#!/usr/bin/env python3
"""Batch cofactor inference over a CSV of proteins.

Two mutually exclusive modes (one CSV, one mode per run — no mixing):

  Structure mode (default):
    CSV must contain `protein_id` and `structure_path` columns. CLIPZyme is
    loaded once and used to extract a 1280-dim embedding per row. Optional
    `sequence` column is used for structure/sequence consistency checks.
    Structure paths may be absolute or relative to the CSV's parent dir.

  Embedding mode (`--embeddings-pickle PATH`):
    CSV needs only `protein_id`. CLIPZyme is not loaded. Embeddings come from
    a single pickle file containing either:
      - {id: 1D embedding} — direct mapping, or
      - {"hiddens": (N, D) array, "uniprots"/"ids"/"protein_ids": (N,)} —
        parallel-array form matching the existing screening-set layout.

Outputs (in --output-dir):
  predictions.jsonl         — one full record per protein (success or error)
  predictions.csv           — flat summary: predicted class, top-k, per-class probabilities
  predictions_summary.json  — counts and run metadata
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import site
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The `src` package lives under cofactor_prediction/, so that directory (not the
# repo root) must be importable. Add it explicitly so `import src` works
# regardless of cwd or how the script is launched (e.g. `python -m pdb ...`,
# which resets sys.path[0] to the script's directory).
COFACTOR_DIR = Path(__file__).resolve().parents[1]
if str(COFACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(COFACTOR_DIR))

from src.manifests import read_json, write_json

SCRIPT_DIR = Path(__file__).resolve().parent
SIBLING_PATH = SCRIPT_DIR / "12_infer_best_mlp.py"
_HELPER_MOD_NAME = "infer_best_mlp_helpers"
_spec = importlib.util.spec_from_file_location(_HELPER_MOD_NAME, SIBLING_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load helper module from {SIBLING_PATH}")
_helpers = importlib.util.module_from_spec(_spec)
sys.modules[_HELPER_MOD_NAME] = _helpers
_spec.loader.exec_module(_helpers)

DEFAULT_MODEL_RUN_ROOT = _helpers.DEFAULT_MODEL_RUN_ROOT
DEFAULT_MODEL_NAME = _helpers.DEFAULT_MODEL_NAME
DEFAULT_SPLIT_TYPE = _helpers.DEFAULT_SPLIT_TYPE
DEFAULT_CLIPZYME_CKPT = _helpers.DEFAULT_CLIPZYME_CKPT
DEFAULT_ESM_MODEL = _helpers.DEFAULT_ESM_MODEL


def _load_clipzyme_once(checkpoint_path: Path, device_name: str):
    from clipzyme.lightning.clipzyme import CLIPZyme

    args = Namespace(
        checkpoint_path=str(checkpoint_path),
        save_hiddens=False,
        save_predictions=False,
        use_as_protein_encoder=True,
        use_as_reaction_encoder=False,
        use_protein_graphs=True,
        skip_protein_graphs=False,
    )
    model = CLIPZyme(args=args, checkpoint_path=str(checkpoint_path))
    if device_name:
        try:
            model.model = model.model.to(device_name)
        except Exception:
            pass
    return model


def _features_to_2d(features) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        arr = features.detach().cpu().numpy()
    else:
        arr = np.asarray(features)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected feature shape from CLIPZyme: {arr.shape}")
    return arr.astype(np.float32)


def _extract_embeddings_batch(
    clipzyme_model, structure_paths: List[Path], esm_dir: Path
) -> List[Tuple[Optional[np.ndarray], Optional[str]]]:
    """Run CLIPZyme on a batch of structures.

    Returns a list parallel to ``structure_paths``: each entry is
    ``(embedding, None)`` on success or ``(None, error_text)`` on per-row
    failure. On a whole-batch failure (e.g. one bad CIF kills the collate),
    the function falls back to one-at-a-time extraction so the rest of the
    batch survives.
    """
    if not structure_paths:
        return []

    cif_strs: List[Optional[str]] = []
    results: List[Tuple[Optional[np.ndarray], Optional[str]]] = [
        (None, None) for _ in structure_paths
    ]
    for i, sp in enumerate(structure_paths):
        try:
            cif_strs.append(str(_helpers._ensure_clipzyme_structure_path(sp)))
        except Exception as exc:
            cif_strs.append(None)
            results[i] = (None, f"{type(exc).__name__}: {exc}")

    valid_idx = [i for i, c in enumerate(cif_strs) if c is not None]
    if not valid_idx:
        return results

    valid_cifs = [cif_strs[i] for i in valid_idx]
    try:
        features = clipzyme_model.extract_protein_features(
            cif_path=valid_cifs, esm_dir=str(esm_dir)
        )
        feats_2d = _features_to_2d(features)
        if feats_2d.shape[0] != len(valid_cifs):
            raise ValueError(
                f"CLIPZyme returned {feats_2d.shape[0]} rows for batch of {len(valid_cifs)}"
            )
        for j, src_i in enumerate(valid_idx):
            results[src_i] = (feats_2d[j], None)
        return results
    except Exception:
        if len(valid_cifs) == 1:
            i = valid_idx[0]
            try:
                features = clipzyme_model.extract_protein_features(
                    cif_path=valid_cifs[0], esm_dir=str(esm_dir)
                )
                results[i] = (_features_to_2d(features)[0], None)
            except Exception as exc:
                results[i] = (None, f"{type(exc).__name__}: {exc}")
            return results
        for j, src_i in enumerate(valid_idx):
            try:
                features = clipzyme_model.extract_protein_features(
                    cif_path=valid_cifs[j], esm_dir=str(esm_dir)
                )
                results[src_i] = (_features_to_2d(features)[0], None)
            except Exception as exc:
                results[src_i] = (None, f"{type(exc).__name__}: {exc}")
        return results


def _read_csv_rows(
    csv_path: Path,
    id_col: str,
    struct_col: str,
    seq_col: Optional[str],
    *,
    struct_required: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if id_col not in fieldnames:
            raise ValueError(f"CSV missing required column '{id_col}'. Found: {fieldnames}")
        if struct_required and struct_col not in fieldnames:
            raise ValueError(
                f"CSV missing required column '{struct_col}'. Found: {fieldnames}"
            )
        struct_present = struct_col in fieldnames
        seq_present = bool(seq_col) and seq_col in fieldnames
        for row in reader:
            pid = (row.get(id_col) or "").strip()
            if not pid:
                continue
            spath = (row.get(struct_col) or "").strip() if struct_present else ""
            if struct_required and not spath:
                continue
            seq_val = (row.get(seq_col) or "").strip() if seq_present else ""
            rows.append(
                {"protein_id": pid, "structure_path": spath, "sequence": seq_val}
            )
    return rows


def _to_1d_float32(value, label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Embedding for {label} expected 1D, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Embedding for {label} expected 1D, got shape {arr.shape}")
    return arr.astype(np.float32)


def _load_embeddings_pickle(path: Path) -> Dict[str, np.ndarray]:
    """Load a {id: 1D embedding} lookup from a pickle file.

    Accepts either a direct mapping or the parallel-array screening-set layout.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(f"Embeddings pickle not found: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Embeddings pickle must contain a dict, got {type(raw).__name__}"
        )

    keys = set(raw.keys())
    parallel_data_keys = ["hiddens", "embeddings", "features"]
    parallel_id_keys = ["uniprots", "ids", "protein_ids", "uniprot_ids"]
    h_key = next((k for k in parallel_data_keys if k in keys), None)
    i_key = next((k for k in parallel_id_keys if k in keys), None)
    if h_key is not None and i_key is not None:
        hiddens = raw[h_key]
        if isinstance(hiddens, torch.Tensor):
            hiddens = hiddens.detach().cpu().numpy()
        hiddens = np.asarray(hiddens)
        if hiddens.ndim != 2:
            raise ValueError(
                f"Parallel-array embeddings expected 2D '{h_key}', got shape {hiddens.shape}"
            )
        ids = list(raw[i_key])
        if len(ids) != hiddens.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(ids)} ids ('{i_key}') vs "
                f"{hiddens.shape[0]} rows ('{h_key}')"
            )
        out = {str(pid): hiddens[i].astype(np.float32) for i, pid in enumerate(ids)}
        return out

    return {str(k): _to_1d_float32(v, str(k)) for k, v in raw.items()}


def _resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def _existing_done_ids(jsonl_path: Path) -> set:
    done: set = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and rec.get("protein_id"):
                done.add(str(rec["protein_id"]))
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch cofactor inference: CSV of protein structures -> ensemble MLP predictions."
    )
    p.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    p.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for outputs (created if missing)."
    )
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-jsonl", type=Path, default=None)

    p.add_argument("--id-column", type=str, default="protein_id")
    p.add_argument("--structure-column", type=str, default="structure_path")
    p.add_argument(
        "--embeddings-pickle",
        type=Path,
        default=None,
        help=(
            "Optional path to a single pickle file containing precomputed embeddings. "
            "When set, CLIPZyme is not loaded and embeddings are looked up by protein_id. "
            "Accepts either {id: 1D array} or "
            "{'hiddens': (N, D), 'uniprots'/'ids'/'protein_ids': (N,)}."
        ),
    )
    p.add_argument(
        "--sequence-column",
        type=str,
        default="sequence",
        help="Optional sequence column name; only used in structure mode.",
    )

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--exclude-no-cofactor", action="store_true", default=False)
    p.add_argument("--strict-sequence-match", action="store_true", default=False)

    p.add_argument("--clipzyme-checkpoint", type=Path, default=DEFAULT_CLIPZYME_CKPT)
    p.add_argument("--esm-model", type=Path, default=DEFAULT_ESM_MODEL)
    p.add_argument("--device", type=str, default=None,
                   help="Single-process device, e.g. 'cuda:2' or 'cpu'. Ignored when --gpus is set.")
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for multi-process structure-mode extraction "
            "(e.g. '0,1,2,3'). Spawns one worker per GPU; each handles a "
            "round-robin slice of rows. Ignored in embedding mode."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of structures per CLIPZyme forward in structure mode (per worker).",
    )

    p.add_argument("--run-root", type=Path, default=DEFAULT_MODEL_RUN_ROOT)
    p.add_argument("--split-type", type=str, default=DEFAULT_SPLIT_TYPE)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--seeds", type=str, default="42,1337,2025")
    p.add_argument("--run-dirs", type=Path, nargs="*", default=None)

    p.add_argument("--limit", type=int, default=None, help="Process at most this many rows after --skip.")
    p.add_argument("--skip", type=int, default=0, help="Skip first N rows of the CSV.")
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip protein_ids already present with status=ok in the output JSONL; append to outputs.",
    )

    return p.parse_args()


def _build_jsonl_record_ok(
    pid: str,
    embedding: np.ndarray,
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    support_runs: np.ndarray,
    class_names: List[str],
    top_idx: List[int],
    winner_idx: int,
    model_name: str,
    split_type: str,
    ensemble_size: int,
    seq_used: Optional[str],
    seq_report: Dict[str, object],
    source: str,
    struct_path: Optional[Path],
    embeddings_pickle: Optional[Path],
) -> Dict[str, object]:
    top_preds = [
        {
            "rank": rank,
            "class_index": int(idx),
            "class_name": class_names[idx],
            "probability_mean": float(mean_probs[idx]),
            "probability_std": float(std_probs[idx]),
            "run_support_count": int(support_runs[idx]),
        }
        for rank, idx in enumerate(top_idx, start=1)
    ]
    return {
        "protein_id": pid,
        "status": "ok",
        "model_name": model_name,
        "split_type": split_type,
        "ensemble_size": ensemble_size,
        "embedding_dim": int(embedding.shape[0]),
        "predicted_class_index": winner_idx,
        "predicted_class_name": class_names[winner_idx],
        "predicted_probability": float(mean_probs[winner_idx]),
        "top_predictions": top_preds,
        "class_run_support_count": {
            class_names[idx]: int(support_runs[idx]) for idx in range(len(class_names))
        },
        "class_probabilities_mean": {
            class_names[idx]: float(mean_probs[idx]) for idx in range(len(class_names))
        },
        "class_probabilities_std": {
            class_names[idx]: float(std_probs[idx]) for idx in range(len(class_names))
        },
        "sequence_check": seq_report,
        "input": {
            "source": source,
            "structure": str(struct_path) if struct_path is not None else None,
            "embeddings_pickle": str(embeddings_pickle) if embeddings_pickle is not None else None,
            "sequence_length": len(seq_used) if seq_used is not None else None,
        },
    }


def _populate_csv_record_ok(
    csv_record: Dict[str, object],
    embedding: np.ndarray,
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    class_names: List[str],
    top_idx: List[int],
    winner_idx: int,
    seq_report: Dict[str, object],
) -> None:
    csv_record.update(
        status="ok",
        error=None,
        predicted_class_name=class_names[winner_idx],
        predicted_probability=float(mean_probs[winner_idx]),
        embedding_dim=int(embedding.shape[0]),
        sequence_matches_structure=seq_report.get("sequence_matches_structure"),
    )
    for rank, idx in enumerate(top_idx, start=1):
        csv_record[f"top{rank}_class"] = class_names[idx]
        csv_record[f"top{rank}_prob_mean"] = float(mean_probs[idx])
        csv_record[f"top{rank}_prob_std"] = float(std_probs[idx])
    for cname_idx, cname in enumerate(class_names):
        csv_record[f"prob__{cname}"] = float(mean_probs[cname_idx])


def _run_inference(
    rank: int,
    world_size: int,
    gpu_ids: Optional[List[int]],
    args: argparse.Namespace,
    all_rows: List[Dict[str, str]],
    csv_dir: Path,
    output_jsonl: Path,
    output_csv: Path,
    fieldnames: List[str],
    class_names: List[str],
    write_csv_header: bool,
    jsonl_mode: str,
    csv_mode: str,
    done_ids: set,
    show_progress: bool = True,
) -> Dict[str, int]:
    """Process this worker's slice of rows.

    Slice is round-robin: rows[rank::world_size]. Loads its own MLPs and
    (in structure mode) CLIPZyme on the assigned GPU.
    """
    use_embeddings = args.embeddings_pickle is not None

    if gpu_ids:
        device_name = f"cuda:{gpu_ids[rank]}"
    elif args.device:
        device_name = args.device
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = _helpers._discover_run_dirs(
            run_root=args.run_root,
            split_type=args.split_type,
            model_name=args.model_name,
            seeds=_helpers._parse_seeds(args.seeds),
        )

    loaded_models = _helpers._load_mlps(run_dirs=run_dirs, device=device)
    model_name = loaded_models[0].model_name

    embeddings_lookup: Dict[str, np.ndarray] = {}
    clipzyme_model = None
    esm_dir = None
    if use_embeddings:
        embeddings_lookup = _load_embeddings_pickle(args.embeddings_pickle)
    else:
        esm_dir = _helpers._resolve_esm_dir(args.esm_model)
        clipzyme_model = _load_clipzyme_once(args.clipzyme_checkpoint, device_name)

    slice_rows = all_rows[rank::world_size]
    rows_to_run = [r for r in slice_rows if r["protein_id"] not in done_ids]
    n_skip = len(slice_rows) - len(rows_to_run)
    n_ok = 0
    n_err = 0

    batch_size = max(1, int(args.batch_size)) if not use_embeddings else 1
    desc = f"rank{rank}" if world_size > 1 else "Inferring"

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open(jsonl_mode) as jf, output_csv.open(csv_mode, newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        if write_csv_header:
            writer.writeheader()
            cf.flush()

        progress = (
            tqdm(
                total=len(rows_to_run),
                desc=desc,
                unit="protein",
                position=rank if world_size > 1 else 0,
            )
            if show_progress
            else None
        )

        for batch_start in range(0, len(rows_to_run), batch_size):
            batch = rows_to_run[batch_start : batch_start + batch_size]
            states: List[Dict[str, object]] = []
            need_extract_indices: List[int] = []
            need_extract_paths: List[Path] = []

            for row in batch:
                pid = row["protein_id"]
                struct_path = (
                    _resolve_path(row["structure_path"], csv_dir)
                    if row["structure_path"]
                    else None
                )
                sequence = row["sequence"] or None
                source = "embedding" if use_embeddings else "structure"

                csv_record: Dict[str, object] = {
                    "protein_id": pid,
                    "structure_path": str(struct_path) if struct_path is not None else "",
                    "status": "error",
                    "error": None,
                    "predicted_class_name": None,
                    "predicted_probability": None,
                    "ensemble_size": len(loaded_models),
                    "embedding_dim": None,
                    "sequence_matches_structure": None,
                }
                state: Dict[str, object] = {
                    "pid": pid,
                    "struct_path": struct_path,
                    "sequence": sequence,
                    "source": source,
                    "csv_record": csv_record,
                    "embedding": None,
                    "error": None,
                    "seq_used": sequence,
                    "seq_report": {},
                }

                try:
                    if use_embeddings:
                        if pid not in embeddings_lookup:
                            raise KeyError(
                                f"protein_id '{pid}' not found in embeddings pickle"
                            )
                        state["embedding"] = embeddings_lookup[pid]
                    else:
                        if struct_path is None:
                            raise ValueError("Row has no structure_path")
                        if not struct_path.exists():
                            raise FileNotFoundError(
                                f"Structure file not found: {struct_path}"
                            )
                        seq_used, seq_report = _helpers._resolve_sequence_from_structure(
                            structure_path=struct_path,
                            sequence=sequence,
                            strict=args.strict_sequence_match,
                        )
                        state["seq_used"] = seq_used
                        state["seq_report"] = seq_report
                        need_extract_indices.append(len(states))
                        need_extract_paths.append(struct_path)
                except Exception as exc:
                    state["error"] = f"{type(exc).__name__}: {exc}"

                states.append(state)

            if need_extract_paths:
                batch_results = _extract_embeddings_batch(
                    clipzyme_model, need_extract_paths, esm_dir
                )
                for j, src_i in enumerate(need_extract_indices):
                    emb, err = batch_results[j]
                    if err is not None:
                        states[src_i]["error"] = err
                    else:
                        states[src_i]["embedding"] = emb

            for state in states:
                pid = state["pid"]
                csv_record = state["csv_record"]
                struct_path = state["struct_path"]
                source = state["source"]

                if state["error"] is not None or state["embedding"] is None:
                    err_text = state["error"] or "no embedding produced"
                    csv_record["error"] = err_text
                    jsonl_record = {
                        "protein_id": pid,
                        "status": "error",
                        "error": err_text,
                        "input": {
                            "source": source,
                            "structure": str(struct_path) if struct_path is not None else None,
                            "embeddings_pickle": (
                                str(args.embeddings_pickle) if use_embeddings else None
                            ),
                        },
                    }
                    jf.write(json.dumps(jsonl_record) + "\n")
                    jf.flush()
                    writer.writerow(csv_record)
                    cf.flush()
                    n_err += 1
                    rank_tag = f"[rank{rank}] " if world_size > 1 else ""
                    tqdm.write(f"{rank_tag}{pid}: ERROR — {err_text}", file=sys.stderr)
                else:
                    embedding = state["embedding"]
                    mean_probs, std_probs, support_runs = _helpers._predict_probs(
                        embedding=embedding,
                        models=loaded_models,
                        merged_classes=class_names,
                        device=device,
                    )

                    ranking = np.argsort(-mean_probs).tolist()
                    if args.exclude_no_cofactor:
                        ranking = [
                            idx for idx in ranking if class_names[idx] != "no_cofactor"
                        ]
                    top_k = max(1, min(int(args.top_k), len(ranking)))
                    top_idx = ranking[:top_k]
                    winner_idx = int(np.argmax(mean_probs))

                    jsonl_record = _build_jsonl_record_ok(
                        pid=pid,
                        embedding=embedding,
                        mean_probs=mean_probs,
                        std_probs=std_probs,
                        support_runs=support_runs,
                        class_names=class_names,
                        top_idx=top_idx,
                        winner_idx=winner_idx,
                        model_name=model_name,
                        split_type=args.split_type,
                        ensemble_size=len(loaded_models),
                        seq_used=state["seq_used"],
                        seq_report=state["seq_report"],
                        source=source,
                        struct_path=struct_path,
                        embeddings_pickle=args.embeddings_pickle if use_embeddings else None,
                    )
                    _populate_csv_record_ok(
                        csv_record=csv_record,
                        embedding=embedding,
                        mean_probs=mean_probs,
                        std_probs=std_probs,
                        class_names=class_names,
                        top_idx=top_idx,
                        winner_idx=winner_idx,
                        seq_report=state["seq_report"],
                    )

                    jf.write(json.dumps(jsonl_record) + "\n")
                    jf.flush()
                    writer.writerow(csv_record)
                    cf.flush()
                    n_ok += 1

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(ok=n_ok, err=n_err, skip=n_skip)

        if progress is not None:
            progress.close()

    return {"ok": n_ok, "err": n_err, "skip": n_skip, "n_total": len(slice_rows)}


def _spawn_worker(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    args: argparse.Namespace,
    all_rows: List[Dict[str, str]],
    csv_dir: Path,
    shard_dir: Path,
    fieldnames: List[str],
    class_names: List[str],
    done_ids: set,
) -> None:
    """mp.spawn entrypoint. Writes a shard pair predictions_rank{N}.{jsonl,csv}."""
    shard_jsonl = shard_dir / f"shard_rank{rank}.jsonl"
    shard_csv = shard_dir / f"shard_rank{rank}.csv"
    counts = _run_inference(
        rank=rank,
        world_size=world_size,
        gpu_ids=gpu_ids,
        args=args,
        all_rows=all_rows,
        csv_dir=csv_dir,
        output_jsonl=shard_jsonl,
        output_csv=shard_csv,
        fieldnames=fieldnames,
        class_names=class_names,
        write_csv_header=False,
        jsonl_mode="w",
        csv_mode="w",
        done_ids=done_ids,
        show_progress=True,
    )
    counts_path = shard_dir / f"shard_rank{rank}.counts.json"
    write_json(counts_path, counts)


def _merge_shards(
    shard_dir: Path,
    world_size: int,
    output_jsonl: Path,
    output_csv: Path,
    fieldnames: List[str],
    jsonl_mode: str,
    csv_mode: str,
    write_csv_header: bool,
) -> Dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0
    n_skip = 0
    n_total = 0
    with output_jsonl.open(jsonl_mode) as jf_out, output_csv.open(csv_mode, newline="") as cf_out:
        if write_csv_header:
            cf_out.write(",".join(fieldnames) + "\n")
        for rank in range(world_size):
            sj = shard_dir / f"shard_rank{rank}.jsonl"
            sc = shard_dir / f"shard_rank{rank}.csv"
            if sj.exists():
                with sj.open("r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        jf_out.write(line if line.endswith("\n") else line + "\n")
            if sc.exists():
                with sc.open("r") as f:
                    for line in f:
                        cf_out.write(line)
            counts_path = shard_dir / f"shard_rank{rank}.counts.json"
            if counts_path.exists():
                c = read_json(counts_path)
                n_ok += int(c.get("ok", 0))
                n_err += int(c.get("err", 0))
                n_skip += int(c.get("skip", 0))
                n_total += int(c.get("n_total", 0))
    return {"ok": n_ok, "err": n_err, "skip": n_skip, "n_total": n_total}


def _class_names_from_run_dirs(run_dirs: List[Path]) -> List[str]:
    merged: List[str] = []
    for rd in run_dirs:
        manifest = read_json(rd / "run_manifest.json")
        for c in _helpers._class_names_from_manifest(manifest):
            if c not in merged:
                merged.append(c)
    return merged


def _parse_gpu_ids(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    parts = [x.strip() for x in arg.split(",") if x.strip()]
    return [int(x) for x in parts]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or (args.output_dir / "predictions.csv")
    output_jsonl = args.output_jsonl or (args.output_dir / "predictions.jsonl")

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = _helpers._discover_run_dirs(
            run_root=args.run_root,
            split_type=args.split_type,
            model_name=args.model_name,
            seeds=_helpers._parse_seeds(args.seeds),
        )
    if not run_dirs:
        raise FileNotFoundError(
            "No model runs discovered. "
            f"run_root={args.run_root} split_type={args.split_type} "
            f"model_name={args.model_name} seeds={args.seeds}"
        )

    class_names = _class_names_from_run_dirs(run_dirs)
    base_cols = [
        "protein_id",
        "structure_path",
        "status",
        "error",
        "predicted_class_name",
        "predicted_probability",
        "ensemble_size",
        "embedding_dim",
        "sequence_matches_structure",
    ]
    top_cols: List[str] = []
    top_k_cols = max(1, int(args.top_k))
    for k in range(1, top_k_cols + 1):
        top_cols.extend([f"top{k}_class", f"top{k}_prob_mean", f"top{k}_prob_std"])
    prob_cols = [f"prob__{c}" for c in class_names]
    fieldnames = base_cols + top_cols + prob_cols

    use_embeddings = args.embeddings_pickle is not None
    seq_col = args.sequence_column or None
    rows = _read_csv_rows(
        args.csv,
        args.id_column,
        args.structure_column,
        seq_col,
        struct_required=not use_embeddings,
    )
    csv_dir = args.csv.resolve().parent

    if args.skip:
        rows = rows[args.skip :]
    if args.limit is not None:
        rows = rows[: args.limit]

    done_ids: set = _existing_done_ids(output_jsonl) if args.resume else set()
    if done_ids:
        print(f"Resume: {len(done_ids)} protein(s) already complete will be skipped.")

    append_mode = args.resume and output_jsonl.exists()
    jsonl_mode = "a" if append_mode else "w"
    csv_mode = "a" if (args.resume and output_csv.exists()) else "w"
    write_csv_header = csv_mode == "w"

    if not use_embeddings and not args.clipzyme_checkpoint.exists():
        raise FileNotFoundError(
            f"CLIPZyme checkpoint not found: {args.clipzyme_checkpoint}"
        )

    gpu_ids = _parse_gpu_ids(args.gpus)
    multi_gpu = (not use_embeddings) and len(gpu_ids) > 1
    world_size = len(gpu_ids) if multi_gpu else 1

    print(f"Run config: world_size={world_size}, "
          f"mode={'embedding' if use_embeddings else 'structure'}, "
          f"batch_size={args.batch_size}, "
          f"rows={len(rows)}, classes={len(class_names)}, "
          f"ensemble={len(run_dirs)}")

    if world_size == 1:
        single_gpu_ids = gpu_ids if (gpu_ids and not use_embeddings) else None
        counts = _run_inference(
            rank=0,
            world_size=1,
            gpu_ids=single_gpu_ids,
            args=args,
            all_rows=rows,
            csv_dir=csv_dir,
            output_jsonl=output_jsonl,
            output_csv=output_csv,
            fieldnames=fieldnames,
            class_names=class_names,
            write_csv_header=write_csv_header,
            jsonl_mode=jsonl_mode,
            csv_mode=csv_mode,
            done_ids=done_ids,
            show_progress=True,
        )
    else:
        import torch.multiprocessing as mp

        shard_dir = args.output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for old in shard_dir.glob("shard_rank*.*"):
            old.unlink()

        print(f"Spawning {world_size} workers (GPUs: {gpu_ids})...")
        mp.spawn(
            _spawn_worker,
            args=(
                world_size,
                gpu_ids,
                args,
                rows,
                csv_dir,
                shard_dir,
                fieldnames,
                class_names,
                done_ids,
            ),
            nprocs=world_size,
            join=True,
        )

        counts = _merge_shards(
            shard_dir=shard_dir,
            world_size=world_size,
            output_jsonl=output_jsonl,
            output_csv=output_csv,
            fieldnames=fieldnames,
            jsonl_mode=jsonl_mode,
            csv_mode=csv_mode,
            write_csv_header=write_csv_header,
        )

    summary = {
        "total_rows": len(rows),
        "succeeded": counts["ok"],
        "failed": counts["err"],
        "skipped_resume": counts["skip"],
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "ensemble_size": len(run_dirs),
        "split_type": args.split_type,
        "model_name": args.model_name,
        "embedding_source": "embedding" if use_embeddings else "structure",
        "embeddings_pickle": str(args.embeddings_pickle) if use_embeddings else None,
        "world_size": world_size,
        "gpu_ids": gpu_ids,
        "batch_size": args.batch_size,
        "run_dirs": [str(p) for p in run_dirs],
        "class_names": class_names,
    }
    summary_path = args.output_dir / "predictions_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "class_names"}, indent=2))


if __name__ == "__main__":
    main()
