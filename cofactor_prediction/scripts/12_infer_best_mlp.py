#!/usr/bin/env python3
from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import site
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import numpy as np
import torch

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
from src.models_mlp import MLP


DEFAULT_MODEL_RUN_ROOT = REPO_ROOT / "cofactor_prediction" / "runs" / "mlp_tuning_20260208_175222" / "model_runs"
DEFAULT_MODEL_NAME = "mlp_tune_ce_weighted_dropout03_lr1e3_wd1e3"
DEFAULT_SPLIT_TYPE = "random_disjoint"
DEFAULT_SEEDS = (42, 1337, 2025)
DEFAULT_CLIPZYME_CKPT = REPO_ROOT / "files" / "clipzyme_model.ckpt"
DEFAULT_ESM_MODEL = Path("esm_checkpoints/checkpoints/esm2_t33_650M_UR50D.pt")


@dataclass
class LoadedMLP:
    run_dir: Path
    seed: int
    model_name: str
    class_names: List[str]
    model: MLP


def _normalize_seq(seq: str) -> str:
    return "".join(ch for ch in seq.upper() if "A" <= ch <= "Z")


def _parse_seeds(seed_arg: str) -> List[int]:
    parts = [x.strip() for x in seed_arg.split(",") if x.strip()]
    return [int(x) for x in parts]


def _resolve_esm_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file() and path.name == "esm2_t33_650M_UR50D.pt":
        return path.parent
    raise FileNotFoundError(
        f"Could not resolve ESM directory from {path}. "
        "Provide either the checkpoint directory or esm2_t33_650M_UR50D.pt."
    )


def _discover_run_dirs(run_root: Path, split_type: str, model_name: str, seeds: Sequence[int]) -> List[Path]:
    found = []
    for seed in seeds:
        candidate = run_root / f"{split_type}__seed{seed}__{model_name}"
        if candidate.is_dir():
            found.append(candidate)
    return found


def _class_names_from_manifest(manifest: Dict) -> List[str]:
    idx_to_class = manifest["config"]["idx_to_class"]
    n_classes = len(idx_to_class)
    return [str(idx_to_class[str(i)]) for i in range(n_classes)]


def _load_single_mlp(run_dir: Path, device: torch.device) -> LoadedMLP:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json in {run_dir}")

    manifest = read_json(manifest_path)
    config = manifest["config"]
    model_name = str(manifest["model_name"])
    seed = int(manifest["seed"])
    class_names = _class_names_from_manifest(manifest)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        artifact_path = manifest.get("artifacts", {}).get("best_model")
        if artifact_path:
            ckpt_path = Path(artifact_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best model checkpoint for {run_dir}")

    state_dict = torch.load(ckpt_path, map_location=device)
    if "net.0.weight" not in state_dict:
        raise KeyError(f"Unexpected state dict format in {ckpt_path}; missing net.0.weight")

    input_dim = int(state_dict["net.0.weight"].shape[1])
    model = MLP(
        input_dim=input_dim,
        num_classes=len(class_names),
        hidden_dims=[int(x) for x in config["hidden_dims"]],
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return LoadedMLP(
        run_dir=run_dir,
        seed=seed,
        model_name=model_name,
        class_names=class_names,
        model=model,
    )


def _load_mlps(run_dirs: Sequence[Path], device: torch.device) -> List[LoadedMLP]:
    return [_load_single_mlp(rd, device=device) for rd in run_dirs]


def _extract_embedding_from_structure(
    structure_path: Path,
    clipzyme_checkpoint: Path,
    esm_dir: Path,
    device: str,
) -> np.ndarray:
    from clipzyme.lightning.clipzyme import CLIPZyme

    args = Namespace(
        checkpoint_path=str(clipzyme_checkpoint),
        save_hiddens=False,
        save_predictions=False,
        use_as_protein_encoder=True,
        use_as_reaction_encoder=False,
        use_protein_graphs=True,
        skip_protein_graphs=False,
    )
    model = CLIPZyme(args=args, checkpoint_path=str(clipzyme_checkpoint))
    if device:
        try:
            model.model = model.model.to(device)
        except Exception:
            # Keep CPU fallback if model move is not supported in current CLIPZyme stack.
            pass
    cif_path = _ensure_clipzyme_structure_path(structure_path)
    features = model.extract_protein_features(cif_path=str(cif_path), esm_dir=str(esm_dir))
    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().numpy()
    else:
        feat_np = np.asarray(features)

    if feat_np.ndim == 2:
        if feat_np.shape[0] != 1:
            raise ValueError(f"Expected single embedding row, got shape {feat_np.shape}")
        feat_np = feat_np[0]
    if feat_np.ndim != 1:
        raise ValueError(f"Expected 1D embedding, got shape {feat_np.shape}")

    return feat_np.astype(np.float32)


def _ensure_clipzyme_structure_path(structure_path: Path) -> Path:
    suffix = structure_path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return structure_path
    if suffix in {".pdb", ".ent"}:
        import Bio.PDB

        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("query_protein", str(structure_path))
        out_dir = Path(tempfile.mkdtemp(prefix="cofactor_infer_", dir="/tmp"))
        out_path = out_dir / (structure_path.stem + ".cif")
        io = Bio.PDB.MMCIFIO()
        io.set_structure(structure)
        io.save(str(out_path))
        return out_path
    raise ValueError(f"Unsupported structure extension for CLIPZyme extraction: {structure_path}")


def _extract_sequence_from_structure(structure_path: Path) -> str:
    import Bio.PDB
    from Bio.Data.IUPACData import protein_letters_3to1

    aa_map = dict(protein_letters_3to1)
    aa_map.update({k.upper(): v for k, v in aa_map.items()})

    if structure_path.suffix.lower() in {".pdb", ".ent"}:
        parser = Bio.PDB.PDBParser(QUIET=True)
    else:
        parser = Bio.PDB.MMCIFParser(QUIET=True)

    structure = parser.get_structure("query_protein", str(structure_path))
    model = next(structure.get_models())

    chain_seqs: List[str] = []
    for chain in model:
        residues: List[str] = []
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            has_ca = False
            has_n = False
            has_c = False
            for atom in residue:
                atom_name = str(atom.name)
                if atom_name == "CA":
                    has_ca = True
                elif atom_name == "N":
                    has_n = True
                elif atom_name == "C":
                    has_c = True

            if has_ca and has_n and has_c:
                residues.append(aa_map.get(str(residue.get_resname()).upper(), "-"))
        if residues:
            chain_seqs.append("".join(residues))

    if not chain_seqs:
        raise ValueError(f"No amino-acid sequence could be extracted from structure: {structure_path}")
    return ":".join(chain_seqs)


def _resolve_sequence_from_structure(
    structure_path: Path,
    sequence: Optional[str],
    strict: bool,
) -> Tuple[Optional[str], Dict[str, object]]:
    report: Dict[str, object] = {
        "provided_sequence": bool(sequence),
        "sequence_matches_structure": None,
        "provided_sequence_length": len(sequence) if sequence else None,
        "structure_sequence_length": None,
        "sequence_source": "provided" if sequence else "structure",
    }
    structure_seq: Optional[str] = None
    try:
        structure_seq = _extract_sequence_from_structure(structure_path)
        norm_struct = _normalize_seq(structure_seq)
        report["structure_sequence_length"] = len(norm_struct)
    except Exception as exc:
        report["sequence_validation_error"] = str(exc)
        if strict:
            raise

    if sequence is None:
        if structure_seq is not None:
            sequence = structure_seq
            report["sequence_matches_structure"] = True
            report["provided_sequence_length"] = len(_normalize_seq(sequence))
        return sequence, report

    if structure_seq is not None:
        norm_input = _normalize_seq(sequence)
        norm_struct = _normalize_seq(structure_seq)
        match = norm_input == norm_struct
        report["sequence_matches_structure"] = bool(match)
        if not match and strict:
            raise ValueError(
                "Provided sequence does not match the sequence extracted from structure. "
                f"provided_len={len(norm_input)} structure_len={len(norm_struct)}"
            )

    return sequence, report


def _merged_class_names(models: Sequence[LoadedMLP]) -> List[str]:
    merged: List[str] = []
    for item in models:
        for class_name in item.class_names:
            if class_name not in merged:
                merged.append(class_name)
    return merged


def _predict_probs(
    embedding: np.ndarray,
    models: Sequence[LoadedMLP],
    merged_classes: Sequence[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(embedding.astype(np.float32)).to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor after batch add, got {tuple(x.shape)}")

    probs = np.full((len(models), len(merged_classes)), np.nan, dtype=np.float64)
    with torch.no_grad():
        for run_idx, item in enumerate(models):
            logits = item.model(x)
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            class_to_local = {name: idx for idx, name in enumerate(item.class_names)}
            for merged_idx, class_name in enumerate(merged_classes):
                if class_name in class_to_local:
                    probs[run_idx, merged_idx] = float(p[class_to_local[class_name]])

    support_counts = np.sum(~np.isnan(probs), axis=0).astype(np.int64)
    mean_probs = np.nanmean(probs, axis=0)
    std_probs = np.nanstd(probs, axis=0)
    mean_probs = np.nan_to_num(mean_probs, nan=0.0)
    std_probs = np.nan_to_num(std_probs, nan=0.0)
    return mean_probs, std_probs, support_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Infer cofactors with the best selected MLP model. "
            "Supports structure+sequence input (CLIPZyme feature extraction) or direct embedding input."
        )
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--structure", type=Path, help="Path to protein structure file (CIF).")
    input_group.add_argument(
        "--embedding-path",
        type=Path,
        help="Path to precomputed 1D protein embedding (.pt/.npy). Bypasses CLIPZyme feature extraction.",
    )

    p.add_argument("--sequence", type=str, default=None, help="Protein sequence (used for structure consistency check).")
    p.add_argument("--sequence-fasta", type=Path, default=None, help="Optional FASTA file; first record is used.")
    p.add_argument("--protein-id", type=str, default="query_protein")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--exclude-no-cofactor", action="store_true", default=False)
    p.add_argument("--strict-sequence-match", action="store_true", default=False)

    p.add_argument("--clipzyme-checkpoint", type=Path, default=DEFAULT_CLIPZYME_CKPT)
    p.add_argument(
        "--esm-model",
        type=Path,
        default=DEFAULT_ESM_MODEL,
        help="Path to esm2_t33_650M_UR50D.pt or its containing directory.",
    )
    p.add_argument("--device", type=str, default=None, help="Torch device for inference, default: cuda if available else cpu.")

    p.add_argument("--run-root", type=Path, default=DEFAULT_MODEL_RUN_ROOT)
    p.add_argument("--split-type", type=str, default=DEFAULT_SPLIT_TYPE)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--seeds", type=str, default="42,1337,2025", help="Comma-separated seed list for ensemble selection.")
    p.add_argument(
        "--run-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit model run directories. If omitted, they are discovered from run-root/split/model/seeds.",
    )
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def _read_sequence_arg(args: argparse.Namespace) -> Optional[str]:
    seq = args.sequence
    if args.sequence_fasta is not None:
        from Bio import SeqIO

        record = next(SeqIO.parse(str(args.sequence_fasta), "fasta"), None)
        if record is None:
            raise ValueError(f"No sequence records found in {args.sequence_fasta}")
        fasta_seq = str(record.seq)
        if seq is None:
            seq = fasta_seq
        elif _normalize_seq(seq) != _normalize_seq(fasta_seq):
            raise ValueError("Both --sequence and --sequence-fasta are provided but do not match.")
    return seq


def _load_embedding(embedding_path: Path) -> np.ndarray:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if embedding_path.suffix == ".npy":
        arr = np.load(embedding_path)
    else:
        raw = torch.load(embedding_path, map_location="cpu")
        if isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected one embedding row, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D embedding, got {arr.shape}")
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    sequence = _read_sequence_arg(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = _discover_run_dirs(
            run_root=args.run_root,
            split_type=args.split_type,
            model_name=args.model_name,
            seeds=_parse_seeds(args.seeds),
        )
    if not run_dirs:
        raise FileNotFoundError(
            "No model runs found for inference. "
            f"Checked run_root={args.run_root}, split_type={args.split_type}, "
            f"model_name={args.model_name}, seeds={args.seeds}"
        )

    loaded_models = _load_mlps(run_dirs=run_dirs, device=device)
    class_names = _merged_class_names(loaded_models)

    seq_report: Dict[str, object] = {}
    if args.embedding_path is not None:
        embedding = _load_embedding(args.embedding_path)
    else:
        if args.structure is None:
            raise ValueError("Expected --structure when --embedding-path is not provided")
        if not args.structure.exists():
            raise FileNotFoundError(f"Structure file not found: {args.structure}")
        sequence, seq_report = _resolve_sequence_from_structure(
            structure_path=args.structure,
            sequence=sequence,
            strict=args.strict_sequence_match,
        )
        esm_dir = _resolve_esm_dir(args.esm_model)
        if not args.clipzyme_checkpoint.exists():
            raise FileNotFoundError(f"CLIPZyme checkpoint not found: {args.clipzyme_checkpoint}")
        embedding = _extract_embedding_from_structure(
            structure_path=args.structure,
            clipzyme_checkpoint=args.clipzyme_checkpoint,
            esm_dir=esm_dir,
            device=device_name,
        )

    mean_probs, std_probs, class_support_runs = _predict_probs(
        embedding=embedding,
        models=loaded_models,
        merged_classes=class_names,
        device=device,
    )

    ranking = np.argsort(-mean_probs).tolist()
    if args.exclude_no_cofactor:
        ranking = [idx for idx in ranking if class_names[idx] != "no_cofactor"]
    top_k = max(1, min(int(args.top_k), len(ranking)))
    ranking = ranking[:top_k]

    top_preds = []
    for rank, idx in enumerate(ranking, start=1):
        top_preds.append(
            {
                "rank": rank,
                "class_index": int(idx),
                "class_name": class_names[idx],
                "probability_mean": float(mean_probs[idx]),
                "probability_std": float(std_probs[idx]),
                "run_support_count": int(class_support_runs[idx]),
            }
        )

    winner_idx = int(np.argmax(mean_probs))
    output = {
        "protein_id": args.protein_id,
        "model_name": loaded_models[0].model_name,
        "split_type": args.split_type,
        "run_dirs": [str(p) for p in run_dirs],
        "ensemble_size": len(loaded_models),
        "embedding_dim": int(embedding.shape[0]),
        "predicted_class_index": winner_idx,
        "predicted_class_name": class_names[winner_idx],
        "predicted_probability": float(mean_probs[winner_idx]),
        "top_predictions": top_preds,
        "class_run_support_count": {class_names[i]: int(class_support_runs[i]) for i in range(len(class_names))},
        "sequence_check": seq_report,
        "input": {
            "structure": str(args.structure) if args.structure is not None else None,
            "embedding_path": str(args.embedding_path) if args.embedding_path is not None else None,
            "sequence_length": len(sequence) if sequence is not None else None,
        },
    }

    print(f"Protein: {output['protein_id']}")
    print(
        "Predicted cofactor: "
        f"{output['predicted_class_name']} (p={output['predicted_probability']:.4f}, ensemble={output['ensemble_size']})"
    )
    print("Top predictions:")
    for row in top_preds:
        print(
            f"  {row['rank']}. {row['class_name']}: "
            f"{row['probability_mean']:.4f} ± {row['probability_std']:.4f}"
        )

    if args.output_json is not None:
        write_json(args.output_json, output)
        print(f"Saved JSON: {args.output_json}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
