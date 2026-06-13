from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import json
import pandas as pd
from collections import defaultdict

from .manifests import ensure_dir, write_json, utc_now_iso
from .pickle_compat import install_numpy_pickle_compat_aliases


DEFAULT_PATHS = {
    "positive_train_pkl": "claude_progresscofactor_analysis/data/protein_train_cofactors.pkl",
    "positive_dev_pkl": "claude_progresscofactor_analysis/data/protein_dev_cofactors.pkl",
    "positive_test_pkl": "claude_progresscofactor_analysis/data/protein_test_cofactors.pkl",
    "negative_train_csv": "claude_progresscofactor_analysis/data/protein_train_no_cofactors.csv",
    "negative_dev_csv": "claude_progresscofactor_analysis/data/protein_dev_no_cofactors.csv",
    "negative_test_csv": "claude_progresscofactor_analysis/data/protein_test_no_cofactors.csv",
    "positive_embedding_map_json": "claude_progresscofactor_analysis/data_random_split/protein_embedding_map.json",
    "negative_embedding_map_json": "claude_progresscofactor_analysis/data_random_split/protein_negative_embedding_map.json",
    "reaction_cofactor_dataset_pkl": "claude_progresscofactor_analysis/data/reaction_cofactor_dataset.pkl",
    "vocab_json": "claude_progresscofactor_analysis/data/organic_cofactor_vocabulary_filtered.json",
}


@dataclass
class InventoryArtifacts:
    positive_inventory_pkl: Path
    negative_inventory_pkl: Path
    summary_json: Path



def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p


def load_paths_config(paths_config: Optional[Path], root: Path) -> Dict[str, Path]:
    import yaml

    if paths_config is None:
        config = dict(DEFAULT_PATHS)
    else:
        with paths_config.open("r") as f:
            loaded = yaml.safe_load(f) or {}
        config = dict(DEFAULT_PATHS)
        config.update(loaded)

    resolved = {k: _resolve(v, root) for k, v in config.items()}
    missing = [k for k, p in resolved.items() if not p.exists()]
    if missing:
        details = ", ".join(f"{k}={resolved[k]}" for k in missing)
        raise FileNotFoundError(f"Missing required input paths: {details}")
    return resolved


def _safe_cofactors(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _normalize_embedding_path(path_str: str, root: Path) -> Optional[str]:
    path = _resolve(path_str, root)
    if path.exists():
        return str(path)
    return None


def _load_vocab(vocab_path: Path) -> List[str]:
    with vocab_path.open("r") as f:
        vocab = json.load(f)
    return list(vocab["cofactors"])


def _build_rule_map(reaction_df: pd.DataFrame, allowed_cofactors: Set[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    protein_to_rules: Dict[str, Set[str]] = defaultdict(set)
    protein_to_cofactors: Dict[str, Set[str]] = defaultdict(set)

    for _, row in reaction_df.iterrows():
        rule_id = row.get("rule_id")
        if rule_id is None:
            continue
        proteins = row.get("uniprot_ids", [])
        if not isinstance(proteins, list):
            continue

        cofactors = row.get("cofactors", [])
        cofactors = [cf for cf in cofactors if cf in allowed_cofactors] if isinstance(cofactors, list) else []

        for pid in proteins:
            pid = str(pid)
            protein_to_rules[pid].add(str(rule_id))
            protein_to_cofactors[pid].update(cofactors)

    return protein_to_rules, protein_to_cofactors


def build_inventory(paths: Dict[str, Path], output_dir: Path, root: Path, drop_missing_embeddings: bool = True) -> InventoryArtifacts:
    ensure_dir(output_dir)
    install_numpy_pickle_compat_aliases()

    allowed_cofactors = set(_load_vocab(paths["vocab_json"]))

    positive_frames = {
        "train": pd.read_pickle(paths["positive_train_pkl"]),
        "dev": pd.read_pickle(paths["positive_dev_pkl"]),
        "test": pd.read_pickle(paths["positive_test_pkl"]),
    }
    negative_frames = {
        "train": pd.read_csv(paths["negative_train_csv"]),
        "dev": pd.read_csv(paths["negative_dev_csv"]),
        "test": pd.read_csv(paths["negative_test_csv"]),
    }

    with paths["positive_embedding_map_json"].open("r") as f:
        pos_embedding_map_raw = json.load(f)
    with paths["negative_embedding_map_json"].open("r") as f:
        neg_embedding_map_raw = json.load(f)

    pos_embedding_map = {
        str(pid): _normalize_embedding_path(path_str, root)
        for pid, path_str in pos_embedding_map_raw.items()
    }
    neg_embedding_map = {
        str(pid): _normalize_embedding_path(path_str, root)
        for pid, path_str in neg_embedding_map_raw.items()
    }

    reaction_df = pd.read_pickle(paths["reaction_cofactor_dataset_pkl"])
    protein_to_rules, protein_to_rxn_cofactors = _build_rule_map(reaction_df, allowed_cofactors)

    # Build globally deduplicated positive inventory
    positive_rows: Dict[str, Dict] = {}
    for split, df in positive_frames.items():
        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            cofactors = [cf for cf in _safe_cofactors(row.get("cofactors", [])) if cf in allowed_cofactors]

            if pid not in positive_rows:
                positive_rows[pid] = {
                    "protein_id": pid,
                    "sequence": row.get("sequence", ""),
                    "reaction": row.get("reaction", ""),
                    "cif": row.get("cif", ""),
                    "cofactors": set(cofactors),
                    "source_splits": {split},
                }
            else:
                positive_rows[pid]["cofactors"].update(cofactors)
                positive_rows[pid]["source_splits"].add(split)
                if not positive_rows[pid]["sequence"] and row.get("sequence", ""):
                    positive_rows[pid]["sequence"] = row.get("sequence", "")
                if not positive_rows[pid]["reaction"] and row.get("reaction", ""):
                    positive_rows[pid]["reaction"] = row.get("reaction", "")
                if not positive_rows[pid]["cif"] and row.get("cif", ""):
                    positive_rows[pid]["cif"] = row.get("cif", "")

    positive_inventory_records = []
    for pid, rec in positive_rows.items():
        embedding_path = pos_embedding_map.get(pid)
        if drop_missing_embeddings and embedding_path is None:
            continue
        cofactors = sorted(set(rec["cofactors"]) | set(protein_to_rxn_cofactors.get(pid, set())))
        rule_ids = sorted(protein_to_rules.get(pid, set()))
        positive_inventory_records.append(
            {
                "protein_id": pid,
                "sequence": rec["sequence"],
                "reaction": rec["reaction"],
                "cif": rec["cif"],
                "cofactors": cofactors,
                "source_splits": sorted(rec["source_splits"]),
                "embedding_path": embedding_path,
                "rule_ids": rule_ids,
                "num_rules": len(rule_ids),
            }
        )

    positive_inventory = pd.DataFrame(positive_inventory_records).sort_values("protein_id").reset_index(drop=True)

    # Build globally deduplicated negative inventory
    negative_rows: Dict[str, Dict] = {}
    for split, df in negative_frames.items():
        for _, row in df.iterrows():
            pid = str(row["protein_id"])
            if pid not in negative_rows:
                negative_rows[pid] = {
                    "protein_id": pid,
                    "sequence": row.get("sequence", ""),
                    "reaction": row.get("reaction", ""),
                    "cif": row.get("cif", ""),
                    "source_splits": {split},
                }
            else:
                negative_rows[pid]["source_splits"].add(split)
                if not negative_rows[pid]["sequence"] and row.get("sequence", ""):
                    negative_rows[pid]["sequence"] = row.get("sequence", "")
                if not negative_rows[pid]["reaction"] and row.get("reaction", ""):
                    negative_rows[pid]["reaction"] = row.get("reaction", "")
                if not negative_rows[pid]["cif"] and row.get("cif", ""):
                    negative_rows[pid]["cif"] = row.get("cif", "")

    negative_inventory_records = []
    for pid, rec in negative_rows.items():
        embedding_path = neg_embedding_map.get(pid)
        if drop_missing_embeddings and embedding_path is None:
            continue
        negative_inventory_records.append(
            {
                "protein_id": pid,
                "sequence": rec["sequence"],
                "reaction": rec["reaction"],
                "cif": rec["cif"],
                "cofactors": [],
                "source_splits": sorted(rec["source_splits"]),
                "embedding_path": embedding_path,
                "rule_ids": [],
                "num_rules": 0,
            }
        )

    negative_inventory = pd.DataFrame(negative_inventory_records).sort_values("protein_id").reset_index(drop=True)

    positive_path = output_dir / "positive_inventory.pkl"
    negative_path = output_dir / "negative_inventory.pkl"
    summary_path = output_dir / "inventory_summary.json"

    positive_inventory.to_pickle(positive_path)
    negative_inventory.to_pickle(negative_path)

    summary = {
        "created_at": utc_now_iso(),
        "paths": {k: str(v) for k, v in paths.items()},
        "drop_missing_embeddings": drop_missing_embeddings,
        "positive_unique_proteins": int(len(positive_inventory)),
        "negative_unique_proteins": int(len(negative_inventory)),
        "positive_original_rows": int(sum(len(df) for df in positive_frames.values())),
        "negative_original_rows": int(sum(len(df) for df in negative_frames.values())),
        "positive_missing_embeddings": int(sum(1 for v in pos_embedding_map.values() if v is None)),
        "negative_missing_embeddings": int(sum(1 for v in neg_embedding_map.values() if v is None)),
        "num_vocab_cofactors": int(len(allowed_cofactors)),
        "positive_source_split_overlap": {
            "train_dev": int(len(set(positive_frames["train"]["protein_id"]) & set(positive_frames["dev"]["protein_id"]))),
            "train_test": int(len(set(positive_frames["train"]["protein_id"]) & set(positive_frames["test"]["protein_id"]))),
            "dev_test": int(len(set(positive_frames["dev"]["protein_id"]) & set(positive_frames["test"]["protein_id"]))),
        },
        "negative_source_split_overlap": {
            "train_dev": int(len(set(negative_frames["train"]["protein_id"]) & set(negative_frames["dev"]["protein_id"]))),
            "train_test": int(len(set(negative_frames["train"]["protein_id"]) & set(negative_frames["test"]["protein_id"]))),
            "dev_test": int(len(set(negative_frames["dev"]["protein_id"]) & set(negative_frames["test"]["protein_id"]))),
        },
    }
    write_json(summary_path, summary)

    return InventoryArtifacts(
        positive_inventory_pkl=positive_path,
        negative_inventory_pkl=negative_path,
        summary_json=summary_path,
    )
