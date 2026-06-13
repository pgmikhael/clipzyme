from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


METRIC_DEFINITIONS = {
    "accuracy": "Standard multiclass accuracy on hard predictions.",
    "macro_f1": "Unweighted mean of per-class F1 over classes present in evaluation labels.",
    "macro_precision": "Unweighted mean of per-class precision over classes present in evaluation labels.",
    "macro_recall": "Unweighted mean of per-class recall over classes present in evaluation labels.",
    "micro_auroc": "AUROC from flattened one-vs-rest labels and probabilities.",
    "macro_auroc": "Unweighted mean of valid one-vs-rest class AUROCs.",
    "weighted_auroc": "Support-weighted mean of valid one-vs-rest class AUROCs.",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_sha(default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return default


def environment_info() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


@dataclass
class RunManifest:
    run_id: str
    split_type: str
    seed: int
    model_family: str
    model_name: str
    dataset_manifest_path: str
    created_at: str = field(default_factory=utc_now_iso)
    code_version: str = field(default_factory=git_sha)
    metric_definitions: Dict[str, str] = field(default_factory=lambda: dict(METRIC_DEFINITIONS))
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=environment_info)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())


@dataclass
class SplitManifest:
    split_type: str
    seed: int
    created_at: str
    source: Dict[str, Any]
    positive: Dict[str, list]
    negative: Dict[str, list]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())


@dataclass
class DatasetManifest:
    split_type: str
    split_manifest_path: str
    include_no_cofactor: bool
    min_train_support: int
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[str, str]
    created_at: str
    split_files: Dict[str, str]
    split_counts: Dict[str, int]
    class_support: Dict[str, Dict[str, int]]
    dropped_classes: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())


def file_digest_or_none(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)
