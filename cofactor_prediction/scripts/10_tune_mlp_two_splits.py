#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import subprocess
import sys
from typing import Dict, List

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifests import ensure_dir, utc_now_iso, write_json


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_clipzyme")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def _load_tuning_config(path: Path) -> Dict:
    with path.open("r") as f:
        payload = yaml.safe_load(f) or {}
    cfgs = payload.get("mlp_configs", payload)
    names = payload.get("config_names", list(cfgs.keys()))
    names = [str(x) for x in names]
    missing = [name for name in names if name not in cfgs]
    if missing:
        raise KeyError(f"Config names missing from mlp_configs: {missing}")
    return {"mlp_configs": cfgs, "config_names": names}


def _write_resolved_config(payload: Dict, out_root: Path) -> Path:
    cfg_path = out_root / "tuning_configs" / "resolved_mlp_tuning_config.yaml"
    ensure_dir(cfg_path.parent)
    with cfg_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return cfg_path


def _dataset_manifest(dataset_root: Path, split_type: str, seed: int) -> Path:
    return dataset_root / split_type / f"seed_{seed}" / "dataset_manifest.json"


def _run_id(split_type: str, seed: int, config_name: str) -> str:
    return f"{split_type}__seed{seed}__{config_name}"


def _build_tuning_leaderboard(run_root: Path, out_dir: Path) -> Path:
    all_metrics_path = out_dir / "publication" / "tables" / "all_runs_metrics.csv"
    if not all_metrics_path.exists():
        raise FileNotFoundError(f"Missing aggregated metrics table: {all_metrics_path}")

    df = pd.read_csv(all_metrics_path)
    if df.empty:
        raise RuntimeError("Aggregated metrics table is empty")

    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "micro_auroc", "macro_auroc", "weighted_auroc"]
    grouped = (
        df.groupby(["split_type", "model_name"], as_index=False)
        .agg(
            n_runs=("macro_f1", "count"),
            **{f"{m}_mean": (m, "mean") for m in metrics},
            **{f"{m}_std": (m, "std") for m in metrics},
        )
        .sort_values(["split_type", "macro_f1_mean"], ascending=[True, False])
    )

    grouped["rank_by_macro_f1"] = grouped.groupby("split_type")["macro_f1_mean"].rank(ascending=False, method="min")
    grouped = grouped.sort_values(["split_type", "rank_by_macro_f1", "model_name"])

    out_csv = out_dir / "publication" / "tables" / "mlp_tuning_leaderboard_by_split.csv"
    grouped.to_csv(out_csv, index=False)

    lines = [
        "| split_type | rank | model_name | macro_f1_mean | macro_f1_std | macro_auroc_mean | micro_auroc_mean | accuracy_mean |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in grouped.iterrows():
        lines.append(
            f"| {r['split_type']} | {int(r['rank_by_macro_f1'])} | {r['model_name']} | "
            f"{r['macro_f1_mean']:.4f} | {0.0 if pd.isna(r['macro_f1_std']) else r['macro_f1_std']:.4f} | "
            f"{r['macro_auroc_mean']:.4f} | {r['micro_auroc_mean']:.4f} | {r['accuracy_mean']:.4f} |"
        )

    out_md = out_dir / "publication" / "tables" / "mlp_tuning_leaderboard_by_split.md"
    out_md.write_text("\n".join(lines) + "\n")
    return out_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run additional MLP tuning on random_disjoint and balanced_rule_disjoint_v3.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Path to datasets directory from an existing run root.")
    p.add_argument("--out-root", type=Path, required=True, help="New output root for tuning runs.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("cofactor_prediction/configs/mlp_tuning_two_splits.yaml"),
        help="YAML containing mlp_configs and config_names.",
    )
    p.add_argument(
        "--split-types",
        type=str,
        default="random_disjoint,balanced_rule_disjoint_v3",
        help="Comma-separated split types to tune.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,1337,2025",
        help="Comma-separated dataset/model seeds (must match available dataset manifests).",
    )
    p.add_argument("--skip-existing", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_root)

    split_types = [x.strip() for x in args.split_types.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    tuning_payload = _load_tuning_config(args.config)
    resolved_cfg_path = _write_resolved_config(tuning_payload, args.out_root)
    cfg_names = tuning_payload["config_names"]

    model_run_root = args.out_root / "model_runs"
    ensure_dir(model_run_root)

    run_log = {
        "created_at": utc_now_iso(),
        "dataset_root": str(args.dataset_root),
        "out_root": str(args.out_root),
        "split_types": split_types,
        "seeds": seeds,
        "config_names": cfg_names,
        "resolved_config": str(resolved_cfg_path),
        "runs_requested": 0,
        "runs_planned": 0,
        "runs_executed": 0,
        "runs_skipped_existing": 0,
    }

    for split_type in split_types:
        for seed in seeds:
            dataset_manifest_path = _dataset_manifest(args.dataset_root, split_type, seed)
            if not dataset_manifest_path.exists():
                raise FileNotFoundError(f"Missing dataset manifest: {dataset_manifest_path}")

            for config_name in cfg_names:
                run_log["runs_requested"] += 1
                run_id = _run_id(split_type, seed, config_name)
                run_dir = model_run_root / run_id

                if args.skip_existing and (run_dir / "metrics.json").exists() and (run_dir / "roc" / "roc_summary.json").exists():
                    run_log["runs_skipped_existing"] += 1
                    continue

                run_log["runs_planned"] += 1

                _run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "cofactor_prediction/scripts/05_run_mlp.py"),
                        "--dataset",
                        str(dataset_manifest_path),
                        "--config",
                        str(resolved_cfg_path),
                        "--config-name",
                        config_name,
                        "--seed",
                        str(seed),
                        "--out-dir",
                        str(model_run_root),
                    ],
                    dry_run=args.dry_run,
                )
                _run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "cofactor_prediction/scripts/06_evaluate_runs.py"),
                        "--run-dir",
                        str(run_dir),
                    ],
                    dry_run=args.dry_run,
                )
                _run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "cofactor_prediction/scripts/07_make_roc_artifacts.py"),
                        "--run-dir",
                        str(run_dir),
                    ],
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    run_log["runs_executed"] += 1

    if not args.dry_run:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "cofactor_prediction/scripts/08_aggregate_publication_tables.py"),
                "--run-root",
                str(model_run_root),
                "--out-dir",
                str(args.out_root / "publication"),
            ],
            dry_run=False,
        )

        leaderboard_csv = _build_tuning_leaderboard(model_run_root, args.out_root)
        run_log["tuning_leaderboard_csv"] = str(leaderboard_csv)

    run_log_path = args.out_root / "tuning_run_manifest.json"
    write_json(run_log_path, run_log)

    print(f"Saved tuning manifest: {run_log_path}")
    print(
        f"runs_requested={run_log['runs_requested']} runs_planned={run_log['runs_planned']} "
        f"runs_executed={run_log['runs_executed']} skipped_existing={run_log['runs_skipped_existing']}"
    )


if __name__ == "__main__":
    main()
