#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import subprocess
import sys
from typing import Dict, Iterable, List

USER_SITE = str(site.getusersitepackages())
HOME_DIR = str(Path.home())
sys.path[:] = [
    p
    for p in sys.path
    if p != USER_SITE and not (p.startswith(HOME_DIR) and ".local" in p and "site-packages" in p)
]

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifests import ensure_dir, utc_now_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full cofactor reimplementation experiment matrix.")
    p.add_argument("--config", type=Path, default=Path("cofactor_prediction/configs/experiment_grid.yaml"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--publication-dir",
        type=Path,
        default=Path("cofactor_prediction/publication"),
        help="Publication output directory.",
    )
    return p.parse_args()


def _load_config(path: Path) -> Dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _as_csv(values: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in values)


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    seeds = [int(x) for x in cfg.get("seeds", [42, 1337, 2025])]
    split_types = list(cfg.get("split_types", ["legacy_original", "random_disjoint", "balanced_rule_disjoint_v3"]))

    split_cfg = cfg.get("split", {})
    split_min_train = int(split_cfg.get("min_train_support", 5))
    split_min_test = int(split_cfg.get("min_test_support", 1))
    n_restarts = int(split_cfg.get("n_restarts", 64))

    dataset_cfg = cfg.get("dataset", {})
    include_no_cofactor = bool(dataset_cfg.get("include_no_cofactor", True))
    dataset_min_train = int(dataset_cfg.get("min_train_support", 5))

    knn_cfg = cfg.get("knn", {})
    run_knn = bool(knn_cfg.get("enabled", True))
    k_list = [int(x) for x in knn_cfg.get("k_list", [1, 3, 5, 10, 20, 50])]

    mlp_cfg = cfg.get("mlp", {})
    run_mlp = bool(mlp_cfg.get("enabled", True))
    mlp_config_file = Path(mlp_cfg.get("config_file", str(args.config)))
    mlp_config_names = list(
        mlp_cfg.get(
            "config_names",
            [
                "mlp_ce_unweighted_dropout0",
                "mlp_ce_weighted_dropout0",
                "mlp_focal_weighted_gamma2_dropout0",
                "mlp_ce_weighted_dropout03",
            ],
        )
    )

    paths_config = Path(cfg.get("paths_config", "cofactor_prediction/configs/paths.yaml"))
    inventory_cfg = cfg.get("inventory", {})
    drop_missing_embeddings = bool(inventory_cfg.get("drop_missing_embeddings", True))

    out_root = args.out_root
    ensure_dir(out_root)

    inventory_dir = out_root / "inventory"
    split_dir = out_root / "splits"
    dataset_dir = out_root / "datasets"
    run_dir = out_root / "model_runs"
    ensure_dir(inventory_dir)
    ensure_dir(split_dir)
    ensure_dir(dataset_dir)
    ensure_dir(run_dir)

    # 01 Build inventory.
    build_inventory_cmd = [
        sys.executable,
        str(REPO_ROOT / "cofactor_prediction/scripts/01_build_inventory.py"),
        "--paths-config",
        str(paths_config),
        "--out-dir",
        str(inventory_dir),
    ]
    if drop_missing_embeddings:
        build_inventory_cmd.append("--drop-missing-embeddings")
    else:
        build_inventory_cmd.append("--no-drop-missing-embeddings")
    _run(build_inventory_cmd)

    dataset_manifests: List[Path] = []

    # 02 Create splits + 03 Prepare datasets.
    for split_type in split_types:
        for seed in seeds:
            split_out_dir = split_dir / split_type
            ensure_dir(split_out_dir)
            split_manifest_path = split_out_dir / f"{split_type}_seed{seed}.json"

            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "cofactor_prediction/scripts/02_create_splits.py"),
                    "--split-type",
                    split_type,
                    "--seed",
                    str(seed),
                    "--out-dir",
                    str(split_out_dir),
                    "--inventory-dir",
                    str(inventory_dir),
                    "--paths-config",
                    str(paths_config),
                    "--min-train-support",
                    str(split_min_train),
                    "--min-test-support",
                    str(split_min_test),
                    "--n-restarts",
                    str(n_restarts),
                ]
            )

            ds_out_dir = dataset_dir / split_type / f"seed_{seed}"
            ensure_dir(ds_out_dir)
            dataset_cmd = [
                sys.executable,
                str(REPO_ROOT / "cofactor_prediction/scripts/03_prepare_datasets.py"),
                "--split-manifest",
                str(split_manifest_path),
                "--inventory-dir",
                str(inventory_dir),
                "--paths-config",
                str(paths_config),
                "--min-train-support",
                str(dataset_min_train),
                "--out-dir",
                str(ds_out_dir),
            ]
            if include_no_cofactor:
                dataset_cmd.append("--include-no-cofactor")
            else:
                dataset_cmd.append("--no-include-no-cofactor")
            _run(dataset_cmd)

            dataset_manifest_path = ds_out_dir / "dataset_manifest.json"
            dataset_manifests.append(dataset_manifest_path)

    # 04 KNN + 05 MLP.
    for dataset_manifest_path in dataset_manifests:
        split_part = dataset_manifest_path.parts[-3]
        seed_part = dataset_manifest_path.parts[-2]
        seed = int(seed_part.replace("seed_", ""))

        if run_knn:
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / "cofactor_prediction/scripts/04_run_knn.py"),
                    "--dataset",
                    str(dataset_manifest_path),
                    "--k-list",
                    _as_csv(k_list),
                    "--seed",
                    str(seed),
                    "--out-dir",
                    str(run_dir),
                ]
            )

        if run_mlp:
            for config_name in mlp_config_names:
                _run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "cofactor_prediction/scripts/05_run_mlp.py"),
                        "--dataset",
                        str(dataset_manifest_path),
                        "--config",
                        str(mlp_config_file),
                        "--config-name",
                        config_name,
                        "--seed",
                        str(seed),
                        "--out-dir",
                        str(run_dir),
                    ]
                )

    # 06 Evaluate + 07 ROC for each run directory.
    run_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and (p / "predictions.npz").exists()])
    for one_run in run_dirs:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "cofactor_prediction/scripts/06_evaluate_runs.py"),
                "--run-dir",
                str(one_run),
            ]
        )
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "cofactor_prediction/scripts/07_make_roc_artifacts.py"),
                "--run-dir",
                str(one_run),
            ]
        )

    # 08 Aggregate publication artifacts.
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "cofactor_prediction/scripts/08_aggregate_publication_tables.py"),
            "--run-root",
            str(run_dir),
            "--out-dir",
            str(args.publication_dir),
        ]
    )

    print("Completed full pipeline")
    print(f"out_root={out_root}")
    print(f"publication_dir={args.publication_dir}")
    print(f"timestamp_utc={utc_now_iso()}")


if __name__ == "__main__":
    main()
