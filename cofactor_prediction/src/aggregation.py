from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import shutil
import pandas as pd

from .manifests import ensure_dir, write_json


PRIMARY_METRIC = "macro_f1"


def _find_run_dirs(run_root: Path) -> List[Path]:
    return [p.parent for p in run_root.rglob("run_manifest.json") if (p.parent / "metrics.json").exists()]


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def aggregate_publication_outputs(run_root: Path, out_dir: Path) -> Dict:
    ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    manifests_dir = out_dir / "manifests"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(manifests_dir)

    run_dirs = _find_run_dirs(run_root)
    if not run_dirs:
        raise RuntimeError(f"No completed run directories found under {run_root}")

    rows = []
    per_class_rows = []

    for run_dir in run_dirs:
        run_manifest = _load_json(run_dir / "run_manifest.json")
        metrics = _load_json(run_dir / "metrics.json")
        roc_summary_path = run_dir / "roc" / "roc_summary.json"
        roc_summary = _load_json(roc_summary_path) if roc_summary_path.exists() else {}

        row = {
            "run_dir": str(run_dir),
            "split_type": run_manifest["split_type"],
            "seed": run_manifest["seed"],
            "model_family": run_manifest["model_family"],
            "model_name": run_manifest["model_name"],
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "micro_auroc": metrics.get("micro_auroc"),
            "macro_auroc": metrics.get("macro_auroc"),
            "weighted_auroc": metrics.get("weighted_auroc"),
            "best_epoch": run_manifest.get("config", {}).get("best_epoch"),
            "roc_summary_path": str(roc_summary_path) if roc_summary_path else "",
        }
        rows.append(row)

        per_class_auroc = metrics.get("per_class_auroc", {})
        per_class_support = metrics.get("per_class_support", {})
        for class_name, auroc_val in per_class_auroc.items():
            per_class_rows.append(
                {
                    "run_dir": str(run_dir),
                    "split_type": run_manifest["split_type"],
                    "seed": run_manifest["seed"],
                    "model_family": run_manifest["model_family"],
                    "model_name": run_manifest["model_name"],
                    "class_name": class_name,
                    "auroc": auroc_val,
                    "support": per_class_support.get(class_name, 0),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "all_runs_metrics.csv", index=False)

    grouped = (
        df.groupby(["split_type", "model_family", "model_name"], as_index=False)
        .agg(
            n_runs=("macro_f1", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_auroc_mean=("macro_auroc", "mean"),
            macro_auroc_std=("macro_auroc", "std"),
            weighted_auroc_mean=("weighted_auroc", "mean"),
            weighted_auroc_std=("weighted_auroc", "std"),
            micro_auroc_mean=("micro_auroc", "mean"),
            micro_auroc_std=("micro_auroc", "std"),
        )
        .sort_values(["split_type", "model_family", "macro_f1_mean"], ascending=[True, True, False])
    )
    grouped.to_csv(tables_dir / "leaderboard_by_split.csv", index=False)

    main_rows = []
    best_index = []
    for split_type in sorted(df["split_type"].unique()):
        for family in ["knn", "mlp"]:
            subset = grouped[(grouped["split_type"] == split_type) & (grouped["model_family"] == family)]
            if subset.empty:
                continue
            best = subset.sort_values("macro_f1_mean", ascending=False).iloc[0]
            main_rows.append(
                {
                    "split_type": split_type,
                    "model_family": family,
                    "best_model_name": best["model_name"],
                    "macro_f1_mean": best["macro_f1_mean"],
                    "macro_f1_std": best["macro_f1_std"],
                    "macro_auroc_mean": best["macro_auroc_mean"],
                    "macro_auroc_std": best["macro_auroc_std"],
                    "weighted_auroc_mean": best["weighted_auroc_mean"],
                    "weighted_auroc_std": best["weighted_auroc_std"],
                    "micro_auroc_mean": best["micro_auroc_mean"],
                    "micro_auroc_std": best["micro_auroc_std"],
                }
            )
            best_index.append((split_type, family, best["model_name"]))

    main_df = pd.DataFrame(main_rows)
    if main_df.empty:
        main_df = pd.DataFrame(
            columns=[
                "split_type",
                "model_family",
                "best_model_name",
                "macro_f1_mean",
                "macro_f1_std",
                "macro_auroc_mean",
                "macro_auroc_std",
                "weighted_auroc_mean",
                "weighted_auroc_std",
                "micro_auroc_mean",
                "micro_auroc_std",
            ]
        )
    else:
        main_df = main_df.sort_values(["split_type", "model_family"])
    main_df.to_csv(tables_dir / "main_results.csv", index=False)

    md = ["| split_type | model_family | best_model_name | macro_f1_mean | macro_f1_std | macro_auroc_mean | weighted_auroc_mean | micro_auroc_mean |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in main_df.iterrows():
        md.append(
            f"| {r['split_type']} | {r['model_family']} | {r['best_model_name']} | "
            f"{r['macro_f1_mean']:.4f} | {r['macro_f1_std']:.4f} | {r['macro_auroc_mean']:.4f} | "
            f"{r['weighted_auroc_mean']:.4f} | {r['micro_auroc_mean']:.4f} |"
        )
    (tables_dir / "main_results.md").write_text("\n".join(md) + "\n")

    # Per-class AUROC aggregation
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_out_path = tables_dir / "per_class_auroc.csv"
    shared_out_path = tables_dir / "shared_class_comparison.csv"
    if not per_class_df.empty:
        per_class_agg = (
            per_class_df.groupby(["split_type", "model_family", "model_name", "class_name"], as_index=False)
            .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), support_mean=("support", "mean"))
            .sort_values(["split_type", "model_family", "model_name", "auroc_mean"], ascending=[True, True, True, False])
        )
        per_class_agg.to_csv(per_class_out_path, index=False)

        # Shared-class comparison: best MLP model per split, intersection of classes.
        mlp_best = [x for x in best_index if x[1] == "mlp"]
        shared = None
        best_rows = []
        for split_type, family, model_name in mlp_best:
            sub = per_class_agg[
                (per_class_agg["split_type"] == split_type)
                & (per_class_agg["model_family"] == family)
                & (per_class_agg["model_name"] == model_name)
            ]
            cls = set(sub["class_name"].tolist())
            shared = cls if shared is None else (shared & cls)
            best_rows.append((split_type, sub))

        if shared:
            shared_rows = []
            for split_type, sub in best_rows:
                for _, row in sub[sub["class_name"].isin(shared)].iterrows():
                    shared_rows.append(
                        {
                            "split_type": split_type,
                            "class_name": row["class_name"],
                            "auroc_mean": row["auroc_mean"],
                            "auroc_std": row["auroc_std"],
                            "model_name": row["model_name"],
                        }
                    )
            pd.DataFrame(shared_rows).to_csv(shared_out_path, index=False)
        else:
            pd.DataFrame(columns=["split_type", "class_name", "auroc_mean", "auroc_std", "model_name"]).to_csv(
                shared_out_path, index=False
            )
    else:
        pd.DataFrame(columns=["split_type", "model_family", "model_name", "class_name", "auroc_mean", "auroc_std", "support_mean"]).to_csv(
            per_class_out_path, index=False
        )
        pd.DataFrame(columns=["split_type", "class_name", "auroc_mean", "auroc_std", "model_name"]).to_csv(
            shared_out_path, index=False
        )

    # Copy best ROC figures for publication
    for split_type, family, model_name in best_index:
        subset = df[(df["split_type"] == split_type) & (df["model_family"] == family) & (df["model_name"] == model_name)]
        if subset.empty:
            continue
        # Pick median-seed-style representative: highest macro_f1 run.
        run_row = subset.sort_values("macro_f1", ascending=False).iloc[0]
        run_dir = Path(run_row["run_dir"])
        roc_dir = run_dir / "roc"
        for src_name, dst_suffix in [
            ("roc_overall.png", "overall"),
            ("roc_per_class.png", "per_class"),
            ("roc_per_class_auroc_bar.png", "bar"),
        ]:
            src = roc_dir / src_name
            if src.exists():
                dst = figures_dir / f"{split_type}__{family}__{model_name}__{dst_suffix}.png"
                shutil.copy2(src, dst)

    index_payload = {
        "run_root": str(run_root),
        "n_run_dirs": len(run_dirs),
        "primary_metric": PRIMARY_METRIC,
        "tables": [
            str(tables_dir / "all_runs_metrics.csv"),
            str(tables_dir / "leaderboard_by_split.csv"),
            str(tables_dir / "main_results.csv"),
            str(tables_dir / "main_results.md"),
            str(per_class_out_path),
            str(shared_out_path),
        ],
        "figures_dir": str(figures_dir),
        "run_dirs": [str(p) for p in run_dirs],
    }
    write_json(manifests_dir / "experiment_index.json", index_payload)
    return index_payload
