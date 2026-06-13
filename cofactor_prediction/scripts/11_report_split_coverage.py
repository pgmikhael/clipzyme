#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import site
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifests import read_json, write_json, ensure_dir, utc_now_iso


NO_COFACTOR = "no_cofactor"


def _split_assessment(coverage_mean: float, train_only_mean: float) -> str:
    if coverage_mean >= 0.8 and train_only_mean <= 1.0:
        return "Good proxy for seen-cofactor generalization; not a direct proxy for unseen cofactor classes."
    if coverage_mean >= 0.5 and train_only_mean <= 4.0:
        return "Moderate proxy for seen-cofactor generalization; limited for unseen classes."
    return "Weak proxy for broad cofactor generalization; distribution shift and train-only classes are substantial."


def _analyze_seed(report: Dict) -> Dict:
    support_train = report["class_support"]["train"]
    support_test = report["class_support"]["test"]

    positive_classes = [c for c in support_train.keys() if c != NO_COFACTOR]

    train_positive_present = sorted([c for c in positive_classes if support_train.get(c, 0) > 0])
    test_positive_present = sorted([c for c in positive_classes if support_test.get(c, 0) > 0])

    train_only_positive = sorted([c for c in positive_classes if support_train.get(c, 0) > 0 and support_test.get(c, 0) == 0])
    test_only_positive = sorted([c for c in positive_classes if support_train.get(c, 0) == 0 and support_test.get(c, 0) > 0])
    tiny_test_positive = sorted([c for c in positive_classes if 0 < support_test.get(c, 0) <= 2])

    n_train_pos = len(train_positive_present)
    n_test_pos = len(test_positive_present)
    coverage = (n_test_pos / n_train_pos) if n_train_pos > 0 else 0.0

    test_total = int(sum(support_test.values()))
    test_no_cofactor = int(support_test.get(NO_COFACTOR, 0))
    test_positive_total = int(sum(support_test.get(c, 0) for c in positive_classes))

    return {
        "n_valid_positive_classes": len(positive_classes),
        "n_train_positive_present": n_train_pos,
        "n_test_positive_present": n_test_pos,
        "coverage_test_over_train_positive_classes": coverage,
        "n_train_only_positive_classes": len(train_only_positive),
        "n_test_only_positive_classes": len(test_only_positive),
        "n_tiny_test_positive_classes_leq2": len(tiny_test_positive),
        "test_total_samples": test_total,
        "test_no_cofactor_samples": test_no_cofactor,
        "test_no_cofactor_fraction": (test_no_cofactor / test_total) if test_total > 0 else 0.0,
        "test_positive_total_samples": test_positive_total,
        "train_only_positive_classes": train_only_positive,
        "test_only_positive_classes": test_only_positive,
        "tiny_test_positive_classes_leq2": tiny_test_positive,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report cofactor class coverage and proxy quality for random_disjoint and balanced_rule_disjoint_v3."
    )
    p.add_argument("--dataset-root", type=Path, required=True, help="Path to datasets directory under a run root.")
    p.add_argument(
        "--split-types",
        type=str,
        default="random_disjoint,balanced_rule_disjoint_v3",
        help="Comma-separated split types to summarize.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    split_types = [x.strip() for x in args.split_types.split(",") if x.strip()]

    seed_rows: List[Dict] = []
    split_rows: List[Dict] = []
    absent_by_split: Dict[str, Dict[str, List[str]]] = {}

    for split_type in split_types:
        split_dir = args.dataset_root / split_type
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dataset directory: {split_dir}")

        seed_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if not seed_dirs:
            raise RuntimeError(f"No seed_* directories found in {split_dir}")

        per_split_seed_rows = []
        absent_by_seed: Dict[str, List[str]] = {}

        for seed_dir in seed_dirs:
            report_path = seed_dir / "class_filter_report.json"
            report = read_json(report_path)

            seed_analysis = _analyze_seed(report)
            seed_label = seed_dir.name.replace("seed_", "")

            split_manifest_path = Path(report["split_manifest_path"])
            n_testable = None
            if split_manifest_path.exists():
                split_manifest = read_json(split_manifest_path)
                n_testable = split_manifest.get("diagnostics", {}).get("n_testable")

            row = {
                "split_type": split_type,
                "seed": int(seed_label),
                "split_manifest_path": str(split_manifest_path),
                "n_testable_from_split_diagnostics": n_testable,
                **{k: v for k, v in seed_analysis.items() if not isinstance(v, list)},
            }
            seed_rows.append(row)
            per_split_seed_rows.append(row)
            absent_by_seed[seed_label] = seed_analysis["train_only_positive_classes"]

        split_df = pd.DataFrame(per_split_seed_rows)
        coverage_mean = float(split_df["coverage_test_over_train_positive_classes"].mean())
        train_only_mean = float(split_df["n_train_only_positive_classes"].mean())

        split_summary = {
            "split_type": split_type,
            "n_seeds": int(len(split_df)),
            "coverage_test_over_train_positive_classes_mean": coverage_mean,
            "coverage_test_over_train_positive_classes_std": float(split_df["coverage_test_over_train_positive_classes"].std(ddof=1) if len(split_df) > 1 else 0.0),
            "n_train_only_positive_classes_mean": train_only_mean,
            "n_train_only_positive_classes_std": float(split_df["n_train_only_positive_classes"].std(ddof=1) if len(split_df) > 1 else 0.0),
            "n_test_positive_present_mean": float(split_df["n_test_positive_present"].mean()),
            "n_valid_positive_classes_mean": float(split_df["n_valid_positive_classes"].mean()),
            "n_tiny_test_positive_classes_leq2_mean": float(split_df["n_tiny_test_positive_classes_leq2"].mean()),
            "test_no_cofactor_fraction_mean": float(split_df["test_no_cofactor_fraction"].mean()),
            "n_testable_from_split_diagnostics_mean": float(split_df["n_testable_from_split_diagnostics"].dropna().mean()) if split_df["n_testable_from_split_diagnostics"].notna().any() else None,
            "can_directly_evaluate_unseen_cofactor_classes": False,
            "proxy_assessment": _split_assessment(coverage_mean, train_only_mean),
        }
        split_rows.append(split_summary)
        absent_by_split[split_type] = absent_by_seed

    seed_df = pd.DataFrame(seed_rows).sort_values(["split_type", "seed"])
    split_df = pd.DataFrame(split_rows).sort_values(["split_type"])

    seed_csv = args.out_dir / "split_coverage_seed_level.csv"
    split_csv = args.out_dir / "split_coverage_summary.csv"
    seed_df.to_csv(seed_csv, index=False)
    split_df.to_csv(split_csv, index=False)

    payload = {
        "created_at": utc_now_iso(),
        "dataset_root": str(args.dataset_root),
        "split_types": split_types,
        "seed_level_csv": str(seed_csv),
        "summary_csv": str(split_csv),
        "absent_train_only_classes_by_seed": absent_by_split,
        "notes": [
            "Classes absent from test cannot be directly evaluated with multiclass metrics.",
            "This report assesses proxy quality for seen-class generalization, not zero-shot unseen class prediction.",
        ],
    }
    json_path = args.out_dir / "split_coverage_report.json"
    write_json(json_path, payload)

    md_lines = [
        "# Split Coverage Report",
        "",
        f"- Generated: {payload['created_at']}",
        f"- Dataset root: `{args.dataset_root}`",
        "",
        "## Summary",
        "",
        "| split_type | test/train positive class coverage (mean±std) | train-only positive classes (mean±std) | tiny test positive classes <=2 (mean) | test no_cofactor fraction (mean) | n_testable (mean) | proxy assessment |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, r in split_df.iterrows():
        n_testable_mean = (
            ""
            if pd.isna(r["n_testable_from_split_diagnostics_mean"])
            else f"{r['n_testable_from_split_diagnostics_mean']:.2f}"
        )
        md_lines.append(
            f"| {r['split_type']} | {r['coverage_test_over_train_positive_classes_mean']:.3f} ± {r['coverage_test_over_train_positive_classes_std']:.3f} "
            f"| {r['n_train_only_positive_classes_mean']:.2f} ± {r['n_train_only_positive_classes_std']:.2f} "
            f"| {r['n_tiny_test_positive_classes_leq2_mean']:.2f} "
            f"| {r['test_no_cofactor_fraction_mean']:.3f} "
            f"| {n_testable_mean} "
            f"| {r['proxy_assessment']} |"
        )

    md_lines.extend(
        [
            "",
            "## Seed-Level Details",
            "",
            "| split_type | seed | n_valid_positive_classes | n_test_positive_present | n_train_only_positive_classes | n_tiny_test_positive_classes_leq2 | coverage_test_over_train_positive_classes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, r in seed_df.iterrows():
        md_lines.append(
            f"| {r['split_type']} | {int(r['seed'])} | {int(r['n_valid_positive_classes'])} | {int(r['n_test_positive_present'])} | "
            f"{int(r['n_train_only_positive_classes'])} | {int(r['n_tiny_test_positive_classes_leq2'])} | {r['coverage_test_over_train_positive_classes']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Neither split directly evaluates truly unseen cofactor classes (classes absent from test are unevaluable in multiclass metrics).",
            "- `random_disjoint` is generally the better proxy for robust seen-class generalization because train/test class coverage is high and train-only class count is low.",
            "- `balanced_rule_disjoint_v3` is a harsher stress test with larger class-support shift; useful for robustness, but weaker as a stable publication headline metric unless framed explicitly as strict OOD-like difficulty.",
        ]
    )

    md_path = args.out_dir / "split_coverage_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Saved seed-level CSV: {seed_csv}")
    print(f"Saved split summary CSV: {split_csv}")
    print(f"Saved markdown report: {md_path}")
    print(f"Saved json report: {json_path}")


if __name__ == "__main__":
    main()
