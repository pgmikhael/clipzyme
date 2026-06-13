#!/usr/bin/env python3
"""Bundle the cofactor MLP ensemble into a single standalone checkpoint.

Reads the per-seed model run directories (each with ``run_manifest.json`` +
``best_model.pt``) and writes one self-contained ``.pt`` file that
``clipzyme.lightning.clipzyme_plus.CLIPZymePlus`` can load on its own — with no
dependency on the ``cofactor_prediction`` package or its run-dir layout.

The bundle is the artifact distributed on Zenodo. Re-run this after retraining
the ensemble to refresh it.

Bundle format (``format_version=1``)::

    {
      "format_version": 1,
      "model_name": str,
      "split_type": str,
      "class_names": [str, ...],          # union across members
      "members": [
          {"seed", "model_name", "input_dim", "hidden_dims",
           "dropout", "class_names", "state_dict"},
          ...
      ],
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "cofactor_prediction"
    / "runs"
    / "mlp_tuning_20260208_175222"
    / "model_runs"
)
DEFAULT_SPLIT_TYPE = "random_disjoint"
DEFAULT_MODEL_NAME = "mlp_tune_ce_weighted_dropout03_lr1e3_wd1e3"
DEFAULT_SEEDS = (42, 1337, 2025)
DEFAULT_OUTPUT = REPO_ROOT / "files" / "clipzyme_plus_cofactor_ensemble.pt"


def _class_names_from_manifest(manifest: dict) -> List[str]:
    idx_to_class = manifest["config"]["idx_to_class"]
    return [str(idx_to_class[str(i)]) for i in range(len(idx_to_class))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--split-type", type=str, default=DEFAULT_SPLIT_TYPE)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    members = []
    for seed in args.seeds:
        run_dir = args.run_root / f"{args.split_type}__seed{seed}__{args.model_name}"
        manifest = json.load((run_dir / "run_manifest.json").open())
        config = manifest["config"]
        class_names = _class_names_from_manifest(manifest)
        state_dict = torch.load(run_dir / "best_model.pt", map_location="cpu")
        input_dim = int(state_dict["net.0.weight"].shape[1])
        members.append(
            {
                "seed": int(manifest["seed"]),
                "model_name": str(manifest["model_name"]),
                "input_dim": input_dim,
                "hidden_dims": [int(x) for x in config["hidden_dims"]],
                "dropout": float(config["dropout"]),
                "class_names": class_names,
                "state_dict": {k: v.cpu() for k, v in state_dict.items()},
            }
        )
        print(
            f"packed seed={seed} input_dim={input_dim} "
            f"hidden={config['hidden_dims']} dropout={config['dropout']} "
            f"n_classes={len(class_names)}"
        )

    merged: List[str] = []
    for member in members:
        for name in member["class_names"]:
            if name not in merged:
                merged.append(name)

    bundle = {
        "format_version": 1,
        "model_name": members[0]["model_name"],
        "split_type": args.split_type,
        "class_names": merged,
        "members": members,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(
        f"\nWrote {args.output} "
        f"({args.output.stat().st_size / 1e6:.2f} MB, {len(members)} members, "
        f"{len(merged)} classes)"
    )


if __name__ == "__main__":
    main()
