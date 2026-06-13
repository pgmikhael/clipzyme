from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.split_builders import (
    create_balanced_rule_disjoint_v3,
    create_random_disjoint_split,
)


def _positive_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"protein_id": "p1", "cofactors": ["cfA"], "rule_ids": ["r1"], "embedding_path": "x1"},
            {"protein_id": "p2", "cofactors": ["cfA"], "rule_ids": ["r1"], "embedding_path": "x2"},
            {"protein_id": "p3", "cofactors": ["cfB"], "rule_ids": ["r2"], "embedding_path": "x3"},
            {"protein_id": "p4", "cofactors": ["cfB"], "rule_ids": ["r2"], "embedding_path": "x4"},
            {"protein_id": "p5", "cofactors": ["cfC"], "rule_ids": ["r3"], "embedding_path": "x5"},
            {"protein_id": "p6", "cofactors": ["cfC"], "rule_ids": ["r4"], "embedding_path": "x6"},
            {"protein_id": "p7", "cofactors": ["cfD"], "rule_ids": [], "embedding_path": "x7"},
            {"protein_id": "p8", "cofactors": ["cfE"], "rule_ids": [], "embedding_path": "x8"},
        ]
    )


def _negative_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"protein_id": "n1", "cofactors": [], "rule_ids": [], "embedding_path": "n1"},
            {"protein_id": "n2", "cofactors": [], "rule_ids": [], "embedding_path": "n2"},
            {"protein_id": "n3", "cofactors": [], "rule_ids": [], "embedding_path": "n3"},
            {"protein_id": "n4", "cofactors": [], "rule_ids": [], "embedding_path": "n4"},
            {"protein_id": "n5", "cofactors": [], "rule_ids": [], "embedding_path": "n5"},
            {"protein_id": "n6", "cofactors": [], "rule_ids": [], "embedding_path": "n6"},
        ]
    )


def test_random_disjoint_split_has_no_overlap(tmp_path: Path) -> None:
    manifest = create_random_disjoint_split(
        positive_inventory=_positive_inventory(),
        negative_inventory=_negative_inventory(),
        output_path=tmp_path / "random.json",
        seed=42,
    )

    assert manifest.diagnostics["positive_overlap"]["train_dev"] == 0
    assert manifest.diagnostics["positive_overlap"]["train_test"] == 0
    assert manifest.diagnostics["positive_overlap"]["dev_test"] == 0
    assert manifest.diagnostics["negative_overlap"]["train_dev"] == 0
    assert manifest.diagnostics["negative_overlap"]["train_test"] == 0
    assert manifest.diagnostics["negative_overlap"]["dev_test"] == 0
    assert (tmp_path / "random.json").exists()


def test_balanced_rule_split_has_no_rule_or_protein_overlap(tmp_path: Path) -> None:
    manifest = create_balanced_rule_disjoint_v3(
        positive_inventory=_positive_inventory(),
        negative_inventory=_negative_inventory(),
        output_path=tmp_path / "balanced.json",
        seed=1337,
        n_restarts=16,
    )

    assert manifest.diagnostics["positive_overlap"]["train_dev"] == 0
    assert manifest.diagnostics["positive_overlap"]["train_test"] == 0
    assert manifest.diagnostics["positive_overlap"]["dev_test"] == 0

    assert manifest.diagnostics["rule_overlap"]["train_dev"] == 0
    assert manifest.diagnostics["rule_overlap"]["train_test"] == 0
    assert manifest.diagnostics["rule_overlap"]["dev_test"] == 0

    assert len(manifest.positive["dev"]) > 0
    assert len(manifest.positive["test"]) > 0
    assert (tmp_path / "balanced.json").exists()
