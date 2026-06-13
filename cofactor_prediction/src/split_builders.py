from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .manifests import SplitManifest, utc_now_iso, write_json
from .pickle_compat import install_numpy_pickle_compat_aliases


SPLITS = ("train", "dev", "test")


@dataclass
class SplitResult:
    positive: Dict[str, List[str]]
    negative: Dict[str, List[str]]
    diagnostics: Dict


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _to_set(values: Iterable[str]) -> Set[str]:
    return {str(v) for v in values}


def _overlap_counts(splits: Dict[str, Sequence[str]]) -> Dict[str, int]:
    train_set = _to_set(splits["train"])
    dev_set = _to_set(splits["dev"])
    test_set = _to_set(splits["test"])
    return {
        "train_dev": len(train_set & dev_set),
        "train_test": len(train_set & test_set),
        "dev_test": len(dev_set & test_set),
    }


def _random_disjoint(ids: Sequence[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    unique_ids = np.array(sorted(set(str(x) for x in ids)), dtype=object)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(round(n * ratios[0]))
    n_dev = int(round(n * ratios[1]))
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n - 2, 0)
    n_dev = min(max(n_dev, 1), n - n_train - 1) if n - n_train >= 2 else max(n - n_train - 1, 0)

    train = unique_ids[:n_train].tolist()
    dev = unique_ids[n_train : n_train + n_dev].tolist()
    test = unique_ids[n_train + n_dev :].tolist()

    return {"train": train, "dev": dev, "test": test}


def _cofactor_support(positive_df: pd.DataFrame, split_ids: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, int]]:
    protein_to_cofactors: Dict[str, Set[str]] = defaultdict(set)
    for _, row in positive_df.iterrows():
        pid = str(row["protein_id"])
        cofactors = row.get("cofactors", [])
        if isinstance(cofactors, list):
            protein_to_cofactors[pid].update(str(cf) for cf in cofactors)

    support: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        c = Counter()
        for pid in sorted(set(split_ids[split])):
            c.update(protein_to_cofactors.get(str(pid), set()))
        support[split] = dict(sorted(c.items()))
    return support


def _testable_cofactors(support: Dict[str, Dict[str, int]], min_train: int, min_test: int) -> List[str]:
    train = support["train"]
    test = support["test"]
    all_cofactors = sorted(set(train.keys()) | set(test.keys()))
    return [cf for cf in all_cofactors if train.get(cf, 0) >= min_train and test.get(cf, 0) >= min_test]


def create_legacy_original_split(
    positive_train_pkl: Path,
    positive_dev_pkl: Path,
    positive_test_pkl: Path,
    negative_train_csv: Path,
    negative_dev_csv: Path,
    negative_test_csv: Path,
    output_path: Path,
    seed: int,
    min_train_support: int = 5,
    min_test_support: int = 1,
) -> SplitManifest:
    install_numpy_pickle_compat_aliases()
    pos = {
        "train": pd.read_pickle(positive_train_pkl),
        "dev": pd.read_pickle(positive_dev_pkl),
        "test": pd.read_pickle(positive_test_pkl),
    }
    neg = {
        "train": pd.read_csv(negative_train_csv),
        "dev": pd.read_csv(negative_dev_csv),
        "test": pd.read_csv(negative_test_csv),
    }

    pos_ids = {split: sorted(set(str(x) for x in pos[split]["protein_id"].tolist())) for split in SPLITS}
    neg_ids = {split: sorted(set(str(x) for x in neg[split]["protein_id"].tolist())) for split in SPLITS}

    positive_merged = pd.concat(pos.values(), ignore_index=True)
    support = _cofactor_support(positive_merged, pos_ids)
    testable = _testable_cofactors(support, min_train=min_train_support, min_test=min_test_support)

    diagnostics = {
        "positive_sizes": {k: len(v) for k, v in pos_ids.items()},
        "negative_sizes": {k: len(v) for k, v in neg_ids.items()},
        "positive_overlap": _overlap_counts(pos_ids),
        "negative_overlap": _overlap_counts(neg_ids),
        "cofactor_support": support,
        "testable_cofactors": testable,
        "n_testable": len(testable),
    }

    manifest = SplitManifest(
        split_type="legacy_original",
        seed=int(seed),
        created_at=utc_now_iso(),
        source={
            "positive_train_pkl": str(positive_train_pkl),
            "positive_dev_pkl": str(positive_dev_pkl),
            "positive_test_pkl": str(positive_test_pkl),
            "negative_train_csv": str(negative_train_csv),
            "negative_dev_csv": str(negative_dev_csv),
            "negative_test_csv": str(negative_test_csv),
        },
        positive=pos_ids,
        negative=neg_ids,
        diagnostics=diagnostics,
    )
    manifest.write(output_path)
    return manifest


def create_random_disjoint_split(
    positive_inventory: pd.DataFrame,
    negative_inventory: pd.DataFrame,
    output_path: Path,
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_train_support: int = 5,
    min_test_support: int = 1,
) -> SplitManifest:
    pos_ids = _random_disjoint(positive_inventory["protein_id"].tolist(), ratios=ratios, seed=seed)
    neg_ids = _random_disjoint(negative_inventory["protein_id"].tolist(), ratios=ratios, seed=seed + 17)

    support = _cofactor_support(positive_inventory, pos_ids)
    testable = _testable_cofactors(support, min_train=min_train_support, min_test=min_test_support)

    diagnostics = {
        "positive_sizes": {k: len(v) for k, v in pos_ids.items()},
        "negative_sizes": {k: len(v) for k, v in neg_ids.items()},
        "positive_overlap": _overlap_counts(pos_ids),
        "negative_overlap": _overlap_counts(neg_ids),
        "cofactor_support": support,
        "testable_cofactors": testable,
        "n_testable": len(testable),
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
    }

    manifest = SplitManifest(
        split_type="random_disjoint",
        seed=int(seed),
        created_at=utc_now_iso(),
        source={"positive_inventory": "global_dedup", "negative_inventory": "global_dedup"},
        positive=pos_ids,
        negative=neg_ids,
        diagnostics=diagnostics,
    )
    manifest.write(output_path)
    return manifest


def _build_components(positive_inventory: pd.DataFrame) -> List[Dict]:
    rows = positive_inventory.reset_index(drop=True)
    n = len(rows)
    uf = UnionFind(n)

    rule_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in rows.iterrows():
        rules = row.get("rule_ids", [])
        if isinstance(rules, list):
            for rule in rules:
                rule_to_indices[str(rule)].append(int(idx))

    for indices in rule_to_indices.values():
        if len(indices) <= 1:
            continue
        head = indices[0]
        for i in indices[1:]:
            uf.union(head, i)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    components: List[Dict] = []
    for root, members in groups.items():
        protein_ids = [str(rows.iloc[i]["protein_id"]) for i in members]
        rules = set()
        cofactor_counts = Counter()
        for i in members:
            row = rows.iloc[i]
            for rule in row.get("rule_ids", []) if isinstance(row.get("rule_ids", []), list) else []:
                rules.add(str(rule))
            cofactors = row.get("cofactors", [])
            if isinstance(cofactors, list):
                cofactor_counts.update(set(str(cf) for cf in cofactors))

        components.append(
            {
                "component_id": int(root),
                "protein_ids": protein_ids,
                "size": len(protein_ids),
                "rules": sorted(rules),
                "cofactor_counts": dict(cofactor_counts),
            }
        )

    return components


def _objective_score(
    train_counts: Counter,
    dev_counts: Counter,
    test_counts: Counter,
    split_sizes: Dict[str, int],
    targets: Dict[str, int],
    cofactor_universe: Sequence[str],
    min_train_support: int,
    min_test_support: int,
) -> float:
    testable = sum(
        1
        for cf in cofactor_universe
        if train_counts.get(cf, 0) >= min_train_support and test_counts.get(cf, 0) >= min_test_support
    )
    test_coverage = sum(1 for cf in cofactor_universe if test_counts.get(cf, 0) > 0)

    ratio_penalty = 0.0
    for split in SPLITS:
        target = max(1, targets[split])
        ratio_penalty += abs(split_sizes[split] - target) / target

    dev_penalty = 0 if split_sizes["dev"] > 0 else 1
    test_penalty = 0 if split_sizes["test"] > 0 else 1

    return 1000.0 * testable + 20.0 * test_coverage - 120.0 * ratio_penalty - 300.0 * dev_penalty - 300.0 * test_penalty


def _local_gain(
    split: str,
    comp: Dict,
    train_counts: Counter,
    dev_counts: Counter,
    test_counts: Counter,
    split_sizes: Dict[str, int],
    targets: Dict[str, int],
    min_train_support: int,
    min_test_support: int,
) -> float:
    comp_size = comp["size"]
    comp_counts = comp["cofactor_counts"]

    gain = 0.0
    target = max(1, targets[split])
    gain += 2.0 * ((targets[split] - split_sizes[split]) / target)

    if split == "train":
        for cf, count in comp_counts.items():
            deficit = max(0, min_train_support - train_counts.get(cf, 0))
            gain += min(deficit, count) * 3.0

    if split == "test":
        for cf, count in comp_counts.items():
            if test_counts.get(cf, 0) == 0:
                gain += 8.0
            if train_counts.get(cf, 0) >= min_train_support and test_counts.get(cf, 0) < min_test_support:
                gain += 10.0
            if train_counts.get(cf, 0) == 0:
                gain -= 2.5

    if split == "dev":
        gain += 1.0
        if split_sizes["dev"] == 0:
            gain += 2.0

    if split_sizes[split] + comp_size > targets[split] * 1.4:
        gain -= 5.0

    return gain


def create_balanced_rule_disjoint_v3(
    positive_inventory: pd.DataFrame,
    negative_inventory: pd.DataFrame,
    output_path: Path,
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_train_support: int = 5,
    min_test_support: int = 1,
    n_restarts: int = 64,
) -> SplitManifest:
    rng_master = np.random.default_rng(seed)

    components = _build_components(positive_inventory)
    if not components:
        raise RuntimeError("No positive components were constructed for balanced split")

    total_pos = int(sum(comp["size"] for comp in components))
    targets = {
        "train": max(1, int(round(total_pos * ratios[0]))),
        "dev": max(1, int(round(total_pos * ratios[1]))),
        "test": max(1, total_pos - int(round(total_pos * ratios[0])) - int(round(total_pos * ratios[1]))),
    }

    cofactor_universe = sorted(
        {
            cf
            for _, row in positive_inventory.iterrows()
            for cf in (row.get("cofactors", []) if isinstance(row.get("cofactors", []), list) else [])
        }
    )

    best = None
    best_score = float("-inf")

    for _ in range(n_restarts):
        order = np.arange(len(components))
        rng_master.shuffle(order)

        assignment = {"train": [], "dev": [], "test": []}
        split_sizes = {"train": 0, "dev": 0, "test": 0}
        train_counts: Counter = Counter()
        dev_counts: Counter = Counter()
        test_counts: Counter = Counter()

        for comp_idx in order:
            comp = components[int(comp_idx)]

            gains = {}
            for split in SPLITS:
                gains[split] = _local_gain(
                    split,
                    comp,
                    train_counts,
                    dev_counts,
                    test_counts,
                    split_sizes,
                    targets,
                    min_train_support,
                    min_test_support,
                )

            max_gain = max(gains.values())
            best_splits = [s for s, g in gains.items() if abs(g - max_gain) < 1e-9]
            chosen = best_splits[int(rng_master.integers(0, len(best_splits)))]

            assignment[chosen].append(comp)
            split_sizes[chosen] += comp["size"]
            if chosen == "train":
                train_counts.update(comp["cofactor_counts"])
            elif chosen == "dev":
                dev_counts.update(comp["cofactor_counts"])
            else:
                test_counts.update(comp["cofactor_counts"])

        # Guarantee non-empty dev/test by moving the smallest train component if needed.
        for split in ("dev", "test"):
            if split_sizes[split] == 0 and assignment["train"]:
                assignment["train"].sort(key=lambda c: c["size"])
                moved = assignment["train"].pop(0)
                assignment[split].append(moved)
                split_sizes["train"] -= moved["size"]
                split_sizes[split] += moved["size"]
                train_counts.subtract(moved["cofactor_counts"])
                if split == "dev":
                    dev_counts.update(moved["cofactor_counts"])
                else:
                    test_counts.update(moved["cofactor_counts"])

        score = _objective_score(
            train_counts,
            dev_counts,
            test_counts,
            split_sizes,
            targets,
            cofactor_universe,
            min_train_support,
            min_test_support,
        )
        if score > best_score:
            best_score = score
            best = {
                "assignment": assignment,
                "split_sizes": split_sizes,
                "train_counts": train_counts.copy(),
                "dev_counts": dev_counts.copy(),
                "test_counts": test_counts.copy(),
                "score": score,
            }

    assert best is not None

    pos_ids = {
        split: sorted({pid for comp in best["assignment"][split] for pid in comp["protein_ids"]})
        for split in SPLITS
    }

    neg_ids = _random_disjoint(
        negative_inventory["protein_id"].tolist(), ratios=ratios, seed=seed + 1009
    )

    # Rule overlap diagnostics
    split_rules = {
        split: set(rule for comp in best["assignment"][split] for rule in comp["rules"])
        for split in SPLITS
    }
    rule_overlap = {
        "train_dev": len(split_rules["train"] & split_rules["dev"]),
        "train_test": len(split_rules["train"] & split_rules["test"]),
        "dev_test": len(split_rules["dev"] & split_rules["test"]),
    }

    support = {
        "train": dict(sorted(best["train_counts"].items())),
        "dev": dict(sorted(best["dev_counts"].items())),
        "test": dict(sorted(best["test_counts"].items())),
    }
    testable = _testable_cofactors(support, min_train=min_train_support, min_test=min_test_support)

    diagnostics = {
        "score": float(best["score"]),
        "n_components": len(components),
        "positive_sizes": {k: len(v) for k, v in pos_ids.items()},
        "negative_sizes": {k: len(v) for k, v in neg_ids.items()},
        "positive_overlap": _overlap_counts(pos_ids),
        "negative_overlap": _overlap_counts(neg_ids),
        "rule_overlap": rule_overlap,
        "cofactor_support": support,
        "testable_cofactors": testable,
        "n_testable": len(testable),
        "targets": targets,
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "min_train_support": min_train_support,
        "min_test_support": min_test_support,
        "n_restarts": n_restarts,
    }

    manifest = SplitManifest(
        split_type="balanced_rule_disjoint_v3",
        seed=int(seed),
        created_at=utc_now_iso(),
        source={"positive_inventory": "global_dedup+rule_components", "negative_inventory": "global_dedup"},
        positive=pos_ids,
        negative=neg_ids,
        diagnostics=diagnostics,
    )
    manifest.write(output_path)
    return manifest


def load_split_manifest(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)
