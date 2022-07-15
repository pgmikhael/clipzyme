import argparse
import numpy as np
import torch
import copy
import warnings
from random import Random
from collections import defaultdict
from typing import List
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.rdkit import generate_scaffold
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data.separate import separate
from tqdm import tqdm


@register_object("moleculenet", "dataset")
class MoleNet(AbstractDataset, MoleculeNet):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        MoleculeNet Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        self.args = args

        # self.version = None
        MoleculeNet.__init__(self, root=args.data_dir, name=args.moleculenet_dataset)
        dataset = []
        for idx in tqdm(range(self.len()), position=0):
            data = separate(
                cls=self.data.__class__,
                batch=self.data,
                idx=idx,
                slice_dict=self.slices,
                decrement=False,
            )
            dataset.append(copy.deepcopy(data))

        self.assign_splits(dataset, args.split_probs, args.split_type, seed=args.split_seed)
        self.dataset = []
        for d in dataset:
            if d.split == split_group:
                if args.moleculenet_task is not None:
                    d.y = d.y[:, torch.tensor(args.moleculenet_task)]
                d.has_y = ~torch.isnan(d.y)
                d.y[torch.isnan(d.y)] = 0
                self.dataset.append(d)

    def __getitem__(self, index):
        try:
            return self.dataset[index]

        except Exception:
            warnings.warn("Could not load sample")

    def assign_splits(self, metadata_json, split_probs, method, seed) -> None:
        np.random.seed(seed)
        if method == "random":
            for idx in range(len(metadata_json)):
                metadata_json[idx]['split'] = np.random.choice(
                    ["train", "dev", "test"], p=split_probs
                )
        elif method == "scaffold":
            self.scaffold_split(metadata_json, split_probs, seed)
        else:
            raise NotImplementedError(
                f"SPLIT TYPE {method} NOT DEFINED. OPTIONS ARE: RANDOM or SCAFFOLD."
            )

    def scaffold_split(self, meta: List[dict], split_probs: List[float], seed):
        scaffold_to_indices = defaultdict(list)
        for m_i, m in enumerate(meta):
            scaffold = generate_scaffold(m['smiles'])
            scaffold_to_indices[scaffold].append(m_i)

        # Split
        train_size, val_size, test_size = (
            split_probs[0] * len(meta),
            split_probs[1] * len(meta),
            split_probs[2] * len(meta),
        )
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

        # Seed randomness
        random = Random(seed)

        if (
            self.args.scaffold_balanced
        ):  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(
                list(scaffold_to_indices.values()),
                key=lambda index_set: len(index_set),
                reverse=True,
            )

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

        for idx_list, split in [(train, "train"), (val, "dev"), (test, "test")]:
            for idx in idx_list:
                meta[idx]['split'] = split

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(MoleNet, MoleNet).add_args(parser)
        parser.add_argument(
            "--moleculenet_dataset",
            type=str,
            default="Tox21",
            choices=[
                "ESOL",
                "FreeSolv",
                "Lipo",
                "PCBA",
                "MUV",
                "HIV",
                "BACE",
                "BBPB",
                "Tox21",
                "ToxCast",
                "SIDER",
                "ClinTox",
            ],
            help="moleculenet dataset",
        )
        parser.add_argument(
            "--moleculenet_task",
            type=int,
            nargs="*",
            default=None,
            help="task indices",
        )
        parser.add_argument(
            "--scaffold_balanced",
            action="store_true",
            default=False,
            help="balance the scaffold sets",
        )
