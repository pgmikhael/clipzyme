import argparse
import numpy as np
import copy
import warnings
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from torch_geometric.datasets import MoleculeNet


@register_object("gsm_link", "dataset")
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
        dataset = copy.deepcopy(self._data_list)
        self.assign_splits(dataset, seed=args.split_seed)
        self.dataset = []
        for d in dataset:
            if d.split == split_group:
                d.y = d.y[np.array(args.moleculenet_task)]
                self.dataset.append(d)

    def __getitem__(self, index):
        try:
            return self.dataset[index]

        except Exception:
            warnings.warn("Could not load sample")

    def assign_splits(self, metadata_json, seed) -> None:
        np.random.seed(seed)
        for idx in range(len(metadata_json)):
            metadata_json[idx].split = np.random.choice(
                ["train", "dev", "test"], p=self.args.split_probs
            )

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
            default=0,
            help="task indices",
        )
