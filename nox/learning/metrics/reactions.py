from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import numpy as np
from rdkit import Chem
import torch
from torchmetrics import Metric


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ""


@register_object("topk_classification", "metric")
class TopK(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        self.topk_values = args.topk_values
        for k in args.topk_values:
            self.add_state(
                f"num_correct_top{k}", default=torch.tensor(0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"total_top{k}", default=torch.tensor(0), dist_reduce_fx="sum"
            )

    @property
    def metric_keys(self):
        return ["preds", "golds"]

    def update(self, predictions_dict, args) -> None:
        """Computes top-k accuracy for some defined value(s) of k"""
        preds = predictions_dict["preds"]  # B, k (list)
        golds = predictions_dict["golds"]  # B
        golds = [canonicalize_smiles(g) for g in golds]
        ranks = []
        for top_preds, gold in zip(preds, golds):
            standardized_preds = [canonicalize_smiles(g) for g in top_preds]
            matches = [p == gold for p in standardized_preds]
            if sum(matches) > 0:
                match_idx = matches.index(True) + 1
            else:
                match_idx = 0
            ranks.append(match_idx)

        ranks_np = np.array(ranks)

        for k in self.topk_values:
            num_samples = len(golds)
            num_correct = np.logical_and(ranks_np > 0, ranks_np <= k).sum()
            last_correct = getattr(self, f"num_correct_top{k}")
            last_total = getattr(self, f"total_top{k}")

            setattr(
                self, f"num_correct_top{k}", last_correct + torch.tensor(num_correct)
            )
            setattr(self, f"total_top{k}", last_total + torch.tensor(num_samples))

    def compute(self) -> Dict:
        stats_dict = {
            f"top_{k}": getattr(self, f"num_correct_top{k}")
            / getattr(self, f"total_top{k}")
            for k in self.topk_values
        }

        return stats_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--topk_values",
            type=int,
            nargs="+",
            default=[1],
            help="Values of k for which to obtain top-k accuracies",
        )
