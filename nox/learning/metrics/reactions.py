from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import numpy as np
from rdkit import Chem
import torch
from torchmetrics import Metric


def canonicalize_smiles(smiles, return_stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=return_stereo)
    else:
        return ""


@register_object("topk_classification", "metric")
class TopK(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        self.topk_values = args.topk_values
        for k in args.topk_values:
            self.add_state(
                f"num_correct_top{k}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"num_nonstereo_correct_top{k}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"total_top{k}", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

    @property
    def metric_keys(self):
        return ["preds", "golds"]

    def update(self, predictions_dict, args) -> None:
        """Computes top-k accuracy for some defined value(s) of k"""
        preds = predictions_dict["preds"]  # B, k (list)
        golds = predictions_dict["golds"]  # B
        golds = [canonicalize_smiles(g) for g in golds]
        nonstereo_golds = [canonicalize_smiles(g, return_stereo=False) for g in golds]
        ranks = []
        nonstereo_ranks = []
        for top_preds, gold, nonstereo_gold in zip(preds, golds, nonstereo_golds):
            standardized_preds = [canonicalize_smiles(g) for g in top_preds]
            standardized_nonstereo_preds = [canonicalize_smiles(g, return_stereo=False) for g in top_preds]
            matches = [p == gold for p in standardized_preds]
            nonstereo_matches = [p == nonstereo_gold for p in standardized_nonstereo_preds]
            if sum(matches) > 0:
                match_idx = matches.index(True) + 1
            else:
                match_idx = 0
            if sum(nonstereo_matches):
                nonstereo_match_idx = nonstereo_matches.index(True) + 1
            else:
                nonstereo_match_idx = 0
            ranks.append(match_idx)
            nonstereo_ranks.append(nonstereo_match_idx)

        ranks_np = np.array(ranks)
        nonstereo_ranks = np.array(nonstereo_ranks)

        for k in self.topk_values:
            num_samples = len(golds)
            num_correct = np.logical_and(ranks_np > 0, ranks_np <= k).sum()
            num_nonstereo_correct = np.logical_and(nonstereo_ranks > 0, nonstereo_ranks <= k).sum()
            last_nonstereo_correct = getattr(self, f"num_nonstereo_correct_top{k}")
            last_correct = getattr(self, f"num_correct_top{k}")
            last_total = getattr(self, f"total_top{k}")

            setattr(
                self, f"num_nonstereo_correct_top{k}", last_nonstereo_correct + torch.tensor(num_nonstereo_correct)
            )
            setattr(
                self, f"num_correct_top{k}", last_correct + torch.tensor(num_correct)
            )
            setattr(self, f"total_top{k}", last_total + torch.tensor(num_samples))

    def compute(self) -> Dict:
        stats_dict = {}
        stats_dict.update({
            f"top_{k}": getattr(self, f"num_correct_top{k}").float()
            / getattr(self, f"total_top{k}")
            for k in self.topk_values
        })
        stats_dict.update({
            f"nonstereo_top_{k}": getattr(self, f"num_nonstereo_correct_top{k}").float()
            / getattr(self, f"total_top{k}")
            for k in self.topk_values
        })

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

@register_object("mol_topk", "metric")
class MolTopK(Metric, Nox):
    def __init__(self, args):
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        self.mol_topk_pred_key = args.mol_topk_pred_key
        self.mol_topk_gold_key = args.mol_topk_gold_key

        self.topk_values = args.mol_topk_values
        self.add_state("percent_coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
        for k in args.mol_topk_values:
            self.add_state(
                f"num_correct_top{k}", default=torch.tensor(0), dist_reduce_fx="sum"
            )
            self.add_state(
                f"total_top{k}", default=torch.tensor(0), dist_reduce_fx="sum"
            )

    @property
    def metric_keys(self):
        return ["preds", "golds"] + [self.mol_topk_pred_key, self.mol_topk_gold_key]

    def update(self, predictions_dict, args) -> None:
        """Computes top-k accuracy for some defined value(s) of k"""
        preds = predictions_dict[args.mol_topk_pred_key]  # B (list)
        golds = predictions_dict[args.mol_topk_gold_key]  # B, k (list)
        golds = [[canonicalize_smiles(g) for g in glist] for glist in golds]
        ranks = []
        for top_preds, gold in zip(preds, golds):
            if not isinstance(top_preds, list):
                top_preds = [top_preds]
            standardized_preds = [canonicalize_smiles(g) for g in top_preds]
            matches = [p in gold for p in standardized_preds]    
            match_idx = matches.index(True) + 1 if sum(matches) > 0 else 0
            ranks.append(match_idx)

            self.percent_coverage += len(set(gold).intersection(set(top_preds))) / len(gold) # max(len(gold), len(set(top_preds)))


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
            f"mol_top_{k}": getattr(self, f"num_correct_top{k}")
            / getattr(self, f"total_top{k}")
            for k in self.topk_values
        }
        stats_dict["mol_percent_coverage"] = self.percent_coverage / getattr(self, f"total_top{self.topk_values[0]}")

        return stats_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--mol_topk_values",
            type=int,
            nargs="+",
            default=[1],
            help="Values of k for which to obtain top-k accuracies",
        )
        parser.add_argument(
            "--mol_topk_gold_key",
            type=str,
            default="all_smiles",
            help="key for gold",
        )
        parser.add_argument(
            "--mol_topk_pred_key",
            type=str,
            default="pred_smiles",
            help="key for predictions",
        )
