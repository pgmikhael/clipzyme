from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import torchmetrics
from torchmetrics import Metric
from torchmetrics.utilities.compute import auc as compute_auc
import torch


@register_object("reactivity_classification", "metric")
class ReactivityClassification(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        # self.num_classes = kwargs["num_classes"]

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args):
        preds = predictions_dict["preds"]
        target = predictions_dict["golds"]
        assert preds.shape == target.shape

        N = preds.shape[1]
        triu_indices = torch.triu_indices(N, N, offset=1, device=preds.device)

        # preds_upper = preds[:, triu_indices[0], triu_indices[1]]
        # target_upper = target[:, triu_indices[0], triu_indices[1]]

        correct_tensor = (preds == target).all(dim=-1)
        correct = 0
        total = 0
        for i, j in zip(triu_indices[0], triu_indices[1]):
            total += 1
            correct += int(correct_tensor[i, j])

        self.correct += torch.tensor(correct).to(self.device)
        self.total += torch.tensor(total).to(self.device)

    def compute(self):
        stats_dict = {
            "reactivity_accuracy": self.correct.float() / self.total,
        }
        return stats_dict


@register_object("candidate_topk", "metric")
class CandidateTopK(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        for k in args.topk_bonds:
            self.add_state(f"num_correct_{k}", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args):
        candidate_bond_changes = predictions["candidate_bond_changes"]
        batch_real_bond_changes = predictions["real_bond_changes"]
                
        # compute for each reaction
        for idx in range(len(candidate_bond_changes)):
            # sort bonds
            cand_changes = [(min(atom1, atom2), max(atom1, atom2), float(change_type)) for (atom1, atom2, change_type, score) in candidate_bond_changes[idx]]
            gold_bonds = [(min(atom1, atom2), max(atom1, atom2), float(change_type)) for (atom1, atom2, change_type) in batch_real_bond_changes[idx]]

            self.num_total += torch.tensor(num_total).to(self.device)
            for k in self.args.topk_bonds:
                if set(gold_bonds) <= set(cand_changes[:k]):
                    current = getattr(self, f"num_correct_{k}")
                    setattr(self, f"num_correct_{k}", current + torch.tensor(1).to(self.device))            


    def compute(self):
        stats_dict = {
            "reactivity_accuracy": self.correct.float() / self.total,
        }
        return stats_dict
