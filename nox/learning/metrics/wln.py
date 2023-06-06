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

    def reset(self):
        super().reset()
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)