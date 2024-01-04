from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import torchmetrics
from torchmetrics import Metric, MetricCollection


@register_object("hierarchical_ec_classification", "metric")
class HierarchicalECClassification(Metric, Nox):
    def __init__(self, args) -> None:
        """
        Computes standard classification metrics for EC prediction
        """
        super().__init__()
        self.task_type = "multilabel"
        self.levels = sorted(args.ec_levels)
        metrics = {}
        for level in sorted(self.levels):
            metrics[f"f1_ec{level}"] = torchmetrics.F1Score(
                task=self.task_type,
                num_labels=len(args.ec_levels[level]),
                average="weighted",
            )
            metrics[f"precision_ec{level}"] = torchmetrics.Precision(
                task=self.task_type,
                num_labels=len(args.ec_levels[level]),
                average="weighted",
            )
            metrics[f"recall_ec{level}"] = torchmetrics.Recall(
                task=self.task_type,
                num_labels=len(args.ec_levels[level]),
                average="weighted",
            )
        self.metrics = MetricCollection(metrics)

    @property
    def metric_keys(self):
        return [f"probs_ec{i}" for i in range(1, 5, 1)] + [
            f"golds_ec1{i}" for i in range(1, 5, 1)
        ]

    def update(self, predictions_dict, args) -> Dict:
        for level in self.levels:
            probs = predictions_dict[f"probs_ec{level}"]  # B, C (float)
            golds = predictions_dict[f"golds_ec{level}"].int()  # B

            self.metrics[f"f1_ec{level}"].update(probs, golds)
            self.metrics[f"precision_ec{level}"].update(probs, golds)
            self.metrics[f"recall_ec{level}"].update(probs, golds)

    def compute(self) -> Dict:
        stats_dict = {k: v.compute() for k, v in self.metrics.items()}
        return stats_dict

    def reset(self):
        super().reset()
        self.metrics.reset()
