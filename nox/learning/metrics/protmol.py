from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import torchmetrics
from torchmetrics import Metric
from nox.learning.metrics.basic import TopK
import torch

@register_object("protmol_clip_classification", "metric")
class ProtMolClassificatino(Metric, Nox):
    """Same as BaseClassification but samplewise multidim_average"""

    def __init__(self, args) -> None:
        super().__init__()
        self.ignore_index = args.ignore_index
        
        self.reaction_center_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=2,
            num_labels=2,
        )
        self.reaction_center_recall = torchmetrics.Recall(
            task="multiclass",
            num_classes=2,
            num_labels=2,
        )

        self.matching_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=2,
            num_labels=2,
        )

        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_labels=args.vocab_size,
            num_classes=args.vocab_size,
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )

    @property
    def metric_keys(self):
        return ["reaction_center_preds", "reaction_center_golds", "matching_preds", "matching_golds", "mlm_preds", "mlm_golds"]

    def update(self, predictions_dict, args) -> Dict:
        if "reaction_center_preds" in predictions_dict:
            self.reaction_center_accuracy.update(
                predictions_dict["reaction_center_preds"], 
                predictions_dict["reaction_center_golds"].long())
            self.reaction_center_recall(
                predictions_dict["reaction_center_preds"], 
                predictions_dict["reaction_center_golds"].long())

        if "matching_preds" in predictions_dict:
            self.matching_accuracy.update(
                predictions_dict["matching_preds"], 
                predictions_dict["matching_golds"].long())
            

        if "mlm_preds" in predictions_dict:
            self.mlm_accuracy.update(
                predictions_dict["mlm_preds"], 
                predictions_dict["mlm_golds"])
        

    def compute(self) -> Dict:
        stats_dict = {}
        if len(self.mlm_accuracy.tn + self.mlm_accuracy.fn + self.mlm_accuracy.fp + self.mlm_accuracy.tn):
            stats_dict["mlm_accuracy"] = self.mlm_accuracy.compute()
        
        if len(self.reaction_center_accuracy.tn + self.reaction_center_accuracy.fn + self.reaction_center_accuracy.fp + self.reaction_center_accuracy.tn):
            stats_dict["reaction_center_accuracy"] = self.reaction_center_accuracy.compute()

        if len(self.matching_accuracy.tn + self.matching_accuracy.fn + self.matching_accuracy.fp + self.matching_accuracy.tn):
            stats_dict["matching_accuracy"] = self.matching_accuracy.compute()

        if len(self.reaction_center_recall.tn + self.reaction_center_recall.fn + self.reaction_center_recall.fp + self.reaction_center_recall.tn):
            stats_dict["reaction_center_recall"] = self.reaction_center_recall.compute()

        return stats_dict

    def reset(self):
        super().reset()
        self.reaction_center_accuracy.reset()
        self.matching_accuracy.reset()
        self.mlm_accuracy.reset()
        

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-100,
            help="type of task",
        )