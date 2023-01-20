from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import torchmetrics
from torchmetrics import Metric
from torchmetrics.utilities.compute import auc as compute_auc
import torch


@register_object("classification", "metric")
class BaseClassification(Metric, Nox):
    def __init__(self, args) -> None:
        """
        Computes standard classification metrics

        Args:
            predictions_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'preds', 'golds']
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            Binary: two labels
            Multiclass: more than two labels
            Multilabel: potentially more than one label per sample (independent classes)
        """
        super().__init__()
        self.task_type = args.task_type
        self.accuracy_metric = torchmetrics.Accuracy(
            task=args.task_type,
            num_classes=args.num_classes,
        )
        self.auroc_metric = torchmetrics.AUROC(
            task=args.task_type, num_classes=args.num_classes
        )
        self.f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_classes=args.num_classes,
        )
        self.macro_f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_classes=args.num_classes,
            average="macro",
        )
        self.ap_metric = torchmetrics.AveragePrecision(
            task=args.task_type, num_classes=args.num_classes
        )
        self.auprc_metric = torchmetrics.PrecisionRecallCurve(
            task=args.task_type, num_classes=args.num_classes
        )
        self.precision_metric = torchmetrics.Precision(
            task=args.task_type,
            num_classes=args.num_classes,
        )
        self.recall_metric = torchmetrics.Recall(
            task=args.task_type,
            num_classes=args.num_classes,
        )

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args) -> Dict:

        probs = predictions_dict["probs"]  # B, C (float)
        preds = predictions_dict["preds"]  # B
        golds = predictions_dict["golds"].int()  # B

        self.accuracy_metric.update(preds, golds)
        self.auroc_metric.update(probs, golds)
        self.auprc_metric.update(probs, golds)
        self.f1_metric.update(probs, golds)
        self.macro_f1_metric.update(probs, golds)
        self.precision_metric.update(probs, golds)
        self.recall_metric.update(probs, golds)
        self.ap_metric.update(probs, golds)

    def compute(self) -> Dict:
        pr, rc, _ = self.auprc_metric.compute()
        if self.task_type != "binary":  # list per class or per label if not binary task
            pr_auc = [compute_auc(rc_i, pr_i) for rc_i, pr_i in zip(rc, pr)]
            pr_auc = torch.mean(torch.stack(pr_auc))
        else:
            pr_auc = compute_auc(rc, pr)
        stats_dict = {
            "accuracy": self.accuracy_metric.compute(),
            "roc_auc": self.auroc_metric.compute(),
            "pr_auc": pr_auc,
            "f1": self.f1_metric.compute(),
            "macro_f1": self.macro_f1_metric.compute(),
            "precision": self.precision_metric.compute(),
            "recall": self.recall_metric.compute(),
            "average_precision": self.ap_metric.compute(),
        }
        return stats_dict

    def reset(self):
        self.accuracy_metric.reset()
        self.auroc_metric.reset()
        self.auprc_metric.reset()
        self.f1_metric.reset()
        self.macro_f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.ap_metric.reset()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--task_type",
            type=str,
            default=None,
            required=True,
            choices=["binary", "multiclass", "multilabel"],
            help="type of task",
        )


@register_object("seq2seq_classification", "metric")
class Seq2SeqClassification(Metric, Nox):
    """Same as BaseClassification but samplewise multidim_average"""

    def __init__(self, args) -> None:
        super().__init__(args)

        self.task_type = args.task_type
        self.accuracy_metric = torchmetrics.Accuracy(
            task=args.task_type,
            num_classes=args.num_classes,
            multidim_average="samplewise",
        )
        self.f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_classes=args.num_classes,
            multidim_average="samplewise",
        )
        self.macro_f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_classes=args.num_classes,
            average="macro",
            multidim_average="samplewise",
        )
        self.precision_metric = torchmetrics.Precision(
            task=args.task_type,
            num_classes=args.num_classes,
            multidim_average="samplewise",
        )
        self.recall_metric = torchmetrics.Recall(
            task=args.task_type,
            num_classes=args.num_classes,
            multidim_average="samplewise",
        )
        # don't have multidim_average arg
        self.auroc_metric = torchmetrics.AUROC(
            task=args.task_type, num_classes=args.num_classes
        )
        self.ap_metric = torchmetrics.AveragePrecision(
            task=args.task_type, num_classes=args.num_classes
        )
        self.auprc_metric = torchmetrics.PrecisionRecallCurve(
            task=args.task_type, num_classes=args.num_classes
        )

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args) -> Dict:

        probs = predictions_dict["probs"]  # B, C (float)
        preds = predictions_dict["preds"]  # B
        golds = predictions_dict["golds"].int()  # B

        self.accuracy_metric.update(preds, golds)
        self.f1_metric.update(probs, golds)
        self.macro_f1_metric.update(probs, golds)
        self.precision_metric.update(probs, golds)
        self.recall_metric.update(probs, golds)

        for sample_prob, sample_gold in zip(probs, golds):
            self.auroc_metric.update(sample_prob, sample_gold)
            self.ap_metric.update(sample_prob, sample_gold)
            self.auprc_metric.update(sample_prob, sample_gold)

    def compute(self) -> Dict:
        pr, rc, _ = self.auprc_metric.compute()
        if self.task_type != "binary":  # list per class or per label if not binary task
            pr_auc = [compute_auc(rc_i, pr_i) for rc_i, pr_i in zip(rc, pr)]
            pr_auc = torch.mean(torch.stack(pr_auc))
        else:
            pr_auc = compute_auc(rc, pr)
        stats_dict = {
            "accuracy": self.accuracy_metric.compute(),
            "roc_auc": self.auroc_metric.compute(),
            "pr_auc": pr_auc,
            "f1": self.f1_metric.compute(),
            "macro_f1": self.macro_f1_metric.compute(),
            "precision": self.precision_metric.compute(),
            "recall": self.recall_metric.compute(),
        }
        return stats_dict

    def reset(self):
        self.accuracy_metric.reset()
        self.auroc_metric.reset()
        self.auprc_metric.reset()
        self.f1_metric.reset()
        self.macro_f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.ap_metric.reset()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--task_type",
            type=str,
            default=None,
            required=True,
            choices=["binary", "multiclass", "multilabel"],
            help="type of task",
        )


@register_object("regression", "metric")
class BaseRegression(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_classes = args.num_classes
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.pearson = torchmetrics.PearsonCorrcoef(num_outputs=args.num_classes)
        self.spearman = torchmetrics.SpearmanCorrcoef(num_outputs=args.num_classes)
        self.r2 = torchmetrics.R2Score(
            num_outputs=args.num_classes, multioutput=args.r2_multioutput
        )
        self.cosine_similarity = torchmetrics.CosineSimilarity(reduction="mean")

    @property
    def metric_keys(self):
        return ["probs", "golds"]

    def update(self, predictions_dict, args) -> Dict:

        probs = predictions_dict["probs"]  # B, C (float)
        golds = predictions_dict["golds"]  # B

        self.mae.update(probs, golds)
        self.mse.update(probs, golds)
        self.pearson.update(probs, golds)
        self.spearman.update(probs, golds)
        self.r2.update(probs, golds)
        if self.num_classes > 1:
            self.cosine_similarity.update(probs, golds)

    def compute(self) -> Dict:
        stats_dict = {
            "mae": self.mae.compute(),
            "mse": self.mse.compute(),
            "pearson": self.pearson.compute(),
            "spearman": self.spearman.compute(),
            "r2": self.r2.compute(),
        }
        if self.num_classes > 1:
            stats_dict["cosine_similarity"] = self.cosine_similarity.compute()
        return stats_dict

    def reset(self):
        self.mae.reset()
        self.mse.reset()
        self.pearson.reset()
        self.spearman.reset()
        self.r2.reset()
        self.cosine_similarity.reset()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--r2_multioutput",
            type=str,
            default="uniform_average",
            choices=["raw_values", "uniform_average", "variance_weighted"],
            help="aggregation in the case of multiple output scores",
        )
