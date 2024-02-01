from typing import Dict
from clipzyme.utils.registry import register_object
from clipzyme.utils.classes import Nox
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
        if args.task_type != "multilabel":
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
        elif self.task_type == "multilabel":
            self.accuracy_metric = torchmetrics.Accuracy(
                task=args.task_type,
                num_labels=args.num_classes,
            )
            self.auroc_metric = torchmetrics.AUROC(
                task=args.task_type, num_labels=args.num_classes
            )
            self.f1_metric = torchmetrics.F1Score(
                task=args.task_type,
                num_labels=args.num_classes,
            )
            self.macro_f1_metric = torchmetrics.F1Score(
                task=args.task_type,
                num_labels=args.num_classes,
                average="macro",
            )
            self.ap_metric = torchmetrics.AveragePrecision(
                task=args.task_type, num_labels=args.num_classes
            )
            self.auprc_metric = torchmetrics.PrecisionRecallCurve(
                task=args.task_type, num_labels=args.num_classes
            )
            self.precision_metric = torchmetrics.Precision(
                task=args.task_type,
                num_labels=args.num_classes,
            )
            self.recall_metric = torchmetrics.Recall(
                task=args.task_type,
                num_labels=args.num_classes,
            )
        else:
            self.accuracy_metric = torchmetrics.Accuracy(
                task=args.task_type,
            )
            self.auroc_metric = torchmetrics.AUROC(
                task=args.task_type,
            )
            self.f1_metric = torchmetrics.F1Score(
                task=args.task_type,
            )
            self.macro_f1_metric = torchmetrics.F1Score(
                task=args.task_type,
                average="macro",
            )
            self.ap_metric = torchmetrics.AveragePrecision(
                task=args.task_type,
            )
            self.auprc_metric = torchmetrics.PrecisionRecallCurve(
                task=args.task_type,
            )
            self.precision_metric = torchmetrics.Precision(
                task=args.task_type,
            )
            self.recall_metric = torchmetrics.Recall(
                task=args.task_type,
            )

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args) -> Dict:
        probs = predictions_dict["probs"]  # B, C (float)
        preds = predictions_dict["preds"]  # B
        golds = predictions_dict["golds"].int()  # B

        if self.task_type == "binary":
            if len(preds.shape) >= 2:
                preds = preds[:, 1]
            if len(probs.shape) >= 2:
                probs = probs[:, 1]

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
        super().reset()
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


@register_object("dummy_classification", "metric")
class DummyBaseClassification(Metric, Nox):
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
        return ["y", "probs", "preds", "golds"]

    def update(self, predictions_dict, args) -> Dict:
        probs = predictions_dict["probs"]  # B, C (float)
        probs = torch.diagonal(probs)  # I am only looking at samples I added
        preds = predictions_dict["preds"]  # B
        preds = torch.diagonal(preds)
        golds = predictions_dict["y"].int()  # B

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
        super().__init__()
        self.ignore_index = args.ignore_index
        self.task_type = args.task_type
        self.accuracy_metric = torchmetrics.Accuracy(
            task=args.task_type,
            num_classes=args.num_classes,
            num_labels=args.num_classes,
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )
        self.f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_classes=args.num_classes,
            num_labels=args.num_classes,
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )
        self.macro_f1_metric = torchmetrics.F1Score(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            average="macro",
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )
        self.precision_metric = torchmetrics.Precision(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )
        self.recall_metric = torchmetrics.Recall(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            multidim_average="samplewise",
            ignore_index=args.ignore_index,
        )
        self.top_1_metric = TopK(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        ).to(self.accuracy_metric.device)
        # don't have multidim_average arg
        self.auroc_metric = torchmetrics.AUROC(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        self.ap_metric = torchmetrics.AveragePrecision(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        self.auprc_metric = torchmetrics.PrecisionRecallCurve(
            task=args.task_type,
            num_labels=args.num_classes,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds", "preds_mask"]

    def update(self, predictions_dict, args) -> Dict:
        probs = predictions_dict["probs"]  # B, N, C (float)
        preds = predictions_dict["preds"]  # B, N
        golds = predictions_dict["golds"].long()  # B, N
        preds_mask = predictions_dict["preds_mask"]

        if self.ignore_index is not None:
            preds[preds_mask == self.ignore_index] = self.ignore_index

        self.accuracy_metric.update(preds, golds)
        self.f1_metric.update(preds, golds)
        self.macro_f1_metric.update(preds, golds)
        self.precision_metric.update(preds, golds)
        self.recall_metric.update(preds, golds)
        self.top_1_metric.update(preds, golds, preds_mask)

        # for sample_prob, sample_gold in zip(probs, golds):
        #     self.auroc_metric.update(sample_prob, sample_gold)
        #     self.ap_metric.update(sample_prob, sample_gold)
        #     self.auprc_metric.update(sample_prob, sample_gold)

    def compute(self) -> Dict:
        # pr, rc, _ = self.auprc_metric.compute()
        # if self.task_type != "binary":  # list per class or per label if not binary task
        #     pr_auc = [compute_auc(rc_i, pr_i) for rc_i, pr_i in zip(rc, pr)]
        #     pr_auc = torch.mean(torch.stack(pr_auc))
        # else:
        #     pr_auc = compute_auc(rc, pr)
        stats_dict = {
            "accuracy": self.accuracy_metric.compute().mean(),
            # "roc_auc": self.auroc_metric.compute(),
            # pr_auc": pr_auc,
            "f1": self.f1_metric.compute().mean(),
            "macro_f1": self.macro_f1_metric.compute().mean(),
            "precision": self.precision_metric.compute().mean(),
            "recall": self.recall_metric.compute().mean(),
            "top_1": self.top_1_metric.compute(),
        }
        return stats_dict

    def reset(self):
        super().reset()
        self.accuracy_metric.reset()
        # self.auroc_metric.reset()
        # self.auprc_metric.reset()
        self.f1_metric.reset()
        self.macro_f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.ap_metric.reset()
        self.top_1_metric.reset()

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
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-100,
            help="type of task",
        )


@register_object("regression", "metric")
class BaseRegression(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_classes = args.num_classes
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=args.num_classes)
        self.spearman = torchmetrics.SpearmanCorrCoef(num_outputs=args.num_classes)
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

        if probs.shape[-1] == 1:
            probs = probs.view(-1)
            golds = golds.view(-1)

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
        super().reset()
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


class TopK(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        self.ignore_index = kwargs["ignore_index"]
        self.task = kwargs["task"]
        self.num_classes = kwargs["num_classes"]

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, preds_mask: torch.Tensor
    ):
        assert preds.shape == target.shape

        if self.ignore_index is not None:
            preds[preds_mask == self.ignore_index] = self.ignore_index

        correct = (preds == target).sum(1) == preds.shape[1]
        correct = correct.sum()
        total = torch.tensor(preds.shape[0])

        self.correct += correct.to(self.device)
        self.total += total.to(self.device)

    def compute(self):
        return self.correct.float() / self.total
