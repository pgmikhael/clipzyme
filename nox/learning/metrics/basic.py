from typing import Dict
from nox.utils.registry import register_object
from collections import OrderedDict
from nox.utils.classes import Nox
import numpy as np
import pdb
from torchmetrics.functional import (
    auc,
    accuracy,
    auroc,
    precision_recall,
    confusion_matrix,
    f1,
    precision_recall_curve,
    average_precision,
)
import torch
import copy

EPSILON = 1e-6
BINARY_CLASSIF_THRESHOLD = 0.5


@register_object("classification", "metric")
class BaseClassification(Nox):
    def __init__(self, args) -> None:
        super().__init__()

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def __call__(self, logging_dict, args) -> Dict:
        """
        Computes standard classification metrics

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'preds', 'golds']
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            In multiclass setting (>2), accuracy, and micro-f1, micro-recall, micro-precision are equivalent
            Macro: calculates metric per class then averages
        """
        stats_dict = OrderedDict()

        probs = logging_dict["probs"]  # B, C (float)
        preds = logging_dict["preds"]  # B
        golds = logging_dict["golds"]  # B
        stats_dict["accuracy"] = accuracy(golds, preds)
        stats_dict["confusion_matrix"] = confusion_matrix(
            preds, golds, args.num_classes
        )
        if args.num_classes == 2:
            if len(probs.shape) == 1:
                stats_dict["precision"], stats_dict["recall"] = precision_recall(
                    probs, golds
                )
                stats_dict["f1"] = f1(probs, golds)
                pr, rc, _ = precision_recall_curve(probs, golds)
                stats_dict["pr_auc"] = auc(rc, pr)
                try:
                    stats_dict["roc_auc"] = auroc(probs, golds, pos_label=1)
                except:
                    pass
            else:
                stats_dict["precision"], stats_dict["recall"] = precision_recall(
                    probs, golds, multiclass=False, num_classes=2
                )
                stats_dict["f1"] = f1(probs, golds, multiclass=False, num_classes=2)
                pr, rc, _ = precision_recall_curve(probs, golds, num_classes=2)
                stats_dict["pr_auc"] = auc(rc[-1], pr[-1])
                try:
                    stats_dict["roc_auc"] = auroc(probs, golds, num_classes=2)
                except:
                    pass
        else:
            stats_dict["precision"], stats_dict["recall"] = precision_recall(
                probs, golds, num_classes=args.num_classes, average="macro"
            )
            stats_dict["f1"] = f1(
                probs, golds, num_classes=args.num_classes, average="macro"
            )
            stats_dict["micro_f1"] = f1(
                probs, golds, num_classes=args.num_classes, average="micro"
            )
            if len(torch.unique(golds)) == args.num_classes:
                pr, rc, _ = precision_recall_curve(
                    probs, golds, num_classes=args.num_classes
                )
                stats_dict["pr_auc"] = torch.mean(
                    torch.stack([auc(rc[i], pr[i]) for i in range(args.num_classes)])
                )
                stats_dict["roc_auc"] = auroc(
                    probs, golds, num_classes=args.num_classes, average="macro"
                )

            if args.store_classwise_metrics:
                classwise_metrics = {}
                (
                    classwise_metrics["precisions"],
                    classwise_metrics["recalls"],
                ) = precision_recall(
                    probs, golds, num_classes=args.num_classes, average="none"
                )
                classwise_metrics["f1s"] = f1(
                    probs, golds, num_classes=args.num_classes, average="none"
                )
                pr, rc, _ = precision_recall_curve(
                    probs, golds, num_classes=args.num_classes
                )
                classwise_metrics["pr_aucs"] = [
                    auc(rc[i], pr[i]) for i in range(args.num_classes)
                ]
                classwise_metrics["accs"] = accuracy(
                    golds, preds, num_classes=args.num_classes, average="none"
                )
                try:
                    classwise_metrics["rocaucs"] = auroc(
                        probs, golds, num_classes=args.num_classes, average="none"
                    )
                except:
                    pass

                for metricname in [
                    "precisions",
                    "recalls",
                    "f1s",
                    "rocaucs",
                    "pr_aucs",
                    "accs",
                ]:
                    if metricname in classwise_metrics:
                        stats_dict.update(
                            {
                                "class{}_{}".format(i + 1, metricname): v
                                for i, v in enumerate(classwise_metrics[metricname])
                            }
                        )
        return stats_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--store_classwise_metrics",
            action="store_true",
            default=False,
            help="Whether to log metrics per class or just log average across classes",
        )


@register_object("ordinal_classification", "metric")
class Ordinal_Classification(BaseClassification):
    def __call__(self, logging_dict, args) -> Dict:
        """
        Computes classification for metrics when predicting multiple independent classes

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc, prefixed by col index
        """
        stats_dict = OrderedDict()

        probs = logging_dict["probs"]  # B, C (float)
        preds = logging_dict["preds"]  # B
        golds = logging_dict["golds"]  # B
        stats_dict["accuracy"] = accuracy(golds, preds)
        stats_dict["confusion_matrix"] = confusion_matrix(
            preds, golds, args.num_classes + 1
        )

        for classindex in range(golds.shape[-1]):
            (
                stats_dict["class{}_precision".format(classindex)],
                stats_dict["class{}_recall".format(classindex)],
            ) = precision_recall(probs, golds)
            stats_dict["class{}_f1".format(classindex)] = f1(probs, golds)
            pr, rc, _ = precision_recall_curve(probs, golds)
            stats_dict["class{}_pr_auc".format(classindex)] = auc(rc, pr)
            try:
                stats_dict["class{}_roc_auc".format(classindex)] = auroc(
                    probs, golds, pos_label=1
                )
            except:
                pass

        return stats_dict


@register_object("survival_classification", "metric")
class Survival_Classification(BaseClassification):
    def __call__(self, logging_dict, args):
        stats_dict = OrderedDict()

        golds = logging_dict["golds"]
        probs = logging_dict["probs"]
        preds = probs[:, -1].view(-1) > 0.5
        probs = probs.reshape((-1, probs.shape[-1]))[:, -1]

        stats_dict["accuracy"] = accuracy(golds, preds)

        if (args.num_classes == 2) and not (
            np.unique(golds)[-1] > 1 or np.unique(preds)[-1] > 1
        ):
            stats_dict["precision"], stats_dict["recall"] = precision_recall(
                probs, golds
            )
            stats_dict["f1"] = f1(probs, golds)
            num_pos = golds.sum()
            if num_pos > 0 and num_pos < len(golds):
                stats_dict["auc"] = auroc(probs, golds, pos_label=1)
                stats_dict["ap_score"] = average_precision(probs, golds)
                precision, recall, _ = precision_recall_curve(probs, golds)
                stats_dict["prauc"] = auc(recall, precision)
        return stats_dict


@register_object("discrim_classification", "metric")
class Discriminator_Classification(BaseClassification):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def metric_keys(self):
        return ["discrim_probs", "discrim_golds"]

    def __call__(self, logging_dict, args):
        stats_dict = OrderedDict()

        golds = logging_dict["discrim_golds"]
        probs = logging_dict["discrim_probs"]
        preds = logging_dict["discrim_probs"].argmax(axis=-1).reshape(-1)

        nargs = copy.deepcopy(args)
        nargs.num_classes = probs.shape[-1]
        stats_dict = super().__call__(
            {"golds": golds, "probs": probs, "preds": preds}, nargs
        )
        stats_dict = {"discrim_{}".format(k): v for k, v in stats_dict.items()}

        return stats_dict


@register_object("multi_discrim_classification", "metric")
class MultiDiscriminator_Classification(BaseClassification):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def metric_keys(self):
        return ["device_probs", "device_golds", "thickness_probs", "thickness_golds"]

    def __call__(self, logging_dict, args):
        stats_dict = OrderedDict()

        for key in ["device", "thickness"]:
            golds = logging_dict["{}_golds".format(key)]
            probs = logging_dict["{}_probs".format(key)]
            preds = logging_dict["{}_probs".format(key)].argmax(axis=-1).reshape(-1)

            nargs = copy.deepcopy(args)
            nargs.num_classes = probs.shape[-1]
            stats_dict = super().__call__(
                {"golds": golds, "probs": probs, "preds": preds}, nargs
            )
            stats_dict = {"{}_{}".format(key, k): v for k, v in stats_dict.items()}

        return stats_dict
