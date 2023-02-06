from typing import Dict
from nox.utils.registry import register_object
from collections import OrderedDict, defaultdict
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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    cosine_similarity,
    pearson_corrcoef,
    spearman_corrcoef
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

    def __call__(self, predictions_dict, args) -> Dict:
        """
        Computes standard classification metrics

        Args:
            predictions_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'preds', 'golds']
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            In multiclass setting (>2), accuracy, and micro-f1, micro-recall, micro-precision are equivalent
            Macro: calculates metric per class then averages
        """
        stats_dict = OrderedDict()

        probs = predictions_dict["probs"]  # B, C (float)
        preds = predictions_dict["preds"]  # B
        golds = predictions_dict["golds"].int()  # B
        stats_dict["accuracy"] = accuracy(golds, preds)
        # stats_dict["confusion_matrix"] = confusion_matrix(
        #     preds, golds, args.num_classes
        # )
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
    def __call__(self, predictions_dict, args) -> Dict:
        """
        Computes classification for metrics when predicting multiple independent classes

        Args:
            predictions_dict: dictionary obtained from computing loss and model outputs
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc, prefixed by col index
        """
        stats_dict = OrderedDict()

        probs = predictions_dict["probs"]  # B, C (float)
        preds = predictions_dict["preds"]  # B
        golds = predictions_dict["golds"]  # B
        stats_dict["accuracy"] = accuracy(golds, preds)
        #stats_dict["confusion_matrix"] = confusion_matrix(
        #    preds, golds, args.num_classes + 1
        #)

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
    def __call__(self, predictions_dict, args):
        stats_dict = OrderedDict()

        golds = predictions_dict["golds"]
        probs = predictions_dict["probs"]
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


@register_object("multitask_classification", "metric")
class MultiTask_Classification(BaseClassification):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def __call__(self, predictions_dict, args):
        stats_dict = OrderedDict()

        golds = predictions_dict["golds"].int()
        probs = predictions_dict["probs"]
        preds = predictions_dict["preds"].int()

        if "has_golds" in predictions_dict:
            metrics = defaultdict(list)
            for col in range(predictions_dict["golds"].shape[1]):
                row = predictions_dict["has_golds"][:,col]
                pr, rc = precision_recall(probs[row, col], golds[row, col])
                metrics["precision"].append(pr)
                metrics["recall"].append(rc)
                metrics["f1"].append( f1(probs[row, col], golds[row, col] ))
                metrics["roc_auc"].append( auroc(probs[row, col], golds[row, col] ) )
            stats_dict = {k:torch.stack(v).mean() for k,v in metrics.items()}
        else:
            stats_dict["precision"], stats_dict["recall"] = precision_recall(probs, golds, multiclass=False)
            stats_dict["f1"] = f1(probs, golds, multiclass=False, average="macro")
            stats_dict["roc_auc"] = auroc(probs, golds, multiclass=False, num_classes=2)
            
        return stats_dict


@register_object("regression", 'metric')
class BaseRegression(Nox):
    def __init__(self, args) -> None:
        super().__init__()

    @property
    def metric_keys(self):
        return ['probs', 'golds']

    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes standard regresssion loss
        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'golds']
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for mse, mae, r2
        '''
        stats_dict = OrderedDict()

        probs = logging_dict['probs']
        golds = logging_dict['golds']

        stats_dict['mse'] = mean_squared_error(probs, golds)
        stats_dict['mae'] = mean_absolute_error(probs, golds)
        stats_dict['pearson'] = pearson_corrcoef(probs.view(-1), golds.view(-1))
        stats_dict['spearman'] = spearman_corrcoef(probs.view(-1), golds.view(-1))

        r2 = r2_score(probs, golds, multioutput='raw_values')
        if probs.shape[-1] > 1:
            stats_dict['r2'] = torch.stack([r for r in r2 if not torch.isinf(r)] ).mean()
            stats_dict['cosine_similarity'] = cosine_similarity(probs, golds, reduction = 'mean')
        else:
            stats_dict['r2'] = r2

        return stats_dict


@register_object("seq2seq_classification", "metric")
class Seq2SeqClassification(BaseClassification):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def __call__(self, predictions_dict, args) -> Dict:
        stats_dict = defaultdict(list)

        probs = predictions_dict["probs"]  # B, N, C (float)
        preds = predictions_dict["preds"]  # B, N
        golds = predictions_dict["golds"].int()  # B, N
        
        top1_correct = 0
        for sample_probs, sample_preds, sample_golds  in zip(probs, preds, golds):
            sample_probs = sample_probs[sample_golds!=-100]
            sample_preds = sample_preds[sample_golds!=-100]
            sample_golds = sample_golds[sample_golds!=-100]
            sample_stats = super().__call__(
                {
                    "probs": sample_probs, 
                    "preds": sample_preds,
                    "golds": sample_golds
                }, args
            )
            top1_correct += torch.all(sample_preds == sample_golds).int()
            for k,v in sample_stats.items():
                if len(v.shape) < 2:
                    stats_dict[k].append(v)
        
        for k,v in stats_dict.items():
            stats_dict[k] = torch.stack(v).mean()
        
        stats_dict["top_1"] = top1_correct / len(golds)

        return stats_dict
    
