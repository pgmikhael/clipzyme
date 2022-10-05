from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb
from nox.utils.classes import Nox


@register_object("precomputed_loss", "loss")
class PreComputedLoss(Nox):
    """Method for when loss is computed through model automatically, e.g., hugging face transformers"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        loss = model_output["loss"]
        logging_dict["loss"] = loss.detach()
        for k, v in model_output.items():
            if k != "loss":
                predictions[k] = v.detach()

        return loss, logging_dict, predictions


@register_object("cross_entropy", "loss")
class CrossEntropyLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        loss = F.cross_entropy(logit, batch["y"].view(-1).long()) * args.ce_loss_lambda
        logging_dict["cross_entropy_loss"] = loss.detach()
        predictions["probs"] = F.softmax(logit, dim=-1).detach()
        predictions["golds"] = batch["y"].view(-1)
        predictions["preds"] = predictions["probs"].argmax(axis=-1).reshape(-1)
        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--ce_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )


@register_object("binary_cross_entropy_logits", "loss")
class BinaryCrossEntropyLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        if "has_y" in batch:
            loss = (
                F.binary_cross_entropy_with_logits(
                    logit, batch["y"], reduction="none", weight=batch["has_y"]
                ).sum()
                / batch["has_y"].sum()
                * args.ce_loss_lambda
            )
            predictions["has_golds"] = batch["has_y"]
        else:
            loss = (
                F.binary_cross_entropy_with_logits(logit, batch["y"])
                * args.ce_loss_lambda
            )
        logging_dict["binary_cross_entropy_loss"] = loss.detach()
        predictions["probs"] = torch.sigmoid(logit).detach()
        predictions["golds"] = batch["y"]
        predictions["preds"] = predictions["probs"] > 0.5
        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--ce_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )


@register_object("survival", "loss")
class SurvivalLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        y_seq, y_mask = batch["y_seq"], batch["y_mask"]
        loss = F.binary_cross_entropy_with_logits(
            logit, y_seq.float(), weight=y_mask.float(), reduction="sum"
        ) / torch.sum(y_mask.float())
        logging_dict["survival_loss"] = loss.detach()
        predictions["probs"] = torch.sigmoid(logit).detach()
        predictions["golds"] = batch["y"]
        predictions["censors"] = batch["time_at_event"]
        return loss, logging_dict, predictions


@register_object("ordinal_cross_entropy", "loss")
class RankConsistentLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        """
        Computes cross-entropy loss

        If batch contains they key 'has_y', the cross entropy loss will be computed for samples where batch['has_y'] = 1
        Expects model_output to contain 'logit'

        Returns:
            loss: cross entropy loss
            l_dict (dict): dictionary containing cross_entropy_loss detached from computation graph
            p_dict (dict): dictionary of model predictions and ground truth labels (preds, probs, golds)
        """
        loss = 0
        l_dict, p_dict = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        yseq = batch["yseq"]
        ymask = batch["ymask"]

        loss = F.binary_cross_entropy_with_logits(
            logit, yseq.float(), weight=ymask.float(), reduction="sum"
        ) / torch.sum(ymask.float())

        probs = F.logsigmoid(logit)  # log_sum to add probs
        probs = probs.unsqueeze(1).repeat(1, len(args.rank_thresholds), 1)
        probs = torch.tril(probs).sum(2)
        probs = torch.exp(probs)

        p_dict["logits"] = logit.detach()
        p_dict["probs"] = probs.detach()
        preds = probs > 0.5  # class = last prob > 0.5
        preds = preds.sum(-1)
        p_dict["preds"] = preds
        p_dict["golds"] = batch["y"]

        return loss, l_dict, p_dict
