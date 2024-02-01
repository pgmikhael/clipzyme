from clipzyme.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb
from clipzyme.utils.classes import Nox


@register_object("cross_entropy", "loss")
class CrossEntropyLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]

        if "y" in model_output:
            predictions["golds"] = model_output["y"]
        elif "y" in batch:
            predictions["golds"] = batch["y"]
        else:
            raise KeyError("predictions_dict ERROR: y not found")

        target = predictions["golds"]
        if args.precomputed_loss:
            loss = model_output["loss"]
        else:
            loss = (
                F.cross_entropy(
                    logit.view(-1, args.num_classes), target.view(-1).long()
                )
                * args.ce_loss_lambda
            )
        logging_dict["cross_entropy_loss"] = loss.detach()
        predictions["probs"] = F.softmax(logit, dim=-1).detach()
        predictions["preds"] = predictions["probs"].argmax(axis=-1)

        if not args.keep_preds_dim:
            predictions["golds"] = predictions["golds"].view(-1)
            predictions["preds"] = predictions["preds"].reshape(-1)
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
        parser.add_argument(
            "--precomputed_loss",
            action="store_true",
            default=False,
            help="whether loss is computed through model automatically, e.g., hugging face transformers",
        )
        parser.add_argument(
            "--keep_preds_dim",
            action="store_true",
            default=False,
            help="do not flatten preds and y",
        )


@register_object("binary_cross_entropy_logits", "loss")
class BinaryCrossEntropyLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        if logit.shape[-1] == 1 and len(logit.shape) == 2:
            logit = logit.squeeze(-1)

        if "has_y" in batch:
            loss = (
                F.binary_cross_entropy_with_logits(
                    logit, batch["y"].float(), reduction="none", weight=batch["has_y"]
                ).sum()
                / batch["has_y"].sum()
                * args.ce_loss_lambda
            )
            predictions["has_golds"] = batch["has_y"]
        else:
            loss = (
                F.binary_cross_entropy_with_logits(logit, batch["y"].float())
                * args.ce_loss_lambda
            )
        logging_dict["binary_cross_entropy_loss"] = loss.detach()
        if args.use_top_prediction:
            probs = torch.sigmoid(logit).detach()
            predictions["probs"] = torch.max(probs, dim=-1)
            prob_ids = torch.argmax(probs, dim=-1)
            predictions["golds"] = batch["y"][prob_ids]
            predictions["preds"] = predictions["probs"] > 0.5
        else:
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
        parser.add_argument(
            "--use_top_prediction",
            action="store_true",
            default=False,
            help="store top prediction in predictions dict",
        )


@register_object("mse", "loss")
class MSELoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        l_dict, p_dict = OrderedDict(), OrderedDict()
        B = batch["y"].shape[0]
        golds = batch["y"].view(B, -1).float()
        logit = model_output["logit"].view(B, -1)
        loss = F.mse_loss(logit, golds) * args.mse_loss_lambda
        l_dict["mse_loss"] = loss.detach()
        p_dict["probs"] = logit.detach()
        p_dict["golds"] = golds
        return loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--mse_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the MSE loss.",
        )


@register_object("rmse", "loss")
class RMSELoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        l_dict, p_dict = OrderedDict(), OrderedDict()
        B = batch["y"].shape[0]
        golds = batch["y"].view(B, -1).float()
        logit = model_output["logit"].view(B, -1)
        loss = (
            torch.sqrt(F.mse_loss(logit, golds) + args.rmse_eps) * args.rmse_loss_lambda
        )
        l_dict["rmse_loss"] = loss.detach()
        p_dict["probs"] = logit.detach()
        p_dict["golds"] = golds
        return loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--rmse_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the RMSE loss.",
        )
        parser.add_argument(
            "--rmse_eps",
            type=float,
            default=1e-8,
            help="epsilon to avoid nan when backpropagating",
        )


@register_object("smoothl1", "loss")
class SmoothL1Loss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        l_dict, p_dict = OrderedDict(), OrderedDict()
        B = batch["y"].shape[0]
        golds = batch["y"].view(B, -1).float()
        logit = model_output["logit"].view(B, -1)
        loss = (
            F.smooth_l1_loss(logit, golds, beta=args.smoothl1_beta)
            * args.smoothl1_loss_lambda
        )
        l_dict["smoothl1_loss"] = loss.detach()
        p_dict["probs"] = logit.detach()
        p_dict["golds"] = golds
        return loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--smoothl1_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the SmoothL1 loss.",
        )
        parser.add_argument(
            "--smoothl1_beta",
            type=float,
            default=1.0,
            help="Specifies the threshold at which to change between L1 and L2 loss. The value must be non-negative.",
        )


@register_object("mae", "loss")
class MAELoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        l_dict, p_dict = OrderedDict(), OrderedDict()
        B = batch["y"].shape[0]
        golds = batch["y"].view(B, -1).float()
        logit = model_output["logit"].view(B, -1)
        loss = F.l1_loss(logit, golds) * args.mae_loss_lambda
        l_dict["mae_loss"] = loss.detach()
        p_dict["probs"] = logit.detach()
        p_dict["golds"] = golds
        return loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--mae_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the MSE loss.",
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
