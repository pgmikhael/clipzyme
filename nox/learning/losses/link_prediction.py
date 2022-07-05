from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb
from nox.utils.classes import Nox


@register_object("cross_entropy", "loss")
class LinkPredictionLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"]
        
        # first column is positive examples and all others are negative examples
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        
        # calculate loss
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # attention weight over negative examples
        neg_weight = torch.ones_like(pred)
        if args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / args.num_negative

        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        logging_dict["cross_entropy_loss"] = loss.detach()
        predictions["probs"] = F.sigmoid(logit, dim=-1).detach()
        predictions["golds"] = target
        predictions["preds"] = (predictions["probs"] > 0.5).reshape(-1)
        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--adversarial_temperature",
            type=float,
            default=0,
            help="temperature for attention weight over negative examples.",
        )
