from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb
from nox.utils.classes import Nox


@register_object("protmol_clip_cross_entropy", "loss")
class ProtMolClipCrossEntropyLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        loss = 0
        if "reaction_center_logit" in model_output:
            logit = model_output["reaction_center_logit"]
            target =  model_output["reaction_center_labels"]
            
            loss_rc = F.cross_entropy(logit, target.long()) * args.reaction_node_ce_loss_lambda
            logging_dict["reaction_center_cross_entropy_loss"] = loss_rc.detach()
            predictions["reaction_center_probs"] = F.softmax(logit, dim=-1).detach()
            predictions["reaction_center_preds"] = predictions["reaction_center_probs"].argmax(axis=-1)
            predictions["reaction_center_golds"] = model_output["reaction_center_labels"]
            loss = loss + loss_rc
        
        if "matching_logits" in model_output:
            logit = model_output["matching_logits"]
            target =  model_output["matching_labels"]
            
            loss_m = F.cross_entropy(logit, target.long()) * args.matching_loss_lambda
            logging_dict["matching_cross_entropy_loss"] = loss_m.detach()
            predictions["matching_probs"] = F.softmax(logit, dim=-1).detach()
            predictions["matching_preds"] = predictions["matching_probs"].argmax(axis=-1) 
            predictions["matching_golds"] = model_output["matching_labels"]
            loss = loss + loss_m

        if "mlm_logits" in model_output:
            logit = model_output["mlm_logits"].view(-1, args.vocab_size)
            target =  model_output["mlm_labels"].view(-1)
            
            loss_mlm = F.cross_entropy(logit, target.long()) * args.mlm_loss_lambda
            logging_dict["mlm_cross_entropy_loss"] = loss_mlm.detach()
            predictions["mlm_probs"] = F.softmax(model_output["mlm_logits"], dim=-1).detach()
            predictions["mlm_preds"] = predictions["mlm_probs"].argmax(axis=-1) 
            predictions["mlm_golds"] = model_output["mlm_labels"]
            loss = loss + loss_mlm

        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--reaction_node_ce_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        parser.add_argument(
            "--matching_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        parser.add_argument(
            "--mlm_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        
