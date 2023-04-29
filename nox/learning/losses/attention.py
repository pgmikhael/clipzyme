from nox.utils.registry import register_object
import torch 
import torch.nn.functional as F
from collections import OrderedDict
from nox.utils.classes import Nox


@register_object("cross_attention_loss", "loss")
class CrossAttentionLoss(Nox):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-9

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        max_seq_len = max(len(s) for s in batch['sequence'])
        attentions = model_output["cross_attentions"] # tuple: batch_size, num_heads, decoder_seq_len, max_protein_seq_len 
        sequence_annotation = batch["sequence_annotation"] / batch["sequence_annotation"].sum(-1).unsqueeze(-1) # normalize to sum to one
        sequence_annotation = sequence_annotation.unsqueeze(1).unsqueeze(1) # batch_size, 1, 1, seq_len
        sequence_mask = batch["sequence_annotation"].unsqueeze(1).unsqueeze(1) > 0.5
        loss = 0
        for attention in attentions:
            protein_attention = attention[...,1:(max_seq_len+1)] + self.eps
            protein_attention = protein_attention / protein_attention.sum(-1).unsqueeze(-1) # normalize to sum to one
            kl_div_loss = F.kl_div(protein_attention.log(), sequence_annotation, reduction='none', log_target=False) * sequence_mask # input in log space
            loss = loss + kl_div_loss.sum() / sum(kl_div_loss.shape[:3]) # mean over first 3 dimensions (batch, heads, decoding tokens)
        
        loss = loss * args.attention_loss_lambda
        logging_dict["cross_entropy_loss"] = loss.detach()

        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--attention_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
