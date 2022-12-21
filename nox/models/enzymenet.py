import torch
import torch.nn as nn
import copy
from nox.utils.registry import register_object, get_object
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel


@register_object("enzyme_substrate_model", "model")
class EnzymeSubstrateScore(AbstractModel):
    def __init__(self, args):
        super(EnzymeSubstrateScore, self).__init__()

        self.args = args
        self.protein_encoder = get_object(args.protein_encoder_name, "model")(args)
        self.substrate_encoder = get_object(args.substrate_encoder_name, "model")(args)
        self.mlp = get_object(args.protein_substrate_aggregator, "model")(args)
        if args.activation_name is not None:
            self.activation = getattr(torch.nn.functional, args.activation_name)

    def forward(self, batch=None):
        output = {}
        sequence_dict = self.protein_encoder(batch["sequence"])
        substrate_dict = self.substrate_encoder(batch["mol"])
        hidden = torch.cat((sequence_dict["hidden"], substrate_dict["hidden"]), dim=1)
        mlp_dict = self.mlp({"x": hidden})
        output = {
            "sequence_hidden": sequence_dict["hidden"],
            "substrate_hidden": substrate_dict["hidden"],
        }
        output.update(mlp_dict)
        if self.args.activation_name is not None:
            output["logit"] = self.activation(output["logit"])
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--substrate_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--protein_substrate_aggregator",
            type=str,
            action=set_nox_type("model"),
            default="chemprop",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--activation_name",
            type=str,
            default=None,
            help="type of activation to be applied on logits",
        )


@register_object("enzyme_active_site_predictor", "model")
class EnzymeActiveSiteModel(AbstractModel):
    def __init__(self, args):
        super(EnzymeActiveSiteModel, self).__init__()

        self.args = args
        self.protein_encoder = get_object(args.protein_encoder_name, "model")(args)
        #self.reaction_encoder = get_object(args.reaction_encoder_name, "model")(args)
        self.mlp = get_object(args.classifier_name, "model")(args)

    def forward(self, batch=None):
        batch_size = len(batch["sequence"])
        seq_len = [len(p) for p in batch["sequence"]]
        sequence_dict = self.protein_encoder(batch["sequence"])
        hidden = sequence_dict["token_hiddens"][:, 1:-1]  # B, seq_len, hidden_dim
        output = self.mlp({"x": hidden})  # B, seq_len, num_classes
 
        labels = batch["residue_mask"]
        residue_mask = sequence_dict['mask_hiddens'][:, 1:-1]
        # cross entropy will ignore the padded residues (ignore_index=-100)
        labels[~residue_mask.squeeze(-1).bool()] = -100
        batch["y"] = labels 

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--protein_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="fair_esm",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--reaction_encoder_name",
            type=str,
            action=set_nox_type("model"),
            default="reaction_mlm",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--classifier_name",
            type=str,
            action=set_nox_type("model"),
            default="mlp_classifier",
            help="Name of encoder to use",
        )
