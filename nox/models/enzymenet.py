import torch
import torch.nn as nn
import copy
from nox.utils.registry import register_object, get_object
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel


@register_object("enzyme_substrate_model", "model")
class EznymeSubstrateScore(AbstractModel):
    def __init__(self, args):
        super(EznymeSubstrateScore, self).__init__()

        self.args = args
        self.protein_encoder = get_object(args.protein_encoder_name, "model")(args)
        self.substrate_encoder = get_object(args.subtrate_encoder_name, "model")(args)
        self.mlp = get_object(args.protein_substrate_aggregator, "model")(args)

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
            "--subtrate_encoder_name",
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
