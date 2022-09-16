import torch
import torch.nn as nn
import copy
from nox.utils.registry import register_object
from nox.models.abstract import AbstractModel


@register_object("identity", "model")
class Identity(AbstractModel):
    """
    Model that returns input as is
    """

    def __init__(self, args):
        super(Identity, self).__init__()
        self.args = args

    def forward(self, data):
        return {"hidden": data}


@register_object("linear", "model")
class Linear(AbstractModel):
    def __init__(self, args):
        super(Linear, self).__init__()
        self.args = args
        self.model = nn.Linear(args.linear_input_dim, args.linear_output_dim)

    def forward(self, data):
        output = {"hidden": self.model(data)}
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--linear_input_dim",
            type=int,
            default=512,
            help="Dimension of input",
        )
        parser.add_argument(
            "--linear_output_dim",
            type=int,
            default=512,
            help="Dimension of output",
        )


@register_object("embedding", "model")
class Embedding(AbstractModel):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.args = args
        self.model = nn.Embedding(args.num_embed, args.embed_dim)

    def forward(self, data):
        output = {"hidden": self.model(data)}
        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--num_embed",
            type=int,
            default=512,
            help="Number of rows in the embedding table",
        )
        parser.add_argument(
            "--embed_dim",
            type=int,
            default=512,
            help="Dimension of row",
        )
