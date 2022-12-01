import torch
import torch.nn as nn
import copy
from nox.utils.registry import register_object, get_object
from nox.utils.classes import set_nox_type
from nox.models.abstract import AbstractModel
from typing import List, Tuple, Optional


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.
    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]
    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        super(SimCLRProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, None, nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


@register_object("simclr", "model")
class SimCLR(AbstractModel):
    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args
        self.reaction_backbone = get_object(args.reaction_backbone_model, "model")(args)
        # freeze backbone
        if args.freeze_reaction_mapper:
            self.reaction_backbone.eval()

        self.enzyme_backbone = get_object(args.enzyme_backbone_model, "model")(args)

        self.project_reaction_hidden = args.project_reaction_hidden
        if args.project_reaction_hidden:
            self.reaction_projection_head = SimCLRProjectionHead(
                args.num_ftrs, args.num_ftrs, args.out_dim
            )
        self.enzyme_projection_head = SimCLRProjectionHead(
            args.num_ftrs, args.num_ftrs, args.out_dim
        )

    def forward(self, batch):
        reaction_hidden = self.reaction_backbone(batch["reaction"])
        sequence_hidden = self.enzyme_backbone(batch["sequence"])

        if self.project_reaction_hidden:
            reaction_projection = self.reaction_projection_head(reaction_hidden)
        else:
            reaction_projection = reaction_hidden

        sequence_projection = self.enzyme_projection_head(sequence_hidden)

        output = {
            "reaction_hidden": reaction_hidden,
            "reaction_projection": reaction_projection,
            "sequence_hidden": sequence_hidden,
            "sequence_projection": sequence_projection,
        }

        return output

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--reaction_backbone_model",
            type=str,
            action=set_nox_type("model"),
            default="resnet18",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--enzyme_backbone_model",
            type=str,
            action=set_nox_type("model"),
            default="resnet18",
            help="Name of encoder to use",
        )
        parser.add_argument(
            "--num_ftrs",
            type=int,
            default=512,
            help="Number of hidden features in the projection head",
        )
        parser.add_argument(
            "--out_dim",
            type=int,
            default=128,
            help="Number of output features in the projection head",
        )
        parser.add_argument(
            "--freeze_reaction_mapper",
            action="store_true",
            default=False,
            help="Freeze the reaction backbone",
        )
        parser.add_argument(
            "--project_reaction_hidden",
            action="store_true",
            default=False,
            help="Project the reaction hidden state",
        )
