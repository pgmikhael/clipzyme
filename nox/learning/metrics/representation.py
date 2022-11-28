from collections import OrderedDict
from nox.utils.classes import uspto
from nox.utils.registry import register_object
import torch


def get_sample_similarity(h1, h2):
    cov = torch.mm(h1, h2.T)
    self_mean = torch.mean(torch.diag(cov))
    other_mean = torch.mean(cov - torch.diag(torch.diag(cov)))
    return self_mean, other_mean


def get_feature_similarity(h):
    return torch.std(h, dim=0).mean()


@register_object("contrastive_stat", "metric")
class ContrastiveStatistics(uspto):
    def __init__(self, args) -> None:
        super().__init__()
        keys_for_contrastive_views_as_tuples = []
        for keypair in args.keys_for_contrastive_views:
            keys_for_contrastive_views_as_tuples.append(tuple(keypair.split("/")))

        args.keys_for_contrastive_views = keys_for_contrastive_views_as_tuples
        self.metric_keys = args.keys_for_contrastive_views

    @property
    def metric_keys(self):
        return ["reaction_projection", "sequence_projection"]

    def __call__(self, logging_dict, args):
        stats_dict = OrderedDict()
        k1, k2 = "reaction_projection", "sequence_projection"
        h1 = torch.nn.functional.normalize(logging_dict[k1], dim=1)
        h2 = torch.nn.functional.normalize(logging_dict[k2], dim=1)
        (
            stats_dict["self_similarity"],
            stats_dict["other_similarity"],
        ) = get_sample_similarity(h1, h2)
        stats_dict["{}_feature_std".format(k1)] = get_feature_similarity(h1)
        stats_dict["{}_feature_std".format(k2)] = get_feature_similarity(h2)

        return stats_dict


@register_object("alignment_uniformity", "metric")
class AlignmentUniformityLoss(object):
    def __init__(self, args) -> None:
        super().__init__()
        """
        https://arxiv.org/abs/2005.10242 
        """

    @property
    def metric_keys(self):
        return ["reaction_projection", "sequence_projection"]

    def __call__(self, logging_dict, args):
        stats_dict = OrderedDict()

        z1 = logging_dict["reaction_projection"]
        z2 = logging_dict["sequence_projection"]

        # Normalize
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        alignment_loss = (z1 - z2).norm(dim=1).pow(args.alignment_alpha).mean()
        z1_uniformity_loss = (
            (torch.pdist(z1, p=2).pow(2))
            .mul(-args.uniformity_temperature)
            .exp()
            .mean()
            .log()
        )
        z2_uniformity_loss = (
            (torch.pdist(z2, p=2).pow(2))
            .mul(-args.uniformity_temperature)
            .exp()
            .mean()
            .log()
        )
        uniformity_loss = (z1_uniformity_loss + z2_uniformity_loss) / 2

        stats_dict["alignment"] = alignment_loss
        stats_dict["uniformity"] = uniformity_loss

        return stats_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--alignment_alpha",
            type=float,
            default=2.0,
            help="Exponent for alignment loss",
        )
        parser.add_argument(
            "--uniformity_temperature",
            type=float,
            default=2.0,
            help="Temperature for uniformity loss",
        )
