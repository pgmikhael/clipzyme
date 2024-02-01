from collections import OrderedDict
from clipzyme.utils.classes import Nox
from torchmetrics import Metric
from clipzyme.utils.registry import register_object
import torch
from typing import Dict


def get_sample_similarity(h1, h2):
    cov = torch.mm(h1, h2.T)
    self_mean = torch.mean(torch.diag(cov))
    other_mean = torch.mean(cov - torch.diag(torch.diag(cov)))
    return self_mean, other_mean


def get_feature_similarity(h):
    return torch.std(h, dim=0).mean()


@register_object("contrastive_stat", "metric")
class ContrastiveStatistics(Nox):
    def __init__(self, args) -> None:
        super().__init__()
        # keys_for_contrastive_views_as_tuples = []
        # for keypair in args.keys_for_contrastive_views:
        #     keys_for_contrastive_views_as_tuples.append(tuple(keypair.split("/")))

        # args.keys_for_contrastive_views = keys_for_contrastive_views_as_tuples
        # self.metric_keys = args.keys_for_contrastive_views
        # self.metric_keys = self.metric_keys()

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

    # @staticmethod
    # def add_args(parser) -> None:
    #     """Add class specific args

    #     Args:
    #         parser (argparse.ArgumentParser): argument parser
    #     """
    #     parser.add_argument(
    #         "--keys_for_contrastive_views",
    #         type=float,
    #         default=2.0,
    #         help="Exponent for alignment loss",
    #     )


@register_object("alignment_uniformity", "metric")
class AlignmentUniformityLoss(Nox):
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


@register_object("clip_classification", "metric")
class ClipClassification(Metric, Nox):
    def __init__(self, args) -> None:
        """
        Computes standard classification metrics

        Args:
            predictions_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'preds', 'golds']
            args: argparser Namespace

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            Binary: two labels
            Multiclass: more than two labels
            Multilabel: potentially more than one label per sample (independent classes)
        """
        super().__init__()
        self.add_state("num_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["clip_preds", "clip_golds"]

    def update(self, predictions_dict, args) -> Dict:
        preds = predictions_dict["clip_preds"]  # B
        golds = predictions_dict["clip_golds"].int()  # B

        self.num_correct += (golds == preds).sum()
        self.total += len(golds)

    def compute(self) -> Dict:
        stats_dict = {
            "clip_accuracy": self.num_correct.float() / self.total,
        }
        return stats_dict


@register_object("clip_quantile", "metric")
class ClipQuantile(Metric, Nox):
    def __init__(self, args) -> None:
        """
        Computes the quantile rank of a score relative to other scores pairwise similarity matrix
        """
        super().__init__()
        self.add_state("quantile", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["clip_probs", "clip_golds"]

    def update(self, predictions_dict, args) -> Dict:
        # select diagonal
        # check fraction > diagonal
        probs = predictions_dict["clip_probs"]  # N x K
        indices = predictions_dict["clip_golds"]
        # generalize for ddp gathering: pos_score = torch.diagonal(probs).unsqueeze(-1)
        pos_score = probs.gather(1, indices.unsqueeze(1))
        num_less_than_score = probs < pos_score
        score_quantile = num_less_than_score.float().mean(1).sum()
        self.quantile += score_quantile
        self.total += probs.shape[0]

    def compute(self) -> Dict:
        stats_dict = {
            "clip_quantile": self.quantile.float() / self.total,
        }
        return stats_dict
