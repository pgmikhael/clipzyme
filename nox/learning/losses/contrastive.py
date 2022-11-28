from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from nox.utils.classes import Nox

EPSILON = 1e-6


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@register_object("ntxent_loss", "loss")
class NTexntLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        """
        Computes contrastive loss batch-wise

        Expects model_output to contain [projection_hidden_1, projection_hidden2, prediction_hidden_1, prediction_hidden_2]

        Returns:
            loss: cross contrastive loss with stop grad
            l_dict (dict): dictionary containing simsiam_loss detached from computation graph
            p_dict (dict): dictionary of model's reconstruction of x
        """
        l_dict, p_dict = OrderedDict(), OrderedDict()

        N = model_output["reaction_projection"].shape[0] - 1
        p1, p2 = (
            model_output["reaction_projection"],
            model_output["sequence_projection"],
        )

        # Normalize
        p1 = nn.functional.normalize(p1, dim=1)
        p2 = nn.functional.normalize(p2, dim=1)

        # calculate negative cosine similarity
        similarity = torch.einsum("ic, jc -> ij", p1, p2)

        if args.use_hard_sampling:
            neg_terms = (similarity.fill_diagonal(0),)
            pos_terms = torch.diag(similarity)
            reweight = (args.beta * neg_terms) / neg_terms.mean()
            Neg = max(
                (-N * args.tau_plus * pos_terms + reweight * neg_terms).sum()
                / (1 - args.tau_plus),
                torch.exp(-1 / args.temperature),
            )
            loss = -torch.log(pos_terms.sum() / (pos_terms.sum() + Neg))
        else:
            y = torch.arange(similarity.shape[0], device=similarity.device)
            loss = F.cross_entropy(similarity, y)
            l_dict["contrastive_loss"] = loss.detach()

        logits = similarity.detach()
        probs = torch.softmax(logits, dim=-1)
        if logits.shape[-1] != args.batch_size:
            logits = torch.cat(
                [
                    logits,
                    torch.zeros(
                        logits.shape[0],
                        args.batch_size - logits.shape[1],
                        device=similarity.device,
                    ),
                ],
                dim=-1,
            )
            probs = torch.cat(
                [
                    probs,
                    torch.zeros(
                        probs.shape[0],
                        args.batch_size - probs.shape[1],
                        device=similarity.device,
                    ),
                ],
                dim=-1,
            )
        p_dict["logits"] = logits
        p_dict["probs"] = probs
        p_dict["golds"] = y

        p_dict["reaction_hidden"] = model_output["reaction_hidden"].detach()
        p_dict["sequence_hidden"] = model_output["sequence_hidden"].detach()

        return loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--ntxent_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        parser.add_argument(
            "--use_hard_sampling",
            action="store_true",
            default=False,
            help="Whether to use hard negative sampling.",
        )
        parser.add_argument(
            "--tau_plus",
            type=float,
            default=1.0,
            help="prior over probability of positive class.",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="hardness value.",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="temperature value.",
        )


@register_object("barlow_loss", "loss")
class BarlowLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()

        z1 = model_output["reaction_projection"]
        z2 = model_output["sequence_projection"]

        # normalize repr. along the batch dimension

        z1_norm = (z1 - torch.mean(z1, dim=0)) / (torch.std(z1, dim=0) + EPSILON)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / (torch.std(z2, dim=0) + EPSILON)

        # cross-correlation matrix
        cor_mat = torch.mm(z1_norm.T, z2_norm) / z1.shape[0]
        # if args.distributed_backend == 'ddp':
        #    torch.distributed.all_reduce(cor_mat)

        # loss
        on_diag = torch.diagonal(cor_mat).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cor_mat).pow_(2).sum()
        loss = args.barlow_loss_lambda * (
            on_diag + args.redundancy_reduction_lambda * off_diag
        )

        logging_dict["barlow_loss"] = loss.detach()

        predictions["reaction_hidden"] = model_output["reaction_hidden"].detach()
        predictions["sequence_hidden"] = model_output["sequence_hidden"].detach()

        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--barlow_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )


@register_object("dclw_loss", "loss")
class DCLWLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        """
        Computes decoupled contrastive learning loss

        Expects model_output to contain [projection_hidden_1, projection_hidden2, prediction_hidden_1, prediction_hidden_2]

        Returns:
            loss: dclw loss
            l_dict (dict): dictionary containing barlow_loss detached from computation graph
            p_dict (dict): dictionary of model's reconstruction of x
        """

        def von_mises_fisher_w(sim_vec):
            n = torch.exp(sim_vec / args.dclw_sigma)
            return 2 - n / n.mean()

        l_dict, p_dict = OrderedDict(), OrderedDict()

        z1, z2 = (
            model_output["reaction_projection"],
            model_output["sequence_projection"],
        )

        # Normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # loss
        similarity_matrix_1 = (
            torch.einsum("ic, jc -> ij", z1, z1) / args.dclw_temperature
        )
        similarity_matrix_2 = (
            torch.einsum("ic, jc -> ij", z2, z2) / args.dclw_temperature
        )
        similarity_matrix_12 = (
            torch.einsum("ic, jc -> ij", z1, z2) / args.dclw_temperature
        )

        pos_terms = torch.diag(similarity_matrix_12)
        if args.use_von_mises_fisher:
            pos_loss = (-pos_terms * von_mises_fisher_w(pos_terms)).sum()
        else:
            pos_loss = -pos_terms.sum()

        neg_terms = [
            similarity_matrix_1.fill_diagonal_(0),
            similarity_matrix_2.fill_diagonal_(0),
            similarity_matrix_12.fill_diagonal_(0),
        ]
        neg_exp_terms = [torch.exp(n) for n in neg_terms]
        neg_loss = (
            torch.log(neg_exp_terms[0].sum(-1) + neg_exp_terms[2].sum(-1)).sum()
            + torch.log(neg_exp_terms[1].sum(-1) + neg_exp_terms[2].sum(-1)).sum()
        )

        loss = neg_loss + 2 * pos_loss

        l_dict["dclw_loss"] = args.dclw_loss_lambda * loss.detach()

        p_dict["reaction_hidden"] = model_output["reaction_hidden"].detach()
        p_dict["sequence_hidden"] = model_output["sequence_hidden"].detach()

        return args.dclw_loss_lambda * loss, l_dict, p_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--dclw_loss_lambda",
            type=float,
            default=1.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        parser.add_argument(
            "--dclw_temperature",
            type=float,
            default=1.0,
            help="temperature for dcl-w loss.",
        )
