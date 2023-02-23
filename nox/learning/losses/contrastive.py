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

@register_object("clip_loss", "loss")
class CLIPLoss(Nox):
    def __init__(self, args) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = {}, {}
        substrate_features = model_output["substrate_features"]
        protein_features = model_output["protein_features"]

        # cosine similarity as logits
        logit_scale = model.model.logit_scale.exp()
        logits_per_substrate = logit_scale * substrate_features @ protein_features.t()
        logits_per_protein = logits_per_substrate.t()

        # labels
        labels = torch.arange(logits_per_substrate.shape[0]).to(logits_per_substrate.device)
        loss = (
            F.cross_entropy(logits_per_substrate, labels)
            + F.cross_entropy(logits_per_protein, labels)
        ) / 2
        logging_dict["clip_loss"] = loss.detach()

        predictions["clip_probs"] = F.softmax(logits_per_substrate, dim=-1).detach().cpu()
        predictions["clip_preds"] = (
            predictions["clip_probs"].argmax(axis=-1).reshape(-1).cpu()
        )
        predictions["clip_golds"] = labels.cpu()

        predictions["projection1"] = substrate_features.detach().cpu()
        predictions["projection2"] = protein_features.detach().cpu()

        return loss, logging_dict, predictions


@register_object("supervised_contrastive_loss", "loss")
class SupConLoss(Nox):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    GitHub: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """
    def __init__(self):
        super(SupConLoss, self).__init__()

    def __call__(self, model_output, batch, model, args):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf
        
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features = model_output['logit']
        logging_dict, predictions = OrderedDict(), OrderedDict()
        device = features.device
        mask = None

        if "y" in model_output:
            predictions["golds"] = model_output["y"]
        elif "y" in batch:
            predictions["golds"] = batch["y"]
        else:
            raise KeyError("predictions_dict ERROR: y not found")
        
        labels = predictions["golds"] 
        # if args.precomputed_loss:
        #     loss = model_output["loss"]
        # else:
        #     loss = F.cross_entropy(logit.view(-1, args.num_classes), target.view(-1).long()) * args.ce_loss_lambda
        predictions["probs"] = F.softmax(features, dim=-1).detach()
        predictions["preds"] = predictions["probs"].argmax(axis=-1)
        
        if not args.keep_preds_dim:
             predictions["golds"] =  predictions["golds"].view(-1)
             predictions["preds"] =  predictions["preds"].reshape(-1)


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [B, N, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            print("Loss degenerates to SimCLR loss because labels / mask are not defined")
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if args.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif args.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(args.contrast_mode))

        # compute logits
        # logit_scale = model.model.logit_scale.exp()
        anchor_dot_contrast = torch.div(torch.matmul(logit_scale * anchor_feature, contrast_feature.T), args.supcon_temperature)
        # logits_per_substrate = logit_scale * substrate_features @ protein_features.t()
        # logits_per_protein = logits_per_substrate.t()
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (args.supcon_temperature / args.supcon_base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        logging_dict["cross_entropy_loss"] = loss.detach()
        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--supcon_temperature",
            type=float,
            default=0.07,
            help="temperature for supervised contrastive loss.",
        )
        parser.add_argument(
            "--contrast_mode",
            type=str,
            choices=["all", "one"],
            default="all",
            help="extra features to use",
        )
        parser.add_argument(
            "--supcon_base_temperature",
            type=float,
            default=0.07,
            help="temperature for supervised contrastive loss.",
        )