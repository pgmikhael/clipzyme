from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
from collections import OrderedDict
from nox.utils.classes import Nox
from collections import defaultdict
from nox.utils.wln_processing import get_product_smiles
from rdkit import Chem
from torch_geometric.utils import scatter
import pickle


def get_pair_label(graph_edits, num_atoms):
    """Construct labels for each pair of atoms in reaction center prediction

    Parameters
    ----------
    graph_edits : list
        list of tuples (bond1, bond2, change)
    num_atoms : int
        number of atoms in reactants

    Returns
    -------
    float32 tensor of shape (n, n, 5)
        Labels constructed. n for the number of atoms in the reactants.
    """
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    bond_change_to_id = {0.0: 0, 1: 1, 2: 2, 3: 3, 1.5: 4}
    pair_to_changes = defaultdict(list)
    for a1, a2, change in graph_edits:
        atom1 = int(a1)
        atom2 = int(a2)
        change = bond_change_to_id[float(change)]
        pair_to_changes[(atom1, atom2)].append(change)
        pair_to_changes[(atom2, atom1)].append(change)

    labels = torch.zeros((num_atoms, num_atoms, 5))
    for pair in pair_to_changes.keys():
        i, j = pair
        labels[i, j, pair_to_changes[(j, i)]] = 1.0

    return labels


@register_object("reactivity_loss", "loss")
class ReactivityLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        s_uv = model_output["s_uv"]  # N x N x num_classes

        mol_sizes = torch.bincount(batch["reactants"].batch)
        labels = []
        mask = []
        for idx, bond_changes in enumerate(batch["reactants"].bond_changes):
            labels.append(get_pair_label(bond_changes, mol_sizes[idx]))
            # mask: do not apply loss to nodes in different sample
            # consider also masking out self-nodes
            mask.append(
                torch.ones(*labels[-1].shape[:2])
            )  # torch.ones_like(labels[-1])

        # labels_block = torch.stack([torch.block_diag(*[l[...,i] for l in labels]).to(s_uv.device) for i in range(labels[-1].shape[-1])], -1)
        # mask_block = torch.block_diag(*mask).unsqueeze(-1).to(s_uv.device)

        dim1 = sum([label.size(0) for label in labels])
        dim2 = sum([label.size(1) for label in labels])
        dim3 = labels[0].size(2)  # could just set to 5?

        labels_block = torch.zeros(dim1, dim2, dim3).to(s_uv.device)
        mask_block = torch.zeros(dim1, dim2).to(s_uv.device)

        # Populate diagonal blocks
        cur_i = 0
        for label, mask_val in zip(labels, mask):
            n_i, n_j, _ = label.size()
            labels_block[cur_i : (cur_i + n_i), cur_i : (cur_i + n_j), :] = label
            mask_block[cur_i : (cur_i + n_i), cur_i : (cur_i + n_j)] = mask_val
            cur_i += n_i

        # mask_block.fill_diagonal_(0)
        mask_block = mask_block.unsqueeze(-1)

        # compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s_uv, labels_block, weight=mask_block, reduction="sum"
        ) / len(labels)
        logging_dict["reactivity_loss"] = loss.detach()

        predictions["golds"] = labels_block.detach()
        predictions["probs"] = torch.sigmoid(s_uv).detach()
        predictions["mask"] = mask_block.detach()
        predictions["preds"] = predictions["probs"] > 0.5
        predictions["candidate_bond_changes"] = model_output[
            "candidate_bond_changes"
        ]  # for topk metric
        predictions["real_bond_changes"] = model_output[
            "real_bond_changes"
        ]  # for topk metric

        return loss, logging_dict, predictions


@register_object("candidate_loss", "loss")
class CandidateLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = model_output[
            "candidate_logit"
        ]  # list for each reaction of [score per candidate]
        label = torch.zeros(1, dtype=torch.long, device=logits[0].device)

        loss = 0  # torch.tensor(0.0, device=logit[0].device)
        probs = []
        for i, logit in enumerate(logits):
            logit = logit.view(1, -1)
            loss = loss + torch.nn.functional.cross_entropy(
                logit, label, reduction="sum"
            )  # may need to unsqueeze
            probs.append(torch.softmax(logit, dim=-1))
        loss = loss / len(logits)

        logging_dict["candidate_loss"] = loss.detach()
        predictions["candidate_golds"] = label.repeat(len(logits))
        predictions["candidate_probs"] = [p.detach() for p in probs]
        predictions["candidate_preds"] = torch.concat(
            [torch.argmax(p).unsqueeze(-1) for p in probs]
        )
        predictions["product_candidates_list"] = model_output["product_candidates_list"]
        predictions["reactant_smiles"] = batch["reactants"].smiles
        predictions["product_smiles"] = batch["products"].smiles
        predictions["all_product_smiles"] = batch["all_smiles"]

        batch_real_bond_changes = []
        for i in range(len(batch["reactants"]["bond_changes"])):
            reaction_real_bond_changes = []
            for elem in batch["reactants"]["bond_changes"][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)
        predictions["real_bond_changes"] = batch_real_bond_changes  # for topk metric

        return loss, logging_dict, predictions


@register_object("multi_candidate_loss", "loss")
class MultiCandidateLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = model_output[
            "candidate_logit"
        ]  # list for each reaction of [score per candidate]

        loss = 0  # torch.tensor(0.0, device=logit[0].device)
        probs, labels = [], []
        for i, logit in enumerate(logits):
            reactant_mol = Chem.MolFromSmiles(batch["reactants"].smiles[i])
            logit = logit.view(1, -1)
            label = torch.zeros_like(logit)
            valid_bond_changes = [t[1] for t in batch["all_smiles"][i]]
            valid_bond_changes = [
                set((x, y, t, 0) for x, y, t in bc) for bc in valid_bond_changes
            ]
            valid_products = [
                get_product_smiles(
                    reactant_mol, bc, None, mode="test", return_full_mol=True
                )
                for bc in valid_bond_changes
            ]  # bond changes
            label[:, 0] = 1
            for j, cand_smile in enumerate(
                model_output["product_candidates_list"][i].smiles
            ):
                if cand_smile in valid_products:
                    label[:, j] = 1
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                logit, label, reduction="sum"
            )  # may need to unsqueeze
            probs.append(torch.softmax(logit, dim=-1))
            labels.append(label)
        loss = loss / len(logits)

        logging_dict["candidate_loss"] = loss.detach()
        predictions["candidate_golds"] = labels
        predictions["candidate_probs"] = [p.detach() for p in probs]
        predictions["candidate_preds"] = [(p > 0.5).int() for p in probs]
        predictions["product_candidates_list"] = model_output["product_candidates_list"]
        predictions["reactant_smiles"] = batch["reactants"].smiles
        predictions["product_smiles"] = batch["products"].smiles
        predictions["all_product_smiles"] = batch["all_smiles"]

        batch_real_bond_changes = []
        for i in range(len(batch["reactants"]["bond_changes"])):
            reaction_real_bond_changes = []
            for elem in batch["reactants"]["bond_changes"][i]:
                reaction_real_bond_changes.append(tuple(elem))
            batch_real_bond_changes.append(reaction_real_bond_changes)
        predictions["real_bond_changes"] = batch_real_bond_changes  # for topk metric

        return loss, logging_dict, predictions


@register_object("hierarchical_ec_loss", "loss")
class HierarchicalECLoss(Nox):
    """hierarchical / subclass loss for ec classification

    assumes one class per protein (which is true for 95% of proteins)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ec_to_level = lambda ec, level: ".".join(ec.split(".")[: int(level)])

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = model_output["ec_logits"]
        probs = torch.softmax(logits, dim=-1)
        batch_size, _ = logits.shape

        level4_ecs = {i: c for c, i in args.ec_levels["4"].items()}
        level4_ecs = [
            level4_ecs[i] for i in range(len(level4_ecs))
        ]  # ecs in order as they appear in logits

        loss = 0
        for level in sorted(args.ec_levels):
            ec2index = args.ec_levels[level]

            target_index = torch.tensor(
                [
                    args.ec_levels[level][self.ec_to_level(ec, level)]
                    for ec in level4_ecs
                ]
            ).to(logits.device)
            level_probs = scatter(probs, target_index, dim=1, reduce="sum")
            level_loss = (
                F.cross_entropy(torch.log_softmax(level_probs, -1), batch[f"ec{level}"])
                * args.hierarchical_ec_loss_lambdas[int(level) - 1]
            )
            logging_dict[f"ec{level}_loss"] = level_loss.detach()
            loss = level_loss + loss

        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--hierarchical_ec_loss_lambdas",
            nargs="*",
            default=[1.0, 1.0, 1.0, 1.0],
            type=float,
            help="weight for each level",
        )


@register_object("binary_hierarchical_ec_loss", "loss")
class BottomUpECLoss(Nox):
    """hierarchical / subclass loss for ec classification

    NOTE: training to predict "not in" EC rather than "in" EC

    prob(x in 2.1.1)    = prob(x in 2.1.1.1 or x in 2.1.1.2 or ... or x in 2.1.1.100)
                        = 1 - prob(x not in 2.1.1.1 and x not in 2.1.1.2 ... and x not in 2.1.1.100)
                        = 1 - prob(x not in 2.1.1.1)p(x not in 2.1.1.2)...p(x not in 2.1.1.100)
                        = 1 - prob(x not in 2.1.1)
    log prob(x not in 2.1.1) = sum_i log prob(x not in 2.1.1.i)
    """

    def __init__(self) -> None:
        super().__init__()

        self.ec_to_level = lambda ec, level: ".".join(ec.split(".")[: int(level)])

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = (
            model_output["ec_logits"]
            if "ec_logits" in model_output
            else model_output["logit"]
        )
        logit_scale = model.model.logit_scale.exp()
        probs = F.logsigmoid(logits * logit_scale)
        batch_size, _ = logits.shape

        # class index to EC
        level4_ecs = {i: c for c, i in args.ec_levels["4"].items()}
        # list of ecs in order as they appear in yvec
        level4_ecs = [level4_ecs[i] for i in range(len(level4_ecs))]

        loss = 0
        for level in sorted(args.ec_levels):
            ec2index = args.ec_levels[level]  # comes from data

            # for each ec in level4 order, where does it fall in yvec
            target_index = torch.tensor(
                [ec2index[self.ec_to_level(ec, level)] for ec in level4_ecs]
            ).to(logits.device)
            # combine probabilities into the right index for this level yvec
            # level_probs = 1 - scatter(1 - probs, target_index, dim=1, reduce="mul")
            level_probs = scatter(probs, target_index, dim=1, reduce="sum")  # log
            level_probs = torch.exp(level_probs)
            # BCE loss doesn't support autocasting
            with torch.cuda.amp.autocast(enabled=False):
                # ! NOTE: labels is 1-y
                level_loss = (
                    F.binary_cross_entropy(level_probs, 1 - batch[f"ec{level}"])
                    * args.hierarchical_ec_loss_lambdas[int(level) - 1]
                )
            logging_dict[f"ec{level}_loss"] = level_loss.detach()
            loss = level_loss + loss

            predictions[f"golds_ec{level}"] = batch[f"ec{level}"]
            predictions[f"probs_ec{level}"] = 1 - level_probs.detach()

        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--hierarchical_ec_loss_lambdas",
            nargs="*",
            default=[1.0, 1.0, 1.0, 1.0],
            type=float,
            help="weight for each level",
        )


@register_object("multi_label_ec_loss", "loss")
class MultiLabelECLoss(Nox):
    """Multi-label loss for EC classification."""

    def __init__(self) -> None:
        super().__init__()

        self.ec_to_level = lambda ec, level: ".".join(ec.split(".")[: int(level)])

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = (
            model_output["ec_logits"]
            if "ec_logits" in model_output
            else model_output["logit"]
        )
        if hasattr(
            model.model, "logit_ec_scale"
        ):  # if using ec_head in CLIP then use logit_ec_scale
            logit_scale = model.model.logit_ec_scale.exp()
        else:
            logit_scale = model.model.logit_scale.exp()
        loss = F.binary_cross_entropy_with_logits(logits * logit_scale, batch["ec4"])

        logging_dict["ec_loss"] = loss.detach()

        predictions["golds_ec"] = batch["ec"]
        predictions["probs_ec"] = torch.sigmoid(logits).detach()

        ###############

        # For metrics
        probs = F.logsigmoid(logits * logit_scale)
        # class index to EC
        level4_ecs = {i: c for c, i in args.ec_levels["4"].items()}
        # list of ecs in order as they appear in yvec
        level4_ecs = [level4_ecs[i] for i in range(len(level4_ecs))]

        for level in sorted(args.ec_levels):
            ec2index = args.ec_levels[level]  # comes from data

            # for each ec in level4 order, where does it fall in yvec
            target_index = torch.tensor(
                [ec2index[self.ec_to_level(ec, level)] for ec in level4_ecs]
            ).to(logits.device)
            # combine probabilities into the right index for this level yvec
            # level_probs = 1 - scatter(1 - probs, target_index, dim=1, reduce="mul")
            level_probs = scatter(probs, target_index, dim=1, reduce="sum")  # log
            level_probs = torch.exp(level_probs)

            predictions[f"golds_ec{level}"] = batch[f"ec{level}"]
            predictions[f"probs_ec{level}"] = 1 - level_probs.detach()

        return loss, logging_dict, predictions
