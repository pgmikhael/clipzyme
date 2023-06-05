from nox.utils.registry import register_object
import torch 
import torch.nn.functional as F
from collections import OrderedDict
from nox.utils.classes import Nox

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
    bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
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
        labels[i, j, pair_to_changes[(j, i)]] = 1.

    return labels

@register_object("reactivity_loss", "loss")
class ReactivityLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        s_uv, s_uv_tildes = model_output["s_uv"], model_output["s_uv_tildes"] # N x N x num_classes

        mol_sizes = torch.bincount(batch["reactants"].batch)
        labels = []
        mask = []
        for idx, bond_changes in enumerate(batch["reactants"].bond_changes):
            labels.append(
                get_pair_label(bond_changes, mol_sizes[idx])
            )
            # mask: do not apply loss to nodes in different sample
            # consider also masking out self-nodes
            mask.append(
                torch.ones_like(labels[-1])
            )

        labels = torch.block_diag(*labels).to(s_uv.device)
        mask = torch.block_diag(*mask).to(s_uv.device)
    
        # compute loss
        local_loss = torch.nn.functional.binary_cross_entropy_with_logits(s_uv, labels, weight=mask, reduction="sum") / mask.sum()
        global_loss = torch.nn.functional.binary_cross_entropy_with_logits(s_uv_tildes, labels, weight=mask, reduction="sum") / mask.sum()
        logging_dict["reactivity_local_loss"] = local_loss.detach()
        logging_dict["reactivity_global_loss"] = global_loss.detach()

        loss = local_loss + global_loss

        return loss, logging_dict, predictions


@register_object("candidate_loss", "loss")
class CandidateLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logit = model_output["logit"] # list for each reaction of [score per candidate]
        labels = [torch.zeros_like(l) for l in logit]
        for l in labels:
            l[0] = 1
        
        loss = 0
        probs = []
        for pred, label in zip(preds, labels):
            loss = loss + torch.nn.functional.cross_entropy(preds, labels, reduction="sum") # may need to unsqueeze
            probs.append(torch.softmax(pred))
        loss = loss / len(preds)

        logging_dict["candidate_loss"] = loss.detach()
        predictions["golds"] = torch.concat(labels)
        predictions["probs"] = torch.concat(probs).detach()
        predictions["preds"] = torch.concat([torch.argmax(p) for p in probs])

        return loss, logging_dict, predictions