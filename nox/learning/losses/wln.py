from nox.utils.registry import register_object
import torch 
import torch.nn.functional as F
from collections import OrderedDict
from nox.utils.classes import Nox
from collections import defaultdict 

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
        s_uv = model_output["s_uv"] # N x N x num_classes

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

        # block diagonal only works for 2D tensors
        # labels_block = torch.block_diag(*labels).to(s_uv.device)
        # mask_block = torch.block_diag(*mask).to(s_uv.device)
        dim1 = sum([label.size(0) for label in labels])
        dim2 = sum([label.size(1) for label in labels])
        dim3 = labels[0].size(2)  # could just set to 5?

        labels_block = torch.zeros(dim1, dim2, dim3).to(s_uv.device)
        mask_block = torch.zeros(dim1, dim2, dim3).to(s_uv.device)

        # Populate diagonal blocks
        cur_i = 0
        for label, mask_val in zip(labels, mask):
            n_i, n_j, _ = label.size()
            labels_block[cur_i:(cur_i + n_i), cur_i:(cur_i + n_j), :] = label
            mask_block[cur_i:(cur_i + n_i), cur_i:(cur_i + n_j), :] = mask_val
            cur_i += n_i

        # compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s_uv, labels_block, weight=mask_block, reduction="sum") / mask_block.sum()
        logging_dict["reactivity_loss"] = loss.detach()

        predictions["golds"] = labels_block.detach()
        predictions["probs"] = torch.sigmoid(s_uv).detach()

        # compute preds
        # max_indices = torch.argmax(s_uv, dim=2)
        # output_tensor = torch.zeros_like(s_uv)
        # output_tensor.scatter_(2, max_indices.unsqueeze(2), 1)
        # predictions["preds"] = output_tensor.detach()
        predictions["preds"] = predictions['probs'] > 0.5

        return loss, logging_dict, predictions


@register_object("candidate_loss", "loss")
class CandidateLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, model_output, batch, model, args):
        logging_dict, predictions = OrderedDict(), OrderedDict()
        logits = model_output["logit"] # list for each reaction of [score per candidate]
        label = torch.zeros(1, dtype=torch.long, device = logits[0].device)
        
        loss = 0 # torch.tensor(0.0, device=logit[0].device)
        probs = []
        for i, logit in enumerate(logits):
            logit = logit.view(1,-1)
            loss = loss + torch.nn.functional.cross_entropy(logit, label, reduction="sum")  # may need to unsqueeze
            probs.append(torch.softmax(logit, dim=-1))
        loss = loss / len(logits)

        logging_dict["candidate_loss"] = loss.detach()
        predictions["golds"] = label.repeat(len(logits))
        predictions["probs"] = [p.detach() for p in probs]
        predictions["preds"] = torch.concat([torch.argmax(p).unsqueeze(-1) for p in probs])

        return loss, logging_dict, predictions