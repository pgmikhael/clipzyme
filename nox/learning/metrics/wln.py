from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
import torchmetrics
from torchmetrics import Metric
import torch
from nox.utils.wln_processing import examine_topk_candidate_product
from rdkit import Chem 

@register_object("reactivity_exact_accuracy", "metric")
class ReactivityExactAccuracy(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["probs", "preds", "golds", "mask"]

    def update(self, predictions_dict, args):
        preds = predictions_dict["preds"]
        target = predictions_dict["golds"]
        assert preds.shape == target.shape
        mask = predictions_dict["mask"]
        N = preds.shape[1]
        triu_indices = torch.triu_indices(N, N, offset=1, device=preds.device)

        correct_tensor = (preds == target).all(dim=-1)
        correct = 0
        total = 0
        for i, j in zip(triu_indices[0], triu_indices[1]):
            total += (1 * mask[i, j][0]).long()
            correct += (int(correct_tensor[i, j]) * mask[i, j][0]).long()

        self.correct += correct.to(self.device)
        self.total += total.to(self.device)

    def compute(self):
        stats_dict = {
            "exact_accuracy": self.correct.float() / self.total,
        }
        return stats_dict


@register_object("reactivity_topk", "metric")
class ReactivityTopK(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False
        self.k = args.topk_bonds # list of k values to compute topk accuracy for
        for k in args.topk_bonds:
            self.add_state(f"num_correct_{k}", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def metric_keys(self):
        return ["candidate_bond_changes", "real_bond_changes"]

    def update(self, predictions_dict, args):
        candidate_bond_changes = predictions_dict["candidate_bond_changes"]
        batch_real_bond_changes = predictions_dict["real_bond_changes"]
                
        # compute for each reaction
        for idx in range(len(candidate_bond_changes)):
            # sort bonds
            cand_changes = [(min(atom1, atom2), max(atom1, atom2), float(change_type)) for (atom1, atom2, change_type, score) in candidate_bond_changes[idx]]
            gold_bonds = [(min(atom1, atom2), max(atom1, atom2), float(change_type)) for (atom1, atom2, change_type) in batch_real_bond_changes[idx]]

            self.num_total += torch.tensor(1).to(self.device)
            for k in args.topk_bonds:
                if set(gold_bonds) <= set(cand_changes[:k]):
                    current = getattr(self, f"num_correct_{k}")
                    setattr(self, f"num_correct_{k}", current + torch.tensor(1).to(self.device))            


    def compute(self):
        stats_dict = {}
        for k in self.k:
            stats_dict[f"acc_top{k}"] = getattr(self, f"num_correct_{k}").float() / self.num_total
        return stats_dict


@register_object("candidate_topk", "metric")
class CandidateTopK(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ground", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ground_sanitized", default=torch.tensor(0), dist_reduce_fx="sum")

        for k in args.candidate_topk_values:
            self.add_state('top_{:d}'.format(k), default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state('top_{:d}_sanitized'.format(k), default=torch.tensor(0), dist_reduce_fx="sum")

        self.candidate_topk_values = args.candidate_topk_values

    @property
    def metric_keys(self):
        return ["candidate_bond_changes", "real_bond_changes", "product_candidates_list", "reactant_smiles", "product_smiles", "probs"]

    def update(self, predictions_dict, args):
        # candidate_bond_changes = predictions["candidate_bond_changes"]
        batch_real_bond_changes = predictions["real_bond_changes"]
        candidate_probs = predictions["probs"]
        product_candidates_list = predictions["product_candidates_list"]
        reactant_smiles = predictions["reactant_smiles"]
        product_smiles = predictions["product_smiles"]

        for reaction_idx, reaction_pred in enumerate(candidate_probs):
            valid_candidate_combos = product_candidates_list[reaction_idx].candidate_bond_change
            real_bond_changes = batch_real_bond_changes[reaction_idx]
            reactant_mol = Chem.MolFromSmiles(reactant_smiles[reaction_idx]) 
            product_mol = Chem.MolFromSmiles(product_smiles[reaction_idx]) 

            topk_values, topk_indices = torch.topk(reaction_pred, top_k, dim=0)
            topk_combos = [(atom1, atom2, change_type) for atom1, atom2, change_type, score in valid_candidate_combos]

            batch_found_info = examine_topk_candidate_product(
                args.candidate_topk_values, topk_combos, reactant_mol, real_bond_changes, product_mol)
            
            for k, v in batch_found_info.items():
                current = getattr(self, k)
                setattr(self, k, current + float(v))      

        self.total += len(batch_real_bond_changes)

    def compute(self):
        stats_dict = {}

        stats_dict["strict_gfound"] = self.ground.float() / self.total
        stats_dict["molvs_gfound"] = self.ground_sanitized.float() / self.total

        for k in self.candidate_topk_values:
            stats_dict[f"strict_top{k}"] = getattr(self, 'top_{:d}'.format(k)).float() / self.total
            stats_dict[f"molvs_top{k}"] = getattr(self, 'top_{:d}_sanitized'.format(k)).float() / self.total
        
        return stats_dict
    
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--candidate_topk_values",
            type=int,
            nargs="+",
            default=[1],
            help="Values of k for which to obtain top-k accuracies",
        )


@register_object("candidate_accuracy", "metric")
class WLNAccuracy(Metric, Nox):
    def __init__(self, args) -> None:
        super().__init__()
        higher_is_better: Optional[bool] = True
        is_differentiable = False
        full_state_update: bool = False

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
 
    @property
    def metric_keys(self):
        return ["probs", "preds", "golds"]

    def update(self, predictions_dict, args) -> Dict:
        preds = predictions_dict["preds"]
        target = predictions_dict["golds"]
        
        correct = (preds == target).all(dim=-1).sum()
        total = preds.shape[-1]

        self.correct += torch.tensor(correct).to(self.device)
        self.total += torch.tensor(total).to(self.device)

    def compute(self) -> Dict:
        stats_dict = {
            "per_reaction_accuracy": self.correct.float() / self.total,
        }
        return stats_dict
