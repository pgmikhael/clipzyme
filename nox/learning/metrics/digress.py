from typing import Dict
from nox.utils.registry import register_object
from nox.utils.classes import Nox
from nox.utils.digress.rdkit_functions import compute_molecular_metrics
import torch
from torchmetrics import Metric, MeanAbsoluteError, MetricCollection
from torch import Tensor


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


@register_object("molecule_sampling_metrics", "metric")
class SamplingMolecularMetrics(Metric, Nox):
    def __init__(self, args):
        super().__init__()
        dataset_infos = args.dataset_statistics
        train_smiles = args.train_smiles
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims["X"])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims["E"])
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        num_atoms_max = di.max_n_nodes
        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer("n_target_dist", n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer("node_target_dist", node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer("edge_target_dist", edge_target_dist)

        valency_target_dist = di.valency_distribution.type_as(
            self.generated_valency_dist.edgepernode_dist
        )
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer("valency_target_dist", valency_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.dataset_info = di
        self.add_state("preds_smiles", [], dist_reduce_fx="cat")

    @property
    def metric_keys(self):
        return ["masked_pred_X", "masked_pred_E", "true_X", "true_E"]

    def update(self, predictions_dict, args):
        molecules = predictions_dict["molecules"]
        self.preds_smiles.extend(molecules)

        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

    def compute(self) -> Dict:

        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            self.preds_smiles, self.train_smiles, self.dataset_info
        )

        generated_node_dist = self.generated_node_dist.compute()
        generated_edge_dist = self.generated_edge_dist.compute()
        generated_valency_dist = self.generated_valency_dist.compute()

        stats_dict = {}
        for i, atom_type in enumerate(self.dataset_info.atom_decoder):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            stats_dict[f"{atom_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
        ):
            generated_probability = generated_edge_dist[j]
            target_probability = self.edge_target_dist[j]

            stats_dict[f"bond_{bond_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            stats_dict[f"valency_{valency}_dist"] = (
                generated_probability - target_probability
            ).item()

        for key, value in stability.items():
            stats_dict[key] = value

        for key, value in rdkit_metrics[0].items():
            stats_dict[key] = value

        return stats_dict

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--store_classwise_metrics",
            action="store_true",
            default=False,
            help="Whether to log metrics per class or just log average across classes",
        )


class CEPerClass(Metric):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {
            "H": HydrogenCE,
            "C": CarbonCE,
            "N": NitroCE,
            "O": OxyCE,
            "F": FluorCE,
            "B": BoronCE,
            "Br": BrCE,
            "Cl": ClCE,
            "I": IodineCE,
            "P": PhosphorusCE,
            "S": SulfurCE,
            "Se": SeCE,
            "Si": SiCE,
        }

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


@register_object("molecule_classification_metrics", "metric")
class TrainMolecularMetricsDiscrete(Metric, Nox):
    def __init__(self, args):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=args.dataset_statistics)
        self.train_bond_metrics = BondMetricsCE()

    def update(self, predictions_dict, args) -> Dict:
        masked_pred_X = predictions_dict["masked_pred_X"]
        masked_pred_E = predictions_dict["masked_pred_E"]
        true_X = predictions_dict["true_X"]
        true_E = predictions_dict["true_E"]

        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)

    def compute(self):
        stats_dict = {}
        for key, val in self.train_atom_metrics.compute().items():
            stats_dict[key] = val.item()
        for key, val in self.train_bond_metrics.compute().items():
            stats_dict[key] = val.item()

        return stats_dict

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()
