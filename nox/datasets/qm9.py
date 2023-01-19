import argparse
import numpy as np
import torch
import warnings
import os
import os.path as osp
import pathlib
from typing import List, Any, Sequence
from tqdm import tqdm
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.utils import subgraph

from nox.utils.digress.rdkit_functions import (
    mol2smiles,
    build_molecule_with_partial_charges,
    compute_molecular_metrics,
)
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
import nox.utils.digress.diffusion_utils as utils
from nox.utils.digress.extra_features import ExtraFeatures
from nox.utils.digress.extra_features_molecular import ExtraMolecularFeatures


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.type_as(batch_n_nodes)
        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)

        return log_p


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class DatasetInfo:
    def __init__(self, datasets, args):

        self.datasets = datasets
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )

        self.name = "qm9"
        if self.remove_h:
            self.atom_encoder = {"C": 0, "N": 1, "O": 2, "F": 3}
            self.atom_decoder = ["C", "N", "O", "F"]
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 9
            self.max_weight = 150
            self.n_nodes = torch.Tensor(
                [
                    0,
                    2.2930e-05,
                    3.8217e-05,
                    6.8791e-05,
                    2.3695e-04,
                    9.7072e-04,
                    0.0046472,
                    0.023985,
                    0.13666,
                    0.83337,
                ]
            )
            self.node_types = torch.Tensor([0.7230, 0.1151, 0.1593, 0.0026])
            self.edge_types = torch.Tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])

            self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.Tensor(
                [2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073]
            )
        else:
            self.atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
            self.atom_decoder = ["H", "C", "N", "O", "F"]
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.Tensor(
                [
                    0,
                    0,
                    0,
                    1.5287e-05,
                    3.0574e-05,
                    3.8217e-05,
                    9.1721e-05,
                    1.5287e-04,
                    4.9682e-04,
                    1.3147e-03,
                    3.6918e-03,
                    8.0486e-03,
                    1.6732e-02,
                    3.0780e-02,
                    5.1654e-02,
                    7.8085e-02,
                    1.0566e-01,
                    1.2970e-01,
                    1.3332e-01,
                    1.3870e-01,
                    9.4802e-02,
                    1.0063e-01,
                    3.3845e-02,
                    4.8628e-02,
                    5.4421e-03,
                    1.4698e-02,
                    4.5096e-04,
                    2.7211e-03,
                    0.0000e00,
                    2.6752e-04,
                ]
            )

            self.node_types = torch.Tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            self.edge_types = torch.Tensor(
                [0.88162, 0.11062, 5.9875e-03, 1.7758e-03, 0]
            )

            self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.Tensor(
                [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
            )

        if args.recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = self.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt("n_counts.txt", self.n_nodes.numpy())
            self.node_types = self.node_types()  # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt("atom_types.txt", self.node_types.numpy())

            self.edge_types = self.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt("edge_types.txt", self.edge_types.numpy())

            valencies = self.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt("valencies.txt", valencies.numpy())
            self.valency_distribution = valencies
            assert False

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.datasets[split][-1]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.datasets["train"][-1]:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.datasets[split][-1]):
                counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.datasets["train"][-1]:
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.Tensor(num_classes)

        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.datasets[split][-1]):
                unique, counts = torch.unique(data.batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index.shape[1]
                num_non_edges = all_pairs - num_edges

                edge_types = data.edge_attr.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.datasets[split][-1]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, extra_features, domain_features):
        example_batch = [self.datasets["train"][-1][0], self.datasets["train"][-1][1]]
        example_batch = Batch.from_data_list(example_batch, None, None)

        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": example_batch["y"].size(1) + 1,
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }


@register_object("moleculenet", "qm9")
class QM9(AbstractDataset, InMemoryDataset):

    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        self.split_group = split_group
        self.args = args
        self.remove_h = args.remove_h

        self.version = self.get_version()

        self.name = self.get_name
        self.root = args.data_dir
        # self.version = None

        transform = self.get_transform(args)
        InMemoryDataset.__init__(self, root=self.root, transform=transform)

        self.load_datasets(args)

        self.dataset = self.create_dataset(split_group)

        # get data info (input / output dimensions)
        self.data_info = DatasetInfo(self.datasets, args)

        extra_features = ExtraFeatures(args.extra_features, dataset_info=self.data_info)
        domain_features = ExtraMolecularFeatures(dataset_infos=self.data_info)

        self.data_info.compute_input_output_dims(
            extra_features=extra_features,
            domain_features=domain_features,
        )

        if len(self.dataset) == 0:
            return

        self.print_summary_statement(self.dataset, split_group)

    def load_datasets(self):
        self.datasets = {}
        try:
            self.datasets["train"] = torch.load(self.processed_paths[0])
            self.datasets["dev"] = torch.load(self.processed_paths[1])
            self.datasets["test"] = torch.load(self.processed_paths[2])
        except Exception as e:
            raise Exception("Unable to load dataset", e)

    def get_transform(self, args) -> None:
        target = getattr(args, "guidance_target", None)
        regressor = getattr(self, "regressor", None)
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = None
        else:
            transform = RemoveYTransform()

        return transform

    @property
    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ["proc_tr_no_h.pt", "proc_val_no_h.pt", "proc_test_no_h.pt"]
        else:
            return ["proc_tr_h.pt", "proc_val_h.pt", "proc_test_h.pt"]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        # node and bond types
        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        for split_idx, split in enumerate(["train", "dev", "test"]):

            target_df = pd.read_csv(self.split_paths[split_idx], index_col=0)
            target_df.drop(columns=["mol_id"], inplace=True)

            with open(self.raw_paths[-1], "r") as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

            suppl = Chem.SDMolSupplier(
                self.raw_paths[0], removeHs=False, sanitize=False
            )

            data_list = []
            for i, mol in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                N = mol.GetNumAtoms()

                type_idx = []
                for atom in mol.GetAtoms():
                    type_idx.append(types[atom.GetSymbol()])

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]

                    # 2* to account for bidirectional edges (row and col take both directions)
                    # +1 to account for no bond
                    edge_type += 2 * [bonds[bond.GetBondType()] + 1]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(
                    torch.float
                )

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                y = torch.zeros((1, 0), dtype=torch.float)

                if self.remove_h:
                    type_idx = torch.Tensor(type_idx).long()
                    to_keep = type_idx > 0
                    edge_index, edge_attr = subgraph(
                        to_keep,
                        edge_index,
                        edge_attr,
                        relabel_nodes=True,
                        num_nodes=len(to_keep),
                    )
                    x = x[to_keep]
                    # Shift onehot encoding to match atom decoder
                    x = x[:, 1:]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[split_idx])

    def create_dataset(self, split_group):
        """
        Creates the dataset of samples
        """

        dataset, slices = self.datasets[split_group]

        return dataset

    def __getitem__(self, index):
        try:
            return self.dataset[index]

        except Exception:
            warnings.warn("Could not load sample")

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(QM9, QM9).add_args(parser)

        parser.add_argument(
            "--recompute_statistics",
            action="store_true",
            default=False,
            help="recompute statistics",
        )
        parser.add_argument(
            "--remove_h",
            action="store_true",
            default=False,
            help="remove hydrogens from the molecules",
        )
        parser.add_argument(
            "--extra_features",
            type=str,
            choices=["eigenvalues", "all", "cycles"],
            default=None,
            help="extra features to use",
        )


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    """

    :param dataset_name: qm9 or qm9_second_half
    :return:
    """
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print(
                "\tConverting QM9 dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
