from typing import List, Literal
from tqdm import tqdm
import pickle
import warnings
import copy, os
import numpy as np
import random
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from nox.utils.registry import register_object, get_object
from nox.datasets.ecreact import ECReact_RXNS, ECReact_MultiProduct_RXNS
from nox.datasets.qm9 import DistributionNodes
from nox.utils.digress.extra_features import ExtraFeatures
from torch_geometric.data import Batch
from nox.utils.pyg import from_smiles, x_map, e_map
from nox.utils.smiles import get_rdkit_feature
import nox.utils.digress.diffusion_utils as diff_utils
import torch.nn.functional as F
from nox.utils.classes import classproperty
from collections import Counter
import rdkit
import torch
from collections import defaultdict
from rich import print
import hashlib
import glob
from Bio.Data.IUPACData import protein_letters_3to1
from esm import pretrained
import Bio
import Bio.PDB
from collections import Counter
from nox.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    precompute_node_embeddings,
    compute_node_embedding,
    get_sequences,
)
from torch_geometric.data import HeteroData, Data
from torch_geometric.data import Dataset
import argparse


class DatasetInfo:
    def __init__(self, smiles, args):

        pyg_molecules = [from_smiles(s) for s in smiles]
        rdkit_molecules = [rdkit.Chem.MolFromSmiles(s) for s in smiles]

        self.dataset = pyg_molecules  # list of PyG data items
        self.remove_h = args.remove_h
        periodic_table = rdkit.Chem.GetPeriodicTable()

        self.name = "react"

        self.atom_encoder = {
            periodic_table.GetElementSymbol(i): i for i in x_map["atomic_num"]
        }
        self.atom_decoder = [k for k in self.atom_encoder]
        self.num_atom_types = len(self.atom_decoder)
        self.valencies = [
            periodic_table.GetValenceList(i)[0] for i in x_map["atomic_num"]
        ]

        self.atom_weights = {
            i: round(periodic_table.GetAtomicWeight(i)) for i in x_map["atomic_num"]
        }

        num_nodes = []
        mol_weights = []
        train_atoms = []
        for m in rdkit_molecules:
            num_nodes.append(m.GetNumAtoms())
            atom_list = [a for a in m.GetAtoms()]
            mol_weights.append(
                sum(self.atom_weights[a.GetAtomicNum()] for a in atom_list)
            )
            train_atoms.extend(atom_list)

        self.max_n_nodes = max(num_nodes)  # max number of nodes in a molecule
        self.max_weight = max(mol_weights)  # max molecule weight
        num_nodes_dist = Counter(num_nodes)

        num_nodes_dist = [
            num_nodes_dist.get(k, 0) for k in range(1, self.max_n_nodes + 1)
        ]

        self.n_nodes = torch.Tensor(num_nodes_dist) / sum(
            num_nodes_dist
        )  # distribution over number of nodes in molecule

        train_atom_types = Counter([a.GetAtomicNum() for a in train_atoms])
        train_atom_types = [train_atom_types.get(i, 0) for i in x_map["atomic_num"]]

        self.node_types = torch.Tensor(train_atom_types) / sum(
            train_atom_types
        )  # distribution over nodes types in molecule

        # get distribution over edge types
        # add no-bond to 0th edge type used in from_smiles ("misc")
        all_pairs_minus_connected = sum(
            m.x.shape[0] * (m.x.shape[0] - 1) - m.edge_index.shape[1]
            for m in pyg_molecules
        )
        edge_types = torch.hstack([m.edge_attr[:, 0] for m in pyg_molecules]).tolist()
        edge_types_dist = Counter(edge_types)
        if 0 not in edge_types_dist:
            edge_types_dist[0] = all_pairs_minus_connected
        else:
            edge_types_dist[0] += all_pairs_minus_connected
        edge_types_dist = [edge_types_dist[k] for k in sorted(edge_types_dist)]
        self.edge_types = torch.Tensor(edge_types_dist) / sum(
            edge_types_dist
        )  # distribution over edge types in molecule, including a no-bond cateogry

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        # self.valency_distribution = self.valency_count(self.max_n_nodes)

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes)
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(
        self, example_batch, extra_features=None, domain_features=None
    ):
        # first feature is atomic number
        example_batch.x = F.one_hot(example_batch.x[:, 0], len(x_map["atomic_num"])).to(
            torch.float
        )
        # first feature is bond type
        example_batch.edge_attr = F.one_hot(
            example_batch.edge_attr[:, 0], len(e_map["bond_type"])
        ).to(torch.float)

        ex_dense, node_mask = diff_utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": torch.zeros((2, 1)),
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": len(x_map["atomic_num"]),
            "E": len(e_map["bond_type"]),
            "y": 1,
        }  # + 1 due to time conditioning

        if extra_features is not None:
            ex_extra_feat = extra_features(example_data)
            self.input_dims["X"] += ex_extra_feat.X.size(-1)
            self.input_dims["E"] += ex_extra_feat.E.size(-1)
            self.input_dims["y"] += ex_extra_feat.y.size(-1)

        if domain_features is not None:
            ex_extra_molecular_feat = domain_features(example_data)
            self.input_dims["X"] += ex_extra_molecular_feat.X.size(-1)
            self.input_dims["E"] += ex_extra_molecular_feat.E.size(-1)
            self.input_dims["y"] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            "X": len(x_map["atomic_num"]),
            "E": len(e_map["bond_type"]),
            "y": 0,
        }

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for i, data in enumerate(self.dataset):
            n = data.x.shape[0]
            edge_attr = F.one_hot(data.edge_attr[:, 0], len(e_map["bond_type"])).to(
                torch.float
            )

            for atom in range(n):
                edges = edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


@register_object("ecreact_graph", "dataset")
class ECReactGraph(ECReact_RXNS):
    def init_class(self, args, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.load_dataset(args)

        self.ec2uniprot = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_ec2uniprot.p",
                "rb",
            )
        )
        self.valid_ec2uniprot = {}
        self.uniprot2sequence = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_proteins.p", "rb"
            )
        )
        self.uniprot2cluster = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_mmseq_clusters.p",
                "rb",
            )
        )

    def post_process(self, args):
        split_group = self.split_group

        # get data info (input / output dimensions)
        train_dataset = self.get_split_group_dataset(self.dataset, "train")

        smiles = set()
        for d in train_dataset:
            smiles.update([".".join(d["reactants"]), ".".join(d["products"])])
        smiles = list(smiles)

        data_info = DatasetInfo(smiles, args)

        extra_features = ExtraFeatures(args.extra_features_type, dataset_info=data_info)

        example_batch = [from_smiles(smiles[0]), from_smiles(smiles[1])]
        example_batch = Batch.from_data_list(example_batch, None, None)

        data_info.compute_input_output_dims(
            example_batch=example_batch,
            extra_features=extra_features,
            domain_features=None,
        )

        args.dataset_statistics = data_info
        args.extra_features = extra_features
        args.domain_features = None

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # check right split
        parse_ec = lambda ec: ".".join(ec.split(".")[: self.args.ec_level + 1])
        dataset = []
        if self.args.assign_splits:
            if self.args.split_type in ["mmseqs", "sequence", "ec", "product"]:
                for sample in processed_dataset:
                    if self.args.split_type == "sequence":
                        if self.to_split[sample["protein_id"]] != split_group:
                            continue
                    elif self.args.split_type == "mmseqs":
                        cluster = self.uniprot2cluster[sample["protein_id"]]
                        if self.to_split[cluster] != split_group:
                            continue
                    elif self.args.split_type == "ec":
                        ec = parse_ec(sample["ec"])
                        if self.to_split[ec] != split_group:
                            continue
                    elif self.args.split_type == "product":
                        if self.to_split[sample["products"][0]] != split_group:
                            continue
                    dataset.append(sample)

            elif self.args.split_type == "random":
                for sample in processed_dataset:
                    reaction_string = sample["reaction_string"]
                    if self.to_split[reaction_string] != split_group:
                        continue
                dataset.append(sample)

        else:
            dataset = [d for d in processed_dataset if d["split"] == split_group]

        return dataset

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            total=len(self.metadata_json),
            desc="Building dataset",
        ):

            ec = reaction["ec"]
            reactants = reaction["reactants"]
            products = reaction["products"]
            reaction_string = ".".join(reactants) + ">>" + ".".join(products)

            valid_uniprots = []
            for uniprot in self.ec2uniprot.get(ec, []):
                temp_sample = {
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "reaction_string": reaction_string,
                    "protein_id": uniprot,
                    "sequence": self.uniprot2sequence[uniprot],
                    "split": reaction.get("split", None),
                }
                if self.skip_sample(temp_sample, split_group):
                    continue

                valid_uniprots.append(uniprot)

            if len(valid_uniprots) == 0:
                continue

            if ec not in self.valid_ec2uniprot:
                self.valid_ec2uniprot[ec] = valid_uniprots

            sample = {
                "reactants": reactants,
                "products": products,
                "ec": ec,
                "reaction_string": reaction_string,
                "rowid": rowid,
                "split": reaction["split"],
            }

            # add reaction sample to dataset
            dataset.append(sample)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:

        # if sequence is unknown
        if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            ec = sample["ec"]
            valid_uniprots = self.valid_ec2uniprot[ec]
            uniprot_id = random.sample(valid_uniprots, 1)[0]
            sequence = self.uniprot2sequence[uniprot_id]

            residue_dict = self.get_uniprot_residues(self.mcsa_data, sequence, ec)
            residues = residue_dict["residues"]
            residue_mask = residue_dict["residue_mask"]
            has_residues = residue_dict["has_residues"]
            residue_positions = residue_dict["residue_positions"]

            # incorporate sequence residues if known
            if self.args.use_residues_in_reaction:
                reactants.extend(residues)
                # products.extend(residues)

            reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
            # randomize order of reactants and products
            if self.args.randomize_order_in_reaction:
                np.random.shuffle(reactants)
                np.random.shuffle(products)
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            if self.args.use_random_smiles_representation:
                try:
                    reactants = [randomize_smiles_rotated(s) for s in reactants]
                    products = [randomize_smiles_rotated(s) for s in products]
                    reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
                except:
                    pass

            reactants = from_smiles(".".join(reactants))
            products = from_smiles(".".join(products))

            # first feature is atomic number
            reactants.x = F.one_hot(reactants.x[:, 0], len(x_map["atomic_num"])).to(
                torch.float
            )
            # first feature is bond type
            reactants.edge_attr = F.one_hot(
                reactants.edge_attr[:, 0], len(e_map["bond_type"])
            ).to(torch.float)

            # first feature is atomic number
            products.x = F.one_hot(products.x[:, 0], len(x_map["atomic_num"])).to(
                torch.float
            )
            # first feature is bond type
            products.edge_attr = F.one_hot(
                products.edge_attr[:, 0], len(e_map["bond_type"])
            ).to(torch.float)
            products.y = torch.zeros((1, 0), dtype=torch.float)

            sample_id = sample["rowid"]
            item = {
                "reaction": reaction,
                "reactants": reactants,
                "products": products,
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "sample_id": sample_id,
            }

            if self.args.precomputed_esm_features_dir is not None:
                esm_features = pickle.load(
                    open(
                        os.path.join(
                            self.args.precomputed_esm_features_dir,
                            f"sample_{uniprot_id}.predictions",
                        ),
                        "rb",
                    )
                )

                mask_hiddens = esm_features["mask_hiddens"]  # sequence len, 1
                protein_hidden = esm_features["hidden"]
                token_hiddens = esm_features["token_hiddens"][mask_hiddens[:, 0].bool()]
                item.update(
                    {
                        # "token_hiddens": token_hiddens,
                        "protein_len": mask_hiddens.sum(),
                        "hidden": protein_hidden,
                    }
                )

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")

    @property
    def SUMMARY_STATEMENT(self) -> None:
        reactions = [d["reaction_string"] for d in self.dataset]
        proteins = [u for d in self.dataset for u in self.valid_ec2uniprot[d["ec"]]]
        ecs = [d["ec"] for d in self.dataset]
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ECReactGraph, ECReactGraph).add_args(parser)
        parser.add_argument(
            "--remove_h",
            action="store_true",
            default=False,
            help="remove hydrogens from the molecules",
        )
        parser.add_argument(
            "--extra_features_type",
            type=str,
            choices=["eigenvalues", "all", "cycles"],
            default=None,
            help="extra features to use",
        )
        parser.add_argument(
            "--protein_feature_dim",
            type=int,
            default=480,
            help="size of protein residue features from ESM models",
        )


@register_object("ecreact_multiproduct_graph", "dataset")
class ECReact_MultiProduct_Graph(ECReact_MultiProduct_RXNS):
    @staticmethod
    def set_args(args):
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_multiproduct.json"
        )


@register_object("ecreact_substrates", "dataset")
class ECReactSubstrate(ECReactGraph):
    def post_process(self, args):
        split_group = self.split_group

        # get data info (input / output dimensions)
        train_dataset = [d for d in self.dataset if d["split"] == "train"]

        smiles = set()
        for d in train_dataset:
            smiles.add(d["smiles"])
        smiles = list(smiles)

        data_info = DatasetInfo(smiles, args)

        if args.extra_features_type is not None:
            extra_features = ExtraFeatures(
                args.extra_features_type, dataset_info=data_info
            )
        else:
            extra_features = None

        example_batch = [from_smiles(smiles[0]), from_smiles(smiles[1])]
        example_batch = Batch.from_data_list(example_batch, None, None)

        data_info.compute_input_output_dims(
            example_batch=example_batch,
            extra_features=extra_features,
            domain_features=None,
        )
        data_info.input_dims["y"] += args.protein_feature_dim

        args.dataset_statistics = data_info
        args.extra_features = extra_features
        args.domain_features = None
        if not args.use_original_num_classes:
            args.num_classes = data_info.max_n_nodes

        if args.sample_negatives:
            self.dataset = self.add_negatives(self.dataset, split_group=split_group)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        # TODO remove when I finish debugging
        # cache_path = f"/Mounts/rbg-storage1/datasets/Metabo/ecreact_{split_group}_dataset_temp.pt"
        # if os.path.exists(cache_path):
        #     print(f"Loaded dataset from cache: {cache_path}")
        #     dataset_dict = torch.load(cache_path)
        #     dataset = dataset_dict["dataset"]
        #     self.valid_ec2uniprot = dataset_dict["valid_ec2uniprot"]
        #     self.uniprot2substrates = dataset_dict["uniprot2substrates"]
        #     self.common_substrates = dataset_dict["common_substrates"] if self.args.topk_substrates_to_remove is not None else None
        #     return dataset

        reaction_side_key = f"{self.args.reaction_side}s"

        dataset = []
        uniprot_substrates = set()
        self.uniprot2substrates = defaultdict(set)

        if self.args.topk_substrates_to_remove is not None:
            substrates = Counter(
                [r for d in self.metadata_json for r in d[reaction_side_key]]
            ).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = [s[0] for s in substrates]

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            total=len(self.metadata_json),
            desc="Building dataset",
        ):

            ec = reaction["ec"]
            reactants = reaction["reactants"]
            products = reaction["products"]
            target_molecules = reaction[reaction_side_key]
            reaction_string = ".".join(reactants) + ">>" + ".".join(products)

            valid_uniprots = []
            for uniprot in self.ec2uniprot.get(ec, []):
                temp_sample = {
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "reaction_string": reaction_string,
                    "protein_id": uniprot,
                    "sequence": self.uniprot2sequence[uniprot],
                    "split": reaction.get("split", None),
                }
                if super().skip_sample(temp_sample, split_group):
                    continue

                valid_uniprots.append(uniprot)

            if len(valid_uniprots) == 0:
                continue

            if ec not in self.valid_ec2uniprot:
                self.valid_ec2uniprot[ec] = valid_uniprots

            for rid, smiles in enumerate(target_molecules):
                if self.args.use_all_proteins:
                    for valid_uni in valid_uniprots:
                        if f"{valid_uni}_{smiles}" in uniprot_substrates:
                            continue
                        else:
                            uniprot_substrates.add(f"{valid_uni}_{smiles}")
                            self.uniprot2substrates[valid_uni].add(smiles)

                        sample_to_check = {
                            "smiles": smiles,
                            "ec": ec,
                            "reaction_string": reaction_string,
                            "rowid": f"{rid}{rowid}",
                            "split": reaction["split"],
                            "y": 1,
                            "sequence": self.uniprot2sequence[valid_uni],
                            "uniprot_id": valid_uni,
                            "protein_id": valid_uni,
                        }
                        if self.skip_sample(sample_to_check, split_group):
                            continue
                        dataset.append(sample_to_check)
                else:
                    dataset.append(
                        {
                            "smiles": smiles,
                            "ec": ec,
                            "reaction_string": reaction_string,
                            "rowid": f"{rid}{rowid}",
                            "split": reaction["split"],
                            "y": 1,
                        }
                    )
        
        # TODO remove when I finish debugging
        # print(f"Saving dataset to cache: {cache_path}")
        # torch.save({
        #     "dataset": dataset,
        #     "valid_ec2uniprot": self.valid_ec2uniprot,
        #     "uniprot2substrates": self.uniprot2substrates,
        #     "common_substrates": self.common_substrates if self.args.topk_substrates_to_remove is not None else None,
        #     }, cache_path)

        return dataset
        

    def skip_sample(self, sample, split_group) -> bool:
        if super().skip_sample(sample, split_group):
            return True

        # underspecified EC number
        if "ec" in sample and "-" in sample["ec"]:
            return True

        # skip graphs of size 1
        mol_size = rdkit.Chem.MolFromSmiles(sample["smiles"]).GetNumAtoms()
        if mol_size <= 1:
            return True

        if self.args.reaction_side == "reactant":
            if (self.args.max_reactant_size is not None) and (
                mol_size > self.args.max_reactant_size
            ):
                return True
        elif self.args.reaction_side == "product":
            if (self.args.max_product_size is not None) and (
                mol_size > self.args.max_product_size
            ):
                return True

        if self.args.topk_substrates_to_remove is not None:
            if sample["smiles"] in self.common_substrates:
                return True

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            mol = from_smiles(sample["smiles"])

            ec = sample["ec"]
            if self.args.use_all_proteins:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                # in the case where we are not using all proteins, we need to sample a valid sequence
                # so we check if the sequence is valid, if not we sample a new one
                # if we cant find a new one then we skip this sample in batch
                valid_seq = True
                maxiters = 20
                iters = 0
                while valid_seq:
                    if iters > maxiters:
                        print("Could not find a valid sequence for this EC number")
                        return None  # will just skip this sample in the batch
                    valid_uniprots = self.valid_ec2uniprot[ec]
                    uniprot_id = random.sample(valid_uniprots, 1)[0]
                    sequence = self.uniprot2sequence[uniprot_id]
                    sample_to_check = sample
                    sample_to_check["sequence"] = sequence
                    sample_to_check["protein_id"] = uniprot_id
                    valid_seq = self.skip_sample(sample_to_check, self.split_group)
                    iters += 1

            # first feature is atomic number
            mol.x = F.one_hot(mol.x[:, 0], len(x_map["atomic_num"])).to(torch.float)
            # first feature is bond type
            mol.edge_attr = F.one_hot(mol.edge_attr[:, 0], len(e_map["bond_type"])).to(
                torch.float
            )

            mol.sample_id = sample["rowid"]
            mol.sequence = sequence

            if self.args.precomputed_esm_features_dir is not None:
                esm_features = pickle.load(
                    open(
                        os.path.join(
                            self.args.precomputed_esm_features_dir,
                            f"sample_{uniprot_id}.predictions",
                        ),
                        "rb",
                    )
                )

                mask_hiddens = esm_features["mask_hiddens"]  # sequence len, 1
                protein_hidden = esm_features["hidden"]
                # token_hiddens = esm_features["token_hiddens"][mask_hiddens[:, 0].bool()]

            # token_hiddens = esm_features["token_hiddens"][mask_hiddens[:,0].bool()]
            # mol.y = protein_hidden.view(1, -1)
            mol.y = sample["y"]
            mol.ec = sample["ec"]
            return mol

        except Exception as e:
            warnings.warn(f"Could not load sample: {sample['rowid']} because of {e}")

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ECReactSubstrate, ECReactSubstrate).add_args(parser)
        parser.add_argument(
            "--use_original_num_classes",
            action="store_true",
            default=False,
            help="use max node type as num_classes if false",
        )
        parser.add_argument(
            "--sample_negatives",
            action="store_true",
            default=False,
            help="whether to sample negative substrates",
        )
        parser.add_argument(
            "--sample_negatives_range",
            type=float,
            nargs=2,
            default=None,
            help="range of similarity to sample negatives from",
        )
        parser.add_argument(
            "--use_all_proteins",
            action="store_true",
            default=False,
            help="whether to use all proteins or just sample one for each ec",
        )
        parser.add_argument(
            "--reaction_side",
            type=str,
            default="reactant",
            choices=["reactant", "product"],
            help="choice of reactant or product to use as target",
        )
        parser.add_argument(
            "--negative_samples_cache_dir",
            type=str,
            default=None,
            help="directory to save negative samples",
        )
        parser.add_argument(
            "--sample_k_negatives",
            type=int,
            default=None,
            help="number of negatives to sample from each ec",
        )


    def add_negatives(self, dataset, split_group):
        # TODO remove when I finish debugging
        if self.args.sample_negatives_range is not None:
            min_sim, max_sim = self.args.sample_negatives_range
        if min_sim == 0.6 and max_sim == 1.0:
            cache_path = f"/Mounts/rbg-storage1/datasets/Metabo/ecreact_{split_group}_negatives_temp.pt"
        elif min_sim == 0.85 and max_sim == 0.99:
            cache_path = f"/Mounts/rbg-storage1/datasets/Metabo/ecreact_{split_group}_negatives_temp_point9.pt"
        else:
            cache_path = None
        if cache_path is not None and os.path.exists(cache_path):
            print(f"Loaded negatives from cache: {cache_path}")
            dataset_dict = torch.load(cache_path)
            negatives_to_add = dataset_dict["negatives"]
            dataset += negatives_to_add
            return dataset

        all_substrates = set(d["smiles"] for d in dataset)
        
        all_substrates_list = list(all_substrates)

        # filter out negatives based on some metric (e.g. similarity)
        if self.args.sample_negatives_range is not None:
            min_sim, max_sim = self.args.sample_negatives_range

            smile_fps = np.array(
                [
                    get_rdkit_feature(mol=smile, method="morgan_binary")
                    / np.linalg.norm(
                        get_rdkit_feature(mol=smile, method="morgan_binary")
                    )
                    for smile in all_substrates
                ]
            )

            smile_similarity = smile_fps @ smile_fps.T
            similarity_idx = np.where(
                (smile_similarity <= min_sim) | (smile_similarity >= max_sim)
            )

            smiles2positives = defaultdict(set)
            for smi_i, smile in tqdm(
                enumerate(all_substrates_list), desc="Retrieving all negatives", total=len(all_substrates_list)
            ):
                if smile not in smiles2positives:
                    smiles2positives[smile].update(
                        all_substrates_list[j]
                        for j in similarity_idx[1][similarity_idx[0] == smi_i]
                    )

        ec_to_positives = defaultdict(set)
        for sample in tqdm(dataset, desc="Sampling negatives"):
            ec = sample["ec"]
            ec_to_positives[ec].add(sample["smiles"])

            if self.args.sample_negatives_range is not None:
                ec_to_positives[ec].update(smiles2positives[sample["smiles"]])

        ec_to_negatives = {k: all_substrates - v for k, v in ec_to_positives.items()}
        
        rowid = len(dataset)
        negatives_to_add = []
        no_negatives = 0
        for ec, negatives in tqdm(ec_to_negatives.items(), desc="Processing negatives"):
            if len(negatives) == 0:
                no_negatives += 1
                continue

            if self.args.use_all_proteins:
                # get all the valid uniprots for each EC if use_all_proteins is True
                valid_uniprots = []
                for uniprot in self.ec2uniprot.get(ec, []):
                    # this is all to check that the sequence is valid
                    temp_sample = {
                        "ec": ec,
                        "protein_id": uniprot,
                        "sequence": self.uniprot2sequence[uniprot],
                    }
                    if super().skip_sample(temp_sample, split_group):
                        continue

                    valid_uniprots.append(uniprot)

                if len(valid_uniprots) == 0:
                    continue

                # TODO: generalize to either reaction side, not just reactants (substrates)
                for valid_uni in valid_uniprots:
                    # for each EC include only k negatives
                    if self.args.sample_k_negatives is not None:
                        if len(negatives) < self.args.sample_k_negatives:
                            print(
                                f"Not enough negatives to sample from, using all negatives for {ec}"
                            )
                            new_negatives = list(negatives)
                        else:
                            new_negatives = random.sample(
                                negatives, self.args.sample_k_negatives
                            )
                    else:
                        new_negatives = list(negatives)
                    for rid, reactant in enumerate(new_negatives):
                        sample = {
                            "smiles": reactant,
                            "ec": ec,
                            "uniprot_id": valid_uni,
                            "protein_id": valid_uni,
                            "sequence": self.uniprot2sequence[valid_uni],
                            "reaction_string": "",
                            "rowid": f"{rid}{rowid}",
                            # "split": split_group,
                            "y": 0,
                        }
                        if super().skip_sample(sample, split_group):
                            continue
                        rowid += 1
                        negatives_to_add.append(sample)

            else:
                # for each EC include only k negatives
                if self.args.sample_k_negatives is not None:
                    if len(negatives) < self.args.sample_k_negatives:
                        print(
                            f"Not enough negatives to sample from, using all negatives for {ec}"
                        )
                        new_negatives = list(negatives)
                    else:
                        new_negatives = random.sample(
                            negatives, self.args.sample_k_negatives
                        )
                else:
                    new_negatives = list(negatives)
                for rid, reactant in enumerate(new_negatives):
                    sample = {
                        "smiles": reactant,
                        "ec": ec,
                        "reaction_string": "",
                        "rowid": f"{rid}{rowid}",
                        # "split": split_group,
                        "y": 0,
                    }
                    rowid += 1
                    negatives_to_add.append(sample)

        print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
        print(f"[magenta] Missing any negatives for {no_negatives} ECs [/magenta]")
        print(f"[magenta] Total number of positive samples: {len(dataset)} [/magenta]")
        dataset += negatives_to_add

        print(f"Saving negatives to cache: {cache_path}")
        if cache_path is not None:
            torch.save({"negatives": negatives_to_add}, cache_path)
        return dataset


@register_object("ecreact_substrates_plain_graph", "dataset")
class ECReactSubstratePlainGraph(ECReactSubstrate):
    @classproperty
    def DATASET_ITEM_KEYS(cls) -> list:
        """
        List of keys to be included in sample when being batched

        Returns:
            list
        """
        standard = [
            "sample_id",
            "protein_hiddens",
            "substrate_hiddens",
            "sequence",
            "smiles",
        ]
        return standard

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactant = from_smiles(sample["smiles"])

            ec = sample["ec"]
            if self.args.use_all_proteins:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                # in the case where we are not using all proteins, we need to sample a valid sequence
                # so we check if the sequence is valid, if not we sample a new one
                # if we cant find a new one then we skip this sample in batch
                valid_seq = True
                maxiters = 20
                iters = 0
                while valid_seq:
                    if iters > maxiters:
                        print("Could not find a valid sequence for this EC number")
                        return None  # will just skip this sample in the batch
                    valid_uniprots = self.valid_ec2uniprot[ec]
                    uniprot_id = random.sample(valid_uniprots, 1)[0]
                    sequence = self.uniprot2sequence[uniprot_id]
                    sample_to_check = sample
                    sample_to_check["sequence"] = sequence
                    sample_to_check["protein_id"] = uniprot_id
                    valid_seq = self.skip_sample(sample_to_check, self.split_group)
                    iters += 1

            reactant.sample_id = sample["rowid"]
            reactant.sequence = sequence

            # esm_features = pickle.load(
            #     open(
            #         os.path.join(
            #             self.args.precomputed_esm_features_dir,
            #             f"sample_{uniprot_id}.predictions",
            #         ),
            #         "rb",
            #     )
            # )

            # mask_hiddens = esm_features["mask_hiddens"]  # sequence len, 1
            # protein_hidden = esm_features["hidden"]

            reactant.y = sample["y"]
            reactant.ec = sample["ec"]
            reactant.all_reactants = sample["reaction_string"].split(">>")[0].split(".")
            reactant.all_smiles = self.uniprot2substrates[uniprot_id]
            reactant.uniprot_id = uniprot_id

            return reactant

        except Exception:
            warnings.warn(f"Could not load sample: {sample['rowid']}")


@register_object("ecreact_protmol_graph", "dataset")
class ECReactProtMolGraph(ECReactSubstrate):
    def __init__(self, args, split_group):
        super(ECReactProtMolGraph, ECReactProtMolGraph).__init__(self, args, split_group)
        esm_dir = "/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"
        self.esm_dir = esm_dir
        model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    @classproperty
    def DATASET_ITEM_KEYS(cls) -> list:
        """
        List of keys to be included in sample when being batched

        Returns:
            list
        """
        standard = [
            "sample_id",
            "protein_hiddens",
            "substrate_hiddens",
            "sequence",
            "smiles",
        ]
        return standard

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            # reactant = from_smiles(sample["smiles"])

            ec = sample["ec"]
            if self.args.use_all_proteins:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                # in the case where we are not using all proteins, we need to sample a valid sequence
                # so we check if the sequence is valid, if not we sample a new one
                # if we cant find a new one then we skip this sample in batch
                valid_seq = True
                maxiters = 20
                iters = 0
                while valid_seq:
                    if iters > maxiters:
                        print("Could not find a valid sequence for this EC number")
                        return None  # will just skip this sample in the batch
                    valid_uniprots = self.valid_ec2uniprot[ec]
                    uniprot_id = random.sample(valid_uniprots, 1)[0]
                    sequence = self.uniprot2sequence[uniprot_id]
                    sample_to_check = sample
                    sample_to_check["sequence"] = sequence
                    sample_to_check["protein_id"] = uniprot_id
                    valid_seq = self.skip_sample(sample_to_check, self.split_group)
                    iters += 1

            # load the protein graph
            graph_path = os.path.join(self.args.protein_graphs_dir, "processed", f"{sample['uniprot_id']}_graph.pt")
            structures_dir = os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")
            complex_path = os.path.join(self.args.protein_graphs_dir, "processed_complexes", f"{sample['uniprot_id']}_{hashlib.md5(sample['smiles'].encode()).hexdigest()}_graph.pt")

            if not os.path.exists(structures_dir):
                print(f"Could not find structure for {sample['uniprot_id']} because missing structure path")
                return None

            if os.path.exists(complex_path):
                data = torch.load(complex_path)
                return data

            if not os.path.exists(graph_path):
                data = self.create_protein_graph(sample)
                torch.save(data, graph_path)
            else:
                data = torch.load(graph_path)

            data = self.add_additional_data_to_graph(data, sample)
            # TODO: remove in the future
            if not hasattr(data, "structure_sequence"):
                protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                AA_seq = ""
                for char in data['receptor'].seq:
                    AA_seq += protein_letters_3to1[char]
                data.structure_sequence = AA_seq
            if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                data["receptor"].x = data.x
            if hasattr(data, "x"):
                delattr(data, "x")
            if hasattr(data, "embedding_path"):
                delattr(data, "embedding_path")
            if hasattr(data, "protein_path"):
                delattr(data, "protein_path")
            if hasattr(data, "sample_hash"):
                delattr(data, "sample_hash")

            # torch.save(data, complex_path)
            return data

        except Exception as e:
            warnings.warn(f"Could not load sample: {sample['uniprot_id']} due to error {e}")

    def add_additional_data_to_graph(self, data, sample):
        skipped_keys = set(["protein_path", "embedding_path"])
        for key in sample.keys():
            if not key in skipped_keys and key not in data.to_dict().keys():
                data[key] = sample[key]
        data["mol_data"] = from_smiles(sample["smiles"])
        return data

    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")
            sample_id = sample["uniprot_id"]
            protein_parser = Bio.PDB.MMCIFParser()
            protein_resolution = "residue"
            graph_edge_args = {"knn_size": 10}
            center_protein = True
            esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"

            # parse pdb
            all_res, all_atom, all_pos = read_structure_file(
                protein_parser, raw_path, sample_id
            )
            # filter resolution of protein (backbone, atomic, etc.)
            atom_names, seq, pos = filter_resolution(
                all_res,
                all_atom,
                all_pos,
                protein_resolution=protein_resolution,
            )
            # generate graph
            data = build_graph(atom_names, seq, pos, sample_id)
            # kNN graph
            data = compute_graph_edges(data, **graph_edge_args)
            if center_protein:
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                data["receptor"].pos = data["receptor"].pos - center
                data.center = center
            data.structure_sequence = sample['sequence']
            node_embeddings_args = {"model": self.esm_model, "model_location": self.esm_dir, "alphabet": self.alphabet, "batch_converter": self.batch_converter}

            embedding_path = os.path.join(self.args.protein_graphs_dir, "precomputed_node_embeddings", f"{sample['uniprot_id']}.pt")
            if os.path.exists(embedding_path):
                node_embedding = torch.load(
                    sample["embedding_path"]
                )
            else:
                node_embedding = compute_node_embedding(
                    data, **node_embeddings_args
                )
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]
                # sequences = get_sequences(
                #     self.protein_parser,
                #     [sample["sample_id"]],
                #     [os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")],
                # )
                
                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding
            
            return data

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of exception {e}")
            return None

    @property
    def SUMMARY_STATEMENT(self) -> None:
        statement = f""" 
        * Len of dataset: {len(self.dataset)}
        """
        return statement

    def set_sample_weights(self, args: argparse.ArgumentParser) -> None:
        """
        Set weights for each sample

        Args:
            args (argparse.ArgumentParser)
        """
        if args.class_bal:
            # label_counts = []
            # for sample in self.metadata_json:
            #     cluster = self.uniprot2cluster[sample["uniprot_id"]]
            #     if self.to_split[cluster] == split_group:
            #         label_counts.append(sample[args.class_bal_key])

            label_dist = [d[args.class_bal_key] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]
        else:
            pass

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ECReactProtMolGraph, ECReactProtMolGraph).add_args(parser)
        parser.add_argument(
            "--protein_graphs_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--protein_structures_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )

    def skip_sample(self, sample, split_group) -> bool:
        if super().skip_sample(sample, split_group):
            return True

        if not os.path.exists(os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")):
            return True

        return False

@register_object("ecreact_iocb", "dataset")
class ECReactIOCB(ECReactSubstrate):
    def __init__(self, args, split_group):
        super(ECReactIOCB, ECReactIOCB).__init__(self, args, split_group)
        if args.use_protein_graphs:
            esm_dir = "/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"
            self.esm_dir = esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()

    @classproperty
    def DATASET_ITEM_KEYS(cls) -> list:
        """
        List of keys to be included in sample when being batched

        Returns:
            list
        """
        standard = [
            "sample_id",
            "protein_hiddens",
            "substrate_hiddens",
            "sequence",
            "smiles",
        ]
        return standard

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        if self.args.reaction_side == "reactant": 
            # get column name of split
            if self.args.split_type != "pretrained_uniprots":
                split_key = "substrate_stratified_phylogeny_based_split" 
            else:
                split_key = "Uniprot ID"
            # get column names for smiles
            smiles_key = "SMILES_substrate_canonical_no_stereo" if self.args.use_stereo else "SMILES of substrate"
            smiles_id_key = "Substrate ChEBI ID"
            smiles_name_key = "Substrate (including stereochemistry)"

        elif self.args.reaction_side == "product":
            # get column name of split
            if self.args.split_type != "pretrained_uniprots":
                split_key = "product_stratified_phylogeny_based_split" 
            else:
                split_key = "Uniprot ID"
            # get column names for smiles
            smiles_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 
            smiles_id_key = "Product ChEBI ID"
            smiles_name_key = "Name of product"

        self.substrate_count = Counter([s[smiles_key] for s in self.metadata_json])

        # if removing top K
        if self.args.topk_substrates_to_remove is not None:
            substrates = Counter([r for d in self.metadata_json for r in d[smiles_key]]).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = [s[0] for s in substrates]

        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprotid = row["Uniprot ID"]
            molname = row[smiles_name_key]
            hashed_molname = hashlib.md5(molname.encode()).hexdigest()
            if self.args.use_protein_graphs:
                sample = {
                    "protein_id": uniprotid,
                    "uniprot_id": uniprotid,
                    "sequence": self.clean_seq(row["Amino acid sequence"]),
                    # "sequence_name": row["Name"],
                    # "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "smiles": row[smiles_key],
                    # "smiles_name": row[smiles_name_key],
                    # "smiles_chebi_id": row[smiles_id_key],
                    "sample_id": f"{uniprotid}_{hashed_molname}",
                    # "species": row["Species"],
                    # "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row[split_key],
                    "y": 1
                }
            else:
                sample = {
                        "protein_id": uniprotid,
                        "uniprot_id": uniprotid,
                        "sequence": self.clean_seq(row["Amino acid sequence"]),
                        "sequence_name": row["Name"],
                        "protein_type": row["Type (mono, sesq, di, \u2026)"],
                        "smiles": row[smiles_key],
                        "smiles_name": row[smiles_name_key],
                        "smiles_chebi_id": row[smiles_id_key],
                        "sample_id": f"{uniprotid}_{hashed_molname}",
                        "species": row["Species"],
                        "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                        "split": row[split_key],
                        "y": 1
                    }
            
            if self.skip_sample(sample, self.split_group):
                continue 
            
            dataset.append(sample)

        return dataset 
    
    def clean_seq(self, seq):
        seq = seq.replace(" ", "")
        seq = seq.replace("\n", "")
        seq = seq.replace("*", "")
        return seq

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            if self.args.use_protein_graphs:
                uniprot_id = sample["protein_id"]
                sequence = sample["sequence"]

                # load the protein graph
                graph_path = os.path.join(self.args.protein_graphs_dir, "processed", f"{uniprot_id}_graph.pt")
                structures_dir = os.path.join(self.args.protein_structures_dir, f"AF-{uniprot_id}-F1-model_v4.cif")
                complex_path = os.path.join(self.args.protein_graphs_dir, "processed_complexes", f"{uniprot_id}_{hashlib.md5(sample['smiles'].encode()).hexdigest()}_graph.pt")

                if not os.path.exists(structures_dir):
                    print(f"Could not find structure for {uniprot_id} because missing structure path")
                    return None

                if os.path.exists(complex_path):
                    data = torch.load(complex_path)
                    return data

                if not os.path.exists(graph_path):
                    data = self.create_protein_graph(sample)
                    torch.save(data, graph_path)
                else:
                    data = torch.load(graph_path)
                    if data is None:
                        data = self.create_protein_graph(sample)
                        torch.save(data, graph_path)

                data = self.add_additional_data_to_graph(data, sample)
                # TODO: remove in the future
                if not hasattr(data, "structure_sequence"):
                    protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                    AA_seq = ""
                    for char in data['receptor'].seq:
                        AA_seq += protein_letters_3to1[char]
                    data.structure_sequence = AA_seq
                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x
                if hasattr(data, "x"):
                    delattr(data, "x")
                if hasattr(data, "ec"):
                    delattr(data, "ec")
                if hasattr(data, "embedding_path"):
                    delattr(data, "embedding_path")
                if hasattr(data, "protein_path"):
                    delattr(data, "protein_path")
                if hasattr(data, "sample_hash"):
                    delattr(data, "sample_hash")
                if hasattr(data, "sequence_name"):
                    delattr(data, "sequence_name")

                # torch.save(data, complex_path)
                return data
            else:
                reactant = from_smiles(sample["smiles"])

                reactant.sample_id = sample["sample_id"]
                reactant.sequence = sample["sequence"]
                reactant.y = sample["y"]
                reactant.uniprot_id = sample["protein_id"]

                return reactant

        except Exception as e:
            warnings.warn(f"Could not load sample: {sample['protein_id']} due to error {e}")

    def post_process(self, args):
        if args.sample_negatives:
            self.dataset = self.add_negatives(self.dataset, split_group=self.split_group)

    def add_additional_data_to_graph(self, data, sample):
        skipped_keys = set(["protein_path", "embedding_path"])
        for key in sample.keys():
            if not key in skipped_keys and key not in data.to_dict().keys():
                data[key] = sample[key]
        data["mol_data"] = from_smiles(sample["smiles"])
        return data

    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")
            sample_id = sample["uniprot_id"]
            protein_parser = Bio.PDB.MMCIFParser()
            protein_resolution = "residue"
            graph_edge_args = {"knn_size": 10}
            center_protein = True
            esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"

            # parse pdb
            all_res, all_atom, all_pos = read_structure_file(
                protein_parser, raw_path, sample_id
            )
            # filter resolution of protein (backbone, atomic, etc.)
            atom_names, seq, pos = filter_resolution(
                all_res,
                all_atom,
                all_pos,
                protein_resolution=protein_resolution,
            )
            # generate graph
            data = build_graph(atom_names, seq, pos, sample_id)
            # kNN graph
            data = compute_graph_edges(data, **graph_edge_args)
            if center_protein:
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                data["receptor"].pos = data["receptor"].pos - center
                data.center = center
            data.structure_sequence = sample['sequence']
            node_embeddings_args = {"model": self.esm_model, "model_location": self.esm_dir, "alphabet": self.alphabet, "batch_converter": self.batch_converter}

            embedding_path = os.path.join(self.args.protein_graphs_dir, "precomputed_node_embeddings", f"{sample['uniprot_id']}.pt")
            if os.path.exists(embedding_path):
                node_embedding = torch.load(
                    sample["embedding_path"]
                )
            else:
                node_embedding = compute_node_embedding(
                    data, **node_embeddings_args
                )
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]
                # sequences = get_sequences(
                #     self.protein_parser,
                #     [sample["sample_id"]],
                #     [os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")],
                # )
                
                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, **node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding
            
            return data

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of exception {e}")
            return None

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []
        for sample in processed_dataset:
            if self.to_split[sample['split']] != split_group:
                continue

            dataset.append(sample)

        return dataset

    def assign_splits(self, dataset, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # get all samples
        print("Generating dataset in order to assign splits...")

        if self.args.reaction_side == "reactant": 
            split_key = "substrate_stratified_phylogeny_based_split" 
        elif self.args.reaction_side == "product":
            split_key = "product_stratified_phylogeny_based_split" 

        # set seed
        np.random.seed(seed)

        # assign groups
        if self.args.split_type == "fold":
            samples = sorted(list(set(row[split_key] for row in metadata_json)))
            np.random.shuffle(samples)
            split_indices = np.ceil(
                np.cumsum(np.array(split_probs) * len(samples))
            ).astype(int)
            split_indices = np.concatenate([[0], split_indices])

            for i in range(len(split_indices) - 1):
                self.to_split.update(
                    {
                        sample: ["train", "dev", "test"][i]
                        for sample in samples[split_indices[i] : split_indices[i + 1]]
                    }
                )

        elif self.args.split_type == "fixed":
            self.to_split = {
                "fold_0": "test",
                "fold_1": "dev",
                "fold_2": "train",
                "fold_3": "train",
                "fold_4": "train",
            }
        elif self.args.split_type == "pretrained_uniprots":
            ec2uniprot = pickle.load(
                open(
                    "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_ec2uniprot.p",
                    "rb",
                )
            )
            uniprots = set(prot for l_prots in ec2uniprot.values() for prot in l_prots)
            iocb_trained_on = list(set([u['protein_id'] for u in dataset if u['protein_id'] in uniprots]))
            iocb_unique_uniprots = list(set([u['protein_id'] for u in dataset if not u['protein_id'] in uniprots]))
            # shuffle iocb_unique_uniprots and then split into two
            np.random.shuffle(iocb_unique_uniprots)
            iocb_dev = iocb_unique_uniprots[:int(len(iocb_unique_uniprots)/2)]
            iocb_test = iocb_unique_uniprots[int(len(iocb_unique_uniprots)/2):]
            self.to_split = {
                uniprot: "train" for uniprot in iocb_trained_on
            }
            self.to_split.update({
                uniprot: "dev" for uniprot in iocb_dev
            })
            self.to_split.update({
                uniprot: "test" for uniprot in iocb_test
            })
        else:
            raise ValueError(f"Split {self.args.split_type} type not supported. Must be one of [fold, fixed]")

    @property
    def SUMMARY_STATEMENT(self) -> None:
        # TODO: change if doing protein graph because too slow
        proteins = [d["protein_id"] for d in self.dataset]
        substrates = [d["smiles"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of substrates: {len(set(substrates))}
        * Number of proteins: {len(set(proteins))}
        """
        return statement

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ECReactIOCB, ECReactIOCB).add_args(parser)
        parser.add_argument(
            "--use_protein_graphs",
            action="store_true",
            default=False,
            help="whether to use and generate protein graphs",
        )
        parser.add_argument(
            "--protein_graphs_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--protein_structures_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        #####
        parser.add_argument(
            "--use_stereo",
            action="store_true",
            default=False,
            help="use stereochemistry version of smiles",
        )
        parser.add_argument(
            "--min_substrate_count",
            type=int,
            default=0,
            help="minimum number of times a substrate must appear in the dataset",
        )

    def skip_sample(self, sample, split_group) -> bool:
        if self.substrate_count[sample['smiles']] <= self.args.min_substrate_count:
            return True

        if sample['sequence'] == "to be added" or sample['sequence'] == "tobeadded":
            return True

        if sample['smiles'] == 'Unknown':
            return True

        if super().skip_sample(sample, split_group):
            return True

        if self.args.use_protein_graphs:
            if not os.path.exists(os.path.join(self.args.protein_structures_dir, f"AF-{sample['protein_id']}-F1-model_v4.cif")):
                return True

        # IOCB
        # if sequence is unknown
        if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        try:
            mol_size = rdkit.Chem.MolFromSmiles(sample["smiles"]).GetNumAtoms()
        except:
            return True 

        # skip graphs of size 1
        if mol_size <= 1:
            return True

        if self.args.reaction_side == "reactant":
            if (self.args.max_reactant_size is not None) and (
                mol_size > self.args.max_reactant_size
            ):
                return True

        elif self.args.reaction_side == "product":
            if (self.args.max_product_size is not None) and (
                mol_size > self.args.max_product_size
            ):
                return True

        if self.args.topk_substrates_to_remove is not None:
            if sample["smiles"] in self.common_substrates:
                return True

        return False

    @staticmethod
    def set_args(args) -> None:
        super(ECReactIOCB, ECReactIOCB).set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/IOCB/IOCB_TPS-1April2023_verified_non_minor_tps_with_neg.json"
        )

    def add_negatives(self, dataset, split_group):
        # uniprot2sequence = { d['protein_id']: d['sequence'] for d in dataset }
        uniprot2sequence = {}
        uniprot2split = {}
        for d in dataset:
            if d['protein_id'] not in uniprot2sequence:
                uniprot2sequence[d['protein_id']] = d['sequence']
                uniprot2split[d['protein_id']] = d['split']
                
        all_substrates = set(d["smiles"] for d in dataset)
        all_substrates_list = list(all_substrates)

        # filter out negatives based on some metric (e.g. similarity)
        if self.args.sample_negatives_range is not None:
            min_sim, max_sim = self.args.sample_negatives_range

            smile_fps = np.array(
                [
                    get_rdkit_feature(mol=smile, method="morgan_binary")
                    / np.linalg.norm(
                        get_rdkit_feature(mol=smile, method="morgan_binary")
                    )
                    for smile in all_substrates
                ]
            )

            smile_similarity = smile_fps @ smile_fps.T
            similarity_idx = np.where(
                (smile_similarity <= min_sim) | (smile_similarity >= max_sim)
            )

            smiles2positives = defaultdict(set)
            for smi_i, smile in tqdm(
                enumerate(all_substrates_list), desc="Retrieving all negatives", total=len(all_substrates_list)
            ):
                if smile not in smiles2positives:
                    smiles2positives[smile].update(
                        all_substrates_list[j]
                        for j in similarity_idx[1][similarity_idx[0] == smi_i]
                    )

        prot_id_to_positives = defaultdict(set)
        for sample in tqdm(dataset, desc="Sampling negatives"):
            prot_id = sample["protein_id"]
            prot_id_to_positives[prot_id].add(sample["smiles"])

            if self.args.sample_negatives_range is not None:
                prot_id_to_positives[prot_id].update(smiles2positives[sample["smiles"]])

        prot_id_to_negatives = {k: all_substrates - v for k, v in prot_id_to_positives.items()}
        
        rowid = len(dataset)
        negatives_to_add = []
        no_negatives = 0
        for prot_id, negatives in tqdm(prot_id_to_negatives.items(), desc="Processing negatives"):
            if len(negatives) == 0:
                no_negatives += 1
                continue

            for rid, reactant in enumerate(negatives):
                hashed_molname = hashlib.md5(reactant.encode()).hexdigest()
                sample = {
                    "smiles": reactant,
                    "uniprot_id": prot_id,
                    "protein_id": prot_id,
                    "sequence": uniprot2sequence[prot_id],
                    "split": uniprot2split[prot_id],
                    "sample_id": f"{prot_id}_{hashed_molname}",
                    "y": 0,
                }
                if super().skip_sample(sample, split_group):
                    continue

                negatives_to_add.append(sample)

        print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
        print(f"[magenta] Missing any negatives for {no_negatives} ECs [/magenta]")
        print(f"[magenta] Total number of positive samples: {len(dataset)} [/magenta]")
        dataset += negatives_to_add

        return dataset

