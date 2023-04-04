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
            if self.args.split_type in ["sequence", "ec", "product"]:
                for sample in processed_dataset:
                    if self.args.split_type == "sequence":
                        if self.to_split[sample["protein_id"]] != split_group:
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
                    # reaction_string = (
                    #     ".".join(sample["reactants"])
                    #     + ">>"
                    #     + ".".join(sample["products"])
                    # )
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
        parser.add_argument(
            "--max_reactant_size",
            type=int,
            default=None,
            help="maximum reactant size",
        )
        parser.add_argument(
            "--max_product_size",
            type=int,
            default=None,
            help="maximum product size",
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
            smiles.add(d["reactant"])
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
                if super().skip_sample(temp_sample, split_group):
                    continue

                valid_uniprots.append(uniprot)

            if len(valid_uniprots) == 0:
                continue

            if ec not in self.valid_ec2uniprot:
                self.valid_ec2uniprot[ec] = valid_uniprots

            for rid, reactant in enumerate(reactants):
                sample_to_check = {
                    "reactant": reactant,
                    "products": products,
                    "ec": ec,
                    "reaction_string": reaction_string,
                    "protein_id": uniprot,
                    "sequence": self.uniprot2sequence[uniprot],
                    "split": reaction.get("split", None),
                }
                if self.skip_sample(sample_to_check, split_group):
                    continue

                sample = {
                    "reactant": reactant,
                    "ec": ec,
                    "reaction_string": reaction_string,
                    "rowid": f"{rid}{rowid}",
                    "split": reaction["split"],
                    "y": 1,
                }

                dataset.append(sample)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if super().skip_sample(sample, split_group):
            return True

        # skip graphs of size 1
        reactant_size = rdkit.Chem.MolFromSmiles(sample["reactant"]).GetNumAtoms()
        if reactant_size <= 1:
            return True

        if (self.args.max_reactant_size is not None) and (
            reactant_size > self.args.max_reactant_size
        ):
            return True

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactant = from_smiles(sample["reactant"])

            ec = sample["ec"]
            valid_uniprots = self.valid_ec2uniprot[ec]
            uniprot_id = random.sample(valid_uniprots, 1)[0]
            sequence = self.uniprot2sequence[uniprot_id]

            # residue_dict = self.get_uniprot_residues(self.mcsa_data, sequence, ec)
            # residues = residue_dict["residues"]
            # residue_mask = residue_dict["residue_mask"]
            # has_residues = residue_dict["has_residues"]
            # residue_positions = residue_dict["residue_positions"]

            # first feature is atomic number
            reactant.x = F.one_hot(reactant.x[:, 0], len(x_map["atomic_num"])).to(
                torch.float
            )
            # first feature is bond type
            reactant.edge_attr = F.one_hot(
                reactant.edge_attr[:, 0], len(e_map["bond_type"])
            ).to(torch.float)

            reactant.sample_id = sample["rowid"]
            reactant.sequence = sequence

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
            # reactant.y = protein_hidden.view(1, -1)
            reactant.y = sample["y"]
            reactant.ec = sample["ec"]
            return reactant

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
        args_str = str(
            [
                self.args.dataset_file_path,
                self.args.split_seed,
                self.args.max_reactant_size,
                self.args.dataset_name,
                self.args.assign_splits,
                self.args.use_residues_in_reaction,
                self.args.use_random_smiles_representation,
                self.args.max_protein_length,
                self.args.split_type,
                self.args.sample_negatives,
                self.args.sample_negatives_range,
                self.args.ec_level,
                self.args.negative_samples_cache_dir,
            ]
        )
        # to avoid deleting cached negatives
        if self.args.sample_k_negatives is not None:
            args_str += str(self.args.sample_k_negatives)
        negative_hash = hashlib.md5(args_str.encode()).hexdigest()
        path_to_negatives = os.path.join(
            self.args.negative_samples_cache_dir,
            "negative_sampling",
            f"negatives_to_add_{negative_hash}",
        )
        # all negatives were stored only if _done.pkl exists
        if os.path.exists(
            os.path.join(path_to_negatives, f"{negative_hash}_{split_group}_done.pkl")
        ):
            # directory exists, load all negatives
            all_sample_paths = glob.glob(os.path.join(path_to_negatives, "*.pkl"))
            all_sample_paths = [i for i in all_sample_paths if "_done" not in i]
            negatives_to_add = []
            for path in tqdm(
                all_sample_paths,
                desc=f"Loading negatives from cache: {path_to_negatives}",
            ):
                sample = pickle.load(open(path, "rb"))
                # if not test then only add split group negatives
                if split_group in ["dev", "train"] and sample["split"] == split_group:
                    negatives_to_add.append(sample)
                elif split_group == "test":  # if test, add all negatives
                    negatives_to_add.append(sample)

            print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
        else:
            if not os.path.exists(path_to_negatives):
                os.makedirs(path_to_negatives)

            if split_group in ["train", "dev"]:
                all_substrates = set(
                    d["reactant"] for d in dataset if d["split"] == split_group
                )
            else:  # if test, need to use all substrates
                all_substrates = set(d["reactant"] for d in dataset)
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

            ec_to_positives = defaultdict(set)
            for sample in tqdm(dataset, desc="Sampling negatives"):
                if (
                    sample["split"] == split_group
                ):  # even in test, leave, since we want only test prots
                    ec = sample["ec"]
                    # if ec not in ec_to_positives:
                    ec_to_positives[ec].add(sample["reactant"])

                    if self.args.sample_negatives_range is not None:
                        # add to positives so that we don't sample them as negatives
                        idx = all_substrates_list.index(sample["reactant"])
                        ec_to_positives[ec].update(
                            all_substrates_list[j]
                            for j in similarity_idx[1][similarity_idx[0] == idx]
                        )

            ec_to_negatives = {
                k: all_substrates - v for k, v in ec_to_positives.items()
            }

            rowid = len(dataset)
            negatives_to_add = []
            for ec, negatives in tqdm(
                ec_to_negatives.items(), desc="Processing negatives"
            ):
                if self.args.sample_k_negatives is not None:
                    if len(negatives) < self.args.sample_k_negatives:
                        print(
                            f"Not enough negatives to sample from, using all negatives for {ec}"
                        )
                        negatives = list(negatives)
                    else:
                        negatives = random.sample(
                            negatives, self.args.sample_k_negatives
                        )

                for rid, reactant in enumerate(negatives):
                    sample = {
                        "reactant": reactant,
                        "ec": ec,
                        "reaction_string": "",
                        "rowid": f"{rid}{rowid}",
                        "split": split_group,
                        "y": 0,
                    }
                    rowid += 1
                    negatives_to_add.append(sample)
                    # dataset.append(sample)
                    sample_hash = hashlib.md5(str(sample).encode()).hexdigest()
                    sample_path = os.path.join(
                        path_to_negatives, f"{negative_hash}_{sample_hash}.pkl"
                    )
                    if not os.path.exists(sample_path):
                        pickle.dump(sample, open(sample_path, "wb"))

            print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
            # add a final pickle to indicate that we are done
            pickle.dump(
                f"done pickling {negative_hash}",
                open(
                    os.path.join(
                        path_to_negatives, f"{negative_hash}_{split_group}_done.pkl"
                    ),
                    "wb",
                ),
            )
            # pickle negatives_to_add for caching -> Too slow
            # pickle.dump(negatives_to_add, open(path_to_negatives, "wb"))

        dataset += negatives_to_add
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
            "protein_features",
            "substrate_features",
            "sequence",
            "smiles",
        ]
        return standard

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactant = from_smiles(sample["reactant"])

            ec = sample["ec"]
            valid_uniprots = self.valid_ec2uniprot[ec]
            uniprot_id = random.sample(valid_uniprots, 1)[0]
            sequence = self.uniprot2sequence[uniprot_id]

            # residue_dict = self.get_uniprot_residues(self.mcsa_data, sequence, ec)
            # residues = residue_dict["residues"]
            # residue_mask = residue_dict["residue_mask"]
            # has_residues = residue_dict["has_residues"]
            # residue_positions = residue_dict["residue_positions"]

            reactant.sample_id = sample["rowid"]
            reactant.sequence = sequence

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
            # token_hiddens = esm_features["token_hiddens"][mask_hiddens[:,0].bool()]
            # reactant.y = protein_hidden.view(1, -1)
            reactant.y = sample["y"]
            reactant.ec = sample["ec"]
            reactant.all_reactants = sample["reaction_string"].split(">>")[0].split(".")

            return reactant

        except Exception:
            warnings.warn(f"Could not load sample: {sample['rowid']}")
