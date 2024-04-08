from typing import List, Literal
from tqdm import tqdm
import argparse
import pickle
import copy, os
import numpy as np
import random
from random import Random
import hashlib
from collections import defaultdict, Counter
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem as rdk
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
import torch
import Bio
from Bio.Data.IUPACData import protein_letters_3to1
import Bio.PDB
from esm import pretrained
from clipzyme.utils.registry import register_object
from clipzyme.datasets.abstract import AbstractDataset


from clipzyme.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    compute_node_embedding,
)
from clipzyme.utils.pyg import from_mapped_smiles
from clipzyme.utils.wln_processing import get_bond_changes

ESM_MODEL2HIDDEN_DIM = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}

protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})


def stringify_sets(sets):
    final = []
    for i in sets:
        s = str(i[0]) + "-" + str(i[1]) + "-" + str(i[2])
        final.append(s)
    return " ".join(final)


def destringify_sets(x: str):
    return [
        (int(l.split("-")[0]), int(l.split("-")[1]), float(l.split("-")[2]))
        for l in x.split(" ")
    ]


@register_object("enzymemap_reactions", "dataset")
class EnzymeMap(AbstractDataset):
    def __init__(self, args, split_group) -> None:
        if args.use_protein_graphs:
            self.esm_dir = args.esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(args.esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
        super(EnzymeMap, EnzymeMap).__init__(self, args, split_group)
        self.metadata_json = None  # overwrite for memory

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.version = args.version
        self.load_dataset(args)

        self.valid_ec2uniprot = defaultdict(set)

        self.ec2uniprot = pickle.load(open("files/ec2uniprot.p", "rb"))
        self.uniprot2sequence = pickle.load(open("files/uniprot2sequence.p", "rb"))
        self.uniprot2sequence_len = {
            k: 0 if v is None else len(v) for k, v in self.uniprot2sequence.items()
        }

        # products to remove based on smiles or pattern
        remove_patterns_path = "files/ecreact/patterns.txt"
        remove_molecules_path = "files/ecreact/molecules.txt"

        self.remove_patterns = []
        self.remove_molecules = []

        for line in open(remove_patterns_path):
            if not line.startswith("//") and line.strip():
                self.remove_patterns.append(line.split("//")[0].strip())

        self.remove_patterns = [
            rdk.MolFromSmarts(smart_pattern) for smart_pattern in self.remove_patterns
        ]

        for line in open(remove_molecules_path):
            if not line.startswith("//") and line.strip():
                smiles = line.split("//")[0].strip()
                mol = rdk.MolFromSmiles(smiles)
                if mol:
                    self.remove_molecules.append(rdk.MolToSmiles(mol))
                    self.remove_molecules.append(
                        Chem.CanonSmiles(
                            self.remove_molecules[-1].replace("[O-]", "[OH]")
                        )
                    )

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter(
                [r for d in self.metadata_json for r in d["products"]]
            ).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter(
                [r for d in self.metadata_json for r in d.get("mapped_products", [])]
            ).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {
                s[0]: True
                for byproducts in [raw_byproducts, mapped_byproducts]
                for s in byproducts
            }

        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = (
                sorted(reaction.get("mapped_reactants", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["reactants"])
            )
            products = (
                sorted(reaction.get("mapped_products", []))
                if self.args.use_mapped_reaction
                else sorted(reaction["products"])
            )
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            # select uniprots
            if self.args.version == "1":
                alluniprots = self.ec2uniprot.get(ec, [])
                protein_refs = []
            elif self.args.version == "2":
                protein_refs = eval(reaction["protein_refs"])
                alluniprots = protein_refs
                if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                    alluniprots = self.ec2uniprot.get(ec, [])

            valid_uniprots = []
            for uniprot in alluniprots:
                temp_sample = {
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "protein_id": uniprot,
                    "protein_db": reaction.get("protein_db", ""),
                    "protein_refs": protein_refs,
                    "organism": reaction.get("organism", ""),
                    "rule_id": reaction["rule_id"],
                }
                if self.skip_sample(temp_sample, split_group):
                    continue

                valid_uniprots.append(uniprot)

            if len(valid_uniprots) == 0:
                continue

            for uniprot in valid_uniprots:
                sample = {
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "rowid": f"{uniprot}_{reaction['rxnid']}",
                    "uniprot_id": uniprot,
                    "protein_id": uniprot,
                    "organism": reaction.get("organism", ""),
                    "rule_id": reaction["rule_id"],
                }
                if "split" in reaction:
                    sample["split"] = reaction["split"]
                # add reaction sample to dataset
                dataset.append(sample)

                for ec_level, _ in enumerate(ec.split(".")):
                    sample[f"ec{ec_level+1}"] = ".".join(
                        ec.split(".")[: (ec_level + 1)]
                    )

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if sample["quality"] < self.args.min_reaction_quality:
            return True

        if "-" in sample["ec"]:
            return True

        # if sequence is unknown
        sequence = self.uniprot2sequence.get(sample["protein_id"], None)
        if (sequence is None) or (len(sequence) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sequence
        ) > self.args.max_protein_length:
            return True

        if self.args.max_reactant_size is not None:
            for mol in sample["reactants"]:
                if not (mol in self.mol2size):
                    self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

                if self.mol2size[mol] > self.args.max_reactant_size:
                    return True

        if self.args.max_product_size is not None:
            for mol in sample["products"]:
                if not (mol in self.mol2size):
                    self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()

                if self.mol2size[mol] > self.args.max_product_size:
                    return True
            # if self.mol2size[mol] < 2:
            #     return True

        if len(sample["products"]) > self.args.max_num_products:
            return True

        if ("bond_changes" in sample) and (len(sample["bond_changes"]) == 0):
            return True

        if (self.version == "2") and (
            sample["protein_db"] not in ["swissprot", "uniprot"]
        ):
            if len(sample["protein_refs"]) > 0:  # ids obtained from reference not used
                return True

            if len(sample["protein_refs"]) == 0 and (
                not self.args.sample_uniprot_per_ec
            ):
                return True

        if self.args.remove_duplicate_reactions:
            # reaction = "{}>>{}".format(
            #     remove_atom_maps_manual(".".join(sample["reactants"])),
            #     remove_atom_maps_manual(".".join(sample["products"])),
            # )

            reaction = "{}|{}".format(sample["reaction_string"], sample["uniprot_id"])
            if reaction in self.unique_reactions:
                return True
            self.unique_reactions.add(reaction)

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = (
                copy.deepcopy(sample["reactants"]),
                copy.deepcopy(sample["products"]),
            )

            ec = sample["ec"]
            uniprot_id = sample.get("uniprot_id", "unk")
            sequence = self.uniprot2sequence.get(uniprot_id, "<unk>")

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

            # remove atom-mapping if applicable
            reactants = ".".join(reactants)
            products = ".".join(products)
            # reactants = remove_atom_maps(reactants)
            # products = remove_atom_maps(products)

            # remove stereochemistry
            if self.args.remove_stereochemistry:
                reactants_mol = Chem.MolFromSmiles(reactants)
                products_mol = Chem.MolFromSmiles(products)
                Chem.RemoveStereochemistry(reactants_mol)
                Chem.RemoveStereochemistry(products_mol)
                reactants = Chem.MolToSmiles(reactants_mol)
                products = Chem.MolToSmiles(products_mol)

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
                "smiles": ".".join(products),
                "all_smiles": list(
                    self.reaction_to_products[
                        f"{ec}{'.'.join(sorted(sample['reactants']))}"
                    ]
                ),
                "quality": sample["quality"],
            }

            split_ec = ec.split(".")
            for k, v in self.args.ec_levels.items():
                item[f"ec{k}"] = v.get(".".join(split_ec[: int(k)]), -1)

            return item

        except Exception as e:
            print(
                f"Getitem enzymemap: Could not load sample {sample['uniprot_id']} because of exception {e}"
            )

    def get_pesto_scores(self, uniprot):
        filepath = f"{self.args.pesto_scores_directory}/AF-{uniprot}-F1-model_v4.pt"
        if not os.path.exists(filepath):
            return None
        scores_dict = torch.load(filepath)
        chain = "A:0"  # * NOTE: hardcoded because currently only option
        residue_ids = scores_dict[chain]["resid"]
        residue_ids_unique = np.unique(residue_ids, return_index=True)[1]
        scores = scores_dict[chain]["ligand"][residue_ids_unique]
        return torch.tensor(scores)

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # get all samples
        self.to_split = {}

        # set seed
        np.random.seed(seed)

        # rule id
        rules = [reaction["rule_id"] for reaction in metadata_json]
        rule2count = Counter(rules)
        samples = sorted(list(set(rules)))
        np.random.shuffle(samples)
        samples_cumsum = np.cumsum([rule2count[s] for s in samples])
        # Find the indices for each quantile
        split_indices = [
            np.searchsorted(
                samples_cumsum, np.round(q, 3) * samples_cumsum[-1], side="right"
            )
            for q in np.cumsum(split_probs)
        ]
        split_indices[-1] = len(samples)
        split_indices = np.concatenate([[0], split_indices])
        for i in range(len(split_indices) - 1):
            self.to_split.update(
                {
                    sample: ["train", "dev", "test"][i]
                    for sample in samples[split_indices[i] : split_indices[i + 1]]
                }
            )

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        for sample in processed_dataset:
            # check right split
            if self.to_split[sample["rule_id"]] != split_group:
                continue
            dataset.append(sample)
        return dataset

    def post_process(self, args):
        # add all possible products
        reaction_to_products = defaultdict(set)
        for sample in self.dataset:
            key = f"{sample['ec']}{'.'.join(sample['reactants'])}"
            reaction_to_products[key].add(".".join(sample["products"]))
        self.reaction_to_products = reaction_to_products

        # set ec levels to id for use in modeling
        ecs = [d["ec"].split(".") for d in self.dataset]
        args.ec_levels = {}
        for level in range(1, 5, 1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c: i for i, c in enumerate(unique_classes)}

    def remove_from_products(self, product):
        mol = Chem.MolFromSmiles(product)
        for mol_pattern in self.remove_patterns:
            if mol.HasSubstructMatch(mol_pattern):
                return True
        if product in self.remove_molecules:
            return True
        return False

    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(
                self.args.protein_structures_dir,
                f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
            )
            protein_args = {
                "sample_id": sample["sample_id"],
                "protein_parser": Bio.PDB.MMCIFParser(),
                "protein_resolution": "residue",
                "graph_edge_args": {"knn_size": 10},
                "center_protein": True,
            }

            sample_id = protein_args["sample_id"]
            protein_parser = protein_args["protein_parser"]
            protein_resolution = protein_args["protein_resolution"]
            graph_edge_args = protein_args["graph_edge_args"]
            center_protein = protein_args["center_protein"]

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
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            data.structure_sequence = self.uniprot2sequence[uniprot_id]

            node_embeddings_args = {
                "model": self.esm_model,
                "model_location": self.esm_dir,
                "alphabet": self.alphabet,
                "batch_converter": self.batch_converter,
            }

            embedding_path = os.path.join(
                self.args.protein_graphs_dir,
                "precomputed_node_embeddings",
                f"{sample['uniprot_id']}.pt",
            )

            if os.path.exists(embedding_path):
                node_embedding = torch.load(sample["embedding_path"])
            else:
                node_embedding = compute_node_embedding(data, **node_embeddings_args)
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]

                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, **node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding

            if len(data["receptor"].seq) != data["receptor"].x.shape[0]:
                return None

            return data

        except Exception as e:
            print(
                f"Create prot graph: Could not load sample {sample['uniprot_id']} because of the exception {e}"
            )
            return None

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)
        parser.add_argument(
            "--esm_dir",
            type=str,
            default="/home/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )
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
        parser.add_argument(
            "--use_protein_msa",
            action="store_true",
            default=False,
            help="whether to use and generate protein MSAs",
        )
        parser.add_argument(
            "--protein_msa_dir",
            type=str,
            default="/home/datasets/embed_msa_transformer",
            help="directory where msa transformer embeddings are stored.",
        )
        parser.add_argument(
            "--replace_esm_with_msa",
            action="store_true",
            default=False,
            help="whether to use ONLY the protein MSAs",
        )
        parser.add_argument(
            "--ec_level",
            type=int,
            default=3,
            choices=[0, 1, 2, 3],
            help="EC level to use (e.g., ec_level 1 of '1.2.3.1' -> '1.2')",
        )
        parser.add_argument(
            "--randomize_order_in_reaction",
            action="store_true",
            default=False,
            help="Permute smiles in reactants and in products as augmentation",
        )
        parser.add_argument(
            "--use_random_smiles_representation",
            action="store_true",
            default=False,
            help="Use non-canonical representation of smiles as augmentation",
        )
        parser.add_argument(
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )
        parser.add_argument(
            "--use_mapped_reaction",
            action="store_true",
            default=False,
            help="use atom-mapped reactions",
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
            help="maximum reactant size",
        )
        parser.add_argument(
            "--max_num_products",
            type=int,
            default=np.inf,
            help="maximum number of products",
        )
        parser.add_argument(
            "--topk_byproducts_to_remove",
            type=int,
            default=None,
            help="remove common byproducts",
        )
        parser.add_argument(
            "--create_sample_per_sequence",
            action="store_true",
            default=False,
            help="create a sample for each protein sequence annotated for given EC",
        )
        parser.add_argument(
            "--sample_uniprot_per_ec",
            action="store_true",
            default=False,
            help="randomly sample a uniprot for each EC at getitem",
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles",
        )
        parser.add_argument(
            "--min_reaction_quality",
            type=float,
            default=-1,
            help="minimum threshold to use for filtering reactions based on quality score",
        )
        parser.add_argument(
            "--split_multiproduct_samples",
            action="store_true",
            default=False,
            help="split products into different samples",
        )
        parser.add_argument(
            "--use_one_hot_mol_features",
            action="store_true",
            default=False,
            help="encode node and edge features of molecule as one-hot",
        )
        parser.add_argument(
            "--version",
            type=str,
            default="1",
            help="enzyme map version number",
        )
        parser.add_argument(
            "--remove_duplicate_reactions",
            action="store_true",
            default=False,
            help="remove duplicates",
        )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        try:
            reactions = [
                "{}>>{}".format(".".join(d["reactants"]), ".".join(d["products"]))
                for d in self.dataset
            ]
        except:
            reactions = "NA"
        try:
            proteins = [d["uniprot_id"] for d in self.dataset]
        except:
            proteins = "NA"
        try:
            ecs = [d["ec"] for d in self.dataset]
        except:
            ecs = "NA"
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("enzymemap_reaction_graph", "dataset")
class EnzymeMapGraph(EnzymeMap):
    def post_process(self, args):
        # set ec levels to id for use in modeling
        ecs = set(d["ec"] for d in self.dataset)
        ecs = [e.split(".") for e in ecs]
        args.ec_levels = {}
        for level in range(1, 5, 1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c: i for i, c in enumerate(unique_classes)}
        if hasattr(args, "do_ec_task") and args.do_ec_task:
            args.num_classes = len(args.ec_levels["4"])

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter(
                [r for d in self.metadata_json for r in d["products"]]
            ).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter(
                [r for d in self.metadata_json for r in d.get("mapped_products", [])]
            ).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {
                s[0]: True
                for byproducts in [raw_byproducts, mapped_byproducts]
                for s in byproducts
            }

        if self.args.remove_duplicate_reactions:
            self.unique_reactions = set()

        dataset = []

        rkey = (
            "mapped_reactants"
            if "mapped_reactants" in self.metadata_json[0]
            else "reactants"
        )
        pkey = (
            "mapped_products"
            if "mapped_products" in self.metadata_json[0]
            else "products"
        )

        self.mol2size = {}

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            ec = reaction["ec"]
            eclevels_dict = {
                f"ec{ec_level+1}": ".".join(ec.split(".")[: (ec_level + 1)])
                for ec_level, _ in enumerate(ec.split("."))
            }
            organism = reaction.get("organism", "")

            reactants = sorted([s for s in reaction[rkey] if s != "[H+]"])
            products = sorted([s for s in reaction[pkey] if s != "[H+]"])
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            reaction_string = "{}>>{}".format(".".join(reactants), ".".join(products))

            bond_changes = reaction.get("bond_changes", None)
            if not bond_changes:
                try:
                    bond_changes = get_bond_changes(reaction_string)
                except:
                    continue

            # select uniprots
            if self.args.version == "1":
                alluniprots = self.ec2uniprot.get(ec, [])
                protein_refs = []
            elif self.args.version == "2":
                protein_refs = eval(reaction["protein_refs"])
                alluniprots = protein_refs
                if (len(alluniprots) == 0) and self.args.sample_uniprot_per_ec:
                    alluniprots = self.ec2uniprot.get(ec, [])

            if self.args.create_sample_per_sequence or self.args.sample_uniprot_per_ec:
                for uniprot in alluniprots:
                    sample = {
                        "reaction_string": "{}>>{}".format(
                            ".".join(sorted(reaction["reactants"])),
                            ".".join(sorted(reaction["products"])),
                        ),
                        "df_row": rowid,
                        "quality": reaction["quality"],
                        "reactants": reactants,
                        "products": products,
                        "ec": ec,
                        "rowid": reaction["rxnid"],
                        "sample_id": f"{uniprot}_{reaction['rxnid']}_{rowid}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                        "bond_changes": list(bond_changes),
                        "split": reaction.get("split", None),
                        "protein_refs": protein_refs,
                        "protein_db": reaction.get("protein_db", ""),
                        "rule_id": reaction["rule_id"],
                    }
                    sample.update(eclevels_dict)

                    if self.skip_sample(sample, split_group):
                        continue

                    if self.args.sample_uniprot_per_ec:
                        self.valid_ec2uniprot[ec].add(uniprot)

                        sample = {
                            "df_row": rowid,
                            "quality": reaction["quality"],
                            "reactants": reactants,
                            "products": products,
                            "ec": ec,
                            "rowid": reaction["rxnid"],
                            "sample_id": str(reaction["rxnid"]),
                            "uniprot_id": "",
                            "protein_id": "",
                            "sequence": "X",
                            "bond_changes": list(bond_changes),
                            "split": reaction.get("split", None),
                            "protein_refs": protein_refs,
                            "rule_id": reaction["rule_id"],
                        }
                        sample.update(eclevels_dict)

                        if self.args.split_type == "ec_hold_out":
                            unique_sample_content = f"{reaction_string}"
                            hashed_sample_content = hashlib.sha256(
                                unique_sample_content.encode("utf-8")
                            ).hexdigest()
                            sample["hash_sample_id"] = hashed_sample_content

                        if self.args.split_multiproduct_samples:
                            for product_id, p in enumerate(products):
                                psample = copy.deepcopy(sample)
                                psample["products"] = [p]
                                psample["sample_id"] += f"_{product_id}"
                                dataset.append(psample)

                        else:
                            dataset.append(sample)

                    else:
                        if self.args.split_type == "ec_hold_out":
                            unique_sample_content = (
                                f"{reaction_string}{uniprot}{organism}"
                            )
                            hashed_sample_content = hashlib.sha256(
                                unique_sample_content.encode("utf-8")
                            ).hexdigest()
                            sample["hash_sample_id"] = hashed_sample_content

                        try:
                            # make prot graph if missing
                            if self.args.use_protein_graphs:
                                graph_path = os.path.join(
                                    self.args.protein_graphs_dir,
                                    "processed",
                                    f"{sample['uniprot_id']}_graph.pt",
                                )
                                structure_path = os.path.join(
                                    self.args.protein_structures_dir,
                                    f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
                                )
                                if not os.path.exists(structure_path):
                                    continue
                                if not os.path.exists(graph_path):
                                    print("Generating none existent protein graph")
                                    data = self.create_protein_graph(sample)
                                    if data is None:
                                        raise Exception(
                                            "Could not generate protein graph"
                                        )
                                    torch.save(data, graph_path)

                        except Exception as e:
                            print(
                                f"Error processing {sample['sample_id']} because of {e}"
                            )
                            continue

                        # add reaction sample to dataset
                        if self.args.split_multiproduct_samples:
                            for product_id, p in enumerate(products):
                                psample = copy.deepcopy(sample)
                                psample["products"] = [p]
                                psample["sample_id"] += f"_{product_id}"
                                preaction_string = "{}>>{}".format(
                                    ".".join(psample["reactants"]), p
                                )
                                uniprot = psample["uniprot_id"]
                                punique_sample_content = (
                                    f"{preaction_string}{uniprot}{psample['organism']}"
                                )
                                phashed_sample_content = hashlib.sha256(
                                    punique_sample_content.encode("utf-8")
                                ).hexdigest()
                                psample["hash_sample_id"] = phashed_sample_content
                                dataset.append(psample)
                        else:
                            dataset.append(sample)
            else:
                sample = {
                    "df_row": rowid,
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "rowid": reaction["rxnid"],
                    "sample_id": str(reaction["rxnid"]),
                    "uniprot_id": "",
                    "protein_id": "",
                    "sequence": "X",
                    "bond_changes": list(bond_changes),
                    "split": reaction.get("split", None),
                    "protein_refs": protein_refs,
                    "rule_id": reaction["rule_id"],
                }

                unique_sample_content = f"{reaction_string}"
                hashed_sample_content = hashlib.sha256(
                    unique_sample_content.encode("utf-8")
                ).hexdigest()
                sample["hash_sample_id"] = hashed_sample_content

                sample.update(eclevels_dict)

                if self.skip_sample(sample, split_group):
                    continue

                if self.args.split_multiproduct_samples:
                    for product_id, p in enumerate(products):
                        psample = copy.deepcopy(sample)
                        psample["products"] = [p]
                        psample["sample_id"] += f"_{product_id}"
                        dataset.append(psample)

                else:
                    dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = (
                copy.deepcopy(sample["reactants"]),
                copy.deepcopy(sample["products"]),
            )

            ec = sample["ec"]
            if self.args.create_sample_per_sequence:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence.get(uniprot_id, "<unk>")
            elif self.args.sample_uniprot_per_ec:
                valid_uniprots = self.valid_ec2uniprot.get(ec, ["<unk>"])
                uniprot_id = random.sample(valid_uniprots, 1)[0]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                uniprot_id = "unk"
                sequence = "<unk>"

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

            reactants, atom_map2new_index = from_mapped_smiles(
                ".".join(reactants),
                encode_no_edge=True,
                use_one_hot_encoding=self.args.use_one_hot_mol_features,
            )
            products, _ = from_mapped_smiles(
                ".".join(products),
                encode_no_edge=True,
                use_one_hot_encoding=self.args.use_one_hot_mol_features,
            )

            bond_changes = [
                (atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype)
                for u, v, btype in sample["bond_changes"]
            ]
            bond_changes = [(min(x, y), max(x, y), t) for x, y, t in bond_changes]
            reactants.bond_changes = bond_changes
            sample_id = sample["sample_id"]
            rowid = sample["rowid"]

            reaction_nodes = torch.zeros(reactants.x.shape[0])
            for s in [bond_changes]:
                for u, v, t in s:
                    reaction_nodes[u] = 1
                    reaction_nodes[v] = 1

            reactants.reaction_nodes = reaction_nodes

            item = {
                "x": reaction,
                "reaction": reaction,
                "reactants": reactants,
                "mol": reactants,
                "products": products,
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "uniprot_id": uniprot_id,
                "sample_id": sample_id,
                "row_id": rowid,
                "smiles": products,
                "all_smiles": [],  # all_smiles,
                "quality": sample["quality"],
                # "bond_changes": stringify_sets(bond_changes)
            }

            # ecs as tensors
            split_ec = ec.split(".")
            for k, v in self.args.ec_levels.items():
                if not self.args.ec_levels_one_hot:  # index encoding
                    item[f"ec{k}"] = v.get(".".join(split_ec[: int(k)]), -1)
                else:  # one hot encoding
                    yvec = torch.zeros(len(v))
                    yvec[v[".".join(split_ec[: int(k)])]] = 1
                    item[f"ec{k}"] = yvec

            if self.args.load_wln_cache_in_dataset:
                item["product_candidates"] = self.cache.get(rowid)

            if self.args.use_protein_graphs:
                if self.args.cache_path:
                    try:
                        graph_path_cache = os.path.join(
                            self.args.cache_path,
                            f"{item['uniprot_id']}_graph.pt",
                        )
                        data = torch.load(graph_path_cache)
                        if data is None:
                            data = self.load_protein_graph(item)
                            torch.save(data, graph_path_cache)
                    except:
                        data = self.load_protein_graph(item)
                        torch.save(data, graph_path_cache)
                else:
                    data = self.load_protein_graph(item)

                if self.args.use_protein_msa:
                    feats = data["receptor"].x
                    msa_embed = torch.load(
                        os.path.join(self.args.protein_msa_dir, f"{uniprot_id}.pt")
                    )
                    if self.args.replace_esm_with_msa:
                        data["receptor"].x = msa_embed
                    else:
                        data["receptor"].x = torch.concat([feats, msa_embed], dim=-1)
                        data["receptor"].msa = msa_embed

                item["graph"] = data

            return item

        except Exception as e:
            print(
                f"Could not load sample {sample['uniprot_id']} because of an exception {e}"
            )

    def load_protein_graph(self, item):
        # load the protein graph
        graph_path = os.path.join(
            self.args.protein_graphs_dir,
            "processed",
            f"{item['uniprot_id']}_graph.pt",
        )
        try:
            data = torch.load(graph_path)
        except:
            data = self.create_protein_graph(item)
            torch.save(data, graph_path)
        if data is None:
            try:
                data = self.create_protein_graph(item)
                torch.save(data, graph_path)
            except:
                return

        if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
            data["receptor"].x = data.x

        if not hasattr(data, "structure_sequence"):
            data.structure_sequence = "".join(
                [protein_letters_3to1[char] for char in data["receptor"].seq]
            )

        keep_keys = {
            "receptor",
            "structure_sequence",
            ("receptor", "contact", "receptor"),
        }

        data_keys = data.to_dict().keys()
        for d_key in data_keys:
            if not d_key in keep_keys:
                delattr(data, d_key)

        coors = data["receptor"].pos
        feats = data["receptor"].x
        edge_index = data["receptor", "contact", "receptor"].edge_index
        assert (
            coors.shape[0] == feats.shape[0]
        ), f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

        assert (
            max(edge_index[0]) < coors.shape[0] and max(edge_index[1]) < coors.shape[0]
        ), "Edge index contains node indices not present in coors"

        return data

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args"""
        super(EnzymeMapGraph, EnzymeMapGraph).add_args(parser)
        parser.add_argument(
            "--ec_levels_one_hot",
            action="store_true",
            default=False,
            help="whether to use one hot encoding for ec levels",
        )
