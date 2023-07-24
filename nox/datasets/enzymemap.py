from typing import List, Literal
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from tqdm import tqdm
import argparse
import pickle
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from nox.utils.smiles import get_rdkit_feature, remove_atom_maps, assign_dummy_atom_maps
from nox.utils.pyg import from_smiles, from_mapped_smiles
import warnings
import copy, os
import numpy as np
import random
from collections import defaultdict, Counter
import rdkit
import torch
import hashlib

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
from nox.utils.wln_processing import get_bond_changes
from torch_geometric.data import HeteroData, Data
from torch_geometric.data import Dataset
import argparse
from nox.models.wln import WLDN_Cache

def stringify_sets(sets):
    final = []
    for i in sets:
        s = str(i[0]) + '-' + str(i[1]) + '-' + str(i[2])
        final.append(s)
    return " ".join(final)

def destringify_sets(x:str):
    return [(int(l.split('-')[0]), int(l.split('-')[1]), float(l.split('-')[2])) for l in x.split(" ")]

@register_object("enzymemap_reactions", "dataset")
class EnzymeMap(AbstractDataset):
    def __init__(self, args, split_group) -> None:
        super(EnzymeMap, EnzymeMap).__init__(self, args, split_group)
        self.metadata_json = None  # overwrite for memory
        if args.load_wln_cache_in_dataset:
            self.cache = WLDN_Cache(args.cache_path)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.load_dataset(args)

        self.valid_ec2uniprot = {}

        self.ec2uniprot = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/ec2uniprot.p",
                "rb",
            )
        )
        self.uniprot2sequence = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/uniprot2sequence.p",
                "rb",
            )
        )
        self.uniprot2cluster = pickle.load(
            open(
                "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/mmseq_clusters.p",  # TODO
                "rb",
            )
        )

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:

        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter([r for d in self.metadata_json for r in d["products"]]).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter([r for d in self.metadata_json for r in d.get("mapped_products", []) ]).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {s[0]:True for byproducts in [raw_byproducts, mapped_byproducts] for s in byproducts}


        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = sorted(reaction.get("mapped_reactants", []))  if self.args.use_mapped_reaction else sorted(reaction["reactants"])
            products = sorted(reaction.get("mapped_products", [])) if self.args.use_mapped_reaction else sorted(reaction["products"])
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            valid_uniprots = []
            for uniprot in self.ec2uniprot.get(ec, []):
                temp_sample = {
                    "quality": reaction["quality"],
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "protein_id": uniprot,
                    "sequence": self.uniprot2sequence[uniprot],
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
                }
                if "split" in reaction:
                    sample["split"] = reaction["split"]
                # add reaction sample to dataset
                dataset.append(sample)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if sample["quality"] < self.args.min_reaction_quality:
            return True 
            
        if "-" in sample["ec"]:
            return True

        # if sequence is unknown
        sequence = sample["sequence"]
        if (sequence is None) or (len(sequence) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sequence
        ) > self.args.max_protein_length:
            return True

        for mol in sample["reactants"]:
            if not (mol in self.mol2size):
                self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()
                
            if self.args.max_reactant_size is not None:
                if self.mol2size[mol] > self.args.max_reactant_size:
                    return True
        
        for mol in sample["products"]:
            if not (mol in self.mol2size):
                self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()
            if self.args.max_product_size is not None:
                if self.mol2size[mol] > self.args.max_product_size:
                    return True
            # if self.mol2size[mol] < 2:
            #     return True 


        if len(sample['products']) > self.args.max_num_products:
            return True 

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            ec = sample["ec"]
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]

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
            reactants = remove_atom_maps(".".join(reactants))
            products = remove_atom_maps(".".join(products))

            # remove stereochemistry
            if self.args.remove_stereochemistry:
                reactants_mol = Chem.MolFromSmiles(reactants)
                products_mol = Chem.MolFromSmiles(products)
                Chem.RemoveStereochemistry(reactants_mol)
                Chem.RemoveStereochemistry(products_mol)
                reactants =  Chem.MolToSmiles(reactants)
                products = MolToSmiles(products)

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
                    self.reaction_to_products[f"{ec}{'.'.join(sorted(sample['reactants']))}"]
                ),
                "quality": sample["quality"],
            }

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            return item

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of exception {e}")

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

        # assign groups
        if self.args.split_type in ["mmseqs", "sequence", "ec", "product"]:
            if self.args.split_type == "mmseqs":
                samples = list(self.uniprot2cluster.values())

            if self.args.split_type == "sequence":
                # split based on uniprot_id
                samples = [
                    u
                    for reaction in metadata_json
                    for u in self.ec2uniprot.get(reaction["ec"], [])
                ]

            elif self.args.split_type == "ec":
                # split based on ec number
                samples = [reaction["ec"] for reaction in metadata_json]

                # option to change level of ec categorization based on which to split
                samples = [
                    ".".join(e.split(".")[: self.args.ec_level + 1]) for e in samples
                ]

            elif self.args.split_type == "product":
                # split by reaction product (splits share no products)
                samples = [".".join(s["products"]) for s in metadata_json]
                

            samples = sorted(list(set(samples)))
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

        # random splitting
        elif self.args.split_type == "random":
            for sample in self.metadata_json:
                reaction_string = (
                    ".".join(sample["reactants"]) + ">>" + ".".join(sample["products"])
                )
                self.to_split.update(
                    {
                        reaction_string: np.random.choice(
                            ["train", "dev", "test"], p=split_probs
                        )
                    }
                )
        else:
            raise ValueError("Split type not supported")

    def get_split_group_dataset(self, processed_dataset, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []
        for sample in processed_dataset:
            # check right split
            if self.args.split_type == "ec":
                ec = sample["ec"]
                split_ec = ".".join(ec.split(".")[: self.args.ec_level + 1])
                if self.to_split[split_ec] != split_group:
                    continue

            elif self.args.split_type == "mmseqs":
                cluster = self.uniprot2cluster[sample["protein_id"]]
                if self.to_split[cluster] != split_group:
                    continue

            elif self.args.split_type in ["product"]:
                products = ".".join(sample["products"])
                if self.to_split[products] != split_group:
                    continue

            elif self.args.split_type == "sequence":
                uniprot = sample["protein_id"]
                if self.to_split[uniprot] != split_group:
                    continue

            elif sample["split"] is not None:
                if sample["split"] != split_group:
                    continue
            dataset.append(sample)
        return dataset

    def post_process(self, args):
        # add all possible products
        reaction_to_products = defaultdict(set)
        for sample in self.dataset:
            key = f"{sample['ec']}{'.'.join(sample['reactants'])}" if args.create_sample_per_sequence else '.'.join(sample['reactants'])
            reaction_to_products[key].add(('.'.join(sample["products"]), stringify_sets(sorted(sample["bond_changes"])) ))
        self.reaction_to_products = reaction_to_products

        # set ec levels to id for use in modeling
        ecs = [d['ec'].split('.') for d in self.dataset]
        args.ec_levels = {}
        for level in range(1,5,1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c:i for i,c in enumerate(unique_classes)}

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(EnzymeMap, EnzymeMap).add_args(parser)
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
            "--use_pesto_scores",
            action="store_true",
            default=False,
            help="use pesto scores",
        )
        parser.add_argument(
            "--pesto_scores_directory",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/pesto_ligands",
            help="load pesto scores from directory predictions",
        )
        parser.add_argument(
            "--create_sample_per_sequence",
            action="store_true",
            default=False,
            help="create a sample for each protein sequence annotated for given EC"
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles"
        )
        parser.add_argument(
            "--min_reaction_quality",
            type=float,
            default=-1,
            help="minimum threshold to use for filtering reactions based on quality score"
        )
        parser.add_argument(
            "--load_wln_cache_in_dataset",
            action="store_true",
            default=False,
            help="load cache for wln in getitem"
        )
        parser.add_argument(
            "--split_multiproduct_samples",
            action="store_true",
            default=False,
            help="split products into different samples"
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
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("enzymemap_single_reactions", "dataset")
class EnzymeMapSingle(EnzymeMap):

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:

        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter([r for d in self.metadata_json for r in d["products"]]).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter([r for d in self.metadata_json for r in d.get("mapped_products", []) ]).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {s[0]:True for byproducts in [raw_byproducts, mapped_byproducts] for s in byproducts}


        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = sorted(reaction.get("mapped_reactants", []))  if self.args.use_mapped_reaction else sorted(reaction["reactants"])
            products = sorted(reaction.get("mapped_products", [])) if self.args.use_mapped_reaction else sorted(reaction["products"])
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            valid_uniprots = set()
            for product in products:
                for uniprot in self.ec2uniprot.get(ec, []):
                    temp_sample = {
                        "reactants": reactants,
                        "products": [product],
                        "ec": ec,
                        "protein_id": uniprot,
                        "sequence": self.uniprot2sequence[uniprot],
                    }
                    if self.skip_sample(temp_sample, split_group):
                        continue

                    valid_uniprots.add(uniprot)

                if len(valid_uniprots) == 0:
                    continue

                for uniprot in valid_uniprots:
                    sample = {
                        "reactants": reactants,
                        "products": [product],
                        "ec": ec,
                        "rowid": f"{uniprot}_{reaction['rxnid']}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                    }
                    if "split" in reaction:
                        sample["split"] = reaction["split"]
                    # add reaction sample to dataset
                    dataset.append(sample)

        return dataset


@register_object("enzymemap_substrate", "dataset")
class EnzymeMapSubstrate(EnzymeMap):
    def __init__(self, args, split_group) -> None:
        esm_dir = "/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"
        self.esm_dir = esm_dir
        model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        super(EnzymeMapSubstrate, EnzymeMapSubstrate).__init__(self, args, split_group)

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:

        # if removing top K
        # if self.args.topk_byproducts_to_remove is not None:
        #     raw_byproducts = Counter([r for d in self.metadata_json for r in d["products"]]).most_common(self.args.topk_byproducts_to_remove)
        #     mapped_byproducts = Counter([r for d in self.metadata_json for r in d.get("mapped_products", []) ]).most_common(self.args.topk_byproducts_to_remove)
        #     self.common_byproducts = {s[0]:True for byproducts in [raw_byproducts, mapped_byproducts] for s in byproducts}
        self.mol2size = {}

        if self.args.topk_substrates_to_remove is not None:
            raw_substrates = Counter([r for d in self.metadata_json for r in d["reactants"]]).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = set([s[0] for s in raw_substrates])

        dataset = []
        seen_before = set()

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = sorted(reaction.get("mapped_reactants", []))  if self.args.use_mapped_reaction else sorted(reaction["reactants"])
            products = sorted(reaction.get("mapped_products", [])) if self.args.use_mapped_reaction else sorted(reaction["products"])
            products = [p for p in products if p not in reactants]

            # if self.args.topk_byproducts_to_remove is not None:
            #     products = [p for p in products if p not in self.common_byproducts]
            if self.args.topk_substrates_to_remove is not None:
                reactants = [s for s in reactants if s not in self.common_substrates]

            valid_uniprots = set()
            for r in reactants:
                for uniprot in self.ec2uniprot.get(ec, []):
                    temp_sample = {
                        "smiles": r,
                        "ec": ec,
                        "protein_id": uniprot,
                        "uniprot_id": uniprot,
                        "sequence": self.uniprot2sequence[uniprot],
                        "y": 1,
                    }
                    if self.skip_sample(temp_sample, split_group):
                        continue

                    valid_uniprots.add(uniprot)

                if len(valid_uniprots) == 0:
                    continue

                for uniprot in valid_uniprots:
                    sample = {
                        "smiles": r,
                        "ec": ec,
                        "rowid": f"{uniprot}_{reaction['rxnid']}",
                        "sample_id": f"{uniprot}_{reaction['rxnid']}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                        "y": 1,
                    }
                    # remove duplicate prot-substrate pairs
                    if f"{uniprot}_{r}" in seen_before:
                        continue
                    seen_before.add(f"{uniprot}_{r}")
                    # add reaction sample to dataset
                    try:
                        if self.args.use_protein_graphs:
                            graph_path = os.path.join(self.args.protein_graphs_dir, "processed", f"{sample['uniprot_id']}_graph.pt")

                            if not os.path.exists(graph_path):
                                data = self.create_protein_graph(sample)
                                torch.save(data, graph_path)
                        dataset.append(sample)
                    except Exception as e:
                        print(f"Error processing {sample['rowid']} because of {e}")
                        continue
        return dataset


    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            # ec = sample["ec"]
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            if sample['y'] == 0 and self.args.sample_negatives_on_get:
                # sample a negative substrate
                sample["smiles"] = list(self.prot_id_to_negatives[uniprot_id])[np.random.randint(0, len(self.prot_id_to_negatives[uniprot_id]))]

            smiles = sample["smiles"]

            if self.args.use_random_smiles_representation:
                try:
                    smiles = randomize_smiles_rotated(smiles)
                except:
                    pass

            sample_id = sample["rowid"]
            item = {
                "sequence": sequence,
                # "ec": ec,
                # "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "uniprot_id": uniprot_id,
                "sample_id": sample_id,
                "smiles": smiles,
                "y": sample['y']
            }

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            if self.args.use_protein_graphs:
                # load the protein graph
                graph_path = os.path.join(self.args.protein_graphs_dir, "processed", f"{item['uniprot_id']}_graph.pt")
                data = torch.load(graph_path)
                if data is None:
                    structure_path = os.path.join(self.args.protein_structures_dir, f"AF-{item['uniprot_id']}-F1-model_v4.cif")
                    assert os.path.exists(structure_path), f"Structure path {graph_path} does not exist"
                    print(f"Structure path does exist, but graph path does not exist {graph_path}")
                    data = self.create_protein_graph(item)
                    torch.save(data, graph_path)

                data = self.add_additional_data_to_graph(data, item)
                # TODO: remove in the future
                # if not hasattr(data, "structure_sequence"):
                #     protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
                #     AA_seq = ""
                #     for char in data['receptor'].seq:
                #         AA_seq += protein_letters_3to1[char]
                #     data.structure_sequence = AA_seq
                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x

                keep_keys = {"receptor", "mol_data", "sequence", "protein_id", "uniprot_id", "sample_id", "smiles", "y", ('receptor', 'contact', 'receptor')}
                
                data_keys = data.to_dict().keys()
                for d_key in data_keys:
                    if not d_key in keep_keys:
                        delattr(data, d_key)

                # if hasattr(data, "x"):
                #     delattr(data, "x")
                # if hasattr(data, "ec"):
                #     delattr(data, "ec")
                # if hasattr(data, "embedding_path"):
                #     delattr(data, "embedding_path")
                # if hasattr(data, "protein_path"):
                #     delattr(data, "protein_path")
                # if hasattr(data, "sample_hash"):
                #     delattr(data, "sample_hash")
                coors = data["receptor"].pos
                feats = data["receptor"].x
                edge_index = data["receptor", "contact", "receptor"].edge_index
                assert coors.shape[0] == feats.shape[0], \
                    f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

                assert max(edge_index[0]) < coors.shape[0] and max(edge_index[1]) < coors.shape[0], \
                    "Edge index contains node indices not present in coors"

                return data
            else: # just the substrate, with the protein sequence in the Data object
                reactant = from_smiles(sample["smiles"])
                for key in item.keys():
                    reactant[key] = item[key]
                return reactant

        except Exception as e:
            print(f"Could not load sample: {sample['sample_id']} due to {e}")


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
            sample_id = sample["sample_id"]
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
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            data.structure_sequence = self.uniprot2sequence[uniprot_id]
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
            
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                return None
            
            return data

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of the exception {e}")
            return None

    def post_process(self, args):
        if args.sample_negatives:
            self.dataset = self.add_negatives(self.dataset, split_group=self.split_group)

    def add_negatives(self, dataset, split_group):
        # # Uncomment to add ec
        # uniprot2ec = {}
        # for s in dataset:
        #     uniprot2ec[s["uniprot_id"]] = s["ec"]
        all_substrates = set(d["smiles"] for d in dataset)
        all_substrates_list = list(all_substrates)

        # filter out negatives based on some metric (e.g. similarity)
        if self.args.sample_negatives_range is not None:
            min_sim, max_sim = self.args.sample_negatives_range

            smiles2feature = {smile: get_rdkit_feature(mol=smile, method="morgan_binary") for smile in all_substrates}
            smile_fps = np.array([smiles2feature[smile] / np.linalg.norm(smiles2feature[smile]) for smile in all_substrates])

            smile_similarity = smile_fps @ smile_fps.T
            
            # similarity_idx = np.where(
            #     (smile_similarity <= min_sim) | (smile_similarity >= max_sim)
            # )

            # smiles2similars = defaultdict(set)
            # for smi_i, smile in tqdm(
            #     enumerate(all_substrates_list), desc="Retrieving all negatives", total=len(all_substrates_list)
            # ):
            #     if smile not in smiles2similars:
            #         smiles2similars[smile].update(
            #             all_substrates_list[j]
            #             for j in similarity_idx[1][similarity_idx[0] == smi_i]
            #         )

            # smiles2similars = defaultdict(set)

            # for smi_i, (smile, sim_row) in tqdm(
            #     enumerate(zip(all_substrates_list, smile_similarity)), desc="Retrieving all negatives", total=len(all_substrates_list)
            #     ):
            #     valid_indices = np.where((sim_row <= min_sim) | (sim_row >= max_sim))[0]
            #     smiles2similars[smile].update(all_substrates_list[j] for j in valid_indices)

            smiles2negatives = defaultdict(set)

            for smi_i, (smile, sim_row) in tqdm(
                enumerate(zip(all_substrates_list, smile_similarity)), desc="Retrieving all negatives", total=len(all_substrates_list)
                ):
                valid_indices = np.where((sim_row > min_sim) & (sim_row < max_sim))[0]
                smiles2negatives[smile].update(all_substrates_list[j] for j in valid_indices)

        self.prot_id_to_negatives = defaultdict(set)
        for sample in tqdm(dataset, desc="Sampling negatives"):
            if self.args.sample_negatives_range is not None:
                prot_id = sample["protein_id"]
                self.prot_id_to_negatives[prot_id].update(smiles2negatives[sample["smiles"]])
            else:
                self.prot_id_to_negatives.update(all_substrates_list)

        for sample in tqdm(dataset):
            prot_id = sample["protein_id"]
            self.prot_id_to_negatives[prot_id].discard(sample["smiles"])

        # prot_id_to_positives = defaultdict(set)
        # for sample in tqdm(dataset, desc="Sampling negatives"):
        #     prot_id = sample["protein_id"]
        #     prot_id_to_positives[prot_id].add(sample["smiles"])

        #     if self.args.sample_negatives_range is not None:
        #         prot_id_to_positives[prot_id].update(smiles2similars[sample["smiles"]])

        # self.prot_id_to_negatives = {k: all_substrates - v for k, v in prot_id_to_positives.items()}
        
        rowid = len(dataset)
        negatives_to_add = []
        no_negatives = 0
        # for prot_id, negatives in tqdm(self.prot_id_to_negatives.items(), desc="Processing negatives"):
        for sample in tqdm(dataset, desc="Processing negatives"):
            negatives = self.prot_id_to_negatives[sample['protein_id']]
            prot_id = sample['protein_id']
            if len(negatives) == 0:
                no_negatives += 1
                continue

            if self.args.sample_k_negatives is not None:
                if len(negatives) < self.args.sample_k_negatives:
                    new_negatives = list(negatives)
                else:
                    new_negatives = random.sample(
                        list(negatives), self.args.sample_k_negatives
                    )
            else:
                new_negatives = list(negatives)

            for rid, reactant in enumerate(new_negatives):
                hashed_molname = hashlib.md5(reactant.encode()).hexdigest()
                sample = {
                    # "ec": ec,
                    # "organism": sample.get("organism", "none"),
                    "protein_id": prot_id,
                    "uniprot_id": prot_id,
                    "rowid": prot_id + "_" + str(rowid + rid),
                    "sample_id": prot_id + "_" + str(rowid + rid),
                    "smiles": reactant,
                    # "split": uniprot2split[prot_id],
                    "y": 0,
                }
                if self.skip_sample(sample, split_group):
                    continue

                negatives_to_add.append(sample)

        print(f"[magenta] Adding {len(negatives_to_add)} negatives [/magenta]")
        print(f"[magenta] Missing any negatives for {no_negatives} ECs [/magenta]")
        print(f"[magenta] Total number of positive samples: {len(dataset)} [/magenta]")
        dataset += negatives_to_add

        return dataset

    def skip_sample(self, sample, split_group) -> bool:
        if "ec" in sample and "-" in sample["ec"]:
            return True

        if self.args.use_protein_graphs:
            structures_dir = os.path.join(self.args.protein_structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")

            if not os.path.exists(structures_dir):
                return True

        # if sequence is unknown
        if "sequence" in sample:
            if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
                return True

            if (self.args.max_protein_length is not None) and len(
                sample["sequence"]
            ) > self.args.max_protein_length:
                return True
        
        mol = sample["smiles"]
        if not (mol in self.mol2size):
            self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()
            
        if self.args.max_reactant_size is not None:
            if self.mol2size[mol] > self.args.max_reactant_size:
                return True
        
        if self.mol2size[mol] < 2:
            return True 

        return False
    
    @staticmethod   
    def add_args(parser) -> None:
        """Add class specific args"""
        super(EnzymeMapSubstrate, EnzymeMapSubstrate).add_args(parser)
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
            "--sample_negatives",
            action="store_true",
            default=False,
            help="whether to sample negative substrates",
        )
        parser.add_argument(
            "--topk_substrates_to_remove",
            type=int,
            default=None,
            help="remove common substrates",
        )
        parser.add_argument(
            "--sample_negatives_range",
            type=float,
            nargs=2,
            default=None,
            help="range of similarity to sample negatives from",
        )
        parser.add_argument(
            "--precomputed_esm_features_dir",
            type=str,
            default=None,
            help="directory with precomputed esm features for computation efficiency",
        )
        parser.add_argument(
            "--sample_k_negatives",
            type=int,
            default=None,
            help="number of negatives to sample from each ec",
        )
        parser.add_argument(
            "--sample_negatives_on_get",
            action="store_true",
            default=False,
            help="whether to sample negatives on get",
        )

    @staticmethod
    def set_args(args):
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/enzymemap_brenda2023.json"
        )


@register_object("enzymemap_reaction_graph", "dataset")
class EnzymeMapGraph(EnzymeMap):

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:

        # if removing top K
        if self.args.topk_byproducts_to_remove is not None:
            raw_byproducts = Counter([r for d in self.metadata_json for r in d["products"]]).most_common(self.args.topk_byproducts_to_remove)
            mapped_byproducts = Counter([r for d in self.metadata_json for r in d.get("mapped_products", []) ]).most_common(self.args.topk_byproducts_to_remove)
            self.common_byproducts = {s[0]:True for byproducts in [raw_byproducts, mapped_byproducts] for s in byproducts}

        dataset = []

        rkey = "mapped_reactants" if "mapped_reactants" in self.metadata_json[0] else "reactants"
        pkey = "mapped_products" if "mapped_products" in self.metadata_json[0] else "products"

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            
            reactants = sorted([s for s in reaction[rkey] if s != '[H+]'])
            products = sorted([s for s in reaction[pkey] if s != '[H+]'])
            products = [p for p in products if p not in reactants]

            if self.args.topk_byproducts_to_remove is not None:
                products = [p for p in products if p not in self.common_byproducts]

            reaction_string = "{}>>{}".format(".".join(reactants),".".join(products))
            
            bond_changes = reaction.get("bond_changes", None)
            if not bond_changes:
                try:
                    bond_changes = get_bond_changes(reaction_string)
                except:
                    continue 
            
            if self.args.create_sample_per_sequence:
                valid_uniprots = []
                for uniprot in self.ec2uniprot.get(ec, []):
                    temp_sample = {
                        "quality": reaction["quality"],
                        "reactants": reactants,
                        "products": products,
                        "ec": ec,
                        "protein_id": uniprot,
                        "sequence": self.uniprot2sequence[uniprot],
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
                        "rowid": reaction['rxnid'],
                        "sample_id": f"{uniprot}_{reaction['rxnid']}",
                        "uniprot_id": uniprot,
                        "protein_id": uniprot,
                        "bond_changes": list(bond_changes),
                        "split": reaction["split"],
                    }
                    # add reaction sample to dataset
                    if self.args.split_multiproduct_samples:
                        for product_id, p in enumerate(products):
                            psample = copy.deepcopy(sample)
                            psample["products"] = [p] 
                            psample["sample_id"] += f"_{product_id}"
                            dataset.append(psample)
                    else:
                        dataset.append(sample)

            else:
                sample = {
                        "quality": reaction["quality"],
                        "reactants": reactants,
                        "products": products,
                        "ec": ec,
                        "rowid": reaction['rxnid'],
                        "sample_id": str(reaction['rxnid']),
                        "uniprot_id": "", 
                        "protein_id": "", 
                        "sequence": "X",
                        "bond_changes": list(bond_changes),
                        "split": reaction["split"],
                    }
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
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            ec = sample["ec"]
            if self.args.create_sample_per_sequence:
                uniprot_id = sample["uniprot_id"]
                sequence = self.uniprot2sequence[uniprot_id]
            else:
                uniprot_id = ""
                sequence = ""

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
            

            reactants, atom_map2new_index = from_mapped_smiles(".".join(reactants), encode_no_edge=True)
            products, _ = from_mapped_smiles(".".join(products),  encode_no_edge=True)

            bond_changes = [(atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype) for u, v, btype in sample["bond_changes"]]
            bond_changes = [(min(x,y), max(x,y), t) for x,y,t in bond_changes]
            reactants.bond_changes = bond_changes
            sample_id = sample["sample_id"]
            rowid = sample["rowid"]
            
            # convert bond changes for all_smiles
            all_smiles_key = f"{ec}{reactants.smiles}" if self.args.create_sample_per_sequence else reactants.smiles
            all_smiles_smiles = [smiles for smiles, _ in self.reaction_to_products[all_smiles_key]]
            all_smiles_bond_changes = [destringify_sets(bc) for _, bc in self.reaction_to_products[all_smiles_key]]
            all_smiles_bond_changes = [[(atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype) for u, v, btype in changes] for changes in all_smiles_bond_changes]
            all_smiles_bond_changes = [set((min(x,y), max(x,y), t) for x,y,t in bc) for bc in all_smiles_bond_changes]
            all_smiles = [(prod_smile, prod_bc) for prod_smile, prod_bc in zip(all_smiles_smiles, all_smiles_bond_changes)]
            
            item = {
                "x": reaction,
                "reaction": reaction,
                "reactants": reactants,
                "products": products,
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "row_id": rowid,
                "smiles": products,
                "all_smiles": all_smiles,
                "quality": sample["quality"]
                # "bond_changes": stringify_sets(bond_changes)
            }

            # ecs as tensors
            split_ec = ec.split('.')
            for k,v in self.args.ec_levels.items():
                item[f"ec{k}"] = v[ '.'.join(split_ec[:int(k)]) ]

            if self.args.load_wln_cache_in_dataset:
                item["product_candidates"] = self.cache.get(rowid)

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            return item

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of an exception {e}")

@register_object("reaction_graph_inference", "dataset")
class ReactionGraphInference(AbstractDataset):
    def create_dataset(self, split_group: Literal["test"]) -> List[dict]:
       
        dataset = []

        for rowid, reaction in tqdm(
            enumerate(self.metadata_json),
            desc="Building dataset",
            total=len(self.metadata_json),
            ncols=100,
        ):
            self.mol2size = {}

            ec = reaction["ec"]
            reactants = sorted([s for s in reaction["reactants"] if s != '[H+]'])
            uniprot = reaction["uniprot_id"]
            sequence = reaction["sequence"]

            sample = {
                "reactants": reactants,
                "ec": ec,
                "protein_id": uniprot,
                "sequence": sequence,
            }
            if self.skip_sample(temp_sample, split_group):
                continue

            dataset.append(sample)

        return dataset

    def skip_sample(self, sample, split_group) -> bool:     
        # if sequence is unknown
        sequence = sample["sequence"]
        if (sequence is None) or (len(sequence) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sequence
        ) > self.args.max_protein_length:
            return True

        for mol in sample["reactants"]:
            if not (mol in self.mol2size):
                self.mol2size[mol] = rdkit.Chem.MolFromSmiles(mol).GetNumAtoms()
                
            if self.args.max_reactant_size is not None:
                if self.mol2size[mol] > self.args.max_reactant_size:
                    return True

        return False

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactants = copy.deepcopy(sample["reactants"])
            ec = sample["ec"]
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            uniprot_id = sample["uniprot_id"]
            sequence = sample["sequence"]
            reaction = ".".join(reactants)
            reactants = assign_fake_atom_maps(reaction)
            reactants, atom_map2new_index = from_mapped_smiles(".".join(reactants), encode_no_edge=True)
            reactants.bond_changes = []
            sample_id = sample["sample_id"]
        
            item = {
                "x": reaction,
                "reaction": reaction,
                "reactants": reactants,
                "sequence": sequence,
                "ec": ec,
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "row_id": sample_id,
            }

            return item

        except Exception as e:
            print(f"Could not load sample {sample['uniprot_id']} because of an exception {e}")
        
    
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args
        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(EnzymeMap, EnzymeMap).add_args(parser)
        parser.add_argument(
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )
        parser.add_argument(
            "--max_reactant_size",
            type=int,
            default=None,
            help="maximum reactant size",
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles"
        )


    @property
    def SUMMARY_STATEMENT(self) -> None:
        try:
            reactions = [".".join(d["reactants"]) for d in self.dataset]
        except:
            reactions = []
        try:
            proteins = [d["uniprot_id"] for d in self.dataset]
        except:
            proteins = []
        try:
            ecs = [d["ec"] for d in self.dataset]
        except:
            ecs = []
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        * Number of ECs: {len(set(ecs))}
        """
        return statement


@register_object("enzymemap_ec", "dataset")
class EnzymeEC(EnzymeMap):
    def create_dataset(self, split_group: Literal["test"]) -> List[dict]:
        
        uni2ec = defaultdict(set)
        for ec, uniprots in self.ec2uniprot.items():
            for u in uniprots:
                uni2ec[u].add(ec)

        ecs = [d.split('.') for d in self.ec2uniprot]
        unique_classes = sorted(list(set(".".join(ec[:self.args.ec_level+1]) for ec in ecs)))
        ec2classid = {c:i for i,c in enumerate(unique_classes)}
        self.args.num_classes = len(ec2classid)
        
        dataset = []
        for uni, ecs in uni2ec.items():
            y = torch.zeros(len(ec2classid))
            for ec in ecs: 
                y[ec2classid[".".join(ec.split('.')[:self.args.ec_level+1])]] = 1
        
            sample = {
                "sample_id": uni,
                "protein_id": uni, 
                "x": self.uniprot2sequence[uni],
                "y": y,
                "quality": 1,
                "ec": "",
                "sequence": self.uniprot2sequence[uni],
                "reactants": [],
                "products": [],
            }
            if self.skip_sample(sample, split_group):
                continue 

            dataset.append(sample)

        return dataset 

    def __getitem__(self, index):
        return self.dataset[index]
    
    def post_process(self, args):
        pass 
    
    @property
    def SUMMARY_STATEMENT(self) -> None:
        labels_per_class = Counter([d['y'].sum().item()  for d in self.dataset])
        labels_per_class = {k: labels_per_class[k] for k in sorted(labels_per_class)}
        statement = f""" 
        * Number of classes: {self.args.num_classes}
        * Number of proteins: {len(self.dataset)}
        * Number of labels per class: {labels_per_class}
        """
        return statement