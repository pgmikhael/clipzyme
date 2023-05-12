from typing import List, Literal
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from tqdm import tqdm
import argparse
import pickle
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
import warnings
import copy, os
import numpy as np
import random
from collections import defaultdict, Counter
import rdkit
import torch


@register_object("enzymemap_reactions", "dataset")
class EnzymeMap(AbstractDataset):
    def __init__(self, args, split_group) -> None:
        super(EnzymeMap, EnzymeMap).__init__(self, args, split_group)
        self.metadata_json = None  # overwrite for memory

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
            if self.mol2size[mol] < 2:
                return True 


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

            sample_id = sample["rowid"]
            item = {
                "reaction": reaction,
                "reactants": ".".join(reactants),
                "products": ".".join(products),
                "sequence": sequence,
                "ec": ec,
                "organism": sample.get("organism", "none"),
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "smiles": ".".join(products),
                "all_smiles": list(
                    self.reaction_to_products[f"{ec}{'.'.join(sorted(reactants))}"]
                ),
            }

            if self.args.use_pesto_scores:
                scores = self.get_pesto_scores(item["protein_id"])
                if (scores is None) or (scores.shape[0] != len(item["sequence"])):
                    # make all zeros of length sequence
                    scores = torch.zeros(len(item["sequence"]))
                item["sequence_annotation"] = scores

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")

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
            reaction_to_products[
                f"{sample['ec']}{'.'.join(sample['reactants'])}"
            ].update(sample["products"])
        self.reaction_to_products = reaction_to_products

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

    @property
    def SUMMARY_STATEMENT(self) -> None:
        reactions = [
            "{}>>{}".format(".".join(d["reactants"]), ".".join(d["products"]))
            for d in self.dataset
        ]
        proteins = [d["uniprot_id"] for d in self.dataset]
        ecs = [d["ec"] for d in self.dataset]
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