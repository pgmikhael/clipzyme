from typing import List, Literal
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict, Counter
import argparse
import warnings
import copy
import rdkit
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.pyg import from_mapped_smiles
from nox.utils.smiles import assign_dummy_atom_maps
from rdkit import Chem


@register_object("drugbank_reactions", "dataset")
class DrugBankReactions(AbstractDataset):
    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        assert (self.args.test) and (not self.args.train)
        # all splits will be the same (all the data)
        return processed_dataset

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []

        for idx, row in tqdm(
            enumerate(self.metadata_json),
            total=len(self.metadata_json),
            desc="Creating dataset",
            ncols=50,
        ):
            keys = [
                "num_reactions_in_pathway",
                "substrate_id",
                "substrate",
                "product",
                "uniprot_ids",
                "sequences",
                "co_reactants",
            ]
            sample = {k: row[k] for k in keys}

            for i in range(len(sample["uniprot_ids"])):
                new_sample = sample.copy()
                new_sample["uniprot_id"] = sample["uniprot_ids"][i]
                new_sample["sequence"] = sample["sequences"][i]
                new_sample["reactants"] = (
                    [sample["substrate"]] + sample["co_reactants"][0]
                    if self.args.use_co_reactants
                    else [sample["substrate"]]
                )
                new_sample["substrate"] = sample["substrate"]
                new_sample["co_reactants"] = sample["co_reactants"]
                new_sample["substrate_id"] = sample["substrate_id"]
                new_sample["product"] = sample["product"]
                new_sample["sample_id"] = f"row{idx}_seq{i}"

                if self.skip_sample(new_sample):
                    continue

                dataset.append(new_sample)

        return dataset

    def skip_sample(self, sample) -> bool:
        # if sequence is unknown
        if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        for key in ["substrate", "product"]:
            try:
                mol_size = rdkit.Chem.MolFromSmiles(sample[key]).GetNumAtoms()
            except:
                return True

            # skip graphs of size 1
            if mol_size <= 1:
                return True

        if (self.args.use_co_reactants) and (len(sample["co_reactants"]) == 0):
            return True

        return False

    @property
    def SUMMARY_STATEMENT(self) -> None:
        proteins = [d["uniprot_id"] for d in self.dataset]
        substrates = [d["substrate"] for d in self.dataset]
        products = [d["product"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of substrates: {len(set(substrates))}
        * Number of products: {len(set(products))}
        * Number of proteins: {len(set(proteins))}
        """
        return statement

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["product"]
            )
            sequence = sample["sequence"]
            uniprot_id = sample["uniprot_id"]
            sample_id = sample["sample_id"]
            reactants = ".".join(reactants)
            all_smiles = list(
                self.reaction_to_products[f"{uniprot_id}_{sample['substrate']}"]
            )

            # remove stereochemistry
            if self.args.remove_stereochemistry_from_product:
                products_mol = Chem.MolFromSmiles(products)
                Chem.RemoveStereochemistry(products_mol)
                products = Chem.MolToSmiles(products_mol)

            if self.args.remove_stereochemistry:
                reactants_mol = Chem.MolFromSmiles(reactants)
                products_mol = Chem.MolFromSmiles(products)
                Chem.RemoveStereochemistry(reactants_mol)
                Chem.RemoveStereochemistry(products_mol)
                reactants = Chem.MolToSmiles(reactants_mol)
                products = Chem.MolToSmiles(products_mol)

            if self.args.use_graph_version:
                reactants = assign_dummy_atom_maps(reactants)
                reactants, atom_map2new_index = from_mapped_smiles(
                    reactants, encode_no_edge=True
                )
                reactants.bond_changes = []
                products = assign_dummy_atom_maps(products)
                products, _ = from_mapped_smiles(products, encode_no_edge=True)
                all_smiles = [(s, []) for s in all_smiles]

                drug_mask = torch.zeros(reactants.x.shape[0])
                for i, a in enumerate(
                    Chem.MolFromSmiles(sample["reactants"][0]).GetAtoms()
                ):
                    drug_mask[atom_map2new_index[i + 1]] = 1
                reactants.mask = drug_mask

            item = {
                "reaction": f"{reactants}>>{products}",
                "reactants": reactants,
                "products": products,
                "sequence": sequence,
                "protein_id": uniprot_id,
                "sample_id": sample_id,
                "smiles": products,
                "all_smiles": all_smiles,
            }
            return item

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")

    def post_process(self, args):
        # add all possible products
        reaction_to_products = defaultdict(set)
        for sample in self.dataset:
            reaction_to_products[f"{sample['uniprot_id']}_{sample['substrate']}"].add(
                sample["product"]
            )
        self.reaction_to_products = reaction_to_products

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(DrugBankReactions, DrugBankReactions).add_args(parser)
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
            "--max_product_size",
            type=int,
            default=None,
            help="maximum product size",
        )
        parser.add_argument(
            "--use_graph_version",
            action="store_true",
            default=False,
            help="use graph structure for inputs",
        )
        parser.add_argument(
            "--use_co_reactants",
            action="store_true",
            default=False,
            help="use reactants from generic reaction",
        )
        parser.add_argument(
            "--remove_stereochemistry",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles",
        )
        parser.add_argument(
            "--remove_stereochemistry_from_product",
            action="store_true",
            default=False,
            help="remove stereochemistry from smiles",
        )
        parser.add_argument(
            "--skip_cytochromes",
            action="store_true",
            default=False,
            help="skip cytochrome proteins",
        )

    @staticmethod
    def set_args(args) -> None:
        super(DrugBankReactions, DrugBankReactions).set_args(args)
        args.dataset_file_path = "/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions_with_reactants_itamarupdate.json"


@register_object("drugbank_reactions_uniprot", "dataset")
class DrugBankReactionsUniProt(DrugBankReactions):
    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        assert (self.args.test) and (not self.args.train)
        # all splits will be the same (all the data)
        return processed_dataset

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []

        for idx, row in tqdm(
            enumerate(self.metadata_json),
            total=len(self.metadata_json),
            desc="Creating dataset",
            ncols=50,
        ):
            keys = [
                "num_reactions_in_pathway",
                "substrate_id",
                "substrate",
                "product",
                "uniprot_ids",
                "sequences",
                "coreactant_smiles",
                "protein_names",
            ]
            sample = {k: row[k] for k in keys}

            for i in range(len(sample["uniprot_ids"])):
                new_sample = sample.copy()
                new_sample["uniprot_id"] = sample["uniprot_ids"][i]
                new_sample["protein_name"] = sample["protein_names"][i]
                new_sample["sequence"] = sample["sequences"][i]
                new_sample["reactants"] = (
                    [sample["substrate"]] + sample["coreactant_smiles"][i]
                    if self.args.use_co_reactants
                    else [sample["substrate"]]
                )
                new_sample["substrate"] = sample["substrate"]
                new_sample["co_reactants"] = sample["coreactant_smiles"][i]
                new_sample["substrate_id"] = sample["substrate_id"]
                new_sample["product"] = sample["product"]
                new_sample["sample_id"] = f"row{idx}_seq{i}"

                if self.skip_sample(new_sample):
                    continue

                dataset.append(new_sample)

        return dataset

    def skip_sample(self, sample) -> bool:
        # if sequence is unknown
        if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        for key in ["substrate", "product"]:
            try:
                mol_size = rdkit.Chem.MolFromSmiles(sample[key]).GetNumAtoms()
            except:
                return True

            # skip graphs of size 1
            if mol_size <= 1:
                return True

        if (self.args.use_co_reactants) and (len(sample["coreactant_smiles"]) == 0):
            return True

        if (self.args.skip_cytochromes) and (
            "cytochrome" in sample["protein_name"].lower()
        ):
            return True
        return False

    @staticmethod
    def set_args(args) -> None:
        super(DrugBankReactions, DrugBankReactions).set_args(args)
        if args.dataset_file_path is None:
            args.dataset_file_path = "/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions_with_uniprot_cofactor.json"
