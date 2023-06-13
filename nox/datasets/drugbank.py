
from typing import List, Literal
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict, Counter
import argparse
import warnings
from rich import print as rprint
from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
import rdkit 
import hashlib
from nox.utils.pyg import from_smiles

@register_object("drugbank_reactions", "dataset")
class DrugBankReactions(AbstractDataset):
    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        # TODO: figure out how to just to test
        # all splits will be the same (all the data)
        return dataset

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        for idx, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            keys = ['num_reactions_in_pathway' 'substrate_id', 'substrate', 'product']# , 'uniprot_ids', 'sequences']
            sample = {k: row[k] for k in keys}

            for i in range(len(sample['uniprot_ids'])):
                new_sample = sample.copy()
                new_sample['uniprot_id'] = sample['uniprot_ids'][i]
                new_sample['sequence'] = sample['sequences'][i] 
                new_sample["substrate"] = sample["substrate"]
                new_sample["product"] = sample["product"]

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
            return sample

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")
    
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
    
    @staticmethod
    def set_args(args) -> None:
        super(DrugBankReactions, DrugBankReactions).set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions.json"
        )



@register_object("drugbank_reactions_graph", "dataset")
class DrugBankReactionsGraph(DrugBankReactions):
    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            sample['smiles'] = from_smiles(sample["substrate"])
            return sample

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")
   