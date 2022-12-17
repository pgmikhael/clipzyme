import json 
from typing import List, Literal
from nox.utils.registry import register_object, get_object
from nox.datasets.brenda import Brenda, BrendaReaction
from nox.utils.messages import METAFILE_NOTFOUND_ERR
from tqdm import tqdm
import argparse
import hashlib
from rich import print as rprint


@register_object("ecreact", "dataset")
class ECReact(BrendaReaction):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads dataset file

        Args:
            args (argparse.ArgumentParser)

        Raises:
            Exception: Unable to load
        """
        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.mcsa_biomolecules = json.load(open(args.mcsa_biomolecules_path, "r"))
        self.mcsa_curated_data = json.load(open(args.mcsa_file_path, "r"))

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []

        mcsa_data = self.load_mcsa_data(self.args)
        for reaction in tqdm(self.metadata_json[:1000]):
            
            ec = reaction["ec"]
            reactants = reaction["reactants"]
            products = reaction["products"]
            uniprotid = reaction["uniprot_id"]
            reaction_string= ".".join(reactants)+ ">>" + ".".join(products)
            sample_id = hashlib.md5(f"{uniprotid}_{reaction_string}".encode()).hexdigest()
            sequence = reaction.get("sequence", None)
            residues = self.get_uniprot_residues(mcsa_data, sequence, ec)

            sample = {
                "protein_id": uniprotid,
                "sequence": sequence,
                "reactants": reactants,
                "products": products,
                "ec": ec,
                "reaction_string":reaction_string,
                "sample_id": sample_id,
                "residues": residues["residues"],
                "residue_mask": residues["residue_mask"],
                "has_residues": residues["has_residues"],
                "residue_positions": residues["residue_positions"],
            }

            if self.skip_sample(sample, split_group):
                continue 
        
            # add sample to dataset
            dataset.append(sample)

        return dataset
    
    def skip_sample(self, sample, split_group) -> bool:
        # check right split
        if hasattr(self, "to_split"):
            if self.args.split_type == "sequence":
                if self.to_split[sample["protein_id"]] != split_group:
                    return True

            if self.args.split_type == "ec":
                ec = ".".join(sample["ec"].split(".")[: self.args.ec_level + 1])
                if self.to_split[ec] != split_group:
                    return True
            
            if self.args.split_type == "product":
                if any(self.to_split[p] != split_group for p in sample["products"]):
                    return True

        # if sequence is unknown
        if sample["sequence"] is None:
            return True

        return False
    
    @staticmethod
    def set_args(args) -> None:
        super(ECReact, ECReact).set_args(args)
        args.dataset_file_path = "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_dataset.json"


@register_object("ecreact+orgos", "dataset")
class EC_Orgo_React(ECReact):
    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        super(EC_Orgo_React,EC_Orgo_React).load_dataset(args)
        self.orgo_reactions = get_object("chemical_reactions", "dataset")(args)
        
    
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = super(EC_Orgo_React,EC_Orgo_React).create_dataset(split_group)

        return self.orgo_reactions.dataset + dataset 
