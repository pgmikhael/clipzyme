import json 
from typing import List, Literal
from nox.utils.registry import register_object, get_object
from nox.datasets.brenda import Brenda, BrendaReaction
from nox.utils.messages import METAFILE_NOTFOUND_ERR
from tqdm import tqdm
import argparse
import hashlib
from rich import print as rprint
import pickle
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
import warnings
from frozendict import frozendict
import copy

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
        
        #self.ec2uniprot = pickle.load(open("/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_ec2uniprot.p", "rb"))
        self.uniprot2sequence = pickle.load(open("/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_proteins.p", "rb"))

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []

        mcsa_data = self.load_mcsa_data(self.args)
        for reaction in tqdm(self.metadata_json):
            
            ec = reaction["ec"]
            reactants = reaction["reactants"]
            products = reaction["products"]
            reaction_string = ".".join(reactants)+ ">>" + ".".join(products)
            
            #for uniprotid in self.ec2uniprot.get(ec, []):
            uniprotid=reaction["uniprot_id"]
            sample_id = hashlib.md5(f"{uniprotid}_{reaction_string}".encode()).hexdigest()
            sequence = self.uniprot2sequence[uniprotid]
            residues = self.get_uniprot_residues(mcsa_data, sequence, ec)

            sample = {
                "protein_id": uniprotid,
                "sequence": sequence,
                "reactants": reactants,
                "products": products,
                "ec": ec,
                "reaction_string":reaction_string,
                "sample_id": sample_id,
            }

            if hasattr(self, "to_split"):
                sample.update({
                    "residues": residues["residues"],
                    "residue_mask": residues["residue_mask"],
                    "has_residues": residues["has_residues"],
                    "residue_positions": residues["residue_positions"],
                })

            if self.skip_sample(sample, split_group):
                continue 

            if self.args.split_type != "random":
                del sample["sequence"]
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
        args.dataset_file_path = "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_dataset_lite_v2.json"

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:

            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            # incorporate sequence residues if known
            if self.args.use_residues_in_reaction:
                residues = sample["residues"]
                reactants.extend(residues)
                products.extend(residues)

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

            item = {
                "reaction": reaction,
                "reactants": ".".join(reactants),
                "products": ".".join(products),
                "sequence": self.uniprot2sequence[sample["protein_id"]],
                "ec": sample["ec"],
                "organism": sample.get("organism", "none"),
                "protein_id": sample["protein_id"],
                "sample_id": sample["sample_id"],
                "residues": ".".join(sample["residues"]),
                "has_residues": sample["has_residues"],
                "residue_positions": ".".join(
                    [str(s) for s in sample["residue_positions"]]
                ),
            }

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")

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
