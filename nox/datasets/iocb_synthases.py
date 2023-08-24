
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
from nox.utils.pyg import from_smiles, from_mapped_smiles
from nox.utils.smiles import get_rdkit_feature, remove_atom_maps, assign_dummy_atom_maps
from nox.utils.wln_processing import get_bond_changes
import copy 

@register_object("iocb", "dataset")
class IOCB(AbstractDataset):

    def assign_splits(self, dataset, split_probs, seed=0) -> None:
        """
        Assigns each sample to a split group based on split_probs
        """
        # get all samples
        rprint("Generating dataset in order to assign splits...")

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
            
        else:
            raise ValueError(f"Split {self.args.split_type} type not supported. Must be one of [fold, fixed]")

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []
        for sample in processed_dataset:
            if self.to_split[sample['split']] != split_group:
                    continue

            dataset.append(sample)

        return dataset

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        
        if self.args.reaction_side == "reactant": 
            # get column name of split
            split_key = "substrate_stratified_phylogeny_based_split" 
            # get column names for smiles
            smiles_key = "SMILES_substrate_canonical_no_stereo" if self.args.use_stereo else "SMILES of substrate"
            smiles_id_key = "Substrate ChEBI ID"
            smiles_name_key = "Substrate (including stereochemistry)"

        elif self.args.reaction_side == "product":
            # get column name of split
            split_key = "product_stratified_phylogeny_based_split" 
            # get column names for smiles
            smiles_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 
            smiles_id_key = "Product ChEBI ID"
            smiles_name_key = "Name of product"

        # if removing top K
        if self.args.topk_substrates_to_remove is not None:
            substrates = Counter([r for d in self.metadata_json for r in d[smiles_key]]).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = [s[0] for s in substrates]

        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprotid = row["Uniprot ID"]
            molname = row[smiles_name_key]
            sample = {
                    "protein_id": uniprotid,
                    "sequence": row["Amino acid sequence"],
                    "sequence_name": row["Name"],
                    "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "smiles": row[smiles_key],
                    "smiles_name": row[smiles_name_key],
                    "smiles_chebi_id": row[smiles_id_key],
                    "sample_id": f"{uniprotid}_{molname}_{rowid}",
                    "species": row["Species"],
                    "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row[split_key],
                }
            
            if self.skip_sample(sample):
                continue 
            
            dataset.append(sample)

        return dataset 

    

    def skip_sample(self, sample) -> bool:
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

    @property
    def SUMMARY_STATEMENT(self) -> None:
        proteins = [d["protein_id"] for d in self.dataset]
        substrates = [d["smiles_chebi_id"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of substrates: {len(set(substrates))}
        * Number of proteins: {len(set(proteins))}
        """
        return statement
    
    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            if self.args.use_graph_version:
                reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(sample["products"])
                sequence = sample["sequence"]
                uniprot_id = sample['protein_id']
                sample_id = sample["sample_id"]
                

                reactants = assign_dummy_atom_maps(reactants)
                reactants, atom_map2new_index = from_mapped_smiles(reactants, encode_no_edge=True)
                reactants.bond_changes = []
                products = assign_dummy_atom_maps(products)
                products, _ = from_mapped_smiles(products,  encode_no_edge=True)
                all_smiles = [(sample["products"],[])]
                item = {
                    "reaction": "{}>>{}".format(sample["reactants"], sample["products"]),
                    "reactants": reactants,
                    "products": products,
                    "sequence": sequence,
                    "protein_id": uniprot_id,
                    "sample_id": sample_id,
                    "smiles": products,
                    "all_smiles": all_smiles,
                }
                return item 


            return sample

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")
    
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(IOCB, IOCB).add_args(parser)
        parser.add_argument(
            "--precomputed_esm_features_dir",
            type=str,
            default=None,
            help="directory with precomputed esm features for computation efficiency",
        )
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
            "--reaction_side",
            type=str,
            default="reactant",
            choices=["reactant", "product"],
            help="choice of reactant or product to use as target",
        )
        parser.add_argument(
            "--use_stereo",
            action="store_true",
            default=False,
            help="use stereochemistry version of smiles",
        )
        parser.add_argument(
            "--topk_substrates_to_remove",
            type=int,
            default=None,
            help="remove common substrates",
        )
        parser.add_argument(
            "--use_graph_version",
            action="store_true",
            default=False,
            help="use graph structure for inputs",
        )
        parser.add_argument(
            "--use_atom_maps",
            action="store_true",
            default=False,
            help="use atom maps of reaction",
        )
    
    @staticmethod
    def set_args(args) -> None:
        super(IOCB, IOCB).set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/IOCB/IOCB_TPS-1April2023_verified_non_minor_tps_with_neg.json"
        )

@register_object("iocb_classification", "dataset")
class IOCBClassification(IOCB):

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        
        if self.args.reaction_side == "reactant": 
            # get column name of split
            split_key = "substrate_stratified_phylogeny_based_split" 
            # get column names for smiles
            smiles_key = "SMILES_substrate_canonical_no_stereo" if self.args.use_stereo else "SMILES of substrate"
            smiles_id_key = "Substrate ChEBI ID"
            smiles_name_key = "Substrate (including stereochemistry)"

        elif self.args.reaction_side == "product":
            # get column name of split
            split_key = "product_stratified_phylogeny_based_split" 
            # get column names for smiles
            smiles_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 
            smiles_id_key = "Product ChEBI ID"
            smiles_name_key = "Name of product"

        # if removing top K
        if self.args.topk_substrates_to_remove is not None:
            substrates = Counter([r for d in self.metadata_json for r in d[smiles_key]]).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = [s[0] for s in substrates]

        uni2smiles = defaultdict(list)
        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprotid = row["Uniprot ID"]
            molname = row[smiles_name_key]
            sample = {
                    "protein_id": uniprotid,
                    "sequence": row["Amino acid sequence"],
                    "sequence_name": row["Name"],
                    "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "smiles": row[smiles_key],
                    "smiles_name": row[smiles_name_key],
                    "smiles_chebi_id": row[smiles_id_key],
                    "sample_id": f"{uniprotid}_{molname}_{rowid}",
                    "species": row["Species"],
                    "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row[split_key],
                }
            
            if self.skip_sample(sample):
                continue 
            
            uni2smiles[uniprotid].append(sample)
            
        for uni, samples in uni2smiles.items():
            assert all(s['split'] == samples[0]['split'] for s in samples)

            sample = {
                "protein_id": uni,
                "x": row["Amino acid sequence"],
                "sequence_name": row["Name"],
                "protein_type": row["Type (mono, sesq, di, \u2026)"],
                "sample_id": uni,
                "species": samples[0]["species"],
                "kingdom": samples[0]['kingdom'],
                "split": samples[0]['split'],
                "smiles": [s["smiles"] for s in samples]
            }

            dataset.append(sample)

        return dataset 


    def post_process(self, args):
        train_dataset = self.get_split_group_dataset(self.dataset, "train")
        smiles = sorted(set(s for d in train_dataset for s in d["smiles"]))
        smiles2class = {smi: i for i, smi in enumerate(smiles)}
        
        for sample in self.dataset:
            y = np.zeros(len(smiles2class))
            for s in sample['smiles']:
                if s in smiles2class:
                    y[smiles2class[s]] = 1
            sample['smiles'] = '_'.join(sample['smiles'])
            sample["y"] = y
        
        args.num_classes = len(smiles2class)

        return self.dataset
    
    @property
    def SUMMARY_STATEMENT(self) -> None:
        proteins = [d["protein_id"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of substrates: {self.args.num_classes}
        * Number of proteins: {len(set(proteins))}
        """
        return statement

@register_object("iocb_products", "dataset")
class IOCBProducts(IOCBClassification):

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        # get column name of split
        split_key = "product_phylo_split_simple" 
        # get column names for smiles
        smiles_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 
        smiles_id_key = "Product ChEBI ID"
        smiles_name_key = "Name of product"

        uni2smiles = defaultdict(list)
        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprotid = row["Uniprot ID"]
            molname = row[smiles_name_key]
            sample = {
                    "protein_id": uniprotid,
                    "sequence": row["Amino acid sequence"],
                    "sequence_name": row["Name"],
                    "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "smiles": row[smiles_key],
                    "smiles_name": row[smiles_name_key],
                    "smiles_chebi_id": row[smiles_id_key],
                    "sample_id": f"{uniprotid}_{molname}_{rowid}",
                    "species": row["Species"],
                    "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row[split_key],
                }
            
            if self.skip_sample(sample):
                continue 
            
            uni2smiles[uniprotid].append(sample)
            
        for uni, samples in uni2smiles.items():
            assert all(s['split'] == samples[0]['split'] for s in samples)

            sample = {
                "protein_id": uni,
                "x": row["Amino acid sequence"],
                "sequence_name": row["Name"],
                "protein_type": row["Type (mono, sesq, di, \u2026)"],
                "sample_id": uni,
                "species": samples[0]["species"],
                "kingdom": samples[0]['kingdom'],
                "split": samples[0]['split'],
                "smiles": [s["smiles"] for s in samples]
            }

            dataset.append(sample)

        return dataset 

    @staticmethod
    def set_args(args) -> None:
        super(IOCB, IOCB).set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/IOCB/iocb_products.json"
        )
        args.reaction_side = "product"

@register_object("iocb_reactions", "dataset")
class IOCBReactions(IOCB):

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        self.mol2size = {}

        dataset = []
        
        # get column name of split
        split_key = "product_stratified_phylogeny_based_split"
        
        # get column names for smiles
        substrate_key = "SMILES_substrate_canonical_no_stereo" if self.args.use_stereo else "SMILES of substrate"
        substrate_id_key = "Substrate ChEBI ID"
        substrate_name_key = "Substrate (including stereochemistry)"

        product_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 
        product_id_key = "Product ChEBI ID"
        product_name_key = "Name of product"

        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprotid = row["Uniprot ID"]
        
            sample = {
                    "protein_id": uniprotid,
                    "sequence": row["Amino acid sequence"],
                    "sequence_name": row["Name"],
                    "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "reactants": [row[substrate_key]],
                    "reactants_name": row[substrate_name_key],
                    "reactants_chebi_id": row[substrate_id_key],
                    "products": [row[product_key]],
                    "products_name": row[product_name_key],
                    "products_chebi_id": row[product_id_key],
                    "sample_id": f"{uniprotid}_{rowid}",
                    "species": row["Species"],
                    "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row[split_key],
                    "all_smiles": [row[product_key]],
                    "smiles": row[product_key],
                }
            
            if self.skip_sample(sample):
                continue 
            
            sample["reactants"] = ".".join(sample["reactants"])
            sample["products"] = ".".join(sample["products"])

            if self.args.use_atom_maps:
                if (row["rxn_smiles"] == "") or (row["bond_changes"] == ""):
                    continue 

                reaction_smiles = row["rxn_smiles"]
                reactants, products = reaction_smiles.split(">>")
                sample["bond_changes"] = row["bond_changes"]
                sample["reactants"] = reactants
                sample["products"] = products
            
            dataset.append(sample)

        return dataset 
    
    def skip_sample(self, sample) -> bool:
        # if sequence is unknown
        if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
            return True

        if (self.args.max_protein_length is not None) and len(
            sample["sequence"]
        ) > self.args.max_protein_length:
            return True

        if "Unknown" in sample["reactants"] + sample["products"]:
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

        return False
    
    @property
    def SUMMARY_STATEMENT(self) -> None:
        proteins = [d["protein_id"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of proteins: {len(set(proteins))}
        """
        return statement
    
    @staticmethod
    def set_args(args) -> None:
        IOCB.set_args(args)
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Enzymes/IOCB/TPS-IOCB-with-mapped-rxn-fixedv1.json"
        )



@register_object("iocb_with_negatives", "dataset")
class IOCBNegatives(IOCB):
    def __init__(self, args, split_group):
        super(IOCBNegatives, IOCBNegatives).__init__(self, args, split_group)

    def assign_splits(self, dataset, split_probs, seed=0) -> None:
        pass

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []
        for sample in processed_dataset:
            if sample['split'] != split_group:
                continue

            dataset.append(sample)

        return dataset

    def create_dataset(self, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []

        
        if self.args.reaction_side == "reactant": 
            smiles_key = "substrate_SMILES" if self.args.use_stereo else "substrate_SMILES_canonical_without_stereochem"

        elif self.args.reaction_side == "product":
            smiles_key = "SMILES of product (including stereochemistry)" if self.args.use_stereo else "SMILES_product_canonical_no_stereo" 

        # if removing top K
        if self.args.topk_substrates_to_remove is not None:
            substrates = Counter([r for d in self.metadata_json for r in d[smiles_key]]).most_common(self.args.topk_substrates_to_remove)
            self.common_substrates = [s[0] for s in substrates]

        for rowid, row in tqdm(enumerate(self.metadata_json), total = len(self.metadata_json), desc="Creating dataset", ncols=50):
            
            uniprot_id = row["uniprot_id"]
            molname = hashlib.md5(row[smiles_key].encode()).hexdigest()
            if "*" in row['enzyme_seq']:
                print(f"removed * from sequence: {row['enzyme_seq']}")
                row['enzyme_seq'] = row['enzyme_seq'].replace("*", "")
                
            sample = {
                    "protein_id": uniprot_id,
                    "sequence": row["enzyme_seq"],
                    # "sequence_name": row["Name"],
                    # "protein_type": row["Type (mono, sesq, di, \u2026)"],
                    "smiles": row[smiles_key],
                    # "smiles_name": row[smiles_name_key],
                    # "smiles_chebi_id": row[smiles_id_key],
                    "sample_id": f"{uniprot_id}_{molname}_{rowid}",
                    # "species": row["Species"],
                    # "kingdom": row["Kingdom (plant, fungi, bacteria)"],
                    "split": row["split"],
                    "y": int(row['is_substrate_accepted'])
                }
            
            if self.skip_sample(sample):
                continue 
            
            dataset.append(sample)

        return dataset 

    

    def skip_sample(self, sample) -> bool:
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

        if sample['sequence'] == "to be added":
            return True

        return False

    @property
    def SUMMARY_STATEMENT(self) -> None:
        proteins = [d["protein_id"] for d in self.dataset]
        substrates = [d["smiles"] for d in self.dataset]
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of substrates: {len(set(substrates))}
        * Number of proteins: {len(set(proteins))}
        """
        return statement
    
    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reactant = from_smiles(sample["smiles"])

            reactant.sample_id = sample["sample_id"]
            reactant.sequence = sample["sequence"]
            reactant.y = sample["y"]
            reactant.uniprot_id = sample["protein_id"]

            return reactant
        
        except Exception as e:
            print(f"Could not load sample: {sample['sample_id']}, exception {e}")
            
            
    
    @staticmethod
    def set_args(args) -> None:
        super(IOCB, IOCB).set_args(args)
        args.dataset_file_path = (
            '/Mounts/rbg-storage1/datasets/Enzymes/IOCB/IOCB_TPS_substrate_enzyme_pairs_major_product_split.json'
        )