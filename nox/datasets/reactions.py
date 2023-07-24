import argparse
from typing import List, Literal
from nox.utils.registry import register_object, get_object
from nox.datasets.abstract import AbstractDataset
import warnings
from tqdm import tqdm
import random
from rxn.chemutils.smiles_randomization import randomize_smiles_rotated
from nox.utils.smiles import standardize_reaction
import copy
import numpy as np
from collections import defaultdict
from nox.utils.pyg import  from_mapped_smiles
from nox.utils.wln_processing import get_bond_changes

@register_object("chemical_reactions", "dataset")
class ChemRXN(AbstractDataset):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        reaction2products = defaultdict(set)
        for rxn_dict in tqdm(self.metadata_json):
            reaction2products[f"{'.'.join(rxn_dict['reactants'])}"].update(rxn_dict['products'])



        dataset = []
        for rxn_dict in tqdm(self.metadata_json):
            dataset.append(
                {
                    "x": rxn_dict["reaction"],
                    "sample_id": rxn_dict["rxnid"],
                    "split": rxn_dict["split"],
                    "reactants": rxn_dict['reactants'],
                    "products": rxn_dict['products'],
                    "protein_id": "spontaneous",
                    "sequence": "<pad>", # esm pad token
                    "all_smiles": reaction2products['.'.join(rxn_dict['reactants'])]
                }
            )
        return dataset

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ):
        return [d for d in processed_dataset if d["split"] == split_group]

    def __getitem__(self, index):
        try:
            sample = copy.deepcopy(self.dataset[index])
            item = {}

            reaction = sample["x"]
            reactants, products = reaction.split(">>")
            reactants, products = reactants.split("."), products.split(".")

            # augment: permute and/or randomize
            if self.args.randomize_order_in_reaction and not (
                self.split_group == "test"
            ):
                random.shuffle(reactants)
                random.shuffle(products)
                reaction = "{}>>{}".format(".".join(reactants), ".".join(products))

            if self.args.use_random_smiles_representation and not (
                self.split_group == "test"
            ):
                try:
                    reactants = [randomize_smiles_rotated(s) for s in reactants]
                    products = [randomize_smiles_rotated(s) for s in products]
                    reaction = "{}>>{}".format(".".join(reactants), ".".join(products))
                except:
                    pass

            item["x"] = reaction
            item["reactants"] = ".".join(reactants)
            item["products"] = ".".join(products)
            item["sample_id"] = sample["sample_id"]
            item["sequence"] = sample["sequence"]

            if standardize_reaction(reaction) == ">>":
                return

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {item['sample_id']}")

    @staticmethod
    def add_args(parser) -> None:
        super(ChemRXN, ChemRXN).add_args(parser)
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

    @property
    def SUMMARY_STATEMENT(self) -> str:
        """
        Prints summary statement with dataset stats
        """

        reactions = [d["x"].split(">>") for d in self.dataset]
        num_reactions = len(reactions)
        median_src = np.median([len(v[0]) for v in reactions])
        median_tgt = np.median([len(v[1]) for v in reactions])

        summary = f"""
        * Number of reactions: {num_reactions}
        * Median source length: {median_src}
        * Medin target length: {median_tgt}
        """
        return summary

@register_object("chemical_reactions_graph", "dataset")
class ChemRXNGraph(ChemRXN):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:

        dataset = []
        for rxn_dict in tqdm(self.metadata_json):
            reaction = rxn_dict["reaction"]
            reactants, products = rxn_dict["reaction"].split(">>")
            reactants = reactants.split(".")
            products = products.split(".")

            if "bond_changes" not in rxn_dict:
                bond_changes = get_bond_changes(reaction)
            else:
                bond_changes = rxn_dict["bond_changes"]

            dataset.append(
                {
                    "reaction": reaction,
                    "reactants": reactants,
                    "products": products,
                    "sample_id": f"{rxn_dict['split']}_{rxn_dict['rxnid']}",
                    "split": rxn_dict["split"],
                    "bond_changes": list(bond_changes),
                }
            )
        return dataset


    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reaction = sample["reaction"]
            reactants, products = copy.deepcopy(sample["reactants"]), copy.deepcopy(
                sample["products"]
            )

            reactants, atom_map2new_index = from_mapped_smiles(".".join(reactants), encode_no_edge=True)
            products, _ = from_mapped_smiles(".".join(products),  encode_no_edge=True)

            bond_changes = [(atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype) for u, v, btype in sample["bond_changes"]]
            bond_changes = [(min(x,y), max(x,y), t) for x,y,t in bond_changes]
            reactants.bond_changes = bond_changes
            sample_id = sample["sample_id"]
            
            item = {
                "reaction": reaction,
                "reactants": reactants,
                "products": products,
                "sample_id": sample_id,
                "all_smiles": []
            }
            return item

        except Exception as e:
            print(f"Could not load sample {sample['sample_id']} because of an exception {e}")

    def get_split_group_dataset(self, processed_dataset, split_group: Literal["train", "dev", "test"]) -> List[dict]:
        dataset = []
        for sample in processed_dataset:
            if sample["split"] != split_group:
                continue
            dataset.append(sample)
        return dataset
    
    @property
    def SUMMARY_STATEMENT(self) -> None:
        reactions = [d["reaction"] for d in self.dataset]
        statement = f""" 
        * Number of reactions: {len(set(reactions))}
        """
        return statement


#################
# For diffusion #
#################
from nox.datasets.ecreact_graph import DatasetInfo
from nox.utils.digress.extra_features import ExtraFeatures
from nox.utils.pyg import from_smiles, x_map, e_map
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import random 

@register_object("chemical_reactions_digress", "dataset")
class ChemRXNDigress(ChemRXN):
    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        reaction2products = defaultdict(set)
        for rxn_dict in tqdm(self.metadata_json):
            reaction2products[rxn_dict['reaction'].split(">>")[0]].update(rxn_dict['reaction'].split(">>")[-1])

        dataset = []
        for rxn_dict in tqdm(self.metadata_json):

            dataset.append(
                {
                    "x": rxn_dict["reaction"],
                    "sample_id": rxn_dict["rxnid"],
                    "split": rxn_dict["split"],
                    "protein_id": "spontaneous",
                    "sequence": "<pad>", # esm pad token
                    "all_smiles": reaction2products[rxn_dict['reaction'].split(">>")[0]],
                }
            )
        return dataset
    
    def post_process(self, args):
        split_group = self.split_group

        # get data info (input / output dimensions)
        train_dataset = self.get_split_group_dataset(self.dataset, "train")

        smiles = set()
        for d in train_dataset:
            smiles.update(d["x"].split(">>"))
        smiles = list(smiles)

        data_info = DatasetInfo(random.sample(smiles,1000), args)

        extra_features = ExtraFeatures(args.extra_features_type, dataset_info=data_info)

        example_batch = [from_smiles(smiles[0]), from_smiles(smiles[1])]
        example_batch = Batch.from_data_list(example_batch, None, None)

        data_info.compute_input_output_dims(
            example_batch=example_batch,
            extra_features=extra_features,
            domain_features=None,
        )

        data_info.input_dims["E"] += len(e_map["bond_type"]) # edge are combination of reactants and to-be-noised product edges

        args.dataset_statistics = data_info
        args.extra_features = extra_features
        args.domain_features = None

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            reaction = sample["x"]

            reactants, products = reaction.split(">>")
            reactants = from_smiles(reactants, return_atom_number=True)
            products = from_smiles(products, return_atom_number=True)

            # get products edge indices in terms of reactant nodes (i.e., mapped)
            reactant_atomnumber2id = {i:j for i,j in zip(reactants.atom_map_number, reactants.x_ids)}
            productid2reactantid = {i: reactant_atomnumber2id[j] for i,j in zip(products.x_ids, products.atom_map_number)}
            
            reactants.product_edge_index = products.edge_index.clone()
            reactants.product_edge_index.apply_(productid2reactantid.get) 
            reactants.product_edge_attr = products.edge_attr.clone()

            # first feature is atomic number
            reactants.x = F.one_hot(reactants.x[:, 0], len(x_map["atomic_num"])).to(
                torch.float
            )
            # first feature is bond type
            reactants.edge_attr = F.one_hot(
                reactants.edge_attr[:, 0], len(e_map["bond_type"])
            ).to(torch.float)
            reactants.product_edge_attr = F.one_hot(
                reactants.product_edge_attr[:, 0], len(e_map["bond_type"])
            ).to(torch.float)
            
            reactants.y = torch.zeros((1, 0), dtype=torch.float)

            item = {
                "reaction": reaction,
                "reactants": reactants,
                "sample_id": f"{sample['split']}_{sample['sample_id']}",
            }

            return item

        except Exception:
            warnings.warn(f"Could not load sample: {sample['sample_id']}")

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(ChemRXNDigress, ChemRXNDigress).add_args(parser)
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