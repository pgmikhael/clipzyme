import argparse
import numpy as np
import torch
import warnings
import os
from typing import List, Any, Sequence
from tqdm import tqdm
import json , pickle
from rdkit import Chem
from rich import print 

import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.data.separate import separate

from nox.utils.registry import register_object, md5
from nox.datasets.abstract import AbstractDataset
import nox.utils.digress.diffusion_utils as utils
from nox.utils.digress.extra_features import ExtraFeatures
from nox.utils.pyg import from_smiles, x_map, e_map
from nox.datasets.ecreact_graph import DatasetInfo
from nox.datasets.qm9 import DistributionNodes, RemoveYTransform, SelectMuTransform, SelectHOMOTransform

@register_object("chembl", "dataset")
class Chembl(AbstractDataset, InMemoryDataset):

    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        self.split_group = split_group
        self.args = args
        self.remove_h = args.remove_h

        self.version = self.get_version()

        self.name = "Chembl"
        self.root = args.data_dir
        # self.version = None

        transform = self.get_transform(args)
        InMemoryDataset.__init__(self, root=self.root, transform=transform)

        
        # self.load_datasets()
        # self.dataset, self.slices = self.datasets[split_group]
        self.dataset = self.create_dataset(split_group)

        data_info_path = os.path.join(self.processed_dir, f"dataset_info_{self.version}.p")
        if args.recompute_statistics:
            # get data info (input / output dimensions)
            smiles = list(set(self.datasets["train"][0].smiles))

            data_info = DatasetInfo(smiles, args)
            pickle.dump(data_info, open(data_info_path, "wb"))
            print(f"[magenta] Saved DatasetInfo at: {data_info_path}")
        else:
            data_info = pickle.load(open(data_info_path, "rb"))

        extra_features = ExtraFeatures(args.extra_features_type, dataset_info=data_info)

        example_batch = [ from_smiles(self.__getitem__(0).smiles), from_smiles(self.__getitem__(1).smiles) ]
        example_batch = Batch.from_data_list(example_batch, None, None)

        data_info.compute_input_output_dims(
            example_batch=example_batch,
            extra_features=extra_features,
            domain_features=None,
        )

        args.dataset_statistics = data_info
        args.extra_features = extra_features
        args.domain_features = None

        self.print_summary_statement(self.dataset, split_group)

    def load_datasets(self):
        self.datasets = {}
        try:
            self.datasets["train"] = torch.load(self.processed_paths[0])
            self.datasets["dev"] = torch.load(self.processed_paths[1])
            self.datasets["test"] = torch.load(self.processed_paths[2])
        except Exception as e:
            raise Exception("Unable to load dataset", e)

    def get_version(self):
        """Checks if changes have been made that would effect the preprocessed graphs"""

        args_hash = md5(
            str(
                [
                    self.args.dataset_name,
                    self.args.data_dir,
                    self.args.remove_h,
                    self.args.extra_features_type,
                ]
            )
        )
        return args_hash

    def get_transform(self, args) -> None:
        target = getattr(args, "guidance_target", None)
        regressor = getattr(self, "regressor", None)
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = None
        else:
            transform = RemoveYTransform()

        return transform

    @property
    def raw_file_names(self):
        return "chembl32_dataset.json"

    @property
    def raw_dir(self) -> str:
        # self.root := args.data_dir
        return self.root

    @property
    def processed_dir(self) -> None:
        """Directory where processed data is stored or expected to be exist in"""
        return os.path.join(self.root, "in_memory")

    @property
    def processed_file_names(self):
        return ["chembl32_train.pt", "chembl32_dev.pt", "chembl32_test.pt"]

    def download(self):
        pass 

    def process(self):
        return 
        chembl_data = json.load(open( os.path.join(self.root, self.raw_file_names), "r"))
        self.assign_splits(chembl_data, self.args.split_probs, self.args.split_seed)
        
        data_list = {"train": [], "dev": [], "test": []}
        for idx, molecule_dict in tqdm(enumerate(chembl_data), total = len(chembl_data), ncols= 60):
            split = molecule_dict["split"]
            mol = from_smiles(molecule_dict["smiles"])

            # first feature is atomic number
            mol.x = F.one_hot(mol.x[:, 0], len(x_map["atomic_num"])).to(torch.float)
            # first feature is bond type (bond type 1 is misc)
            mol.edge_attr = F.one_hot(mol.edge_attr[:, 0], len(e_map["bond_type"])).to(torch.float)
            mol.y = torch.zeros((1, 0), dtype=torch.float)
            mol.idx = idx 
            mol.smiles = molecule_dict["smiles"]

            data_list[split].append(mol)

        for i, split in enumerate(["train", "dev", "test"]):
            assert split in self.processed_paths[i]
            torch.save(self.collate(data_list[split]), self.processed_paths[i])

    def create_dataset(self, split_group):
        """
        Creates the dataset of samples
        """
        chembl_data = json.load(open( os.path.join(self.root, self.raw_file_names), "r"))

        self.assign_splits(chembl_data, self.args.split_probs, self.args.split_seed)

        dataset = []
        for data_item in tqdm(chembl_data, ncols=60):
            if data_item["split"] != split_group:
                continue 
            
            mol = Chem.MolFromSmiles(data_item["smiles"])
            if mol.GetNumAtoms() > 40:
                continue 

            dataset.append(data_item)

        return dataset

    def __getitem__(self, index):
        try:
            molecule_dict = self.dataset[index]

            mol = from_smiles(molecule_dict["smiles"])

            # first feature is atomic number
            mol.x = F.one_hot(mol.x[:, 0], len(x_map["atomic_num"])).to(torch.float)
            # first feature is bond type (bond type 1 is misc)
            mol.edge_attr = F.one_hot(mol.edge_attr[:, 0], len(e_map["bond_type"])).to(torch.float)
            mol.y = torch.zeros((1, 0), dtype=torch.float)
            mol.sample_id = f"{self.split_group}_{index}"
    
            return mol

        except Exception:
            warnings.warn("Could not load sample")

    def __len__(self):
        return len(self.dataset)
        
    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(Chembl, Chembl).add_args(parser)

        parser.add_argument(
            "--recompute_statistics",
            action="store_true",
            default=False,
            help="recompute statistics",
        )
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
