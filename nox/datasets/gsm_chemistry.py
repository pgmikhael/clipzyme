import argparse
import warnings
from typing import List, Literal, Dict, Iterable, Tuple, Set
import numpy as np
import torch
import inspect

from nox.utils.registry import register_object, get_object, md5
from nox.utils.classes import set_nox_type, classproperty
from nox.utils.rdkit import get_rdkit_feature
from nox.datasets.abstract import AbstractDataset
from nox.datasets.moleculenet import MoleNet
from nox.utils.pyg import from_smiles
from nox.datasets.gsm_link import GSMLinkDataset

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import index_to_mask
from cobra.io import load_matlab_model
import json
from tqdm import tqdm
import itertools
import os


@register_object("gsm_chemistry_fc", "dataset")
class GSMChemistryFCDataset(GSMLinkDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        Dataset to predict molecular properties using a 'fully connected' genome-scale metabolic network
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """

        self.molecule_dataset = MoleNet(self.args, split_group)
        super(GSMChemistryFCDataset, self).__init__(args, split_group)
        self.pathway2node_id = self.get_pathway2node_id()
        self.pathway2node_indx = {
            path: set(self.nodeid2nodeidx[n] for n in nodes)
            for path, nodes in self.pathway2node_id.items()
        }
        self.pathway2mask = {
            path: index_to_mask(torch.tensor(indices), self.split_graph.num_nodes)
            for path, indices in self.pathway2node_indx.items()
        }

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[Dict]:

        # update labels to pathways
        for d in self.molecule_dataset.dataset:
            # update aux_label (pathway, reaction, enzyme, etc)
            d.aux_label = self.lookup_aux_label(d.smiles)
            # update aux_has_y
            d.aux_has_y = ~torch.isnan(d.aux_label)

        # following super will create self.split_graph which is metabolic graph
        # assing_splits == False will not split the graph (full graph)
        return super().create_dataset(split_group)

    def lookup_aux_label(self, smiles):
        # TODO: once we have csv with smile:pathway mapping, we can use that instead
        return None

    def get_pathway2node_id(self):
        pathway2node_id = {}
        gsm_model = load_matlab_model(
            f"/Mounts/rbg-storage1/datasets/Metabo/BiGG/{self.args.organism_name}.mat"
        )
        for pathway in gsm_model.groups:
            pathway2node_id.setdefault(pathway.id, set())
            for reaction in pathway.members:
                for metabolite in reaction.metabolites:
                    pathway2node_id[pathway.id].add(metabolite.id)
                for gene in pathway.genes:
                    pathway2node_id[pathway.id].add(gene.id)

        return pathway2node_id

    def __getitem__(self, index: int) -> Data:
        return {"mol": self.molecule_dataset.dataset[index]}
