import argparse
from typing import List, Literal, Dict
import torch
import copy

from nox.datasets.molecules import StokesAntibiotics
from nox.datasets.gsm import GSMLinkDataset

from nox.utils.registry import register_object
from nox.utils.classes import set_nox_type

from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from cobra.io import load_matlab_model


@register_object("gsm_chemistry_fc", "dataset")
class GSMChemistryFCDataset(GSMLinkDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        Dataset to predict molecular properties using a 'fully connected' genome-scale metabolic network
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """

        self.molecule_dataset = StokesAntibiotics(args, split_group)

        # need assign split for molecules but not for gsm
        metabo_args = copy.deepcopy(args)
        metabo_args.assign_splits = False
        super(GSMChemistryFCDataset, self).__init__(metabo_args, split_group)

        args.unk_metabolites = [
            k for k, v in self.split_graph.metabolite_features.items() if v is None
        ]
        args.unk_enzymes = [
            k for k, v in self.split_graph.enzyme_features.items() if v is None
        ]

        args.num_pathways = len(self.pathway2node_indx)

        self.set_sample_weights(args)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[Dict]:

        _ = super().create_dataset(split_group)

        self.pathway2node_id = self.get_pathway2node_id()

        self.pathway2node_indx = {
            path: set(self.nodeid2nodeidx[n] for n in nodes)
            for path, nodes in self.pathway2node_id.items()
        }

        # k pathways x n nodes
        self.split_graph.pathway_mask = torch.stack(
            [
                index_to_mask(
                    torch.tensor(list(indices)), self.split_graph.num_nodes
                ).int()
                for path, indices in self.pathway2node_indx.items()
            ]
        )

        # data to metabolite or enzyme
        self.split_graph.node2type = {
            i: type_name
            for type_ids, type_name in [
                (self.split_graph.metabolite_features, "metabolite"),
                (self.split_graph.enzyme_features, "enzyme"),
            ]
            for i in type_ids
        }

        return self.molecule_dataset.dataset

    def get_pathway2node_id(self):
        pathway2node_id = {}
        gsm_model = load_matlab_model(
            f"/Mounts/rbg-storage1/datasets/Metabo/BiGG/{self.args.organism_name}.mat"
        )
        for pathway in gsm_model.groups:
            pathway2node_id.setdefault(pathway.id, set())
            for reaction in pathway.members:
                for metabolite in reaction.metabolites:
                    if metabolite.id in self.nodeid2nodeidx:
                        pathway2node_id[pathway.id].add(metabolite.id)
                for gene in reaction.genes:
                    if gene.id in self.nodeid2nodeidx:
                        pathway2node_id[pathway.id].add(gene.id)

        # ? Add "other"

        return pathway2node_id

    def __len__(self) -> int:
        return len(self.molecule_dataset)

    def __getitem__(self, index: int) -> Data:
        return {"mol": self.molecule_dataset.dataset[index]}

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GSMChemistryFCDataset, GSMChemistryFCDataset).add_args(parser)
        parser.add_argument(
            "--scaffold_balanced",
            action="store_true",
            default=False,
            help="balance the scaffold sets",
        )

    @staticmethod
    def set_args(args) -> None:
        super().set_args(args)
        if args.metabolite_feature_type == "precomputed":
            args.use_rdkit_features = True

    def process(self) -> None:
        super().process()
