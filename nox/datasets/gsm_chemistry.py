import argparse
from typing import List, Literal, Dict
import torch
import copy

from nox.datasets.molecules import StokesAntibiotics
from nox.datasets.gsm_link import GSMLinkDataset

from nox.utils.registry import register_object

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

        self.molecule_dataset = StokesAntibiotics(self.args, split_group)

        # need assign split for molecules but not for gsm
        metabo_args = copy.deepcopy(args)
        metabo_args.assign_splits = False
        super(GSMChemistryFCDataset, self).__init__(metabo_args, split_group)

        args.unk_metabolites = [
            k for k, v in self.data.metabolite_features.items() if v is None
        ]
        args.unk_enzymes = [
            k for k, v in self.data.enzyme_features.items() if v is None
        ]

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
        self.data.pathway_mask = torch.stack(
            [
                index_to_mask(torch.tensor(indices), self.data.num_nodes)
                for path, indices in self.pathway2node_indx.items()
            ]
        )

        # data to metabolite or enzyme
        self.data.node2type = {
            i: type_name
            for type_ids, type_name in [
                (self.data.id2metabolites, "metabolite"),
                (self.data.id2enzymes, "enzyme"),
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
                    pathway2node_id[pathway.id].add(metabolite.id)
                for gene in pathway.genes:
                    pathway2node_id[pathway.id].add(gene.id)

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