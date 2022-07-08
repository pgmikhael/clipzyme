import traceback, warnings
import argparse
from typing import List, Literal
from abc import ABCMeta, abstractmethod
import json
from collections import Counter
import numpy as np
import torch
from torch.utils import data
from nox.utils.loading import get_sample_loader
from nox.utils.registry import register_object, md5
from nox.utils.classes import Nox, set_nox_type
from nox.datasets.utils import METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG
from nox.datasets import AbstractDataset
from torch_geometric.data import HeteroData
import tqdm


@register_object("gsm", "dataset")
class GSMDataset(AbstractDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        Genome Scale Model Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        __metaclass__ = ABCMeta

        super(GSMDataset, self).__init__()

        self.split_group = split_group
        self.args = args

        self.init_class(args, split_group)

        self.input_loader = get_sample_loader(split_group, args)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        self.set_sample_weights(args)

        self.print_summary_statement(self.dataset, split_group)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.load_dataset(args)

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

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        """
        Creates the dataset of samples from json metadata file.
        """
        # node types: metabolite, protein, reaction
        # edge types: reactants_of, creates_products

        protein2id = {}
        metabolite2id = {}
        reaction2id = {}

        metabolite_counter = 0
        protein_counter = 0
        reaction_counter = 0

        metabolites_reactants_of_reactions = []
        reactions_creates_products_metabolites = []
        proteins_reactants_of_reactions = []

        for rxn_dict in tqdm(self.metadata_json, position=0):
            # give each node a unique id
            if rxn_dict["rxn_id"] not in reaction2id:
                reaction2id[rxn_dict["rxn_id"]] = reaction_counter
                reaction_counter += 1

            reactants = rxn_dict["reactants"]
            for reactant in reactants:
                if reactant not in metabolite2id:
                    metabolite2id[reactant["metabolite"]] = metabolite_counter
                    metabolite_counter += 1
                # create edges "metabolites", "reactants_of", "reactions"
                metabolites_reactants_of_reactions.append(
                    [
                        metabolite2id[reactant["metabolite"]],
                        reaction2id[rxn_dict["rxn_id"]],
                    ]
                )

            products = rxn_dict["products"]
            for product in products:
                if product not in metabolite2id:
                    metabolite2id[product["metabolite"]] = metabolite_counter
                    metabolite_counter += 1
                # create edges "reactions", "creates_products", "metabolites"
                reactions_creates_products_metabolites.append(
                    [
                        metabolite2id[product["metabolite"]],
                        reaction2id[rxn_dict["rxn_id"]],
                    ]
                )

            proteins = rxn_dict["proteins"]
            for protein in proteins:
                if protein not in protein2id:
                    protein2id[protein["bigg_gene_id"]] = protein_counter
                    protein_counter += 1
                # create edges "proteins", "reactants_of", "reactions"
                proteins_reactants_of_reactions.append(
                    [
                        protein2id[protein["bigg_gene_id"]],
                        reaction2id[rxn_dict["rxn_id"]],
                    ]
                )

        graph = HeteroData()

        graph["metabolites"].num_nodes = len(metabolite2id)
        graph["proteins"].num_nodes = len(protein2id)
        graph["reactions"].num_nodes = len(reaction2id)

        graph["metabolites", "reactants_of", "reactions"].edge_index = (
            torch.tensor(metabolites_reactants_of_reactions, dtype=torch.long)
            .t()
            .contiguous()
        )  # [2, num_edges_reactants_of]
        graph["proteins", "reactants_of", "reactions"].edge_index = (
            torch.tensor(reactions_creates_products_metabolites, dtype=torch.long)
            .t()
            .contiguous()
        )  # [2, num_edges_reactants_of]
        graph["reactions", "creates_products", "metabolites"].edge_index = (
            torch.tensor(proteins_reactants_of_reactions, dtype=torch.long)
            .t()
            .contiguous()
        )  # [2, num_edges_creates_products]

        # missing, do we need this?
        # graph["reactions", "creates_products", "proteins"].edge_index = ...  # [2, num_edges_creates_products]

        sample = {
            # "sample_id": 0,
            # "y": int(y),
            "graph": graph,  # edges, num of each type of node
            "protein2id": protein2id,
            "metabolite2id": metabolite2id,
            "reaction2id": reaction2id,
        }

        return sample

    def skip_sample(self, sample) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        return False

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        metabolites = len(self.dataset["metabolite2id"])
        reactions = len(self.dataset["reaction2id"])
        proteins = len(self.dataset["protein2id"])
        summary = f"Contructed Genome Scale Model {self.split_group} dataset with {reactions} reactions, {metabolites} metabolites, {proteins} proteins"
        return summary

    def print_summary_statement(self, dataset, split_group):
        statement = "{} DATASET CREATED FOR {}\n.{}".format(
            split_group.upper(), self.args.dataset_name.upper(), self.SUMMARY_STATEMENT
        )
        print(statement)

    def __len__(self) -> int:
        # define length as number of reactions
        return len(self.dataset["reaction2id"])

    def __getitem__(self, index):
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        sample = self.dataset[index]
        graph = sample["graph"]
        graph["metabolites"].x = self.get_node_embeddings(
            sample["metabolite2id"]
        )  # if learnt then returns torch.nn.Embedding((size))

        def forward(bla):
            mol_encoder

        # TODO
        # sample = self.dataset[index]
        # try:
        #     return sample
        # except Exception:
        #     warnings.warn(
        #         LOAD_FAIL_MSG.format(sample["sample_id"], traceback.print_exc())
        #     )

    def assign_splits(self, metadata_json) -> None:
        """
        Assign samples to data splits

        Args:
            metadata_json (dict): raw json dataset loaded
        """
        # TODO
        # for idx in range(len(metadata_json)):
        #     metadata_json[idx]["split"] = np.random.choice(
        #         ["train", "dev", "test"], p=self.args.split_probs
        #     )

    def set_sample_weights(self, args: argparse.ArgumentParser) -> None:
        """
        Set weights for each sample

        Args:
            args (argparse.ArgumentParser)
        """
        # TODO
        # if args.class_bal:
        #     label_dist = [d[args.class_bal_key] for d in self.dataset]
        #     label_counts = Counter(label_dist)
        #     weight_per_label = 1.0 / len(label_counts)
        #     label_weights = {
        #         label: weight_per_label / count for label, count in label_counts.items()
        #     }

        #     print("Class counts are: {}".format(label_counts))
        #     print("Label weights are {}".format(label_weights))
        #     self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]
        # else:
        #     pass

    @property
    def DATASET_ITEM_KEYS(self) -> list:
        """
        List of keys to be included in sample when being batched

        Returns:
            list
        """
        # TODO
        # standard = ["sample_id"]
        # return standard

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--class_bal", action="store_true", default=False, help="class balance"
        )
        parser.add_argument(
            "--class_bal_key",
            type=str,
            default="y",
            help="dataset key to use for class balancing",
        )
        parser.add_argument(
            "--dataset_file_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json",
            help="Path to dataset file either as json or csv",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json",
            help="Path to dataset file either as json or csv",
        )
        parser.add_argument(
            "--num_classes", type=int, default=6, help="Number of classes to predict"
        )
        # Alternative training/testing schemes
        parser.add_argument(
            "--assign_splits",
            action="store_true",
            default=False,
            help="Whether to assign different splits than those predetermined in dataset",
        )
        parser.add_argument(
            "--split_type",
            type=str,
            default="random",
            choices=["random", "institution_split"],
            help="How to split dataset if assign_split = True. Usage: ['random', 'institution_split'].",
        )
        parser.add_argument(
            "--split_probs",
            type=float,
            nargs="+",
            default=[0.6, 0.2, 0.2],
            help="Split probs for datasets without fixed train dev test. ",
        )
        # loader
        parser.add_argument(
            "--input_loader_name",
            type=str,
            action=set_nox_type("input_loader"),
            default="cv_loader",
            help="input loader",
        )
