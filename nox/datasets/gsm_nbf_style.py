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
from torch_geometric.data import HeteroData, InMemoryDataset, Data
import tqdm
import itertools


@register_object("gsm_nbf", "dataset")
class GSMNBFDataset(AbstractDataset, InMemoryDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        Genome Scale Model Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        self.split_group = split_group
        self.args = args

        self.args = args

        self.name = "gsm_nbf"
        # self.version = None
        InMemoryDataset.__init__(self, root=args.data_dir)

        self.init_class(args, split_group)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        args.num_relations = self.num_relations
        self.print_summary_statement(self.dataset, split_group)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        self.load_dataset(args)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        # TODO: implement
        # return ["train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"]
        pass

    def download(self):
        pass

    def process(self):
        # this function converts the raw data into triplets
        # the key thing is that test and train files are organised in a specific order

        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat(
            [train_fact_index, train_fact_index.flip(0)], dim=-1
        )
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))

        ################## OLD ##################################
        ###### GOAL is to covert below code to above format
        # node types: reactant, product, enzyme,
        # edge types:
        # (m1, m2, is_co_reactant_of), bi-directional
        # (p1, p2, is_co_product_of), bi-directional
        # (e1, e2, is_co_enzyme_of), bi-directional
        # (m1, e1, is_co_reactant_enzyme), bi-directional
        # (m1, p1, is_metabolite_reactant_for)
        # (p1, m1, is_product_of_metabolite)
        # (e1, p1, is_enzyme_reactant_for)
        # (p1, e1, is_enzyme_for_product)

        node2id = {}
        relation2id = {
            "is_co_reactant_of": 0,
            "is_co_product_of": 1,
            "is_co_enzyme_of": 2,
            "is_co_reactant_enzyme": 3,
            "is_metabolite_reactant_for": 4,
            "is_product_of_metabolite": 5,
            "is_enzyme_reactant_for": 6,
            "is_enzyme_for_product": 7,
        }

        triplets = []

        for rxn_dict in tqdm(self.metadata_json, position=0):
            reactants = rxn_dict["reactants"]
            products = rxn_dict["products"]
            enzymes = rxn_dict["proteins"]

            is_co_reactant_of = (
                set()
            )  # used to make (m1, m2, is_co_reactant_of), bi-directional
            is_co_product_of = (
                set()
            )  # used to make (p1, p2, is_co_product_of), bi-directional
            is_co_enzyme_of = (
                set()
            )  # used to make (e1, e2, is_co_enzyme_of), bi-directional

            is_co_reactant_enzyme = (
                []
            )  # (m1, e1, is_co_reactant_enzyme), bi-directional
            is_metabolite_reactant_for = (
                []
            )  # (m1, p1, is_metabolite_reactant_for), bi-directional called is_product_of_metabolite
            is_enzyme_for_product = (
                []
            )  # (p1, e1, is_enzyme_for_product) bi-directional called is_enzyme_reactant_for

            for reactant in reactants:
                if reactant not in node2id:
                    node2id[reactant["metabolite"]] = len(node2id)

                node_id = node2id[reactant["metabolite"]]

                # add reactants (metabolites) to any relevant relations
                is_co_reactant_of.add(node_id)
                is_metabolite_reactant_for.append(
                    [node_id, relation2id["is_metabolite_reactant_for"]] * len(products)
                )
                is_co_reactant_enzyme.append(
                    [node_id, relation2id["is_co_reactant_enzyme"]] * len(enzymes)
                )

            for product in products:
                if product not in node2id:
                    node2id[product["metabolite"]] = len(node2id)

                node_id = node2id[product["metabolite"]]

                # add products (metabolites) to any relevant relations
                is_co_product_of.add(node_id)
                is_enzyme_for_product.append(
                    [node_id, relation2id["is_enzyme_for_product"]] * len(enzymes)
                )

                # for each relation already created that requires a product, add those products
                for indx in range(len(is_metabolite_reactant_for)):
                    is_metabolite_reactant_for[indx].append(node_id)

            for enzyme in enzymes:
                if enzyme not in node2id:
                    node2id[enzyme["bigg_gene_id"]] = len(node2id)

                node_id = node2id[enzyme["bigg_gene_id"]]

                # add enzymes (proteins) to any relevant relations
                is_co_enzyme_of.add(node_id)

                # for each relation already created that requires a product, add those products
                for indx in range(len(is_co_reactant_enzyme)):
                    is_co_reactant_enzyme[indx].append(node_id)

                for indx in range(len(is_enzyme_for_product)):
                    is_enzyme_for_product[indx].append(node_id)

        # Add flip directions
        # used to make (m1, m2, is_co_reactant_of), bi-directional
        perms_is_co_reactant_of = [
            [h, relation2id["is_co_reactant_of"], t]
            for h, t in itertools.permutations(is_co_reactant_of, 2)
        ]

        # used to make (p1, p2, is_co_product_of), bi-directional
        perms_is_co_product_of = [
            [h, relation2id["is_co_product_of"], t]
            for h, t in itertools.permutations(is_co_product_of, 2)
        ]

        # used to make (e1, e2, is_co_enzyme_of), bi-directional
        perms_is_co_enzyme_of = [
            [h, relation2id["is_co_enzyme_of"], t]
            for h, t in itertools.permutations(is_co_enzyme_of, 2)
        ]

        # (m1, e1, is_co_reactant_enzyme), bi-directional
        perms_co_reactant_enzymes = is_co_reactant_enzyme + [
            l.reverse() for l in is_co_reactant_enzyme
        ]

        # (m1, p1, is_metabolite_reactant_for), bi-directional called is_product_of_metabolite
        # is_product_of_metabolite = is_metabolite_reactant_for + [
        # (p1, e1, is_enzyme_for_product) bi-directional called is_enzyme_reactant_for

        # graph = HeteroData()

        # graph["metabolites"].num_nodes = len(node2id)
        # graph["proteins"].num_nodes = len(node2id)
        # graph["reactions"].num_nodes = len(reaction2id)

        # graph["metabolites", "reactants_of", "reactions"].edge_index = (
        #     torch.tensor(metabolites_reactants_of_reactions, dtype=torch.long)
        #     .t()
        #     .contiguous()
        # )  # [2, num_edges_reactants_of]
        # graph["proteins", "reactants_of", "reactions"].edge_index = (
        #     torch.tensor(reactions_creates_products_metabolites, dtype=torch.long)
        #     .t()
        #     .contiguous()
        # )  # [2, num_edges_reactants_of]
        # graph["reactions", "creates_products", "metabolites"].edge_index = (
        #     torch.tensor(proteins_reactants_of_reactions, dtype=torch.long)
        #     .t()
        #     .contiguous()
        # )  # [2, num_edges_creates_products]

        # # missing, do we need this?
        # # graph["reactions", "creates_products", "proteins"].edge_index = ...  # [2, num_edges_creates_products]

        # sample = {
        #     # "sample_id": 0,
        #     # "y": int(y),
        #     "graph": graph,  # edges, num of each type of node
        #     "protein2id": protein2id,
        #     "metabolite2id": metabolite2id,
        #     "reaction2id": reaction2id,
        # }

        # return sample

        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(inv_train_entity_vocab),
            target_edge_index=edge_index[:, train_slice],
            target_edge_type=edge_type[train_slice],
        )
        valid_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(inv_train_entity_vocab),
            target_edge_index=edge_index[:, valid_slice],
            target_edge_type=edge_type[valid_slice],
        )
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            num_nodes=len(inv_test_entity_vocab),
            target_edge_index=edge_index[:, test_slice],
            target_edge_type=edge_type[test_slice],
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            (self.collate([train_data, valid_data, test_data])), self.processed_paths[0]
        )

    def __repr__(self):
        return "%s()" % self.name

    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads dataset file

        Args:
            args (argparse.ArgumentParser)

        Raises:
            Exception: Unable to load
        """
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])

        except Exception as e:
            raise Exception("Can't load dataset", e)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        """
        Creates the dataset of samples from json metadata file.
        """
        dataset = []

        if self.split_group == "train":
            self.split_graph = self.seperate_collated_data(0)
        elif self.split_group == "dev":
            self.split_graph = self.seperate_collated_data(1)
        elif self.split_group == "test":
            self.split_graph = self.seperate_collated_data(2)
        else:
            raise ValueError(f"Invalid split group: {self.split_group}")

        triplets = torch.cat(
            [
                self.split_graph.target_edge_index,
                self.split_graph.target_edge_type.unsqueeze(0),
            ]
        ).t()

        # make triplets
        for i, t in enumerate(triplets):
            dataset.append({"triplet": t})

        return dataset

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
        item = self.dataset[index]
        return item

        # sample = self.dataset[index]
        # graph = sample["graph"]
        # graph["metabolites"].x = self.get_node_embeddings(
        #     sample["metabolite2id"]
        # )  # if learnt then returns torch.nn.Embedding((size))

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
