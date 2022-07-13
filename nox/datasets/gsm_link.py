import argparse
from typing import List, Literal, Dict, Iterable, Tuple, Set
import numpy as np
import torch

from nox.utils.registry import register_object, get_object
from nox.utils.rdkit import get_rdkit_feature
from nox.datasets import AbstractDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_smiles
import tqdm
import itertools
import os


@register_object("gsm_link", "dataset")
class GSMLinkDataset(AbstractDataset, InMemoryDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        Genome Scale Model Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        self.split_group = split_group
        self.args = args
        self.protein_encoder = get_object("fair_esm", "model")(args)

        self.name = "gsm_link"
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
        # if self.args.split_by_reactions:
        train_reactions, val_reactions, test_reactions = self.assign_splits(
            self.metadata_json
        )
        originalid2nodeid = {}
        originalids2metadict = {}
        # TODO: must do any skipping here or in get_triplets otherwise they will be assigned a node id
        train_triplets, originalid2nodeid, originalids2metadict = self.get_triplets(
            train_reactions, originalid2nodeid, originalids2metadict
        )
        val_triplets, originalid2nodeid, originalids2metadict = self.get_triplets(
            val_reactions, originalid2nodeid, originalids2metadict
        )
        test_triplets, originalid2nodeid, originalids2metadict = self.get_triplets(
            test_reactions, originalid2nodeid, originalids2metadict
        )

        nodeid2metadict = {
            node_id: originalids2metadict[original_id]
            for original_id, node_id in originalid2nodeid.items()
        }

        # TODO: implement fixes to allow for other kinds of splits
        # else:
        #     all_triplets = self.get_triplets(self.metadata_json)
        #     train_triplets, val_triplets, test_triplets = self.assign_splits(
        #         all_triplets
        #     )
        # TODO: in the other branch of this if statement, test node indices and train+val node indices will overlap
        # although they refer to different nodes. Furthermore the max node index in each set is used to determine
        # the num_nodes in each graph for graph construction.
        # In order to split in another way (not by reaction), this needs to be fixed.

        # TODO: For inductive (later)
        ####### GOAL: get train_graph, val_edges (in graph structure) that are a subset of the train_graph original edges and do not exist in train_graph
        #######       and test_graph and test_edges (in graph structure) that are a subset of the test_graph

        # transductive case
        train_edge_index = train_triplets[:, :2].t()
        train_relation_type = train_triplets[:, 2]
        train_num_nodes = (
            max(train_triplets[:, 0].max(), train_triplets[:, 1].max()) + 1
        ).item()

        val_edge_index = val_triplets[:, :2].t()
        val_relation_type = val_triplets[:, 2]

        test_edge_index = test_triplets[:, :2].t()
        test_relation_type = test_triplets[:, 2]

        id2metabolites, id2enzymes = self.get_node_features(nodeid2metadict)

        train_data = Data(
            metabolite_graphs=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_relation_type,
            num_nodes=train_num_nodes,
            target_edge_index=train_edge_index,
            target_edge_type=train_relation_type,
        )

        valid_data = Data(
            metabolite_graphs=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_relation_type,
            num_nodes=train_num_nodes,
            target_edge_index=val_edge_index,
            target_edge_type=val_relation_type,
        )
        test_data = Data(
            metabolite_graphs=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_edge_index,
            num_nodes=train_num_nodes,
            target_edge_index=test_edge_index,
            target_edge_type=test_relation_type,
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(
            (self.collate([train_data, valid_data, test_data])), self.processed_paths[0]
        )

    def get_triplets(
        self,
        reactions: List[Dict],
        node2id: Dict = {},
        original_node_ids2metadicts: Dict = {},
    ) -> Tuple[List[Tuple[int, int, int]], Set]:
        """
        params: reactions - list of reactions, typically a json/metadata json format
        returns: triplets - tensor of triplets where each triplet is (head, tail, relation), of the following:
            node types: reactant, product, enzyme
            relation types:
                (m1, m2, is_co_reactant_of), bi-directional
                (p1, p2, is_co_product_of), bi-directional
                (e1, e2, is_co_enzyme_of), bi-directional
                (m1, e1, is_co_reactant_enzyme), bi-directional
                (m1, p1, is_metabolite_reactant_for)
                (p1, m1, is_product_of_metabolite)
                (e1, p1, is_enzyme_reactant_for)
                (p1, e1, is_enzyme_for_product)
        """
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

        original_node_ids2metadicts = {
            "metabolites": {},
            "enzymes": {},
        }

        for rxn_dict in tqdm(reactions, position=0):
            reactants = rxn_dict["reactants"]
            products = rxn_dict["products"]
            enzymes = rxn_dict["proteins"]

            # used to make (m1, m2, is_co_reactant_of), bi-directional
            is_co_reactant_of = set()
            # used to make (p1, p2, is_co_product_of), bi-directional
            is_co_product_of = set()
            # used to make (e1, e2, is_co_enzyme_of), bi-directional
            is_co_enzyme_of = set()
            # (m1, e1, is_co_reactant_enzyme), bi-directional
            is_co_reactant_enzyme = []
            # (m1, p1, is_metabolite_reactant_for), bi-directional called is_product_of_metabolite
            is_metabolite_reactant_for = []
            # (p1, e1, is_enzyme_for_product) bi-directional called is_enzyme_reactant_for
            is_enzyme_for_product = []

            for reactant in reactants:
                metabolite_id = reactant["metabolite_id"]
                if metabolite_id not in node2id:
                    node2id[metabolite_id] = len(node2id)

                node_id = node2id[metabolite_id]

                # store the dict of the metabolite for later use
                if metabolite_id not in original_node_ids2metadicts["metabolites"]:
                    original_node_ids2metadicts["metabolites"][metabolite_id] = reactant

                # add reactants (metabolites) to any relevant relations
                is_co_reactant_of.add(node_id)
                if len(products) > 0:
                    is_metabolite_reactant_for.append(
                        [node_id, relation2id["is_metabolite_reactant_for"]]
                        * len(products)
                    )
                if len(enzymes) > 0:
                    is_co_reactant_enzyme.append(
                        [node_id, relation2id["is_co_reactant_enzyme"]] * len(enzymes)
                    )

            for product in products:
                metabolite_id = product["metabolite_id"]
                if metabolite_id not in node2id:
                    node2id[metabolite_id] = len(node2id)

                node_id = node2id[metabolite_id]

                # store the dict of the metabolite for later use
                if metabolite_id not in original_node_ids2metadicts["metabolites"]:
                    original_node_ids2metadicts["metabolites"][metabolite_id] = product

                # add products (metabolites) to any relevant relations
                is_co_product_of.add(node_id)
                if len(enzymes) > 0:
                    is_enzyme_for_product.append(
                        [node_id, relation2id["is_enzyme_for_product"]] * len(enzymes)
                    )

                # for each relation already created that requires a product, add those products
                for indx in range(len(is_metabolite_reactant_for)):
                    is_metabolite_reactant_for[indx].append(node_id)

            for enzyme in enzymes:
                if self.skip_sample(enzyme=enzyme):
                    continue

                enzyme_id = enzyme["bigg_gene_id"]
                if enzyme not in node2id:
                    node2id[enzyme_id] = len(node2id)

                node_id = node2id[enzyme_id]

                # store the dict of the metabolite for later use
                if enzyme_id not in original_node_ids2metadicts["enzymes"]:
                    original_node_ids2metadicts["enzymes"][enzyme_id] = enzyme

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
                trip[::-1] for trip in is_co_reactant_enzyme
            ]

            # (m1, p1, is_metabolite_reactant_for), bi-directional called is_metabolite_reactant_for
            is_product_of_metabolite = [
                trip[::-1] for trip in is_metabolite_reactant_for
            ]

            # (p1, e1, is_enzyme_for_product) bi-directional called is_enzyme_reactant_for
            is_enzyme_reactant_for = [trip[::-1] for trip in is_enzyme_for_product]

            triplets += (
                perms_is_co_reactant_of
                + perms_is_co_product_of
                + perms_is_co_enzyme_of
                + perms_co_reactant_enzymes
                + is_product_of_metabolite
                + is_enzyme_reactant_for
            )

        # change to (head, tail, relation) tuples, rather than [head, relation, tail]
        triplets = [(triplet[0], triplet[2], triplet[1]) for triplet in triplets]
        return triplets, node2id, original_node_ids2metadicts

    def get_node_features(self, nodeid2metadict) -> List[dict]:
        """
        Obtain node features

        Args:
            nodeid2metadict (dict):

        Raises:
            KeyError: raise error if metadict doesn't contain metabolite or protein id

        Returns:
            id2metabolite_features (dict): map metabolite id to node features
            id2enzyme_features (dict): map protein id to node features
        """
        id2metabolite_features = {}
        id2enzyme_features = {}

        for id, metadata_dict in nodeid2metadict.items():
            if not (
                id2metabolite_features.get(id, False)
                or id2enzyme_features.get(id, False)
            ):
                if "metabolite" in metadata_dict:
                    id2metabolite_features[id] = id
                    if metadata_dict["smiles"]:
                        if self.args.metabolite_feature_type == "precomputed":
                            id2metabolite_features[id] = get_rdkit_feature(
                                metadata_dict["smiles"],
                                method=self.args.rdkit_fingerprint_name,
                            )
                        elif self.args.metabolite_feature_type == "trained":
                            id2metabolite_features[id] = from_smiles(
                                metadata_dict["smiles"]
                            )

                elif "bigg_gene_id" in metadata_dict:
                    id2enzyme_features[id] = id
                    if self.args.protein_feature_type == "precomputed":
                        id2enzyme_features[id] = self.protein_encoder(
                            metadata_dict["protein_sequence"]
                        )
                    elif self.args.protein_feature_type == "trained":
                        id2enzyme_features[id] = metadata_dict["protein_sequence"]

                else:
                    raise KeyError(
                        f"[metabolite] OR [protein] NOT FOUND IN METADICT FOR NODE {id}. AVAILABLE KEYS ARE: {nodeid2metadict.keys()}"
                    )

        return id2metabolite_features, id2enzyme_features

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

    def skip_sample(self, **kwargs) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if kwargs.get("enzyme", False):
            if kwargs["enzyme"]["protein_sequence"] is None:
                return True
        return False

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        summary = f"Contructed Genome Scale Model {self.split_group} dataset with {len(self.split_graph)} samples in the split graph"
        return summary

    def print_summary_statement(self, dataset, split_group):
        statement = "{} DATASET CREATED FOR {}\n.{}".format(
            split_group.upper(), self.args.dataset_name.upper(), self.SUMMARY_STATEMENT
        )
        print(statement)

    # def __len__(self) -> int:
    # TODO: if needed, implement

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

    def assign_splits(self, metadata_json: Iterable) -> Tuple:
        """
        Assigns each item in the iterable metadata_json to a split group.

        param: metadata_json: iterable, typically a json of reactions
        returns: splits.values() - a tuple-like (dict_values) object of split groups with length 3
        """
        splits = {
            "train": [],
            "dev": [],
            "test": [],
        }
        for idx in range(len(metadata_json)):
            splits[
                np.random.choice(["train", "dev", "test"], p=self.args.split_probs)
            ].append(metadata_json[idx])

        return splits.values()

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
        pass

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
        pass

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(GSMLinkDataset, GSMLinkDataset).add_args(parser)
        parser.add_argument(
            "--metabolite_feature_type",
            type=str,
            default="none",
            choices=["none", "precomputed", "trained"],
            help="how to initialize metabolite node features",
        )
        parser.add_argument(
            "--rdkit_fingerprint_name",
            type=str,
            default="rdkit_fingerprint",
            choices=["morgan_binary", "morgan_counts", "rdkit_fingerprint"],
            help="fingerprint name for initializing molecule features",
        )
        parser.add_argument(
            "--protein_feature_type",
            type=str,
            default="none",
            choices=["none", "precomputed", "trained"],
            help="how to initialize protein node features",
        )
