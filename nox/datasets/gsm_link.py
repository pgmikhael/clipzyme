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
from torch_geometric.data import InMemoryDataset, Data
from nox.utils.pyg import from_smiles

import json
from tqdm import tqdm
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
        if args.protein_feature_type == "precomputed":
            self.protein_encoder = get_object(self.args.protein_encoder_name, "model")(args)

        self.version = self.get_version()

        self.name = "gsm_link"
        self.root = args.data_dir
        # self.version = None

        self.init_class(args, split_group)

        InMemoryDataset.__init__(self, root=self.root)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        args.num_relations = self.num_relations
        self.print_summary_statement(self.dataset, split_group)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        self.load_dataset(args)

    def get_version(self):
        """Checks if changes have been made that would effect the preprocessed graphs"""
        # hash skip_sample function
        skip_sample_hash = md5(inspect.getsource(self.skip_sample))
        # hash args that would modify the preprocessed graphs
        args_hash = md5(
            str(
                [
                    self.args.metabolite_feature_type,
                    self.args.rdkit_fingerprint_name,
                    self.args.protein_feature_type,
                    self.args.protein_encoder_name,
                    self.args.pretrained_hub_dir,
                    self.args.train_encoder,
                ]
            )
        )

        return md5(skip_sample_hash + args_hash)

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        # self.root := args.data_dir
        return self.root

    @property
    def processed_dir(self) -> None:
        """Directory where processed data is stored or expected to be exist in"""
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> None:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return self.version + "graph.pt"

    @property
    def raw_file_names(self) -> None:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return [f"{self.args.organism_name}_dataset.json"]

    def download(self) -> None:
        raise Exception(
            f"Dataset is trying to download, this means that {self.args.organism_name}_dataset.json does not exist in {self.raw_dir}"
        )

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        # splits graph by reactions
        train_reactions, val_reactions, test_reactions = self.assign_splits(
            self.metadata_json, self.args.split_probs, self.args.split_seed
        )
        originalid2nodeid = {}
        originalids2metadict = {}
        # Must do any skipping here or in self.get_triplets otherwise they will be assigned a node id
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

        # TODO: For inductive (later):
        #  get train_graph, val_edges (in graph structure) that are a subset of the train_graph original edges and do not exist in train_graph
        #  and test_graph and test_edges (in graph structure) that are a subset of the test_graph

        # transductive case
        train_edge_index = train_triplets[:, :2].t()
        train_relation_type = train_triplets[:, 2]
        # note: all nodes need to exist in the graph for link prediction
        train_num_nodes = len(nodeid2metadict) #train_triplets[:, :2].max().item() + 1

        val_edge_index = val_triplets[:, :2].t()
        val_relation_type = val_triplets[:, 2]

        test_edge_index = test_triplets[:, :2].t()
        test_relation_type = test_triplets[:, 2]

        id2metabolites, id2enzymes = self.get_node_features(nodeid2metadict)

        train_data = Data(
            metabolite_features=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_relation_type,
            num_nodes=train_num_nodes,
            target_edge_index=train_edge_index,
            target_edge_type=train_relation_type,
        )

        valid_data = Data(
            metabolite_features=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_relation_type,
            num_nodes=train_num_nodes,
            target_edge_index=val_edge_index,
            target_edge_type=val_relation_type,
        )
        test_data = Data(
            metabolite_features=id2metabolites,
            enzyme_features=id2enzymes,
            edge_index=train_edge_index,
            edge_type=train_relation_type,
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
    ) -> Tuple[List[Tuple[int, int, int]], Dict, Dict]:
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

        for rxn_dict in tqdm(reactions):
            # skip reactions that dont have any reactants or any products (pseudo-reactions)
            if self.skip_sample(reaction=rxn_dict):
                print(
                    f"Skipping reaction {rxn_dict['rxn_id']}, {len(rxn_dict.get('reactants', []))} reactants, {len(rxn_dict.get('products', []))} products and {len(rxn_dict.get('proteins', []))} proteins"
                )
                continue

            reactants = rxn_dict["reactants"]
            products = rxn_dict["products"]
            enzymes = rxn_dict.get("proteins", [])

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
                if metabolite_id not in original_node_ids2metadicts:
                    original_node_ids2metadicts[metabolite_id] = reactant

                # add reactants (metabolites) to any relevant relations
                is_co_reactant_of.add(node_id)
                if len(products) > 0:
                    is_metabolite_reactant_for += [
                        [node_id, relation2id["is_metabolite_reactant_for"]]
                        for j in range(len(products))
                    ]

                if len(enzymes) > 0:
                    is_co_reactant_enzyme += [
                        [node_id, relation2id["is_co_reactant_enzyme"]]
                        for j in range(len(enzymes))
                    ]

            for indx, product in enumerate(products):
                metabolite_id = product["metabolite_id"]
                if metabolite_id not in node2id:
                    node2id[metabolite_id] = len(node2id)

                node_id = node2id[metabolite_id]

                # store the dict of the metabolite for later use
                if metabolite_id not in original_node_ids2metadicts:
                    original_node_ids2metadicts[metabolite_id] = product

                # add products (metabolites) to any relevant relations
                is_co_product_of.add(node_id)
                if len(enzymes) > 0:
                    is_enzyme_for_product += [
                        [node_id, relation2id["is_enzyme_for_product"]]
                        for j in range(len(enzymes))
                    ]

                # for each relation already created that requires a product, add those products
                for i in range(len(reactants)):
                    is_metabolite_reactant_for[indx + i * len(products)].append(node_id)

            for indx, enzyme in enumerate(enzymes):
                # skip enzymes with no sequence and then remove the triplets that expected to have that enzyme appended to them
                if self.skip_sample(enzyme=enzyme):
                    # is_co_reactant_enzyme
                    is_co_reactant_enzyme_remove_indices = [
                        indx + i * len(enzymes) for i in range(len(reactants))
                    ]  # remove all triplets for this enzyme
                    is_co_reactant_enzyme = [
                        i
                        for j, i in enumerate(is_co_reactant_enzyme)
                        if j not in is_co_reactant_enzyme_remove_indices
                    ]

                    # is_enzyme_for_product
                    is_enzyme_for_product_remove_indices = [
                        indx + i * len(enzymes) for i in range(len(products))
                    ]  # remove all triplets for this enzyme
                    is_enzyme_for_product = [
                        i
                        for j, i in enumerate(is_enzyme_for_product)
                        if j not in is_enzyme_for_product_remove_indices
                    ]
                    continue

                enzyme_id = enzyme["bigg_gene_id"]
                if enzyme_id not in node2id:
                    node2id[enzyme_id] = len(node2id)

                node_id = node2id[enzyme_id]

                # store the dict of the metabolite for later use
                if enzyme_id not in original_node_ids2metadicts:
                    original_node_ids2metadicts[enzyme_id] = enzyme

                # add enzymes (proteins) to any relevant relations
                is_co_enzyme_of.add(node_id)

                # for each relation already created that requires a product, add those products
                for i in range(len(reactants)):
                    is_co_reactant_enzyme[indx + i * len(enzymes)].append(node_id)

                for i in range(len(products)):
                    is_enzyme_for_product[indx + i * len(enzymes)].append(node_id)

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

            assert all(
                [
                    len(t) == 3
                    for t in perms_is_co_reactant_of
                    + perms_is_co_product_of
                    + perms_is_co_enzyme_of
                    + perms_co_reactant_enzymes
                    + is_product_of_metabolite
                    + is_enzyme_reactant_for
                ]
            )

            triplets += (
                perms_is_co_reactant_of
                + perms_is_co_product_of
                + perms_is_co_enzyme_of
                + perms_co_reactant_enzymes
                + is_product_of_metabolite
                + is_enzyme_reactant_for
                + is_metabolite_reactant_for
                + is_enzyme_for_product
            )

        # change to (head, tail, relation) tuples, rather than [head, relation, tail]
        triplets = [(triplet[0], triplet[2], triplet[1]) for triplet in triplets]
        for triplet in triplets:
            assert triplet[2] in relation2id.values()
        triplets = torch.tensor(triplets)
        return triplets, node2id, original_node_ids2metadicts

    def get_node_features(self, nodeid2metadict: Dict[int, Dict]) -> Tuple[Dict, Dict]:
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

        print("Getting node features... this may take a while")

        for id, metadata_dict in tqdm(nodeid2metadict.items()):
            if not (
                id2metabolite_features.get(id, False)
                or id2enzyme_features.get(id, False)
            ):
                if "metabolite_id" in metadata_dict:
                    id2metabolite_features[id] = None
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
                    id2enzyme_features[id] = None
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
            self.metadata_json = json.load(
                open(
                    os.path.join(self.root, f"{self.args.organism_name}_dataset.json"),
                    "rb",
                )
            )

        except Exception as e:
            raise Exception("Unable to load dataset", e)

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[Dict]:
        """
        Creates the dataset of samples from json metadata file.
        """
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])

        except Exception as e:
            raise Exception("Unable to load dataset", e)

        dataset = []

        if self.split_group == "train":
            self.split_graph = self.get(0)
        elif self.split_group == "dev":
            self.split_graph = self.get(1)
        elif self.split_group == "test":
            self.split_graph = self.get(2)

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
            # if missing protein sequence, skip sample
            if kwargs["enzyme"]["protein_sequence"] is None:
                return True

        if kwargs.get("reaction", False):
            # if is a pseudo-reaction (ie no reactants or products), skip sample
            if (
                len(kwargs["reaction"].get("reactants", [])) == 0
                or len(kwargs["reaction"].get("products", [])) == 0
            ):
                return True

        return False

    @property
    def SUMMARY_STATEMENT(self) -> str:
        """
        Prints summary statement with dataset stats
        """
        triplets = len(self.dataset)
        num_nodes = self.split_graph.num_nodes

        summary = f"Containing {triplets} triplets and {num_nodes} number of nodes in the split graph"
        return summary

    def print_summary_statement(
        self, dataset, split_group: Literal["train", "dev", "test"]
    ) -> None:
        statement = f"{split_group.upper()} DATASET CREATED FOR {self.args.dataset_name.upper()}\n{self.SUMMARY_STATEMENT}"
        print(statement)

    def __getitem__(self, index: int) -> Dict:
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        item = self.dataset[index]
        return item

    def assign_splits(self, metadata_json: Iterable, split_probs, seed) -> Tuple:
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
            splits[np.random.choice(["train", "dev", "test"], p=split_probs)].append(
                metadata_json[idx]
            )

        return splits.values()

    @classproperty
    def DATASET_ITEM_KEYS(self) -> List:
        """
        List of keys to be included in sample when being batched

        Returns:
            list
        """
        # Leaving here in case we want to change this in the future
        return []

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
        parser.add_argument(
            "--organism_name",
            type=str,
            required=True,
            default=None,
            help="name of organism that exists in BiGG Models",
        )
        parser.add_argument(
            "--protein_encoder_name",
            type=str,
            default="fair_esm",
            help="name of the protein encoder",
            action=set_nox_type("model"),
        )
