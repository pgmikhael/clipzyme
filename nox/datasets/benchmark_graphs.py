import argparse
import copy
from typing import List, Literal
import os
import torch

from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset

from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR
from torch_geometric.data.separate import separate


@register_object("ind_rel_link_pred", "dataset")
class IndRelLinkPredDataset(AbstractDataset, InMemoryDataset):
    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        ],
    }

    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        FB15k-237 or WN18RR dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        self.split_group = split_group
        self.args = args

        self.name = args.dataset_variant
        self.version = args.dataset_version
        assert self.name in ["FB15k-237", "WN18RR"]
        assert self.version in ["v1", "v2", "v3", "v4"]
        InMemoryDataset.__init__(self, root=args.data_dir)

        self.init_class(args, split_group)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        # self.set_sample_weights(args)
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
        return ["train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
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
        Creates the dataset
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

    def seperate_collated_data(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        # self.data is graph
        # self.dataset is number of graphs
        summary = f"Contructed {self.split_group} of {self.args.dataset_variant} dataset with length of {len(self.data)}"
        return summary

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

    @staticmethod
    def add_args(parser):
        super(IndRelLinkPredDataset, IndRelLinkPredDataset).add_args(parser)
        parser.add_argument(
            "--dataset_variant", type=str, default="FB15k-237", help="Dataset variant"
        )
        parser.add_argument(
            "--dataset_version", type=str, default="v1", help="Dataset version"
        )
