import traceback, warnings
import argparse
from typing import List, Literal, Any, Callable, Optional, Tuple, Union
from abc import ABCMeta, abstractmethod
import json, os
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
from collections.abc import Sequence
from torch_geometric.data.makedirs import makedirs
import tqdm
import re


@register_object("ind_rel_link_pred", "dataset")
class IndRelLinkPredDataset(AbstractDataset):
    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """
        FB15k-237 or WN18RR dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        __metaclass__ = ABCMeta

        super(IndRelLinkPredDataset, self).__init__()

        self.urls = {
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

        self.split_group = split_group
        self.args = args

        self.name = args.dataset_variant
        self.version = args.dataset_version
        assert self.name in ["FB15k-237", "WN18RR"]
        assert self.version in ["v1", "v2", "v3", "v4"]
        # self.root = root
        # self.transform = transform
        # self.pre_transform = pre_transform
        # self.pre_filter = pre_filter
        # self._indices: Optional[Sequence] = None

        # self.data = None
        # self.slices = None
        # self._data_list: Optional[List[Data]] = None

        self.init_class(args, split_group)

        # self.input_loader = get_sample_loader(split_group, args)

        # self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        # self.set_sample_weights(args)

        self.print_summary_statement(self.dataset, split_group)

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
           Default is to load JSON dataset

        Args:
            args (argparse.ArgumentParser)
            split_group (str)
        """
        self.load_dataset(args)

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_file_names)
        return [os.path.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _process(self):
        f = os.path.join(self.processed_dir, "pre_transform.pt")
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first"
            )

        f = os.path.join(self.processed_dir, "pre_filter.pt")
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first"
            )

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print("Processing...", file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, "pre_transform.pt")
        torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, "pre_filter.pt")
        torch.save(_repr(self.pre_filter), path)

        print("Done!", file=sys.stderr)

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

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        y = self.data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(self.data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return self.data.y.size(-1)

    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int) -> Data:
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

    @staticmethod
    def collate(data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def copy(self, idx: Optional[IndexType] = None) -> "InMemoryDataset":
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in self.indices()]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset

    def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
        if isinstance(node, Mapping):
            for key, value in node.items():
                for inner_key, inner_value in nested_iter(value):
                    yield inner_key, inner_value
        elif isinstance(node, Sequence):
            for i, inner_value in enumerate(node):
                yield i, inner_value
        else:
            yield None, node

    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads dataset file

        Args:
            args (argparse.ArgumentParser)

        Raises:
            Exception: Unable to load
        """
        try:
            if self.download.__qualname__.split(".")[0] != "Dataset":
                self._download()

            if self.process.__qualname__.split(".")[0] != "Dataset":
                self._process()

            self.dataset, self.slices = torch.load(self.processed_paths[0])

        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        """
        Creates the dataset of samples from json metadata file.
        """
        pass

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
        summary = f"Contructed {self.args.dataset_variant} dataset with length of {len(self.dataset)}"
        return summary

    def print_summary_statement(self, dataset, split_group):
        statement = "{} DATASET CREATED FOR {}\n.{}".format(
            split_group.upper(), self.args.dataset_name.upper(), self.SUMMARY_STATEMENT
        )
        print(statement)

    def __getitem__(self, index):
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        item = {}
        cls = self.args.dataset_variant
        if cls == "FB15k-237":
            data = self.dataset.data
            if self.split_group == "train":
                train_data = Data(
                    edge_index=data.edge_index,
                    edge_type=data.edge_type,
                    num_nodes=data.num_nodes,
                    target_edge_index=data.train_edge_index,
                    target_edge_type=data.train_edge_type,
                )
                item["graph"] = train_data
            elif self.split_group == "dev":
                valid_data = Data(
                    edge_index=data.edge_index,
                    edge_type=data.edge_type,
                    num_nodes=data.num_nodes,
                    target_edge_index=data.valid_edge_index,
                    target_edge_type=data.valid_edge_type,
                )
                item["graph"] = valid_data
            elif self.split_group == "test":
                test_data = Data(
                    edge_index=data.edge_index,
                    edge_type=data.edge_type,
                    num_nodes=data.num_nodes,
                    target_edge_index=data.test_edge_index,
                    target_edge_type=data.test_edge_type,
                )
                item["graph"] = test_data
            else:
                raise ValueError(f"Invalid split group: {self.split_group}")
            # self.dataset.data, self.dataset.slices = self.dataset.collate(
            #    [train_data, valid_data, test_data]
            # )
        elif cls == "WN18RR":
            # dataset = WordNet18RR(**cfg.dataset)
            # convert wn18rr into the same format as fb15k-237
            data = self.dataset.data
            num_nodes = int(data.edge_index.max()) + 1
            num_relations = int(data.edge_type.max()) + 1
            edge_index = data.edge_index[:, data.train_mask]
            edge_type = data.edge_type[data.train_mask]
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
            edge_type = torch.cat([edge_type, edge_type + num_relations])
            if self.split_group == "train":
                train_data = Data(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    num_nodes=num_nodes,
                    target_edge_index=data.edge_index[:, data.train_mask],
                    target_edge_type=data.edge_type[data.train_mask],
                )
                item["graph"] = train_data
            elif self.split_group == "dev":
                valid_data = Data(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    num_nodes=num_nodes,
                    target_edge_index=data.edge_index[:, data.val_mask],
                    target_edge_type=data.edge_type[data.val_mask],
                )
                item["graph"] = valid_data
            elif self.split_group == "test":
                test_data = Data(
                    edge_index=edge_index,
                    edge_type=edge_type,
                    num_nodes=num_nodes,
                    target_edge_index=data.edge_index[:, data.test_mask],
                    target_edge_type=data.edge_type[data.test_mask],
                )
                item["graph"] = test_data
            else:
                raise ValueError(f"Invalid split group: {self.split_group}")
            # self.dataset.data, self.dataset.slices = self.dataset.collate(
            #     [train_data, valid_data, test_data]
            # )
            # train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2] # from 'run.py'
            self.dataset.num_relations = num_relations * 2

        else:
            raise ValueError("Unknown dataset `%s`" % cls)

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
            default="default_image_loader",
            help="input loader",
        )


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([os.path.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", obj.__repr__())
