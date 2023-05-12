import os
import warnings
from rich import print
from typing import Sequence, List, Literal, Dict

import argparse
import random
import json
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd

from nox.utils.registry import register_object
from nox.datasets.abstract import AbstractDataset
from nox.utils.pyg import from_smiles

import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.data import Dataset
from esm import pretrained
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
from collections import Counter

warnings.filterwarnings("ignore", category=Bio.PDB.PDBExceptions.PDBConstructionWarning)

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load sample: {}\nException: {}"

from nox.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    precompute_node_embeddings,
    compute_node_embedding,
    get_sequences,
)


@register_object("quickprot", "dataset")
class QuickProtNox(AbstractDataset):
    def __init__(self, args: argparse.Namespace, split_group):
        self.base_class = QuickProtDataset
        self.already_assigned_splits = False
        self.split_group = split_group
        super().__init__(args, split_group)

    def set_sample_weights(self, args: argparse.ArgumentParser) -> None:
        """
        Set weights for each sample
        Standard method is too slow because get is slow (ish)
        Replace self.dataset with self.metadata_json (which contains labels)
        """
        if args.class_bal:
            try:
                label_dist = [d[args.class_bal_key] for d in self.dataset.metadata_json if d["split"] == self.split_group]
            except KeyError:
                label_dist = [d[args.class_bal_key] for d in self.dataset.metadata_json if "split" in d and d["split"] == self.split_group]
                assert len(label_dist) > 0, "No samples found with split {}".format(self.split_group)
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset.metadata_json]
        else:
            pass

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        full_dataset = self.base_class(
            root=self.args.root_dir,
            structures_dir=self.args.structures_dir,
            transform=None,
            pre_transform=None,
            metadata_dir=self.args.dataset_file_path,
            protein_resolution="residue",
            graph_edge_args={"knn_size": 10},
            center_protein=True,
            precompute_and_cache_node_embeddings=self.args.precompute_and_cache_node_embeddings,
            esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            get_sequences_from_structure=False,
        )
        print("Assigning splits")
        split_indices = self.assign_splits(
            full_dataset.metadata_json, self.args.split_probs, seed=0
        )
        self.train_dataset = full_dataset[split_indices["train"]]
        self.val_dataset = full_dataset[split_indices["dev"]]
        self.test_dataset = full_dataset[split_indices["test"]]

        return [None, None, None]

    def init_class(self, args: argparse.ArgumentParser, split_group: str) -> None:
        """Perform Class-Specific init methods
        Default is to load JSON dataset
        """
        pass

    def get_split_group_dataset(
        self, processed_dataset, split_group: Literal["train", "dev", "test"]
    ):
        if split_group == "train":
            return self.train_dataset
        elif split_group == "dev":
            return self.val_dataset
        elif split_group == "test":
            return self.test_dataset

    def __getitem__(self, index):
        try:
            return self.dataset.get(index)

        except Exception as e:
            item = self.dataset.metadata_json[index]
            print(f"Could not load sample: {item['sample_id']}, due to exception {e}")

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        """
        Assign samples to data splits

        Args:
            metadata_json (dict): raw json dataset loaded
        """
        if not self.already_assigned_splits:
            splits = {"train": [], "dev": [], "test": []}
            if self.args.assign_splits:
                if self.args.split_type == "random":
                    np.random.seed(seed)
                    for idx in range(len(metadata_json)):
                        sample = metadata_json[idx]
                        prot_path = os.path.join(
                            self.args.structures_dir,
                            f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
                        )
                        if os.path.exists(prot_path):
                            sample["split"] = np.random.choice(
                                ["train", "dev", "test"], p=split_probs
                            )
                            splits[sample["split"]].append(idx)

                elif self.args.split_type == "mmseqs":
                    # TODO: Run mmseqs on 'structure_sequence'
                    to_split = {}
                    uniprot2cluster = pickle.load(
                        open(
                            "/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_mmseq_clusters.p",
                            "rb",
                        )
                    )
                    clusters = list(uniprot2cluster.values())
                    clusters = sorted(list(set(clusters)))
                    np.random.seed(seed)
                    np.random.shuffle(clusters)
                    split_indices = np.ceil(
                        np.cumsum(np.array(split_probs) * len(clusters))
                    ).astype(int)
                    split_indices = np.concatenate([[0], split_indices])

                    for i in range(len(split_indices) - 1):
                        to_split.update(
                            {
                                cluster: ["train", "dev", "test"][i]
                                for cluster in clusters[
                                    split_indices[i] : split_indices[i + 1]
                                ]
                            }
                        )
                    for idx in range(len(metadata_json)):
                        sample = metadata_json[idx]
                        prot_path = os.path.join(
                            self.args.structures_dir,
                            f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
                        )
                        if os.path.exists(prot_path):
                            cluster = uniprot2cluster[sample["uniprot_id"]]
                            sample["split"] = to_split[cluster]
                            splits[sample["split"]].append(idx)
                else:
                    raise NotImplementedError(
                        f"Yet to implment {self.args.split_type} splitting"
                    )
            else:
                for idx in range(len(metadata_json)):
                    sample = metadata_json[idx]
                    prot_path = os.path.join(
                        self.args.structures_dir,
                        f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
                    )
                    if os.path.exists(prot_path):
                        splits[sample["split"]].append(idx)

            self.already_assigned_splits = True
            return splits
        else:
            pass

    @staticmethod
    def add_args(parser) -> None:
        super(QuickProtNox, QuickProtNox).add_args(parser)
        parser.add_argument(
            "--root_dir",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Metabo/quickprot_caches/",
            help="Permute smiles in reactants and in products as augmentation",
        )
        parser.add_argument(
            "--structures_dir",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Metabo/AlphaFoldEnzymes/",
            help="Dir of pdb/cif structures",
        )
        parser.add_argument(
            "--precompute_and_cache_node_embeddings",
            action="store_true",
            default=False,
            help="To precompute and cache esm embeddings",
        )


class QuickProtDataset(Dataset):
    def __init__(
        self,
        root,
        structures_dir,
        metadata_dir,
        transform=None,
        pre_transform=None,
        protein_resolution="residue",
        graph_edge_args={"knn_size": 10},
        center_protein=True,
        precompute_and_cache_node_embeddings=True,
        esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
        get_sequences_from_structure=False,
        node_embeddings_args={},
    ):
        """
        Abstract Dataset
        params: root - data directory in which raw data (ie protein files are saved) and where processed data will be saved
        params: transform - a function that will be applied in __getitem__ before batching (takes a PyG Data obj)
        params: pre_transform - a function that will be applied before caching the dataset (takes a PyG Data obj)
        params: pre_filter - a function that takes a PyG Data obj and decides whether or not to include it in the dataset (used for splitting)

        constructs: standard PyG Dataset obj, which can be fed to a DataLoader for batching
        # 1. Set args
        # 2. Load metadata
        # 3. Assign splits based on metadata
        # 4. Run preprocessing and caching using PyG Dataset
        # 5. Print summary statement
        """
        # Args
        self.metadata_dir = metadata_dir
        self.structures_dir = structures_dir
        self.protein_resolution = protein_resolution
        self.graph_edge_args = graph_edge_args
        self.center_protein = center_protein
        self.precompute_and_cache_node_embeddings = precompute_and_cache_node_embeddings
        self.esm_dir = esm_dir
        self.get_sequences_from_structure = get_sequences_from_structure
        self.node_embeddings_args = node_embeddings_args
        # to avoid forking CUDA process
        # for seq issues
        model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        assert os.path.exists(metadata_dir)
        assert os.path.exists(structures_dir)
        assert os.path.exists(esm_dir)

        try:
            self.metadata_json = self.process_data_input(self.metadata_dir)
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(self.metadata_dir, e))
        self.protein_parser = self.get_sample_loader(self.raw_file_names)
        # Get sequences
        seqs = [
            (d["sequence"] if "sequence" in d else None) for d in self.metadata_json
        ]
        self.sample_ids  # run to set sample_ids
        if self.get_sequences_from_structure:
            # this can be very slow
            sequences = get_sequences(
                self.protein_parser, self.sample_ids, self.raw_file_names, seqs
            )
        else:
            sequences = seqs

        # Save each samples sequence in metadata
        for i, s in enumerate(self.metadata_json):
            s["structure_sequence"] = sequences[i]

        # TODO: debug if not precomputed
        if self.precompute_and_cache_node_embeddings:
            sample_id2embedding_path = precompute_node_embeddings(
                self.metadata_json,
                cache_dir=os.path.join(root, "precomputed_node_embeddings"),
                model_location=self.esm_dir,
            )
            # Save each samples embedding path in metadata
            for i, s in enumerate(self.metadata_json):
                s["embedding_path"] = sample_id2embedding_path[s["sample_id"]]

        # Run preprocessing and caching
        super().__init__(root, transform, pre_transform, self.pre_filter)
        # Leave for quickprot but Nox prints this already
        # self.print_summary_statement(split_group)

    # TODO: See if I can get rid of this, just read from metadata_json instead
    @property
    def raw_file_names(self):
        if "protein_path" in self.metadata_json[0]:
            return [d["protein_path"] for d in self.metadata_json]
        elif "uniprot_id" in self.metadata_json[0]:
            return [
                os.path.join(
                    self.structures_dir, f"AF-{d['uniprot_id']}-F1-model_v4.cif"
                )
                for d in self.metadata_json
            ]
        else:
            raise ValueError(
                "Metadata must contain either `protein_path` or `uniprot_id`"
            )

    @property
    def sample_ids(self):
        # set sample ids
        if "sample_id" in self.metadata_json[0]:
            return [d["sample_id"] for d in self.metadata_json]
        elif "uniprot_id" in self.metadata_json[0]:
            for d in self.metadata_json:
                d["sample_id"] = d["uniprot_id"]
            return [d["sample_id"] for d in self.metadata_json]
        else:
            raise ValueError("Metadata must contain either `sample_id` or `uniprot_id`")

    def get_sample_loader(self, filenames):
        if filenames[0].endswith(".cif"):
            parser = Bio.PDB.MMCIFParser()
        elif filenames[0].endswith(".pdb"):
            parser = Bio.PDB.PDBParser()
        else:
            raise ValueError("Invalid structure file format, cannot infer parser")
        return parser

    def process_data_input(self, file) -> list:
        try:
            if isinstance(file, str):
                if file.endswith(".csv"):
                    # Read the CSV file and return a list of file paths to the pdb or cif files
                    with open(file, "r") as f:
                        df = pd.read_csv(f)
                        return df.to_dict(orient="records")
                elif file.endswith(".json"):
                    # Load the JSON file and return a list of file paths to the pdb or cif files
                    with open(file, "r") as f:
                        return json.load(f)
                elif file.endswith(".pkl"):
                    # Load the pickle file and return a list of file paths to the pdb or cif files
                    with open(file, "rb") as f:
                        return pickle.load(f)
                else:
                    raise ValueError("Invalid file format")
            else:
                raise TypeError(
                    "Expected a path to a csv/pkl/json, got {}.".format(type(file))
                )
        except Exception as e:
            raise ValueError(f"Could not load the dataset because of {e} exception")

    # TODO: See if I can get rid of this, just read from metadata_json instead
    @property
    def processed_file_names(self):
        # checks if structure file exists and if it does then it adds the filename to the list
        return [
            filename.split("-")[-3] + "_graph.pt"
            for filename in self.raw_file_names
            if os.path.exists(filename)
        ]

    # Figure out how to do this so that it downloads AF structures if a certain argument is given
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)

    def process(self):
        skipped = 0
        skipped_exceptions = 0
        print("This can take a while, but only runs once! :)")
        for idx, sample in enumerate(
            tqdm(
                self.metadata_json,
                total=len(self.metadata_json),
            )
        ):
            # TODO: Move get structure_seq here
            try:
                if "protein_path" not in sample:
                    self.compute_protein_paths()

                raw_path = sample["protein_path"]
                sample_id = sample["sample_id"]

                if os.path.exists(
                    os.path.join(self.processed_dir, f"{sample_id}_graph.pt")
                ):
                    continue

                # skip samples here if needed
                if self.pre_filter is not None and self.pre_filter(sample):
                    skipped += 1
                    continue
                # parse pdb
                all_res, all_atom, all_pos = read_structure_file(
                    self.protein_parser, raw_path, sample_id
                )
                # filter resolution of protein (backbone, atomic, etc.)
                atom_names, seq, pos = filter_resolution(
                    all_res,
                    all_atom,
                    all_pos,
                    protein_resolution=self.protein_resolution,
                )
                # generate graph
                data = build_graph(atom_names, seq, pos, sample_id)
                # kNN graph
                data = compute_graph_edges(data, **self.graph_edge_args)
                if self.center_protein:
                    center = data["receptor"].pos.mean(dim=0, keepdim=True)
                    data["receptor"].pos = data["receptor"].pos - center
                    data.center = center

                # TODO: Add node embeddings either pre-computed or on the fly
                if self.precompute_and_cache_node_embeddings:
                    node_embedding = torch.load(
                        sample["embedding_path"]
                    )
                else:
                    node_embedding = compute_node_embedding(
                        data, **self.node_embeddings_args
                    )
                # Fix sequence length mismatches
                if len(data["receptor"].seq) != node_embedding.shape[0]:
                    print("Computing seq embedding for mismatched seq length")
                    sequences = get_sequences(
                        self.protein_parser,
                        [sample["sample_id"]],
                        [os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")],
                    )
                    self.node_embeddings_args.update({"model": self.esm_model, "model_location": self.esm_dir, "alphabet": self.alphabet, "batch_converter": self.batch_converter})
                    data.structure_sequence = sequences[0]
                    data["receptor"].x = compute_node_embedding(
                        data, **self.node_embeddings_args
                    )
                else:
                    data["receptor"].x = node_embedding
                
                if self.pre_filter is not None and self.pre_filter(data):
                    skipped += 1
                    continue

                torch.save(
                    data, os.path.join(self.processed_dir, f"{sample_id}_graph.pt")
                )
            except Exception as e:
                skipped_exceptions += 1
                print(f"Skipped sample because of exception {e}")

        print(f"Skipped {skipped} samples due to missing structures")
        print(f"Skipped {skipped_exceptions} samples due to exceptions")

    def compute_protein_paths(self):
        for sample in self.metadata_json:
            sample["protein_path"] = os.path.join(self.structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif",)

    def pre_filter(self, data) -> bool:
        """
        Return True if sample should be skipped and not included in dataset
        """
        if isinstance(data, dict):
            if not os.path.exists(data["protein_path"]):
                return True
        return False

    def add_additional_data_to_graph(self, data, sample):
        """
        Placeholder to add additional data to graph

        eg. adding a ligand to the graph
        ligand = utils.load_ligand_path(sample['ligand_path'])
        data = utils.add_ligand_to_graph(data, ligand)
        """
        skipped_keys = set(["protein_path", "embedding_path"])
        for key in sample.keys():
            if not key in skipped_keys and key not in data.to_dict().keys():
                data[key] = sample[key]
        data["mol_data"] = from_smiles(sample["smiles"])
        return data

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        return f"Dataset contains {len(self)} samples."

    def print_summary_statement(self):
        statement = f"DATASET CREATED FOR QuickProtDataset.\n{self.SUMMARY_STATEMENT}"
        print(statement)

    def len(self):
        return len(self.metadata_json)

    def get(self, idx):
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        try:
            sample = self.metadata_json[idx]
            if len(sample["sequence"]) > 1022:
                print(
                    f"Skipped sample {sample['sample_id']} because len of seq is {len(sample['sequence'])}"
                )
                return None
            
            data = torch.load(os.path.join(self.processed_dir, f"{sample['sample_id']}_graph.pt"))

            # adds other keys in sample to graph
            data = self.add_additional_data_to_graph(data, sample)
            # TODO: remove in the future
            if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                data["receptor"].x = data.x
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if hasattr(data, "embedding_path"):
                delattr(data, "embedding_path")
            if hasattr(data, "protein_path"):
                delattr(data, "protein_path")
            if hasattr(data, "sample_hash"):
                delattr(data, "sample_hash")
            return data

        except Exception as e:
            print(LOAD_FAIL_MSG.format(sample["sample_id"], e))
