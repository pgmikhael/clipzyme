from typing import List, Literal
import traceback, warnings, os, pickle
import pandas as pd
import argparse
import copy
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
from esm import pretrained
from nox.utils.registry import register_object
from nox.utils.messages import METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG
from nox.datasets.abstract import AbstractDataset
from nox.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    compute_node_embedding,
)


@register_object("clean_ec", "dataset")
class CLEAN_EC(AbstractDataset):
    """https://www.science.org/doi/10.1126/science.adf2465"""

    def __init__(self, args, split_group) -> None:
        if args.use_protein_graphs:
            self.esm_dir = args.esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(args.esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
        super(CLEAN_EC, CLEAN_EC).__init__(self, args, split_group)

    def create_protein_graph(self, sample):
        try:
            raw_path = os.path.join(
                self.args.protein_structures_dir,
                f"AF-{sample['uniprot_id']}-F1-model_v4.cif",
            )
            protein_args = {
                "sample_id": sample["sample_id"],
                "protein_parser": Bio.PDB.MMCIFParser(),
                "protein_resolution": "residue",
                "graph_edge_args": {"knn_size": 10},
                "center_protein": True,
            }

            sample_id = protein_args["sample_id"]
            protein_parser = protein_args["protein_parser"]
            protein_resolution = protein_args["protein_resolution"]
            graph_edge_args = protein_args["graph_edge_args"]
            center_protein = protein_args["center_protein"]

            # parse pdb
            all_res, all_atom, all_pos = read_structure_file(
                protein_parser, raw_path, sample_id
            )
            # filter resolution of protein (backbone, atomic, etc.)
            atom_names, seq, pos = filter_resolution(
                all_res,
                all_atom,
                all_pos,
                protein_resolution=protein_resolution,
            )
            # generate graph
            data = build_graph(atom_names, seq, pos, sample_id)
            # kNN graph
            data = compute_graph_edges(data, **graph_edge_args)
            if center_protein:
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                data["receptor"].pos = data["receptor"].pos - center
                data.center = center
            uniprot_id = sample["uniprot_id"]
            sequence = self.uniprot2sequence[uniprot_id]
            data.structure_sequence = self.uniprot2sequence[uniprot_id]

            node_embeddings_args = {
                "model": self.esm_model,
                "model_location": self.esm_dir,
                "alphabet": self.alphabet,
                "batch_converter": self.batch_converter,
            }

            embedding_path = os.path.join(
                self.args.protein_graphs_dir,
                "precomputed_node_embeddings",
                f"{sample['uniprot_id']}.pt",
            )

            if os.path.exists(embedding_path):
                node_embedding = torch.load(sample["embedding_path"])
            else:
                node_embedding = compute_node_embedding(data, **node_embeddings_args)
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
                protein_letters_3to1.update(
                    {k.upper(): v for k, v in protein_letters_3to1.items()}
                )
                AA_seq = ""
                for char in seq:
                    AA_seq += protein_letters_3to1[char]

                data.structure_sequence = AA_seq
                data["receptor"].x = compute_node_embedding(
                    data, **node_embeddings_args
                )
            else:
                data["receptor"].x = node_embedding

            if len(data["receptor"].seq) != data["receptor"].x.shape[0]:
                return None

            return data

        except Exception as e:
            print(
                f"Create prot graph: Could not load sample {sample['uniprot_id']} because of the exception {e}"
            )
            return None

    def load_dataset(self, args: argparse.ArgumentParser) -> None:
        """Loads dataset file

        Args:
            args (argparse.ArgumentParser)

        Raises:
            Exception: Unable to load
        """
        try:
            self.metadata_json = {}
            for filepath, split in [
                (args.dataset_file_path, "train"),
                (args.test_dataset_path, "test"),
            ]:
                csv_dataset = pd.read_csv(filepath, delimiter="\t")
                self.metadata_json[split] = csv_dataset.to_dict("records")
            self.alphafold_files = pickle.load(
                open("/Mounts/rbg-storage1/datasets/Metabo/alphafold_enzymes.p", "rb")
            )
            self.quickprot_caches = pickle.load(
                open("/Mounts/rbg-storage1/datasets/Metabo/quickprot_caches.p", "rb")
            )
            self.uniprot2sequence = {
                d["Entry"]: d["Sequence"]
                for _, dct in self.metadata_json.items()
                for d in dct
            }
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

    def post_process(self, args):
        pickle.dump(
            self.quickprot_caches,
            open("/Mounts/rbg-storage1/datasets/Metabo/quickprot_caches.p", "wb"),
        )

        ecs = sorted(
            list(
                set(
                    [
                        ec
                        for d in self.metadata_json["train"]
                        for ec in d["EC number"].split(";")
                    ]
                )
            )
        )
        ecs = [ec.split(".") for ec in ecs]
        args.ec_levels = {}
        for level in range(1, 5, 1):
            unique_classes = sorted(list(set(".".join(ec[:level]) for ec in ecs)))
            args.ec_levels[str(level)] = {c: i for i, c in enumerate(unique_classes)}
        args.num_classes = len(args.ec_levels["4"])

    def create_dataset(
        self, split_group: Literal["train", "dev", "test"]
    ) -> List[dict]:
        dataset = []
        sgroup = "test" if split_group == "test" else "train"
        for entry in tqdm(self.metadata_json[sgroup]):
            if self.skip_sample(entry):
                continue

            ec = entry["EC number"]
            sample = {
                "uniprot_id": entry["Entry"],
                "ec": ec.split(";"),
                "sequence": entry["Sequence"],
                "sample_id": entry["Entry"],
            }

            # make prot graph if missing
            if self.args.use_protein_graphs:
                graph_path = os.path.join(
                    self.args.protein_graphs_dir,
                    "processed",
                    f"{sample['uniprot_id']}_graph.pt",
                )
                if sample["uniprot_id"] not in self.alphafold_files:
                    continue
                # if sample["uniprot_id"] not in self.quickprot_caches:
                #     print("Generating none existent protein graph")
                #     data = self.create_protein_graph(sample)
                #     if data is None:
                #         raise Exception("Could not generate protein graph")
                #     torch.save(data, graph_path)
                #     self.quickprot_caches.add(sample["uniprot_id"])

            dataset.append(sample)

        return dataset

    def skip_sample(self, sample) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        return False

    def get_split_group_dataset(self, processed_dataset, split_group):
        return [
            sample for sample in processed_dataset if sample["split"] == split_group
        ]

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
        np.random.seed(seed)
        if self.split_group == "test":
            for idx in range(len(metadata_json)):
                metadata_json[idx]["split"] = "test"
        else:
            assert (
                len(split_probs) == 2
            ), "`split_probs` must only consist of train and dev split fractions"

            ec2size = Counter([ec for d in metadata_json for ec in d["ec"]])

            for idx in range(len(metadata_json)):
                if any(ec2size[e] < 10 for e in metadata_json[idx]["ec"]):
                    metadata_json[idx]["split"] = "train"
                else:
                    metadata_json[idx]["split"] = np.random.choice(
                        ["train", "dev"], p=split_probs
                    )

        return metadata_json

    def __getitem__(self, index):
        """
        Fetch single sample from dataset

        Args:
            index (int): random index of sample from dataset

        Returns:
            sample (dict): a sample
        """
        sample = self.dataset[index]
        try:
            item = copy.deepcopy(sample)
            uniprot_id = item["uniprot_id"]

            # ecs as tensors
            for k, v in self.args.ec_levels.items():
                yvec = torch.zeros(len(v))
                for ec in item["ec"]:
                    split_ec = ec.split(".")
                    j = v[".".join(split_ec[: int(k)])]
                    yvec[j] = 1
                item[f"ec{k}"] = yvec

            if self.args.use_protein_graphs:
                # load the protein graph
                graph_path = os.path.join(
                    self.args.protein_graphs_dir,
                    "processed",
                    f"{item['uniprot_id']}_graph.pt",
                )
                try:
                    data = torch.load(graph_path)
                except:
                    data = self.create_protein_graph(item)
                    # torch.save(data, graph_path)

                if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
                    data["receptor"].x = data.x

                keep_keys = {
                    "receptor",
                    ("receptor", "contact", "receptor"),
                }

                data_keys = data.to_dict().keys()
                for d_key in data_keys:
                    if not d_key in keep_keys:
                        delattr(data, d_key)

                coors = data["receptor"].pos
                feats = data["receptor"].x
                edge_index = data["receptor", "contact", "receptor"].edge_index
                assert (
                    coors.shape[0] == feats.shape[0]
                ), f"Number of nodes do not match between coors ({coors.shape[0]}) and feats ({feats.shape[0]})"

                assert (
                    max(edge_index[0]) < coors.shape[0]
                    and max(edge_index[1]) < coors.shape[0]
                ), "Edge index contains node indices not present in coors"

                if self.args.use_protein_msa:
                    msa_embed = torch.load(
                        os.path.join(self.args.protein_msa_dir, f"{uniprot_id}.pt")
                    )
                    data["receptor"].x = torch.concat([feats, msa_embed], dim=-1)

                item["graph"] = data

            return item
        except Exception:
            warnings.warn(
                LOAD_FAIL_MSG.format(sample["sample_id"], traceback.print_exc())
            )

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)

        parser.add_argument(
            "--test_dataset_path",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/datasets/new.csv",
            choices=[
                "/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/split100.csv",
                "/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/datasets/new.csv",
                "/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/datasets/price.csv",
                "/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/datasets/halogenase.csv",
            ],
            help="path to test set used in CLEAN",
        )
        parser.add_argument(
            "--esm_dir",
            type=str,
            default="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
            help="directory to load esm model from",
        )
        parser.add_argument(
            "--use_protein_graphs",
            action="store_true",
            default=False,
            help="whether to use and generate protein graphs",
        )
        parser.add_argument(
            "--protein_graphs_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--protein_structures_dir",
            type=str,
            default=None,
            help="directory to load protein graphs from",
        )
        parser.add_argument(
            "--use_protein_msa",
            action="store_true",
            default=False,
            help="whether to use and generate protein MSAs",
        )
        parser.add_argument(
            "--protein_msa_dir",
            type=str,
            default="/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/hhblits_embeds",
            help="directory where msa transformer embeddings are stored.",
        )

    @staticmethod
    def set_args(args) -> None:
        args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/Metabo/CLEAN/app/data/split100.csv"
        )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        num_proteins = len(set([s["sequence"] for s in self.dataset]))
        num_ecs = len(set([ec for s in self.dataset for ec in s["ec"]]))
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of proteins: {num_proteins}
        * Number of ECs: {num_ecs}
        """
        return statement
