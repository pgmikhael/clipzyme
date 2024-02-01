from typing import List
import traceback, warnings, os, pickle
import argparse
import copy
import torch
from tqdm import tqdm
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
from esm import pretrained
from clipzyme.utils.registry import register_object
from clipzyme.utils.messages import METAFILE_NOTFOUND_ERR, LOAD_FAIL_MSG
from clipzyme.datasets.abstract import AbstractDataset
from clipzyme.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    compute_node_embedding,
)

protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})


@register_object("screening_enzymes", "dataset")
class ScreeningEnzymes(AbstractDataset):
    def __init__(self, args, split_group) -> None:
        if args.use_protein_graphs:
            self.esm_dir = args.esm_dir
            model, alphabet = pretrained.load_model_and_alphabet(args.esm_dir)
            self.esm_model = model
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
        super(ScreeningEnzymes, ScreeningEnzymes).__init__(self, args, split_group)

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

            try:
                node_embedding = torch.load(sample["embedding_path"])
            except:
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
            self.metadata_json = pickle.load(open(args.dataset_file_path, "rb"))
            self.alphafold_files = pickle.load(
                open("/home/datasets/alphafold_enzymes.p", "rb")
            )
            self.quickprot_caches = pickle.load(
                open("/home/datasets/quickprot_caches.p", "rb")
            )
            self.msa_files = pickle.load(
                open(
                    "/home/datasets/uniprot2msa_embedding.p",
                    "rb",
                )
            )
            self.uniprot2sequence = self.metadata_json
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

    def create_dataset(self, split_group) -> List[dict]:
        dataset = []

        for uniprot_id, sequence in tqdm(self.metadata_json.items(), ncols=100):
            sample = {
                "uniprot_id": uniprot_id,
                "sequence": sequence,
                "sample_id": uniprot_id,
            }

            if self.skip_sample(sample):
                continue

            dataset.append(sample)

        return dataset

    def skip_sample(self, sample) -> bool:
        """
        Return True if sample should be skipped and not included in data
        """
        if "sequence" in sample:
            if (sample["sequence"] is None) or (len(sample["sequence"]) == 0):
                return True

            if (self.args.max_protein_length is not None) and len(
                sample["sequence"]
            ) > self.args.max_protein_length:
                return True

        if self.args.use_protein_graphs:
            if sample["uniprot_id"] not in self.alphafold_files:
                return True

        if self.args.use_protein_msa:
            if sample["uniprot_id"] not in self.msa_files:
                return True

        return False

    def get_split_group_dataset(self, processed_dataset, split_group):
        return processed_dataset

    def assign_splits(self, metadata_json, split_probs, seed=0) -> None:
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

            if self.args.use_protein_graphs:
                if self.args.cache_path:
                    try:
                        graph_path_cache = os.path.join(
                            self.args.cache_path,
                            f"{item['uniprot_id']}_graph.pt",
                        )
                        data = torch.load(graph_path_cache)
                        if data is None:
                            data = self.load_protein_graph(item)
                            torch.save(data, graph_path_cache)
                    except:
                        data = self.load_protein_graph(item)
                        torch.save(data, graph_path_cache)
                else:
                    data = self.load_protein_graph(item)

                if self.args.use_protein_msa:
                    feats = data["receptor"].x
                    msa_embed = torch.load(self.msa_files[uniprot_id])
                    if self.args.replace_esm_with_msa:
                        data["receptor"].x = msa_embed
                    else:
                        data["receptor"].x = torch.concat([feats, msa_embed], dim=-1)
                        data["receptor"].msa = msa_embed

                item["graph"] = data

            return item
        except Exception:
            warnings.warn(
                LOAD_FAIL_MSG.format(sample["sample_id"], traceback.print_exc())
            )

    def load_protein_graph(self, item):
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
            torch.save(data, graph_path)
        if data is None:
            try:
                data = self.create_protein_graph(item)
                torch.save(data, graph_path)
            except:
                return

        if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
            data["receptor"].x = data.x

        if not hasattr(data, "structure_sequence"):
            data.structure_sequence = "".join(
                [protein_letters_3to1[char] for char in data["receptor"].seq]
            )

        keep_keys = {
            "receptor",
            "structure_sequence",
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
            max(edge_index[0]) < coors.shape[0] and max(edge_index[1]) < coors.shape[0]
        ), "Edge index contains node indices not present in coors"

        return data

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        AbstractDataset.add_args(parser)
        parser.add_argument(
            "--esm_dir",
            type=str,
            default="/home/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt",
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
            default="/home/datasets/EnzymeMap/embed_msa_transformer",
            help="directory where msa transformer embeddings are stored.",
        )
        parser.add_argument(
            "--max_protein_length",
            type=int,
            default=None,
            help="skip proteins longer than max_protein_length",
        )
        parser.add_argument(
            "--replace_esm_with_msa",
            action="store_true",
            default=False,
            help="whether to use ONLY the protein MSAs",
        )

    @staticmethod
    def set_args(args) -> None:
        args.dataset_file_path = (
            "/home/datasets/uniprot2sequence_standard_set_structs.p"
        )

    @property
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        num_proteins = len(set([s["sequence"] for s in self.dataset]))
        statement = f""" 
        * Number of samples: {len(self.dataset)}
        * Number of proteins: {num_proteins}
        """
        return statement
