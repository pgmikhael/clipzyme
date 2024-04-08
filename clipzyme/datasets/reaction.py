import os
import argparse
from typing import List, Union
from rich import print
from tqdm import tqdm
import copy
import pandas as pd
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
import torch
from torch.utils import data
from torch_geometric.data import Data as pygData
from esm import pretrained
from clipzyme.utils.registry import register_object
from clipzyme.utils.wln_processing import get_bond_changes
from clipzyme.utils.screening import process_mapped_reaction
from clipzyme.utils.protein_utils import (
    read_structure_file,
    filter_resolution,
    build_graph,
    compute_graph_edges,
    compute_node_embedding,
)


protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})


@register_object("reactions_dataset", "dataset")
class ReactionDataset(data.Dataset):
    def __init__(
        self,
        args: argparse.Namespace = None,
        dataset_file_path: str = None,
        esm_dir: str = None,
        protein_cache_dir: str = None,
        use_as_protein_encoder: bool = False,
        use_as_reaction_encoder: bool = False,
    ) -> None:
        """
        Create a dataset of reactions and proteins from a CSV file

        Parameters
        ----------
        args: argparse.Namespace
            Arguments from command line
        dataset_file_path: str
            Path to CSV file with headers ['reaction', 'sequence', 'protein_id', 'cif']
        esm_dir: str
            Path to ESM model directory
        protein_cache_dir: str
            Directory to save/load protein graphs
        use_as_protein_encoder: bool
            Use dataset as protein encoder and do not consider reactions in data filtering
        use_as_reaction_encoder: bool
            Use dataset as reaction encoder and do not consider proteins in data filtering

        Raises
        ------
        ValueError
            If CSV file does not have headers ['reaction', 'sequence', 'protein_id', 'cif']
        """
        super(ReactionDataset, self).__init__()

        if args is None:
            csv_path = dataset_file_path
            esm_dir = esm_dir
            protein_cache_dir = protein_cache_dir
            self.use_as_protein_encoder = use_as_protein_encoder
            self.use_as_reaction_encoder = use_as_reaction_encoder
        else:
            csv_path = args.dataset_file_path
            esm_dir = args.esm_dir
            protein_cache_dir = args.protein_cache_dir
            self.use_as_protein_encoder = args.use_as_protein_encoder
            self.use_as_reaction_encoder = args.use_as_reaction_encoder

        # check if csv file has correct headers
        with open(csv_path, "r") as f:
            headers = f.readline().strip().split(",")
            if (
                "reaction" not in headers
                or "protein_id" not in headers
                or "sequence" not in headers
                or "cif" not in headers
            ):
                raise ValueError(
                    "CSV file must have headers 'reaction', 'sequence', 'protein_id', and 'cif'"
                )

        print("Loading ESM model")
        esm_path = os.path.join(esm_dir, "esm2_t33_650M_UR50D.pt")
        model, alphabet = pretrained.load_model_and_alphabet(esm_path)
        self.esm_model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        print("Preparing dataset")
        csv_dataset = pd.read_csv(csv_path)
        self.dataset = self.create_dataset(csv_dataset)

        self.protein_cache_dir = protein_cache_dir

        print(self.SUMMARY_STATEMENT)

    def create_dataset(self, csv_dataset: pd.DataFrame) -> List[dict]:
        """
        Create dataset of reactions and proteins from CSV file

        Parameters
        ----------
        csv_path : str
            Path to CSV file with headers ['reaction', 'sequence', 'cif']

        Returns
        -------
        List[dict]
            _description_
        """

        dataset = []

        for rowid, row in tqdm(
            csv_dataset.iterrows(),
            desc="Building dataset",
            total=len(csv_dataset),
            ncols=100,
        ):
            reactants, products = row["reaction"].split(">>")
            reactants = reactants.split(".")
            products = products.split(".")
            sample = {
                "protein_id": row["protein_id"],
                "sequence": row["sequence"],
                "reaction": row["reaction"],
                "reactants": reactants,
                "products": products,
                "cif_path": row["cif"],
                "sample_id": f"sample_{rowid}",
            }
            # get bond changes
            try:
                sample["bond_changes"] = get_bond_changes(sample["reaction"])
            except Exception as e:
                sample["bond_changes"] = []

            if self.skip_sample(sample):
                continue

            # add reaction sample to dataset
            dataset.append(sample)

        return dataset

    def skip_sample(self, sample: dict) -> bool:
        """
        Skip sample if criteria are not met

        Parameters
        ----------
        sample : dict
            Sample dictionary from dataset

        Returns
        -------
        bool
            True if sample should be skipped
        """
        # if dataset is not used only as reaction encoder
        if not self.use_as_reaction_encoder:
            # if sequence is unknown
            sequence = sample["sequence"]
            if len(sequence) == 0:
                return True

            if len(sequence) > 650:
                return True

            # check if cif file exists
            if not os.path.exists(sample["cif_path"]):
                return True

        # if dataset is not used only as protein encoder
        if not self.use_as_protein_encoder:
            # if no bond changes
            if len(sample["bond_changes"]) == 0:
                return True

        return False

    def create_protein_graph(self, sample: dict) -> Union[pygData, None]:
        """
        Create pyg protein graph from CIF file

        Parameters
        ----------
        sample : dict
            dataset sample

        Returns
        -------
        data
            pygData object with protein graph
        """
        try:
            raw_path = sample["cif_path"]
            sample_id = sample["sample_id"]
            protein_parser = Bio.PDB.MMCIFParser()
            protein_resolution = "residue"
            graph_edge_args = {"knn_size": 10}
            center_protein = True

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

            sequence = sample["sequence"]
            data.structure_sequence = sequence

            node_embeddings_args = {
                "model": self.esm_model,
                "model_location": "",
                "alphabet": self.alphabet,
                "batch_converter": self.batch_converter,
            }
            node_embedding = compute_node_embedding(data, **node_embeddings_args)
            # Fix sequence length mismatches
            if len(data["receptor"].seq) != node_embedding.shape[0]:
                print("Computing seq embedding for mismatched seq length")
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
                if d_key not in keep_keys:
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

            return data

        except Exception as e:
            print(
                f"Could not create protein graph for:  {sample['protein_id']} because of the exception {e}"
            )
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_id = sample["sample_id"]

        try:
            reactants, products = process_mapped_reaction(
                sample["reaction"],
                sample["bond_changes"],
            )
            protein_id = sample["protein_id"]

            item = {
                "reactants": reactants,
                "products": products,
                "protein_id": protein_id,
                "sample_id": sample_id,
                "cif_path": sample["cif_path"],
                "sequence": sample["sequence"],
            }

            if self.protein_cache_dir:
                graph_path_cache = os.path.join(
                    self.protein_cache_dir, f"{protein_id}.pt"
                )
                try:
                    data = torch.load(graph_path_cache)
                    if data is None:
                        data = self.create_protein_graph(item)
                        torch.save(data, graph_path_cache)
                except:
                    data = self.create_protein_graph(item)
                    torch.save(data, graph_path_cache)
            else:
                data = self.create_protein_graph(item)

                item["graph"] = data

            return item

        except Exception as e:
            print(f"Could not load sample {sample_id} because of an exception {e}")

    @property
    def SUMMARY_STATEMENT(self) -> None:
        reactions = [d["reaction"] for d in self.dataset]
        proteins = [d["sequence"] for d in self.dataset]

        statement = f""" 
        DATASET CREATED:
        * Number of samples: {len(self.dataset)}
        * Number of reactions: {len(set(reactions))}
        * Number of proteins: {len(set(proteins))}
        """
        return statement
