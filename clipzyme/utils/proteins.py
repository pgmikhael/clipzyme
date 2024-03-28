import argparse
from typing import Union, Tuple, Any, List
import os, sys
import pickle
import random
import warnings
import itertools
from collections import defaultdict
import csv
import hashlib

# from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData
from rich import print as rprint

from scipy.spatial.transform import Rotation

import Bio

import Bio.PDB

warnings.filterwarnings("ignore", category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
from Bio.Data.IUPACData import protein_letters_3to1


def read_files(data_to_load, args):
    pdb_id2prot_dict = {}
    for sample in data_to_load:
        pdb_path = sample["path"]
        pdb_id = sample["sample_id"]
        item = pdb_id2prot_dict.get(pdb_id)
        # avoid recomputing proteins
        if item is None:
            tup_all_res_all_atom_all_pos = parse_pdb(pdb_id, pdb_path)
            sample["receptor"] = tup_all_res_all_atom_all_pos
            pdb_id2prot_dict[pdb_id] = sample

    # write to cache
    data_cache = os.path.join(
        args.protein_cache_path, f"{args.hash_saved_data}_cached_prot_data.pkl"
    )
    if args.debug:
        data_cache = data_cache.replace(".pkl", "_debug.pkl")

    if not args.no_graph_cache and not os.path.exists(data_cache):
        if not os.path.exists(os.path.dirname(data_cache)):
            try:
                os.makedirs(os.path.dirname(data_cache))
            except OSError as exc:  # Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise
        with open(data_cache, "wb+") as f:
            pickle.dump(pdb_id2prot_dict, f)
    return pdb_id2prot_dict


def parse_pdb(pdb_id, pdb_path, models=None, chains=None):
    """
    Parse PDB file via Biopython
    (can cause issues with multi-processing)

    Args:
        pdb_id (str): PDB ID
        pdb_path (str): path to PDB file
        models (list): list of models to select
        chains (list): list of chains to select

    Returns:
        Tuple: (residue sequence, coordinates, atom sequence)
    """
    all_res, all_atom, all_pos = [], [], []
    # biopython PDB parser
    if pdb_path[-3:] == "pdb":
        parser = Bio.PDB.PDBParser()
    elif pdb_path[-3:] == "cif":
        parser = Bio.PDB.MMCIFParser()
    structure = parser.get_structure(pdb_id, pdb_path)  # name, path
    # (optional) select subset of models
    if models is not None:
        models = [structure[m] for m in models]
    else:
        models = structure
    for model in models:
        # (optional) select subset of chains
        if chains is not None:
            chains = [model[c] for c in chains if c in model]
        else:
            chains = model
        # retrieve all atoms in desired chains
        for chain in chains:
            res = chain.get_residues()
            for res in chain:
                # ignore HETATM records
                if res.id[0] != " ":
                    continue
                for atom in res:
                    all_res.append(res.get_resname())
                    all_pos.append(torch.from_numpy(atom.get_coord()))
                    all_atom.append((atom.get_name(), atom.element))
    all_pos = torch.stack(all_pos, dim=0)
    return all_res, all_atom, all_pos


def process_data(pdb_id2prot_dict, args):
    """
    1. Converts protein to resolution of interest (atomic level, backbone, or residue)
    2. Converts protein to HeteroData graph object
    3. Caches prot_dicts and graphs

    Args:
        pdb_id2prot_dict (dict): dictionary of protein data
        args (argparse.Namespace): command line arguments

    Returns:

    """
    # check if cache exists
    graph_cache = os.path.join(
        args.protein_cache_path,
        f"{args.hash_saved_data}_cached_prot_graph_{args.protein_resolution}.pkl",
    )
    if args.debug:
        graph_cache = graph_cache.replace(".pkl", "_debug.pkl")

    if not args.no_graph_cache and os.path.exists(graph_cache):
        with open(graph_cache, "rb") as f:
            pdb_id2prot_dict = pickle.load(f)
            # print("Loaded processed data from cache")
            return pdb_id2prot_dict

    # select subset of residues that match desired resolution (e.g. residue-level vs. atom-level)
    for item in pdb_id2prot_dict.values():
        subset = convert_pdb(*item["receptor"], args)
        item["receptor_atom"] = subset[0]
        item["receptor_seq"] = subset[1]
        item["receptor_xyz"] = subset[2]
        # if len(subset[2]) != len(item['sequence']):
        # return None

    # convert to HeteroData graph objects
    for item in pdb_id2prot_dict.values():
        item["graph"] = to_graph(item, args)

    if not args.no_graph_cache:
        with open(graph_cache, "wb+") as f:
            pickle.dump(pdb_id2prot_dict, f)

    return pdb_id2prot_dict


def convert_pdb(all_res, all_atom, all_pos, args):
    """
    Unify PDB representation across different dataset formats.
    Given all residues, coordinates, and atoms, select subset
    of all atoms to keep.

    Args:
        all_res (list): list of all residue names
        all_atom (list): list of all atom names
        all_pos (torch.Tensor): tensor of all atom coordinates

    Returns:
        tuple(list, torch.Tensor) seq, pos
    """
    if args.protein_resolution == "atom":
        atoms_to_keep = None
    elif args.protein_resolution == "backbone":
        atoms_to_keep = {"N", "CA", "O", "CB"}
    elif args.protein_resolution == "residue":
        atoms_to_keep = {"CA"}
    else:
        raise Exception(f"invalid resolution {args.protein_resolution}")

    # this converts atoms to residues or to backbone atoms
    if atoms_to_keep is not None:
        to_keep_idx, atom_names = [], []
        for i, a in enumerate(all_atom):
            if a[0] in atoms_to_keep:  # atom name
                to_keep_idx.append(i)
                atom_names.append(a[1])  # element symbol
        seq = [all_res[i] for i in to_keep_idx]
        pos = all_pos[torch.tensor(to_keep_idx)]
    else:
        atom_names = all_atom
        seq = all_res
        pos = all_pos

    assert pos.shape[0] == len(seq) == len(atom_names)

    return atom_names, seq, pos


def to_graph(item, args):
    """
    Convert raw dictionary to PyTorch geometric object

    Args:
        item (dict): dictionary of protein data
        args (argparse.Namespace): command-line arguments

    Returns:
        HeteroData: PyTorch geometric object
    """
    data = HeteroData()
    data["name"] = item["sample_id"]
    # retrieve position and compute kNN
    data["receptor"].pos = item["receptor_xyz"].float()
    data["receptor"].x = item["receptor_seq"]  # _seq is residue id
    # kNN graph
    edge_index = knn_graph(data["receptor"].pos, args.knn_size)
    data["receptor", "contact", "receptor"].edge_index = edge_index
    # center receptor at origin (pdb can place protein in strange coordinates)
    center = data["receptor"].pos.mean(dim=0, keepdim=True)
    data["receptor"].pos = data["receptor"].pos - center
    data.center = center  # save old center just in case
    return data


def process_embed(pdb_id2prot_dict, args):
    """
    1. Tokenize protein sequences
    2. Compute ESM embeddings
    3. Cache embeddings

    Args:
        pdb_id2prot_dict (dict): dictionary of protein data
        args (argparse.Namespace): command line arguments

    Returns:
        pdb_id2prot_dict (dict): dictionary of protein data, now with esm node features
        data_params (dict): stores relevant esm parameters

    Raises:
        NotImplementedError: [if not using ESM tokenization]
    """
    data_params = {}
    if args.protein_dim > 0:
        pdb_id2prot_dict = compute_embeddings(pdb_id2prot_dict, args)
        data_params["num_residues"] = 23  # <cls> <sep> <pad>
        print("finished tokenizing residues with ESM")
    else:
        # # Rachel's code for alternative tokenization
        # # tokenize residues for non-ESM
        # tokenizer = tokenize(pdb_id2prot_dict.values(), "receptor_seq")
        # esm_model = None
        # data_params["num_residues"] = len(tokenizer)
        # data_params["tokenizer"] = tokenizer

        # # protein sequence tokenization
        # # tokenize atoms
        # atom_tokenizer = tokenize(pdb_id2prot_dict.values(), "receptor_atom")
        # data_params["atom_tokenizer"] = atom_tokenizer
        # print("finished tokenizing all inputs")
        raise NotImplementedError("Non-ESM tokenization not implemented")

    return pdb_id2prot_dict, data_params


def compute_embeddings(pdb_id2prot_dict, args):
    """
    Pre-compute ESM2 embeddings.
    """
    esm_cache = os.path.join(
        args.protein_cache_path, f"{args.hash_saved_data}_cached_prot_esm.pkl"
    )
    if args.debug:
        esm_cache = esm_cache.replace(".pkl", "_debug.pkl")
    # check if we already computed embeddings
    if not args.no_graph_cache and os.path.exists(esm_cache):
        with open(esm_cache, "rb") as f:
            pdb_id2prot_graph_node_feat = pickle.load(f)
        _save_esm_rep(pdb_id2prot_dict, pdb_id2prot_graph_node_feat)
        print("Loaded cached ESM embeddings")
        return pdb_id2prot_dict

    # print("Computing ESM embeddings")
    # load pretrained model
    torch.hub.set_dir(args.pretrained_hub_dir)
    esm_model, alphabet = torch.hub.load(
        "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
    )
    esm_model = esm_model.cuda().eval()
    tokenizer = alphabet.get_batch_converter()
    # convert to 3 letter codes
    aa_code = defaultdict(lambda: "<unk>")
    aa_code.update({k.upper(): v for k, v in protein_letters_3to1.items()})
    # fix ordering
    pdb_id2prot_dict_sorted = sorted(pdb_id2prot_dict)
    all_graphs = [pdb_id2prot_dict[pdb]["graph"] for pdb in pdb_id2prot_dict_sorted]
    rec_seqs = [g["receptor"].x for g in all_graphs]  # residue ids
    # this is the prot sequence
    rec_seqs = ["".join(aa_code[s] for s in seq) for seq in rec_seqs]
    # batchify sequences
    rec_batches = _esm_batchify(rec_seqs, tokenizer, args)
    with torch.no_grad():
        pad_idx = alphabet.padding_idx
        rec_reps = _run_esm(rec_batches, pad_idx, esm_model)

    # dump to cache
    pdb_id2prot_graph_node_feat = {}
    for idx, pdb in enumerate(pdb_id2prot_dict_sorted):
        # cat one-hot representation and ESM embedding
        # if batch size is 1, unsqueeze to make it 2D
        if len(rec_reps[idx][0].shape) == 1:
            rec_graph_x = torch.cat(
                [rec_reps[idx][0].unsqueeze(-1), rec_reps[idx][1]], dim=1
            )
        else:
            rec_graph_x = torch.cat([rec_reps[idx][0], rec_reps[idx][1]], dim=1)
        pdb_id2prot_graph_node_feat[pdb] = rec_graph_x

    if not args.no_graph_cache:
        with open(esm_cache, "wb+") as f:
            pickle.dump(pdb_id2prot_graph_node_feat, f)

    # overwrite graph.x for each element in batch
    _save_esm_rep(pdb_id2prot_dict, pdb_id2prot_graph_node_feat)

    return pdb_id2prot_dict


def _save_esm_rep(pdb_id2prot_dict, pdb_id2prot_graph_node_feat):
    """
    Assign new ESM representation to graph.x INPLACE
    """
    for pdb_id, prot_graph_node_features in pdb_id2prot_graph_node_feat.items():
        rec_graph = pdb_id2prot_dict[pdb_id]["graph"]["receptor"]
        rec_graph.x = prot_graph_node_features
        # assert len(rec_graph.pos) == len(rec_graph.x)
        if not len(rec_graph.pos) == len(rec_graph.x):
            print(f"Sample's coords do not match sequence! {pdb_id2prot_dict}")
            pdb_id2prot_dict = None
            break

    return pdb_id2prot_dict


def _esm_batchify(seqs, tokenizer, args):
    batch_size = args.batch_size
    # group up sequences
    batches = [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]
    batches = [[("", seq) for seq in batch] for batch in batches]
    # tokenize
    batch_tokens = [tokenizer(batch)[2] for batch in batches]
    return batch_tokens


def _run_esm(batches, padding_idx, esm_model):
    """
    Wrapper around ESM specifics

    Args:
        batches (list): list of tokenized sequences
        padding_idx (int): padding index
        esm_model (nn.Module): ESM model
    Returns:
        list: list of (esm_repr, tokens) without <CLS> and <EOS>
    """
    # run ESM model
    all_reps = []
    for batch in batches:
        reps = esm_model(batch.cuda(), repr_layers=[33])
        reps = reps["representations"][33].cpu().squeeze()[:, 1:]
        all_reps.append(reps)
    # crop to length
    # exclude <cls> <eos>
    cropped_representations = []
    for i, batch in enumerate(batches):
        batch_lens = (batch != padding_idx).sum(1) - 2
        for j, length in enumerate(batch_lens):
            rep_crop = all_reps[i][j, :length]
            token_crop = batch[j, 1 : length + 1, None]
            cropped_representations.append((rep_crop, token_crop))
    return cropped_representations


def tokenize(data, key, tokenizer=None):
    """
    Tokenize every item[key] in data.
    Modifies item[key] and copies original value to item[key_raw].
    If tokenizer is None, creates a new tokenizer.
    If tokenizer is a dict, uses it to tokenize.

    Args:
        data (list): list of dicts (items)
        key (str): key to tokenize in each item[key]

    Returns:
        dict: tokenizer
    """
    if len(data) == 0:  # sometimes no val split, etc.
        return
    # if tokenizer is not provided, create index
    all_values = [item[key] for item in data]
    all_values = set(itertools.chain(*all_values))
    if tokenizer is None:
        tokenizer = {}  # never used
    if type(tokenizer) is dict:
        for item in sorted(all_values):
            if item not in tokenizer:
                tokenizer[item] = len(tokenizer) + 1  # 1-index
        f_token = lambda seq: [tokenizer[x] for x in seq]
    else:
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {three.upper(): one for three, one in protein_letters_3to1.items()}
        )

        def f_token(seq):
            seq = "".join([aa_code[s] for s in seq])
            seq = tokenizer([("", seq)])[2][0]
            return seq

    # tokenize items and modify data in place
    raw_key = f"{key}_raw"
    for item in data:
        raw_item = item[key]
        item[raw_key] = raw_item  # save old
        # tokenize and convert to tensor if applicable
        item[key] = f_token(raw_item)
        if not torch.is_tensor(item[key]):
            item[key] = torch.tensor(item[key])
    return tokenizer


def get_protein_graphs_from_path(data_to_load, args):
    """
     Args:
        data_to_load (list): list of dicts with key 'path'
        args (argparse.ArgumentParser)

    Raises:
        NotImplementedError: [if not using ESM tokenization]
    """
    # cache for post-processed data, optional
    args.hash_saved_data = hashlib.md5(f"{data_to_load}".encode()).hexdigest()

    # load and process files
    # organizes and caches pdb_id2prot_dict
    pdb_id2prot_dict = read_files(data_to_load, args)

    # filters resolution, generates graphs and caches pdb_id2prot_dict
    data_params = {}
    pdb_id2prot_dict = process_data(pdb_id2prot_dict, args)
    if pdb_id2prot_dict is None:
        return None, None
    # generate ESM embeddings and cache cat([one-hot, esm_embeddings])
    # pdb_id2prot_dict, data_params = process_embed(pdb_id2prot_dict, args)
    sample = [pdb_id2prot_dict[key] for key in pdb_id2prot_dict][0]
    return sample, data_params
