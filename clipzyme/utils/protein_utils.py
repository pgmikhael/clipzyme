from typing import Union
import os, hashlib
from collections import defaultdict
from tqdm import tqdm
from rich import print
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
import torch
from torch_geometric.data import Data as pygData
from torch_geometric.data import HeteroData
from torch_cluster import knn_graph
from esm import FastaBatchedDataset, pretrained

protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})


# Read PDB and CIF files
def read_structure_file(protein_parser, raw_path, sample_id):
    structure = protein_parser.get_structure(sample_id, raw_path)
    all_res, all_pos, all_atom = [], [], []
    for chains in structure:
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


# Resolution
def filter_resolution(all_res, all_atom, all_pos, protein_resolution):
    protein_resolution = protein_resolution.lower()
    if protein_resolution == "atom":
        atoms_to_keep = None
    elif protein_resolution == "backbone":
        atoms_to_keep = {"N", "CA", "O", "CB"}
    elif protein_resolution == "residue":
        atoms_to_keep = {"CA"}
    else:
        raise Exception(
            f"Invalid resolution {protein_resolution}, expected: 'atom', 'backbone' or 'reside'"
        )

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


# - Download AF2 from uniprots
def download_af_structures():
    pass


# KNN Neighbors / Radius
def compute_graph_edges(data, **kwargs):
    edge_index = knn_graph(data["receptor"].pos, kwargs["knn_size"])
    # data["protein", "contact", "protein"].edge_index = edge_index
    data["receptor", "contact", "receptor"].edge_index = edge_index

    return data


def build_graph(atom_names, seq, pos, sample_id):
    data = HeteroData()
    data["sample_id"] = sample_id
    # retrieve position and compute kNN
    data["receptor"].pos = pos.float()
    data["receptor"].seq = seq  # _seq is residue id
    data["receptor"].atom_names = atom_names  # _atom is atom name

    return data


def get_cache_id(model, labels, sequences):
    input_str = f"{model}-{labels}-{sequences}"
    return hashlib.md5(input_str.encode()).hexdigest()


def precompute_node_embeddings(metadata_json, cache_dir, model_location=None):
    # TODO: Add multi-gpu support
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    seqs = [d["structure_sequence"] for d in metadata_json]
    if "sample_id" in metadata_json[0]:
        names = [d["sample_id"] for d in metadata_json]
    elif "pdb_id" in metadata_json[0]:
        names = [d["pdb_id"] for d in metadata_json]
    elif "uniprot_id" in metadata_json[0]:
        names = [d["uniprot_id"] for d in metadata_json]

    # already computed all ESM embeddings
    all_embeddings_cache_id = get_cache_id(model, names, seqs)
    all_embeddings_cache_id_file = os.path.join(
        cache_dir, f"{all_embeddings_cache_id}.pt"
    )
    if os.path.exists(all_embeddings_cache_id_file):
        return torch.load(all_embeddings_cache_id_file)

    # check which embeddings are missing (in case of crashes)
    embeddings = defaultdict(dict)
    missing_seqs = []
    missing_names = []
    for i, seq in tqdm(
        enumerate(seqs), desc="Checking for cached embeddings...", total=len(seqs)
    ):
        # This was dumb, can just use the name (uniprot_id)
        # seq_cache_id = get_cache_id(model, [names[i]], [seq])
        seq_cache_file = os.path.join(cache_dir, f"{names[i]}.pt")

        if os.path.exists(seq_cache_file):
            # with open(seq_cache_file, "rb") as f:
            #     seq_embedding = torch.load(f)
            # embeddings[names[i]]["embedding"] = seq_embedding
            embeddings[names[i]] = seq_cache_file
        else:
            missing_seqs.append(seq)
            missing_names.append(names[i])

    dataset = FastaBatchedDataset(missing_names, missing_seqs)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]

    with torch.no_grad():
        for batch_idx, (batch_labels, batch_strs, batch_toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({batch_toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                batch_toks = batch_toks.to(device="cuda", non_blocking=True)

            out = model(batch_toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for j, sample_id in enumerate(batch_labels):
                truncate_len = min(truncation_seq_length, len(batch_strs[j]))
                seq_embedding = representations[33][j, 1 : truncate_len + 1].clone()
                # embeddings[sample_id]["embedding"] = seq_embedding

                # Cache the embedding for this sequence
                # This was dumb, can just use the name (uniprot_id)
                # seq_cache_id = get_cache_id(model, [sample_id], [batch_strs[j]])
                seq_cache_file = os.path.join(cache_dir, f"{sample_id}.pt")
                embeddings[sample_id] = seq_cache_file

                with open(seq_cache_file, "wb") as f:
                    torch.save(seq_embedding, f)

    # Cache the embeddings paths for all sequences
    with open(all_embeddings_cache_id_file, "wb") as f:
        torch.save(embeddings, f)

    return embeddings


# Node embeddings (needs to be robust, overwritable)
def compute_node_embedding(data, **kwargs):
    # TODO: Add multi-gpu support
    # TODO: Ready to go atomic embeddings
    # Ready to go ESM embeddings
    model = kwargs["model"]
    model_location = kwargs["model_location"]
    alphabet = kwargs["alphabet"]
    batch_converter = kwargs["batch_converter"]

    model.eval()
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 1022

    seq = data.structure_sequence
    batch_labels, batch_strs, batch_tokens = batch_converter([(0, seq)])
    batch_toks = batch_tokens

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)

    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]

    with torch.no_grad():
        out = model(batch_toks, repr_layers=repr_layers, return_contacts=False)
        representations = {layer: t for layer, t in out["representations"].items()}
        truncate_len = min(truncation_seq_length, len(seq))
        seq_embedding = representations[33][0, 1 : truncate_len + 1].clone()

    return seq_embedding


def get_sequences(
    protein_parser, sample_ids, protein_filepaths, protein_sequences=None
):
    new_sequences = []
    for idx, path in tqdm(
        enumerate(protein_filepaths),
        desc="Reading protein sequences",
        total=len(protein_filepaths),
    ):
        structure_seq = get_sequences_from_structure(protein_parser, path)
        if (
            protein_sequences is not None
            and protein_sequences[idx] is not None
            and structure_seq != protein_sequences[idx]
        ):
            warnings.warn(
                f"Found mismatch in structure seq and metadata seq for sample {sample_ids[idx]['sample_id']}"
            )
        new_sequences.append(structure_seq)
    return new_sequences


def get_sequences_from_structure(protein_parser, file_path):
    structure = protein_parser.get_structure("does_not_matter", file_path)
    structure = structure[0]
    sequence = None
    protein_letters_3to1.update({k.upper(): v for k, v in protein_letters_3to1.items()})
    for i, chain in enumerate(structure):
        seq = ""
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None
            ):  # only append residue if it is an amino acid
                try:
                    seq += protein_letters_3to1[residue.get_resname().upper()]
                except Exception as e:
                    seq += "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        if sequence is None:
            sequence = seq
        else:
            sequence += ":" + seq

    return sequence


def create_protein_graph(cif_path: str, esm_path: str) -> Union[pygData, None]:
    """
    Create pyg protein graph from CIF file

    Parameters
    ----------
    cif_path : str
        Path to CIF file
    esm_path : str
        Path to ESM model (esm2_t33_650M_UR50D.pt)

    Returns
    -------
    data
        pygData object with protein graph
    """
    assert esm_path.endswith(
        "esm2_t33_650M_UR50D.pt"
    ), "ESM model filename must end with esm2_t33_650M_UR50D.pt"
    esm_model, alphabet = pretrained.load_model_and_alphabet(esm_path)
    batch_converter = alphabet.get_batch_converter()

    try:
        raw_path = cif_path
        sample_id = "proteinX"
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

        # get sequence
        AA_seq = ""
        for char in seq:
            AA_seq += protein_letters_3to1[char]

        data.structure_sequence = AA_seq

        node_embeddings_args = {
            "model": esm_model,
            "model_location": "",
            "alphabet": alphabet,
            "batch_converter": batch_converter,
        }

        # compute embeddings
        data["receptor"].x = compute_node_embedding(data, **node_embeddings_args)

        if len(data["receptor"].seq) != data["receptor"].x.shape[0]:
            return None

        if hasattr(data, "x") and not hasattr(data["receptor"], "x"):
            data["receptor"].x = data.x

        if not hasattr(data, "structure_sequence"):
            data.structure_sequence = "".join(
                [protein_letters_3to1[char] for char in data["receptor"].seq]
            )

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

    except Exception as e:
        print(f"Could not create protein graph because of the exception: {e}")
        return None
