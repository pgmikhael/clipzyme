import json
import torch
import os
from esm import pretrained
import Bio
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_3to1
import warnings
warnings.filterwarnings("ignore", category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
from tqdm import tqdm
import sys
sys.path.append('/Mounts/rbg-storage1/users/itamarc/nox/nox/utils/')
from protein_utils import get_sequences, compute_node_embedding
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--base_graph_directory",
        default="/Mounts/rbg-storage1/datasets/Metabo/",
        choices=["/Mounts/rbg-storage1/datasets/Metabo/", "/storage/itamarc/metabo/"],
        help="base directory for graph pt files",
    )

parser.add_argument(
        "--base_structures_directory",
        default="/Mounts/rbg-storage1/datasets/Metabo/",
        choices=["/Mounts/rbg-storage1/datasets/Metabo/", "/storage/itamarc/metabo/"],
        help="base directory for graph pt files",
    )

args = parser.parse_args()

print(f"NOTE BASE DIR IS {args.base_graph_directory}")

dd = json.load(open("/Mounts/rbg-storage1/datasets/Enzymes/ecreact_with_negatives_updated_filtered.json", "rb"))

esm_dir="/Mounts/rbg-storage1/snapshots/metabolomics/esm2/checkpoints/esm2_t33_650M_UR50D.pt"
model, alphabet = pretrained.load_model_and_alphabet(esm_dir)
esm_model = model
alphabet = alphabet
batch_converter = alphabet.get_batch_converter()

node_embeddings_args = {"model": esm_model, "model_location": esm_dir, "alphabet": alphabet, "batch_converter": batch_converter}
protein_parser = Bio.PDB.MMCIFParser()
structures_dir = os.path.join(args.base_structures_directory, "AlphaFoldEnzymes")
unique_prots = set()
d2 = []
for s in tqdm(dd):
    if s['uniprot_id'] not in unique_prots:
        unique_prots.add(s['uniprot_id'])
        d2.append(s)
print("working from 7518 onwards")
for sample in tqdm(d2[7518:]):
    sample_id = sample['uniprot_id']
    graph_path = os.path.join(args.base_graph_directory, f"quickprot_caches/processed", f"{sample_id}_graph.pt")
    struct_path = os.path.join(structures_dir, f"AF-{sample['uniprot_id']}-F1-model_v4.cif")
    if os.path.exists(graph_path) and os.path.exists(struct_path):
        data = torch.load(graph_path)
        if len(data["receptor"].seq) != data.x.shape[0]:
            print(data.x.shape)
            sequences = get_sequences(
                protein_parser,
                [sample["uniprot_id"]],
                [struct_path],
            )

            data.structure_sequence = sequences[0]
            data["receptor"].x = compute_node_embedding(
                data, **node_embeddings_args
            )
            print(data["receptor"].x.shape)
            torch.save(data, graph_path)
