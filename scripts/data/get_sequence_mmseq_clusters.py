"""
A script to split a JSON file containing sequences using mmseqs.
See here on how to install mmseqs - https://github.com/soedinglab/MMseqs2
"""

import os
import shutil
import json
import argparse
import subprocess
import pickle 

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

TEMP_FOLDER = "temp"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script args for splitting sequences using mmseqs",
    )
    parser.add_argument(
        "--file", "-f", type=str, required=True, help="Path to file"
    )
    parser.add_argument(
        "--output-file", "-o", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--min-seq-id",
        "-m",
        type=float,
        default=0.5,
        help="List matches above this sequence identity",
    )
    parser.add_argument(
        "--similarity",
        "-c",
        type=float,
        default=0.8,
        help="List matches above this fraction of aligned (covered) residues",
    )

    return parser.parse_args()


def json_to_fasta(file: str, output_fasta: str) -> None:
    with open(file, "r") as f:
        data = json.load(f)

    fasta_seqs = []
    for entry in data:
        id = entry["gene_names"] if entry["gene_names"] else entry["entry"]
        fasta_seqs.append(SeqRecord(Seq(entry["sequence"]), id=id))

    with open(output_fasta, "w") as output_fasta:
        SeqIO.write(fasta_seqs, output_fasta, "fasta")

def pickle_to_fasta(file: str, output_fasta: str) -> None:
    data = pickle.load(open(file, 'rb'))

    fasta_seqs = []
    for id, sequence in data.items():
        fasta_seqs.append(SeqRecord(Seq(sequence), id=id))

    with open(output_fasta, "w") as output_fasta:
        SeqIO.write(fasta_seqs, output_fasta, "fasta")

def run_mmseqs(
    input_fasta: str, temp_folder: str, min_seq_id: float, similarity: float
) -> None:
    output_file = os.path.join(temp_folder, "mmseqs_output")
    cmd = f"mmseqs easy-cluster {input_fasta} {output_file} {temp_folder} --min-seq-id {min_seq_id} -c {similarity}"
    subprocess.run(cmd, shell=True, check=True)
    return output_file + "_cluster.tsv"

def convert_tsv_to_pickle(tsv_file: str, output_file: str):
    proteinid_to_cluster = {}
    with open(tsv_file, "r") as f:
        for row in f.read().splitlines():
            cluster, item = row.split()
            proteinid_to_cluster[item] = cluster
    
    clusters = list(proteinid_to_cluster.values())
    print(f"Num Cluters: {len(set(clusters))}")
    pickle.dump(proteinid_to_cluster, open(output_file, 'wb'))

def main():
    args = parse_args()

    temp_folder = os.path.join(os.getcwd(), TEMP_FOLDER)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    fasta_file = os.path.join(temp_folder, "temp.fasta")
    pickle_to_fasta(args.file, fasta_file)

    tsv_file = run_mmseqs(fasta_file, temp_folder, args.min_seq_id, args.similarity)

    convert_tsv_to_pickle(tsv_file, args.output_file)

    shutil.rmtree(temp_folder)


if __name__ == "__main__":
    main()
