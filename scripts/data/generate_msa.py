# Generate HHblits MSA for sequences
# https://github.com/soedinglab/hh-suite/wiki#generating-a-multiple-sequence-alignment-using-hhblits

# MSA Transformer:
# - An MSA is generated for each UniRef50 (Suzek et al., 2007) sequence by searching UniClust30 (Mirdita et al., 2017) with HHblits (Steinegger et al., 2019).
# - hhfilter to subsample 256 sequences.
# - `conda install -c conda-forge -c bioconda hhsuite`

import sys
import os

import subprocess
import shutil
import multiprocessing
from argparse import ArgumentParser
import pickle
import torch
from tqdm import tqdm
from pathlib import Path

# From: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
from typing import List, Tuple, Dict, Union
import string
from Bio import SeqIO
import numpy as np
from scipy.spatial.distance import cdist


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]


def sample_msa(
    msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max"
) -> List[Tuple[str, str]]:
    """
    Sample sequences from MSA either randomly or those with highest similarity.
    First (query) sequence is always kept.

    Args:
        msa (List[Tuple[str, str]]): msa as (header, sequence) pairs
        num_seqs (int): number of sequences to sample
        mode (str, optional): sampling strategy. Defaults to "max".

    Raises:
        NotImplementedError

    Returns:
        List[Tuple[str, str]]: list as (header, sequence) pairs to keep
    """
    assert mode in ("max", "random")
    if len(msa) <= num_seqs:
        return msa

    if mode == "random":
        indices = np.random.choice(list(range(1, len(msa), 1)), num_seqs - 1)
        return [msa[0]] + [msa[idx] for idx in indices]
    elif mode == "max":
        # integers for seq
        array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

        optfunc = np.argmax
        all_indices = np.arange(len(msa))  # number of sequences in msa
        indices = [0]
        pairwise_distances = np.zeros((0, len(msa)))
        for _ in range(num_seqs - 1):
            dist = cdist(array[indices[-1:]], array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return [msa[idx] for idx in indices]
    else:
        raise NotImplementedError


TEMP_FOLDER = os.path.join(os.getcwd(), "temp")
if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)

parser = ArgumentParser()
parser.add_argument(
    "--generate_hhblits_msa",
    action="store_true",
    default=False,
    help="whether to generate msas",
)
parser.add_argument(
    "--generate_mmseqs_msa",
    action="store_true",
    default=False,
    help="whether to generate msas with colabfold mmseqs",
)
parser.add_argument(
    "--generate_msa_transformer_embeddings",
    action="store_true",
    default=False,
    help="whether to generate msas",
)
parser.add_argument(
    "--path_to_sequences",
    type=str,
    default=None,
    required=True,
    help="file of dictionary {header:sequence}",
)
parser.add_argument("--cpus", type=int, default=2, help="Number of cpus.")
parser.add_argument(
    "--msa_target_directory",
    type=str,
    default="/home/datasets/EnzymeMap/hhblits_msas",
    help="directory where msa files are stored.",
)
parser.add_argument(
    "--database_directory",
    type=str,
    default="/home/HHSuite/uniclust30_2018_08/uniclust30_2018_08",
    help="directory where UniClust30 is stored.",
)
parser.add_argument(
    "--embedding_target_directory",
    type=str,
    default="/home/datasets/EnzymeMap/hhblits_embeds",
    help="directory where msa transformer embeddings are stored.",
)
parser.add_argument(
    "--num_msa_sequences",
    type=int,
    default=128,
    help="number of sequences to sample for MSA",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="cuda device",
)


def write_fasta(header: Union[str, list], sequence: [str, list], filepath: str):
    if isinstance(header, list):
        assert len(header) == len(sequence)
    else:
        assert isinstance(header, str) and isinstance(sequence, str)
        header = [header]
        sequence = [sequence]

    with open(filepath, "w") as f:
        for head, seq in zip(header, sequence):
            f.write(f">{head}\n")
            f.write(f"{seq}\n")


def launch_job(header_sequence_db, msa_filename):
    header, sequence, db = header_sequence_db
    # make sequence file
    temp_file = os.path.join(TEMP_FOLDER, f"{header}.seq")
    write_fasta(header, sequence, temp_file)

    shell_cmd = f"hhblits -cpu 4 -i {temp_file} -d {db} -oa3m {msa_filename} -n 3"

    print("Launched command: {}".format(shell_cmd))
    if not os.path.exists(msa_filename):
        subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(msa_filename):
        print("FAILED TO PROCESS {}".format(msa_filename))

    # remove temp file
    try:
        os.remove(temp_file)
        os.remove(temp_file.replace(".seq", ".hhr"))
    except:
        pass

    return msa_filename


def worker(job_queue, done_queue):
    """
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    """
    while not job_queue.empty():
        header_and_sequence, path = job_queue.get()
        done_queue.put(launch_job(header_and_sequence, path))


if __name__ == "__main__":
    args = parser.parse_args()

    header_to_sequence_or_path = pickle.load(open(args.path_to_sequences, "rb"))

    if args.generate_hhblits_msa:
        header_sequence_db = [
            (k, v, args.database_directory)
            for k, v in header_to_sequence_or_path.items()
        ]
        msa_file_paths = [
            os.path.join(args.msa_target_directory, f"{k}.a3m")
            for k, _, _ in header_sequence_db
        ]

        job_list = [
            (z, p)
            for z, p in zip(header_sequence_db, msa_file_paths)
            # if not os.path.exists(p)
        ]

        job_queue = multiprocessing.Queue()
        done_queue = multiprocessing.Queue()

        for job in job_list:
            job_queue.put(job)

        for cpu in range(args.cpus):
            multiprocessing.Process(target=worker, args=(job_queue, done_queue)).start()

        for i in range(len(job_list)):
            filepath = done_queue.get()
            if not os.path.exists(filepath):
                print(f"Could not generate MSA {os.path.basename(filepath)}!")
            print(f"({i+1}/{len(job_list)}) \t SUCCESS: Generated MSA {filepath}")

    if args.generate_mmseqs_msa:
        headers = list(header_to_sequence_or_path.values())
        sequences = [header_to_sequence_or_path[u] for u in headers]
        tmpfile = Path(args.path_to_sequences).with_suffix(".fasta").name
        tmpfile = os.path.join(TEMP_FOLDER, tmpfile)
        write_fasta(headers, sequences, tmpfile)

        shell_cmd = f"""python colab_search.py \
        {tmpfile} \
        {args.database_directory} \
        {args.msa_target_directory}\
        -s 8 \
        --use-env 0"""

        print("Launched command: {}".format(shell_cmd))

    if args.generate_msa_transformer_embeddings:
        if not os.path.exists(args.embedding_target_directory):
            os.mkdir(args.embedding_target_directory)

        torch.hub.set_dir("/home/snapshots/metabolomics")
        msa_model, msa_alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S"
        )
        msa_model.eval()
        msa_model.requires_grad_(False)
        msa_model = msa_model.to(args.device)
        msa_transformer_batch_converter = msa_alphabet.get_batch_converter()

        for identifier, msa_file in tqdm(header_to_sequence_or_path.items(), ncols=100):
            pt_file = os.path.join(args.embedding_target_directory, f"{identifier}.pt")
            if os.path.exists(pt_file):
                continue
            msa = read_msa(msa_file)
            if len(msa[0][1]) > 1022:
                continue
            # can change this to pass more/fewer sequences
            msa = sample_msa(msa, num_seqs=args.num_msa_sequences, mode="max")
            _, _, msa_transformer_batch_tokens = msa_transformer_batch_converter([msa])
            msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(args.device)
            with torch.no_grad():
                msa_out = msa_model(
                    msa_transformer_batch_tokens,
                    repr_layers=[12],
                    need_head_weights=True,
                )
            msa_rep = msa_out["representations"][12][0, 0, 1:].cpu()
            torch.save(msa_rep, pt_file)

    shutil.rmtree(TEMP_FOLDER)
