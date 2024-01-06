# Generate HHblits MSA for sequences
# https://github.com/soedinglab/hh-suite/wiki#generating-a-multiple-sequence-alignment-using-hhblits

# MSA Transformer:
# - An MSA is generated for each UniRef50 (Suzek et al., 2007) sequence by searching UniClust30 (Mirdita et al., 2017) with HHblits (Steinegger et al., 2019).
# - hhfilter to subsample 256 sequences.
# - `conda install -c conda-forge -c bioconda hhsuite`

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(".")))
import subprocess
import shutil
import multiprocessing
from argparse import ArgumentParser
import pickle
from nox.utils.hhblits_msa import read_msa, sample_msa
import torch
from tqdm import tqdm

TEMP_FOLDER = os.path.join(os.getcwd(), "temp")
if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)

parser = ArgumentParser()
parser.add_argument(
    "--generate_msa",
    action="store_true",
    default=False,
    help="whether to generate msas",
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
    default="/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/hhblits_msas",
    required=True,
    help="directory where msa files are stored.",
)
parser.add_argument(
    "--database_directory",
    type=str,
    default="/data/rsg/mammogram/HHSuite/uniclust30_2018_08/uniclust30_2018_08",
    help="directory where UniClust30 is stored.",
)
parser.add_argument(
    "--embedding_target_directory",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/hhblits_embeds",
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


def launch_job(header_sequence_db, msa_filename):
    header, sequence, db = header_sequence_db
    # make sequence file
    temp_file = os.path.join(TEMP_FOLDER, f"{header}.seq")
    with open(temp_file, "w") as f:
        f.write(f">{header}\n")
        f.write(sequence)

    shell_cmd = f"hhblits -cpu 4 -i {temp_file} -d {db} -oa3m {msa_filename} -n 3"

    print("Launched command: {}".format(shell_cmd))
    if not os.path.exists(msa_filename):
        subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(msa_filename):
        print("FAILED TO PROCESS {}".format(msa_filename))

    # remove temp file
    os.remove(temp_file)
    os.remove(temp_file.replace(".seq", ".hhr"))

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

    header_to_sequence = pickle.load(open(args.path_to_sequences, "rb"))
    header_sequence_db = [
        (k, v, args.database_directory) for k, v in header_to_sequence.items()
    ]
    msa_file_paths = [
        os.path.join(args.msa_target_directory, f"{k}.a3m")
        for k, _, _ in header_sequence_db
    ]
    if args.generate_msa:
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

    shutil.rmtree(TEMP_FOLDER)

    if args.generate_msa_transformer_embeddings:
        if not os.path.exists(args.embedding_target_directory):
            os.mkdir(args.embedding_target_directory)

        embedding_file_paths = [
            os.path.join(args.embedding_target_directory, f"{k}.pt")
            for k, _, _ in header_sequence_db
        ]

        torch.hub.set_dir("/Mounts/rbg-storage1/snapshots/metabolomics")
        msa_model, msa_alphabet = torch.hub.load(
            "facebookresearch/esm:main", "esm_msa1b_t12_100M_UR50S"
        )
        msa_model.eval()
        msa_model.requires_grad_(False)
        msa_model = msa_model.to(args.device)
        msa_transformer_batch_converter = msa_alphabet.get_batch_converter()

        for msa_file, pt_file in tqdm(
            zip(msa_file_paths, embedding_file_paths),
            ncols=100,
            total=len(embedding_file_paths),
        ):
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
