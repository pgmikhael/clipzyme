# Generate HHblits MSA for sequences
# https://github.com/soedinglab/hh-suite/wiki#generating-a-multiple-sequence-alignment-using-hhblits

# MSA Transformer:
# - An MSA is generated for each UniRef50 (Suzek et al., 2007) sequence by searching UniClust30 (Mirdita et al., 2017) with HHblits (Steinegger et al., 2019).
# - hhfilter to subsample 256 sequences.
# - `conda install -c conda-forge -c bioconda hhsuite`


import subprocess
import os
import shutil
import multiprocessing
from argparse import ArgumentParser
import pickle


TEMP_FOLDER = os.path.join(os.getcwd(), "temp")
if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)

parser = ArgumentParser()
parser.add_argument(
    "--path_to_sequences",
    type=str,
    default=None,
    required=True,
    help="file of dictionary {header:sequence}",
)
parser.add_argument("--cpus", type=int, default=2, help="Number of cpus.")
parser.add_argument(
    "--target_directory",
    type=str,
    default=None,
    required=True,
    help="directory where msa files are stored.",
)
parser.add_argument(
    "--database_directory",
    type=str,
    default="/data/rsg/mammogram/HHSuite/uniclust30_2018_08/uniclust30_2018_08",
    help="directory where UniClust30 is stored.",
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
        os.path.join(args.target_directory, f"{k}.a3m")
        for k, _, _ in header_sequence_db
    ]

    job_list = [
        (z, p)
        for z, p in zip(header_sequence_db, msa_file_paths)
        if not os.path.exists(p)
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
