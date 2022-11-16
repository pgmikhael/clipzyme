import sys, os
import subprocess
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
import json, pickle
import argparse
from tqdm import tqdm


def json_to_fasta(args):
    # File input
    dataset = json.load(open(args.json_dir, "rb"))

    fasta_file_name = args.save_dir + args.json_dir.split("/")[-1].replace("json", "fasta")
    fasta_file = open(fasta_file_name, "w")
    seq_ids = set()
    for sample in tqdm(dataset):
        if sample["seq_id"] in seq_ids:
            continue

        seq_ids.add(sample["seq_id"])
        seq = sample["Sequence"]
        # Output the header
        id = sample["seq_id"]
        header = f">seq_id_{id}\n"
        fasta_file.write(header)
        fasta_file.write(seq + "\n")

    fasta_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script args for json to fasta",
    )

    parser.add_argument("--json_dir", type=str, required=True, help="Path to json file")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/Mounts/rbg-storage1/datasets/YoungLab/IDR_Project/",
        help="Save directory for fasta file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    json_to_fasta(args)
