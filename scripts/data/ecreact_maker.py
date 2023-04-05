import json
import argparse
from tqdm import tqdm
from p_tqdm import p_map
import requests
import pandas as pd
import re, os
import requests
from requests.adapters import HTTPAdapter, Retry
import re 
import pickle

UNIPROT_QUERY_URL = "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+ec:{}&format=json&fields=id,sequence,cc_alternative_products&size=500"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"
RE_NEXT_LINK = re.compile(r'<(.+)>; rel="next"')


parser = argparse.ArgumentParser(
    description="Make EC React Dataset (https://github.com/rxn4chemistry/biocatalysis-model)"
)
parser.add_argument(
    "--react_dir_or_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact-1.0.csv",
    help="Path to EC React entries file or IBM splits directory",
)
parser.add_argument(
    "-o",
    "--output_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_dataset.json",
    help="Path to output file",
)
parser.add_argument(
    "--from_ibm_splits",
    action="store_true",
    default=False,
    help="if from IBM processed splits",
)
parser.add_argument(
    "--from_random_splits",
    action="store_true",
    default=False,
    help="if from random splits",
)
parser.add_argument(
    "--get_ec_to_uniprots",
    action="store_true",
    default=False,
    help="whether to get uniprot ids",
)
parser.add_argument(
    "--add_isoforms",
    action="store_true",
    default=False,
    help="whether to add isoforms",
)


def get_next_link(headers):
    if "Link" in headers:
        match = RE_NEXT_LINK.match(headers["Link"])
        if match:
            return match.group(1)

def parse_fasta(f):
    """Parse fasta data

    Args:
        f (str): fasta data

    Returns:
        str: protein sequence
    """
    _seq = ""
    for _line in f.split("\n"):
        if _line.startswith(">"):
            continue
        _seq += _line.strip()
    return _seq


def get_protein_fasta(uniprot):
    """Get protein info from uniprot

    Args:
        uniprot (str): uniprot
    """

    fasta = requests.get(UNIPROT_ENTRY_URL.format(uniprot))

    if fasta.status_code == 200:  # Success
        sequence = parse_fasta(fasta.text)
        return sequence

    return


def transform_ec_number(ec_str):
    """
    transform input formatted as [vEC1] [uEC2] [tEC3] [qEC4] into EC1.EC2.EC3.EC4
    """
    ec_digits = re.findall(r"\d+|-", ec_str)
    ec = ".".join(ec_digits)
    return ec


def get_uniprots_from_ec(ec):
    iso_dataset = []
    uni2seq = {}
    
    # get the proteins sequences; while loop
    batch_url=UNIPROT_QUERY_URL.format(ec)
    while batch_url:
        uniprot_results = requests.get(batch_url)
        if uniprot_results.status_code == 200:
            batch_url = get_next_link(uniprot_results.headers)
            uniprot_results = uniprot_results.json()["results"]
            num_uniprot_results = len(uniprot_results)
            for uniprot_result in uniprot_results:
                uniprot_id = uniprot_result["primaryAccession"]
                sequence = uniprot_result["sequence"]["value"]
                uni2seq[uniprot_id] = sequence

                for comment in uniprot_result.get("comments", []):
                    for isoform in comment.get("isoforms", []):
                        isoform_uniprots = isoform["isoformIds"]
                        for isoform_u in isoform_uniprots:
                            iso_dataset.append(isoform_u)
            
            if batch_url is None:
                batch_url = False

        else:
            batch_url=False
    
    iso_dataset = list(set(iso_dataset))
    iso_dataset = [u for u in iso_dataset if u not in uni2seq]

    return (uni2seq, iso_dataset)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.from_ibm_splits:
        """run first: rbt-preprocess.py /Mounts/rbg-storage1/datasets/Enzymes/ECReact/smarts_ecreact-1.0.csv /Mounts/rbg-storage1/datasets/Enzymes/ECReact/ibm_splits  --ec-level 4"""
        formatted_dataset = {
            "train": {"src": [], "tgt": []},
            "valid": {"src": [], "tgt": []},
            "test": {"src": [], "tgt": []},
        }
        vocab = set()

        split_files = os.listdir(args.react_dir_or_path)

        for filename in split_files:
            if filename.endswith("combined.txt"):
                continue

            filepath = os.path.join(args.react_dir_or_path, filename)
            rxn_side, split = filename.split("-")
            split, _ = os.path.splitext(split)
            with open(filepath, "r") as f:
                for line in f:
                    formatted_dataset[split][rxn_side].append(line.rstrip("\n"))

        dataset = []
        for split, data_items in formatted_dataset.items():
            assert len(data_items["src"]) == len(data_items["tgt"])
            for i, (src, tgt) in enumerate(zip(data_items["src"], data_items["tgt"])):
                reactants_str, ec_str = src.split("|")
                if split != "test":
                    vocab.update(reactants_str.split(" "))
                ec = transform_ec_number(ec_str)
                reactants = reactants_str.replace(" ", "").split(".")
                products = tgt.replace(" ", "").split(".")

                dataset.append(
                    {
                        "reactants": reactants,
                        "products": products,
                        "ec": ec,
                        "split": "dev" if split == "valid" else split,
                        "from": filepath,
                        "rxnid": f"{split}_{i}",
                    }
                )

        vocab = sorted(list(vocab))
        vocabpath, _ = os.path.splitext(args.output_file_path)
        with open(f"{vocabpath}_vocab.txt", "w") as f:
            for tok in vocab:
                f.write(tok)
                f.write("\n")
        
        json.dump(dataset, open(args.output_file_path, "w"))

    elif args.from_random_splits:
        react_dataset = pd.read_csv(args.react_dir_or_path)
    
        # transform csv to json
        react_dataset_rows = react_dataset.to_dict("records")
        reactions_dataset = []
        for row in react_dataset_rows:
            rxn_smiles = row["rxn_smiles"]
            ec = row["ec"]
            db_source = row["source"]

            full_reactants, products = rxn_smiles.split(">>")
            products = products.split(".")
            reactants_str, ec_str = full_reactants.split("|")
            reactants = reactants_str.split(".")
            reactions_dataset.append(
                {
                    "reactants": reactants,
                    "products": products,
                    "ec": ec,
                    "db_source": db_source,
                }
            )
        json.dump(
            reactions_dataset,
            open(args.react_dir_or_path.replace(".csv", ".json"), "w"),
        )

    if args.get_ec_to_uniprots:
        react_dataset_rows = json.load(open(args.react_dir_or_path, "r"))
        ecs = list(set(r['ec'] for r in react_dataset_rows))
        
        # match ec to uniprots, sequences, and isoforms
        uniprot_results = p_map(get_uniprots_from_ec, ecs)

        ec2uniprot = {}         # ec to canonical uniprot
        ec2iso_uniprot = {}     # ec to isoform uniprot
        uni2seq = {}            # uniprot to sequence
        iso_uniprots = set()    # set of isoform ids 
        for ec, uresult in zip(ecs, uniprot_results):
            ec2uniprot[ec] = list(uresult[0].keys())
            ec2iso_uniprot[ec] = uresult[1]
            iso_uniprots.update(uresult[1])
            uni2seq.update(uresult[0])

        if args.add_isoforms:
            # pass through isoforms
            isoform_sequences = p_map(get_protein_fasta, iso_uniprots)
            isoform2sequence = {i:s for i,s in zip(iso_uniprots, isoform_sequences)}
            uni2seq.update(isoform2sequence)


    pickle.dump(ec2uniprot, open(f"{args.output_file_path}/ec2uniprot.p", "wb"))
    pickle.dump(uni2seq, open(f"{args.output_file_path}/uniprot2sequence.p", "wb"))
