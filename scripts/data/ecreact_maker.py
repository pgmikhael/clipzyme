import json
import argparse
from tqdm import tqdm
from p_tqdm import p_map
import requests
import pandas as pd


UNIPROT_QUERY_URL = "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+ec:{}&format=json&fields=id,sequence,cc_alternative_products&size=500"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"

parser = argparse.ArgumentParser(
    description="Make EC React Dataset (https://github.com/rxn4chemistry/biocatalysis-model)"
)
parser.add_argument(
    "--react_csv_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact-1.0.csv",
    help="Path to EC React entries file",
)
parser.add_argument(
    "-o",
    "--output_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/ECReact/ecreact_dataset.json",
    help="Path to output file",
)


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

    if fasta.status_code == 200: # Success
        sequence = parse_fasta(fasta.text)
        return sequence 

    return 


if __name__ == "__main__":
    args = parser.parse_args()

    react_dataset = pd.read_csv(args.react_csv_path)
    # for i, row in tqdm(react_dataset.iterrows(), total=len(react_dataset)):
    def parse_react_row(row):
        dataset = []
        iso_dataset = []
        rxn_smiles = row["rxn_smiles"]
        ec = row["ec"]
        db_source = row["source"]

        full_reactants, products = rxn_smiles.split(">>")
        products = products.split(".")
        reactants_str, ec_str = full_reactants.split("|")
        reactants = reactants_str.split(".")
        # get the proteins sequences; while loop
        num_uniprot_results = 500
        while num_uniprot_results >= 500:
            uniprot_results = requests.get(UNIPROT_QUERY_URL.format(ec))
            if uniprot_results.status_code == 200:
                uniprot_results = uniprot_results.json()["results"]
                num_uniprot_results = len(uniprot_results)
                for uniprot_result in uniprot_results:
                    uniprot_id = uniprot_result["primaryAccession"]
                    sequence = uniprot_result["sequence"]["value"]
                    dataset.append(
                        {
                            "reactants": reactants,
                            "products": products,
                            "sequence": sequence,
                            "ec": ec,
                            "uniprot_id": uniprot_id,
                            "db_source": db_source
                        }
                    )
                    for comment in uniprot_result["comments"]:
                        for isoform in comment.get("isoforms", []):
                            isoform_uniprots = isoform["isoformIds"]
                            for isoform_u in isoform_uniprots:
                                iso_dataset.append(
                                    {
                                        "reactants": reactants,
                                        "products": products,
                                        "ec": ec,
                                        "uniprot_id": isoform_u,
                                        "db_source": db_source
                                    }
                                )

            else:
                dataset.append(
                        {
                            "reactants": reactants,
                            "products": products,
                            "ec": ec,
                            "uniprot_id": uniprot_id,
                            "db_source": db_source
                        }
                    )
                num_uniprot_results = 0
            return (dataset, iso_dataset)

    # match ec to uniprots, sequences, and isoforms
    react_dataset_rows = react_dataset.to_dict('records')
    reference_dataset = p_map(parse_react_row, react_dataset_rows)
    dataset, iso_dataset = [], []
    for d in reference_dataset:
        dataset.extend(d[0])
        iso_dataset.extend(d[1])
    
    # pass through isoforms
    isoform_uniprots = [d['uniprot_id'] for d in iso_dataset]
    isform_sequences = p_map(get_protein_fasta, isoform_uniprots)
    for i, sample in enumerate(iso_dataset):
        sample["sequence"] = isform_sequences[i]
        dataset.append(sample)

    json.dump(dataset, open(args.output_file_path, "w"))
