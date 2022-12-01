import json
import argparse
from tqdm import tqdm
from p_tqdm import p_map
from bioservices import UniProt

CHEBI_DB = json.load(open("/Mounts/rbg-storage1/datasets/Metabo/chebi_db.json", "r"))

parser = argparse.ArgumentParser(description="Pulls biomolecule data from M-CSA data")
parser.add_argument(
    "--mcsa_entries_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/entries.json",
    help="Path to MCSA entries file",
)
parser.add_argument(
    "--mcsa_homologs_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/homologues_residues.json",
    help="Path to MCSA homologs file",
)
parser.add_argument(
    "-o",
    "--output_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/mcsa_biomolecules.json",
    help="Path to output file",
)


if __name__ == "__main__":
    args = parser.parse_args()
    mcsa_dataset = {}

    mcsa_entries = json.load(open(args.mcsa_entries_path, "r"))
    mcsa_homologs = json.load(open(args.mcsa_homologs_path, "r"))

    mcsa_mols = set()
    # get molecules from m-csa
    for entry in mcsa_entries:
        for mol in entry["reaction"]["compounds"]:
            mcsa_mols.add(mol["chebi_id"])

    mcsa_molecules_dict = {}
    for mol in tqdm(mcsa_mols, desc='Getting Chebi data'):
        mcsa_molecules_dict[mol] = CHEBI_DB.get(f"CHEBI:{mol}", None)

    # get data
    u = UniProt(verbose=False)

    uniprots = set()
    for homolog in mcsa_homologs:
        for h in homolog["residue_sequences"]:
            uniprots.add(h["uniprot_id"])

    for entry in mcsa_entries:
        uniprots.add(entry["reference_uniprot_id"])
        for p in entry["protein"]["sequences"]:
            uniprots.add(p["uniprot_id"])

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

    def get_protein_info(uniprot):
        """Get protein info from uniprot

        Args:
            uniprot (str): uniprot
        """
        if uniprot == "":
            return {"uniprot": uniprot, "sequence": None}
        fasta = u.get_fasta(uniprot)
        if fasta is None:
            return {"uniprot": uniprot, "sequence": None}
        seq = parse_fasta(fasta)
        return {"uniprot": uniprot, "sequence": seq}

    uniprots = list(uniprots)
    protein_info = p_map(get_protein_info, uniprots)
    protein_info = {d["uniprot"]: d for d in protein_info}

    mcsa_dataset["molecules"] = mcsa_molecules_dict
    mcsa_dataset["proteins"] = protein_info

    json.dump(mcsa_dataset, open(args.output_file_path, "w"))
