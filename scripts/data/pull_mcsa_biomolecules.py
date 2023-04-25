import json
import argparse
from tqdm import tqdm
from p_tqdm import p_map
from bioservices import UniProt
from collections import defaultdict
import warnings

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
    "--mcsa_pdb_to_uniprots",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/pdb2uniprotlite.json",
    help="Map from PDB id to Uniprot; obtained through https://www.uniprot.org/id-mapping",
)
parser.add_argument(
    "-o",
    "--output_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/MCSA/mcsa_biomolecules.json",
    help="Path to output file",
)

RESIDE_TABLE = {
    "": None,
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Sec": "U",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Pyl": "O",
    "X": None,
    "Asx": ["N", "D"],
    "Glx": ["E", "Q"],
    "Xle": ["L", "I"],
}


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


def check_fasta_with_residues(sequence, residues):
    """Check residues for uniprot do appear at correct indices of the sequence

    Args:
        sequence (str): protein sequence
        residues (list): list of tuples of (residue index, residue 3-letter code)

    """
    # one residue and it's unknown
    if (len(residues) == 1) and (residues[0][0] in ["", None]):
        return False

    for res_id, res_name in residues:
        if isinstance(res_id, int) and (res_name != ""):
            if (res_id > len(sequence)) or (
                sequence[res_id - 1] != RESIDE_TABLE[res_name]
            ):
                return False
    return True


def get_protein_fasta(uniprot_tuple):
    """Get protein info from uniprot

    Args:
        uniprot (str): uniprot
    """
    uniprot = uniprot_tuple[0]
    residues = uniprot_tuple[1]

    if uniprot == "":
        return

    is_assembly = "-" in uniprot
    parent_uniprot = uniprot.split("-")[0] if is_assembly else uniprot

    fasta = u.get_fasta(uniprot)

    if (fasta == 404) and is_assembly:  # Not Found
        fasta = u.get_fasta(parent_uniprot)
        if fasta == 404:
            return
    elif fasta == 404:
        return

    sequence = parse_fasta(fasta)
    if not check_fasta_with_residues(sequence, residues):
        return

    return {
        "uniprot": uniprot,
        "sequence": sequence,
        "is_assembly": is_assembly,
        "parent_uniprot": parent_uniprot,
    }


if __name__ == "__main__":
    args = parser.parse_args()
    mcsa_dataset = {}

    mcsa_entries = json.load(open(args.mcsa_entries_path, "r"))
    mcsa_homologs = json.load(open(args.mcsa_homologs_path, "r"))

    # init uniprot service
    u = UniProt(verbose=False)

    # pdb to uniprot mapping
    pdb2uniprot = json.load(open(args.mcsa_pdb_to_uniprots, "r"))

    # get protein ids and assemblies from pdb
    uniprots = set()
    pdbs = set()
    to_residues = defaultdict(set)
    uniprot2assemblies = defaultdict(set)
    for entry in mcsa_entries:
        if "," in entry["reference_uniprot_id"]:
            for uni_name in entry["reference_uniprot_id"].split(","):
                uniprots.add(uni_name.strip(" "))
        else:
            uniprots.add(entry["reference_uniprot_id"])
        for residue in entry["residues"]:
            for seq in residue["residue_sequences"]:
                uniprots.add(seq["uniprot_id"])
                to_residues[seq["uniprot_id"]].add((seq["resid"], seq["code"]))

            for chain in residue["residue_chains"]:
                pdbid = chain["pdb_id"]
                assembly = chain.get("assembly", None)
                for uniprot in pdb2uniprot.get(pdbid, []):
                    uniprot2assemblies[uniprot["uniprot"]].add(assembly)
                    if assembly not in ["", None]:
                        assembly_name = f"{uniprot['uniprot']}-{assembly}"
                        to_residues[assembly_name].add((chain["resid"], chain["code"]))

    for homolog in mcsa_homologs:
        for seq in homolog["residue_sequences"]:
            uniprots.add(seq["uniprot_id"])
            to_residues[seq["uniprot_id"]].add((seq["resid"], seq["code"]))

        for chain in homolog["residue_chains"]:
            pdbid = chain["pdb_id"]
            assembly = chain.get("assembly", None)
            for uniprot in pdb2uniprot.get(pdbid, []):
                uniprot2assemblies[uniprot["uniprot"]].add(assembly)
                if assembly not in ["", None]:
                    assembly_name = f"{uniprot['uniprot']}-{assembly}"
                    to_residues[assembly_name].add((chain["resid"], chain["code"]))

    # add uniprots from residue_sequences not found among residue_chains pdbs
    for uni in uniprots:
        if uni not in uniprot2assemblies:
            uniprot2assemblies[uni].add(None)  # proteins to fetch

    protein_to_fetch = []
    for uni, assemblies in uniprot2assemblies.items():
        if len(set(to_residues[uni])) > 0:
            protein_to_fetch.append((uni, list(set(to_residues[uni]))))
        for asm in assemblies:
            if asm in ["", None]:
                continue
            protein_to_fetch.append(
                (f"{uni}-{asm}", list(set(to_residues[f"{uni}-{asm}"])))
            )

    # get associated fastas
    protein_info = p_map(get_protein_fasta, protein_to_fetch)

    protein_dict = {}
    for p in protein_info:
        if p is None:
            continue
        if p["parent_uniprot"] not in protein_dict:
            protein_dict[p["parent_uniprot"]] = {}
        protein_dict[p["parent_uniprot"]][p["uniprot"]] = p

    # get molecules from m-csa
    mcsa_mols = set()
    for entry in mcsa_entries:
        for mol in entry["reaction"]["compounds"]:
            mcsa_mols.add(mol["chebi_id"])

    mcsa_molecules_dict = {}
    for mol in tqdm(mcsa_mols, desc="Getting Chebi data"):
        mcsa_molecules_dict[mol] = CHEBI_DB.get(f"CHEBI:{mol}", None)

    mcsa_dataset["molecules"] = mcsa_molecules_dict
    mcsa_dataset["proteins"] = protein_dict

    json.dump(mcsa_dataset, open(args.output_file_path, "w"))
