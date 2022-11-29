from zeep import Client
import hashlib
from bs4 import BeautifulSoup
import re, json
import requests
import argparse
from tqdm import tqdm
from p_tqdm import p_map
import pubchempy
from bioservices import UniProt
import pickle, os

WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
#LIGAND_URL = "https://www.brenda-enzymes.org/ligand.php?brenda_ligand_id={}"
LIGAND_URL = "https://brenda-enzymes.org/ligand.php?brenda_group_id={}"
CHEBI_DB = json.load(open("/Mounts/rbg-storage1/datasets/Metabo/chebi_db.json", "r"))

parser = argparse.ArgumentParser(description="Pulls substrate data from BRENDA")
parser.add_argument(
    "-i",
    "--input_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_2022_2.json",
    help="Path to input file",
)
parser.add_argument(
    "-o",
    "--output_file_path",
    default="/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_substrates.json",
    help="Path to output file",
)
parser.add_argument(
    "--get_molecules",
    action="store_true",
    default=False,
    help="Whether to get molecules from BRENDA",
)
parser.add_argument(
    "--get_smiles",
    action="store_true",
    default=False,
    help="Map molecules to smiles using pubchem or chebi",
)
parser.add_argument(
    "--get_proteins",
    action="store_true",
    default=False,
    help="Whether to get proteins from BRENDA",
)
parser.add_argument(
    "--brenda_username",
    default=None,
    help="brenda username",
)
parser.add_argument(
    "--brenda_pw",
    default=None,
    help="brenda password",
)


def get_brenda_ids(sect):
    pattern = """(?<=<div class="cell">).*?(?=<\/div>)"""
    l = len('<div class="cell">')
    sects = []
    for i in range(3):
        i1 = sect.index('<div class="cell">')
        sect = sect[i1:]
        if i == 2:
            i2 = sect[l:].index("</div>") + 6
        else:
            i2 = sect[l:].index('<div class="cell">')
        parsed_sect = re.findall(pattern, sect[: i2 + l])[0]
        sects.append(parsed_sect)
        sect = sect[i2 + l :]
    return sects


def get_molecule_info(molinfo):
    """Get molecule id from BRENDA

    Args:
        mol (str): molecule name
        args (argparse): argparse object

    Returns:
        str: BRENDA molecule id
    """
    if molinfo["brenda_id"] is None:
        return molinfo
    try:
        molid = molinfo["brenda_id"]

        url = LIGAND_URL.format(molid)
        mol_page = requests.get(url)

        # get inchikey
        title = re.findall("""(?<=<title>).*?(?=<\/title>)""", mol_page.text)[0]
        title = re.findall("\(.*?\)", title)
        for c in title:
            if molid in c:
                inchi = c.split()[-1].strip(")")
                molinfo["inchi"] = inchi

        # find name with pattern and window around divs
        try:
            index = mol_page.text.index(
                '<div class="header"><a name="INCHIKEY"></a>InChIKey</div>'
            )
            cells = get_brenda_ids(mol_page.text[(index + 63) : (index + 1000)])
            name = cells[1]
            if molinfo.get("inchi", False):
                assert inchi == cells[2]
            
            if not molinfo.get("name", False):
                molinfo["name"] = name
        except:
            pass

        # find synonyms
        try:
            synonym_index = mol_page.text.index("Synonyms:")
            synonym_section = mol_page.text[
                (synonym_index + 15) : (synonym_index + 10000)
            ]
            synonyms = re.findall("""(?<=<div>).*?(?=<\/div>)""", synonym_section)[
                0
            ].split(", ")
            synonyms = [s.strip(" ") for s in synonyms]
            molinfo["synonyms"] = synonyms
        except:
            pass

        try:
            chebi_index = mol_page.text.index(
                "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:"
            )
            chebi_link = mol_page.text[chebi_index : chebi_index + 100]
            molinfo["chebi_link"] = chebi_link.split(" ")[0].strip('"')
        except:
            pass

        try:
            pubchem_index = mol_page.text.index(
                "http://www.ncbi.nlm.nih.gov/sites/entrez"
            )
            pubchem_link = mol_page.text[pubchem_index : pubchem_index + 500]
            molinfo["pubchem_link"] = pubchem_link.split(" ")[0].strip("'")
        except:
            pass

    except Exception as e:
        molinfo["error"] = e
        print(e)
        return molinfo

    return molinfo


def get_smiles(molinfo):
    """Get smiles from BRENDA

    Args:
        molinfo (dict): molecule
    """
    if molinfo.get("chebi_link", None):
        molinfo["chebi_data"] = CHEBI_DB[molinfo["chebi_link"].split("chebiId=")[-1]]
    elif molinfo.get("pubchem_link", None):
        molinfo["pubchem_data"] = pubchempy.get_compounds(
            molinfo["pubchem_link"].split("term=")[-1], "inchikey"
        )[0].to_dict()
    return molinfo


if __name__ == "__main__":
    args = parser.parse_args()

    if args.get_molecules:

        brenda_group_ids = pickle.load(open("/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_group_ids.p", "rb"))
        brenda_group_ids = [m for m in brenda_group_ids if m['brenda_id'] is not None]

        if os.path.exists(args.output_file_path):
            last_brenda_molecules = pickle.load(open(args.output_file_path, "rb"))
            last_brenda_molecules_dict = {}
            for j in last_brenda_molecules:
                if 'name' in j:
                    last_brenda_molecules_dict[j['name']] = j 
                for synonym in j.get("synonyms", []):
                    last_brenda_molecules_dict[synonym] = j 

        brenda_mol_ids = []
        brenda_retrieved_molecules = []
        for g in tqdm(brenda_group_ids):
            if (not last_brenda_molecules_dict.get(g['name'], False)) or last_brenda_molecules_dict[g['name']].get("errors", False):
                brenda_mol_ids.append(g)
            elif ('pubchem_link' not in last_brenda_molecules_dict[g['name']] ) and ('chebi_link' not in last_brenda_molecules_dict[g['name']]):
                brenda_mol_ids.append(g)
            else:
                brenda_retrieved_molecules.append(last_brenda_molecules_dict[g['name']])

        brenda_molecules = p_map(get_molecule_info, brenda_mol_ids, num_cpus=7)

        brenda_molecules = brenda_molecules + brenda_retrieved_molecules

        pickle.dump(brenda_molecules, open(args.output_file_path, "wb"))

    if args.get_smiles:
        brenda_molecules = pickle.load(open(args.input_file_path, "rb"))
        brenda_molecules = p_map(get_smiles, brenda_molecules)
        pickle.dump(brenda_molecules, open(args.output_file_path, "wb"))

    if args.get_proteins:
        brenda_dataset = json.load(open(args.input_file_path, "r"))

        u = UniProt(verbose=False)

        uniprots = []
        for _, ec_dict in brenda_dataset["data"].items():
            if "proteins" in ec_dict:
                for k, v in ec_dict["proteins"].items():
                    uniprots.extend(v[0]["accessions"])

        uniprots = list(set(uniprots))

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
            fasta = u.get_fasta(uniprot)
            if fasta is None:
                return {"uniprot": uniprot, "sequence": None}
            seq = parse_fasta(fasta)
            return {"uniprot": uniprot, "sequence": seq}

        protein_info = p_map(get_protein_info, uniprots)
        protein_info = {d["uniprot"]: d for d in protein_info}

        json.dump(protein_info, open(args.output_file_path, "w"))
