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

WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
LIGAND_URL = "https://www.brenda-enzymes.org/ligand.php?brenda_ligand_id={}"
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

if __name__ == "__main__":
    args = parser.parse_args()

    pattern = """(?<=<div class="cell">).*?(?=<\/div>)"""

    brenda_dataset = json.load(open(args.input_file_path, "r"))

    if args.get_molecules:

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
                soup = BeautifulSoup(mol_page.text, "html.parser")

                # get inchikey
                title = re.findall("\(.*?\)", soup.title.text)
                for c in title:
                    if molid in c:
                        inchi = c.split()[-1].strip(")")
                        molinfo["inchi"] = inchi

                # find with pattern and window around divs
                index = mol_page.text.index(
                    '<div class="header"><a name="INCHIKEY"></a>InChIKey</div>'
                )
                cells = re.findall(pattern, mol_page.text[(index + 63) : (index + 500)])
                assert len(cells) == 3
                name = cells[1]
                if molinfo.get("inchi", False):
                    assert inchi == cells[2]

                # second_col = soup.body.find(text="BRENDA Name")
                # name = list(second_col.next_elements)[3].next.next.next
                molinfo["name"] = name

                # get link to structure db
                for link in soup.find_all("a"):
                    if "chebi" in link.get("href", ""):
                        molinfo["chebi_link"] = link.get("href")
                    if "pccompound" in link.get("href", ""):
                        molinfo["pubchem_link"] = link.get("href")

                if molinfo.get("chebi_link", None):
                    molinfo["chebi_data"] = CHEBI_DB[
                        molinfo["chebi_link"].split("chebiId=")[-1]
                    ]
                elif molinfo.get("pubchem_link", None):
                    molinfo["pubchem_data"] = pubchempy.get_compounds(
                        molinfo["pubchem_link"].split("term=")[-1], "inchikey"
                    )[0].to_dict()
            except:
                return molinfo
            return molinfo

        # get molecules from brenda
        # brenda_mols = set()
        # for ecid, ecdict in tqdm(brenda_dataset["data"].items()):
        #     for key in ["cofactor", "activating_compound", "inhibitor"]:
        #         if key in ecdict:
        #             brenda_mols.update([d["value"] for d in ecdict[key]])

        #     for key in ["natural_reaction", "reaction"]:
        #         if key in ecdict:
        #             brenda_mols.update(
        #                 [e for d in ecdict[key] for e in d.get("educts", [])]
        #             )
        #             brenda_mols.update(
        #                 [e for d in ecdict[key] for e in d.get("products", [])]
        #             )

        # brenda_mols = list(brenda_mols)

        # client = Client(WSDL)
        # password = hashlib.sha256(args.brenda_pw.encode("utf-8")).hexdigest()
        # parameters = [(args.brenda_username, password, mol) for mol in brenda_mols]

        # brenda_mol_ids = []
        # for param in tqdm(parameters):
        #     molinfo = {"name": param[-1]}
        #     try:
        #         molid = client.service.getLigandStructureIdByCompoundName(*param)
        #         molinfo["brenda_id"] = molid
        #     except:
        #         molinfo["brenda_id"] = None

        #     brenda_mol_ids.append(molinfo)

        brenda_mol_ids = [{"brenda_id": i} for i in range(1, 260600, 1)]

        brenda_molecules = p_map(get_molecule_info, brenda_mol_ids)

        brenda_molecules_dict = {d["name"]: d for d in brenda_molecules}

        json.dump(brenda_molecules_dict, open(args.output_file_path, "w"))

    if args.get_proteins:
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
