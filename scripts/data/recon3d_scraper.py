import os, json
from cobra.io import load_matlab_model
from cobra.core.metabolite import Metabolite
from cobra.core.reaction import Reaction
import pandas as pd
from collections import defaultdict
from bioservices import UniProt
from bioservices.apps.fasta import FASTA
from bigg_scraper import link_metabolite_to_db
from p_tqdm import p_map
import requests
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import Mol
from typing import Union
import warnings

warnings.filterwarnings("ignore", "*WARNING: not removing hydrogen*")

# https://www.nature.com/articles/nbt.4072
RECON3_METABOLITES = pd.read_excel(
    "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/41587_2018_BFnbt4072_MOESM11_ESM.xlsx",
    sheet_name="Supplementary Data File 14",
)
RECON3_METABOLITES.fillna("", inplace=True)

# RECON3_PROTEINS = pd.read_excel(
#   "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/41587_2018_BFnbt4072_MOESM11_ESM.xlsx",
#    sheet_name="Supplementary Data File 11",
# )
# RECON3_PROTEINS.fillna("", inplace=True)

# https://github.com/SBRG/ssbio
RECON3_PROTEINS_GEMPRO = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/Recon3D/Recon3D_GP/data/DF_GEMPRO.csv"
)
RECON3_PROTEINS_GEMPRO.fillna("", inplace=True)

RECON3_DATASET_PATH = (
    "/Mounts/rbg-storage1/datasets/Metabo/datasets/recon3d_dataset.json"
)

uniprot_service = UniProt(verbose=False)


def assert_rdkit_fp_safe(mol: Union[Mol, str]) -> None:
    try:
        if isinstance(mol, str):
            assert len("smiles") > 0, "assertion error: empty smiles"
            mol = Chem.MolFromSmiles(mol)
            assert mol is not None, "assertion error: smiles is none"
        if isinstance(mol, Mol):
            fp = Chem.RDKFingerprint(mol)
            assert fp is not None, "assertion error: fingerprint is none"
    except TypeError:
        raise TypeError(f"smiles must be str or rdkit.Chem.rdchem.Mol. received {mol}")
    except AssertionError as msg:
        raise AssertionError(msg)

def get_metabolite_metadata(metabolite: Metabolite) -> dict:
    """Get the metabolite metdata from local Recon3D supplementary file or from .mol file if present

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """

    bigg_id = metabolite.id
    try:
        meta_dict = RECON3_METABOLITES[
            RECON3_METABOLITES["BiGG ID"] == bigg_id
        ].to_dict("records")[0]
        meta_dict = {k.lower(): v for k, v in meta_dict.items()}
        meta_dict["found_in_supplementary"] = True
    except:
        meta_dict = {"found_in_supplementary": False, "bigg id": bigg_id}

    molid = metabolite.id.split("[")[0]
    molfile = "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/mol/{}.mol".format(
        molid
    )

    # manually curated
    if molid == "CE6252":
        meta_dict["smiles"] = "Oc1nc(O)c2[n-]c(O)nc2n1"
        return meta_dict
    elif molid == "CE2120":
        meta_dict["smiles"] = "[H]N(CCC1=CN([H])C2=CC(OS(O)(=O)=O)=C(OC)C=C12)C(C)=O"
        return meta_dict

    # if smiles found
    if len(meta_dict.get("smiles", "")):
        try:
            assert_rdkit_fp_safe(meta_dict["smiles"])
        except Exception as e:
            meta_dict["smiles"] = ""
            print("Could not load original smile", e)

    # if smiles not found
    if not len(meta_dict.get("smiles", "")):
        try:
            # try to get smiles from molfile
            mol = Chem.MolFromMolFile(molfile)
            assert_rdkit_fp_safe(mol)
            meta_dict["smiles"] = Chem.MolToSmiles(mol)

        except:
            try:
                # try to pull for vmh website
                vmh_page = requests.get(
                    f"https://www.vmh.life/_api/metabolites/?abbreviation={molid}&format=json"
                ).json()
                smiles = vmh_page["results"][0]["smile"]
                assert_rdkit_fp_safe(smiles)
                meta_dict["smiles"] = vmh_page["results"][0]["smile"]
            except:
                try:
                    # try to pull for chemical databases
                    bigg_scraped_metadata = link_metabolite_to_db(metabolite)
                    if "smiles" in bigg_scraped_metadata and (
                        len(bigg_scraped_metadata["smiles"]) > 0
                        or bigg_scraped_metadata["smiles"] is not None
                    ):
                        assert_rdkit_fp_safe(bigg_scraped_metadata["smiles"])
                        meta_dict.update(bigg_scraped_metadata)
                except:
                    meta_dict["smiles"] = None

    return meta_dict


def populate_empty_with_none(x: str) -> Union[str, None]:
    """Return None for empty strings

    Args:
        x (str): a string

    Returns:
        Union[str, None]: string itself or None
    """
    if x == "" or x == "None":
        return None
    elif isinstance(x, (int, float)):
        return str(x)
    else:
        return x


def get_reaction_elements(rxn: Reaction) -> dict:
    rxn_dict = defaultdict(list)
    rxn_dict["rxn_id"] = rxn.id
    # 1. Get reactants
    reactants = rxn.reactants
    reactant_stoich = list(rxn.get_coefficients(reactants))
    for i, r in enumerate(reactants):
        metabolite_dict = {
            "metabolite_id": r.id,
            "coefficient": reactant_stoich[i],
        }
        metabolite_dict.update(get_metabolite_metadata(r))
        rxn_dict["reactants"].append(metabolite_dict)

    # 2. Get products
    products = rxn.products
    product_stoich = list(rxn.get_coefficients(products))
    for i, p in enumerate(products):
        metabolite_dict = {
            "metabolite_id": p.id,
            "coefficient": product_stoich[i],
        }
        metabolite_dict.update(get_metabolite_metadata(p))
        rxn_dict["products"].append(metabolite_dict)

    # 3. Get proteins
    protein_meta = RECON3_PROTEINS_GEMPRO[
        RECON3_PROTEINS_GEMPRO["m_reaction"] == rxn.id
    ]
    proteins = []
    for i, row in protein_meta.iterrows():
        if row["m_gene"] == "":
            assert row["seq_uniprot"] == ""
            continue

        protein_dict = {
            "uniprot": populate_empty_with_none(row["seq_uniprot"]),
            "bigg_gene_id": populate_empty_with_none(row["m_gene"]),
            "entrez": populate_empty_with_none(row["m_gene"]),
            "is_experimental": populate_empty_with_none(row["struct_is_experimental"]),
            "pdb": populate_empty_with_none(row["struct_pdb"]),
        }
        # check if existing
        fasta_path = f"Recon3D_GP/genes/{row['m_gene']}/{row['m_gene']}_protein/sequences/{row['seq_file']}"
        if os.path.exists(fasta_path):
            fasta_service = FASTA()
            fasta_service.read_fasta(fasta_path)
            protein_dict["protein_sequence"] = (fasta_service.sequence,)

        else:
            if protein_dict["uniprot"]:
                try:
                    protein_dict[
                        "protein_sequence"
                    ] = uniprot_service.get_fasta_sequence(protein_dict["uniprot"])
                except:
                    try:
                        protein_dict[
                            "protein_sequence"
                        ] = uniprot_service.get_fasta_sequence(
                            protein_dict["uniprot"].split("-")[0]
                        )
                    except:
                        protein_dict["protein_sequence"] = None
            else:
                protein_dict["protein_sequence"] = None

        proteins.append(protein_dict)

    rxn_dict["proteins"] = proteins
    return rxn_dict


# Init dataset
dataset = []

# Load model
model = load_matlab_model(
    "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/Recon3D_301/Recon3D_301.mat"
)

reaction1 = get_reaction_elements(model.reactions[0])

# Get list of reactions
dataset = p_map(get_reaction_elements, model.reactions)

json.dump(dataset, open(RECON3_DATASET_PATH, "w"))
