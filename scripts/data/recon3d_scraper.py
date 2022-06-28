import os, json
from cobra.io import load_matlab_model
from cobra.core.metabolite import Metabolite
from cobra.core.reaction import Reaction
import pandas as pd
from collections import defaultdict
from bioservices import UniProt
from p_tqdm import p_umap
import rdkit.Chem as Chem
from typing import Union
import warnings
warnings.filterwarnings("ignore")

RECON3_METABOLITES = pd.read_excel(
    "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/41587_2018_BFnbt4072_MOESM11_ESM.xlsx",
    sheet_name="Supplementary Data File 14",
)
RECON3_METABOLITES.fillna("", inplace=True)

RECON3_PROTEINS = pd.read_excel(
    "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/41587_2018_BFnbt4072_MOESM11_ESM.xlsx",
    sheet_name="Supplementary Data File 11",
)
RECON3_PROTEINS.fillna("", inplace=True)

RECON3_DATASET_PATH = "/Mounts/rbg-storage1/datasets/Metabo/recon3d_dataset.json"

uniprot_service = UniProt(verbose=False)


def get_metabolite_metadata(metabolite: Metabolite) -> dict:
    """Get the metabolite metdata from local Recon3D supplementary file or from .mol file if present

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """

    bigg_id = metabolite.id
    try:
        meta_dict = RECON3_METABOLITES[RECON3_METABOLITES["BiGG ID"] == bigg_id].to_dict(
            "records"
        )[0]
        meta_dict = {k.lower(): v for k, v in meta_dict.items()}
        meta_dict['found_in_supplementary'] = True
    except:
        meta_dict = {'found_in_supplementary': False, "bigg id": bigg_id}


    molid = metabolite.id.split("[")[0]
    molfile = "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/mol/{}.mol".format(
        molid
    )
    if not len(meta_dict.get("smiles", "")) and os.path.isfile(molfile):
        mol = Chem.MolFromMolFile(molfile)
        smiles = Chem.MolToSmiles(mol)
        meta_dict["smiles"] = smiles
    else:
        meta_dict["smiles"] = None
    return meta_dict


def populate_empty_with_none(x: str) -> Union[str, None]:
    """Return None for empty strings

    Args:
        x (str): a string

    Returns:
        Union[str, None]: string itself or None
    """
    return None if x == "" else x

def get_reaction_elements(rxn: Reaction) -> dict:
    rxn_dict = defaultdict(list)

    # 1. Get reactants
    reactants = rxn.reactants
    reactant_stoich = list(rxn.get_coefficients(reactants))
    for i, r in enumerate(reactants):
        metabolite_dict = {
            "metabolite": r.id,
            "coefficient": reactant_stoich[i],
        }
        metabolite_dict.update(get_metabolite_metadata(r))
        rxn_dict["reactants"].append(metabolite_dict)

    # 2. Get products
    products = rxn.products
    product_stoich = list(rxn.get_coefficients(products))
    for i, p in enumerate(products):
        metabolite_dict = {
            "metabolite": p.id,
            "coefficient": product_stoich[i],
        }
        metabolite_dict.update(get_metabolite_metadata(p))
        rxn_dict["products"].append(metabolite_dict)

    # 3. Get proteins
    protein_meta = RECON3_PROTEINS[RECON3_PROTEINS["m_reaction"] == rxn.id]

    proteins = []
    for i, row in protein_meta.iterrows():
        protein_dict = {
            "uniprot": populate_empty_with_none(row["seq_uniprot"]),
            "entrez": populate_empty_with_none(row["m_gene"]),
            "is_experimental": populate_empty_with_none(row["struct_is_experimental"]),
            "pdb": populate_empty_with_none(row["struct_pdb"]),
        }
        if protein_dict["uniprot"]:
            try:
                protein_dict['sequence'] = uniprot_service.get_fasta_sequence(protein_dict["uniprot"])
            except:
                try:
                    protein_dict['sequence'] = uniprot_service.get_fasta_sequence(protein_dict["uniprot"].split('-')[0])
                except:
                    protein_dict['sequence'] = None
        else:
            protein_dict['sequence'] = None

        proteins.append(protein_dict)

    rxn_dict["proteins"] = proteins
    return rxn_dict

# Init dataset
dataset = []

# Load model
model = load_matlab_model(
    "/Mounts/rbg-storage1/datasets/Metabo/VMH/Recon3D/Recon3D_301/Recon3D_301.mat"
)

# Get list of reactions
dataset = p_umap(get_reaction_elements, model.reactions)

json.dump(dataset, open(RECON3_DATASET_PATH, "w"))
