from email.policy import default
from xml.dom import NotFoundErr
import requests, os, json
from cobra.io import load_matlab_model
import pandas as pd
from collections import defaultdict

BIGG_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models_metabolites.txt", sep="\t"
)
METANETX_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/MetaNetX/BiGG/chem_prop.tsv", sep="\t", skiprows=351
)


def link_metabolite_to_metanetx(metabolite):
    """
    Link metabolite to external database
    In order, try:
        1. HMDB
        2. MetaNetX
        3. PubChemPy: pull from API: http://bigg.ucsd.edu/api/v2/models/{model}/metabolites/{metabolite}

    Args:
        metabolite (cobra.Metabolite): metabolite object

    Raises:
        NotFoundErr: _description_

    Returns:
        dict: metadata for metabolite
    """
    # Link to smiles and inchi-key
    meta = BIGG_METABOLITES[BIGG_METABOLITES["bigg_id" == metabolite.id]][
        "database_links"
    ]
    metanetx = [f for f in meta.split(";") if "MetaNetX" in f]
    if len(metanetx):
        metanetid = os.path.basename(metanetx[0].split(":")[-1])
        row = METANETX_METABOLITES[METANETX_METABOLITES["#ID"] == metanetid]
        return {
            "metanetx_id": metanetid,
            "metanetx_name": row["name"],
            "metanetx_mass": row["mass"],
            "InChI": row["InChI"],
            "InChIKey": row["InChIKey"],
            "smiles": row["SMILES"],
        }
    else:
        assert len(meta) == 0
        raise NotFoundErr("Metabolite not found")


# Downloading reactions
models_json = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models.json", "rb")
)
models = [v["bigg_id"] for v in models_json["results"]]
# Get list of reactions

# Load each organism .mat file
for organism_name in models:
    model = load_matlab_model(
        f"/Mounts/rbg-storage1/datasets/Metabo/BiGG/{organism_name}.mat"
    )

    model_dataset = []

    reactions = model.reactions

    # For each reaction
    for rxn in reactions:
        rxn_dict = defaultdict(list)
        # Reaction metadata: compartments, id, name, reverse_id, reverse_variable, reversibility
        # Metabolite metadata: charge,formula,formula_weight,id,name

        # 1. Get reactants
        reactants = rxn.reactants
        reactant_stoich = rxn.get_coefficients(reactants)
        for i, r in enumerate(reactants):
            metabolite_dict = {
                "metabolite": r.id,
                "coefficient": reactant_stoich[i],
            }
            metabolite_dict.update(link_metabolite_to_metanetx(r))
            rxn_dict["reactants"].append(metabolite_dict)

        # 2. Get products
        products = rxn.products
        product_stoich = rxn.get_coefficients(products)
        for i, p in enumerate(products):
            metabolite_dict = {
                "metabolite": p.id,
                "coefficient": product_stoich[i],
            }
            metabolite_dict.update(link_metabolite_to_metanetx(p))
            rxn_dict["products"].append(metabolite_dict)

        # 3. Get reaction sequences by gene id
        proteins = []
        for gene in list(rxn.genes):
            r = requests.get(
                f"http://bigg.ucsd.edu/api/v2/models/{organism_name}/genes/{gene.id}"
            )
            protein_metadata = r.json()
            proteins.append(protein_metadata)

# 3. Get reaction sequences by reaction id
# r = requests.post(
#     "http://bigg.ucsd.edu/advanced_search_sequences", data={"query": rxn.id}
# )

# if r.status_code == 200:
#     json.loads(r.content.decode("utf-8"))
