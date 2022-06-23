import requests, os, json
from cobra.io import load_matlab_model
from cobra.core.metabolite import Metabolite
import pandas as pd
from collections import defaultdict
from bioservices import ChEBI
import warnings
from tqdm import tqdm

BIGG_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models_metabolites.txt", sep="\t"
)
METANETX_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/MetaNetX/chem_prop.tsv",
    sep="\t",
    skiprows=351,
)

HMDB_METABOLITES = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/HMDB/metabolites.json", "r")
)


def link_metabolite_to_db(metabolite: Metabolite) -> dict:
    """
    Link metabolite to external database
    In order, try:
        1. MetaNetX
        2. HMDB
        3. ChEBI

    Args:
        metabolite (cobra.Metabolite): metabolite object

    Raises:
        NotFoundErr: _description_

    Returns:
        dict: metadata for metabolite
    """
    # Link to smiles and inchi-key
    meta = BIGG_METABOLITES[BIGG_METABOLITES["bigg_id"] == metabolite.id][
        "database_links"
    ]
    metanetx = [f for f in meta.values[0].split(";") if "MetaNetX" in f]
    # Try to link to MetaNetX
    meta_dict = {}
    if len(metanetx):
        metanetid = os.path.basename(metanetx[0].split(":")[-1])
        row = METANETX_METABOLITES[METANETX_METABOLITES["#ID"] == metanetid]
        if len(row) > 0:
            meta_dict.update(
                {
                    "metanetx_id": metanetid,
                    "metanetx_name": row["name"].item(),
                    "metanetx_mass": row["mass"].item(),
                    "InChI": row["InChI"].item(),
                    "InChIKey": row["InChIKey"].item(),
                    "smiles": row["SMILES"].item(),
                }
            )

    # HMDB
    hmdb = [f for f in meta.values[0].split(";") if "hmdb" in f]
    if len(hmdb):
        hmdbid = os.path.basename(hmdb[0].split(":")[-1])
        hmdbid = hmdbid[:4] + (11 - len(hmdbid)) * "0" + hmdbid[4:]
        row = HMDB_METABOLITES.get(hmdbid, False)
        if row:
            meta_dict.update(
                {
                    "hmdb_id": hmdbid,
                    "InChI": row["inchi"]
                    if meta_dict.get("InChI", None) is None
                    else meta_dict["InChI"],
                    "InChIKey": row["inchikey"]
                    if meta_dict.get("InChIKey", None) is None
                    else meta_dict["InChIKey"],
                    "smiles": row["smiles"]
                    if meta_dict.get("smiles", None) is None
                    else meta_dict["smiles"],
                }
            )

    # Try to link to ChEBI
    elif not meta_dict.get("smiles", False):
        ch = ChEBI()
        try:
            name_search = ch.getLiteEntity(metabolite.name, maximumResults=5000)
            name_search_ids = [str(r.chebiId) for r in name_search]
        except:
            name_search_ids = []

        try:
            formula_search = ch.getLiteEntity(
                metabolite.formula, searchCategory="FORMULA",  maximumResults=5000
            )
            formula_search_ids = [str(r.chebiId) for r in formula_search]
        except:
            formula_search_ids = []

        if len(formula_search_ids) == 0 and len(name_search_ids) == 0:
            warnings.warn(f"No matches found for metabolite {metabolite.id}")
            return {"smiles": None, "errors": "metabolite not found"}

        # if formula and name exists and have matches, then find overlap, else use non-empty option
        if len(formula_search_ids) > 0 and len(name_search_ids) > 0:
            matched_ids = [m for m in name_search_ids if m in formula_search_ids]
        else:
            matched_ids = name_search_ids + formula_search_ids
        
        all_complete_entities = []
        for j in range(0, len(matched_ids), 50):
            complete_entities = ch.getCompleteEntityByList(matched_ids[j:j+50])
            all_complete_entities.extend(complete_entities)

        exact_matches = [
            i
            for i, met in enumerate(all_complete_entities)
            if met["Formulae"][0].data == metabolite.formula
            or met["chebiAsciiName"].lower() == metabolite.name.lower()
        ]
        
        # if no matches, then find the closest one
        if len(exact_matches) == 0:
            closest_mass = sorted(all_complete_entities, key=lambda x: abs(metabolite.formula_weight - float(x["mass"])))[0]
            exact_matches = [all_complete_entities.index(closest_mass)]
            meta_dict.setdefault("errors", [])
            meta_dict['errors'].append("exact metabolite not found, metabolite with closes mass used")
            warnings.warn(f"exact metabolite not found, metabolite with closes mass used for metabolite {metabolite.id}")

        match = dict(all_complete_entities[exact_matches[0]])

        for key in [
            "chebiId",
            "chebiAsciiName",
            "smiles",
            "inchi",
            "inchiKey",
            "charge",
            "mass",
            "monoisotopicMass",
        ]:
            meta_dict[key] = match[key]

    return meta_dict


geneid2proteinmeta = dict()

# Downloading reactions
models_json = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models.json", "rb")
)
models = [v["bigg_id"] for v in models_json["results"]]
# Get list of reactions

# Load each organism .mat file
for organism_name in tqdm(models[1:]):
    model = load_matlab_model(
        f"/Mounts/rbg-storage1/datasets/Metabo/BiGG/{organism_name}.mat"
    )

    model_dataset = []

    reactions = model.reactions

    # For each reaction
    for rxn in tqdm(reactions, position=0):
        rxn_dict = defaultdict(list)
        # Reaction metadata: compartments, id, name, reverse_id, reverse_variable, reversibility
        # Metabolite metadata: charge,formula,formula_weight,id,name

        # 1. Get reactants
        reactants = rxn.reactants
        reactant_stoich = list(rxn.get_coefficients(reactants))
        for i, r in enumerate(reactants):
            metabolite_dict = {
                "metabolite": r.id,
                "coefficient": reactant_stoich[i],
            }
            metabolite_dict.update(link_metabolite_to_db(r))
            rxn_dict["reactants"].append(metabolite_dict)

        # 2. Get products
        products = rxn.products
        product_stoich = list(rxn.get_coefficients(products))
        for i, p in enumerate(products):
            metabolite_dict = {
                "metabolite": p.id,
                "coefficient": product_stoich[i],
            }
            metabolite_dict.update(link_metabolite_to_db(p))
            rxn_dict["products"].append(metabolite_dict)

        # 3. Get reaction sequences by gene id
        proteins = []
        for gene in list(rxn.genes):
            if gene.id not in geneid2proteinmeta:
                r = requests.get(
                    f"http://bigg.ucsd.edu/api/v2/models/{organism_name}/genes/{gene.id}"
                )
                protein_metadata = r.json()
                if (
                    protein_metadata["protein_sequence"] is None
                    or len(protein_metadata["protein_sequence"]) == 0
                ):
                    warnings.warn(f"No protein sequence found for gene {gene.id}")
                    protein_metadata.update(
                        {"protein_sequence": None, "errors": "gene not found"}
                    )
                proteins.append(protein_metadata)
                geneid2proteinmeta[gene.id] = protein_metadata
            else:
                proteins.append(geneid2proteinmeta[gene.id])

        rxn_dict["proteins"] = proteins

# 3. Get reaction sequences by reaction id
# r = requests.post(
#     "http://bigg.ucsd.edu/advanced_search_sequences", data={"query": rxn.id}
# )

# if r.status_code == 200:
#     json.loads(r.content.decode("utf-8"))
