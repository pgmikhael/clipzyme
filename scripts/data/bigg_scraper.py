import requests, os, json, wget
from cobra.io import load_matlab_model
from cobra.core.metabolite import Metabolite
import pandas as pd
from collections import defaultdict
from bioservices import ChEBI, KEGG, UniProt
import pubchempy
import warnings
from tqdm import tqdm
from xml.etree import cElementTree as ET
import argparse
from rdkit import Chem

parser = argparse.ArgumentParser(
    description="Scrape metabolite data from external databases"
)

parser.add_argument(
    "--download_bigg_models",
    action="store_true",
    default=False,
    help="whether to download models",
)

parser.add_argument(
    "--bigg_model_dir",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/BiGG",
    help="path to metabolites metadata json file",
)

parser.add_argument(
    "--organism_name",
    type=str,
    required=True,
    default=None,
    help="name of organism that exists in BiGG Models",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/datasets/",
    help="directory to save the dataset",
)

BIGG_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models_metabolites.txt", sep="\t"
)
METANETX_METABOLITES = pd.read_csv(
    "/Mounts/rbg-storage1/datasets/Metabo/MetaNetX/chem_prop.tsv",
    sep="\t",
    skiprows=351,
)
METANETX_METABOLITES.fillna("", inplace=True)
HMDB_METABOLITES = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/HMDB/metabolites.json", "r")
)


def xml2dict(t):
    """Transform XML into dictionary

    Args:
        t (ET.XML): xml object

    Returns:
        dict: dictionary version of xml object
    """
    d = {}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(xml2dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}

    if t.text:
        text = t.text.strip()
        if not text == "":
            d[t.tag] = text
    return d


def get_metanetx(db_meta: pd.Series) -> dict:
    """Get the metabolite metdata from local METANETX

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    metanetx = [f for f in db_meta.values[0].split(";") if "MetaNetX" in f]

    if len(metanetx):
        metanetid = os.path.basename(metanetx[0].split(":")[-1])
        row = METANETX_METABOLITES[METANETX_METABOLITES["#ID"] == metanetid]
        if (len(row) > 0) and len(row["SMILES"].item()):
            return {
                "metanetx_id": metanetid,
                "metanetx_name": row["name"].item(),
                "metanetx_mass": row["mass"].item(),
                "metanetx_inchi": row["InChI"].item(),
                "metanetx_inchikey": row["InChIKey"].item(),
                "metanetx_smiles": row["SMILES"].item(),
            }
    return dict()


def get_hmdb(db_meta: pd.Series) -> dict:
    """Get the metabolite metdata from local HMDB

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    hmdb = [f for f in db_meta.values[0].split(";") if "hmdb" in f]
    if len(hmdb):
        hmdbid = os.path.basename(hmdb[0].split(":")[-1])
        hmdbid = hmdbid[:4] + (11 - len(hmdbid)) * "0" + hmdbid[4:]
        row = HMDB_METABOLITES.get(hmdbid, False)
        if row and len(row["smiles"]):
            return {
                "hmdb_id": hmdbid,
                "hmdb_inchi": row["inchi"],
                "hmdb_inchikey": row["inchikey"],
                "hmdb_smiles": row["smiles"],
            }
    return dict()


def get_vmh(metabolite: Metabolite) -> dict:
    """Get the metabolite metdata from local HMDB

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    if metabolite.id.endswith(f"_{metabolite.compartment}"):
        bigg_id = "_".join(metabolite.id.split("_")[:-1])
    elif metabolite.id.endswith(f"[{metabolite.compartment}]"):
        bigg_id = metabolite.id.split("[")[0]
    else:
        bigg_id = metabolite.id

    vmh_page = requests.get(
        f"https://www.vmh.life/_api/metabolites/?abbreviation={bigg_id}&format=json"
    ).json()
    try:
        meta_dict = dict()
        assert len(vmh_page["results"]) == 1
        for key in (
            "charge",
            "avgmolweight",
            "monoisotopicweight",
            "biggId",
            "keggId",
            "pubChemId",
            "cheBlId",
            "chembl",
            "inchiString",
            "inchiKey",
            "hmdb",
            "food_db",
            "biocyc",
            "drugbank",
        ):

            meta_dict[f"vmh_{key}"] = vmh_page["results"][0].get(key, None)
        meta_dict["vmh_smiles"] = vmh_page["results"][0]["smile"]
        return meta_dict
    except:
        return dict()


def get_biocyc(db_meta: pd.Series) -> dict:
    """Get the metabolite metdata from BioCyc website

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    biocyc = [f for f in db_meta.values[0].split(";") if "biocyc" in f]
    if len(biocyc):
        biocycid = os.path.basename(biocyc[0])
        page = requests.get(
            f"https://websvc.biocyc.org/getxml?id={biocycid}&detail=full"
        )
        xml_data = xml2dict(ET.XML(page.text))["ptools-xml"]
        if xml_data.get("Compound", False) and len(
            xml_data["Compound"]["cml"]["molecule"].get("string", "")
        ):
            return {
                "biocyc_id": xml_data["metadata"]["query"],
                "biocyc_inchi": xml_data["Compound"].get("inchi", None),
                "biocyc_inchikey": xml_data["Compound"].get("inchi-key", None),
                "biocyc_smiles": xml_data["Compound"]["cml"]["molecule"]["string"],
            }
    return dict()


def get_kegg(db_meta: pd.Series) -> dict:
    """Get the metabolite metdata from KEGG website

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    kegg = [f for f in db_meta.values[0].split(";") if "kegg" in f]
    meta_dict = {}
    if len(kegg):
        keggid = os.path.basename(kegg[0].split(":")[-1])

        # map to pubchem
        map_kegg_pubchem = kegg_service.conv("pubchem", keggid)
        ktype, kname = list(map_kegg_pubchem.keys())[0].split(":")
        assert kname == keggid
        pubchem_id = map_kegg_pubchem[f"{ktype}:{keggid}"].split(":")[-1]

        compounds = pubchempy.get_compounds(pubchem_id, namespace="cid")
        if len(compounds) > 1:
            warnings.warn("Found more than one compound for cid: {pubchem_id}")

        if len(compounds) == 1:
            meta_dict = {
                "pubchem_inchikey": compounds[0].inchikey,
                "pubchem_inchi": compounds[0].inchi,
                "pubchem_smiles": compounds[0].canonical_smiles,
                "pubchem_isomeric_smiles": compounds[0].isomeric_smiles,
            }
            return meta_dict

        else:
            # map to chebi

            map_kegg_chebi = kegg_service.conv("chebi", keggid)

            # map to chebi
            if isinstance(map_kegg_chebi, dict):
                ktype, kname = list(map_kegg_chebi.keys())[0].split(":")
                assert kname == keggid
                chebi_id = map_kegg_chebi[f"{ktype}:{keggid}"]
                mol = chebi_service.getCompleteEntity(chebi_id.upper())
                mol = dict(mol)

                if len(mol["smiles"]):
                    for dictkey, dbkey in [
                        ("id", "chebiId"),
                        ("ascii_ame", "chebiAsciiName"),
                        ("smiles", "smiles"),
                        ("inchi", "inchi"),
                        ("inchikey", "inchiKey"),
                        ("charge", "charge"),
                        ("mass", "mass"),
                        ("monoisotopic_mass", "monoisotopicMass"),
                    ]:
                        meta_dict[f"chebi_{dictkey}"] = mol.get(dbkey, None)
                return meta_dict

    return dict()


def get_pubchem(db_meta: pd.Series) -> dict:
    """Get the metabolite metdata from PubChem website based on Inchi Key

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    inchikey = [f for f in db_meta.values[0].split(";") if "inchikey" in f]
    if len(inchikey):
        inchikey_num = os.path.basename(inchikey[0].split(":")[-1])
        compounds = pubchempy.get_compounds(inchikey_num, namespace="inchikey")
        if len(compounds) == 0:
            warnings.warn("Did not find compounds with inchikey: {inchikey_num}")
            return dict()
        if len(compounds) > 1:
            warnings.warn("Found more than one compound for inchikey: {inchikey_num}")

        return {
            "pubchem_inchikey": compounds[0].inchikey,
            "pubchem_inchi": compounds[0].inchi,
            "pubchem_smiles": compounds[0].canonical_smiles,
            "pubchem_isomeric_smiles": compounds[0].isomeric_smiles,
        }
    return dict()


def search_chebi(metabolite: Metabolite) -> dict:
    """Search for metabolite in ChEBI using name and/or formula

    Args:
        db_meta (pandas row): pandas row matching metabolite to external links

    Returns:
        dict: metabolite properties
    """
    meta_dict = dict()
    try:
        name_search = chebi_service.getLiteEntity(metabolite.name, maximumResults=5000)
        name_search_ids = [str(r.chebiId) for r in name_search]
    except:
        name_search_ids = []

    try:
        formula_search = chebi_service.getLiteEntity(
            metabolite.formula, searchCategory="FORMULA", maximumResults=5000
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
        complete_entities = chebi_service.getCompleteEntityByList(
            matched_ids[j : j + 50]
        )
        all_complete_entities.extend(complete_entities)

    exact_matches = [
        i
        for i, met in enumerate(all_complete_entities)
        if met["Formulae"][0].data == metabolite.formula
        or met["chebiAsciiName"].lower() == metabolite.name.lower()
    ]

    # if no matches, then find the closest one
    if len(exact_matches) == 0:
        if metabolite.formula_weight:
            closest_mass = sorted(
                all_complete_entities,
                key=lambda x: abs(metabolite.formula_weight - float(x["mass"])),
            )[0]
        else:
            return meta_dict

        exact_matches = [all_complete_entities.index(closest_mass)]
        meta_dict.setdefault("errors", [])
        meta_dict["errors"].append(
            "exact metabolite not found, metabolite with closest mass used"
        )
        warnings.warn(
            f"exact metabolite not found, metabolite with closest mass used for metabolite {metabolite.id}"
        )

    match = dict(all_complete_entities[exact_matches[0]])

    for dictkey, dbkey in [
        ("id", "chebiId"),
        ("ascii_ame", "chebiAsciiName"),
        ("smiles", "smiles"),
        ("inchi", "inchi"),
        ("inchikey", "inchiKey"),
        ("charge", "charge"),
        ("mass", "mass"),
        ("monoisotopic_mass", "monoisotopicMass"),
    ]:
        meta_dict[f"chebi_{dictkey}"] = match[dbkey]

    return meta_dict


def link_metabolite_to_db(metabolite: Metabolite) -> dict:
    """
    Link metabolite to external database
    Collect fit:
        - MetaNetX
        - HMDB
        - BioCyc
        - VMH
        - KEGG
        - PubChem
        - ChEBI DB Search
    Args:
        metabolite (cobra.Metabolite): metabolite object

    Returns:
        dict: metadata for metabolite
    """
    db_meta = BIGG_METABOLITES[BIGG_METABOLITES["bigg_id"] == metabolite.id][
        "database_links"
    ]

    # Try HMDB
    meta_dict = get_hmdb(db_meta)

    if not any("smiles" in k for k in meta_dict.keys()):
        # Try MetaNetX
        meta_dict.update(get_metanetx(db_meta))

    if not any("smiles" in k for k in meta_dict.keys()):
        # Try VMH
        meta_dict.update(get_vmh(metabolite))

    if not any("smiles" in k for k in meta_dict.keys()):
        # Try BioCyc
        meta_dict.update(get_biocyc(db_meta))

    if not any("smiles" in k for k in meta_dict.keys()):
        # Try KEGG
        meta_dict.update(get_kegg(db_meta))

    if not any("smiles" in k for k in meta_dict.keys()):
        # Try PubChem
        meta_dict.update(get_pubchem(db_meta))

    # Try to link to ChEBI
    if not any("smiles" in k for k in meta_dict.keys()):
        meta_dict.update(search_chebi(metabolite))

    # standardize SMILES
    if any("smiles" in k for k in meta_dict.keys()):
        smiles_key = [k for k in meta_dict.keys() if "smiles" in k][0]
        meta_dict["smiles"] = Chem.CanonSmiles(meta_dict[smiles_key])
    else:
        meta_dict["smiles"] = None

    return meta_dict


if __name__ == "__main__":
    args = parser.parse_args()

    if args.download_bigg_models:
        # get list of non-multistrain models
        model_list = requests.get("http://bigg.ucsd.edu/api/v2/models?multistrain=off")
        json.dump(
            model_list.json(),
            open(
                os.path.join(args.bigg_model_dir, "bigg_models_single_strain.json"), "w"
            ),
        )

        # download all models
        model_list = requests.get("http://bigg.ucsd.edu/api/v2/models")
        model_list = model_list.json()
        json.dump(
            model_list, open(os.path.join(args.bigg_model_dir, "bigg_models.json"), "w")
        )

        for model in model_list["results"]:
            wget.download(f"http://bigg.ucsd.edu/static/models/{model["bigg_id"]}.mat",  args.bigg_model_dir)
                

    geneid2proteinmeta = dict()
    kegg_service = KEGG()
    chebi_service = ChEBI()
    uniprot_service = UniProt()

    dataset = []

    model = load_matlab_model(
        os.path.join(args.bigg_model_dir, f"{args.organism_name}.mat")
    )

    # Get list of reactions
    reactions = model.reactions

    # For each reaction
    for rxn in tqdm(reactions, position=0):
        rxn_dict = defaultdict(list)
        rxn_dict["rxn_id"] = rxn.id
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
        could_not_find_gene = []
        for gene in list(rxn.genes):
            if gene.id not in geneid2proteinmeta:
                r = requests.get(
                    f"http://bigg.ucsd.edu/api/v2/models/{args.organism_name}/genes/{gene.id}"
                )
                protein_metadata = r.json()
                if (
                    protein_metadata["protein_sequence"] is None
                    or len(protein_metadata["protein_sequence"]) == 0
                ):
                    warnings.warn(f"No protein sequence found for gene {gene.id}")
                    could_not_find_gene.append(gene)
                    protein_metadata.update(
                        {"protein_sequence": None, "errors": "gene not found"}
                    )
                protein_metadata["bigg_gene_id"] = gene.id
                proteins.append(protein_metadata)
                geneid2proteinmeta[gene.id] = protein_metadata
            else:
                proteins.append(geneid2proteinmeta[gene.id])

        # uniprot_service.mapping(fr="P_ENTREZGENEID", to="ID", query="55577")
        # new_sequences = GEMPRO.uniprot - find - prot - gene - seq - now()
        # proteins.extend(new_sequences)

        rxn_dict["proteins"] = proteins

        # 3. Get reaction sequences by reaction id
        # r = requests.post(
        #     "http://bigg.ucsd.edu/advanced_search_sequences", data={"query": rxn.id}
        # )

        # if r.status_code == 200:
        #     json.loads(r.content.decode("utf-8"))

        dataset.append(rxn_dict)

    json.dump(
        dataset,
        open(os.path.join(args.save_dir, f"{args.organism_name}_dataset.json"), "w"),
    )
