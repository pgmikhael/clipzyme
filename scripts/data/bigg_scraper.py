import requests, os, json
from cobra.io import load_matlab_model
from cobra.core.metabolite import Metabolite
import pandas as pd
from collections import defaultdict
from bioservices import ChEBI, KEGG, UniProt
import pubchempy
import warnings
from tqdm import tqdm
from xml.etree import cElementTree as ET

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


def get_from_metanetx(db_meta) -> dict:
    """
    Get the metabolite metdata from MetaNetX
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


def get_hmdb(db_meta) -> dict:
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


def get_biocyc(db_meta) -> dict:
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


def get_kegg(db_meta) -> dict:
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


def get_pubchem(db_meta) -> dict:
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


def link_metabolite_to_db(metabolite: Metabolite) -> dict:
    """
    Link metabolite to external database
    Collect fit:
        - MetaNetX
        - HMDB
        - BioCyc
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
        meta_dict.update(get_from_metanetx(db_meta))

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
        try:
            name_search = chebi_service.getLiteEntity(
                metabolite.name, maximumResults=5000
            )
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


geneid2proteinmeta = dict()
kegg_service = KEGG()
chebi_service = ChEBI()
uniprot_service = UniProt()

# Downloading reactions
models_json = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/BiGG/bigg_models.json", "rb")
)
models = [v["bigg_id"] for v in models_json["results"]]


dataset = defaultdict(list)
# Load each organism .mat file
for organism_name in tqdm(models):
    model = load_matlab_model(
        f"/Mounts/rbg-storage1/datasets/Metabo/BiGG/{organism_name}.mat"
    )

    # Get list of reactions
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
        could_not_find_gene = []
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
                    could_not_find_gene.append(gene)
                    protein_metadata.update(
                        {"protein_sequence": None, "errors": "gene not found"}
                    )
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

        dataset[organism_name].append(rxn_dict)
