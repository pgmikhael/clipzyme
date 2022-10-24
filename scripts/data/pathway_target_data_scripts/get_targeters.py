import requests, json, sqlite3, argparse
from p_tqdm import p_map
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="Scrape metabolite data from external databases"
)

parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    default="/Mounts/rbg-storage1/datasets/Metabo/datasets/iML1515_dataset.json",
    help="path to json dataset file",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="/Mounts/rbg-storage1/datasets/Metabo/datasets/iML1515_chembl_targeters.json",
    help="path to output file",
)
parser.add_argument(
    "--from_chembl",
    action="store_true",
    default=False,
    help="whether to download data from chembl",
)
parser.add_argument(
    "--from_pubchem",
    action="store_true",
    default=False,
    help="whether to download data from chembl",
)


CHEMBL_UNIPROT_MAPPING = {}
with open(
    "/Mounts/rbg-storage1/datasets/ChEMBL/chembl_uniprot_mapping.txt", "r"
) as txtfile:
    for l in txtfile:
        if l.startswith("#"):
            continue
        uniprot, chembl, _, _ = l.split("\t")
        CHEMBL_UNIPROT_MAPPING[uniprot] = chembl


chembl_path = (
        "/Mounts/rbg-storage1/datasets/ChEMBL/chembl_31/chembl_31_sqlite/chembl_31.db"
    )
CHEMBL_DB = sqlite3.connect(chembl_path)



def get_model_uniprots(metabolic_model):
    proteins = [
        uniprot["id"]
        for rxn_dict in metabolic_model
        for protein_dict in rxn_dict["proteins"]
        for uniprot in protein_dict["database_links"].get("UniProt", [{"id": None}])
    ]
    return tuple(set(proteins))


def get_pubchem_assays(protein_id):
    # https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest-tutorial

    # PUBCHEM
    # get protein information
    protein_pg = requests.get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/protein/accession/{protein_id}/aids/JSON"
    ).json()
    # get assay information
    try:
        aids = protein_pg["InformationList"]["Information"][0]["AID"]
    except:
        return
    return {protein_id: aids}


def get_pubchem_assays_results(aid):
    compound_pg = requests.get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/JSON"
    ).json()

    columns = compound_pg["Table"]["Columns"]["Column"]
    results = []
    for row in compound_pg["Table"]["Row"]:
        results.append({ colname:row["Cell"][i] for i, colname in enumerate(columns)})
    
    # break up cid queries into 100 chunks
    cid_index = columns.index('CID')
    cids = [row["Cell"][cid_index] for row in compound_pg["Table"]["Row"]]
    cid_queries = [','.join(cids[i:(i+100)]) for i in range(0,len(cids), 100)]
    
    smiles_results = []
    for cid_query in cid_queries:
        smiles_data = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_query}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        ).json()
        smiles_results.extend(smiles_data["PropertyTable"]["Properties"])

    for row, rdict in enumerate(results):
        for key in smiles_results[0].keys():
            rdict[key] = smiles_results[row][key]

    return results


def get_chembl_compounds(protein_id):
    cursor = CHEMBL_DB.cursor()
    if protein_id in CHEMBL_UNIPROT_MAPPING:
        chembl_id = CHEMBL_UNIPROT_MAPPING[protein_id]
    else:
        try:
            # get chemblid from uniprot
            cursor.execute(
                f"""SELECT chembl_id FROM target_dictionary td
            INNER JOIN  target_components tc on td.tid=tc.tid
            INNER JOIN component_sequences cseq on tc.component_id=cseq.component_id
            AND cseq.accession = {protein_id};"""
            )
            chembl_id = cursor.fetchall()
            if len(chembl_id) > 0:
                assert len(chembl_id) == 1
            chembl_id = chembl_id[0][0]
        except:
            return

    # get chembl
    cmd = f"""SELECT m.chembl_id AS compound_chembl_id,
    s.canonical_smiles,
    r.compound_key,
    a.description                   AS assay_description,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.activity_comment,
    t.chembl_id                    AS target_chembl_id,
    t.pref_name                    AS target_name,
    t.organism                     AS target_organism
    FROM compound_structures s
    JOIN molecule_dictionary m ON s.molregno = m.molregno
    JOIN compound_records r ON m.molregno = r.molregno
    JOIN docs d ON r.doc_id = d.doc_id
    JOIN activities act ON r.record_id = act.record_id
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary t ON a.tid = t.tid
        AND t.chembl_id = '{chembl_id}'
        AND act.standard_type = 'IC50'
        AND act.standard_units = 'nM';"""
    cursor.execute(cmd)
    compounds = cursor.fetchall()


    if len(compounds) == 0:
        return 

    results = []
    for hit in compounds:
        rdict = {
            key: hit[i] for i, key in enumerate([
                'compund_chembl_id',
                'canonical_smiles',
                'compound_key',
                'assay_description',
                'assay_standard_type',
                'assay_standard_relation',
                'assay_standard_value',
                'assay_standard_units',
                'assay_activity_comment',
                'target_chembl_id'
                'target_name',
                'target_organism'])
        }
        rdict['protein_id'] = protein_id
        results.append(rdict)
        
    return results


if __name__ == "__main__":

    args = parser.parse_args()
    metabolic_model = json.load(open(args.dataset_path, "r"))

    # list of protein ids 
    metabolic_proteins = get_model_uniprots(metabolic_model)
    
    if args.from_chembl:
        # get chembl compounds
        chembl_compounds = p_map(get_chembl_compounds, metabolic_proteins) # list of lists or None

        chembl_targeters = {
            p: c for p,c in zip(metabolic_proteins, chembl_compounds) 
        }

        json.dump(chembl_targeters, open(args.output_file, "w"))

    if args.from_pubchem:
        # pubchem
        pubchem_aids = p_map(get_pubchem_assays, metabolic_proteins) # list of (protein_id, aids)

        aids_union = []
        for r in pubchem_aids:
            if r is not None:
                aids_union.extend(list(r.values())[0])
        aids_union = list(set(aids_union))

        aids_union = [str(a) for a in aids_union]
        aid_queries = [','.join(aids_union[i:(i+100)]) for i in range(0,len(aids_union), 100)]
        
        pubchem_compounds = p_map(get_pubchem_assays_results, aid_queries) # list of lists

        pubchem_compounds = list(set([item for ls in pubchem_compounds for item in ls]))
        pubchem_targeters = defaultdict(list)
        for rdict in pubchem_compounds:
            pubchem_targeters[rdict['Target Accession']].append(rdict)

        assert all([i in metabolic_proteins for i in pubchem_targeters])

        for p in metabolic_proteins:
            if p not in pubchem_targeters:
                pubchem_targeters[p] = None

        json.dump(pubchem_targeters, open(args.output_file, "w"))