import xml.etree.ElementTree as ET
import pickle
import rdkit
from rdkit.Chem import SDMolSupplier
import requests 
import json 

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"

def xml2dict(t):
    d = {}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(xml2dict, children):
            for k, v in dc.items():
                if (not k == "") and not ("reference" in k):
                    dd[k.split("}")[-1]].append(v)
        d = {
            t.tag.split("}")[-1]: {
                k.split("}")[-1]: v[0] if len(v) == 1 else v for k, v in dd.items()
            }
        }
    if t.text:
        text = t.text.strip()
        if (not text == "") and not ("reference" in text):
            d[t.tag] = text
    return d

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

def get_smiles(substrate):
    substrate_data = brenda_smiles.get(substrate, None)
    if substrate_data is None:
        return

    if substrate_data.get("chebi_data", False):
        if substrate_data["chebi_data"].get("SMILES", False):
            smi = substrate_data["chebi_data"]["SMILES"]
            if isinstance(substrate_data["chebi_data"]["SMILES"], str) and (
                len(smi) > 0
            ):
                return smi

        if isinstance(substrate_data["chebi_data"]["SMILES"], list):
            has_smiles = [
                "<SMILES>" in l
                for l in substrate_data["chebi_data"].get("InChIKey", [])
            ]
            if any(has_smiles):
                smi = substrate_data["chebi_data"]["InChIKey"][
                    has_smiles.index(1) + 1
                ]
                return smi

        return substrate_data["chebi_data"].get("SMILES", None)

    elif substrate_data.get("pubchem_data", False):
        if isinstance(substrate_data["pubchem_data"], dict):
            return substrate_data["pubchem_data"].get("CanonicalSMILES", None)
        elif isinstance(substrate_data["pubchem_data"], list):
            return substrate_data["pubchem_data"][0].get("CanonicalSMILES", None)
        else:
            raise NotImplementedError
    return

path = "/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank5.xml"

drug_to_properties = {}
reactions = []
for event, elem in ET.iterparse(path, events=("start", "end")):
    if (elem.tag == "{http://www.drugbank.ca}drug") and (event == "end") and (elem.findall("{http://www.drugbank.ca}description")):
        ids = [t.text for t in elem.findall('{http://www.drugbank.ca}drugbank-id')]
        property_types = elem.findall('{http://www.drugbank.ca}calculated-properties/{http://www.drugbank.ca}property/{http://www.drugbank.ca}kind')
        property_vals = elem.findall('{http://www.drugbank.ca}calculated-properties/{http://www.drugbank.ca}property/{http://www.drugbank.ca}value')
        smiles = [l.text for l in property_types]
        if "SMILES" in smiles:
            index = smiles.index("SMILES")
            smile = property_vals[index].text
        else:
            smile = None
        drug_to_properties.update({i:smile for i in ids})
        
        for r in elem.findall("{http://www.drugbank.ca}reactions/{http://www.drugbank.ca}reaction"):
            reactions.append(xml2dict(r)['reaction'])

parser = SDMolSupplier("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/metabolite-structures.sdf")
drugbank_metabolites = {}
for mol in parser:
    try:
        smi = mol.GetProp('SMILES')
        dbid = mol.GetProp('DATABASE_ID')
        drugbank_metabolites[dbid] = smi
    except:
        continue
drug_to_properties.update(drugbank_metabolites)

pickle.dump(drug_to_properties, open("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/smiles.p", "wb"))
pickle.dump(reactions, open("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/reactions.p", 'wb'))

# get uniprots
proteins = set()
for s in all_pickle:
    if s.get('enzymes', None):
        if isinstance(s['enzymes'].get('enzyme', []), list):
            for e in s['enzymes'].get('enzyme', []):
                if 'uniprot-id' in e:
                    proteins.add(e['uniprot-id'])
        else:
            if 'uniprot-id' in s['enzymes']['enzyme']:
                proteins.add(s['enzymes']['enzyme']['uniprot-id'])


protein_to_sequence = {}
for uniprot in tqdm(proteins, ncols= 60):
    fasta = requests.get(UNIPROT_ENTRY_URL.format(uniprot))
    protein_to_sequence[uniprot] = parse_fasta(fasta.text)



# build dataset 
reactions = [s for s in reactions if 'sequence' in s]
molid2smiles = drug_to_properties

unknown_enzymes = set()
dataset = []
for s in tqdm(reactions, ncols=60):
    sample = {}
    sample['num_reactions_in_pathway'] = s['sequence']
    # assumes that same sequence is same reaction
    sample['substrate_id'] = s['left-element']['drugbank-id']
    try:
        sample['substrate'] = Chem.MolToSmiles(
            Chem.MolFromSmiles(
                molid2smiles[
                    s['left-element']['drugbank-id']
                ]
            )
        )
        if sample['substrate'] is None:
            continue
    except:
        continue

    try:
        sample['product'] = Chem.MolToSmiles(Chem.MolFromSmiles(molid2smiles[s['right-element']['drugbank-id']]))
        if sample['product'] is None:
            continue
    except:
        continue
    if 'enzymes' in s:
        if isinstance(s['enzymes']['enzyme'], list):
            enzymes = [e.get('uniprot-id', None) for e in s['enzymes']['enzyme']]
            enzymes = [e for e in enzymes if e is not None]
            if len(enzymes) == 0:
                unknown_enzymes.update([e['drugbank-id'] for e in s['enzymes']['enzyme']])
            
        elif isinstance(s['enzymes']['enzyme'], dict):
            try:
                enzymes = [s['enzymes']['enzyme']['uniprot-id']]
            except:
                unknown_enzymes.add(s['enzymes']['enzyme']['drugbank-id'])
        else:
            print("Weird format")

    else:
        enzymes = []
    sample['uniprot_ids'] = enzymes
    sample['sequences'] = []
    for seq in enzymes:
        sample['sequences'].append(protein_to_sequence[seq])

    dataset.append(sample)


json.dump(dataset, open("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions.json","w"))


# ------------------------------------------------------------------
# dataset with generic reaction substrates and enzymemap exclusion

brenda = json.load(open("/Mounts/rbg-storage1/datasets/Enzymes/Brenda/brenda_2022_2.json","r")) # brenda
drugbank = json.load(open("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions.json", 'r'))
enzymemap = json.load(open("/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/enzymemap_brenda2023.json", 'r'))
ec2uniprot = pickle.load(open("/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/ec2uniprot.p","rb"))

enzymemap_products = set([ Chem.CanonSmiles(product) for sample in enzymemap for product in sample["products"] ])

brenda_uniprots = set()
brenda_uniprots2generic_reaction = defaultdict(set)
ec2generic_reaction = defaultdict(set)
for ec, ecdict in brenda['data'].items():
    if ecdict.get('generic_reaction', False):
        generic_reactants = [a for e in ecdict['generic_reaction'] for a in e.get('educts',[]) ]
        ec2generic_reaction[ec].update(generic_reactants)
        
    if 'proteins' not in ecdict:
        continue
        
    for idx, plist in ecdict['proteins'].items():
        for acc in plist:
            ecproteins = [u for u in acc['accessions']]
            brenda_uniprots.update(ecproteins)

drugbank_proteins = [u for d in drugbank for u in d['uniprot_ids']]

ecuniprots = set([u for ec,unis in ec2uniprot.items() for u in unis])  # ec number uniprots
uniprots = list(set([u for d in drugbank for u in d['uniprot_ids']])) # drugbank uniprots
uniprot2ecs = { u: set([ec for ec,unis in ec2uniprot.items() if u in unis]) for u in uniprots } # drugbank uni to ec list

ec2smiles = {}
for ec, names in ec2generic_reaction.items():
    ec2smiles[ec] = set(get_smiles(s) for s in names)
    ec2smiles[ec] = set(s for s in ec2smiles[ec] if s is not None)

drugbank_with_coreactant = []
for reaction in drugbank:
    coreactantslist = []
    uniprot_associated_with_ec = any( p in uniprot2ecs for p in reaction['uniprot_ids'] )
    new_product =  Chem.CanonSmiles(reaction['product']) not in enzymemap_products
    
    ecs = [ec for u in reaction['uniprot_ids'] for ec in uniprot2ecs.get(u,None) if ec is not None]
    ec_has_mol = len(set(s for ec in ecs for s in ec2smiles.get(ec, ""))) > 0
    if uniprot_associated_with_ec and new_product and ec_has_mol:
        for u in reaction['uniprot_ids']:
            ecs = [ec for ec in uniprot2ecs.get(u,None) if ec is not None]
            coreactants = list(set(s for s in ec2smiles.get(ec, "") for ec in ecs))
            coreactantslist.append(coreactants)
        drugbank_with_coreactant.append(reaction)
        drugbank_with_coreactant[-1]['co_reactants'] = coreactantslist

json.dump(drugbank_with_coreactant, open("/Mounts/rbg-storage1/users/pgmikhael/DrugBank/drugbank_reactions_with_reactants.json", 'w'), indent=2)