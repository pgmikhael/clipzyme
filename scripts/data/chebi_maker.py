from tqdm import tqdm
import json 
from p_tqdm import p_umap

"""
https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/
"""

filename = "/Mounts/rbg-storage1/datasets/Metabo/ChEBI_complete.sdf"

with open(filename, "r") as f:
    molfile = f.read()
mols = molfile.split("$$$$")

if mols[-1] == '\n':
    mols = mols[:-1]
    
# chebi_db = {}
# for m in tqdm(mols, total = len(mols)):
def make_db(m):
    mol_db = {}
    moldata = [k for k in m.split("\n") if k != ""]
    entries = [l for l in moldata if l.startswith('> <')]

    for i,j in zip(range(len(entries) - 1), range(1, len(entries))):
        start = m.index(entries[i]) + len(entries[i])
        end = m.index(entries[j])
        entry_values = [k for k in m[start:end].split("\n") if k!= ""]
        if len(entry_values) == 1:
            entry_values = entry_values[0]
            
        mol_db[ entries[i].lstrip("> <").rstrip(">") ] = entry_values
        
    # chebi_db[mol_db["ChEBI ID"]] = mol_db
    return (mol_db["ChEBI ID"], mol_db)

if __name__ == "__main__":

    with open(filename, "r") as f:
        molfile = f.read()
    mols = molfile.split("$$$$")

    if mols[-1] == '\n':
        mols = mols[:-1]
    
    chebi_db = p_umap(make_db, mols)
    chebi_db = {chebid: moldb for chebid, moldb in chebi_db}

    json.dump(chebi_db, open("/Mounts/rbg-storage1/datasets/Metabo/chebi_db.json", "w"))