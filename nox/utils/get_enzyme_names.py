import json
import requests
from collections import defaultdict
from tqdm import tqdm
import pickle

json_path = "/Mounts/rbg-storage1/datasets/Enzymes/EnzymeMap/version2/enzymemap_v2_brenda2_processed_single_remove_products_all_changes.json"

data = json.load(open(json_path, "rb"))

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{}"

uniprot2name = defaultdict(list)
all_uniprots = set()
for sample in data:
    uniprot = sample["protein_refs"].replace("[", "").replace("]", "").replace("'", "")
    all_uniprots.add(uniprot)

for uniprot in tqdm(all_uniprots):
    if not uniprot in uniprot2name:
        response = requests.get(UNIPROT_ENTRY_URL.format(uniprot)).json()
        try:
            for name in response["proteinDescription"]["submissionNames"]:
                full_name = name["fullName"]["value"]
                if not full_name in uniprot2name[uniprot]:
                    uniprot2name[uniprot].append(full_name.lower())
        except:
            try:
                full_name = response["proteinDescription"]["recommendedName"][
                    "fullName"
                ]["value"]
                if not full_name in uniprot2name[uniprot]:
                    uniprot2name[uniprot].append(full_name.lower())
            except:
                # print(response)
                continue

with open("uniprot2name.pkl", "wb") as f:
    pickle.dump(uniprot2name, f)
