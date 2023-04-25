from tqdm import tqdm
import json

meta = json.load(
    open("/Mounts/rbg-storage1/datasets/Metabo/datasets/recon3d_dataset.json", "rb")
)

missing_smiles_name = set()
missing_prots_name = set()
TOTAL_MISSING_SMILES = 0
TOTAL_MISSING_SEQS = 0

metabolites = set()
proteins = set()

for sample in tqdm(meta):
    for reactant in sample["reactants"]:
        metabolites.add(reactant["metabolite_id"])
        if reactant["smiles"] is None or len(reactant["smiles"]) == 0:
            missing_smiles_name.add(reactant["metabolite_id"])
            TOTAL_MISSING_SMILES += 1

    if "products" in sample:
        for product in sample["products"]:
            metabolites.add(product["metabolite_id"])
            if product["smiles"] is None or len(product["smiles"]) == 0:
                missing_smiles_name.add(product["metabolite_id"])
                TOTAL_MISSING_SMILES += 1
    if "proteins" in sample:
        for protein in sample["proteins"]:
            proteins.add(protein["bigg_gene_id"])
            if (
                protein["protein_sequence"] is None
                or len(protein["protein_sequence"]) == 0
            ):
                missing_prots_name.add(protein["bigg_gene_id"])
                TOTAL_MISSING_SEQS += 1

print(
    f"Total missing SMILES: {TOTAL_MISSING_SMILES} out of {len(metabolites)} metabolites, and total missing sequences: {TOTAL_MISSING_SEQS} out of {len(proteins)} proteins"
)
