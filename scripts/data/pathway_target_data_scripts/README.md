# Data Scripts

## Conda Environment Installation

```
conda env create -f environment.yml
conda activate data_scripts
```

## Instructions

1. Deduplicate data.

```
python deduplicate_data.py \
    --data_path ~/data/molecules_raw.csv \
    --save_path ~/data/molecules.csv
```

37,562 unique rows with 18,287 unique SMILES, 654 unique targets, and 101 unique pathways.

2. Create pathway dataset. A molecule may have multiple pathways. Only pathways with at least 10 molecules are included.

```
python create_moa_dataset.py \
    --data_path ~/data/molecules.csv \
    --moa_columns Pathway \
    --moa_name Pathway \
    --save_path ~/data/molecules_pathway.csv
```

3,136 unique SMILES and 25 unique pathways.

* Neuronal Signaling = 420
* Microbiology = 387
* Metabolism = 301
* Immunology & Inflammation = 167
* DNA Damage = 163
* Etc.


3. Create target dataset. A molecule may have multiple targets. Only targets with at least 10 molecules are included.

```
python create_moa_dataset.py \
    --data_path ~/data/molecules.csv \
    --moa_columns Target_x Target_y \
    --moa_name Target \
    --save_path ~/data/molecules_target.csv
```

2,340 unique SMILES and 86 unique targets.

* Anti-infection = 358
* Immunology & Inflammation related = 104
* AChR = 77
* Adrenergic Receptor = 75
* DNA/RNA Synthesis = 73
* Etc.


4. Get ChEMBL compound IDs

```
python get_chembl_ids.py \
    --data_path ~/data/molecules.csv \
    --save_path ~/data/molecules_with_chembl_ids.csv
```

6,432 unique ChEMBL IDs found.

5. Map ChEMBL compound IDs to ChEMBL target IDs and UniProt target IDs.

Download and unzip ChEMBL.
```
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29_sqlite.tar.gz
tar xvzf chembl_29_sqlite.tar.gz
mv chembl_29_sqlite/chembl_29.db .
rm -r chembl_29_sqlite
rm chembl_29_sqlite.tar.gz
```

Map SMILES to ChEMBL and UniProt targets.
```
python get_targets.py \
    --data_path ~/data/molecules_with_chembl_ids.csv \
    --chembl_path ~/data/chembl_29.db \
    --chembl_save_path ~/data/molecules_with_chembl_targets.csv \
    --uniprot_save_path ~/data/molecules_with_uniprot_targets.csv
```

ChEMBL targets: 18 targets with at least 100 molecules, 2,312 molecules

UniProt targets: 18 targets with at least 100 molecules, 2,312 molecules
