## To install requirements:

```
conda create -n clipzyme python=3.10
conda activate clipzyme
python -m pip install rdkit
python -m pip install numpy==1.26.0 pandas==2.1.1 scikit-image==0.19.1 scikit-learn==1.3.2 scipy==1.11.2 tqdm==4.62.3 GitPython==3.1.27 comet-ml==3.28.1 wandb==0.12.19
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install pytorch-lightning==2.0.9 torchmetrics==0.11.4
python -m pip install torch_geometric==2.3.1
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install wget bioservices==1.9.0 pubchempy==1.0.4 openpyxl==3.0.10 transformers==4.25.1 rxn-chem-utils==1.0.4 rxn-utils==1.1.3
python -m pip install biopython p_tqdm einops ninja easydict pyyaml
python -m pip install imageio==2.24.0 ipdb pdbpp networkx==2.8.7 overrides pygsp pyemd moviepy 
python -m pip install molvs==0.1.1 epam.indigo==1.9.0 fair-esm==2.0.0 
```
 
## Steps to reproduce results in the paper
1. Install the requirements (above)by running the following commands:
    - EnzymeMap: `wget https://www.enzymemap.org/data/EnzymeMap.zip && unzip EnzymeMap.zip` <b>THIS IS BOILERPLATE</b>
    - Unannotated EnzymeMap: `wget https://www.enzymemap.org/data/Unannotated_EnzymeMap.zip && unzip Unannotated_EnzymeMap.zip` <b>THIS IS BOILERPLATE</b>
    - Terpene Synthases: `wget https://www.enzymemap.org/data/Terpene_Synthases.zip && unzip Terpene_Synthases.zip` <b>THIS IS BOILERPLATE</b>
3. To train the models presented in the tables below, run the following command:
    ```
    python scripts/dispatcher -c {config_path} -l {log_path}
    ```
    where `{config_path}` is the path to the config file in the table below and `{log_path}` is the path in which to save the log file. For example, to run the first row in Table 1, run:
    ```
    python scripts/dispatcher -c configs/train/clip_egnn.json -l ./logs/
    ```
4. Once you've trained the model, run the eval config to evaluate the model on the test set. For example, to evaluate the first row in Table 1, run:
    ```
    python scripts/dispatcher -c configs/eval/clip_egnn.json -l ./logs/
    ```
5. Finally to compute the metrics, copy the experiment id from the log file (appears in the log file name also) and input it in the evaluation notebook. Run the notebook to print the metrics. 


### Table 1 configs
| Protein Encoder                                   | Reaction Encoder                             | BEDROC<sub>85</sub>(%) | BEDROC<sub>20</sub>(%) | EF<sub>0.05</sub> | EF<sub>0.1</sub> | Train Config Path | Eval Config Path |
| ------------------------------------------------- | -------------------------------------------- | ---------------------- | ---------------------- | ---------------- | ---------------- | ----------- | ------------- |
| ESM                                               | Ours (see Reaction Encoding Section)         | 36.91                  | 53.04                  | 11.93             | 6.84             | `configs/train/clip_esm.json`            | `configs/eval/clip_esm.json` |
| MSA-Transformer (weights frozen) + EGNN           | Ours (see Reaction Encoding Section)         | 28.76                  | 46.53                  | 10.34             | 6.67             |    `configs/train/clip_msa.json`         | `configs/eval/clip_msa.json` |
| ESM (weights frozen) + EGNN                       | CGR [Hoonakker et al. 2011]                  | 38.91                  | 57.58                  | 13.16             | 7.73             |  `configs/train/clip_cgr_egnn.json`           | `configs/eval/clip_cgr_egnn.json` |
| ESM (weights frozen) + EGNN                       | Reaction SMILES                              | 29.94                  | 46.01                  | 10.34             | 6.32             | `configs/train/clip_cgr_rxn_string.json`       | `configs/eval/clip_rxn_str.json` |
| ESM (weights frozen) + EGNN                       | WLDN [Jin et al. 2017]                       | 29.84                  | 46.70                  | 10.71             | 6.41             | `configs/train/clip_wldn.json`            | `configs/eval/clip_wldn.json` |
| ESM (weights frozen) + EGNN                       | Ours (see Reaction Encoding Section)         | 44.69                  | 62.98                  | 14.09             | 8.06             | `configs/train/clip_egnn.json`            | `configs/eval/clip_egnn.json` |

### Table 2 configs
- To reproduce the results in table 2 see the notebook `notebooks/CLIPZyme_CLEAN.ipynb`. It involves using the models trained in Table 1 and evaluating them on the test set.

| EC Level Assigned to Reaction | Method           | BEDROC<sub>85</sub>(%) | BEDROC<sub>20</sub>(%) | EF<sub>0.05</sub> | EF<sub>0.1</sub> | Config Paths |
| ----------------------------- | ---------------- | ---------------------- | ---------------------- | ---------------- | ---------------- | ------------ |
| -                             | CLIPZyme         | 44.69                  | 62.98                  | 14.09            | 8.06             |              |
| Level 1 (x.-.-.-)             | CLEAN            | 0.96                   | 6.53                   | 1.22             | 1.72             |              |
|                               | CLIPZyme + CLEAN | 57.03                  | 78.50                  | 17.84            | 9.56             |              |
| Level 2 (x.x.-.-)             | CLEAN            | 4.86                   | 14.10                  | 3.23             | 2.49             |              |
|                               | CLIPZyme + CLEAN | 75.57                  | 90.20                  | 19.40            | 9.84             |              |
| Level 3 (x.x.x.-)             | CLEAN            | 25.86                  | 36.75                  | 8.03             | 4.81             |              |
|                               | CLIPZyme + CLEAN | 82.69                  | 93.23                  | 19.43            | 9.84             |              |
| Level 4 (x.x.x.x)             | CLEAN            | 89.74                  | 93.42                  | 18.97            | 9.60             |              |
|                               | CLIPZyme + CLEAN | 89.57                  | 95.24                  | 19.43            | 9.84             |              |


### Table 3 configs
- To reproduce the results in table 3 see the notebook `notebooks/CLIPZyme_CLEAN.ipynb`. It involves using the models trained in Table 1 and evaluating them on the test set.

| Dataset                | BEDROC<sub>85</sub>(%) | BEDROC<sub>20</sub>(%) | EF<sub>0.05</sub> | EF<sub>0.1</sub> | Config Paths |
| ---------------------- | ---------------------- | ---------------------- | ---------------- | --------------- | ------------ |
| Terpene Synthases      | 72.46                  | 85.89                  | 18.29            | 9.42            |              |
| Unannotated EnzymeMap  | 42.94                  | 61.39                  | 13.92            | 7.73            |              |


### Table 4 configs
- To reproduce the results in table 4 see the notebook `notebooks/CLIPZyme_CLEAN.ipynb`. It involves using the models trained in Table 1 and evaluating them on the test set.

| Exclusion Criteria         | BEDROC<sub>85</sub>(%) | BEDROC<sub>20</sub>(%) | EF<sub>0.05</sub> | EF<sub>0.1</sub> | Config Paths |
| -------------------------- | ---------------------- | ---------------------- | ---------------- | --------------- | ------------ |
| Exact Match                | 39.13                  | 58.86                  | 13.40            | 7.81            |              |
| MMSeqs 90% Similarity      | 34.08                  | 55.25                  | 12.81            | 7.55            |              |
| Foldseek 90% Similarity    | 33.13                  | 47.57                  | 10.28            | 6.23            |              |


### Table 5 configs
- To reproduce the results in table 5 see the notebook `notebooks/CLIPZyme_CLEAN.ipynb`. It involves using the models trained in Table 1 and evaluating them on the test set.

| BEDROC<sub>85</sub>(%) | BEDROC<sub>20</sub>(%) | EF<sub>0.05</sub> | EF<sub>0.1</sub> | Config Paths |
| ---------------------- | ---------------------- | ---------------- | --------------- | ------------ |
| 36.25                  | 51.61                  | 11.30            | 6.83            |              |
