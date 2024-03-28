[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/pgmikhael/CLIPZyme/blob/main/LICENSE.txt) 
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2402.06748)
<!-- ![version](https://img.shields.io/badge/version-1.0.2-success) -->

# CLIPZyme

Reaction-Conditioned Virtual Screening of Enzymes



Table of contents
=================

<!--ts-->
   * [Installation](#installation)
   * [Screening with CLIPZyme](#screening-with-clipzyme)
        * [Using CLIPZyme's screening set](#using-clipzyme's-screening-set)
        * [Using your own screening set](#using-your-own-screening-set)
   * [Reproducing published results](#reproducing-published-results)
        * [Data processing](#data-processing)
        * [Training and evaluation](#training-and-evaluation)
   * [Citation](#citation)
    
<!--te-->

# Installation:

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
# Screening with CLIPZyme

## Using CLIPZyme's screening set

## Using your own screening set

1. In python shell or jupyter notebook (slow)
    

2. Batched (faster)

---------------------

# Reproducing published results

## Data processing

We obtain the data from the following sources:
- [EnzymeMap:](`https://doi.org/10.5281/zenodo.7841780`) Heid et al. Enzymemap: Curation, validation and data-driven prediction of enzymatic reactions. 2023.
- [Terpene Synthases:](`https://zenodo.org/records/10567437`) Samusevich et al. Discovery and characterization of terpene synthases powered by machine learning. 2024. 

Our processed data is available at [here](`https://doi.org/10.5281/zenodo.5555555`). It consists of the following files:
- `enzymemap.json`: contains the EnzymeMap dataset.
- `terpene_synthases.json`: contains the Terpene Synthases dataset.
- `enzymemap_screening.p`: contains the screening set.
- `sequenceid2sequence.p`: contains the mapping form sequence ID to amino acids.


## Training and evaluation
1. To train the models presented in the tables below, run the following command:
    ```
    python scripts/dispatcher -c {config_path} -l {log_path}
    ```
    - `{config_path}` is the path to the config file in the table below 
    - `{log_path}` is the path in which to save the log file. 
    
    For example, to run the first row in Table 1, run:
    ```
    python scripts/dispatcher -c configs/train/clip_egnn.json -l ./logs/
    ```
2. Once you've trained the model, run the eval config to evaluate the model on the test set. For example, to evaluate the first row in Table 1, run:
    ```
    python scripts/dispatcher -c configs/eval/clip_egnn.json -l ./logs/
    ```
3. We perform all analysis in the jupyter notebook included [CLIPZyme_CLEAN.ipynb](analysis/CLIPZyme_CLEAN.ipynb). We first calculate the hidden representations of the screening using the eval configs above and collect them into one matrix (saved as a pickle file). These are loaded into the jupyter notebook as well as the test set. All tables are then generated in the notebook.


## Citation

```
@article{mikhael2024clipzyme,
  title={CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes},
  author={Mikhael, Peter G and Chinn, Itamar and Barzilay, Regina},
  journal={arXiv preprint arXiv:2402.06748},
  year={2024}
}
```
