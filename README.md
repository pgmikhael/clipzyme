[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/pgmikhael/CLIPZyme/blob/main/LICENSE.txt) 
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2402.06748)
<!-- ![version](https://img.shields.io/badge/version-1.0.2-success) -->

# CLIPZyme

Implementation of the paper [**CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes**](https://github.com/pgmikhael/CLIPZyme/blob/main/LICENSE.txt)



Table of contents
=================

<!--ts-->
   * [Installation](#installation)
   * [Screening with CLIPZyme](#screening-with-clipzyme)
        * [Using CLIPZyme's screening set](#using-clipzyme's-screening-set)
        * [Using your own screening set](#using-your-own-screening-set)
            * [Interactive (slow)](#interactive-slow)
            * [Batched (fast)](#batched-fast)
   * [Reproducing published results](#reproducing-published-results)
        * [Data processing](#data-processing)
        * [Training and evaluation](#training-and-evaluation)
   * [Citation](#citation)
    
<!--te-->

# Installation:

1. Clone the repository:
```bash
git clone https://github.com/pgmikhael/CLIPZyme.git
```
2. Install the dependencies:
```bash
conda create env -f environment.yml
pip install clipzyme
```

3. Download ESM-2 checkpoint `esm2_t33_650M_UR50D`. The `esm_dir` argument should point to this directory.
# Screening with CLIPZyme

## Using CLIPZyme's screening set

1. Download the screening set and extract the files into `files/`.

```bash

wget https://github.com/pgmikhael/clipzyme/releases/download/v1.0.0/clipzyme_screening_set.zip

unzip clipzyme_screening_set.zip -d files/
```

```python
import pickle
from clipzyme import CLIPZyme

## Load the screening set
##-----------------------
screenset = pickle.load(open("files/clipzyme_screening_set.p", 'rb'))
screen_hiddens = screenset["hiddens"] # hidden representations (261907, 1280)
screen_unis = screenset["uniprots"] # uniprot ids (261907,)

## Load the model and obtain the hidden representations of a reaction
##-------------------------------------------------------------------
model = CLIPZyme(checkpoint_path="files/clipzyme_model.ckpt")
reaction = "[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH:6]=[O:7].[O:9]=[O:10].[OH2:8]>>[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][C:6](=[O:7])[OH:8].[OH:9][OH:10]"
reaction_embedding = model.extract_reaction_features(reaction=reaction) # (1,1280)

enzyme_scores = screen_hiddens @ reaction_embedding.T # (261907, 1)

```

## Using your own screening set

Prepare your data as a CSV in the following format, and save it as `files/new_data.csv`. For the cases where we wish only to obtain the hidden representations of the sequences, the `reaction` column can be left empty (and vice versa).

| reaction | sequence | protein_id | cif |
|----------|----------|------------|-----|
| [CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH:6]=[O:7].[O:9]=[O:10].[OH2:8]>>[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][C:6](=[O:7])[OH:8].[OH:9][OH:10] |MGLSDGEWQLVLNVWGKVEAD<br>IPGHGQEVLIRLFKGHPETLE<br>KFDKFKHLKSEDEMKASEDLK<br>KHGATVLTALGGILKKKGHHE<br>AELKPLAQSHATKHKIPIKYL<br>EFISEAIIHVLHSRHPGDFGA<br>DAQGAMNKALELFRKDIAAKY<br>KELGYQG | P69905 | 1a0s.cif |


### Interactive (slow)
    
```python
from clipzyme import CLIPZyme
from clipzyme import ReactionDataset

## Create reaction dataset
#-------------------------
reaction_dataset = ReactionDataset(
  dataset_file_path = "files/new_data.csv",
  esm_dir = "/path/to/esm2_dir",
  protein_cache_dir = "/path/to/protein_cache",
)

## Load the model
#----------------
model = CLIPZyme(checkpoint_path="files/clipzyme_model.ckpt")

## For reaction-enzyme pair
#--------------------------
for batch in reaction_dataset:
  output = model(batch) 
  enzyme_scores = output.scores
  protein_hiddens = output.protein_hiddens
  reaction_hiddens = output.reaction_hiddens

## For sequences only
#--------------------
for batch in reaction_dataset:
  protein_hiddens = model.extract_protein_features(batch) 
  
## For reactions only
#--------------------
for batch in reaction_dataset:
  reaction_hiddens = model.extract_reaction_features(batch)

```

### Batched (fast)

1. Update the screening config `configs/screening.json` with the path to your data and indicate what you want to save and where:


```JSON
{
  "dataset_file_path": ["files/new_data.csv"],
  "inference_dir": ["/where/to/save/embeddings_and_scores"],
  "save_hiddens": [true], # whether to save the hidden representations
  "save_predictions": [true], # whether to save the reaction-enzyme pair scores
  "use_as_protein_encoder": [true], # whether to use the model as a protein encoder only
  "use_as_reaction_encoder": [true], # whether to use the model as a reaction encoder only
  "esm_dir": ["/data/esm/checkpoints"], path to ESM-2 checkpoints
  "gpus": [8], # number of gpus to use,
  "protein_cache_dir": ["/path/to/protein_cache"], # where to save the protein cache [optional]
  ...
}
```

If you want to use specific GPUs, you can specify them in the `available_gpus` field. For example, to use GPUs 0, 1, and 2, set `available_gpus` to `["0,1,2"]`.



2. Run the dispatcher with the screening config:

```bash
python scripts/dispatcher.py -c configs/screening.json -l ./logs/
```

3. Load the saved embeddings and scores:

```python
from clipzyme import collect_screening_results

screen_hiddens, screen_unis, enzyme_scores = collect_screening_results("configs/screening.json")

```


---------------------

# Reproducing published results

## Data processing

We obtain the data from the following sources:
- [EnzymeMap:](`https://doi.org/10.5281/zenodo.7841780`) Heid et al. Enzymemap: Curation, validation and data-driven prediction of enzymatic reactions. 2023.
- [Terpene Synthases:](`https://zenodo.org/records/10567437`) Samusevich et al. Discovery and characterization of terpene synthases powered by machine learning. 2024. 

Our processed data is available at [here](`https://doi.org/10.5281/zenodo.5555555`). It consists of the following files:
- `enzymemap.json`: contains the EnzymeMap dataset.
- `terpene_synthases.json`: contains the Terpene Synthases dataset.
- `clipzyme_screening_set.p`: contains the screening set as dict of UniProt IDs and precomputed protein embeddings.
- `uniprot2sequence.p`: contains the mapping form sequence ID to amino acids.


## Training and evaluation
1. To train the models presented in the tables below, run the following command:
    ```
    python scripts/dispatcher.py -c {config_path} -l {log_path}
    ```
    - `{config_path}` is the path to the config file in the table below 
    - `{log_path}` is the path in which to save the log file. 
    
    For example, to run the first row in Table 1, run:
    ```
    python scripts/dispatcher.py -c configs/train/clip_egnn.json -l ./logs/
    ```
2. Once you've trained the model, run the eval config to evaluate the model on the test set. For example, to evaluate the first row in Table 1, run:
    ```
    python scripts/dispatcher.py -c configs/eval/clip_egnn.json -l ./logs/
    ```
3. We perform all analysis in the jupyter notebook included [Results.ipynb](analysis/Results.ipynb). We first calculate the hidden representations of the screening using the eval configs above and collect them into one matrix (saved as a pickle file). These are loaded into the jupyter notebook as well as the test set. All tables are then generated in the notebook.


## Citation

```
@article{mikhael2024clipzyme,
  title={CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes},
  author={Mikhael, Peter G and Chinn, Itamar and Barzilay, Regina},
  journal={arXiv preprint arXiv:2402.06748},
  year={2024}
}
```
