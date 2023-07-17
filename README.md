<div align="center">

**nox**

a repository for all sorts of research 

therefore a space of darkness and chaos

-----------------------------------

</div>

To install:

```
conda create -n nox python=3.10
conda install -c conda-forge rdkit
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m pip install pytorch-lightning torchmetrics pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

python -m pip install ninja easydict pyyaml opencv-python==4.5.4.60 albumentations==1.1.0 scikit-image==0.19.1 scikit-learn==1.0.1 GitPython==3.1.27 comet-ml==3.28.1 wandb tqdm wget bioservices==1.9.0 pubchempy==1.0.4 openpyxl==3.0.10 rxn-chem-utils==1.0.4 rxn-utils==1.1.3 frozendict==2.3.4 plotille==5.0.0 biopython p_tqdm einops imageio==2.24.0 ipdb pdbpp networkx==2.8.7 overrides pygsp pyemd moviepy epam.indigo MolVS fair-esm overrides cobra transformers --no-cache-dir
```
 
