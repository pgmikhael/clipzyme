<div align="center">

**nox**

a repository for all sorts of research 

therefore a space of darkness and chaos

-----------------------------------

</div>

To install:

```
conda create -n nox python=3.8
conda install -c conda-forge rdkit
python -m pip install numpy==1.21.4 opencv-python==4.5.4.60 albumentations==1.1.0 pandas==1.3.5 scikit-image==0.19.1 scikit-learn==1.0.1 scipy==1.7.3 tqdm==4.62.3 GitPython==3.1.27 comet-ml==3.28.1 wandb==0.12.19 --no-cache-dir
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
python -m pip install  pytorch-lightning==1.6.4 torchmetrics==0.6.2 --no-cache-dir
python -m pip install torch-scatter==2.0.9 torch-sparse==0.6.14 torch-geometric==2.0.4 ninja easydict pyyaml -f https://data.pyg.org/whl/torch-1.11.0+cu115.html
python -m pip install wget cobra==0.25.0 bioservices==1.9.0 pubchempy==1.0.4 openpyxl==3.0.10 transformers==4.22.2 rxn-chem-utils==1.0.4 rxn-utils==1.1.3 frozendict==2.3.4 plotille==5.0.0
python -m pip install biopython p_tqdm einops
```
