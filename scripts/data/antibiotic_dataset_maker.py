import pandas as pd
import json
import os

SPARK_DIR = "/Mounts/rbg-storage1/datasets/Metabo/antibiotics/SPARKDataDownload"

# Downloaded from: https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx
STOKES19 = "/Mounts/rbg-storage1/datasets/Metabo/antibiotics/stokes2019antibiotic.csv"

# Create Stokes Dataset
stokes = pd.read_csv(STOKES19)
stokes.columns = [c.lower() for c in stokes.columns]
stokes = stokes.to_dict("records")
json.dump(
    stokes,
    open(
        "/Mounts/rbg-storage1/datasets/Metabo/antibiotics/stokes2019_dataset.json", "w"
    ),
)
