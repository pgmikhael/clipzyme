"""Gets ChEMBL IDs from SMILES."""
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from chembl_webresource_client.new_client import new_client
import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

from constants import CHEMBL_COMPOUND_ID_COLUMN, SMILES_COLUMN


class Args(Tap):
    data_path: Path  # Path to data CSV file.
    save_path: Path  # Path where data with ChEMBL IDs will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def get_chembl_id(smiles: str) -> Optional[str]:
    """Gets a ChEMBL ID for a SMILES."""
    try:
        records = new_client.molecule.filter(
            molecule_structures__canonical_smiles__flexmatch=smiles
        ).only(["molecule_chembl_id"])

        if len(records) > 0:
            return records[0]["molecule_chembl_id"]
    except Exception as e:
        print(f"Exception for {smiles}")
        print(e)

    return None


def get_chembl_ids(args: Args):
    """Gets ChEMBL IDs from SMILES."""
    # Load data
    data = pd.read_csv(args.data_path)

    # Get ChEMBL IDs
    smiles_list = sorted(set(data[SMILES_COLUMN].dropna().unique()))
    smiles_to_chembl_ids = {}
    with Pool() as pool:
        for smiles, chembl_id in tqdm(
            zip(smiles_list, pool.imap(get_chembl_id, smiles_list)),
            total=len(smiles_list),
        ):
            if chembl_id is not None:
                smiles_to_chembl_ids[smiles] = chembl_id

    # Add ChEMBL IDs to data
    data[CHEMBL_COMPOUND_ID_COLUMN] = [
        smiles_to_chembl_ids.get(smiles, np.nan) for smiles in data[SMILES_COLUMN]
    ]

    print(
        f"Number of unique ChEMBL IDs = "
        f"{len({chembl_id for chembl_ids in smiles_to_chembl_ids.values() for chembl_id in chembl_ids}):,}"
    )

    # Save data
    data.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    get_chembl_ids(Args().parse_args())
