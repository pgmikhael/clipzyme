"""Creates MOA dataset."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tap import Tap

from constants import OTHERS, SMILES_COLUMN


class Args(Tap):
    data_path: Path  # Path to deduplicated data CSV file.
    moa_columns: List[
        str
    ]  # List of column names to use as MOA columns (e.g., "Target_x", "Target_y", "Pathway").
    moa_name: str  # Name of the MOA column in the saved file (e.g., "Target", "Pathway").
    min_mols_per_moa: int = 10  # Minimum number of molecules per MOA.
    save_path: Path  # Path to CSV file where MOA dataset will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def create_moa_dataset(args: Args) -> None:
    """Creates MOA dataset."""
    # Load data
    data = pd.read_csv(args.data_path)
    print(f"Data size = {len(data):,}")

    # Map SMILES to MOAs, removing MOAs that are NaN or Others
    smiles_to_moas = defaultdict(set)

    for moa_column in args.moa_columns:
        data[moa_column].replace(
            {OTHERS: np.nan, OTHERS.lower(): np.nan, "0": np.nan}, inplace=True
        )
        data_with_moa = data.dropna(subset=[moa_column])

        for smiles, moa in zip(data_with_moa[SMILES_COLUMN], data_with_moa[moa_column]):
            smiles_to_moas[smiles].add(moa)

    print(f"Number of SMILES with at least one MOA = {len(smiles_to_moas):,}")

    # Get MOA counts
    moa_counts = Counter(moa for moas in smiles_to_moas.values() for moa in moas)

    # Filter by number of molecules per MOA
    include_moas = {
        moa for moa, count in moa_counts.most_common() if count >= args.min_mols_per_moa
    }
    smiles_to_moas = {
        smiles: sorted(filtered_moas)
        for smiles, moas in smiles_to_moas.items()
        if len(filtered_moas := moas & include_moas) > 0
    }

    # Print stats
    print(
        f"Number of SMILES after removing MOAs with fewer than {args.min_mols_per_moa} "
        f"molecules = {len(smiles_to_moas):,}"
    )

    moa_counts = Counter(moa for moas in smiles_to_moas.values() for moa in moas)

    print(f"Number of MOAs = {len(moa_counts)}\n")

    for moa, count in moa_counts.most_common():
        print(f"{moa} = {count:,}")

    # Create DataFrame with binary MOAs
    smiles_list = sorted(smiles_to_moas.keys())
    moa_list = sorted(moa_counts.keys())
    data = pd.DataFrame(
        data={
            SMILES_COLUMN: smiles_list,
            **{
                moa: [
                    1 if moa in smiles_to_moas[smiles] else 0 for smiles in smiles_list
                ]
                for moa in moa_list
            },
        }
    )

    # Save data
    data.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    create_moa_dataset(Args().parse_args())
