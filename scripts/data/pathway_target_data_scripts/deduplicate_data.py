"""Deduplicates condensates data."""
from pathlib import Path

import pandas as pd
from tap import Tap

from constants import PATHWAY_COLUMN, SMILES_COLUMN, TARGET_COLUMN


class Args(Tap):
    data_path: Path  # Path to data CSV file.
    save_path: Path  # Path to CSV file where deduplicated data will be saved.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def deduplicate_data(args: Args) -> None:
    """Deduplicates condensates data."""
    # Load data
    data = pd.read_csv(args.data_path)
    print(f"Data size = {len(data):,}")

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)
    print(f"Data size after removing duplicate rows = {len(data):,}")

    # Remove rows without SMILES
    data.dropna(subset=[SMILES_COLUMN], inplace=True)
    print(f"Data size after removing rows without SMILES = {len(data):,}")

    # Print statistics
    print(f"Number of unique SMILES = {data[SMILES_COLUMN].nunique():,}")
    # print(f'Number of unique targets = {len(set(data[f"{TARGET_COLUMN}_x"].dropna().unique()) | set(data[f"{TARGET_COLUMN}_y"].dropna().unique())):,}')
    # print(f'Number of unique pathways = {data[PATHWAY_COLUMN].nunique()}')

    # Save data
    data.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    deduplicate_data(Args().parse_args())
