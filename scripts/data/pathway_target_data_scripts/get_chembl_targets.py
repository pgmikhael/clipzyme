"""Gets ChEMBL and UniProt targets from ChEMBL compound IDs."""
from collections import defaultdict
from pathlib import Path

from chembl_webresource_client.new_client import new_client
import pandas as pd
from tap import Tap
from tqdm import trange, tqdm

from constants import (
    CHEMBL_COMPOUND_ID_COLUMN,
    SMILES_COLUMN,
)


class Args(Tap):
    data_path: Path  # Path to data CSV file containing compound IDs.
    chunk_size: int = 50  # Chunk size to use when making web requests to ChEMBL.
    chembl_save_path: Path  # Path to CSV file where data with ChEMBL target IDs will be saved.
    uniprot_save_path: Path  # Path to CSV file where data with UniProt target IDs will be saved.

    def process_args(self) -> None:
        self.chembl_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.uniprot_save_path.parent.mkdir(parents=True, exist_ok=True)


def get_chembl_targets(args: Args):
    """Gets ChEMBL and UniProt targets from ChEMBL compound IDs."""
    # Load data
    data = pd.read_csv(args.data_path)

    # Drop data without ChEMBL compound ID or SMILES
    data.dropna(subset=[CHEMBL_COMPOUND_ID_COLUMN, SMILES_COLUMN], inplace=True)

    # Sort data by SMILES for canonical selection between SMILES with same ChEMBL ID
    data.sort_values(by=SMILES_COLUMN, inplace=True)

    # Map ChEMBL compound ID to SMILES
    # NOTE: By default, uses the last SMILES that matches the ChEMBL compound ID
    chembl_compound_id_to_smiles = dict(
        zip(data[CHEMBL_COMPOUND_ID_COLUMN], data[SMILES_COLUMN])
    )

    # Get ChEMBL targets
    chembl_compound_ids = sorted(chembl_compound_id_to_smiles)
    chembl_compound_id_to_chembl_target_ids = defaultdict(set)

    for i in trange(0, len(chembl_compound_ids), args.chunk_size):
        # Use activities to get target from compound
        activities = new_client.activity.filter(
            molecule_chembl_id__in=chembl_compound_ids[i : i + args.chunk_size]
        ).only(["molecule_chembl_id", "target_chembl_id"])

        # Extract target ChEMBL IDs from activities:
        for activity in tqdm(activities):
            chembl_compound_id_to_chembl_target_ids[activity["molecule_chembl_id"]].add(
                activity["target_chembl_id"]
            )

    chembl_compound_id_to_chembl_target_ids = dict(
        chembl_compound_id_to_chembl_target_ids
    )

    # Create DataFrame with ChEMBL targets
    chembl_target_ids = sorted(
        chembl_target_id
        for chembl_target_ids in chembl_compound_id_to_chembl_target_ids.values()
        for chembl_target_id in chembl_target_ids
    )
    chembl_target_data = pd.DataFrame(
        data={
            SMILES_COLUMN: [
                chembl_compound_id_to_smiles[chembl_compound_id]
                for chembl_compound_id in chembl_compound_ids
            ],
            **{
                chembl_target_id: [
                    1
                    if chembl_target_id
                    in chembl_compound_id_to_chembl_target_ids[chembl_compound_id]
                    else 0
                    for chembl_compound_id in chembl_compound_ids
                ]
                for chembl_target_id in chembl_target_ids
            },
        }
    )

    # Save ChEMBL target data
    chembl_target_data.to_csv(args.chembl_save_path, index=False)

    # Get UniProt targets
    chembl_compound_id_to_uniprot_target_ids = defaultdict(set)
    for chembl_compound_id, chembl_target_ids in tqdm(
        chembl_compound_id_to_chembl_target_ids.items()
    ):
        chembl_target_ids = sorted(chembl_target_ids)

        for i in trange(0, len(chembl_target_ids), args.chunk_size):
            targets = new_client.target.filter(
                target_chembl_id__in=chembl_target_ids[i : i + args.chunk_size]
            ).only(["target_components"])
            chembl_compound_id_to_uniprot_target_ids[chembl_compound_id] |= {
                component["accession"]
                for target in targets
                for component in target["target_components"]
            }

    chembl_compound_id_to_uniprot_target_ids = dict(
        chembl_compound_id_to_uniprot_target_ids
    )

    # Create DataFrame with UniProt targets
    uniprot_target_ids = sorted(
        uniprot_target_id
        for uniprot_target_ids in chembl_compound_id_to_uniprot_target_ids.values()
        for uniprot_target_id in uniprot_target_ids
    )
    uniprot_target_data = pd.DataFrame(
        data={
            SMILES_COLUMN: [
                chembl_compound_id_to_smiles[chembl_compound_id]
                for chembl_compound_id in chembl_compound_ids
            ],
            **{
                uniprot_target_id: [
                    1
                    if uniprot_target_id
                    in chembl_compound_id_to_uniprot_target_ids[chembl_compound_id]
                    else 0
                    for chembl_compound_id in chembl_compound_ids
                ]
                for uniprot_target_id in uniprot_target_ids
            },
        }
    )

    # Save UniProt target data
    uniprot_target_data.to_csv(args.uniprot_save_path, index=False)


if __name__ == "__main__":
    get_chembl_targets(Args().parse_args())
