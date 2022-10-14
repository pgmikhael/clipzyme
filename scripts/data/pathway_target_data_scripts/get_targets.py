"""Get ChEMBL and UniProt targets from ChEMBL IDs."""
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3

import pandas as pd
from tap import Tap

from constants import (
    ACTIVE_COMMENTS,
    CHEMBL_COMPOUND_ID_COLUMN,
    SMILES_COLUMN,
)


class Args(Tap):
    data_path: Path  # Path to data CSV file containing compound IDs.
    chembl_path: Path  # Path to ChEMBL SQL database.
    min_mols_per_target: int = 100  # Minimum number of molecules per target.
    chembl_save_path: Path  # Path to CSV file where data with ChEMBL target IDs will be saved.
    uniprot_save_path: Path  # Path to CSV file where data with UniProt target IDs will be saved.

    def process_args(self) -> None:
        self.chembl_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.uniprot_save_path.parent.mkdir(parents=True, exist_ok=True)


def get_targets(args: Args) -> None:
    """Get ChEMBL and UniProt targets from ChEMBL IDs."""
    # Load data
    data = pd.read_csv(args.data_path)

    # Only include rows with compound IDs
    data.dropna(subset=[CHEMBL_COMPOUND_ID_COLUMN], inplace=True)

    # Map compound ID to SMILES
    # NOTE: By default, uses the last SMILES that matches the ChEMBL compound ID
    compound_id_to_smiles = dict(zip(data[CHEMBL_COMPOUND_ID_COLUMN], data[SMILES_COLUMN]))
    compound_ids = sorted(compound_id_to_smiles)
    print(f'Number of ChEMBL compound IDs = {len(compound_ids):,}')

    # Create SQL command
    query = [
        'SELECT md.chembl_id AS compound_id,',
        'td.chembl_id AS chembl_target_id,',
        'cseq.accession AS uniprot_target_id',
        'FROM component_sequences cseq',
        '  JOIN target_components tc ON cseq.component_id = tc.component_id',
        '  JOIN target_dictionary td ON tc.tid = td.tid',
        '  JOIN assays a ON td.tid = a.tid',
        '  JOIN activities act ON a.assay_id = act.assay_id',
        '  JOIN molecule_dictionary md ON act.molregno = md.molregno',
        f'    AND act.activity_comment IN {tuple(ACTIVE_COMMENTS)}',
        f'WHERE md.chembl_id IN {tuple(compound_ids)}'
    ]
    command = '\n'.join(query)

    # Connect to ChEMBL SQL database
    conn = sqlite3.connect(args.chembl_path)
    cursor = conn.cursor()

    # Execute SQL command
    cursor.execute(command)
    results = cursor.fetchall()
    print(f'Number of results = {len(results):,}')

    # Map compound ID to target IDs
    compound_id_to_chembl_target_ids = defaultdict(set)
    compound_id_to_uniprot_target_ids = defaultdict(set)
    for result in results:
        compound_id, chembl_target_id, uniprot_target_id = result
        compound_id_to_chembl_target_ids[compound_id].add(chembl_target_id)
        compound_id_to_uniprot_target_ids[compound_id].add(uniprot_target_id)

    # Create datasets for ChEMBL and UniProt targets
    for name, compound_id_to_target_ids, save_path in [
        ('ChEMBL', compound_id_to_chembl_target_ids, args.chembl_save_path),
        ('UniProt', compound_id_to_uniprot_target_ids, args.uniprot_save_path)
    ]:
        # Create DataFrame for targets
        compound_ids = set(compound_id_to_target_ids)
        target_ids = sorted({
            target_id
            for target_ids in compound_id_to_target_ids.values()
            for target_id in target_ids if target_id is not None
        })

        target_data = pd.DataFrame(data={
            SMILES_COLUMN: [compound_id_to_smiles[compound_id] for compound_id in compound_ids],
            **{
                target_id: [
                    1 if target_id in compound_id_to_target_ids[compound_id] else 0
                    for compound_id in compound_ids
                ]
                for target_id in target_ids
            }
        })

        # Filter out targets with too few compounds
        include_target_ids = [
            target_id for target_id in target_ids if target_data[target_id].sum() >= args.min_mols_per_target
        ]
        target_data = target_data[[SMILES_COLUMN] + include_target_ids]

        # Print stats
        print()
        print(f'Target type = {name}')
        print(f'Number of unique compounds = {len(compound_ids):,}')
        print(f'Number of unique targets = {len(include_target_ids):,}')
        target_counts = sorted((target_data[target_id].sum() for target_id in include_target_ids), reverse=True)
        print(f'Top 10 target counts = {target_counts[:10]}')

        # Save ChEMBL target data
        target_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    get_targets(Args().parse_args())
