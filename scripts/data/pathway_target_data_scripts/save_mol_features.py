"""Computes and saves molecular features for a dataset."""

from multiprocessing import Pool
import os
import shutil
import sys
from typing import List, Tuple
import json
from tqdm import tqdm
from tap import (
    Tap,
)  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import rdkit
from os.path import realpath, dirname

sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(
    os.path.join(dirname(dirname(dirname(realpath(__file__)))), "chemprop_pkg")
)

from chemprop.data import get_smiles
from chemprop.features import (
    get_available_features_generators,
    get_features_generator,
    load_features,
    save_features,
)
from chemprop.utils import makedirs

import hashlib


class Args(Tap):
    data_path: str  # Path to data CSV
    smiles_column: str = None  # Name of the column containing SMILES strings. By default, uses the first column.
    features_generator: str = "rdkit_2d_normalized"  # Type of features to generate
    save_path: str  # Path to .npz file where features will be saved as a compressed numpy archive
    save_frequency: int = 10000  # Frequency with which to save the features
    restart: bool = False  # Whether to not load partially complete featurization and instead start from scratch
    sequential: bool = False  # Whether to run sequentially rather than in parallel
    from_csv: bool = False  # Whether loading from csv
    dataset_name: str  # dataset name to help determine loading of smiles

    def configure(self) -> None:
        self.add_argument(
            "--features_generator", choices=get_available_features_generators()
        )


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def generate_and_save_features(args: Args):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

    :param args: Arguments.
    """

    # Get data and features function
    if args.from_csv:
        # Create directory for save_path
        makedirs(args.save_path, isfile=True)
        smiles = get_smiles(
            path=args.data_path, smiles_columns=args.smiles_column, flatten=True
        )
    elif args.dataset_name in ["nuclei", "condensate"]:
        # Create directory for save_path
        makedirs(args.save_path, isfile=False)
        dataset_obj = json.load(open(args.data_path, "r"))
        smiles = [m["SMILES"] for m in dataset_obj if m["scaffold"] is not None]
        smiles = list(set(smiles))
    else:
        raise NotImplementedError(
            "SMILES can be loaded from csv or manually for the following datasets [depmap]"
        )

    features_generator = get_features_generator(args.features_generator)

    # Load partially complete data
    if args.restart:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)
    else:
        if os.path.exists(args.save_path):
            precomputed_smiles = [
                i.split(".npz")[0] for i in os.listdir(args.save_path)
            ]

    if not os.path.exists(args.save_path):
        makedirs(args.save_path)

    # Build features map function
    smiles = [
        s for s in smiles if md5(s) not in precomputed_smiles
    ]  # restrict to data for which features have not been computed yet

    if args.sequential:
        features_map = map(features_generator, smiles)
    else:
        features_map = Pool().imap(features_generator, smiles)

    # Get features
    for i, feats in tqdm(enumerate(features_map), total=len(smiles)):
        save_features(
            os.path.join(args.save_path, "{}.npz".format(md5(smiles[i]))), feats
        )


if __name__ == "__main__":
    generate_and_save_features(Args().parse_args())
