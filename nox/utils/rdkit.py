from typing import Union
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

Molecule = Union[str, Chem.Mol]


def get_rdkit_feature(
    mol: Molecule, radius: int = 2, num_bits: int = 2048, method="morgan_binary"
) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    if method == "morgan_binary":
        features_vec = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=num_bits
        )
    elif method == "morgan_counts":
        features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    elif method == "rdkit_fingerprint":
        features_vec = Chem.RDKFingerprint(mol)

    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features
