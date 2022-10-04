from typing import Union, Tuple
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import re

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


def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    From: https://github.com/chemprop/chemprop/blob/master/chemprop/rdkit.py

    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def generate_scaffold(
    mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
    include_chirality: bool = False,
) -> str:
    """
    From: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )

    return scaffold

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)