# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert molecules into DGLGraphs
#
# pylint: disable= no-member, arguments-differ, invalid-name
#
# Adapted from: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/data/uspto.py

import errno
import numpy as np
import os
import random
import torch

from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles
from tqdm import tqdm
from torch_geometric.data import Batch
from clipzyme.utils.pyg import from_smiles, from_mapped_smiles

from functools import partial
import torch

from sklearn.neighbors import NearestNeighbors

try:
    from molvs import Standardizer
except ImportError as e:
    print("MolVS is not installed, which is required for candidate ranking")


BOND_TYPE = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
}

clean_rxns_postsani = [
    # two adjacent aromatic nitrogens should allow for H shift
    Chem.AllChem.ReactionFromSmarts("[n;H1;+0:1]:[n;H0;+1:2]>>[n;H0;+0:1]:[n;H0;+0:2]"),
    # two aromatic nitrogens separated by one should allow for H shift
    Chem.AllChem.ReactionFromSmarts(
        "[n;H1;+0:1]:[c:3]:[n;H0;+1:2]>>[n;H0;+0:1]:[*:3]:[n;H0;+0:2]"
    ),
    Chem.AllChem.ReactionFromSmarts("[#7;H0;+:1]-[O;H1;+0:2]>>[#7;H0;+:1]-[O;H0;-:2]"),
    # neutralize C(=O)[O-]
    Chem.AllChem.ReactionFromSmarts(
        "[C;H0;+0:1](=[O;H0;+0:2])[O;H0;-1:3]>>[C;H0;+0:1](=[O;H0;+0:2])[O;H1;+0:3]"
    ),
    # turn neutral halogens into anions EXCEPT HCl
    Chem.AllChem.ReactionFromSmarts("[I,Br,F;H1;D0;+0:1]>>[*;H0;-1:1]"),
    # inexplicable nitrogen anion in reactants gets fixed in prods
    Chem.AllChem.ReactionFromSmarts("[N;H0;-1:1]([C:2])[C:3]>>[N;H1;+0:1]([*:2])[*:3]"),
]


def robust_edit_mol(rmol, edits):
    """Simulate reaction via graph editing

    Parameters
    ----------
    rmol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants
    bond_changes : list of 3-tuples
        Each tuple is of form (atom1, atom2, change_type)
    keep_atom_map : bool
        Whether to keep atom mapping number. Default to False.

    Returns
    -------
    pred_smiles : list of str
        SMILES for the edited molecule
    """
    ################ for switching IDXes ###########################
    # edits are indices, so we need to replace the atom map number with the index
    old_index2atom_number = {}
    for atom in rmol.GetAtoms():
        old_index2atom_number[atom.GetIdx()] = int(
            atom.GetProp("molAtomMapNumber")
        )  # Update mapping dictionary

    order = sorted(
        old_index2atom_number.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    # old_index2new_index = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)}
    atom_map_number2new_index = {
        atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)
    }
    ######################################################################
    new_mol = Chem.RWMol(rmol)

    # Keep track of aromatic nitrogens, might cause explicit hydrogen issues
    aromatic_nitrogen_idx = set()
    aromatic_carbonyl_adj_to_aromatic_nH = {}
    aromatic_carbondeg3_adj_to_aromatic_nH0 = {}
    for a in new_mol.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == "N":
            aromatic_nitrogen_idx.add(a.GetIdx())
            for nbr in a.GetNeighbors():
                if (
                    a.GetNumExplicitHs() == 1
                    and nbr.GetSymbol() == "C"
                    and nbr.GetIsAromatic()
                    and any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds())
                ):
                    aromatic_carbonyl_adj_to_aromatic_nH[nbr.GetIdx()] = a.GetIdx()
                elif (
                    a.GetNumExplicitHs() == 0
                    and nbr.GetSymbol() == "C"
                    and nbr.GetIsAromatic()
                    and len(nbr.GetBonds()) == 3
                ):
                    aromatic_carbondeg3_adj_to_aromatic_nH0[nbr.GetIdx()] = a.GetIdx()
        else:
            a.SetNumExplicitHs(0)
    new_mol.UpdatePropertyCache()

    amap = {}
    for atom in rmol.GetAtoms():
        amap[atom_map_number2new_index[atom.GetIntProp("molAtomMapNumber")]] = (
            atom.GetIdx()
        )  # new index to old index

    # Apply the edits as predicted
    for x, y, t in edits:
        bond = new_mol.GetBondBetweenAtoms(amap[x], amap[y])
        a1 = new_mol.GetAtomWithIdx(amap[x])
        a2 = new_mol.GetAtomWithIdx(amap[y])
        if bond is not None:
            new_mol.RemoveBond(amap[x], amap[y])

            # Are we losing a bond on an aromatic nitrogen?
            if bond.GetBondTypeAsDouble() == 1.0:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 0:
                        a1.SetNumExplicitHs(1)
                    elif a1.GetFormalCharge() == 1:
                        a1.SetFormalCharge(0)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 0:
                        a2.SetNumExplicitHs(1)
                    elif a2.GetFormalCharge() == 1:
                        a2.SetFormalCharge(0)

            # Are we losing a c=O bond on an aromatic ring? If so, remove H from adjacent nH if appropriate
            if bond.GetBondTypeAsDouble() == 2.0:
                if amap[x] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(
                        aromatic_carbonyl_adj_to_aromatic_nH[amap[x]]
                    ).SetNumExplicitHs(0)
                elif amap[y] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(
                        aromatic_carbonyl_adj_to_aromatic_nH[amap[y]]
                    ).SetNumExplicitHs(0)

        if t > 0:
            new_mol.AddBond(amap[x], amap[y], BOND_TYPE[t])

            # Special alkylation case?
            if t == 1:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 1:
                        a1.SetNumExplicitHs(0)
                    else:
                        a1.SetFormalCharge(1)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 1:
                        a2.SetNumExplicitHs(0)
                    else:
                        a2.SetFormalCharge(1)

            # Are we getting a c=O bond on an aromatic ring? If so, add H to adjacent nH0 if appropriate
            if t == 2:
                if amap[x] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(
                        aromatic_carbondeg3_adj_to_aromatic_nH0[amap[x]]
                    ).SetNumExplicitHs(1)
                elif amap[y] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(
                        aromatic_carbondeg3_adj_to_aromatic_nH0[amap[y]]
                    ).SetNumExplicitHs(1)

    pred_mol = new_mol.GetMol()

    # Clear formal charges to make molecules valid
    # Note: because S and P (among others) can change valence, be more flexible
    for atom in pred_mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
        if (
            atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1
        ):  # exclude negatively-charged azide
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        elif (
            atom.GetSymbol() == "N" and atom.GetFormalCharge() == -1
        ):  # handle negatively-charged azide addition
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 3 and any(
                [nbr.GetSymbol() == "N" for nbr in atom.GetNeighbors()]
            ):
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == "N":
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if (
                bond_vals == 4 and not atom.GetIsAromatic()
            ):  # and atom.IsInRingSize(5)):
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == "C" and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
        elif atom.GetSymbol() == "O" and atom.GetFormalCharge() != 0:
            bond_vals = (
                sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                + atom.GetNumExplicitHs()
            )
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ["Cl", "Br", "I", "F"] and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 1:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == "S" and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
        elif (
            atom.GetSymbol() == "P"
        ):  # quartenary phosphorous should be pos. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) == 3 and len(bond_vals) == 3:  # make sure neutral
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == "B":  # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ["Mg", "Zn"]:
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 1 and len(bond_vals) == 1:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == "Si":
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == len(bond_vals):
                atom.SetNumExplicitHs(max(0, 4 - len(bond_vals)))

    # Bounce to/from SMILES to try to sanitize
    pred_smiles = Chem.MolToSmiles(pred_mol)
    pred_list = pred_smiles.split(".")
    pred_mols = [Chem.MolFromSmiles(pred_smiles) for pred_smiles in pred_list]

    for i, mol in enumerate(pred_mols):
        # Check if we failed/succeeded in previous step
        if mol is None:
            # print('##### Unparseable mol: {}'.format(pred_list[i]))
            continue

        # Else, try post-sanitiztion fixes in structure
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is None:
            continue
        for rxn in clean_rxns_postsani:
            out = rxn.RunReactants((mol,))
            if out:
                try:
                    Chem.SanitizeMol(out[0][0])
                    pred_mols[i] = Chem.MolFromSmiles(Chem.MolToSmiles(out[0][0]))
                except Exception as e:
                    pass
                    # print(e)
                    # print('Could not sanitize postsani reaction product: {}'.format(Chem.MolToSmiles(out[0][0])))
                    # print('Original molecule was: {}'.format(Chem.MolToSmiles(mol)))
    pred_smiles = [
        Chem.MolToSmiles(pred_mol) for pred_mol in pred_mols if pred_mol is not None
    ]

    return pred_smiles


def get_bond_changes(reaction):
    """Get the bond changes in a reaction.

    Parameters
    ----------
    reaction : str
        SMILES for a reaction, e.g. [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7]
        (=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5]
        [c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]. It consists of reactants,
        products and the atom mapping.

    Returns
    -------
    bond_changes : set of 3-tuples
        Each tuple consists of (atom1, atom2, change type)
        There are 5 possible values for change type. 0 for losing the bond, and 1, 2, 3, 1.5
        separately for forming a single, double, triple or aromatic bond.
    """
    reactants = Chem.MolFromSmiles(reaction.split(">")[0])
    products = Chem.MolFromSmiles(reaction.split(">")[2])

    conserved_maps = [
        a.GetProp("molAtomMapNumber")
        for a in products.GetAtoms()
        if a.HasProp("molAtomMapNumber")
    ]
    bond_changes = set()  # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [
                bond.GetBeginAtom().GetProp("molAtomMapNumber"),
                bond.GetEndAtom().GetProp("molAtomMapNumber"),
            ]
        )
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
            continue
        bonds_prev["{}~{}".format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [
                bond.GetBeginAtom().GetProp("molAtomMapNumber"),
                bond.GetEndAtom().GetProp("molAtomMapNumber"),
            ]
        )
        bonds_new["{}~{}".format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            # lost bond
            bond_changes.add((bond.split("~")[0], bond.split("~")[1], 0.0))
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                # changed bond
                bond_changes.add(
                    (bond.split("~")[0], bond.split("~")[1], bonds_new[bond])
                )
    for bond in bonds_new:
        if bond not in bonds_prev:
            # new bond
            bond_changes.add((bond.split("~")[0], bond.split("~")[1], bonds_new[bond]))

    return bond_changes


def edit_mol(reactant_mols, edits, product_info, mode="train", return_full_mol=False):
    """Simulate reaction via graph editing

    Parameters
    ----------
    reactant_mols : rdkit.Chem.rdchem.Mol
        RDKit molecule instances for reactants.
    edits : list of 4-tuples
        Bond changes for getting the product out of the reactants in a reaction.
        Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
        and atom2 are the end atoms to form or lose a bond, change_type is the
        type of bond change and score represents the confidence for the bond change
        by a model.
    product_info : dict
        product_info['atoms'] gives a set of atom ids in the ground truth product molecule.
    return_full_mol : bool
        keep atom-mapping and do not remove fragments not in product (use for generating products to rank)

    Returns
    -------
    str
        SMILES for the main products
    """
    ################ for switching IDXes ###########################
    old_index2atom_number = {}
    for atom in reactant_mols.GetAtoms():
        old_index2atom_number[atom.GetIdx()] = int(
            atom.GetProp("molAtomMapNumber")
        )  # Update mapping dictionary

    order = sorted(
        old_index2atom_number.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {
        old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    new_index2old_index = {
        new_idx: old_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    atom_map_number2new_index = {
        atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)
    }
    ######################################################################

    bond_change_to_type = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }

    new_mol = Chem.RWMol(reactant_mols)
    [atom.SetNumExplicitHs(0) for atom in new_mol.GetAtoms()]

    for atom1, atom2, change_type, score in edits:
        atom1, atom2 = new_index2old_index[atom1], new_index2old_index[atom2]
        bond = new_mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None:
            new_mol.RemoveBond(atom1, atom2)
        if change_type > 0:
            new_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])

    pred_mol = new_mol.GetMol()
    pred_smiles = Chem.MolToSmiles(pred_mol)

    if return_full_mol:
        return pred_smiles

    pred_list = pred_smiles.split(".")
    pred_mols = []
    for pred_smiles in pred_list:
        mol = Chem.MolFromSmiles(pred_smiles)
        if mol is None:
            continue
        atom_set = set(
            [atom_map_number2new_index[atom.GetAtomMapNum()] for atom in mol.GetAtoms()]
        )

        if mode == "train":
            if (
                len(atom_set & product_info["atoms"]) == 0
            ):  # no overlap with gold product atoms
                continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        pred_mols.append(mol)

    return ".".join(sorted([Chem.MolToSmiles(mol) for mol in pred_mols]))


def get_product_smiles(
    reactant_mols, edits, product_info, mode="train", return_full_mol=False
):
    """Get the product smiles of the reaction

    Parameters
    ----------
    reactant_mols : rdkit.Chem.rdchem.Mol
        RDKit molecule instances for reactants.
    edits : list of 4-tuples
        Bond changes for getting the product out of the reactants in a reaction.
        Each 4-tuple is of form (atom1, atom2, change_type, score), where atom1
        and atom2 are the end atoms to form or lose a bond, change_type is the
        type of bond change and score represents the confidence for the bond change
        by a model.
    product_info : dict
        product_info['atoms'] gives a set of atom ids in the ground truth product molecule.

    Returns
    -------
    str
        SMILES for the main products
    """
    smiles = edit_mol(reactant_mols, edits, product_info, mode, return_full_mol)
    if len(smiles) != 0:
        return smiles
    try:
        Chem.Kekulize(reactant_mols)
    except Exception as e:
        return smiles
    return edit_mol(reactant_mols, edits, product_info, mode, return_full_mol)


def is_connected_change_combo(combo_ids, cand_change_adj):
    """Check whether the combo of bond changes yields a connected component.

    Parameters
    ----------
    combo_ids : tuple of int
        Ids for bond changes in the combination.
    cand_change_adj : bool ndarray of shape (N, N)
        Adjacency matrix for candidate bond changes. Two candidate bond
        changes are considered adjacent if they share a common atom.
        * N for the number of candidate bond changes.

    Returns
    -------
    bool
        Whether the combo of bond changes yields a connected component
    """
    if len(combo_ids) == 1:
        return True
    multi_hop_adj = np.linalg.matrix_power(
        cand_change_adj[combo_ids, :][:, combo_ids], len(combo_ids) - 1
    )
    # The combo is connected if the distance between
    # any pair of bond changes is within len(combo) - 1

    return np.all(multi_hop_adj)


def is_valid_combo(combo_changes, reactant_info):
    """Whether the combo of bond changes is chemically valid.

    Parameters
    ----------
    combo_changes : list of 4-tuples
        Each tuple consists of atom1, atom2, type of bond change (in the form of related
        valence) and score for the change.
    reactant_info : dict
        Reaction-related information of reactants

    Returns
    -------
    bool
        Whether the combo of bond changes is chemically valid.
    """
    num_atoms = len(reactant_info["free_val"])
    force_even_parity = np.zeros((num_atoms,), dtype=bool)
    force_odd_parity = np.zeros((num_atoms,), dtype=bool)
    pair_seen = defaultdict(bool)
    free_val_tmp = reactant_info["free_val"].copy()
    for atom1, atom2, change_type, score in combo_changes:
        if pair_seen[(atom1, atom2)]:
            # A pair of atoms cannot have two types of changes. Even if we
            # randomly pick one, that will be reduced to a combo of less changes
            return False
        pair_seen[(atom1, atom2)] = True

        # Special valence rules
        atom1_type_val = atom2_type_val = change_type
        if change_type == 2:
            # to form a double bond
            if reactant_info["is_o"][atom1]:
                if reactant_info["is_c2_of_pyridine"][atom2]:
                    atom2_type_val = 1.0
                elif reactant_info["is_p"][atom2]:
                    # don't count information of =o toward valence
                    # but require odd valence parity
                    atom2_type_val = 0.0
                    force_odd_parity[atom2] = True
                elif reactant_info["is_s"][atom2]:
                    atom2_type_val = 0.0
                    force_even_parity[atom2] = True
            elif reactant_info["is_o"][atom2]:
                if reactant_info["is_c2_of_pyridine"][atom1]:
                    atom1_type_val = 1.0
                elif reactant_info["is_p"][atom1]:
                    atom1_type_val = 0.0
                    force_odd_parity[atom1] = True
                elif reactant_info["is_s"][atom1]:
                    atom1_type_val = 0.0
                    force_even_parity[atom1] = True
            elif reactant_info["is_n"][atom1] and reactant_info["is_p"][atom2]:
                atom2_type_val = 0.0
                force_odd_parity[atom2] = True
            elif reactant_info["is_n"][atom2] and reactant_info["is_p"][atom1]:
                atom1_type_val = 0.0
                force_odd_parity[atom1] = True
            elif reactant_info["is_p"][atom1] and reactant_info["is_c"][atom2]:
                atom1_type_val = 0.0
                force_odd_parity[atom1] = True
            elif reactant_info["is_p"][atom2] and reactant_info["is_c"][atom1]:
                atom2_type_val = 0.0
                force_odd_parity[atom2] = True

        reactant_pair_val = reactant_info["pair_to_bond_val"].get((atom1, atom2), None)
        if reactant_pair_val is not None:
            free_val_tmp[atom1] += reactant_pair_val - atom1_type_val
            free_val_tmp[atom2] += reactant_pair_val - atom2_type_val
        else:
            free_val_tmp[atom1] -= atom1_type_val
            free_val_tmp[atom2] -= atom2_type_val

    free_val_tmp = np.array(
        [l.item() if isinstance(l, torch.Tensor) else l for l in free_val_tmp]
    )

    # False if 1) too many connections 2) sulfur valence not even
    # 3) phosphorous valence not odd
    if (
        any(free_val_tmp < 0)
        or any(aval % 2 != 0 for aval in free_val_tmp[force_even_parity])
        or any(aval % 2 != 1 for aval in free_val_tmp[force_odd_parity])
    ):
        return False
    return True


def bookkeep_reactant(mol):  # , candidate_pairs):
    """Bookkeep reaction-related information of reactants.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for reactants.
    candidate_pairs : list of 2-tuples
        Pairs of atoms that ranked high by a model for reaction center prediction.
        By assumption, the two atoms are different and the first atom has a smaller
        index than the second.

    Returns
    -------
    info : dict
        Reaction-related information of reactants
    """
    ################ for switching IDXes ###########################
    # edits are indices, so we need to replace the atom map number with the index
    old_index2atom_number = {}
    for atom in mol.GetAtoms():
        old_index2atom_number[atom.GetIdx()] = int(
            atom.GetProp("molAtomMapNumber")
        )  # Update mapping dictionary

    order = sorted(
        old_index2atom_number.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {
        old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    # atom_map_number2new_index = {atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)}
    ######################################################################

    num_atoms = mol.GetNumAtoms()
    info = {
        # free valence of atoms
        "free_val": [0 for _ in range(num_atoms)],
        # Whether it is a carbon atom
        "is_c": [False for _ in range(num_atoms)],
        # Whether it is a carbon atom connected to a nitrogen atom in pyridine
        "is_c2_of_pyridine": [False for _ in range(num_atoms)],
        # Whether it is a phosphorous atom
        "is_p": [False for _ in range(num_atoms)],
        # Whether it is a sulfur atom
        "is_s": [False for _ in range(num_atoms)],
        # Whether it is an oxygen atom
        "is_o": [False for _ in range(num_atoms)],
        # Whether it is a nitrogen atom
        "is_n": [False for _ in range(num_atoms)],
        "pair_to_bond_val": dict(),
        "ring_bonds": set(),
    }

    # bookkeep atoms
    for j, atom in enumerate(mol.GetAtoms()):
        j = old_index2new_index[j]
        info["free_val"][j] += atom.GetTotalNumHs() + abs(atom.GetFormalCharge())
        # An aromatic carbon atom next to an aromatic nitrogen atom can get a
        # carbonyl b/c of bookkeeping of hydroxypyridines
        if atom.GetSymbol() == "C":
            info["is_c"][j] = True
            if atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "N" and nbr.GetDegree() == 2:
                        info["is_c2_of_pyridine"][j] = True
                        break
        # A nitrogen atom should be allowed to become positively charged
        elif atom.GetSymbol() == "N":
            info["free_val"][j] += 1 - atom.GetFormalCharge()
            info["is_n"][j] = True
        # Phosphorous atoms can form a phosphonium
        elif atom.GetSymbol() == "P":
            info["free_val"][j] += 1 - atom.GetFormalCharge()
            info["is_p"][j] = True
        elif atom.GetSymbol() == "O":
            info["is_o"][j] = True
        elif atom.GetSymbol() == "S":
            info["is_s"][j] = True

    # bookkeep bonds
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        atom1, atom2 = old_index2new_index[atom1], old_index2new_index[atom2]
        atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
        type_val = bond.GetBondTypeAsDouble()
        info["pair_to_bond_val"][(atom1, atom2)] = type_val
        # if (atom1, atom2) in candidate_pairs:
        #     info['free_val'][atom1] += type_val
        #     info['free_val'][atom2] += type_val
        if bond.IsInRing():
            info["ring_bonds"].add((atom1, atom2))

    return info


def bookkeep_product(mol):
    """Bookkeep reaction-related information of atoms/bonds in products

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for products.

    Returns
    -------
    info : dict
        Reaction-related information of atoms/bonds in products
    """
    ################ for switching IDXes ###########################
    old_index2atom_number = {}
    for atom in mol.GetAtoms():
        old_index2atom_number[atom.GetIdx()] = int(
            atom.GetProp("molAtomMapNumber")
        )  # Update mapping dictionary

    order = sorted(
        old_index2atom_number.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {
        old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    atom_map_number2new_index = {
        atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)
    }
    ######################################################################
    info = {"atoms": set()}
    for atom in mol.GetAtoms():
        info["atoms"].add(atom_map_number2new_index[atom.GetAtomMapNum()])

    return info


def pre_process_one_reaction(
    info, num_candidate_bond_changes, max_num_bond_changes, max_num_change_combos, mode
):
    """Pre-process one reaction for candidate ranking.

    Parameters
    ----------
    info : 4-tuple
        * candidate_bond_changes : list of tuples
            The candidate bond changes for the reaction
        * real_bond_changes : list of tuples
            The real bond changes for the reaction
        * reactant_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for reactants
        * product_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance for product
    num_candidate_bond_changes : int
        Number of candidate bond changes to consider for the ground truth reaction.
    max_num_bond_changes : int
        Maximum number of bond changes per reaction.
    max_num_change_combos : int
        Number of bond change combos to consider for each reaction.
    mode : str
        Whether the dataset is to be used for training, validation or test.

    Returns
    -------
    valid_candidate_combos : list
        valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
        of candidate bond changes for the reaction.
    candidate_bond_changes : list of 4-tuples
        Refined candidate bond changes considered for combos.
    reactant_info : dict
        Reaction-related information of reactants.
    """
    assert mode in [
        "train",
        "val",
        "test",
    ], "Expect mode to be 'train' or 'val' or 'test', got {}".format(mode)
    candidate_bond_changes_, real_bond_changes, reactant_mol, product_mol = info
    # candidate_pairs = [(atom1, atom2) for (atom1, atom2, _, _) in candidate_bond_changes_]
    reactant_info = bookkeep_reactant(reactant_mol)  # , candidate_pairs)
    if mode == "train":
        product_info = bookkeep_product(product_mol)

    # Filter out candidate new bonds already in reactants
    candidate_bond_changes = []
    count = 0
    for atom1, atom2, change_type, score in candidate_bond_changes_:
        if (
            (atom1, atom2) not in reactant_info["pair_to_bond_val"] and change_type > 0
        ) or (
            (atom1, atom2) in reactant_info["pair_to_bond_val"]
            and reactant_info["pair_to_bond_val"][(atom1, atom2)] != change_type
        ):
            candidate_bond_changes.append((atom1, atom2, change_type, score))
            count += 1
            if count == num_candidate_bond_changes:
                break

    # Check if two bond changes have atom in common
    cand_change_adj = np.eye(len(candidate_bond_changes), dtype=bool)
    for i in range(len(candidate_bond_changes)):
        atom1_1, atom1_2, _, _ = candidate_bond_changes[i]
        for j in range(i + 1, len(candidate_bond_changes)):
            atom2_1, atom2_2, _, _ = candidate_bond_changes[j]
            if (
                atom1_1 == atom2_1
                or atom1_1 == atom2_2
                or atom1_2 == atom2_1
                or atom1_2 == atom2_2
            ):
                cand_change_adj[i, j] = cand_change_adj[j, i] = True

    # Enumerate combinations of k candidate bond changes and record
    # those that are connected and chemically valid
    valid_candidate_combos = []
    cand_change_ids = range(len(candidate_bond_changes))
    for k in range(1, max_num_bond_changes + 1):
        for combo_ids in combinations(cand_change_ids, k):
            # Check if the changed bonds form a connected component
            if not is_connected_change_combo(combo_ids, cand_change_adj):
                continue
            combo_changes = [candidate_bond_changes[j] for j in combo_ids]
            # Check if the combo is chemically valid
            if is_valid_combo(combo_changes, reactant_info):
                valid_candidate_combos.append(combo_changes)

    if mode == "train":
        random.shuffle(valid_candidate_combos)
        # Index for the combo of candidate bond changes
        # that is equivalent to the gold combo
        real_combo_id = -1
        for j, combo_changes in enumerate(valid_candidate_combos):
            if set(
                [
                    (atom1, atom2, change_type)
                    for (atom1, atom2, change_type, score) in combo_changes
                ]
            ) == set(real_bond_changes):
                real_combo_id = j
                break

        # If we fail to find the real combo, make it the first entry
        if real_combo_id == -1:
            valid_candidate_combos = [
                [
                    (atom1, atom2, change_type, 0.0)
                    for (atom1, atom2, change_type) in real_bond_changes
                ]
            ] + valid_candidate_combos
        else:
            valid_candidate_combos[0], valid_candidate_combos[real_combo_id] = (
                valid_candidate_combos[real_combo_id],
                valid_candidate_combos[0],
            )

        product_smiles = get_product_smiles(
            reactant_mol, valid_candidate_combos[0], product_info, mode
        )
        if len(product_smiles) > 0:
            # Remove combos yielding duplicate products
            product_smiles = set([product_smiles])
            new_candidate_combos = [valid_candidate_combos[0]]

            count = 0
            for combo in valid_candidate_combos[1:]:
                smiles = get_product_smiles(reactant_mol, combo, product_info, mode)
                if smiles in product_smiles or len(smiles) == 0:
                    continue
                product_smiles.add(smiles)
                new_candidate_combos.append(combo)
                count += 1
                if count == max_num_change_combos:
                    break
            valid_candidate_combos = new_candidate_combos
    valid_candidate_combos = valid_candidate_combos[:max_num_change_combos]

    candidate_smiles = [
        get_product_smiles(reactant_mol, combo, None, mode="test", return_full_mol=True)
        for combo in valid_candidate_combos
    ]

    return (
        valid_candidate_combos,
        candidate_bond_changes,
        reactant_info,
        candidate_smiles,
    )


def get_batch_candidate_bonds(reaction_strings, preds, batch_ids):
    all_candidates = []
    batch_sizes = torch.bincount(batch_ids)  # Compute the number of nodes in each graph
    node_offset = 0  # Keep track of the starting node index for each graph

    for idx, reaction in enumerate(reaction_strings):
        num_nodes = batch_sizes[idx]
        graph_preds = preds[
            node_offset : node_offset + num_nodes, node_offset : node_offset + num_nodes
        ]
        candidates = tensor_to_tuples(graph_preds)
        all_candidates.append(candidates)
        node_offset += num_nodes  # Move the offset to the start of the next graph

    return all_candidates


def tensor_to_tuples(t):
    N = t.shape[0]
    K = t.shape[2]

    # Get indices of the original tensor
    indices = torch.triu_indices(N, N, offset=1).T  # upper triangular indices
    maxes, arg_maxes = torch.max(t, dim=2)  # (N, N)

    # Convert to a list of tuples
    tuples = [
        (int(row), int(col), arg_maxes[row, col].item(), maxes[row, col].item())
        for row, col in indices
    ]
    tuples = sorted(tuples, key=lambda x: x[-1], reverse=True)
    return tuples


def get_atom_pair_to_bond(mol):
    """Bookkeep bonds in the reactant.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for reactants.

    Returns
    -------
    pair_to_bond_type : dict
        Mapping 2-tuples of atoms to bond type. 1, 2, 3, 1.5 are
        separately for single, double, triple and aromatic bond.
    """

    ################ for switching IDXes ###########################
    old_index2atom_number = {}
    for atom in mol.GetAtoms():
        old_index2atom_number[atom.GetIdx()] = int(
            atom.GetProp("molAtomMapNumber")
        )  # Update mapping dictionary

    order = sorted(
        old_index2atom_number.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {
        old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    atom_map_number2new_index = {
        atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)
    }
    ######################################################################

    pair_to_bond_type = dict()
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1, atom2 = old_index2new_index[atom1], old_index2new_index[atom2]
        atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
        type_val = bond.GetBondTypeAsDouble()
        pair_to_bond_type[(atom1, atom2)] = type_val

    return pair_to_bond_type


def generate_candidates_from_scores(model_output, batch, args, mode="train"):
    """Generate candidate products from predicted changes

    Parameters
    ----------
    model_ouput : dict
        Predictions from reactivity center model.
    args : Namespace
        global arguments.
    batch : dict
        Dataset mini batch of reaction samples.
    mode : str
        Whether ground truth products are known (training) or not (inference)

    Returns
    -------
    list_of_data_batches:
        list of DataBatch of candidate products for each reaction in the batch
    valid_candidate_combos:
        list of valid candidate combinations for each reaction in the batch
    """
    num_candidate_bond_changes = args.num_candidate_bond_changes  # core size 20
    max_num_bond_changes = args.max_num_bond_changes  # combinations 5
    max_num_change_combos_per_reaction = (
        args.max_num_change_combos_per_reaction
    )  # cutoff 500

    # returns a list of list of tuples
    candidate_bond_changes = get_batch_candidate_bonds(
        batch["reaction"], model_output["s_uv"], batch["reactants"].batch
    )
    # make bonds that are "4" -> "1.5"
    for i in range(len(candidate_bond_changes)):
        candidate_bond_changes[i] = [
            (elem[0], elem[1], 1.5, elem[3]) if elem[2] == 4 else elem
            for elem in candidate_bond_changes[i]
        ]

    # Get valid candidate products, candidate bond changes considered and reactant info
    candidate_smiles = []
    valid_candidate_combos = []
    # valid_candidate_combos, candidate_bond_changes_many, reactant_info,  = [], [], []
    for i in range(len(candidate_bond_changes)):
        real_bond_changes = [
            tuple(elem) for elem in batch["reactants"]["bond_changes"][i]
        ]
        reactant_mol = Chem.MolFromSmiles(batch["reactants"]["smiles"][i])
        if mode == "train":
            product_mol = Chem.MolFromSmiles(batch["products"]["smiles"][i])
        else:
            product_mol = None

        info = (candidate_bond_changes[i], real_bond_changes, reactant_mol, product_mol)
        (
            valid_candidate_combos_one,
            candidate_bond_changes_one,
            reactant_info_one,
            candidate_smiles_one,
        ) = pre_process_one_reaction(
            info,
            num_candidate_bond_changes,
            max_num_bond_changes,
            max_num_change_combos_per_reaction,
            mode,
        )
        valid_candidate_combos.append(valid_candidate_combos_one)
        # candidate_bond_changes_many.append(candidate_bond_changes_one)
        # reactant_info.append(reactant_info_one)
        # real_bond_changes_fake_scores = [(elem[0], elem[1], elem[2], 1000) for elem in real_bond_changes]
        candidate_smiles.append(candidate_smiles_one)

    list_of_data_batches = []
    for i, list_of_candidates in enumerate(candidate_smiles):
        data_batch = []
        for j, candidate in enumerate(list_of_candidates):
            data, _ = from_mapped_smiles(
                candidate, encode_no_edge=False, sanitize=False
            )
            # fail to convert to rdkit mol
            if (data is None) or (len(data.x) == 0):
                continue
            data.candidate_bond_change = valid_candidate_combos[i][
                j
            ]  # add the bond change tuple to the Data object
            data_batch.append(data)
        if len(data_batch) == 0:
            continue
        data_batch = Batch.from_data_list(data_batch)
        data_batch.index_in_batch = i
        list_of_data_batches.append(data_batch)

    return list_of_data_batches


def sanitize_smiles_molvs(smiles, largest_fragment=False):
    """Sanitize a SMILES with MolVS

    Parameters
    ----------
    smiles : str
        SMILES string for a molecule.
    largest_fragment : bool
        Whether to select only the largest covalent unit in a molecule with
        multiple fragments. Default to False.

    Returns
    -------
    str
        SMILES string for the sanitized molecule.
    """
    standardizer = Standardizer()
    standardizer.prefer_organic = True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        mol = standardizer.standardize(mol)  # standardize functional group reps
        if largest_fragment:
            mol = standardizer.largest_fragment(
                mol
            )  # remove product counterions/salts/etc.
        mol = standardizer.uncharge(mol)  # neutralize, e.g., carboxylic acids
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def examine_topk_candidate_product(
    topks,
    topk_combos,
    reactant_mol,
    real_bond_changes,
    product_mol,
    remove_stereochemistry=False,
):
    """Perform topk evaluation for predicting the product of a reaction

    Parameters
    ----------
    topks : list of int
        Options for top-k evaluation, e.g. [1, 3, ...].
    topk_combos : list of list
        topk_combos[i] gives the combo of valid bond changes ranked i-th,
        which is a list of 3-tuples. Each tuple is of form
        (atom1, atom2, change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. The change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond or
        forming a single, double, triple, aromatic bond.
    reactant_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants.
    real_bond_changes : list of tuples
        Ground truth bond changes in a reaction. Each tuple is of form (atom1, atom2,
        change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
        a single, double, triple, and aromatic bond.
    product_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the product.
    get_smiles : bool
        Whether to get the SMILES of candidate products.

    Returns
    -------
    found_info : dict
        Binary values indicating whether we can recover the product from the ground truth
        graph edits or top-k predicted edits
    """
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(reactant_mol)
        Chem.RemoveStereochemistry(product_mol)

    found_info = defaultdict(bool)

    # Avoid corrupting the RDKit molecule instances in the dataset
    reactant_mol = deepcopy(reactant_mol)
    product_mol = deepcopy(product_mol)

    for atom in product_mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    product_smiles = Chem.MolToSmiles(product_mol)
    product_smiles_sanitized = set(
        sanitize_smiles_molvs(product_smiles, True).split(".")
    )
    product_smiles = set(product_smiles.split("."))

    ########### Use *true* edits to try to recover product
    # Generate product by modifying reactants with graph edits
    pred_smiles = robust_edit_mol(reactant_mol, real_bond_changes)
    pred_smiles_sanitized = set(sanitize_smiles_molvs(smiles) for smiles in pred_smiles)
    pred_smiles = set(pred_smiles)

    if not product_smiles <= pred_smiles:
        # Try again with kekulized form
        Chem.Kekulize(reactant_mol)
        pred_smiles_kek = robust_edit_mol(reactant_mol, real_bond_changes)
        pred_smiles_kek = set(pred_smiles_kek)
        if not product_smiles <= pred_smiles_kek:
            if product_smiles_sanitized <= pred_smiles_sanitized:
                # print('\nwarn: mismatch, but only due to standardization')
                found_info["ground_sanitized"] = True
            else:
                pass
                # print('\nwarn: could not regenerate product {}'.format(product_smiles))
                # print('sani product: {}'.format(product_smiles_sanitized))
                # print(Chem.MolToSmiles(reactant_mol))
                # print(Chem.MolToSmiles(product_mol))
                # print(real_bond_changes)
                # print('pred_smiles: {}'.format(pred_smiles))
                # print('pred_smiles_kek: {}'.format(pred_smiles_kek))
                # print('pred_smiles_sani: {}'.format(pred_smiles_sanitized))
        else:
            found_info["ground"] = True
            found_info["ground_sanitized"] = True
    else:
        found_info["ground"] = True
        found_info["ground_sanitized"] = True

    ########### Now use candidate edits to try to recover product
    max_topk = max(topks)
    current_rank = 0
    correct_rank = max_topk + 1
    sanitized_correct_rank = max_topk + 1
    candidate_smiles_list = []
    candidate_smiles_sanitized_list = []

    for i, combo in enumerate(topk_combos):
        prev_len_candidate_smiles = len(set(candidate_smiles_list))

        # Generate products by modifying reactants with predicted edits.
        candidate_smiles = robust_edit_mol(reactant_mol, combo)
        candidate_smiles = set(candidate_smiles)
        candidate_smiles_sanitized = set(
            sanitize_smiles_molvs(smiles) for smiles in candidate_smiles
        )

        if product_smiles_sanitized <= candidate_smiles_sanitized:
            sanitized_correct_rank = min(sanitized_correct_rank, current_rank + 1)
        if product_smiles <= candidate_smiles:
            correct_rank = min(correct_rank, current_rank + 1)

        # Record unkekulized form
        candidate_smiles_list.append(".".join(candidate_smiles))
        candidate_smiles_sanitized_list.append(".".join(candidate_smiles_sanitized))

        # Edit molecules with reactants kekulized. Sometimes previous editing fails due to
        # RDKit sanitization error (edited molecule cannot be kekulized)
        try:
            Chem.Kekulize(reactant_mol)
        except Exception as e:
            pass

        candidate_smiles = robust_edit_mol(reactant_mol, combo)
        candidate_smiles = set(candidate_smiles)
        candidate_smiles_sanitized = set(
            sanitize_smiles_molvs(smiles) for smiles in candidate_smiles
        )
        if product_smiles_sanitized <= candidate_smiles_sanitized:
            sanitized_correct_rank = min(sanitized_correct_rank, current_rank + 1)
        if product_smiles <= candidate_smiles:
            correct_rank = min(correct_rank, current_rank + 1)

        # If we failed to come up with a new candidate, don't increment the counter!
        if len(set(candidate_smiles_list)) > prev_len_candidate_smiles:
            current_rank += 1

        if correct_rank < max_topk + 1 and sanitized_correct_rank < max_topk + 1:
            break

    for k in topks:
        if correct_rank <= k:
            found_info["top_{:d}".format(k)] = True
        if sanitized_correct_rank <= k:
            found_info["top_{:d}_sanitized".format(k)] = True

    return found_info
