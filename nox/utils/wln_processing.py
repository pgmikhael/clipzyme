
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
from rdkit.Chem import rdmolops
from tqdm import tqdm
from torch_geometric.data import Batch
from nox.utils.pyg import from_smiles, from_mapped_smiles

from functools import partial
import torch

from sklearn.neighbors import NearestNeighbors

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass




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
    reactants = Chem.MolFromSmiles(reaction.split('>')[0])
    products  = Chem.MolFromSmiles(reaction.split('>')[2])

    conserved_maps = [
        a.GetProp('molAtomMapNumber')
        for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
            continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            # lost bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                # changed bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
    for bond in bonds_new:
        if bond not in bonds_prev:
            # new bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))

    return bond_changes
    

def edit_mol(reactant_mols, edits, product_info, mode="train"):
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

    Returns
    -------
    str
        SMILES for the main products
    """
    bond_change_to_type = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                           3: Chem.rdchem.BondType.TRIPLE, 1.5: Chem.rdchem.BondType.AROMATIC}

    new_mol = Chem.RWMol(reactant_mols)
    [atom.SetNumExplicitHs(0) for atom in new_mol.GetAtoms()]

    for atom1, atom2, change_type, score in edits:
        bond = new_mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None:
            new_mol.RemoveBond(atom1, atom2)
        if change_type > 0:
            new_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])

    pred_mol = new_mol.GetMol()
    pred_smiles = Chem.MolToSmiles(pred_mol)
    pred_list = pred_smiles.split('.')
    pred_mols = []
    for pred_smiles in pred_list:
        mol = Chem.MolFromSmiles(pred_smiles)
        if mol is None:
            continue
        atom_set = set([atom.GetAtomMapNum() - 1 for atom in mol.GetAtoms()])
        if mode == "train":
            if len(atom_set & product_info['atoms']) == 0: # no overlap with gold product atoms
                continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        pred_mols.append(mol)

    return '.'.join(sorted([Chem.MolToSmiles(mol) for mol in pred_mols]))


def get_product_smiles(reactant_mols, edits, product_info, mode="train"):
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
    smiles = edit_mol(reactant_mols, edits, product_info, mode)
    if len(smiles) != 0:
        return smiles
    try:
        Chem.Kekulize(reactant_mols)
    except Exception as e:
        return smiles
    return edit_mol(reactant_mols, edits, product_info, mode)


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
        cand_change_adj[combo_ids, :][:, combo_ids], len(combo_ids) - 1)
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
    num_atoms = len(reactant_info['free_val'])
    force_even_parity = np.zeros((num_atoms,), dtype=bool)
    force_odd_parity = np.zeros((num_atoms,), dtype=bool)
    pair_seen = defaultdict(bool)
    free_val_tmp = reactant_info['free_val'].copy()
    for (atom1, atom2, change_type, score) in combo_changes:
        if pair_seen[(atom1, atom2)]:
            # A pair of atoms cannot have two types of changes. Even if we
            # randomly pick one, that will be reduced to a combo of less changes
            return False
        pair_seen[(atom1, atom2)] = True

        # Special valence rules
        atom1_type_val = atom2_type_val = change_type
        if change_type == 2:
            # to form a double bond
            if reactant_info['is_o'][atom1]:
                if reactant_info['is_c2_of_pyridine'][atom2]:
                    atom2_type_val = 1.
                elif reactant_info['is_p'][atom2]:
                    # don't count information of =o toward valence
                    # but require odd valence parity
                    atom2_type_val = 0.
                    force_odd_parity[atom2] = True
                elif reactant_info['is_s'][atom2]:
                    atom2_type_val = 0.
                    force_even_parity[atom2] = True
            elif reactant_info['is_o'][atom2]:
                if reactant_info['is_c2_of_pyridine'][atom1]:
                    atom1_type_val = 1.
                elif reactant_info['is_p'][atom1]:
                    atom1_type_val = 0.
                    force_odd_parity[atom1] = True
                elif reactant_info['is_s'][atom1]:
                    atom1_type_val = 0.
                    force_even_parity[atom1] = True
            elif reactant_info['is_n'][atom1] and reactant_info['is_p'][atom2]:
                atom2_type_val = 0.
                force_odd_parity[atom2] = True
            elif reactant_info['is_n'][atom2] and reactant_info['is_p'][atom1]:
                atom1_type_val = 0.
                force_odd_parity[atom1] = True
            elif reactant_info['is_p'][atom1] and reactant_info['is_c'][atom2]:
                atom1_type_val = 0.
                force_odd_parity[atom1] = True
            elif reactant_info['is_p'][atom2] and reactant_info['is_c'][atom1]:
                atom2_type_val = 0.
                force_odd_parity[atom2] = True

        reactant_pair_val = reactant_info['pair_to_bond_val'].get((atom1, atom2), None)
        if reactant_pair_val is not None:
            free_val_tmp[atom1] += reactant_pair_val - atom1_type_val
            free_val_tmp[atom2] += reactant_pair_val - atom2_type_val
        else:
            free_val_tmp[atom1] -= atom1_type_val
            free_val_tmp[atom2] -= atom2_type_val

    free_val_tmp = np.array(free_val_tmp)
    # False if 1) too many connections 2) sulfur valence not even
    # 3) phosphorous valence not odd
    if any(free_val_tmp < 0) or \
            any(aval % 2 != 0 for aval in free_val_tmp[force_even_parity]) or \
            any(aval % 2 != 1 for aval in free_val_tmp[force_odd_parity]):
        return False
    return True

def bookkeep_reactant(mol, candidate_pairs):
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
    num_atoms = mol.GetNumAtoms()
    info = {
        # free valence of atoms
        'free_val': [0 for _ in range(num_atoms)],
        # Whether it is a carbon atom
        'is_c': [False for _ in range(num_atoms)],
        # Whether it is a carbon atom connected to a nitrogen atom in pyridine
        'is_c2_of_pyridine': [False for _ in range(num_atoms)],
        # Whether it is a phosphorous atom
        'is_p': [False for _ in range(num_atoms)],
        # Whether it is a sulfur atom
        'is_s': [False for _ in range(num_atoms)],
        # Whether it is an oxygen atom
        'is_o': [False for _ in range(num_atoms)],
        # Whether it is a nitrogen atom
        'is_n': [False for _ in range(num_atoms)],
        'pair_to_bond_val': dict(),
        'ring_bonds': set()
    }

    # bookkeep atoms
    for j, atom in enumerate(mol.GetAtoms()):
        info['free_val'][j] += atom.GetTotalNumHs() + abs(atom.GetFormalCharge())
        # An aromatic carbon atom next to an aromatic nitrogen atom can get a
        # carbonyl b/c of bookkeeping of hydroxypyridines
        if atom.GetSymbol() == 'C':
            info['is_c'][j] = True
            if atom.GetIsAromatic():
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'N' and nbr.GetDegree() == 2:
                        info['is_c2_of_pyridine'][j] = True
                        break
        # A nitrogen atom should be allowed to become positively charged
        elif atom.GetSymbol() == 'N':
            info['free_val'][j] += 1 - atom.GetFormalCharge()
            info['is_n'][j] = True
        # Phosphorous atoms can form a phosphonium
        elif atom.GetSymbol() == 'P':
            info['free_val'][j] += 1 - atom.GetFormalCharge()
            info['is_p'][j] = True
        elif atom.GetSymbol() == 'O':
            info['is_o'][j] = True
        elif atom.GetSymbol() == 'S':
            info['is_s'][j] = True

    # bookkeep bonds
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1, atom2 = min(atom1, atom2), max(atom1, atom2)
        type_val = bond.GetBondTypeAsDouble()
        info['pair_to_bond_val'][(atom1, atom2)] = type_val
        if (atom1, atom2) in candidate_pairs:
            info['free_val'][atom1] += type_val
            info['free_val'][atom2] += type_val
        if bond.IsInRing():
            info['ring_bonds'].add((atom1, atom2))

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
    info = {
        'atoms': set()
    }
    for atom in mol.GetAtoms():
        info['atoms'].add(atom.GetAtomMapNum() - 1)

    return info


def pre_process_one_reaction(info, num_candidate_bond_changes, max_num_bond_changes,
                             max_num_change_combos, mode):
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
    assert mode in ['train', 'val', 'test'], \
        "Expect mode to be 'train' or 'val' or 'test', got {}".format(mode)
    candidate_bond_changes_, real_bond_changes, reactant_mol, product_mol = info
    candidate_pairs = [(atom1, atom2) for (atom1, atom2, _, _)
                       in candidate_bond_changes_]
    reactant_info = bookkeep_reactant(reactant_mol, candidate_pairs)
    if mode == 'train':
        product_info = bookkeep_product(product_mol)

    # Filter out candidate new bonds already in reactants
    candidate_bond_changes = []
    count = 0
    for (atom1, atom2, change_type, score) in candidate_bond_changes_:
        if ((atom1, atom2) not in reactant_info['pair_to_bond_val']) or (reactant_info['pair_to_bond_val'][(atom1, atom2)] != change_type):
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
            if atom1_1 == atom2_1 or atom1_1 == atom2_2 or \
                    atom1_2 == atom2_1 or atom1_2 == atom2_2:
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

    if mode == 'train':
        random.shuffle(valid_candidate_combos)
        # Index for the combo of candidate bond changes
        # that is equivalent to the gold combo
        real_combo_id = -1
        for j, combo_changes in enumerate(valid_candidate_combos):
            if set([(atom1, atom2, change_type) for
                    (atom1, atom2, change_type, score) in combo_changes]) == \
                    set(real_bond_changes):
                real_combo_id = j
                break

        # If we fail to find the real combo, make it the first entry
        if real_combo_id == -1:
            valid_candidate_combos = \
                [[(atom1, atom2, change_type, 0.0)
                  for (atom1, atom2, change_type) in real_bond_changes]] + \
                valid_candidate_combos
        else:
            valid_candidate_combos[0], valid_candidate_combos[real_combo_id] = \
                valid_candidate_combos[real_combo_id], valid_candidate_combos[0]

        product_smiles = get_product_smiles(
            reactant_mol, valid_candidate_combos[0], product_info, mode)
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

    candidate_smiles = [get_product_smiles(reactant_mol, combo, None, mode) for combo in valid_candidate_combos]

    return valid_candidate_combos, candidate_bond_changes, reactant_info, candidate_smiles

def get_candidate_bonds(reaction, preds, num_nodes, max_k, easy, include_scores=False):
    """Get candidate bonds for a reaction.

    Parameters
    ----------
    reaction : str
        Reaction
    preds : float32 tensor of shape (E * 5)
        E for the number of edges in a complete graph and 5 for the number of possible
        bond changes.
    num_nodes : int
        Number of nodes in the graph.
    max_k : int
        Maximum number of atom pairs to be selected.
    easy : bool
        If True, reactants not contributing atoms to the product will be excluded in
        top-k atom pair selection, which will make the task easier.
    include_scores : bool
        Whether to include the scores for the atom pairs selected. Default to False.

    Returns
    -------
    list of 3-tuples or 4-tuples
        The first three elements in a tuple separately specify the first atom,
        the second atom and the type for bond change. If include_scores is True,
        the score for the prediction will be included as a fourth element.
    """
    # Decide which atom-pairs will be considered.
    reaction_atoms = []
    reaction_bonds = defaultdict(bool)
    reactants, _, product = reaction.split('>')
    product_mol = Chem.MolFromSmiles(product)
    product_atoms = set([atom.GetAtomMapNum() for atom in product_mol.GetAtoms()])

    for reactant in reactants.split('.'):
        reactant_mol = Chem.MolFromSmiles(reactant)
        reactant_atoms = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms()]
        # In the hard mode, all reactant atoms will be included.
        # In the easy mode, only reactants contributing atoms to the product will be included.
        if (len(set(reactant_atoms) & product_atoms) > 0) or (not easy):
            reaction_atoms.extend(reactant_atoms)
            for bond in reactant_mol.GetBonds():
                end_atoms = sorted([bond.GetBeginAtom().GetAtomMapNum(),
                                    bond.GetEndAtom().GetAtomMapNum()])
                bond = tuple(end_atoms + [bond.GetBondTypeAsDouble()])
                # Bookkeep bonds already in reactants
                reaction_bonds[bond] = True

    candidate_bonds = []
    if len(preds)<max_k:
        max_k = len(preds)
    topk_values, topk_indices = torch.topk(preds, max_k)
    for j in range(max_k):
        preds_j = topk_indices[j].cpu().item()
        # A bond change can be either losing the bond or forming a
        # single, double, triple or aromatic bond
        change_id = preds_j % num_change_types
        change_type = id_to_bond_change[change_id]
        pair_id = preds_j // num_change_types
        # Atom map numbers
        atom1 = pair_id // num_nodes + 1
        atom2 = pair_id % num_nodes + 1
        # Avoid duplicates and an atom cannot form a bond with itself
        if atom1 >= atom2:
            continue
        if atom1 not in reaction_atoms:
            continue
        if atom2 not in reaction_atoms:
            continue
        candidate = (int(atom1), int(atom2), float(change_type))
        if reaction_bonds[candidate]:
            continue
        if include_scores:
            candidate += (float(topk_values[j].cpu().item()),)
        candidate_bonds.append(candidate)

    return candidate_bonds


def get_batch_candidate_bonds(reaction_strings, preds, batch_ids):
    all_candidates = []
    batch_sizes = torch.bincount(batch_ids)  # Compute the number of nodes in each graph
    node_offset = 0  # Keep track of the starting node index for each graph

    for idx, reaction in enumerate(reaction_strings):
        num_nodes = batch_sizes[idx]
        graph_preds = preds[node_offset : node_offset + num_nodes, node_offset : node_offset + num_nodes]
        candidates = tensor_to_tuples(graph_preds)
        all_candidates.append(candidates)
        node_offset += num_nodes  # Move the offset to the start of the next graph

    return all_candidates


def tensor_to_tuples(t):
    N = t.shape[0]
    K = t.shape[2]
    
    # Get indices of the original tensor
    indices = torch.triu_indices(N, N, offset=1) # upper triangular indices
    indices = torch.stack(indices).reshape(2, -1).T  # transpose and reshape

    maxes, arg_maxes = torch.max(t, dim=2) # (N, N)

    # Convert to a list of tuples
    tuples = [(int(row), int(column), arg_maxes[row, col], maxes[row, col]) for row, column in indices]

    return tuples


def generate_candidates_from_scores(model_output, batch, args, mode = "train"):
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
    list of B + 1 DGLGraph
        The first entry in the list is the DGLGraph for the reactants and the rest are
        DGLGraphs for candidate products. Each DGLGraph has edge features in edata['he'] and
        node features in ndata['hv'].
    candidate_scores : float32 tensor of shape (B, 1)
        The sum of scores for bond changes in each combo, where B is the number of combos.
    labels : int64 tensor of shape (1, 1), optional
        Index for the true candidate product, which is always 0 with pre-processing. This is
        returned only when we are not in the training mode.
    valid_candidate_combos : list, optional
        valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
        of candidate bond changes for the reaction. Each tuple is of form (atom1, atom2,
        change_type, score). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
        a single, double, triple, and aromatic bond.
    reactant_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the reactants
    real_bond_changes : list of tuples
        Ground truth bond changes in a reaction. Each tuple is of form (atom1, atom2,
        change_type). atom1, atom2 are the atom mapping numbers - 1 of the two
        end atoms. change_type can be 0, 1, 2, 3, 1.5, separately for losing a bond, forming
        a single, double, triple, and aromatic bond.
    product_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the product
    """    
    num_candidate_bond_changes = args.num_candidate_bond_changes # core size 20
    max_num_bond_changes = args.max_num_bond_changes # combinations 5
    max_num_change_combos_per_reaction = args.max_num_change_combos_per_reaction # cutoff 500
    
    candidate_bond_changes = get_batch_candidate_bonds(batch["reaction"], model_output['s_uv'], batch['reactants'].batch)
    
    real_bond_changes = batch['bond_changes']
    reactant_mol = batch['reactants']
    if mode == 'train':
        product_mol = batch['products']
    else:
        product_mol = None

    # Get valid candidate products, candidate bond changes considered and reactant info
    info = (candidate_bond_changes, real_bond_changes, reactant_mol, product_mol)
    valid_candidate_combos, candidate_bond_changes, reactant_info, candidate_smiles = \
        pre_process_one_reaction(info, num_candidate_bond_changes, max_num_bond_changes, max_num_change_combos_per_reaction, mode)

    
    # TODO: from here replace with pyg data (run from_smiles on all mols, will also featurize)
    list_of_data_batches = []
    for list_of_candidates in candidate_smiles:
        data_batch = []
        for candidate in list_of_candidates:
            data, _ = from_mapped_smiles(candidate)
            data_batch.append(data)
        list_of_data_batches.append(Batch.from_data_list(data_batch))
