
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
from nox.utils.pyg import from_smiles

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
    

# def one_hot_encoding(x, allowable_set, encode_unknown=False):
#     """One-hot encoding.

#     Parameters
#     ----------
#     x
#         Value to encode.
#     allowable_set : list
#         The elements of the allowable_set should be of the
#         same type as x.
#     encode_unknown : bool
#         If True, map inputs not in the allowable set to the
#         additional last element.

#     Returns
#     -------
#     list
#         List of boolean values where at most one value is True.
#         The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
#         and ``len(allowable_set) + 1`` otherwise.

#     Examples
#     --------
#     >>> from dgllife.utils import one_hot_encoding
#     >>> one_hot_encoding('C', ['C', 'O'])
#     [True, False]
#     >>> one_hot_encoding('S', ['C', 'O'])
#     [False, False]
#     >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
#     [False, False, True]
#     """
#     if encode_unknown and (allowable_set[-1] is not None):
#         allowable_set.append(None)

#     if encode_unknown and (x not in allowable_set):
#         x = None

#     return list(map(lambda s: x == s, allowable_set))

# # pylint: disable=I1101
# def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
#                  canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
#     """Convert an RDKit molecule object into a DGLGraph and featurize for it.

#     This function can be used to construct any arbitrary ``DGLGraph`` from an
#     RDKit molecule instance.

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     graph_constructor : callable
#         Takes an RDKit molecule as input and returns a DGLGraph
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to
#         update ndata for a DGLGraph.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to
#         update edata for a DGLGraph.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). If False, it will do nothing.
#         Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Converted DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.

#     See Also
#     --------
#     mol_to_bigraph
#     mol_to_complete_graph
#     mol_to_nearest_neighbor_graph
#     """
#     if mol is None:
#         print('Invalid mol found')
#         return None

#     # Whether to have hydrogen atoms as explicit nodes
#     if explicit_hydrogens:
#         mol = Chem.AddHs(mol)

#     if canonical_atom_order:
#         new_order = rdmolfiles.CanonicalRankAtoms(mol)
#         mol = rdmolops.RenumberAtoms(mol, new_order)
#     g = graph_constructor(mol)

#     if node_featurizer is not None:
#         g.ndata.update(node_featurizer(mol))

#     if edge_featurizer is not None:
#         g.edata.update(edge_featurizer(mol))

#     if num_virtual_nodes > 0:
#         num_real_nodes = g.num_nodes()
#         real_nodes = list(range(num_real_nodes))
#         g.add_nodes(num_virtual_nodes)

#         # Change Topology
#         virtual_src = []
#         virtual_dst = []
#         for count in range(num_virtual_nodes):
#             virtual_node = num_real_nodes + count
#             virtual_node_copy = [virtual_node] * num_real_nodes
#             virtual_src.extend(real_nodes)
#             virtual_src.extend(virtual_node_copy)
#             virtual_dst.extend(virtual_node_copy)
#             virtual_dst.extend(real_nodes)
#         g.add_edges(virtual_src, virtual_dst)

#         for nk, nv in g.ndata.items():
#             nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
#             nv[-num_virtual_nodes:, -1] = 1
#             g.ndata[nk] = nv

#         for ek, ev in g.edata.items():
#             ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
#             ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
#             g.edata[ek] = ev

#     return g

# def construct_bigraph_from_mol(mol, add_self_loop=False):
#     """Construct a bi-directed DGLGraph with topology only for the molecule.

#     The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
#     **i** th node in the returned DGLGraph.

#     The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
#     **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
#     **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
#     **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

#     If self loops are added, the last **n** edges will separately be self loops for
#     atoms ``0, 1, ..., n-1``.

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.

#     Returns
#     -------
#     g : DGLGraph
#         Empty bigraph topology of the molecule
#     """
#     g = dgl.graph(([], []), idtype=torch.int32)

#     # Add nodes
#     num_atoms = mol.GetNumAtoms()
#     g.add_nodes(num_atoms)

#     # Add edges
#     src_list = []
#     dst_list = []
#     num_bonds = mol.GetNumBonds()
#     for i in range(num_bonds):
#         bond = mol.GetBondWithIdx(i)
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#         src_list.extend([u, v])
#         dst_list.extend([v, u])

#     if add_self_loop:
#         nodes = g.nodes().tolist()
#         src_list.extend(nodes)
#         dst_list.extend(nodes)

#     g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

#     return g

# def mol_to_bigraph(mol, add_self_loop=False,
#                    node_featurizer=None,
#                    edge_featurizer=None,
#                    canonical_atom_order=True,
#                    explicit_hydrogens=False,
#                    num_virtual_nodes=0):
#     """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Bi-directed DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.

#     Examples
#     --------
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import mol_to_bigraph

#     >>> mol = Chem.MolFromSmiles('CCO')
#     >>> g = mol_to_bigraph(mol)
#     >>> print(g)
#     DGLGraph(num_nodes=3, num_edges=4,
#              ndata_schemes={}
#              edata_schemes={})

#     We can also initialize node/edge features when constructing graphs.

#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import mol_to_bigraph

#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> def featurize_bonds(mol):
#     >>>     feats = []
#     >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#     >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
#     >>>     for bond in mol.GetBonds():
#     >>>         btype = bond_types.index(bond.GetBondType())
#     >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
#     >>>         feats.extend([btype, btype])
#     >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> mol = Chem.MolFromSmiles('CCO')
#     >>> g = mol_to_bigraph(mol, node_featurizer=featurize_atoms,
#     >>>                    edge_featurizer=featurize_bonds)
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['type'])
#     tensor([[0.],
#             [0.],
#             [0.],
#             [0.]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> g = mol_to_bigraph(mol, explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=9, num_edges=16,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     smiles_to_bigraph
#     """
#     return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
#                         node_featurizer, edge_featurizer,
#                         canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

# def smiles_to_bigraph(smiles, add_self_loop=False,
#                       node_featurizer=None,
#                       edge_featurizer=None,
#                       canonical_atom_order=True,
#                       explicit_hydrogens=False,
#                       num_virtual_nodes=0):
#     """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

#     Parameters
#     ----------
#     smiles : str
#         String of SMILES
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Bi-directed DGLGraph for the molecule if :attr:`smiles` is valid and None otherwise.

#     Examples
#     --------
#     >>> from dgllife.utils import smiles_to_bigraph

#     >>> g = smiles_to_bigraph('CCO')
#     >>> print(g)
#     DGLGraph(num_nodes=3, num_edges=4,
#              ndata_schemes={}
#              edata_schemes={})

#     We can also initialize node/edge features when constructing graphs.

#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import smiles_to_bigraph

#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> def featurize_bonds(mol):
#     >>>     feats = []
#     >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#     >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
#     >>>     for bond in mol.GetBonds():
#     >>>         btype = bond_types.index(bond.GetBondType())
#     >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
#     >>>         feats.extend([btype, btype])
#     >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> g = smiles_to_bigraph('CCO', node_featurizer=featurize_atoms,
#     >>>                       edge_featurizer=featurize_bonds)
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['type'])
#     tensor([[0.],
#             [0.],
#             [0.],
#             [0.]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> g = smiles_to_bigraph('CCO', explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=9, num_edges=16,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     mol_to_bigraph
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     return mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer,
#                           canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

# def construct_complete_graph_from_mol(mol, add_self_loop=False):
#     """Construct a complete graph with topology only for the molecule

#     The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
#     **i** th node in the returned DGLGraph.

#     The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
#     If self loops are not created, we will not have (0, 0), (1, 1), ...

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.

#     Returns
#     -------
#     g : DGLGraph
#         Empty complete graph topology of the molecule
#     """
#     num_atoms = mol.GetNumAtoms()
#     src = []
#     dst = []
#     for i in range(num_atoms):
#         for j in range(num_atoms):
#             if i != j or add_self_loop:
#                 src.append(i)
#                 dst.append(j)
#     g = dgl.graph((torch.IntTensor(src), torch.IntTensor(dst)), idtype=torch.int32)

#     return g

# def mol_to_complete_graph(mol, add_self_loop=False,
#                           node_featurizer=None,
#                           edge_featurizer=None,
#                           canonical_atom_order=True,
#                           explicit_hydrogens=False,
#                           num_virtual_nodes=0):
#     """Convert an RDKit molecule into a complete DGLGraph and featurize for it.

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Complete DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.

#     Examples
#     --------
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import mol_to_complete_graph

#     >>> mol = Chem.MolFromSmiles('CCO')
#     >>> g = mol_to_complete_graph(mol)
#     >>> print(g)
#     DGLGraph(num_nodes=3, num_edges=6,
#              ndata_schemes={}
#              edata_schemes={})

#     We can also initialize node/edge features when constructing graphs.

#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import mol_to_complete_graph
#     >>> from functools import partial

#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> def featurize_edges(mol, add_self_loop=False):
#     >>>     feats = []
#     >>>     num_atoms = mol.GetNumAtoms()
#     >>>     atoms = list(mol.GetAtoms())
#     >>>     distance_matrix = Chem.GetDistanceMatrix(mol)
#     >>>     for i in range(num_atoms):
#     >>>         for j in range(num_atoms):
#     >>>             if i != j or add_self_loop:
#     >>>                 feats.append(float(distance_matrix[i, j]))
#     >>>     return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> mol = Chem.MolFromSmiles('CCO')
#     >>> add_self_loop = True
#     >>> g = mol_to_complete_graph(
#     >>>         mol, add_self_loop=add_self_loop, node_featurizer=featurize_atoms,
#     >>>         edge_featurizer=partial(featurize_edges, add_self_loop=add_self_loop))
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['dist'])
#     tensor([[0.],
#             [2.],
#             [1.],
#             [2.],
#             [0.],
#             [1.],
#             [1.],
#             [1.],
#             [0.]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> g = mol_to_complete_graph(mol, explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=9, num_edges=72,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     smiles_to_complete_graph
#     """
#     return mol_to_graph(mol,
#                         partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
#                         node_featurizer, edge_featurizer,
#                         canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

# def smiles_to_complete_graph(smiles, add_self_loop=False,
#                              node_featurizer=None,
#                              edge_featurizer=None,
#                              canonical_atom_order=True,
#                              explicit_hydrogens=False,
#                              num_virtual_nodes=0):
#     """Convert a SMILES into a complete DGLGraph and featurize for it.

#     Parameters
#     ----------
#     smiles : str
#         String of SMILES
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Complete DGLGraph for the molecule if :attr:`smiles` is valid and None otherwise.

#     Examples
#     --------
#     >>> from dgllife.utils import smiles_to_complete_graph

#     >>> g = smiles_to_complete_graph('CCO')
#     >>> print(g)
#     DGLGraph(num_nodes=3, num_edges=6,
#              ndata_schemes={}
#              edata_schemes={})

#     We can also initialize node/edge features when constructing graphs.

#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import smiles_to_complete_graph
#     >>> from functools import partial

#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> def featurize_edges(mol, add_self_loop=False):
#     >>>     feats = []
#     >>>     num_atoms = mol.GetNumAtoms()
#     >>>     atoms = list(mol.GetAtoms())
#     >>>     distance_matrix = Chem.GetDistanceMatrix(mol)
#     >>>     for i in range(num_atoms):
#     >>>         for j in range(num_atoms):
#     >>>             if i != j or add_self_loop:
#     >>>                 feats.append(float(distance_matrix[i, j]))
#     >>>     return {'dist': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> add_self_loop = True
#     >>> g = smiles_to_complete_graph(
#     >>>         'CCO', add_self_loop=add_self_loop, node_featurizer=featurize_atoms,
#     >>>         edge_featurizer=partial(featurize_edges, add_self_loop=add_self_loop))
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['dist'])
#     tensor([[0.],
#             [2.],
#             [1.],
#             [2.],
#             [0.],
#             [1.],
#             [1.],
#             [1.],
#             [0.]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> g = smiles_to_complete_graph('CCO', explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=9, num_edges=72,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     mol_to_complete_graph
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     return mol_to_complete_graph(mol, add_self_loop, node_featurizer, edge_featurizer,
#                                  canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

# def k_nearest_neighbors(coordinates, neighbor_cutoff, max_num_neighbors=None,
#                         p_distance=2, self_loops=False):
#     """Find k nearest neighbors for each atom

#     We do not guarantee that the edges are sorted according to the distance
#     between atoms.

#     Parameters
#     ----------
#     coordinates : numpy.ndarray of shape (N, D)
#         The coordinates of atoms in the molecule. N for the number of atoms
#         and D for the dimensions of the coordinates.
#     neighbor_cutoff : float
#         If the distance between a pair of nodes is larger than neighbor_cutoff,
#         they will not be considered as neighboring nodes.
#     max_num_neighbors : int or None.
#         If not None, then this specifies the maximum number of neighbors
#         allowed for each atom. Default to None.
#     p_distance : int
#         We compute the distance between neighbors using Minkowski (:math:`l_p`)
#         distance. When ``p_distance = 1``, Minkowski distance is equivalent to
#         Manhattan distance. When ``p_distance = 2``, Minkowski distance is
#         equivalent to the standard Euclidean distance. Default to 2.
#     self_loops : bool
#         Whether to allow a node to be its own neighbor. Default to False.

#     Returns
#     -------
#     srcs : list of int
#         Source nodes.
#     dsts : list of int
#         Destination nodes, corresponding to ``srcs``.
#     distances : list of float
#         Distances between the end nodes, corresponding to ``srcs`` and ``dsts``.

#     Examples
#     --------
#     >>> from dgllife.utils import get_mol_3d_coordinates, k_nearest_neighbors
#     >>> from rdkit import Chem
#     >>> from rdkit.Chem import AllChem

#     >>> mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C')
#     >>> AllChem.EmbedMolecule(mol)
#     >>> AllChem.MMFFOptimizeMolecule(mol)
#     >>> coords = get_mol_3d_coordinates(mol)
#     >>> srcs, dsts, dists = k_nearest_neighbors(coords, neighbor_cutoff=1.25)
#     >>> print(srcs)
#     [8, 7, 11, 10, 20, 19]
#     >>> print(dsts)
#     [7, 8, 10, 11, 19, 20]
#     >>> print(dists)
#     [1.2084666104583117, 1.2084666104583117, 1.226457824344217,
#      1.226457824344217, 1.2230522248065987, 1.2230522248065987]

#     See Also
#     --------
#     get_mol_3d_coordinates
#     mol_to_nearest_neighbor_graph
#     smiles_to_nearest_neighbor_graph
#     """
#     num_atoms = coordinates.shape[0]
#     model = NearestNeighbors(radius=neighbor_cutoff, p=p_distance)
#     model.fit(coordinates)
#     dists_, nbrs = model.radius_neighbors(coordinates)
#     srcs, dsts, dists = [], [], []
#     for i in range(num_atoms):
#         dists_i = dists_[i].tolist()
#         nbrs_i = nbrs[i].tolist()
#         if not self_loops:
#             dists_i.remove(0)
#             nbrs_i.remove(i)
#         if max_num_neighbors is not None and len(nbrs_i) > max_num_neighbors:
#             packed_nbrs = list(zip(dists_i, nbrs_i))
#             # Sort neighbors based on distance from smallest to largest
#             packed_nbrs.sort(key=lambda tup: tup[0])
#             dists_i, nbrs_i = map(list, zip(*packed_nbrs))
#             dsts.extend([i for _ in range(max_num_neighbors)])
#             srcs.extend(nbrs_i[:max_num_neighbors])
#             dists.extend(dists_i[:max_num_neighbors])
#         else:
#             dsts.extend([i for _ in range(len(nbrs_i))])
#             srcs.extend(nbrs_i)
#             dists.extend(dists_i)

#     return srcs, dsts, dists

# # pylint: disable=E1102
# def mol_to_nearest_neighbor_graph(mol,
#                                   coordinates,
#                                   neighbor_cutoff,
#                                   max_num_neighbors=None,
#                                   p_distance=2,
#                                   add_self_loop=False,
#                                   node_featurizer=None,
#                                   edge_featurizer=None,
#                                   canonical_atom_order=True,
#                                   keep_dists=False,
#                                   dist_field='dist',
#                                   explicit_hydrogens=False,
#                                   num_virtual_nodes=0):
#     """Convert an RDKit molecule into a nearest neighbor graph and featurize for it.

#     Different from bigraph and complete graph, the nearest neighbor graph
#     may not be symmetric since i is the closest neighbor of j does not
#     necessarily suggest the other way.

#     Parameters
#     ----------
#     mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule holder
#     coordinates : numpy.ndarray of shape (N, D)
#         The coordinates of atoms in the molecule. N for the number of atoms
#         and D for the dimensions of the coordinates.
#     neighbor_cutoff : float
#         If the distance between a pair of nodes is larger than neighbor_cutoff,
#         they will not be considered as neighboring nodes.
#     max_num_neighbors : int or None.
#         If not None, then this specifies the maximum number of neighbors
#         allowed for each atom. Default to None.
#     p_distance : int
#         We compute the distance between neighbors using Minkowski (:math:`l_p`)
#         distance. When ``p_distance = 1``, Minkowski distance is equivalent to
#         Manhattan distance. When ``p_distance = 2``, Minkowski distance is
#         equivalent to the standard Euclidean distance. Default to 2.
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     keep_dists : bool
#         Whether to store the distance between neighboring atoms in ``edata`` of the
#         constructed DGLGraphs. Default to False.
#     dist_field : str
#         Field for storing distance between neighboring atoms in ``edata``. This comes
#         into effect only when ``keep_dists=True``. Default to ``'dist'``.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Nearest neighbor DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.

#     Examples
#     --------
#     >>> from dgllife.utils import mol_to_nearest_neighbor_graph
#     >>> from rdkit import Chem
#     >>> from rdkit.Chem import AllChem

#     >>> mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C')
#     >>> AllChem.EmbedMolecule(mol)
#     >>> AllChem.MMFFOptimizeMolecule(mol)
#     >>> coords = get_mol_3d_coordinates(mol)
#     >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25)
#     >>> print(g)
#     DGLGraph(num_nodes=23, num_edges=6,
#              ndata_schemes={}
#              edata_schemes={})

#     Quite often we will want to use the distance between end atoms of edges, this can be
#     achieved with

#     >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25, keep_dists=True)
#     >>> print(g.edata['dist'])
#     tensor([[1.2024],
#             [1.2024],
#             [1.2270],
#             [1.2270],
#             [1.2259],
#             [1.2259]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C')
#     >>> mol = Chem.AddHs(mol)
#     >>> AllChem.EmbedMolecule(mol)
#     >>> AllChem.MMFFOptimizeMolecule(mol)
#     >>> coords = get_mol_3d_coordinates(mol)
#     >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25,
#     >>>                                   explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=41, num_edges=42,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     get_mol_3d_coordinates
#     k_nearest_neighbors
#     smiles_to_nearest_neighbor_graph
#     """
#     if mol is None:
#         print('Invalid mol found')
#         return None

#     if explicit_hydrogens:
#         mol = Chem.AddHs(mol)

#     num_atoms = mol.GetNumAtoms()
#     num_coords = coordinates.shape[0]
#     assert num_atoms == num_coords, \
#         'Expect the number of atoms to match the first dimension of coordinates, ' \
#         'got {:d} and {:d}'.format(num_atoms, num_coords)

#     if canonical_atom_order:
#         new_order = rdmolfiles.CanonicalRankAtoms(mol)
#         mol = rdmolops.RenumberAtoms(mol, new_order)

#     srcs, dsts, dists = k_nearest_neighbors(coordinates=coordinates,
#                                             neighbor_cutoff=neighbor_cutoff,
#                                             max_num_neighbors=max_num_neighbors,
#                                             p_distance=p_distance,
#                                             self_loops=add_self_loop)
#     g = dgl.graph(([], []), idtype=torch.int32)

#     # Add nodes first since some nodes may be completely isolated
#     g.add_nodes(num_atoms)

#     # Add edges
#     g.add_edges(srcs, dsts)

#     if node_featurizer is not None:
#         g.ndata.update(node_featurizer(mol))

#     if edge_featurizer is not None:
#         g.edata.update(edge_featurizer(mol))

#     if keep_dists:
#         assert dist_field not in g.edata, \
#             'Expect {} to be reserved for distance between neighboring atoms.'
#         g.edata[dist_field] = torch.tensor(dists).float().reshape(-1, 1)

#     if num_virtual_nodes > 0:
#         num_real_nodes = g.num_nodes()
#         real_nodes = list(range(num_real_nodes))
#         g.add_nodes(num_virtual_nodes)

#         # Change Topology
#         virtual_src = []
#         virtual_dst = []
#         for count in range(num_virtual_nodes):
#             virtual_node = num_real_nodes + count
#             virtual_node_copy = [virtual_node] * num_real_nodes
#             virtual_src.extend(real_nodes)
#             virtual_src.extend(virtual_node_copy)
#             virtual_dst.extend(virtual_node_copy)
#             virtual_dst.extend(real_nodes)
#         g.add_edges(virtual_src, virtual_dst)

#         for nk, nv in g.ndata.items():
#             nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
#             nv[:-num_virtual_nodes, -1] = 1
#             g.ndata[nk] = nv

#         for ek, ev in g.edata.items():
#             ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
#             ev[:-num_virtual_nodes * num_real_nodes * 2, -1] = 1
#             g.edata[ek] = ev

#     return g

# def smiles_to_nearest_neighbor_graph(smiles,
#                                      coordinates,
#                                      neighbor_cutoff,
#                                      max_num_neighbors=None,
#                                      p_distance=2,
#                                      add_self_loop=False,
#                                      node_featurizer=None,
#                                      edge_featurizer=None,
#                                      canonical_atom_order=True,
#                                      keep_dists=False,
#                                      dist_field='dist',
#                                      explicit_hydrogens=False,
#                                      num_virtual_nodes=0):
#     """Convert a SMILES into a nearest neighbor graph and featurize for it.

#     Different from bigraph and complete graph, the nearest neighbor graph
#     may not be symmetric since i is the closest neighbor of j does not
#     necessarily suggest the other way.

#     Parameters
#     ----------
#     smiles : str
#         String of SMILES
#     coordinates : numpy.ndarray of shape (N, D)
#         The coordinates of atoms in the molecule. N for the number of atoms
#         and D for the dimensions of the coordinates.
#     neighbor_cutoff : float
#         If the distance between a pair of nodes is larger than neighbor_cutoff,
#         they will not be considered as neighboring nodes.
#     max_num_neighbors : int or None.
#         If not None, then this specifies the maximum number of neighbors
#         allowed for each atom. Default to None.
#     p_distance : int
#         We compute the distance between neighbors using Minkowski (:math:`l_p`)
#         distance. When ``p_distance = 1``, Minkowski distance is equivalent to
#         Manhattan distance. When ``p_distance = 2``, Minkowski distance is
#         equivalent to the standard Euclidean distance. Default to 2.
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     keep_dists : bool
#         Whether to store the distance between neighboring atoms in ``edata`` of the
#         constructed DGLGraphs. Default to False.
#     dist_field : str
#         Field for storing distance between neighboring atoms in ``edata``. This comes
#         into effect only when ``keep_dists=True``. Default to ``'dist'``.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Returns
#     -------
#     DGLGraph or None
#         Nearest neighbor DGLGraph for the molecule if :attr:`smiles` is valid and None otherwise.

#     Examples
#     --------
#     >>> from dgllife.utils import smiles_to_nearest_neighbor_graph
#     >>> from rdkit import Chem
#     >>> from rdkit.Chem import AllChem

#     >>> smiles = 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
#     >>> mol = Chem.MolFromSmiles(smiles)
#     >>> AllChem.EmbedMolecule(mol)
#     >>> AllChem.MMFFOptimizeMolecule(mol)
#     >>> coords = get_mol_3d_coordinates(mol)
#     >>> g = mol_to_nearest_neighbor_graph(mol, coords, neighbor_cutoff=1.25)
#     >>> print(g)
#     DGLGraph(num_nodes=23, num_edges=6,
#              ndata_schemes={}
#              edata_schemes={})

#     Quite often we will want to use the distance between end atoms of edges, this can be
#     achieved with

#     >>> g = smiles_to_nearest_neighbor_graph(smiles, coords, neighbor_cutoff=1.25, keep_dists=True)
#     >>> print(g.edata['dist'])
#     tensor([[1.2024],
#             [1.2024],
#             [1.2270],
#             [1.2270],
#             [1.2259],
#             [1.2259]])

#     By default, we do not explicitly represent hydrogens as nodes, which can be done as follows.

#     >>> smiles = 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
#     >>> mol = Chem.MolFromSmiles(smiles)
#     >>> mol = Chem.AddHs(mol)
#     >>> AllChem.EmbedMolecule(mol)
#     >>> AllChem.MMFFOptimizeMolecule(mol)
#     >>> coords = get_mol_3d_coordinates(mol)
#     >>> g = smiles_to_nearest_neighbor_graph(smiles, coords, neighbor_cutoff=1.25,
#     >>>                                      explicit_hydrogens=True)
#     >>> print(g)
#     DGLGraph(num_nodes=41, num_edges=42,
#              ndata_schemes={}
#              edata_schemes={})

#     See Also
#     --------
#     get_mol_3d_coordinates
#     k_nearest_neighbors
#     mol_to_nearest_neighbor_graph
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     return mol_to_nearest_neighbor_graph(
#         mol, coordinates, neighbor_cutoff, max_num_neighbors, p_distance,
#         add_self_loop, node_featurizer, edge_featurizer, canonical_atom_order,
#         keep_dists, dist_field, explicit_hydrogens, num_virtual_nodes)

# class ToGraph:
#     r"""An abstract class for writing graph constructors."""
#     def __call__(self, data_obj):
#         raise NotImplementedError

#     def __repr__(self):
#         return self.__class__.__name__ + '()'

# class MolToBigraph(ToGraph):
#     """Convert RDKit molecule objects into bi-directed DGLGraphs and featurize for them.

#     Parameters
#     ----------
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Examples
#     --------
#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import MolToBigraph

#     >>> # A custom node featurizer
#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> # A custom edge featurizer
#     >>> def featurize_bonds(mol):
#     >>>     feats = []
#     >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#     >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
#     >>>     for bond in mol.GetBonds():
#     >>>         btype = bond_types.index(bond.GetBondType())
#     >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
#     >>>         feats.extend([btype, btype])
#     >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> mol_to_g = MolToBigraph(node_featurizer=featurize_atoms, edge_featurizer=featurize_bonds)
#     >>> mol = Chem.MolFromSmiles('CCO')
#     >>> g = mol_to_g(mol)
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['type'])
#     tensor([[0.],
#             [0.],
#             [0.],
#             [0.]])
#     """
#     def __init__(self,
#                  add_self_loop=False,
#                  node_featurizer=None,
#                  edge_featurizer=None,
#                  canonical_atom_order=True,
#                  explicit_hydrogens=False,
#                  num_virtual_nodes=0):
#         self.add_self_loop = add_self_loop
#         self.node_featurizer = node_featurizer
#         self.edge_featurizer = edge_featurizer
#         self.canonical_atom_order = canonical_atom_order
#         self.explicit_hydrogens = explicit_hydrogens
#         self.num_virtual_nodes = num_virtual_nodes

#     def __call__(self, mol):
#         """Construct graph for the molecule and featurize it.

#         Parameters
#         ----------
#         mol : rdkit.Chem.rdchem.Mol
#             RDKit molecule holder

#         Returns
#         -------
#         DGLGraph or None
#             Bi-directed DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.
#         """
#         return mol_to_bigraph(mol,
#                               self.add_self_loop,
#                               self.node_featurizer,
#                               self.edge_featurizer,
#                               self.canonical_atom_order,
#                               self.explicit_hydrogens,
#                               self.num_virtual_nodes)

# class SMILESToBigraph(ToGraph):
#     """Convert SMILES strings into bi-directed DGLGraphs and featurize for them.

#     Parameters
#     ----------
#     add_self_loop : bool
#         Whether to add self loops in DGLGraphs. Default to False.
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph. Default to None.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph. Default to None.
#     canonical_atom_order : bool
#         Whether to use a canonical order of atoms returned by RDKit. Setting it
#         to true might change the order of atoms in the graph constructed. Default
#         to True.
#     explicit_hydrogens : bool
#         Whether to explicitly represent hydrogens as nodes in the graph. If True,
#         it will call rdkit.Chem.AddHs(mol). Default to False.
#     num_virtual_nodes : int
#         The number of virtual nodes to add. The virtual nodes will be connected to
#         all real nodes with virtual edges. If the returned graph has any node/edge
#         feature, an additional column of binary values will be used for each feature
#         to indicate the identity of virtual node/edges. The features of the virtual
#         nodes/edges will be zero vectors except for the additional column. Default to 0.

#     Examples
#     --------
#     >>> import torch
#     >>> from rdkit import Chem
#     >>> from dgllife.utils import SMILESToBigraph

#     >>> # A custom node featurizer
#     >>> def featurize_atoms(mol):
#     >>>     feats = []
#     >>>     for atom in mol.GetAtoms():
#     >>>         feats.append(atom.GetAtomicNum())
#     >>>     return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> # A custom edge featurizer
#     >>> def featurize_bonds(mol):
#     >>>     feats = []
#     >>>     bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
#     >>>                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
#     >>>     for bond in mol.GetBonds():
#     >>>         btype = bond_types.index(bond.GetBondType())
#     >>>         # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
#     >>>         feats.extend([btype, btype])
#     >>>     return {'type': torch.tensor(feats).reshape(-1, 1).float()}

#     >>> smi_to_g = SMILESToBigraph(node_featurizer=featurize_atoms,
#     ...                            edge_featurizer=featurize_bonds)
#     >>> g = smi_to_g('CCO')
#     >>> print(g.ndata['atomic'])
#     tensor([[6.],
#             [8.],
#             [6.]])
#     >>> print(g.edata['type'])
#     tensor([[0.],
#             [0.],
#             [0.],
#             [0.]])
#     """
#     def __init__(self,
#                  add_self_loop=False,
#                  node_featurizer=None,
#                  edge_featurizer=None,
#                  canonical_atom_order=True,
#                  explicit_hydrogens=False,
#                  num_virtual_nodes=0):
#         self.add_self_loop = add_self_loop
#         self.node_featurizer = node_featurizer
#         self.edge_featurizer = edge_featurizer
#         self.canonical_atom_order = canonical_atom_order
#         self.explicit_hydrogens = explicit_hydrogens
#         self.num_virtual_nodes = num_virtual_nodes

#     def __call__(self, smiles):
#         """Construct graph for the molecule and featurize it.

#         Parameters
#         ----------
#         smiles : str
#             SMILES string.

#         Returns
#         -------
#         DGLGraph or None
#             Bi-directed DGLGraph for the molecule if :attr:`smiles` is valid and None otherwise.
#         """
#         return smiles_to_bigraph(smiles,
#                                  self.add_self_loop,
#                                  self.node_featurizer,
#                                  self.edge_featurizer,
#                                  self.canonical_atom_order,
#                                  self.explicit_hydrogens,
#                                  self.num_virtual_nodes)


# def featurize_nodes_and_compute_combo_scores(
#         node_featurizer, reactant_mol, valid_candidate_combos):
#     """Featurize atoms in reactants and compute scores for combos of bond changes

#     Parameters
#     ----------
#     node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for nodes like atoms in a molecule, which can be used to update
#         ndata for a DGLGraph.
#     reactant_mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule instance for reactants in a reaction
#     valid_candidate_combos : list
#         valid_candidate_combos[i] gives a list of tuples, which is the i-th valid combo
#         of candidate bond changes for the reaction.

#     Returns
#     -------
#     node_feats : float32 tensor of shape (N, M)
#         Node features for reactants, N for the number of nodes and M for the feature size
#     combo_bias : float32 tensor of shape (B, 1)
#         Scores for combos of bond changes, B equals len(valid_candidate_combos)
#     """
#     node_feats = node_featurizer(reactant_mol)['hv']
#     combo_bias = torch.zeros(len(valid_candidate_combos), 1).float()
#     for combo_id, combo in enumerate(valid_candidate_combos):
#         combo_bias[combo_id] = sum([
#             score for (atom1, atom2, change_type, score) in combo])

#     return node_feats, combo_bias

# def construct_graphs_rank(info, edge_featurizer):
#     """Construct graphs for reactants and candidate products in a reaction and featurize
#     their edges

#     Parameters
#     ----------
#     info : 4-tuple
#         * reactant_mol : rdkit.Chem.rdchem.Mol
#             RDKit molecule instance for reactants in a reaction
#         * candidate_combos : list
#             candidate_combos[i] gives a list of tuples, which is the i-th valid combo
#             of candidate bond changes for the reaction.
#         * candidate_bond_changes : list of 4-tuples
#             Refined candidate bond changes considered for candidate products
#         * reactant_info : dict
#             Reaction-related information of reactants.
#     edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
#         Featurization for edges like bonds in a molecule, which can be used to update
#         edata for a DGLGraph.

#     Returns
#     -------
#     reaction_graphs : list of DGLGraphs
#         DGLGraphs for reactants and candidate products with edge features in edata['he'],
#         where the first graph is for reactants.
#     """
#     reactant_mol, candidate_combos, candidate_bond_changes, reactant_info = info
#     # Graphs for reactants and candidate products
#     reaction_graphs = []

#     # Get graph for the reactants
#     reactant_graph = mol_to_bigraph(reactant_mol, edge_featurizer=edge_featurizer,
#                                     canonical_atom_order=False)
#     reaction_graphs.append(reactant_graph)

#     candidate_bond_changes_no_score = [
#         (atom1, atom2, change_type)
#         for (atom1, atom2, change_type, score) in candidate_bond_changes]

#     # Prepare common components across all candidate products
#     breaking_reactant_neighbors = []
#     common_src_list = []
#     common_dst_list = []
#     common_edge_feats = []
#     num_bonds = reactant_mol.GetNumBonds()
#     for j in range(num_bonds):
#         bond = reactant_mol.GetBondWithIdx(j)
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#         u_sort, v_sort = min(u, v), max(u, v)
#         # Whether a bond in reactants might get broken
#         if (u_sort, v_sort, 0.0) not in candidate_bond_changes_no_score:
#             common_src_list.extend([u, v])
#             common_dst_list.extend([v, u])
#             common_edge_feats.extend([reactant_graph.edata['he'][2 * j],
#                                       reactant_graph.edata['he'][2 * j + 1]])
#         else:
#             breaking_reactant_neighbors.append((
#                 u_sort, v_sort, bond.GetBondTypeAsDouble()))

#     for combo in candidate_combos:
#         combo_src_list = deepcopy(common_src_list)
#         combo_dst_list = deepcopy(common_dst_list)
#         combo_edge_feats = deepcopy(common_edge_feats)
#         candidate_bond_end_atoms = [
#             (atom1, atom2) for (atom1, atom2, change_type, score) in combo]
#         for (atom1, atom2, change_type) in breaking_reactant_neighbors:
#             if (atom1, atom2) not in candidate_bond_end_atoms:
#                 # If a bond might be broken in some other combos but not this,
#                 # add it as a negative sample
#                 combo.append((atom1, atom2, change_type, 0.0))

#         for (atom1, atom2, change_type, score) in combo:
#             if change_type == 0:
#                 continue
#             combo_src_list.extend([atom1, atom2])
#             combo_dst_list.extend([atom2, atom1])
#             feats = one_hot_encoding(change_type, [1.0, 2.0, 3.0, 1.5, -1])
#             if (atom1, atom2) in reactant_info['ring_bonds']:
#                 feats[-1] = 1
#             feats = torch.tensor(feats).float()
#             combo_edge_feats.extend([feats, feats.clone()])

#         combo_graph = dgl.graph(([], []))
#         combo_graph.add_nodes(reactant_graph.num_nodes())
#         if len(combo_edge_feats) > 0:
#             combo_edge_feats = torch.stack(combo_edge_feats, dim=0)
#             combo_graph.add_edges(combo_src_list, combo_dst_list)
#             combo_graph.edata['he'] = combo_edge_feats
#         else:
#             combo_graph.edata['he'] = torch.zeros((0, 5))
#         reaction_graphs.append(combo_graph)

#     return reaction_graphs


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
        if ((atom1, atom2) not in reactant_info['pair_to_bond_val']) or \
                (reactant_info['pair_to_bond_val'][(atom1, atom2)] != change_type):
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
    num_candidate_bond_changes = args.num_candidate_bond_changes
    max_num_bond_changes = args.max_num_bond_changes
    max_num_change_combos_per_reaction = args.max_num_change_combos_per_reaction
    
    raw_candidate_bond_changes = "" # TODO from model_output
   
    gold_bond_change = batch['y']
    reactant_mol = batch['reactant']
    if mode == 'train':
        product_mol = batch['product']
    else:
        product_mol = None


    # Get valid candidate products, candidate bond changes considered and reactant info
    # def pre_process_one_reaction(info, num_candidate_bond_changes, max_num_bond_changes,
    #                          max_num_change_combos, mode):
    info = (raw_candidate_bond_changes, gold_bond_change, reactant_mol, product_mol)
    valid_candidate_combos, candidate_bond_changes, reactant_info, candidate_smiles = \
        pre_process_one_reaction(info, num_candidate_bond_changes, max_num_bond_changes, max_num_change_combos_per_reaction, mode)

    
    # TODO: from here replace with pyg data (run from_smiles on all mols, will also featurize)
    list_of_data_batches = []
    for list_of_candidates in candidate_smiles:
        data_batch = []
        for candidate in list_of_candidates:
            data_batch.append(from_smiles(candidate, return_atom_number=True))
        list_of_data_batches.append(Batch.from_data_list(data_batch))


    # Construct DGLGraphs and featurize their edges
    # g_list = construct_graphs_rank(
    #     (reactant_mol, valid_candidate_combos,
    #         candidate_bond_changes, reactant_info),
    #     self.edge_featurizer)

    # # Get node features and candidate scores
    # node_feats, candidate_scores = featurize_nodes_and_compute_combo_scores(
    #     self.node_featurizer, reactant_mol, valid_candidate_combos)
    # for g in g_list:
    #     g.ndata['hv'] = node_feats

    # if self.mode == 'train':
    #     labels = torch.zeros(1, 1).long()
    #     return g_list, candidate_scores, labels
    # else:
    #     reactant_mol = self.reactant_mols[item]
    #     gold_bond_change = self.gold_bond_change[item]
    #     product_mol = self.product_mols[item]
    #     return g_list, candidate_scores, valid_candidate_combos, \
    #             reactant_mol, gold_bond_change, product_mol


