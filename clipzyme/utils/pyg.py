# functions from torch_geometric.utils not yet importable
from rdkit import Chem, RDLogger
from typing import List
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from itertools import combinations

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map = {
    "bond_type": [
        "misc",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
    return_atom_number=False,
    use_one_hot_encoding=False,
):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.
    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    # mapping order = from2to
    xs = []
    x_ids = []
    x_numbers = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
        x.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        x.append(x_map["degree"].index(atom.GetTotalDegree()))
        x.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        x.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        x.append(
            x_map["num_radical_electrons"].index(max(atom.GetNumRadicalElectrons(), 4))
        )
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(x)
        x_ids.append(atom.GetIdx())
        if return_atom_number:
            x_numbers.append(
                int(atom.GetProp("molAtomMapNumber"))
                if atom.HasProp("molAtomMapNumber")
                else -1
            )

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)
    # x_ids = torch.tensor(x_ids, dtype=torch.long).view(-1)

    # Map i and j to the new order
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    if use_one_hot_encoding:
        one_hot_x = torch.hstack(
            [F.one_hot(x[:, i], len(v)) for i, (k, v) in enumerate(x_map.items())]
        )
        if x.shape[-1] > len(x_map):
            one_hot_x = torch.hstack([one_hot_x, x[:, len(x_map) :]])
        x = one_hot_x

        one_hot_e = torch.hstack(
            [
                F.one_hot(edge_attr[:, i], len(v))
                for i, (k, v) in enumerate(e_map.items())
            ]
        )
        if edge_attr.shape[-1] > len(e_map):
            one_hot_e = torch.hstack([one_hot_e, x[:, len(e_map) :]])
        edge_attr = one_hot_e

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, x_ids=x_ids
    )
    if return_atom_number:
        data.atom_map_number = x_numbers  # torch.tensor(x_numbers).view(-1)
    return data


def from_mapped_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
    encode_no_edge=False,
    sanitize=True,
    use_one_hot_encoding=False,
):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.
    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        encode_no_edge (bool, optional): adds edges between all molecules
        sanitize (bool, optional): sanitize molecules when using rdkit. typically should be set to `True`
    """
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)

    if mol is None:
        return None, None  # mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    x_ids = []
    x_numbers = []
    atom_index_map = {}
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
        x.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        x.append(x_map["degree"].index(atom.GetTotalDegree()))
        x.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        x.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        x.append(
            x_map["num_radical_electrons"].index(max(atom.GetNumRadicalElectrons(), 4))
        )
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(x)
        x_ids.append(atom.GetIdx())

        assert atom.HasProp("molAtomMapNumber"), "SMILES must contain atom map numbers"

        atom_map_number = int(atom.GetProp("molAtomMapNumber"))
        x_numbers.append(atom_map_number)
        atom_index_map[atom.GetIdx()] = atom_map_number  # Update mapping dictionary

    new_xs = [None for _ in range(len(xs))]
    order = sorted(
        atom_index_map.items(), key=lambda x: x[1]
    )  # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {
        old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)
    }
    atom_map_number2new_index = {
        atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)
    }

    for i, (atom_idx, atom_map_number) in enumerate(order):
        new_xs[i] = xs[atom_idx]

    x = torch.tensor(new_xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    edge_indices_complete, edge_attrs_complete = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Remap i and j according to the atom_map_number
        i = old_index2new_index[i]
        j = old_index2new_index[j]

        e = []
        e_complete = [0, 1]  # [different components, same component, bond features]

        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        e_complete.append(
            e_map["bond_type"].index(str(bond.GetBondType())) + 1
        )  # ! add 1 in case we use encode_no_edge
        e_complete.append(
            e_map["stereo"].index(str(bond.GetStereo())) + 1
        )  # ! add 1 in case we use encode_no_edge
        e_complete.append(
            e_map["is_conjugated"].index(bond.GetIsConjugated()) + 1
        )  # ! add 1 in case we use encode_no_edge

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
        edge_indices_complete += [[i, j], [j, i]]
        edge_attrs_complete += [e_complete, e_complete]

    if encode_no_edge:
        # atom id to molecule id (if multiple in smiles)
        atom2molecule = {}
        for mol_id, s in enumerate(smiles.split(".")):
            single_mol = Chem.MolFromSmiles(s, sanitize=sanitize)
            for atom in single_mol.GetAtoms():
                # idx = old_index2new_index[ atom.GetIdx() ] # ! problem: atom.GetIdx() gets reset for each smiles in list; use atom number
                idx = atom_map_number2new_index[int(atom.GetProp("molAtomMapNumber"))]
                atom2molecule[idx] = mol_id  # new index to molecule id

        edge_attr_dim = len(e_map)
        set_edge_indices = set([tuple(idxs) for idxs in edge_indices])
        total_num_atoms = mol.GetNumAtoms()
        for i, j in combinations(
            range(total_num_atoms), 2
        ):  # get all combs of new indices
            if (i, j) not in set_edge_indices:
                if atom2molecule[i] == atom2molecule[j]:  # same molecule (component)
                    e = [0, 1] + [0 for _ in range(edge_attr_dim)]
                else:
                    e = [1, 0] + [0 for _ in range(edge_attr_dim)]
                edge_indices_complete += [[i, j], [j, i]]
                edge_attrs_complete += [e, e]
                set_edge_indices.update(
                    [(i, j), (j, i)]
                )  # prob not necessary but just to be safe

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    edge_index_complete = torch.tensor(edge_indices_complete)
    edge_index_complete = edge_index_complete.t().to(torch.long).view(2, -1)
    edge_attr_complete = torch.tensor(edge_attrs_complete, dtype=torch.long).view(-1, 5)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        perm = (edge_index_complete[0] * x.size(0) + edge_index_complete[1]).argsort()
        edge_index_complete, edge_attr_complete = (
            edge_index_complete[:, perm],
            edge_attr_complete[perm],
        )

    if use_one_hot_encoding:
        one_hot_x = torch.hstack(
            [F.one_hot(x[:, i], len(v)) for i, (k, v) in enumerate(x_map.items())]
        )
        if x.shape[-1] > len(x_map):
            one_hot_x = torch.hstack([one_hot_x, x[:, len(x_map) :]])
        x = one_hot_x

        one_hot_e = torch.hstack(
            [
                F.one_hot(edge_attr[:, i], len(v))
                for i, (k, v) in enumerate(e_map.items())
            ]
        )
        if edge_attr.shape[-1] > len(e_map):
            one_hot_e = torch.hstack([one_hot_e, x[:, len(e_map) :]])
        edge_attr = one_hot_e

    if encode_no_edge:
        if use_one_hot_encoding:
            one_hot_e = torch.hstack(
                [
                    F.one_hot(edge_attr_complete[:, (i + 2)], len(v) + 1)
                    for i, (k, v) in enumerate(e_map.items())
                ]
            )  # start at i = 2 since first 2 are for encoding same or diff mol
            one_hot_e = torch.hstack([edge_attr_complete[:, :2], one_hot_e])
            edge_attr_complete = one_hot_e

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles,
            x_ids=x_ids,
            edge_index_complete=edge_index_complete,
            edge_attr_complete=edge_attr_complete,
        )
    else:
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, x_ids=x_ids
        )
    data.atom_map_number = x_numbers  # torch.tensor(x_numbers).view(-1)
    return data, atom_map_number2new_index
