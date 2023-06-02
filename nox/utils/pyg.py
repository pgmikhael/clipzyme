# functions from torch_geometric.utils not yet importable
from rdkit import Chem, RDLogger
from typing import List
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree


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


def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False, return_atom_number=False):
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
        x.append(x_map["num_radical_electrons"].index( max(atom.GetNumRadicalElectrons(),4)))
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(x)
        x_ids.append(atom.GetIdx())
        if return_atom_number:
            x_numbers.append(int(atom.GetProp("molAtomMapNumber")) if atom.HasProp("molAtomMapNumber") else -1)


    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)
    #x_ids = torch.tensor(x_ids, dtype=torch.long).view(-1)


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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, x_ids=x_ids)
    if return_atom_number:
        data.atom_map_number = x_numbers # torch.tensor(x_numbers).view(-1)
    return data


def from_mapped_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):
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
        x.append(x_map["num_radical_electrons"].index( max(atom.GetNumRadicalElectrons(),4)))
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(x)
        x_ids.append(atom.GetIdx())
        
        assert atom.HasProp("molAtomMapNumber"), "SMILES must contain atom map numbers"
        
        # if atom.HasProp("molAtomMapNumber"):
        atom_map_number = int(atom.GetProp("molAtomMapNumber"))
        x_numbers.append(atom_map_number)
        atom_index_map[atom.GetIdx()] = atom_map_number  # Update mapping dictionary
    #     else:
    #         # if no atom map number is present, we will put it at the end. For now just keep track of these indices
    #         x_numbers.append(-1)
    #         atom_index_map[atom.GetIdx()] = None

    # num_nones = x_numbers.count(-1)
    # for atom_idx, atom_map_number in atom_index_map.items():
    #     if atom_map_number is None:
    #         atom_index_map[atom_idx] = max(x_numbers) + 1
    #         x_numbers[atom_idx] = max(x_numbers) + 1

    new_xs = [None for _ in range(len(xs))]
    order = sorted(atom_index_map.items(), key=lambda x: x[1]) # sort by atom number because k,v = atom_idx, atom_map_number
    old_index2new_index = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(order)}
    atom_map_number2new_index = {atom_map_number: new_idx for new_idx, (_, atom_map_number) in enumerate(order)}

    for i, (atom_idx, atom_map_number) in enumerate(order):
        new_xs[i] = xs[atom_idx]

    x = torch.tensor(new_xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Remap i and j according to the atom_map_number
        i = old_index2new_index[i]
        j = old_index2new_index[j]

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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, x_ids=x_ids)
    data.atom_map_number = x_numbers # torch.tensor(x_numbers).view(-1)
    return data, atom_map_number2new_index
