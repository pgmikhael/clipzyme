import rdkit
from rdkit import Chem
import argparse
from rdkit import RDLogger
import numpy as np
from tqdm import tqdm
from indigo import *

lg = RDLogger.logger()
lg.setLevel(4)

idxfunc = lambda a: a.GetAtomMapNum()
bond_types = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
bond_types_as_double = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}


# Define a standardization procedure so we can evaluate based on...
# a) RDKit-sanitized equivalence, and
# b) MOLVS-sanitized equivalence
from molvs import Standardizer

standardizer = Standardizer()
standardizer.prefer_organic = True


def sanitize_smiles(smi, largest_fragment=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
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


def addmapbyINDIGO(rxnsmi):
    indigo = Indigo()
    smi = rxnsmi
    rxn = indigo.loadReaction(smi)
    rxn.automap("discard ignore_charges ignore_isotopes ignore_valence ignore_radicals")
    rxnsmi_map_indigo = rxn.smiles()
    rxnsmi_map_indigo = rxn.canonicalSmiles()
    return rxnsmi_map_indigo


"""
Functions below evaluate the quality of predictions from the rank_diff_wln model by applying the predicted
graph edits to the reactants, cleaning up the generated product, and comparing it to what was recorded
as the true (major) product of that reaction
"""

# Define some post-sanitization reaction cleaning scripts
# These are to align our graph edit representation of a reaction with the data for improved coverage
from rdkit.Chem import AllChem

clean_rxns_presani = [
    AllChem.ReactionFromSmarts(
        "[O:1]=[c:2][n;H0:3]>>[O:1]=[c:2][n;H1:3]"
    ),  # hydroxypyridine written with carbonyl, must invent H on nitrogen
]
clean_rxns_postsani = [
    AllChem.ReactionFromSmarts(
        "[n;H1;+0:1]:[n;H0;+1:2]>>[n;H0;+0:1]:[n;H0;+0:2]"
    ),  # two adjacent aromatic nitrogens should allow for H shift
    AllChem.ReactionFromSmarts(
        "[n;H1;+0:1]:[c:3]:[n;H0;+1:2]>>[n;H0;+0:1]:[*:3]:[n;H0;+0:2]"
    ),  # two aromatic nitrogens separated by one should allow for H shift
    AllChem.ReactionFromSmarts("[#7;H0;+:1]-[O;H1;+0:2]>>[#7;H0;+:1]-[O;H0;-:2]"),
    AllChem.ReactionFromSmarts(
        "[C;H0;+0:1](=[O;H0;+0:2])[O;H0;-1:3]>>[C;H0;+0:1](=[O;H0;+0:2])[O;H1;+0:3]"
    ),  # neutralize C(=O)[O-]
    AllChem.ReactionFromSmarts(
        "[I,Br,F;H1;D0;+0:1]>>[*;H0;-1:1]"
    ),  # turn neutral halogens into anions EXCEPT HCl
    AllChem.ReactionFromSmarts(
        "[N;H0;-1:1]([C:2])[C:3]>>[N;H1;+0:1]([*:2])[*:3]"
    ),  # inexplicable nitrogen anion in reactants gets fixed in prods
]
for clean_rxn in clean_rxns_presani + clean_rxns_postsani:
    if clean_rxn.Validate() != (0, 0):
        raise ValueError("Invalid cleaning reaction - check your SMARTS!")
BOND_TYPE = [
    0,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def edit_mol(rmol, edits):
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
        amap[atom.GetIntProp("molAtomMapNumber")] = atom.GetIdx()

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


"""
Functions below the data used in Wengong Jin's NIPS paper on predicting reaction outcomes for the modified
forward prediction script. Rather than just training to predict which bonds change, we make a direct prediction
on HOW those bonds change
"""


def atom_numberize_post(reactants_smi):
    m = Chem.MolFromSmiles(reactants_smi)
    atomnums = sorted(
        [
            str(a.GetProp("molAtomMapNumber"))
            for a in m.GetAtoms()
            if a.HasProp("molAtomMapNumber")
        ]
    )

    mapnum = 1
    for a in m.GetAtoms():
        if not a.HasProp("molAtomMapNumber"):
            while str(mapnum) in atomnums:
                mapnum += 1
            a.SetIntProp("molAtomMapNumber", mapnum)
            mapnum += 1
        else:
            continue

    return Chem.MolToSmiles(m)


def get_changed_bonds(rxn_smi):
    reactants = Chem.MolFromSmiles(atom_numberize_post(rxn_smi.split(">")[0]))
    products = Chem.MolFromSmiles(rxn_smi.split(">")[2])

    conserved_maps = [
        a.GetProp("molAtomMapNumber")
        for a in products.GetAtoms()
        if a.HasProp("molAtomMapNumber")
    ]
    bond_changes = set()  # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        begin_atom_num = (
            bond.GetBeginAtom().GetProp("molAtomMapNumber")
            if bond.GetBeginAtom().HasProp("molAtomMapNumber")
            else None
        )
        end_atom_num = (
            bond.GetEndAtom().GetProp("molAtomMapNumber")
            if bond.GetEndAtom().HasProp("molAtomMapNumber")
            else None
        )
        nums = sorted([begin_atom_num, end_atom_num])

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
            bond_changes.add((bond.split("~")[0], bond.split("~")[1], 0.0))  # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add(
                    (bond.split("~")[0], bond.split("~")[1], bonds_new[bond])
                )  # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add(
                (bond.split("~")[0], bond.split("~")[1], bonds_new[bond])
            )  # new bond

    return reactants, products, bond_changes


def product_is_recoverable(rmol, pmol, gedits, bonds_as_doubles=True):
    try:
        gfound, gfound_sani = 0, 0
        thisrow = []

        r = Chem.MolToSmiles(rmol)
        p = Chem.MolToSmiles(pmol)

        thisrow.append(r)
        thisrow.append(p)

        # Save pbond information
        pbonds = {}
        for bond in pmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType())
            pbonds[(a1, a2)] = pbonds[(a2, a1)] = t + 1

        for atom in pmol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")

        psmiles = Chem.MolToSmiles(pmol)
        psmiles_sani = set(sanitize_smiles(psmiles, True).split("."))
        psmiles = set(psmiles.split("."))

        thisrow.append(".".join(psmiles))
        thisrow.append(".".join(psmiles_sani))

        ########### Use *true* edits to try to recover product

        if bonds_as_doubles:
            cbonds = []
            for gedit in gedits.split(";"):
                x, y, t = gedit.split("-")
                x, y, t = int(x), int(y), float(t)
                cbonds.append((x, y, bond_types_as_double[t]))
        else:
            # check if psmiles is recoverable
            cbonds = []
            for gedit in gedits.split(";"):
                x, y = gedit.split("-")
                x, y = int(x), int(y)
                if (x, y) in pbonds:
                    t = pbonds[(x, y)]
                else:
                    t = 0
                cbonds.append((x, y, t))

        # Generate products by modifying reactants with predicted edits.
        pred_smiles = edit_mol(rmol, cbonds)
        pred_smiles_sani = set(sanitize_smiles(smi) for smi in pred_smiles)
        pred_smiles = set(pred_smiles)
        if not psmiles <= pred_smiles:
            # Try again with kekulized form
            Chem.Kekulize(rmol)
            pred_smiles_kek = edit_mol(rmol, cbonds)
            pred_smiles_kek = set(pred_smiles_kek)
            if not psmiles <= pred_smiles_kek:
                if psmiles_sani <= pred_smiles_sani:
                    gfound_sani += 1

                else:
                    pass

            else:
                gfound += 1
                gfound_sani += 1

        else:
            gfound += 1
            gfound_sani += 1

        return gfound or gfound_sani
    except:
        return False


def get_atom_mapped_reaction(rxn_smi, args):
    try:
        rxn_smi = addmapbyINDIGO(rxn_smi)
    except:
        return

    rxn_smi = rxn_smi.strip().split(" ")[0]
    try:
        reactants, products, bond_changes = get_changed_bonds(rxn_smi)
        gedits = ";".join(["{}-{}-{}".format(x[0], x[1], x[2]) for x in bond_changes])
        is_recoverable = product_is_recoverable(
            reactants, products, gedits, bonds_as_doubles=not args.bonds_not_as_doubles
        )
        if not is_recoverable:
            return

    except:
        return

    rxn_smi = "{}>>{}".format(Chem.MolToSmiles(reactants), Chem.MolToSmiles(products))
    return {
        "reactants": Chem.MolToSmiles(reactants),
        "products": Chem.MolToSmiles(products),
        "bond_changes": gedits,
    }
