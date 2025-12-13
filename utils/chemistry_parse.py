import rdkit
from rdkit import Chem
import numpy as np
from typing import Dict

from rdkit.Chem.rdmolfiles import SmilesParserParams

def get_mol(smiles: str, kekulize: bool = False, removeHs: bool = False) -> Chem.Mol:
    """SMILES string to Mol.
    Parameters
    ----------
    smiles: str,
        SMILES string for molecule
    kekulize: bool,
        Whether to kekulize the molecule
    """
    # 创建SMILES解析参数对象，并设置removeHs=False（保留显式氢）
    params = SmilesParserParams()
    params.removeHs = removeHs

    mol = Chem.MolFromSmiles(smiles, params=params)
    if mol is not None and kekulize:
        Chem.Kekulize(mol)
    return mol


def get_bond_info(mol: Chem.Mol) -> Dict:
    """Get information on bonds in the molecule.
    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()
        if a_start > a_end:
            a_start, a_end = a_end, a_start

        bond_type = bond.GetBondTypeAsDouble()
        bond_info[(a_start, a_end)] = [bond_type, bond.GetIdx()]

    return bond_info


def get_reaction_core(reac: str, prod: str):
    reac_mol, prod_mol = get_mol(reac), get_mol(prod)
    if reac_mol is None or prod_mol is None:
        raise NotImplementedError('[PREPROCESS] Invalid Smiles Given')

    prod_bonds = get_bond_info(prod_mol)
    reac_bonds = get_bond_info(reac_mol)

    prod_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in prod_mol.GetAtoms()
    }

    reac_amap_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in reac_mol.GetAtoms()
    }

    RCs = set()

    for bond, (ftype, bidx) in prod_bonds.items():
        if bond not in reac_bonds or reac_bonds[bond][0] != ftype:
            RCs.update(bond)

    for bond in reac_bonds:
        if bond[0] in prod_amap_idx:
            if bond[1] not in prod_amap_idx:
                RCs.add(bond[0])
            elif bond not in prod_bonds:
                RCs.update(bond)
        elif bond[1] in prod_amap_idx:
            RCs.add(bond[1])

    for atom in prod_mol.GetAtoms():
        amap_num = atom.GetAtomMapNum()
        reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[amap_num])
        if atom.GetTotalNumHs(includeNeighbors=True) != reac_atom.GetTotalNumHs(includeNeighbors=True):
            RCs.add(amap_num)
        if atom.GetFormalCharge() != reac_atom.GetFormalCharge():
            RCs.add(amap_num)

    reac_rc, prod_rc = [], []
    for x in RCs:
        reac_atom = reac_mol.GetAtomWithIdx(reac_amap_idx[x])
        for n_am in reac_atom.GetNeighbors():
            if n_am.GetAtomMapNum() != 0:
                reac_rc.append(n_am.GetAtomMapNum())

        prod_atom = prod_mol.GetAtomWithIdx(prod_amap_idx[x])
        for n_am in prod_atom.GetNeighbors():
            if n_am.GetAtomMapNum() != 0:
                prod_rc.append(n_am.GetAtomMapNum())

    return (RCs | set(reac_rc)), (RCs | set(prod_rc))



def canonical_smiles(x):
    mol = get_mol(x, kekulize=False)
    return Chem.MolToSmiles(mol)


def removeHs(x):
    mol = Chem.MolFromSmiles(x)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def removeHs_rxn(x):
    reac, prod = x.split('>>')
    return f'{removeHs(reac)}>>{removeHs(prod)}'


def clear_map_number(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return canonical_smiles(Chem.MolToSmiles(mol))


def canonical_rxn(rxn):
    reac, prod = rxn.split('>>')
    reac = clear_map_number(reac)
    prod = clear_map_number(prod)
    return f'{reac}>>{prod}'
