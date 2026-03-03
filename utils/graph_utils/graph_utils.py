# from ogb.utils.features import (
#     allowable_features, atom_to_feature_vector, bond_feature_vector_to_dict,
#     bond_to_feature_vector, atom_feature_vector_to_dict
# )
from ..ogb_utils_features import (
    allowable_features, atom_to_feature_vector, bond_to_feature_vector
)
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from rdkit.Chem.rdmolfiles import SmilesParserParams
# 创建SMILES解析参数对象，并设置removeHs=False（保留显式氢）
params = SmilesParserParams()
params.removeHs = False

def smiles2graph(smiles_string, with_amap=False, with_local_pe=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string, params=params)
    if with_amap:
        max_amap = max([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(max_amap + 1)
                max_amap = max_amap + 1

        amap_idx = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
        }

    # atoms
    atom_features_list = []
    local_pe_mapper = {}
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        if with_local_pe and atom.GetChiralTag() in [
            ChiralType.CHI_TETRAHEDRAL_CCW,
            ChiralType.CHI_TETRAHEDRAL_CW
        ]:
            neighbors = list(atom.GetNeighbors())
            # if len(neighbors) != 4:
            #     raise ValueError("Implicit Hs in SMILES")
            for idx, nei in enumerate(neighbors):
                local_pe_mapper[(nei.GetIdx(), atom.GetIdx())] = idx
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        local_pe_list = [] if with_local_pe else None
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            if with_local_pe:
                local_pe_list.append(local_pe_mapper.get((i, j), 4))
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            if with_local_pe:
                local_pe_list.append(local_pe_mapper.get((j, i), 4))

        # data.edge_index: Graph connectivity
        # in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with
        # shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)
        if with_local_pe:
            local_pe = np.array(local_pe_list, dtype=np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        if with_local_pe:
            local_pe = np.empty((0, ), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    if with_local_pe:
        graph['local_pe'] = local_pe
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    if with_amap:
        return graph, amap_idx
    else:
        return graph
