import torch
import torch_geometric
import numpy as np
from numpy import concatenate as npcat

from .chemistry_parse import get_reaction_core
from .graph_utils import smiles2graph


class RAlignDatasetBase(torch.utils.data.Dataset):
    def __init__(self, reactions):
        super(RAlignDatasetBase, self).__init__()
        self.reactions = reactions

    def __len__(self):
        return len(self.reactions)

    @staticmethod
    def _reorder_graph_by_am(graph, amap_to_idx, ordered_ams):
        if len(ordered_ams) != graph['num_nodes']:
            raise ValueError(
                'The reordered atom-map list must match the number of graph nodes'
            )

        am_to_rank = {am: idx for idx, am in enumerate(ordered_ams)}
        old_to_new = {
            old_idx: am_to_rank[am]
            for am, old_idx in amap_to_idx.items()
        }

        new_node_feat = np.zeros_like(graph['node_feat'])
        for am, old_idx in amap_to_idx.items():
            new_node_feat[am_to_rank[am]] = graph['node_feat'][old_idx]

        edge_index = graph['edge_index']
        new_edge_index = np.array([
            [old_to_new[idx] for idx in edge_index[0]],
            [old_to_new[idx] for idx in edge_index[1]]
        ], dtype=np.int64)

        graph['node_feat'] = new_node_feat
        graph['edge_index'] = new_edge_index
        return am_to_rank

    def get_aligned_graphs(self, index):
        reac, prod = self.reactions[index].strip().split('>>')
        reac_rcs, prod_rcs = get_reaction_core(reac, prod, hop=1)

        reac_mol, reac_amap = smiles2graph(reac, with_amap=True)
        prod_mol, prod_amap = smiles2graph(prod, with_amap=True)

        shared_ams = sorted(set(reac_amap.keys()) & set(prod_amap.keys()))
        reac_only_ams = sorted(set(reac_amap.keys()) - set(prod_amap.keys()))
        prod_only_ams = sorted(set(prod_amap.keys()) - set(reac_amap.keys()))

        reac_order = shared_ams + reac_only_ams
        prod_order = shared_ams + prod_only_ams

        self._reorder_graph_by_am(reac_mol, reac_amap, reac_order)
        prod_am2rank = self._reorder_graph_by_am(prod_mol, prod_amap, prod_order)

        reac_mol['is_rc'] = np.array(
            [am in reac_rcs for am in reac_order], dtype=bool
        )
        prod_mol['is_rc'] = np.array(
            [am in prod_rcs for am in prod_order], dtype=bool
        )
        reac_mol['isprod'] = np.array(
            [am in prod_am2rank for am in reac_order], dtype=bool
        )
        return reac_mol, prod_mol

    def __getitem__(self, index):
        raise NotImplementedError('the __getitem__ is not implemented for Base Dataset')


def graph_col_fn(batch):
    batch_size, edge_idx, node_feat, edge_feat = len(batch), [], [], []
    node_ptr, node_batch, lstnode, isprod, vols = [0], [], 0, [], []
    max_node, is_rc = max(x['num_nodes'] for x in batch), []
    batch_mask = torch.zeros(batch_size, max_node).bool()

    for idx, gp in enumerate(batch):
        node_cnt = gp['num_nodes']
        if node_cnt == 0:
            node_ptr.append(lstnode)
            continue

        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)

        if 'is_rc' in gp:
            is_rc.append(torch.Tensor(gp['is_rc']).bool())
        if 'isprod' in gp:
            isprod.append(gp['isprod'])
        if 'volumn' in gp:
            vols.append(gp['volumn'])

        batch_mask[idx, :node_cnt] = True
        lstnode += node_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        'edge_attr': torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'num_nodes': lstnode,
        'batch_mask': batch_mask
    }

    if len(is_rc) > 0:
        result['is_rc'] = torch.cat(is_rc, dim=0)

    if len(isprod) > 0:
        result['is_prod'] = torch.from_numpy(npcat(isprod, axis=0))

    if len(vols) > 0:
        result['volumn'] = torch.from_numpy(npcat(vols, axis=0))

    return torch_geometric.data.Data(**result)


class JointDataset(RAlignDatasetBase):
    def __init__(self, reactions, is_elementary, barrier):
        super(JointDataset, self).__init__(reactions)
        self.cls_label = is_elementary
        self.reg_label = barrier

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        cls_label = self.cls_label[idx]
        reg_label = self.reg_label[idx]
        return reac_mol, prod_mol, cls_label, reg_label


def joint_colfn(batch):
    reacs, prods, cls_labels, reg_labels = [], [], [], []
    for x in batch:
        reacs.append(x[0])
        prods.append(x[1])
        cls_labels.append(x[2])
        reg_labels.append(x[3])

    return graph_col_fn(reacs), graph_col_fn(prods), \
        torch.tensor(cls_labels, dtype=torch.long), \
        torch.tensor(reg_labels, dtype=torch.float32)
