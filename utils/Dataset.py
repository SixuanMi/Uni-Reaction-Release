import torch
import torch_geometric
import numpy as np
from numpy import concatenate as npcat
import pandas as pd
from rdkit import Chem

from .chemistry_parse import get_reaction_core
from .graph_utils import smiles2graph, pretrain_s2g


class RAlignDatasetBase(torch.utils.data.Dataset):
    def __init__(self, reactions, use_local_pe=False):
        super(RAlignDatasetBase, self).__init__()
        self.reactions = reactions
        self.use_local_pe = use_local_pe

    def __len__(self):
        return len(self.reactions)

    def get_aligned_graphs(self, index):
        reac, prod = self.reactions[index].strip().split('>>')
        reac_rcs, prod_rcs = get_reaction_core(reac, prod, hop=1)

        reac_mol, reac_amap = smiles2graph(
            reac, with_amap=True, with_local_pe=self.use_local_pe
        )
        prod_mol, prod_amap = smiles2graph(
            prod, with_amap=True, with_local_pe=self.use_local_pe
        )

        # align the atoms so that atom with same idx
        # have the same atom map
        am2rank = {}
        for am, arank in prod_amap.items():
            am2rank[am] = arank

        for am in reac_amap:
            if am not in am2rank:
                am2rank[am] = len(am2rank)

        prod_mol['is_rc'] = [False] * len(prod_amap)
        reac_mol['is_rc'] = [False] * len(reac_amap)

        for k in reac_rcs:
            reac_mol['is_rc'][am2rank[k]] = True

        for k in prod_rcs:
            prod_mol['is_rc'][prod_amap[k]] = True

        remap = {v: am2rank[x] for x, v in reac_amap.items()}
        reac_x = np.zeros_like(reac_mol['node_feat'])
        reac_e = reac_mol['edge_index'].tolist()

        for k, v in remap.items():
            reac_x[v] = reac_mol['node_feat'][k]

        reac_e = [
            [remap[x] for x in reac_e[0]],
            [remap[x] for x in reac_e[1]]
        ]

        reac_mol['node_feat'] = reac_x
        reac_mol['edge_index'] = np.array(reac_e, dtype=np.int64)

        isprod = [False] * reac_mol['num_nodes']
        for x in prod_amap:
            isprod[am2rank[x]] = True

        reac_mol['isprod'] = np.array(isprod, dtype=bool)
        return reac_mol, prod_mol

    def __getitem__(self, index):
        msg = 'the __getitem__ is not implemented for Base Dataset'
        raise NotImplementedError(msg)


class CNYieldDataset(RAlignDatasetBase):
    def __init__(
        self, reactions, ligand, catalyst, base, additive,
        labels, condition_type='pretrain'
    ):
        super(CNYieldDataset, self).__init__(reactions)
        self.ligand = ligand
        self.catalyst = catalyst
        self.base = base
        self.additive = additive
        self.labels = labels
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __getitem__(self, index):
        reac_mol, prod_mol = self.get_aligned_graphs(index)
        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        return reac_mol, prod_mol, gf(self.ligand[index]), \
            gf(self.base[index]), gf(self.additive[index]), \
            gf(self.catalyst[index]), self.labels[index]
    
def graph_col_fn(batch):
    batch_size, edge_idx, node_feat, edge_feat = len(batch), [], [], []
    local_pe = []
    node_ptr,  node_batch, lstnode, isprod, vols = [0], [], 0, [], []
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
        if 'local_pe' in gp:
            local_pe.append(gp['local_pe'])

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
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'num_nodes': lstnode,
        'batch_mask': batch_mask
    }

    if len(local_pe) > 0:
        result['local_pe'] = torch.from_numpy(npcat(local_pe, axis=0))

    if len(is_rc) > 0:
        result['is_rc'] = torch.cat(is_rc, dim=0)

    if len(isprod) > 0:
        result['is_prod'] = torch.from_numpy(npcat(isprod, axis=0))

    if len(vols) > 0:
        result['volumn'] = torch.from_numpy(npcat(vols, axis=0))

    return torch_geometric.data.Data(**result)


def cn_colfn(batch):
    reac, prod, all_conditions, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        all_conditions.extend(x[2: -1])
        lbs.append(x[-1])

    return graph_col_fn(reac), graph_col_fn(prod), \
        graph_col_fn(all_conditions), torch.FloatTensor(lbs)


class SelDataset(RAlignDatasetBase):
    def __init__(
        self, reactions, labels, catalyst=None, condition_type='pretrain'
    ):
        super(SelDataset, self).__init__(reactions)
        self.catalyst = catalyst
        self.labels = labels
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        if self.catalyst is None:
            return reac_mol, prod_mol, self.labels[idx]
        else:
            return reac_mol, prod_mol, gf(self.catalyst[idx]), self.labels[idx]


def sel_with_cat_colfn(batch):
    reac, prod, catalyst, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        lbs.append(x[-1])
        catalyst.append(x[2])

    return graph_col_fn(reac), graph_col_fn(prod), \
        graph_col_fn(catalyst), torch.FloatTensor(lbs)


def sel_wo_cat_colfn(batch):
    reac, prod, catalyst, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        lbs.append(x[2])

    return graph_col_fn(reac), graph_col_fn(prod), torch.FloatTensor(lbs)


class ReactionPredDataset(RAlignDatasetBase):
    def __init__(self, reactions, labels, cls_id, end_id=None):
        super(ReactionPredDataset, self).__init__(reactions)
        self.labels = labels
        self.cls_id = cls_id
        self.end_id = end_id

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)

        tlabel = [self.cls_id] + self.labels[idx]
        if self.end_id is not None:
            tlabel += [self.end_id]
        return reac_mol, prod_mol, tlabel


def gen_fn(batch):
    reac = graph_col_fn([x[0] for x in batch])
    prod = graph_col_fn([x[1] for x in batch])
    return reac, prod, [x[2] for x in batch]


def pred_fn(batch):
    reac = graph_col_fn([x[0] for x in batch])
    prod = graph_col_fn([x[1] for x in batch])
    return reac, prod, torch.LongTensor([x[2] for x in batch])


class ReactionSeqInferenceDataset(RAlignDatasetBase):
    def __init__(self, reactions, labels=None, return_raw=True):
        super(ReactionSeqInferenceDataset, self).__init__(reactions)
        self.labels = labels
        self.return_raw = return_raw

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        out_ans = [reac_mol, prod_mol]
        if self.return_raw:
            out_ans.append(self.reactions[idx])
        if self.labels is not None:
            out_ans.append(self.labels[idx])
        return out_ans


def seq_inf_fn(batch):
    out_ans = [
        graph_col_fn([x[0] for x in batch]),
        graph_col_fn([x[1] for x in batch])
    ]
    if len(batch[0]) > 2:
        out_ans.append([x[2] for x in batch])
    if len(batch[0]) > 3:
        out_ans.append([x[3] for x in batch])
    return out_ans


class JointDataset(RAlignDatasetBase):
    """继承RAlignDatasetBase的双标签数据集（分类+回归）"""
    def __init__(
        self, reactions, is_elementary, barrier, use_local_pe=False
    ):
        super(JointDataset, self).__init__(reactions, use_local_pe=use_local_pe)
        self.cls_label = is_elementary
        self.reg_label = barrier

    def __getitem__(self, idx):
        # 基础信息提取
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        
        # 双标签提取
        cls_label = self.cls_label[idx]  # 分类标签（整数）
        reg_label = self.reg_label[idx]  # 回归标签（浮点数）
        
        return reac_mol, prod_mol, cls_label, reg_label


def joint_colfn(batch):
    """
    处理双标签批次数据，拼接图结构和标签
    输入: batch = [(reac_graph, prod_graph, cls_label, reg_label), ...]
    输出: 拼接后的图、分类标签、回归标签
    """
    reacs, prods, cls_labels, reg_labels = [], [], [], []
    
    # 提取批次中的各个部分
    for x in batch:
        reacs.append(x[0])
        prods.append(x[1])
        cls_labels.append(x[2])
        reg_labels.append(x[3])
    
    return graph_col_fn(reacs), graph_col_fn(prods), \
        torch.tensor(cls_labels, dtype=torch.long), \
        torch.tensor(reg_labels, dtype=torch.float32) # NaN会被保留为torch.nan
        
