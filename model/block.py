import torch

from .layers import RAlignGATBlock, DualGATBlock, TransDecLayer
from .utils import graph2batch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 2000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        token_len = token_embedding.shape[1]
        return self.dropout(token_embedding + self.pos_embedding[:token_len])


class TranDec(torch.nn.Module):
    def __init__(
        self, n_layers, emb_dim, heads, dropout=0,
        kvdim=None, dim_ff=None
    ):
        super(TranDec, self).__init__()
        self.layers = torch.nn.ModuleList([
            TransDecLayer(emb_dim, heads, dropout, kvdim, dim_ff)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        for i in range(self.n_layers):
            tgt = self.layers[i](
                tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return tgt


class RAlignEncoder(torch.nn.Module):
    def __init__(
        self, n_layer, emb_dim, heads, edge_dim, reac_batch_infos={},
        reac_num_keys={}, prod_batch_infos={}, prod_num_keys={},
        dropout=0.1, negative_slope=0.2, update_last_edge=False,
        use_lg_lin=True
    ):
        super(RAlignEncoder, self).__init__()
        self.n_layers = n_layer
        self.layers = torch.nn.ModuleList()
        for i in range(n_layer):
            update_edge = (i < n_layer - 1) or update_last_edge
            self.layers.append(RAlignGATBlock(
                emb_dim=emb_dim, heads=heads, edge_dim=edge_dim,
                reac_batch_infos=reac_batch_infos, reac_num_keys=reac_num_keys,
                prod_batch_infos=prod_batch_infos, prod_num_keys=prod_num_keys,
                negative_slope=negative_slope, dropout=dropout,
                edge_update=update_edge, use_lg_lin=use_lg_lin
            ))

        self.update_last_edge = update_last_edge

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(
        self, reac_graph, prod_graph,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        reac_x = self.atom_encoder(reac_graph.x)
        prod_x = self.atom_encoder(prod_graph.x)
        reac_e = self.bond_encoder(reac_graph.edge_attr)
        prod_e = self.bond_encoder(prod_graph.edge_attr)
        for i in range(self.n_layers):
            reac_x, prod_x, reac_e, prod_e = self.layers[i](
                reac_x=reac_x, reac_e=reac_e,
                reac_eidx=reac_graph.edge_index,
                reac_bmask=reac_graph.batch_mask,
                shared_mask=reac_graph.is_prod,
                reac_batched_condition=reac_batched_condition,
                reac_num_conditions=reac_num_conditions,
                prod_x=prod_x, prod_e=prod_e,
                prod_eidx=prod_graph.edge_index,
                prod_bmask=prod_graph.batch_mask,
                prod_batched_condition=prod_batched_condition,
                prod_num_conditions=prod_num_conditions
            )

        return reac_x, prod_x, reac_e, prod_e


class DualGATEncoder(torch.nn.Module):
    def __init__(
        self, n_layer, emb_dim, heads, edge_dim, reac_batch_infos={},
        reac_num_keys={}, prod_batch_infos={}, prod_num_keys={},
        dropout=0.1, negative_slope=0.2, update_last_edge=False
    ):
        super(DualGATEncoder, self).__init__()
        self.n_layers = n_layer
        self.layers = torch.nn.ModuleList()
        for i in range(n_layer):
            update_edge = (i < n_layer - 1) or update_last_edge
            self.layers.append(DualGATBlock(
                emb_dim=emb_dim, heads=heads, edge_dim=edge_dim,
                reac_batch_infos=reac_batch_infos, reac_num_keys=reac_num_keys,
                prod_batch_infos=prod_batch_infos, prod_num_keys=prod_num_keys,
                negative_slope=negative_slope, dropout=dropout,
                edge_update=update_edge
            ))

        self.update_last_edge = update_last_edge

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(
        self, reac_graph, prod_graph,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        reac_x = self.atom_encoder(reac_graph.x)
        prod_x = self.atom_encoder(prod_graph.x)
        reac_e = self.bond_encoder(reac_graph.edge_attr)
        prod_e = self.bond_encoder(prod_graph.edge_attr)
        for i in range(self.n_layers):
            reac_x, prod_x, reac_e, prod_e = self.layers[i](
                reac_x=reac_x, reac_e=reac_e,
                reac_eidx=reac_graph.edge_index,
                reac_bmask=reac_graph.batch_mask,
                shared_mask=reac_graph.is_prod,
                reac_batched_condition=reac_batched_condition,
                reac_num_conditions=reac_num_conditions,
                prod_x=prod_x, prod_e=prod_e,
                prod_eidx=prod_graph.edge_index,
                prod_bmask=prod_graph.batch_mask,
                prod_batched_condition=prod_batched_condition,
                prod_num_conditions=prod_num_conditions
            )

        return reac_x, prod_x, reac_e, prod_e
