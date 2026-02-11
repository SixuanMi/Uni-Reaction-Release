import torch
from .GATconv import SelfLoopGATConv
from .shared import SparseEdgeUpdateLayer
from ..utils import graph2batch


class RAlingLayer(torch.nn.Module):
    def __init__(self, dim, dropout=0, use_lg_lin=True):
        super(RAlingLayer, self).__init__()
        self.comm_lin = torch.nn.Sequential(
            torch.nn.Linear(dim + dim, dim + dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim + dim, dim + dim)
        )

        self.use_lg_lin = use_lg_lin
        if self.use_lg_lin:
            self.lg_lin = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(dim, dim)
            )
        else:
            # keep reactant-only atoms unchanged and avoid creating unused params
            self.lg_lin = None
        self.dim = dim

    def forward(self, x_prod, x_reac, reac_mask):
        new_reac = torch.zeros_like(x_reac)
        shared_result = torch.cat([x_prod, x_reac[reac_mask]], dim=-1)
        shared_result = self.comm_lin(shared_result)
        new_prod = shared_result[:, :self.dim]
        new_reac[reac_mask] = shared_result[:, self.dim:]

        if torch.any(~reac_mask).item():
            if self.use_lg_lin:
                new_reac[~reac_mask] = self.lg_lin(x_reac[~reac_mask])
            else:
                new_reac[~reac_mask] = x_reac[~reac_mask]

        return new_prod, new_reac


class RAlignGATBlock(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, edge_dim, reac_batch_infos={}, reac_num_keys={},
        prod_batch_infos={}, prod_num_keys={}, dropout=0.1,
        negative_slope=0.2, edge_update=True, use_lg_lin=True
    ):
        super(RAlignGATBlock, self).__init__()
        self.reac_batch_adapter = torch.nn.ModuleDict({
            k: torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=v['heads'], dropout=dropout,
                batch_first=True, kdim=v['dim'], vdim=v['dim']
            ) for k, v in reac_batch_infos.items()
        })
        self.prod_batch_adapter = torch.nn.ModuleDict({
            k: torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=v['heads'], dropout=dropout,
                batch_first=True, kdim=v['dim'], vdim=v['dim']
            ) for k, v in prod_batch_infos.items()
        })
        self.reac_num_adapter = torch.nn.ModuleDict({
            k: torch.nn.ModuleDict(
                {
                    'beta': torch.nn.Linear(v, emb_dim),
                    'gamma': torch.nn.Linear(v, emb_dim)
                }
            ) for k, v in reac_num_keys.items()
        })
        self.prod_num_adapter = torch.nn.ModuleDict({
            k: torch.nn.ModuleDict(
                {
                    'beta': torch.nn.Linear(v, emb_dim),
                    'gamma': torch.nn.Linear(v, emb_dim)
                }
            ) for k, v in prod_num_keys.items()
        })
        assert emb_dim % heads == 0, 'emb_dim must be divisible by heads'
        self.reac_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.prod_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )

        self.edge_update = edge_update

        self.fusion_layer = RAlingLayer(
            emb_dim, dropout, use_lg_lin=use_lg_lin
        )

        self.reac_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.reac_fusion_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_fusion_ln = torch.nn.LayerNorm(emb_dim)

        if self.edge_update:
            self.reac_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.prod_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.reac_edge_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_edge_ln = torch.nn.LayerNorm(emb_dim)

        self.reac_cond_ln = None if len(reac_batch_infos) == 0\
            else torch.nn.LayerNorm(emb_dim)

        self.prod_cond_ln = None if len(prod_batch_infos) == 0 \
            else torch.nn.LayerNorm(emb_dim)

        self.drop_f = torch.nn.Dropout(dropout)

    def forward(
        self, reac_x, reac_e, reac_eidx, reac_bmask, shared_mask,
        prod_x, prod_e, prod_eidx, prod_bmask,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        reac_conv = self.reac_mpnn(
            x=reac_x, edge_attr=reac_e, edge_index=reac_eidx
        )

        prod_conv = self.prod_mpnn(
            x=prod_x, edge_attr=prod_e, edge_index=prod_eidx
        )

        prod_x = self.prod_mpnn_ln(self.drop_f(prod_conv) + prod_x)
        reac_x = self.reac_mpnn_ln(self.drop_f(reac_conv) + reac_x)

        prod_u, reac_u = self.fusion_layer(
            x_prod=prod_x, x_reac=reac_x, reac_mask=shared_mask
        )

        prod_x = self.prod_fusion_ln(self.drop_f(prod_u) + prod_x)
        reac_x = self.reac_fusion_ln(reac_x + self.drop_f(reac_u))

        reac_x = graph2batch(reac_x, reac_bmask)
        prod_x = graph2batch(prod_x, prod_bmask)

        reac_bias = torch.zeros_like(reac_x)
        prod_bias = torch.zeros_like(prod_x)

        for k, v in self.reac_batch_adapter.items():
            this_info = reac_batched_condition[k]
            bias, _w = v(
                query=reac_x, key=this_info['embedding'],
                value=this_info['embedding'],
                key_padding_mask=this_info.get('padding_mask', None)
            )
            reac_bias += self.drop_f(bias)

        for k, v in self.prod_batch_adapter.items():
            this_info = prod_batched_condition[k]
            bias, _w = v(
                query=prod_x, key=this_info['embedding'],
                value=this_info['embedding'],
                key_padding_mask=this_info.get('padding_mask', None)
            )
            prod_bias += self.drop_f(bias)

        if self.prod_cond_ln is not None:
            prod_x = self.prod_cond_ln(prod_x + prod_bias)
        if self.reac_cond_ln is not None:
            reac_x = self.reac_cond_ln(reac_bias + reac_x)

        reac_bias = torch.zeros_like(reac_x)
        prod_bias = torch.zeros_like(prod_x)

        for k, v in self.reac_num_adapter.items():
            gamma = v['gamma'](reac_num_conditions[k])
            beta = v['beta'](reac_num_conditions[k])
            reac_bias += gamma * reac_x + beta

        for k, v in self.prod_num_adapter.items():
            gamma = v['gamma'](prod_num_conditions[k])
            beta = v['beta'](prod_num_conditions[k])
            prod_bias += gamma * prod_x + beta

        prod_x = (prod_x + prod_bias)[prod_bmask]
        reac_x = (reac_x + reac_bias)[reac_bmask]

        if self.edge_update:
            reac_e_u = self.reac_ue(
                edge_feats=reac_e, node_feats=reac_x, edge_index=reac_eidx
            )
            prod_e_u = self.prod_ue(
                edge_feats=prod_e, node_feats=prod_x, edge_index=prod_eidx
            )
            reac_e = self.reac_edge_ln(reac_e + self.drop_f(reac_e_u))
            prod_e = self.prod_edge_ln(prod_e + self.drop_f(prod_e_u))

        return reac_x, prod_x, reac_e, prod_e
