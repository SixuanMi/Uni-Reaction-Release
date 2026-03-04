import torch

from .layers import DotMhAttn
from .utils import graph2batch


class JointModel(torch.nn.Module):
    def __init__(self, encoder, dim, heads, dropout=0.1):
        super(JointModel, self).__init__()
        self.encoder = encoder
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, 2)
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 1),
            torch.nn.Softplus()
        )
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )
        self.xln = torch.nn.LayerNorm(dim)

    def forward(self, reac_graph, prod_graph, _unused_conditions=None, cross_mask=None):
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, prod_graph=prod_graph
        )

        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        reac_mask = torch.logical_not(reac_graph.batch_mask)
        prod_mask = torch.logical_not(prod_graph.batch_mask)

        shared_query = self.pool_keys.repeat(x_reac.shape[0], 1, 1)
        reac_cross_mask, prod_cross_mask = None, None
        if cross_mask is not None and cross_mask.ndim == 4:
            mask_source = cross_mask[:, :1] if cross_mask.shape[1] != 1 else cross_mask
            total_len = x_reac.shape[1] + x_prod.shape[1]
            if mask_source.shape[2] != total_len:
                raise ValueError(
                    f'cross_mask key dim {mask_source.shape[2]} does not match '
                    f'combined sequence length {total_len}'
                )
            reac_cross_mask = mask_source[:, :, :x_reac.shape[1], :]
            prod_cross_mask = mask_source[:, :, x_reac.shape[1]:, :]

        r_pool, _ = self.pooler(
            query=shared_query, key=x_reac, value=x_reac,
            key_padding_mask=reac_mask, attn_mask=reac_cross_mask
        )
        p_pool, _ = self.pooler(
            query=shared_query, key=x_prod, value=x_prod,
            key_padding_mask=prod_mask, attn_mask=prod_cross_mask
        )
        r_pool = self.xln(r_pool.squeeze(dim=1))
        p_pool = self.xln(p_pool.squeeze(dim=1))
        d_pool = p_pool - r_pool

        cls_feat = torch.cat([r_pool + p_pool, torch.abs(d_pool)], dim=-1)
        reg_feat = torch.cat([r_pool, p_pool, d_pool], dim=-1)
        cls_out = self.cls_head(cls_feat)
        reg_out = self.reg_head(reg_feat)
        return cls_out, reg_out
