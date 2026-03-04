import torch

from .GATconv import SelfLoopGATConv
from .shared import SparseEdgeUpdateLayer, FiLM


class RAlingLayer(torch.nn.Module):
    def __init__(self, dim, dropout=0):
        super(RAlingLayer, self).__init__()
        self.comm_lin = torch.nn.Sequential(
            torch.nn.Linear(dim + dim, dim + dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim + dim, dim + dim)
        )
        self.dim = dim

    def forward(self, x_prod, x_reac, reac_mask):
        if not torch.all(reac_mask).item():
            raise ValueError(
                "fusion_mode='legacy' requires aligned reactant/product atoms"
            )
        if x_prod.shape != x_reac.shape:
            raise ValueError("Legacy fusion requires matched reactant/product shapes")

        shared_result = self.comm_lin(torch.cat([x_prod, x_reac], dim=-1))
        new_prod = shared_result[:, :self.dim]
        new_reac = shared_result[:, self.dim:]
        return new_prod, new_reac


class SymmetricFiLMLayer(torch.nn.Module):
    def __init__(self, dim):
        super(SymmetricFiLMLayer, self).__init__()
        self.film = FiLM(dim, dim)

    def forward(self, x_prod, x_reac, reac_mask):
        if not torch.all(reac_mask).item():
            raise ValueError(
                "fusion_mode='film' requires aligned reactant/product atoms"
            )
        if x_prod.shape != x_reac.shape:
            raise ValueError("FiLM fusion requires matched reactant/product shapes")
        return self.film(x_prod, x_reac), self.film(x_reac, x_prod)


class RAlignGATBlock(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, edge_dim, dropout=0.1,
        negative_slope=0.2, edge_update=True, fusion_mode='legacy'
    ):
        super(RAlignGATBlock, self).__init__()
        if fusion_mode not in ['legacy', 'film']:
            raise ValueError(f'Invalid fusion mode {fusion_mode}')
        if emb_dim % heads != 0:
            raise ValueError('emb_dim must be divisible by heads')

        self.reac_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.prod_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )

        self.edge_update = edge_update
        self.fusion_mode = fusion_mode
        if self.fusion_mode == 'legacy':
            self.fusion_layer = RAlingLayer(emb_dim, dropout)
        else:
            self.fusion_layer = SymmetricFiLMLayer(emb_dim)

        self.reac_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.reac_fusion_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_fusion_ln = torch.nn.LayerNorm(emb_dim)

        if self.edge_update:
            self.reac_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.prod_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.reac_edge_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_edge_ln = torch.nn.LayerNorm(emb_dim)

        self.drop_f = torch.nn.Dropout(dropout)

    def forward(
        self, reac_x, reac_e, reac_eidx, shared_mask,
        prod_x, prod_e, prod_eidx
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
        reac_x = self.reac_fusion_ln(self.drop_f(reac_u) + reac_x)

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
