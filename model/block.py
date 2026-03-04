import torch

from .layers import RAlignGATBlock
from utils.ogb_utils_features import AtomEncoder, BondEncoder


class RAlignEncoder(torch.nn.Module):
    def __init__(
        self, n_layer, emb_dim, heads, edge_dim,
        dropout=0.1, negative_slope=0.2, update_last_edge=False,
        fusion_mode='legacy'
    ):
        super(RAlignEncoder, self).__init__()
        self.n_layers = n_layer
        self.layers = torch.nn.ModuleList()
        for i in range(n_layer):
            update_edge = (i < n_layer - 1) or update_last_edge
            self.layers.append(RAlignGATBlock(
                emb_dim=emb_dim, heads=heads, edge_dim=edge_dim,
                negative_slope=negative_slope, dropout=dropout,
                edge_update=update_edge, fusion_mode=fusion_mode
            ))

        self.update_last_edge = update_last_edge
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, reac_graph, prod_graph, **_unused):
        reac_x = self.atom_encoder(reac_graph.x)
        prod_x = self.atom_encoder(prod_graph.x)
        reac_e = self.bond_encoder(reac_graph.edge_attr)
        prod_e = self.bond_encoder(prod_graph.edge_attr)
        for i in range(self.n_layers):
            reac_x, prod_x, reac_e, prod_e = self.layers[i](
                reac_x=reac_x,
                reac_e=reac_e,
                reac_eidx=reac_graph.edge_index,
                shared_mask=reac_graph.is_prod,
                prod_x=prod_x,
                prod_e=prod_e,
                prod_eidx=prod_graph.edge_index
            )

        return reac_x, prod_x, reac_e, prod_e
