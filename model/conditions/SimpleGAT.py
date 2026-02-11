import torch
from utils.ogb_utils_features import AtomEncoder, BondEncoder
from ..layers import SelfLoopGATConv, SparseEdgeUpdateLayer


class SimpleCondGAT(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4,
        embedding_dim: int = 64, dropout: float = 0.7,
        negative_slope: float = 0.2
    ):
        super(SimpleCondGAT, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.drop_f = torch.nn.Dropout(dropout)
        assert embedding_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(SelfLoopGATConv(
                in_channels=embedding_dim, heads=num_heads,
                out_channels=embedding_dim // num_heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.lns.append(torch.nn.LayerNorm(embedding_dim))
            if layer < self.num_layers - 1:
                self.edge_update.append(SparseEdgeUpdateLayer(
                    embedding_dim, embedding_dim, dropout=dropout
                ))
        self.atom_encoder = AtomEncoder(embedding_dim)
        self.bond_encoder = BondEncoder(embedding_dim)

    def forward(self, G) -> torch.Tensor:
        node_feats = self.atom_encoder(G.x)
        edge_feats = self.bond_encoder(G.edge_attr)
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=G.edge_index,
            ))
            node_feats = self.drop_f(torch.relu(conv_res)) + node_feats

            if layer < self.num_layers - 1:
                edge_res = self.lns[layer](self.edge_update[layer](
                    edge_feats=edge_feats, node_feats=node_feats,
                    edge_index=G.edge_index
                ))
                edge_feats = self.drop_f(torch.relu(edge_res)) + edge_feats

        return node_feats
