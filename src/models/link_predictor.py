"""
Link-prediction model: GCN encoder + dot-product score head.

The encoder shares the same 2-layer GCNConv structure as SubgraphGCN but
returns raw embeddings (no log-softmax). The score head is a parameter-free
dot product, suitable as a baseline for ogbl-collab.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             add_self_loops=False, normalize=True)

    def forward(self, x, edge_index):
        """Encode nodes -> [N, out_channels] embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def score(self, z, edge_pairs):
        """
        Compute logits for edges.

        Args:
            z: [N, D] node embeddings.
            edge_pairs: [2, E] LongTensor of (src, dst) pairs.

        Returns:
            [E] logits.
        """
        src, dst = edge_pairs[0], edge_pairs[1]
        return (z[src] * z[dst]).sum(dim=-1)
