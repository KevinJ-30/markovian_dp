"""
SubgraphGCN model using PyG's standard GCNConv.
No vmap constraints — used for utility validation only.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SubgraphGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
