"""
MLP baseline for node classification.
"""

import torch.nn as nn
import torch.nn.functional as F


class NodeMLP(nn.Module):
    """2-layer MLP that ignores graph structure."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        """Accept edge_index for API compatibility but ignore it."""
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
