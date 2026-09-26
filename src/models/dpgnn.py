"""Root-independent multi-hop DP-GNN views with shared inference parameters."""

import torch
import torch.nn.functional as F
from torch import nn


class _DPGNNLayer(nn.Module):
    def __init__(self, inputs: int, hidden: int, architecture: str):
        super().__init__()
        self.architecture = architecture
        if architecture == "graphsage":
            self.root_encoder = nn.Linear(inputs, hidden)
            self.neighbour_encoder = nn.Linear(inputs, hidden, bias=False)
        elif architecture == "gin":
            self.mlp = nn.Sequential(
                nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        else:
            raise ValueError("multi-hop DP-GNN supports graphsage and gin")

    def forward(self, features: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
        if self.architecture == "graphsage":
            return torch.tanh(
                self.root_encoder(features) + self.neighbour_encoder(neighbours))
        return F.relu(self.mlp(features + neighbours))


class _MultiHopDPGNN(nn.Module):
    """One real neighborhood aggregation per layer, using DP-GNN arc orientation."""

    def __init__(self, inputs: int, hidden: int, classes: int, *, radius: int,
                 architecture: str, dropout: float):
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be positive")
        self.layers = nn.ModuleList([
            _DPGNNLayer(inputs if hop == 0 else hidden, hidden, architecture)
            for hop in range(radius)
        ])
        self.decoder = nn.Linear(hidden, classes)
        self.architecture = architecture
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        del edge_weight
        senders, receivers = edge_index[:, edge_index[0] != edge_index[1]]
        degree = torch.bincount(senders, minlength=x.size(0)).to(x.dtype).clamp_min_(1)
        for layer in self.layers:
            neighbours = torch.zeros_like(x)
            neighbours.index_add_(0, senders, x[receivers])
            if self.architecture == "graphsage":
                neighbours = neighbours / degree[:, None]
            x = layer(x, neighbours)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.decoder(x)


class _PaddedMultiHopDPGNN(nn.Module):
    """Keep roots as axis zero through every Linear for Opacus global clipping.

    Nodes shared between root neighborhoods are deliberately separate samples.
    GradSampleModule sums the node axis within a root, never across roots.
    """

    def __init__(self, model: _MultiHopDPGNN):
        super().__init__()
        self.layers = model.layers
        self.decoder = model.decoder
        self.architecture = model.architecture
        self.dropout = model.dropout

    def forward(self, features: torch.Tensor, node_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        senders, receivers = edge_index.unbind(dim=1)
        degree = features.new_zeros(node_mask.shape)
        degree.scatter_add_(1, senders, edge_mask.to(features.dtype))
        degree.clamp_min_(1)
        x = features.masked_fill(~node_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            messages = x.gather(1, receivers.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            messages = messages.masked_fill(~edge_mask.unsqueeze(-1), 0)
            neighbours = torch.zeros_like(x)
            neighbours.scatter_add_(
                1, senders.unsqueeze(-1).expand_as(messages), messages)
            if self.architecture == "graphsage":
                neighbours = neighbours / degree.unsqueeze(-1)
            x = layer(x, neighbours)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.masked_fill(~node_mask.unsqueeze(-1), 0)
        return self.decoder(x[:, 0])
