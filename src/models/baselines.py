"""Neural networks used by the partitioned baseline trainers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GraphSAGE(nn.Module):
    """Mean-aggregating GraphSAGE without a PyG runtime dependency."""

    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.self_layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.neighbor_layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    @staticmethod
    def _mean_neighbors(x: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        sums = torch.zeros_like(x)
        sums.index_add_(0, target, x[source])
        degree = torch.bincount(target, minlength=x.size(0)).to(x.dtype).clamp_min_(1)
        return sums / degree[:, None]

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        if edge_index is None:
            raise ValueError("GraphSAGE requires edge_index")
        for index, (self_layer, neighbor_layer) in enumerate(zip(self.self_layers, self.neighbor_layers)):
            x = self_layer(x) + neighbor_layer(self._mean_neighbors(x, edge_index))
            if index < len(self.self_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DPARMLP(nn.Module):
    """The released DPAR ``W1 ... Wo`` MLP, expressed with torch modules."""

    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 2:
            raise ValueError("DPAR requires at least two MLP layers")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))
        return self.layers[-1](F.dropout(x, p=self.dropout, training=self.training))


class _OneHopGCN(nn.Module):
    """The DP-GNN one-hop GCN convention: receivers send to senders."""

    def __init__(self, inputs: int, hidden: int, classes: int):
        super().__init__()
        self.encoder = nn.Linear(inputs, hidden)
        self.core = nn.Linear(hidden, hidden)
        self.decoder = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.encoder(x))
        aggregated = torch.zeros_like(x)
        if edge_index.numel():
            senders, receivers = edge_index
            aggregated.index_add_(0, senders, x[receivers] * edge_weight[:, None])
        x = aggregated + torch.tanh(self.core(aggregated))
        return self.decoder(x)
