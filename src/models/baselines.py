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
    def _mean_neighbors(
        x: Tensor,
        edge_index: Tensor,
        num_targets: int | None = None,
    ) -> Tensor:
        source, target = edge_index
        target_count = x.size(0) if num_targets is None else num_targets
        sums = x.new_zeros((target_count, x.size(1)))
        sums.index_add_(0, target, x[source])
        degree = torch.bincount(target, minlength=target_count).to(
            x.dtype
        ).clamp_min_(1)
        return sums / degree[:, None]

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        if edge_index is None:
            raise ValueError("GraphSAGE requires edge_index")
        for index, (self_layer, neighbor_layer) in enumerate(
            zip(self.self_layers, self.neighbor_layers)
        ):
            x = self_layer(x) + neighbor_layer(
                self._mean_neighbors(x, edge_index)
            )
            if index < len(self.self_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_sampled(
        self,
        x: Tensor,
        edge_index: Tensor,
        num_sampled_nodes: list[int],
        num_sampled_edges: list[int],
        *,
        hierarchical: bool,
    ) -> Tensor:
        """Return seed logits from a BFS-ordered sampled neighborhood.

        ``hierarchical=False`` applies every layer to the complete sampled
        subgraph. ``hierarchical=True`` progressively drops the deepest hop,
        preserving identical seed logits while avoiding unused activations.
        """
        layer_count = len(self.self_layers)
        if len(num_sampled_nodes) != layer_count + 1:
            raise ValueError("sampled node counts must contain one entry per hop")
        if len(num_sampled_edges) != layer_count:
            raise ValueError("sampled edge counts must contain one entry per layer")
        if not hierarchical:
            return self.forward(x, edge_index)[:num_sampled_nodes[0]]

        for index, (self_layer, neighbor_layer) in enumerate(
            zip(self.self_layers, self.neighbor_layers)
        ):
            retained_hops = layer_count - index
            target_count = sum(num_sampled_nodes[:retained_hops])
            edge_count = sum(num_sampled_edges[:retained_hops])
            layer_edges = edge_index[:, :edge_count]
            x = self_layer(x[:target_count]) + neighbor_layer(
                self._mean_neighbors(x, layer_edges, target_count)
            )
            if index < layer_count - 1:
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

    def __init__(self, inputs: int, hidden: int, classes: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Linear(inputs, hidden)
        self.core = nn.Linear(hidden, hidden)
        self.decoder = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.encoder(x))
        aggregated = torch.zeros_like(x)
        if edge_index.numel():
            senders, receivers = edge_index
            aggregated.index_add_(0, senders, x[receivers] * edge_weight[:, None])
        x = aggregated + torch.tanh(self.core(aggregated))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.decoder(x)


class _PaddedOneHopGCN(nn.Module):
    """Root-only view of one-hop stars sharing the full-graph model's layers."""

    def __init__(self, model: _OneHopGCN):
        super().__init__()
        self.encoder = model.encoder
        self.core = model.core
        self.decoder = model.decoder
        self.dropout = model.dropout

    def forward(
        self, features: torch.Tensor, node_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = torch.tanh(self.encoder(features))
        encoded = encoded.masked_fill(~node_mask.unsqueeze(-1), 0)
        averaged = encoded.sum(dim=1) / node_mask.sum(dim=1, keepdim=True)
        hidden = averaged + torch.tanh(self.core(averaged))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.decoder(hidden)


class _OneHopGraphSAGE(nn.Module):
    """One-hop mean GraphSAGE with separate root and neighbour transforms."""

    def __init__(self, inputs: int, hidden: int, classes: int, dropout: float = 0.5):
        super().__init__()
        self.root_encoder = nn.Linear(inputs, hidden)
        self.neighbour_encoder = nn.Linear(inputs, hidden, bias=False)
        self.decoder = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        del edge_weight
        neighbours = torch.zeros_like(x)
        if edge_index.numel():
            senders, receivers = edge_index
            nonself = senders != receivers
            senders, receivers = senders[nonself], receivers[nonself]
            if senders.numel():
                degree = torch.bincount(senders, minlength=x.size(0)).to(
                    dtype=x.dtype).clamp_min_(1.0)
                neighbours.index_add_(
                    0, senders, x[receivers] / degree[senders, None])
        hidden = torch.tanh(
            self.root_encoder(x) + self.neighbour_encoder(neighbours))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.decoder(hidden)


class _PaddedOneHopGraphSAGE(nn.Module):
    """Root-only padded-star view sharing a one-hop GraphSAGE model."""

    def __init__(self, model: _OneHopGraphSAGE):
        super().__init__()
        self.root_encoder = model.root_encoder
        self.neighbour_encoder = model.neighbour_encoder
        self.decoder = model.decoder
        self.dropout = model.dropout

    def forward(
        self, features: torch.Tensor, node_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbour_mask = node_mask.clone()
        neighbour_mask[:, 0] = False
        neighbours = features.masked_fill(
            ~neighbour_mask.unsqueeze(-1), 0).sum(dim=1)
        neighbours = neighbours / neighbour_mask.sum(
            dim=1, keepdim=True).clamp_min_(1)
        hidden = torch.tanh(
            self.root_encoder(features[:, 0])
            + self.neighbour_encoder(neighbours))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.decoder(hidden)
