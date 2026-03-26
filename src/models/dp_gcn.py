"""
Simple GNN models for node-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VMapCompatibleGCN(nn.Module):
    """
    GCN layer that's compatible with vmap.

    Uses out-of-place operations (no scatter_add_) for vmap compatibility.
    Implements GCN message passing using gather and matmul operations.
    """

    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass using gather-based message passing (vmap-compatible).

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes] (unused, kept for compatibility)

        Returns:
            output: Node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        device = x.device

        if num_edges == 0:
            x = torch.matmul(x, self.weight) + self.bias
            return x

        src, dst = edge_index[0], edge_index[1]

        src_features = x[src]

        src_degree = torch.bincount(src, minlength=num_nodes).float()
        dst_degree = torch.bincount(dst, minlength=num_nodes).float()

        src_degree_sqrt_inv = torch.pow(src_degree + 1e-6, -0.5)
        dst_degree_sqrt_inv = torch.pow(dst_degree + 1e-6, -0.5)

        edge_norm_src = src_degree_sqrt_inv[src]
        edge_norm_dst = dst_degree_sqrt_inv[dst]
        edge_norm = edge_norm_src * edge_norm_dst

        normalized_src = src_features * edge_norm.unsqueeze(-1)

        edge_to_dst = (dst.unsqueeze(1) == torch.arange(num_nodes, device=device).unsqueeze(0)).float()

        aggregated = torch.einsum('ef,en->nf', normalized_src, edge_to_dst)

        if self.add_self_loops:
            total_degree = src_degree + dst_degree
            self_loop_norm = torch.pow(total_degree + 1e-6, -1.0)
            aggregated = aggregated + x * self_loop_norm.unsqueeze(-1)

        x = torch.matmul(aggregated, self.weight)
        x = x + self.bias

        return x


class NodeGCN(nn.Module):
    """
    Simple 2-layer GCN for node classification.

    Designed for node-level privacy with NeighborLoader.
    Uses vmap-compatible GCN layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = VMapCompatibleGCN(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = VMapCompatibleGCN(hidden_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, batch)
        return F.log_softmax(x, dim=1)
