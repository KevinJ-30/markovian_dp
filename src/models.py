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

        Implements GCN without scatter_add_ by using gather and manual aggregation.

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
            # No edges, just apply linear transformation
            x = torch.matmul(x, self.weight) + self.bias
            return x

        src, dst = edge_index[0], edge_index[1]

        # Gather source node features for each edge
        src_features = x[src]  # [num_edges, in_channels]

        # Compute degrees for normalization (vmap-compatible, no in-place operations)
        # Use bincount which is out-of-place
        src_degree = torch.bincount(src, minlength=num_nodes).float()
        dst_degree = torch.bincount(dst, minlength=num_nodes).float()

        # For GCN symmetric normalization: D^(-1/2) A D^(-1/2)
        # Compute normalization factors: 1 / sqrt(deg_src * deg_dst)
        src_degree_sqrt_inv = torch.pow(src_degree + 1e-6, -0.5)
        dst_degree_sqrt_inv = torch.pow(dst_degree + 1e-6, -0.5)

        # Get normalization factors for each edge
        edge_norm_src = src_degree_sqrt_inv[src]  # [num_edges]
        edge_norm_dst = dst_degree_sqrt_inv[dst]  # [num_edges]
        edge_norm = edge_norm_src * edge_norm_dst  # [num_edges]

        # Normalize source features
        normalized_src = src_features * edge_norm.unsqueeze(-1)  # [num_edges, in_channels]

        # Aggregate messages at destination nodes using einsum (fully functional, vmap-compatible)
        # Create indicator matrix: edge_to_dst[edge_i, node_j] = 1 if edge i points to node j
        # This is done using comparison operations (no scatter)

        # Create a matrix where entry [i, j] is 1 if edge i has destination j
        edge_to_dst = (dst.unsqueeze(1) == torch.arange(num_nodes, device=device).unsqueeze(0)).float()
        # edge_to_dst shape: [num_edges, num_nodes]

        # Aggregate using einsum: for each node, sum the normalized features of edges pointing to it
        # einsum('ef,en->nf', normalized_src, edge_to_dst)
        # normalized_src: [num_edges, features]
        # edge_to_dst: [num_edges, num_nodes]
        # Result: [num_nodes, features]
        aggregated = torch.einsum('ef,en->nf', normalized_src, edge_to_dst)  # [num_nodes, in_channels]

        # Add self-loops if requested
        if self.add_self_loops:
            # Self-loop normalization: 1 / degree
            total_degree = src_degree + dst_degree
            self_loop_norm = torch.pow(total_degree + 1e-6, -1.0)
            aggregated = aggregated + x * self_loop_norm.unsqueeze(-1)

        # Apply linear transformation
        x = torch.matmul(aggregated, self.weight)  # [num_nodes, out_channels]
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
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes] (unused for node-level prediction)

        Returns:
            output: Node-level predictions [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index, batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, batch)
        return F.log_softmax(x, dim=1)
