"""
Simple GNN models for node-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NodeGCN(nn.Module):
    """
    Simple 2-layer GCN for node classification.
    
    Designed for node-level privacy with NeighborLoader.
    Returns node-level predictions (one per node).
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # Disable add_self_loops since we pre-add them in a vmap-compatible way
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False)
    
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
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Return node-level predictions (no pooling)
        return F.log_softmax(x, dim=1)