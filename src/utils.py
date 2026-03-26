"""
Shared utilities.
"""

import torch


def compute_full_degrees(edge_index, num_nodes) -> torch.Tensor:
    """Count source occurrences in edge_index (= undirected degree for symmetric graphs)."""
    src = edge_index[0]
    return torch.bincount(src, minlength=num_nodes).float()
