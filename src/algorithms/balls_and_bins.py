"""
Algorithm 1: Basic Nodes-and-Bins.

Each node is uniformly assigned to a bin. Bin k's subgraph keeps all edges
where the source is in bin k (neighbors in other bins become sinks).
"""

import torch
from src.algorithms.base import SubgraphAlgorithm


class BallsAndBins(SubgraphAlgorithm):
    def partition(self, edge_index, num_nodes, num_bins, device) -> list:
        bin_assignments = torch.randint(0, num_bins, (num_nodes,), device=device)
        return self._build_subgraphs(bin_assignments, edge_index, num_bins)

    @staticmethod
    def _build_subgraphs(bin_assignments, edge_index, num_bins):
        """Shared subgraph construction: edges where source is in bin k."""
        result = []
        for k in range(num_bins):
            bin_mask = (bin_assignments == k)
            edge_mask = bin_mask[edge_index[0]]
            directed_ei = edge_index[:, edge_mask]
            result.append((bin_mask, directed_ei))
        return result

    def name(self) -> str:
        return "balls_and_bins"
