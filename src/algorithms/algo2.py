"""
Algorithm 2: Nodes-and-Bins, Removing Sinks.

Same bin assignment as Algorithm 1, but edges are only kept when BOTH
endpoints are in the same bin. Produces identical gradients to Algorithm 1
(sinks had no effect on forward pass or loss) but smaller subgraphs.
"""

import torch
from src.algorithms.base import SubgraphAlgorithm


class RemoveSinks(SubgraphAlgorithm):
    def partition(self, edge_index, num_nodes, num_bins, device) -> list:
        bin_assignments = torch.randint(0, num_bins, (num_nodes,), device=device)
        return self._build_subgraphs(bin_assignments, edge_index, num_bins)

    @staticmethod
    def _build_subgraphs(bin_assignments, edge_index, num_bins):
        """Subgraph construction: edges where BOTH endpoints are in bin k."""
        result = []
        for k in range(num_bins):
            bin_mask = (bin_assignments == k)
            edge_mask = bin_mask[edge_index[0]] & bin_mask[edge_index[1]]
            directed_ei = edge_index[:, edge_mask]
            result.append((bin_mask, directed_ei))
        return result

    def name(self) -> str:
        return "remove_sinks"
