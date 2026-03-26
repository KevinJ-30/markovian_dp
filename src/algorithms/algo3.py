"""
Algorithm 3: Nodes-and-Bins, Removing Sinks + Subsampling.

Same as Algorithm 2, but each node is first dropped to a dummy bin with
probability p_perp. Nodes in the dummy bin are excluded from ALL bins,
giving privacy amplification by subsampling.
"""

import torch
from src.algorithms.base import SubgraphAlgorithm


class RemoveSinksSubsampled(SubgraphAlgorithm):
    def __init__(self, subsample_prob: float = 0.0):
        """
        Args:
            subsample_prob: Probability p_perp that a node is assigned to the
                dummy bin and excluded from all computation.
        """
        self.subsample_prob = subsample_prob

    def partition(self, edge_index, num_nodes, num_bins, device) -> list:
        # Step 1: assign bins uniformly
        bin_assignments = torch.randint(0, num_bins, (num_nodes,), device=device)

        # Step 2: with probability p_perp, reassign to dummy bin (-1)
        if self.subsample_prob > 0:
            drop_mask = torch.bernoulli(
                torch.full((num_nodes,), self.subsample_prob, device=device)
            ).bool()
            bin_assignments[drop_mask] = -1  # dummy bin

        # Step 3: build subgraphs (same as Algo 2 — both endpoints must match bin k)
        result = []
        for k in range(num_bins):
            bin_mask = (bin_assignments == k)
            edge_mask = bin_mask[edge_index[0]] & bin_mask[edge_index[1]]
            directed_ei = edge_index[:, edge_mask]
            result.append((bin_mask, directed_ei))
        return result

    def name(self) -> str:
        return "remove_sinks_subsampled"
