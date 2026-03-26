"""
Abstract base class for subgraph partitioning algorithms.
"""

from abc import ABC, abstractmethod
import torch


class SubgraphAlgorithm(ABC):
    @abstractmethod
    def partition(self, edge_index, num_nodes, num_bins, device) -> list:
        """
        Partition the graph into subgraphs.

        Args:
            edge_index: [2, num_edges] tensor.
            num_nodes: Total number of nodes.
            num_bins: Number of partitions.
            device: Torch device.

        Returns:
            List of (bin_mask, directed_edge_index) tuples, one per bin.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...
