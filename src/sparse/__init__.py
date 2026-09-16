"""
Paper-faithful sparsification: SparseGNN (Algorithm 1) and SparseExpand (Algorithm 2)
from *Privacy Amplification by Composite Subsampling*.

This is the current default sparsification mechanism. Root sampling and
SparseExpand are decoupled from the concrete node-classification mechanism
through the BaseMechanism interface.

Public API:
    sparse_expand, build_adjacency, RootedSubgraph       (Algorithm 5 / 2)
    train_sparse_gnn                                     (Algorithm 1 engine)
    BaseMechanism, GNNMechanism                          (the base mechanism g0)

Expansion defaults to direction='in' (Algorithm 5, manuscript v35 Section 6):
subgraphs grow along INCOMING edges so that messages flow toward the root.
"""

from .sparse_expand import (
    RootedSubgraph, SparseAdjacency, build_adjacency, build_out_adjacency,
    cap_degrees, cap_degrees_undirected, edge_set_is_symmetric, max_degrees,
    sparse_expand,
)
from .sparse_gnn import train_sparse_gnn, train_sparse_gnn_with_budget
from .base_mechanism import BaseMechanism
from .gnn_mechanism import GNNMechanism
from .accounting import (
    SparseGNNNoiseCalibration, calibrate_sparsegnn_noise,
    mixture_gaussian_pld, sparsegnn_epsilon, sparsegnn_epsilon_schedule,
)

__all__ = [
    "RootedSubgraph",
    "SparseAdjacency",
    "build_adjacency",
    "build_out_adjacency",
    "cap_degrees",
    "cap_degrees_undirected",
    "edge_set_is_symmetric",
    "max_degrees",
    "sparse_expand",
    "train_sparse_gnn",
    "train_sparse_gnn_with_budget",
    "BaseMechanism",
    "SparseGNNNoiseCalibration",
    "calibrate_sparsegnn_noise",
    "mixture_gaussian_pld",
    "sparsegnn_epsilon",
    "sparsegnn_epsilon_schedule",
    "GNNMechanism",
]
