"""
Paper-faithful sparsification: SparseGNN (Algorithm 1) and SparseExpand (Algorithm 2)
from *Privacy Amplification by Composite Subsampling*.

This is the current default sparsification mechanism.  It is intentionally
model-agnostic (Assumption 3.2, G(y) = sum_v g0(y_v)): the sparsification
(root sampling + SparseExpand) is decoupled from the base mechanism g0, so a
GNN node classifier and a non-GNN graph-anomaly detector both plug into the
same training engine.

Public API:
    sparse_expand, build_adjacency, RootedSubgraph       (Algorithm 5 / 2)
    train_sparse_gnn                                     (Algorithm 1 engine)
    BaseMechanism, GNNMechanism, AnomalyMechanism        (the base mechanism g0)

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
    resolve_sparsegnn_theorem, sparsegnn_epsilon, sparsegnn_epsilon_schedule,
    sparsegnn_theorem_label,
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
    "resolve_sparsegnn_theorem",
    "sparsegnn_epsilon",
    "sparsegnn_epsilon_schedule",
    "sparsegnn_theorem_label",
    "GNNMechanism",
]
