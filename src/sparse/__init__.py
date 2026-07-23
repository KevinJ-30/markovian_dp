"""
Paper-faithful sparsification: SparseGNN (Algorithm 1) and SparseExpand (Algorithm 2)
from *Privacy Amplification by Composite Subsampling*.

This is the current default sparsification mechanism.  It is intentionally
model-agnostic (Assumption 3.2, G(y) = sum_v g0(y_v)): the sparsification
(root sampling + SparseExpand) is decoupled from the base mechanism g0, so a
GNN node classifier and a non-GNN graph-anomaly detector both plug into the
same training engine.

Public API:
    sparse_expand, build_out_adjacency, RootedSubgraph   (Algorithm 2)
    train_sparse_gnn                                     (Algorithm 1 engine)
    BaseMechanism, GNNMechanism, AnomalyMechanism        (the base mechanism g0)
"""

from .sparse_expand import (
    RootedSubgraph, build_out_adjacency, cap_degrees, max_degrees, sparse_expand,
)
from .sparse_gnn import train_sparse_gnn
from .base_mechanism import BaseMechanism
from .gnn_mechanism import GNNMechanism
from .anomaly_mechanism import AnomalyMechanism

__all__ = [
    "RootedSubgraph",
    "build_out_adjacency",
    "cap_degrees",
    "max_degrees",
    "sparse_expand",
    "train_sparse_gnn",
    "BaseMechanism",
    "GNNMechanism",
    "AnomalyMechanism",
]
