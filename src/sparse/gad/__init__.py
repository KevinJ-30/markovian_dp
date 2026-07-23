"""
Graph anomaly detection on sparsified graphs (GADBench, arXiv:2306.12251).

Implements XGB-Graph: a tree ensemble on parameter-free multi-hop neighbor-aggregated
node features (GADBench Section 3.1). This is NOT a GNN and NOT gradient training; the
graph enters only through the neighbor-aggregation step, so edge sparsification degrades
the features and we can measure the resulting utility drop.

Public API:
    aggregate_features, sparsify_edges_bernoulli, aggregate_features_expand
    XGBGraphDetector
    auroc, auprc, rec_at_k
"""

from .neighbor_aggregation import (
    aggregate_features,
    aggregate_features_expand,
    sparsify_edges_bernoulli,
)
from .xgb_graph import XGBGraphDetector
from .metrics import auroc, auprc, rec_at_k

__all__ = [
    "aggregate_features",
    "aggregate_features_expand",
    "sparsify_edges_bernoulli",
    "XGBGraphDetector",
    "auroc",
    "auprc",
    "rec_at_k",
]
