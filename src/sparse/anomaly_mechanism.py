"""
AnomalyMechanism: non-GNN, GRADIENT-BASED base mechanism g0 for graph anomaly detection.

NOTE: The realized (non-DP) graph-anomaly path is the tree ensemble XGB-Graph in
`src/sparse/gad/` (GADBench), which is NOT gradient-based and therefore does not use this
BaseMechanism/train_sparse_gnn engine at all — it reuses only the sparsification primitive
(`sparse_expand`). This module remains a STUB for a possible FUTURE gradient-based /
DP-boosting anomaly variant that would plug into the iterative SparseGNN engine.

Such a mechanism would reuse the entire SparseGNN pipeline (root sampling, SparseExpand,
gradient aggregation, and the DP clip/noise path) and only need to define its own
per-subgraph contribution g0(H).

Contract for a concrete implementation:
    * `self.module` holds the learnable parameters theta of the anomaly model
      (e.g. a small scorer over aggregated subgraph statistics — degree
      histograms, feature moments, motif counts, reconstruction error of a
      subgraph autoencoder, etc.).  It need not be a message-passing GNN.
    * `subgraph_loss(H)` returns a scalar whose gradient w.r.t. theta is the
      subgraph's contribution g0(H).  Under Assumption 3.2 this contribution is
      clipped to ||g0(H)||_2 <= C by the engine's DP path, so any bounded,
      differentiable subgraph-level objective is admissible (one-class /
      reconstruction / contrastive losses are all natural choices).
    * `evaluate(data)` reports the task metric (e.g. ROC-AUC / average precision
      of anomaly scores against ground-truth labels).

Because the interface matches GNNMechanism, swapping the anomaly detector in is
a one-line change at the call site in run.py; the engine is unchanged.
"""

from typing import Dict

import torch

from .base_mechanism import BaseMechanism


class AnomalyMechanism(BaseMechanism):
    """Non-GNN graph-anomaly base mechanism (interface fixed, body unbuilt)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AnomalyMechanism is a stub. Implement subgraph_loss (a bounded, "
            "differentiable subgraph-level anomaly objective) and evaluate "
            "(anomaly-detection metric), following the contract in this "
            "module's docstring. The SparseGNN engine needs no changes."
        )

    def subgraph_loss(self, subgraph) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, data) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError
