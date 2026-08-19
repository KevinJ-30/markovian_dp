"""
XGB-Graph detector (GADBench): XGBoost on parameter-free multi-hop aggregated features.

Pipeline:
  1. Build features [N, (L+1)*d] by neighbor-aggregating over the (possibly sparsified) graph.
  2. Fit an XGBoost binary classifier on the training nodes.
  3. Score every node by its predicted anomaly probability (class 1).

Sparsification enters only in step 1 (which edges are available for aggregation), so a lower
p2 degrades the features and the downstream utility — the effect we measure.
"""

import torch

from .neighbor_aggregation import (
    aggregate_features,
    aggregate_features_expand,
    sparsify_edges_bernoulli,
)
from ..sparse_expand import build_adjacency

# GADBench XGBoost defaults (Table 9): n_estimators=100, lr default 0.3, gbtree, L2=1.
_DEFAULT_XGB = dict(
    n_estimators=100,
    learning_rate=0.3,
    max_depth=6,
    subsample=1.0,
    reg_lambda=1.0,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="aucpr",
)


class XGBGraphDetector:
    """XGB-Graph anomaly detector.

    Args:
        num_layers:      number of neighbor-aggregation hops L (GADBench default 2).
        aggr:            'mean' (default) | 'sum' | 'max' | 'min'.
        scale_pos_weight: if True, set XGBoost scale_pos_weight = #neg/#pos on the train set
                          to counter class imbalance (anomalies are rare).
        xgb_params:      overrides merged into the GADBench defaults.
    """

    def __init__(self, num_layers=2, aggr="mean", scale_pos_weight=True, **xgb_params):
        self.num_layers = num_layers
        self.aggr = aggr
        self.scale_pos_weight = scale_pos_weight
        self.xgb_params = {**_DEFAULT_XGB, **xgb_params}
        self.model = None

    # ── feature construction ──────────────────────────────────────────────────

    def build_features(self, data, p2=1.0, sparsifier="global", generator=None):
        """Return aggregated node features [N, (L+1)*d] on the sparsified graph.

        sparsifier:
          'global' — one Bernoulli(p2) edge drop, then global L-hop aggregation (fast, default).
          'expand' — per-root SparseExpand(p2, r=num_layers) aggregation (slow on dense graphs).
        """
        x = data.x
        if sparsifier == "global":
            edge_index = sparsify_edges_bernoulli(data.edge_index, p2, generator=generator)
            return aggregate_features(x, edge_index, self.num_layers, aggr=self.aggr)
        if sparsifier == "expand":
            adj = build_adjacency(data.edge_index, int(data.num_nodes), direction='in')
            nodes = torch.arange(int(data.num_nodes))
            return aggregate_features_expand(
                x, adj, nodes, p2, self.num_layers, aggr=self.aggr, generator=generator
            )
        raise ValueError(f"unknown sparsifier {sparsifier!r} (use 'global' or 'expand')")

    # ── fit / predict ─────────────────────────────────────────────────────────

    def fit(self, X, y, train_mask):
        from xgboost import XGBClassifier

        Xtr = X[train_mask].detach().cpu().numpy()
        ytr = y[train_mask].detach().cpu().numpy().astype(int)

        params = dict(self.xgb_params)
        if self.scale_pos_weight:
            n_pos = int(ytr.sum())
            n_neg = int(len(ytr) - n_pos)
            params["scale_pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0

        self.model = XGBClassifier(**params)
        self.model.fit(Xtr, ytr)
        return self

    def predict_scores(self, X):
        """Per-node anomaly probability (class 1)."""
        if self.model is None:
            raise RuntimeError("call fit() before predict_scores()")
        Xn = X.detach().cpu().numpy()
        return self.model.predict_proba(Xn)[:, 1]
