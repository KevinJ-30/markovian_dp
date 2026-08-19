"""
Tests for graph anomaly detection (XGB-Graph on sparsified graphs, GADBench).

Run in PytorchEnv (needs xgboost + scikit-learn):
    conda run -n PytorchEnv python -m pytest tests/test_gad.py -q
"""

import numpy as np
import torch

from src.sparse.gad.neighbor_aggregation import (
    aggregate_features, aggregate_features_expand, sparsify_edges_bernoulli,
)
from src.sparse.gad.metrics import auroc, auprc, rec_at_k
from src.sparse.sparse_expand import build_adjacency


def _undirected_toy():
    # Triangle 0-1-2 plus pendant 2-3, stored as both directed arcs.
    u = torch.tensor([0, 1, 1, 2, 2, 3])
    v = torch.tensor([1, 0, 2, 1, 3, 2])
    edge_index = torch.stack([u, v])
    x = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    return edge_index, x, 4


def test_aggregate_L0_is_identity():
    edge_index, x, _ = _undirected_toy()
    out = aggregate_features(x, edge_index, num_layers=0)
    assert torch.equal(out, x)
    assert out.shape == (4, 1)


def test_aggregate_L1_mean_matches_manual():
    edge_index, x, _ = _undirected_toy()
    out = aggregate_features(x, edge_index, num_layers=1, aggr="mean")
    assert out.shape == (4, 2)          # [h0 || h1], d=1
    # node 0's neighbors = {1}; node 2's neighbors = {1,3} -> mean(2,8)=5
    assert out[0, 1].item() == 2.0
    assert out[2, 1].item() == 5.0
    # h0 half is unchanged
    assert torch.equal(out[:, :1], x)


def test_output_dim_is_L_plus_1_times_d():
    edge_index, x, _ = _undirected_toy()
    for L in (0, 1, 2, 3):
        out = aggregate_features(x, edge_index, num_layers=L)
        assert out.shape == (4, (L + 1) * x.size(1))


def test_sparsify_edges_bounds_and_determinism():
    edge_index, _, _ = _undirected_toy()
    assert sparsify_edges_bernoulli(edge_index, 1.0).size(1) == edge_index.size(1)
    assert sparsify_edges_bernoulli(edge_index, 0.0).size(1) == 0
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = sparsify_edges_bernoulli(edge_index, 0.5, generator=g1)
    b = sparsify_edges_bernoulli(edge_index, 0.5, generator=g2)
    assert torch.equal(a, b)


def test_global_matches_expand_at_p2_1():
    edge_index, x, n = _undirected_toy()
    L = 2
    glob = aggregate_features(x, edge_index, num_layers=L, aggr="mean")
    adj = build_adjacency(edge_index, n, direction='in')
    nodes = torch.arange(n)
    exp = aggregate_features_expand(x, adj, nodes, p2=1.0, r=L, aggr="mean")
    assert torch.allclose(glob, exp, atol=1e-6)


def test_metrics_hand_checked():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # perfect-ish ranking: top-2 are indices 3 (pos) and 1 (neg) -> rec@2 = 1/2
    assert abs(rec_at_k(y, scores) - 0.5) < 1e-9
    assert 0.0 <= auroc(y, scores) <= 1.0
    assert 0.0 <= auprc(y, scores) <= 1.0
    # a perfect ranker
    assert abs(auroc(np.array([0, 1]), np.array([0.1, 0.9])) - 1.0) < 1e-9


def test_xgbgraph_smoke():
    from src.sparse.gad.xgb_graph import XGBGraphDetector
    # small imbalanced synthetic graph
    torch.manual_seed(0)
    n = 200
    x = torch.randn(n, 5)
    # chain edges both directions
    idx = torch.arange(n - 1)
    edge_index = torch.stack([torch.cat([idx, idx + 1]), torch.cat([idx + 1, idx])])
    y = (torch.rand(n) < 0.2).long()
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[: n // 2] = True

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = n

    det = XGBGraphDetector(num_layers=2, aggr="mean")
    X = det.build_features(data, p2=0.8)
    assert X.shape == (n, 3 * 5)
    det.fit(X, y, train_mask)
    scores = det.predict_scores(X)
    assert scores.shape == (n,)
    assert np.all((scores >= 0) & (scores <= 1))
