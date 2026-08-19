"""
Tests for the base mechanisms added alongside the inductive/RelBench suites.

The RelBench graph builder itself is not covered here — it downloads a database
on first use, so it is exercised by scripts/relbench_f1.sh rather than pytest.
Only its pure-python name parsing is unit-tested.
"""

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.sparse.binary_mechanism import BinaryGNNMechanism, _auroc
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism, _micro_f1
from src.sparse.relbench_data import parse_relbench_name
from src.sparse.sparse_expand import build_adjacency, sparse_expand


def _toy_data(num_labels=None, binary=False):
    """A 6-node directed graph with one node per split."""
    torch.manual_seed(0)
    edge_index = torch.tensor([[1, 2, 3, 4, 5, 0],
                               [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    x = torch.randn(6, 4)
    if binary:
        y = torch.tensor([0, 1, 0, 1, 0, 1])
    elif num_labels:
        y = (torch.rand(6, num_labels) > 0.5).float()
    else:
        y = torch.tensor([0, 1, 0, 1, 0, 1])
    data = Data(x=x, y=y, edge_index=edge_index)
    for i, split in enumerate(('train', 'val', 'test')):
        mask = torch.zeros(6, dtype=torch.bool)
        mask[i * 2:(i + 1) * 2] = True
        setattr(data, f'{split}_mask', mask)
    return data


# ── AUROC / micro-F1 helpers ──────────────────────────────────────────────────

def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert math.isclose(_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
    assert math.isclose(_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)


def test_auroc_all_ties_is_one_half():
    y = np.array([0, 0, 1, 1])
    assert math.isclose(_auroc(y, np.full(4, 0.5)), 0.5)


def test_auroc_single_class_is_nan():
    assert math.isnan(_auroc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))


def test_auroc_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = rng.normal(size=200) + y            # correlated with the label
    assert math.isclose(_auroc(y, scores),
                        sklearn.roc_auc_score(y, scores), rel_tol=1e-9)


def test_micro_f1_hand_checked():
    pred = torch.tensor([[1., 0., 1.]])
    target = torch.tensor([[1., 1., 0.]])
    # tp=1, fp=1, fn=1 -> 2/(2+1+1) = 0.5
    assert math.isclose(_micro_f1(pred, target), 0.5)
    assert math.isclose(_micro_f1(target, target), 1.0)


# ── mechanisms plug into the engine ───────────────────────────────────────────

@pytest.mark.parametrize("kind", ["binary", "multilabel"])
def test_mechanism_subgraph_loss_and_metrics(kind):
    if kind == "binary":
        data = _toy_data(binary=True)
        mech = BinaryGNNMechanism(data, 4, 2, hidden=8, num_layers=2)
        expected_metric = "auroc"
    else:
        data = _toy_data(num_labels=3)
        mech = MultiLabelGNNMechanism(data, 4, 3, hidden=8, num_layers=2)
        expected_metric = "micro_f1"

    assert mech.metric_name == expected_metric

    adj = build_adjacency(data.edge_index, 6, direction='in')
    sg = sparse_expand(adj, 0, p2=1.0, r=2, direction='in')
    loss = mech.subgraph_loss(sg)
    assert torch.isfinite(loss) and loss.requires_grad

    # An unlabelled-split root contributes a differentiable zero.
    assert float(mech.subgraph_loss(
        sparse_expand(adj, 4, p2=1.0, r=2, direction='in'))) == 0.0

    metrics = mech.evaluate(data)
    # Contract: the three split keys are required; a mechanism may report extra
    # secondary metrics (multilabel adds <split>_auroc, since micro-F1 is
    # degenerate at low epsilon — see multilabel_mechanism._micro_auroc).
    assert {"train", "val", "test"} <= set(metrics)
    for v in metrics.values():
        assert math.isnan(v) or 0.0 <= v <= 1.0
    if kind == "multilabel":
        assert {"train_auroc", "val_auroc", "test_auroc"} <= set(metrics)


def test_mechanism_trains_through_the_engine():
    from src.sparse.sparse_gnn import train_sparse_gnn

    data = _toy_data(num_labels=3)
    mech = MultiLabelGNNMechanism(data, 4, 3, hidden=8, num_layers=2)
    mech.build_optimizer(lr=0.05, kind='adam')
    metrics = train_sparse_gnn(mech, data, direction='in', p1=1.0, p2=1.0,
                               r=2, T=20,
                               candidate_nodes=torch.where(data.train_mask)[0],
                               seed=0)
    assert {"train", "val", "test"} <= set(metrics)


# ── RelBench name parsing ─────────────────────────────────────────────────────

def test_parse_relbench_name():
    assert parse_relbench_name('relbench:rel-f1/driver-top3') == \
        ('rel-f1', 'driver-top3')
    with pytest.raises(ValueError):
        parse_relbench_name('relbench:rel-f1')


# ── large-graph evaluation ────────────────────────────────────────────────────

@pytest.mark.parametrize("aggr", ["mean", "gcn"])
def test_csr_eval_path_matches_edge_index(aggr):
    """Above the dense-message budget, evaluate() must switch to CSR and give
    the same numbers.

    Message passing over an edge_index gathers x[edge_index[0]], materializing
    an [E, F] tensor.  On Reddit that is 114.6M x 602 x 4B = 276 GB, which is
    what full-graph evaluation actually tried to allocate.  The CSR adjacency
    fuses gather and scatter; PyG's result is identical either way.
    """
    from src.sparse.gnn_mechanism import GNNMechanism

    torch.manual_seed(0)
    n, f, c = 200, 6, 3
    ei = torch.unique(torch.stack([torch.randint(0, n, (2000,)),
                                   torch.randint(0, n, (2000,))]), dim=1)
    data = _toy_like(n, f, c, ei)

    torch.manual_seed(1)
    mech = GNNMechanism(data, f, c, hidden=8, num_layers=2, dropout=0.0,
                        aggr=aggr)
    dense = mech.evaluate(data)
    assert mech.eval_edges(data) is data.edge_index      # budget not exceeded

    mech._DENSE_MESSAGE_BUDGET = 0                       # force CSR
    mech._eval_adj_cache = None
    assert mech.eval_edges(data) is not data.edge_index  # now a CSR adjacency
    sparse = mech.evaluate(data)

    for split in ("train", "val", "test"):
        assert dense[split] == pytest.approx(sparse[split], abs=1e-9)


def _toy_like(n, f, c, edge_index):
    from torch_geometric.data import Data
    data = Data(x=torch.randn(n, f), y=torch.randint(0, c, (n,)),
                edge_index=edge_index)
    data.train_mask = torch.ones(n, dtype=torch.bool)
    data.val_mask = data.test_mask = data.train_mask
    return data
