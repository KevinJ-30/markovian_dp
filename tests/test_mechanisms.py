"""Tests for classification and regression mechanisms and their shared engine."""

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models.binary_mechanism import BinaryGNNMechanism, _auroc
from src.models.bootstrap import BootstrapConfig, BootstrapMetrics
from src.models.gnn_mechanism import GNNMechanism
from src.models.multilabel_mechanism import MultiLabelGNNMechanism, _micro_f1
from src.models.regression_mechanism import RegressionGNNMechanism
from src.models.objectives import _task_metric, trivial_baseline
from src.processing.sparse_expand import build_adjacency, sparse_expand


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


def test_multiclass_evaluation_excludes_ignored_metric_label():
    data = Data(
        x=torch.ones((3, 1)),
        y=torch.tensor([0, 1, 19]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    data.train_mask = data.val_mask = data.test_mask = torch.ones(
        3, dtype=torch.bool)

    class FixedPredictions(torch.nn.Module):
        def forward(self, x, edge_index):
            logits = torch.zeros((3, 20))
            logits[0, 0] = 1
            logits[1, 1] = 1
            logits[2, 0] = 1  # deliberately wrong, but class 19 is unscored
            return logits

    mechanism = GNNMechanism(
        data, 1, 20, hidden=2, num_layers=1, dropout=0.0,
        metric_ignore_label=19)
    mechanism.module = FixedPredictions()

    assert mechanism.evaluate(data) == {
        "train": 1.0,
        "val": 1.0,
        "test": 1.0,
    }


@pytest.mark.parametrize(
    ("targets", "predictions", "expected"),
    [
        ([0., 1., 2.], [3., 3., 3.], -6.),
        ([7., 7.], [7., 7.], 1.),
        ([7., 7.], [7., 8.], 0.),
        ([1.], [1.], float("nan")),
        ([], [], float("nan")),
    ],
)
def test_regression_r2_negative_constant_and_undefined_scores(targets, predictions, expected):
    score, _ = _task_metric(
        torch.tensor(predictions).reshape(-1, 1), torch.tensor(targets),
        multilabel=False, regression=True)
    if math.isnan(expected):
        assert math.isnan(score)
    else:
        assert score == pytest.approx(expected)


def test_regression_reference_uses_training_mean_and_is_affine_invariant():
    data = Data(
        y=torch.tensor([10., 12., 0., 2.]),
        train_mask=torch.tensor([True, True, False, False]),
        test_mask=torch.tensor([False, False, True, True]),
    )
    # Predicting the train mean (11) has SSE=202 and test SST=2, not R²=0.
    assert trivial_baseline(data, "r2") == pytest.approx(-100.)
    data.y = data.y * 3 + 7
    assert trivial_baseline(data, "r2") == pytest.approx(-100.)


def test_regression_mechanism_scores_each_split_with_its_own_mean():
    data = Data(
        x=torch.tensor([[1.], [1.], [10.], [10.], [0.], [0.]]),
        y=torch.tensor([0., 2., 10., 12., 5., 5.]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    for index, role in enumerate(("train", "val", "test")):
        mask = torch.zeros(6, dtype=torch.bool)
        mask[2 * index:2 * index + 2] = True
        setattr(data, f"{role}_mask", mask)

    class FixedPredictions(torch.nn.Module):
        def forward(self, x, edge_index):
            return x[:, 0]

    mechanism = RegressionGNNMechanism(data, 1, hidden=2, num_layers=1)
    mechanism.module = FixedPredictions()
    assert mechanism.evaluate() == {"train": 0., "val": -1., "test": 0.}


@pytest.mark.parametrize(
    "mechanism_type, labels, predictions, expected, secondary",
    [
        (GNNMechanism, [0, 1], [[2., -1.], [-1., 2.]], 1., None),
        (BinaryGNNMechanism, [0, 1], [-1., 1.], 1., "val_bin_acc"),
        (MultiLabelGNNMechanism, [[0., 1.], [1., 0.]],
         [[-1., 1.], [1., -1.]], 1., "val_auroc"),
        (RegressionGNNMechanism, [0., 2.], [2., 4.], -3., None),
    ],
)
def test_validation_only_evaluation_needs_no_train_or_test_mask(
        mechanism_type, labels, predictions, expected, secondary):
    data = Data(
        x=torch.tensor(predictions).reshape(2, -1),
        y=torch.tensor(labels),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        train_mask=torch.ones(2, dtype=torch.bool),
        val_mask=torch.ones(2, dtype=torch.bool),
    )
    mechanism = mechanism_type(data, data.x.size(1), 2, num_layers=1)

    class FixedPredictions(torch.nn.Module):
        def forward(self, x, edge_index):
            return x[:, 0] if x.size(1) == 1 else x

    mechanism.module = FixedPredictions()
    del data.train_mask
    accumulator = BootstrapMetrics(
        mechanism.metric_name, BootstrapConfig(n_resamples=10))
    metrics = mechanism.evaluate(
        data, splits=("val",), bootstrap=accumulator)

    expected_metrics = {"val": expected}
    if secondary:
        expected_metrics[secondary] = 1.
    assert metrics == expected_metrics
    assert accumulator.compute()["n_observations"] == 0


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
    from src.training.sparse_gnn import train_sparse_gnn

    data = _toy_data(num_labels=3)
    mech = MultiLabelGNNMechanism(data, 4, 3, hidden=8, num_layers=2)
    mech.build_optimizer(lr=0.05, kind='adam')
    metrics = train_sparse_gnn(
        mech, data, data, direction='in', p1=1.0, p2=1.0, r=2, T=20,
        seed=0)
    assert {"train", "val", "test"} <= set(metrics)


@pytest.mark.parametrize(
    "kind", ["multiclass", "binary", "multilabel", "regression"])
def test_every_mechanism_trains_one_private_padded_step(kind):
    from src.training.sparse_gnn import train_sparse_gnn

    if kind == "binary":
        data = _toy_data(binary=True)
        mechanism = BinaryGNNMechanism(
            data, 4, 2, hidden=8, num_layers=2, dropout=0.0)
    elif kind == "multilabel":
        data = _toy_data(num_labels=3)
        mechanism = MultiLabelGNNMechanism(
            data, 4, 3, hidden=8, num_layers=2, dropout=0.0)
    elif kind == "regression":
        data = _toy_data()
        data.y = data.y.float()
        mechanism = RegressionGNNMechanism(
            data, 4, 1, hidden=8, num_layers=2, dropout=0.0)
    else:
        data = _toy_data()
        mechanism = GNNMechanism(
            data, 4, 2, hidden=8, num_layers=2, dropout=0.0)

    mechanism.build_optimizer(lr=0.01, kind="sgd")
    metrics = train_sparse_gnn(
        mechanism, data, data, direction="in", p1=1.0, p2=1.0, r=1,
        T=1, dp=True, clip=1.0, sigma=1.0, seed=4)
    assert {"train", "val", "test"} <= set(metrics)


# ── large-graph evaluation ────────────────────────────────────────────────────

@pytest.mark.parametrize("aggr", ["mean", "gcn", "gin"])
def test_csr_eval_path_matches_edge_index(aggr):
    """Above the dense-message budget, evaluate() must switch to CSR and give
    the same numbers.

    Message passing over an edge_index gathers x[edge_index[0]], materializing
    an [E, F] tensor.  On Reddit that is 114.6M x 602 x 4B = 276 GB, which is
    what full-graph evaluation actually tried to allocate.  The CSR adjacency
    fuses gather and scatter; PyG's result is identical either way.
    """
    from src.models.gnn_mechanism import GNNMechanism

    torch.manual_seed(0)
    n, f, c = 200, 6, 3
    ei = torch.unique(torch.stack([torch.randint(0, n, (2000,)),
                                   torch.randint(0, n, (2000,))]), dim=1)
    data = _toy_like(n, f, c, ei)

    torch.manual_seed(1)
    mech = GNNMechanism(data, f, c, hidden=8, num_layers=2, dropout=0.0,
                        aggr=aggr)
    dense = mech.evaluate(data)

    mech._DENSE_MESSAGE_BUDGET = 0                       # force CSR
    mech._eval_adj_cache = None
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
