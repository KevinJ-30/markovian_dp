"""Validation selection must not change SparseGNN's update trajectory."""

import math

import pytest
import torch
from torch_geometric.data import Data

from src.models.bootstrap import BootstrapConfig
from src.models.gnn_mechanism import GNNMechanism
from src.models.regression_mechanism import RegressionGNNMechanism
from src.training.sparse_gnn import train_sparse_gnn


class _OffsetPredictions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.))
        self.register_buffer("updates", torch.tensor(0))
        self.training_offsets = []

    def forward(self, x, edge_index):
        if self.training:
            self.training_offsets.append(float(self.offset.detach()))
            self.updates.add_(1)
        return x[:, 0] + self.offset


class _RecordingRegression(RegressionGNNMechanism):
    def __init__(self, data):
        super().__init__(data, 1, num_layers=1)
        self.module = _OffsetPredictions()
        self.evaluations = []
        self.build_optimizer(lr=.25, kind="sgd", momentum=1.)

    def evaluate(self, data=None, *, splits=("train", "val", "test"), bootstrap=None):
        self.evaluations.append(
            (int(self.module.updates), tuple(splits), bootstrap is not None))
        return super().evaluate(data, splits=splits, bootstrap=bootstrap)


def _regression_fixture(validation_center=1.5625):
    # One supervised root gives exact SGD offsets .5, 1.25, 1.875, 2.0625.
    # Reversed validation features make every candidate's R² negative.
    data = Data(
        x=torch.tensor([[0.], [2.5 - validation_center],
                        [-1.5 - validation_center], [0.], [2.]]),
        y=torch.tensor([1., 0., 1., 1.25, 3.25]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        train_mask=torch.tensor([True, False, False, False, False]),
        val_mask=torch.tensor([False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, True, True]),
    )
    return data, _RecordingRegression(data)


@pytest.mark.parametrize("track_every", [0, 1])
def test_restores_strict_validation_max_and_earliest_tie_before_test_bootstrap(track_every):
    data, mechanism = _regression_fixture()
    callbacks = []
    result = train_sparse_gnn(
        mechanism, data, data, p1=1., p2=1., r=0, T=4,
        eval_every=1, track_every=track_every,
        checkpoint_callback=lambda checkpoint: callbacks.append(dict(checkpoint)),
        bootstrap=BootstrapConfig(n_resamples=30, seed=9))

    # Step 2 improves over step 1; distinct weights at step 3 tie with step 2;
    # step 4 is worse. All updates still execute, then model buffers restore too.
    assert mechanism.module.training_offsets == [0., .5, 1.25, 1.875]
    assert float(mechanism.module.offset) == 1.25
    assert int(mechanism.module.updates) == 2
    assert result["selection"] == {
        "metric": "r2", "step": 2, "validation_score": -24.390625,
        "evaluate_every": 1,
    }
    assert result["val"] == -24.390625
    assert result["test"] == 1.
    ci = result["test_confidence_intervals"]
    assert ci["n_observations"] == 2
    assert ci["metrics"]["r2"] == {
        "lower": 1., "upper": 1., "valid_resamples": 30,
    }
    training_splits = ("train", "val", "test") if track_every else ("val",)
    assert mechanism.evaluations == [
        (step, training_splits, False) for step in range(1, 5)
    ] + [(2, ("train", "val", "test"), True)]
    if track_every:
        assert callbacks == result["history"]
        assert [checkpoint["step"] for checkpoint in callbacks] == [1, 2, 3, 4]
        assert callbacks[-1]["test"] < result["test"]
        assert all("test_confidence_intervals" not in checkpoint for checkpoint in callbacks)
    else:
        assert callbacks == []


@pytest.mark.parametrize("tracking", [False, True])
def test_tracking_and_progress_do_not_add_selection_candidates(tracking):
    data, mechanism = _regression_fixture(validation_center=.5)
    result = train_sparse_gnn(
        mechanism, data, data, p1=1., p2=1., r=0, T=4, eval_every=2,
        track_every=1 if tracking else 0,
        verbose=tracking, progress_every=1)

    # Step 1 is the global optimum but is not a selection candidate, even when
    # tracking or logging evaluates it. Only steps 2 and 4 compete.
    assert result["selection"]["step"] == 2
    assert result["selection"]["evaluate_every"] == 2
    assert float(mechanism.module.offset) == 1.25
    if tracking:
        assert result["history"][0]["val"] > result["val"]


def test_final_step_competes_outside_regular_validation_cadence():
    data, mechanism = _regression_fixture(validation_center=2.0625)
    result = train_sparse_gnn(
        mechanism, data, data, p1=1., p2=1., r=0, T=4, eval_every=3)

    assert result["selection"]["step"] == 4
    assert float(mechanism.module.offset) == 2.0625
    assert mechanism.evaluations == [
        (3, ("val",), False), (4, ("val",), False),
        (4, ("train", "val", "test"), False),
    ]


def test_undefined_validation_restores_first_evaluated_model():
    data, mechanism = _regression_fixture()
    data.val_mask.zero_()
    result = train_sparse_gnn(
        mechanism, data, data, p1=1., p2=1., r=0, T=4, eval_every=2)

    assert result["selection"] == {
        "metric": "r2", "step": 2, "validation_score": None,
        "evaluate_every": 2,
    }
    assert math.isnan(result["val"])
    assert float(mechanism.module.offset) == 1.25
    assert mechanism.module.training_offsets == [0., .5, 1.25, 1.875]


def test_empty_draws_do_not_skip_default_epoch_validation_or_final_candidate():
    data, mechanism = _regression_fixture()
    data.train_mask.zero_()
    result = train_sparse_gnn(
        mechanism, data, data, p1=.4, p2=1., r=0, T=4)

    assert result["selection"]["evaluate_every"] == 3  # ceil(1 / .4)
    assert result["selection"]["step"] == 3  # identical model ties at step 4
    assert mechanism.module.training_offsets == []
    assert mechanism.evaluations == [
        (0, ("val",), False), (0, ("val",), False),
        (0, ("train", "val", "test"), False),
    ]


@pytest.mark.parametrize("cadence", ["eval_every", "track_every", "progress_every"])
def test_negative_cadence_rejected_before_training(cadence):
    data, mechanism = _regression_fixture()
    with pytest.raises(ValueError, match=cadence):
        train_sparse_gnn(
            mechanism, data, data, p1=1., p2=1., r=0, T=4, **{cadence: -1})
    assert mechanism.module.training_offsets == []


@pytest.mark.parametrize("sampling_seed", [None, 17])
def test_tracking_and_progress_preserve_dropout_updates_and_sampling_rng(sampling_seed):
    class RecordingGNN(GNNMechanism):
        def __init__(self, data):
            super().__init__(data, 3, 2, hidden=4, num_layers=2, dropout=.5)
            self.trajectory = []

        def subgraph_losses(self, subgraphs):
            self.trajectory.append((
                [(subgraph.root, subgraph.nodes.tolist(), subgraph.edge_index.tolist())
                 for subgraph in subgraphs],
                [parameter.detach().clone() for parameter in self.parameters()],
            ))
            return super().subgraph_losses(subgraphs)

    def run(track_every, verbose):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            data = Data(
                x=torch.randn(8, 3), y=torch.arange(8) % 2,
                edge_index=torch.tensor([
                    [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
                    [1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 0],
                ]),
                train_mask=torch.tensor([True] * 4 + [False] * 4),
                val_mask=torch.tensor([False] * 4 + [True, True, False, False]),
                test_mask=torch.tensor([False] * 6 + [True, True]),
            )
            mechanism = RecordingGNN(data)
            mechanism.build_optimizer(lr=.1, kind="sgd")
            result = train_sparse_gnn(
                mechanism, data, data, p1=.75, p2=.5, r=2, T=6,
                seed=sampling_seed, eval_every=2, track_every=track_every,
                verbose=verbose, progress_every=1)
            rng = torch.get_rng_state().clone()
        return mechanism, result, rng

    plain, plain_result, plain_rng = run(0, False)
    tracked, tracked_result, tracked_rng = run(1, True)
    assert plain_result["selection"] == tracked_result["selection"]
    assert torch.equal(plain_rng, tracked_rng)
    assert len(plain.trajectory) == len(tracked.trajectory)
    for (plain_samples, plain_params), (tracked_samples, tracked_params) in zip(
            plain.trajectory, tracked.trajectory):
        assert plain_samples == tracked_samples
        for left, right in zip(plain_params, tracked_params):
            assert torch.equal(left, right)
    for left, right in zip(plain.parameters(), tracked.parameters()):
        assert torch.equal(left, right)
