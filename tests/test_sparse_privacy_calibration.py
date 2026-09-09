import csv
import sys
from types import SimpleNamespace

import pytest
import torch

from src.sparse.accounting import SparseGNNNoiseCalibration
from src.sparse import run as sparse_run
from src.sparse import sparse_gnn


def _calibration(multiplier=2.5):
    noise_std = multiplier * 1.5
    return SparseGNNNoiseCalibration(
        noise_multiplier=multiplier,
        noise_std=noise_std,
        noise_variance=noise_std ** 2,
        epsilon=0.9,
        target_epsilon=1.0,
        delta=1e-5,
        theorem="thm6.4-substitution",
        evaluations=7,
    )


def test_train_sparse_gnn_with_budget_calibrates_once_and_forwards(monkeypatch):
    calibration = _calibration()
    calibration_calls = []
    train_calls = []

    def calibrate(**kwargs):
        calibration_calls.append(kwargs)
        return calibration

    def train(mechanism, data, **kwargs):
        train_calls.append((mechanism, data, kwargs))
        return {"test": 0.8}

    monkeypatch.setattr(sparse_gnn, "calibrate_sparsegnn_noise", calibrate)
    monkeypatch.setattr(sparse_gnn, "train_sparse_gnn", train)
    checkpoint_callback = object()
    metrics, result = sparse_gnn.train_sparse_gnn_with_budget(
        "mechanism", "data", target_epsilon=1.0, target_delta=1e-5,
        K_in=3, K_out=4, p1=0.2, p2=0.3, r=2, T=10, clip=1.5,
        direction="out", theorem="thm45", accounting_grid=2e-4,
        calibration_rtol=2e-3, calibration_atol=3e-6, max_sigma=99.0,
        adj="adj", candidate_nodes="roots", seed=8, eval_every=9,
        track_every=10, eval_alt_edge_index="alt", verbose=True,
        checkpoint_callback=checkpoint_callback,
    )

    assert metrics == {"test": 0.8}
    assert result is calibration
    assert calibration_calls == [{
        "target_epsilon": 1.0, "target_delta": 1e-5,
        "p1": 0.2, "p2": 0.3, "r": 2, "K_in": 3, "K_out": 4,
        "steps": 10, "clip": 1.5, "direction": "out", "theorem": "thm45",
        "grid": 2e-4, "sigma_rtol": 2e-3, "sigma_atol": 3e-6,
        "max_sigma": 99.0,
    }]
    assert train_calls == [("mechanism", "data", {
        "p1": 0.2, "p2": 0.3, "r": 2, "T": 10, "adj": "adj",
        "direction": "out", "candidate_nodes": "roots", "dp": True,
        "clip": 1.5, "sigma": calibration.noise_multiplier, "seed": 8,
        "eval_every": 9, "track_every": 10, "eval_alt_edge_index": "alt",
        "verbose": True, "checkpoint_callback": checkpoint_callback,
    })]


def test_target_cli_calibrates_each_cell_once_and_records_metadata(
        monkeypatch, tmp_path):
    from torch_geometric.data import Data

    data = Data(
        x=torch.ones((3, 1)),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, False]),
        val_mask=torch.tensor([False, False, True]),
        test_mask=torch.tensor([False, False, True]),
    )
    dataset = SimpleNamespace(num_features=1, num_classes=2)
    calibrations = []
    training = []

    class FakeMechanism:
        metric_name = "accuracy"

        def __init__(self, *args, **kwargs):
            pass

        def build_optimizer(self, **kwargs):
            pass

    def calibrate(**kwargs):
        calibrations.append(kwargs)
        multiplier = 2.0 + kwargs["p2"]
        return SparseGNNNoiseCalibration(
            noise_multiplier=multiplier,
            noise_std=multiplier * kwargs["clip"],
            noise_variance=(multiplier * kwargs["clip"]) ** 2,
            epsilon=0.95,
            target_epsilon=kwargs["target_epsilon"],
            delta=kwargs["target_delta"],
            theorem="thm6.4-substitution",
            evaluations=5,
        )

    def train(*args, **kwargs):
        training.append(kwargs)
        return {"train": 0.8, "val": 0.7, "test": 0.6}

    monkeypatch.setattr(sparse_run, "load_dataset", lambda *args, **kwargs: (dataset, data))
    monkeypatch.setattr(sparse_run, "_MECHANISMS", {"gnn": FakeMechanism})
    monkeypatch.setattr(sparse_run, "_report_subgraph_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(sparse_run, "trivial_baseline", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(sparse_run, "calibrate_sparsegnn_noise", calibrate)
    monkeypatch.setattr(sparse_run, "train_sparse_gnn", train)
    monkeypatch.setattr(sys, "argv", [
        "run", "--dataset", "tiny", "--dp", "--p1", "0.1", "--p2", "0.2", "0.4",
        "--r", "1", "--T", "2", "--K_in", "2", "--K_out", "2", "--seeds", "2",
        "--target_epsilon", "1.0", "--target_delta", "1e-5", "--out_dir", str(tmp_path),
    ])

    sparse_run.main()

    assert len(calibrations) == 2
    assert [call["p2"] for call in calibrations] == [0.2, 0.4]
    assert [call["sigma"] for call in training] == [2.2, 2.2, 2.4, 2.4]
    with (tmp_path / "sparse_gnn_tiny_dp_results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {row["sigma"] for row in rows} == {"2.2", "2.4"}
    assert {row["target_epsilon"] for row in rows} == {"1.0"}
    assert {row["calibrated_epsilon"] for row in rows} == {"0.95"}
    assert {row["accounting_theorem"] for row in rows} == {"thm6.4-substitution"}
    assert {row["noise_variance"] for row in rows} == {"4.840000000000001", "5.76"}


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--dp", "--target_epsilon", "1.0"],
        ["run", "--dp", "--target_delta", "1e-5"],
        ["run", "--target_epsilon", "1.0", "--target_delta", "1e-5"],
        ["run", "--dp", "--sigma", "1.0", "--target_epsilon", "1.0", "--target_delta", "1e-5"],
    ],
)
def test_target_cli_validates_noise_selection(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        sparse_run.parse_args()


def test_cli_preserves_current_optimizer_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run"])
    args = sparse_run.parse_args()
    assert args.optimizer == "auto"
    assert args.lr == 1e-2
    assert args.weight_decay == 5e-4
