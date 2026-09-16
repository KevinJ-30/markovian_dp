import csv
import sys
from types import SimpleNamespace

import pytest
import torch

from src.privacy.accounting import SparseGNNNoiseCalibration
from src.experiments import sparse as sparse_run


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
            evaluations=5,
        )

    def train(*args, **kwargs):
        training.append((args, kwargs))
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
    assert [call[1]["sigma"] for call in training] == [2.2, 2.2, 2.4, 2.4]
    for call_args, _ in training:
        _, train_graph, test_graph = call_args
        assert train_graph is not test_graph
        assert train_graph.edge_index.size(1) < test_graph.edge_index.size(1)
    with (tmp_path / "sparse_gnn_tiny_dp_results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {row["sigma"] for row in rows} == {"2.2", "2.4"}
    assert {row["target_epsilon"] for row in rows} == {"1.0"}
    assert {row["calibrated_epsilon"] for row in rows} == {"0.95"}
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


