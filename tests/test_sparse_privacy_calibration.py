import csv
import sys
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.privacy.accounting import SparseGNNNoiseCalibration
from src.experiments import sparse as sparse_run


def test_target_cli_calibrates_each_cell_once_and_records_metadata(
        monkeypatch, tmp_path):
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


def _domain_dataset_and_data():
    split = {
        "train": ["de"],
        "val": ["fr"],
        "test": ["fr"],
        "seed": 7,
        "val_ratio": 0.3,
    }
    dataset = SimpleNamespace(
        num_features=1,
        num_classes=20,
        domain_dataset=True,
        domain_split=split,
        domain_split_id="abc123",
        task_type="MULTICLASS",
        primary_metric="accuracy",
        metric_ignore_label=19,
    )
    data = Data(
        x=torch.ones((4, 1)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        y=torch.tensor([0, 1, 2, 3]),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )
    return dataset, data


def test_domain_cli_forwards_split_preserves_target_context_and_records_identity(
        monkeypatch, tmp_path):
    dataset, data = _domain_dataset_and_data()
    loaded = []
    trained = []
    mechanism_kwargs = []

    class FakeMechanism:
        metric_name = "accuracy"

        def __init__(self, *args, **kwargs):
            mechanism_kwargs.append(kwargs)

        def build_optimizer(self, **kwargs):
            pass

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return dataset, data

    def train(mechanism, train_graph, test_graph, **kwargs):
        trained.append((train_graph, test_graph))
        return {"train": 0.8, "val": 0.7, "test": 0.6}

    monkeypatch.setattr(sparse_run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sparse_run, "load_dataset", load)
    monkeypatch.setattr(sparse_run, "_MECHANISMS", {"gnn": FakeMechanism})
    monkeypatch.setattr(
        sparse_run, "_report_subgraph_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sparse_run, "trivial_baseline", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(sparse_run, "train_sparse_gnn", train)
    monkeypatch.setattr(sys, "argv", [
        "run", "--dataset", "mag-countries", "--model", "gnn",
        "--train_domains", "de", "--val_domains", "fr",
        "--test_domains", "fr", "--domain_split_seed", "7",
        "--domain_val_ratio", "0.3", "--p1", "1", "--p2", "1", "--r", "1",
        "--T", "1", "--seeds", "1", "--out_dir", str(tmp_path),
    ])

    sparse_run.main()

    assert loaded == [(
        "mag-countries",
        {
            "device": "cpu",
            "domain_split": {
                "train": ["de"],
                "val": ["fr"],
                "test": ["fr"],
                "seed": 7,
                "val_ratio": 0.3,
            },
        },
    )]
    assert mechanism_kwargs[0]["metric_ignore_label"] == 19
    train_graph, test_graph = trained[0]
    assert {
        tuple(edge) for edge in test_graph.edge_index.t().tolist()
    } == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert {
        tuple(edge) for edge in train_graph.edge_index.t().tolist()
    } == {(0, 1), (1, 0)}

    path = tmp_path / "sparse_gnn_mag-countries_abc123_results.csv"
    with path.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["domain_split_id"] == "abc123"
    assert row["domain_split"] == (
        '{"seed":7,"test":["fr"],"train":["de"],'
        '"val":["fr"],"val_ratio":0.3}')


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--model", "gnn"], "use --model binary_gnn"),
        (
            ["--model", "binary_gnn", "--common_inductive_split"],
            "cannot be used with a domain dataset",
        ),
    ],
)
def test_twitch_guards_reject_wrong_model_and_common_split(
        monkeypatch, tmp_path, extra_args, message):
    dataset, data = _domain_dataset_and_data()
    dataset.num_classes = 2
    dataset.task_type = "BINARY"
    dataset.primary_metric = "auroc"
    dataset.metric_ignore_label = None
    monkeypatch.setattr(sparse_run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sparse_run, "load_dataset", lambda *args, **kwargs: (dataset, data))
    monkeypatch.setattr(sys, "argv", [
        "run", "--dataset", "twitch-explicit", "--out_dir", str(tmp_path),
        *extra_args,
    ])

    with pytest.raises(SystemExit, match=message):
        sparse_run.main()


def test_domain_cli_requires_all_role_lists(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["run", "--train_domains", "de", "--val_domains", "fr"])
    with pytest.raises(SystemExit) as error:
        sparse_run.parse_args()
    assert error.value.code == 2
    assert "must be provided together" in capsys.readouterr().err


