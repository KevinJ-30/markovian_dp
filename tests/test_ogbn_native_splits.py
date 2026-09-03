import json
import sys
import types

import pytest
import torch
from torch_geometric.data import Data

from src.datasets import _load_ogb_node
from src.experiments.baselines import BaselineConfig, BaselineTrainer
import src.experiments.dpar as dpar
from src.experiments.inductive import load_or_create_inductive_split
import src.experiments.run as experiment_runner
from src.experiments.upstream import export_partitions


def _native_graph():
    data = Data(
        x=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 3, 5],
                [1, 0, 3, 2, 5, 4, 2, 4, 0],
            ]
        ),
    )
    data.train_mask = torch.tensor([True, True, False, False, False, False])
    data.val_mask = torch.tensor([False, False, True, True, False, False])
    data.test_mask = torch.tensor([False, False, False, False, True, True])
    return data


def _held_out_class_graph():
    data = _native_graph()
    data.y = torch.tensor([0, 1, 2, 2, 2, 2])
    return data


def test_native_split_preserves_masks_and_removes_cross_edges(tmp_path):
    split = load_or_create_inductive_split(
        _native_graph(), "native-unit", root=tmp_path, seed=7, split_strategy="native"
    )
    reloaded = load_or_create_inductive_split(
        _native_graph(), "native-unit", root=tmp_path, seed=7, split_strategy="native"
    )

    assert split.path.name == "native-unit-native-seed7.pt"
    assert [part.node_ids.tolist() for part in (split.train, split.val, split.test)] == [
        [0, 1], [2, 3], [4, 5]
    ]
    assert [part.stats["edges"] for part in (split.train, split.val, split.test)] == [2, 2, 2]
    assert [part.node_ids.tolist() for part in (reloaded.train, reloaded.val, reloaded.test)] == [
        [0, 1], [2, 3], [4, 5]
    ]




def test_global_class_space_includes_held_out_labels(monkeypatch, tmp_path):
    split = load_or_create_inductive_split(
        _held_out_class_graph(), "held-out-class", root=tmp_path / "splits", split_strategy="native"
    )
    assert split.num_classes == 3

    baseline = BaselineTrainer(BaselineConfig(method="mlp", layers=1), "cpu")
    assert baseline._model(split.train.data, split.num_classes).layers[-1].out_features == 3

    captured = {}
    dpar_mlp = dpar.DPARMLP

    def capture_dpar_mlp(inputs, classes, hidden, layers, dropout):
        captured["classes"] = classes
        return dpar_mlp(inputs, classes, hidden, layers, dropout)

    monkeypatch.setattr(dpar, "DPARMLP", capture_dpar_mlp)
    dpar.DPARTrainer(dpar.DPARConfig(epochs=1, hidden_size=4, topk=2, batch_size=8, dropout=0.0), "cpu").fit(
        split
    )
    assert captured["classes"] == 3

    manifest = export_partitions(split, tmp_path / "partitions")
    assert json.loads(manifest.read_text())["num_classes"] == 3

@pytest.mark.parametrize("case", ["missing", "overlap", "gap", "non_boolean", "wrong_length"])
def test_native_split_rejects_invalid_masks(tmp_path, case):
    data = _native_graph()
    if case == "missing":
        del data.train_mask
    elif case == "overlap":
        data.val_mask[0] = True
    elif case == "gap":
        data.test_mask[5] = False
    elif case == "non_boolean":
        data.train_mask = data.train_mask.to(torch.long)
    else:
        data.test_mask = torch.ones(5, dtype=torch.bool)

    with pytest.raises(ValueError, match="native inductive split"):
        load_or_create_inductive_split(data, f"native-invalid-{case}", root=tmp_path, split_strategy="native")


def _install_fake_ogb(monkeypatch, split_idx):
    class FakePygNodePropPredDataset:
        def __init__(self, name, root):
            self.name = name
            self.root = root
            self.data = Data(
                x=torch.randn(4, 3),
                y=torch.tensor([[0], [1], [0], [1]]),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            )

        def __getitem__(self, index):
            assert index == 0
            return self.data

        def get_idx_split(self):
            return split_idx

    ogb_module = types.ModuleType("ogb")
    nodeprop_module = types.ModuleType("ogb.nodeproppred")
    nodeprop_module.PygNodePropPredDataset = FakePygNodePropPredDataset
    ogb_module.nodeproppred = nodeprop_module
    monkeypatch.setitem(sys.modules, "ogb", ogb_module)
    monkeypatch.setitem(sys.modules, "ogb.nodeproppred", nodeprop_module)


def test_ogb_loader_converts_complete_official_split(monkeypatch):
    _install_fake_ogb(
        monkeypatch,
        {"train": torch.tensor([0, 1]), "valid": torch.tensor([2]), "test": torch.tensor([3])},
    )

    _, data = _load_ogb_node("ogbn-arxiv")

    assert data.y.shape == (4,)
    assert data.train_mask.dtype == data.val_mask.dtype == data.test_mask.dtype == torch.bool
    assert data.train_mask.tolist() == [True, True, False, False]
    assert data.val_mask.tolist() == [False, False, True, False]
    assert data.test_mask.tolist() == [False, False, False, True]


def test_ogb_loader_rejects_invalid_official_indices(monkeypatch):
    _install_fake_ogb(
        monkeypatch,
        {"train": torch.tensor([0, 1]), "valid": torch.tensor([1]), "test": torch.tensor([3])},
    )

    with pytest.raises(ValueError, match="OGB dataset ogbn-products has an invalid official split"):
        _load_ogb_node("ogbn-products")


def test_runner_reports_native_split_strategy(monkeypatch, tmp_path):
    data = _native_graph()
    monkeypatch.setattr(experiment_runner, "load_dataset", lambda dataset, device: (object(), data))

    result = experiment_runner.run(
        {
            "dataset": "native-runner-unit",
            "method": "mlp",
            "device": "cpu",
            "seed": 0,
            "split_strategy": "native",
            "split_root": str(tmp_path),
            "parameters": {"epochs": 1, "hidden_size": 4, "layers": 1, "dropout": 0.0, "batch_size": 8},
        }
    )

    assert result["split_strategy"] == "native"
    assert {name: result["partitions"][name]["nodes"] for name in ("train", "val", "test")} == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
