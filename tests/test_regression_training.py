import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.experiments.dpgnn_adapter import _load_partitions
from src.training.baselines import BaselineConfig, BaselineTrainer
import src.training.dpar as dpar_module


class _ConstantRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index=None):
        return self.bias.expand(x.size(0), 1)


@pytest.mark.parametrize("method", ["mlp", "dpar"])
def test_negative_r2_selects_and_restores_best_checkpoint(monkeypatch, method):
    data = Data(
        x=torch.ones(2, 1), y=torch.tensor([0.0, 1.0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    split = SimpleNamespace(
        train=SimpleNamespace(data=data.clone(), stats={"nodes": 2}),
        val=SimpleNamespace(data=data.clone()),
        test=SimpleNamespace(data=data.clone()), num_classes=1,
    )
    model = _ConstantRegressor()
    # Keep real scoring/checkpointing, but prescribe three trained checkpoints:
    # validation R² is -49, -9, -25; neither the first nor the last is best.
    checkpoints = iter([4.0, 2.0, 3.0])

    @torch.no_grad()
    def step(optimizer, closure=None):
        model.bias.fill_(next(checkpoints))

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    if method == "dpar":
        monkeypatch.setattr(dpar_module, "DPARMLP", lambda *args: model)
        trainer = dpar_module.DPARTrainer(dpar_module.DPARConfig(
            regression=True, epochs=3, batch_size=2, sampled_train_rate=1.0,
            ppr_num=2, topk=2, inference_steps=0,
        ))
    else:
        trainer = BaselineTrainer(BaselineConfig(
            method=method, regression=True, epochs=3, batch_size=2,
        ))
        monkeypatch.setattr(trainer, "_model", lambda *args: model)
    result = trainer.fit(split)
    assert result["validation_accuracy"] == pytest.approx(-9.0)
    assert result["test_accuracy"] == pytest.approx(-9.0)
    assert model.bias.item() == 2.0


@pytest.fixture
def heterpoisson(monkeypatch):
    # The adapter's scoring and checkpointing do not need upstream's optional
    # training/accounting dependencies. Isolate those imports, not the scorer.
    datasets = ModuleType("datasets")
    datasets.model = ModuleType("datasets.model")
    scheduler = ModuleType("train_scheduler")
    scheduler.Phase = SimpleNamespace(TRAIN="train")
    privacy = ModuleType("privacy")
    privacy.sampling = ModuleType("privacy.sampling")
    for name, module in {
        "datasets": datasets, "datasets.model": datasets.model,
        "train_scheduler": scheduler, "privacy": privacy,
        "privacy.sampling": privacy.sampling,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[1] / "third_party/PNPiGNNs"
        / "Preserving_Node_level_Privacy_in_Graph_Neural_Networks/inductive_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("heterpoisson_regression_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root_batch(predictions, truth):
    predictions = torch.as_tensor(predictions, dtype=torch.float64)
    truth = torch.as_tensor(truth, dtype=torch.float64)
    # Deliberately wrong neighbor predictions must never affect root scoring.
    x = torch.stack((predictions, torch.full_like(predictions, 1e6)), dim=1).unsqueeze(-1)
    targets = torch.stack((truth, torch.full_like(truth, -1e6)), dim=1)
    return x, targets


def test_heterpoisson_scores_global_center_node_r2_across_batches(heterpoisson):
    from sklearn.metrics import r2_score

    truth = torch.tensor([0., 1., 4., 5., 9.], dtype=torch.float64) + 1e9
    predictions = torch.tensor([-8., 2., 2., 8., 4.], dtype=torch.float64) + 1e9
    scheduler = SimpleNamespace(model=torch.nn.Identity(), device="cpu")
    batches = [None] + [
        _root_batch(predictions[start:end], truth[start:end])
        for start, end in [(0, 1), (1, 3), (3, 5)]
    ]
    expected = r2_score(truth.numpy(), predictions.numpy())
    assert expected < 0
    assert heterpoisson._r2_score(scheduler, batches) == pytest.approx(expected, abs=1e-7)
    assert heterpoisson._r2_score(
        scheduler, [_root_batch(predictions, truth)]
    ) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("predictions,truth,expected", [
    ([], [], float("nan")),
    ([2.0], [2.0], float("nan")),
    ([2.0, 2.0], [2.0, 2.0], 1.0),
    ([2.0, 3.0], [2.0, 2.0], 0.0),
])
def test_heterpoisson_r2_small_and_constant_targets(heterpoisson, predictions, truth, expected):
    scheduler = SimpleNamespace(model=torch.nn.Identity(), device="cpu")
    actual = heterpoisson._r2_score(scheduler, [None, _root_batch(predictions, truth)])
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def test_heterpoisson_restores_best_negative_r2_and_worker_weights(heterpoisson):
    model = _ConstantRegressor()
    loader = [_root_batch([0.0, 0.0], [0.0, 1.0])]
    scheduler = SimpleNamespace(
        model=model, device="cpu", train_loader=object(),
        val_loader=loader, test_loader=loader,
        worker_param_func=[model.bias.detach().clone()],
    )

    @torch.no_grad()
    def one_epoch(**kwargs):
        model.bias.fill_([4.0, 2.0, 3.0][scheduler.epoch])
        scheduler.worker_param_func[0].copy_(model.bias)

    scheduler.one_epoch = one_epoch
    validation, test = heterpoisson._train_regression(scheduler, 3)
    assert validation == pytest.approx(-9.0)
    assert test == pytest.approx(-9.0)
    assert model.bias.item() == 2.0
    assert scheduler.worker_param_func[0].item() == 2.0


def test_dpgnn_rejects_retired_mae_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": 2, "num_classes": 1, "primary_metric": "mae", "binary": False,
        "partitions": {name: f"{name}.pt" for name in ("train", "val", "test")},
    }))
    with pytest.raises(ValueError, match="unsupported primary_metric"):
        _load_partitions(manifest)
