import json
from types import SimpleNamespace

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


def test_dpgnn_rejects_retired_mae_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": 2, "num_classes": 1, "primary_metric": "mae", "binary": False,
        "partitions": {name: f"{name}.pt" for name in ("train", "val", "test")},
    }))
    with pytest.raises(ValueError, match="unsupported primary_metric"):
        _load_partitions(manifest)
