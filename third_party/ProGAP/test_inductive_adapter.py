import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from core.methods.progap.edge import EdgeLevelProGAP
from core.methods.progap.node import NodeLevelProGAP

def test_numeric_delta_calibrates_node_and_edge_safely(monkeypatch):
    load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False}),
    )
    node = NodeLevelProGAP(num_classes=2, epsilon=8.0, delta=5e-4, depth=1, batch_size=2)
    node.num_train_nodes = 10
    node.trainer = SimpleNamespace(epochs=1)
    node.calibrate()
    assert node.effective_delta == 5e-4
    assert node.composed_mechanism.get_approxDP(node.effective_delta) <= node.epsilon

    edge = EdgeLevelProGAP(num_classes=2, epsilon=8.0, delta=5e-4, depth=1)
    edge.num_edges = 20
    edge.calibrate()
    assert edge.effective_delta == 5e-4
    assert edge.composed_mechanism.get_approxDP(edge.effective_delta) <= edge.epsilon


def test_manifest_adapter_trains_only_train_partition(monkeypatch, tmp_path):
    adapter = importlib.import_module("inductive_adapter")
    graphs = {
        name: Data(x=torch.ones(3, 2), y=torch.tensor([0, 1, 0]), edge_index=torch.tensor([[0, 1], [1, 2]]))
        for name in ("train", "val", "test")
    }
    calls = []

    class Mechanism:
        params = {"coeff_list": [1, 2]}

        def get_approxDP(self, delta):
            assert delta == 5e-4
            return 7.9

    class FakeMethod:
        def __init__(self, **kwargs):
            assert kwargs["epsilon"] == 8.0
            assert kwargs["delta"] == 5e-4
            self.composed_mechanism = Mechanism()
            self.effective_delta = 5e-4
            self.noise_scale = 1.25
        def run(self, data):
            calls.append(data)

    monkeypatch.setattr(adapter, "NodeLevelProGAP", FakeMethod)
    monkeypatch.setattr(adapter, "_load", lambda _manifest, name: graphs[name])
    monkeypatch.setattr(adapter, "_metrics", lambda _method, data: (0.5, 0.4))
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("PARTITION_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("RESULT_PATH", str(result_path))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_EPOCHS", "1")
    monkeypatch.setenv("PROGAP_BATCH_SIZE", "2")
    monkeypatch.setenv("PROGAP_MAX_DEGREE", "5")
    monkeypatch.setenv("PROGAP_DEPTH", "1")

    adapter.main()

    assert len(calls) == 1
    assert calls[0].num_nodes == graphs["train"].num_nodes
    result = json.loads(result_path.read_text())
    assert result["privacy"]["total"]["epsilon"] == 7.9
    assert result["privacy"]["total"]["delta"] == 5e-4
    assert result["calibration"]["noise_std"] == 1.25
