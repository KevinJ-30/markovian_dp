import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from core.methods.progap.edge import EdgeLevelProGAP
from core.methods.progap.node import NodeLevelProGAP
from core.modules.prog import ProgressiveModule

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


def test_manifest_adapter_uses_train_for_setup_and_validation_for_fit(monkeypatch, tmp_path):
    adapter = importlib.import_module("inductive_adapter")
    graphs = {
        name: Data(
            x=torch.full((3, 2), float(index)),
            y=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        for index, name in enumerate(("train", "val", "test"), start=1)
    }
    calls = []

    class Mechanism:
        params = {"coeff_list": [1, 2]}

        def get_approxDP(self, delta):
            assert delta == 5e-4
            return 7.9

    class FakeMethod:
        def __init__(self, **kwargs):
            assert kwargs["num_classes"] == 3
            assert kwargs["epsilon"] == 8.0
            assert kwargs["delta"] == 5e-4
            self.classifier = SimpleNamespace(multilabel=None)
            self.composed_mechanism = Mechanism()
            self.effective_delta = 5e-4
            self.noise_scale = 1.25

        def to_device(self, data):
            return data

        def setup(self, data):
            calls.append(("setup", int(data.x[0, 0])))

        def fit(self):
            calls.append(("fit", int(self.validation.x[0, 0])))

    monkeypatch.setattr(adapter, "InductiveNodeLevelProGAP", FakeMethod)
    monkeypatch.setattr(adapter, "_load", lambda _path, _manifest, name: graphs[name])

    def metrics(_method, data):
        calls.append(("metrics", int(data.x[0, 0])))
        return 0.5, 0.4

    monkeypatch.setattr(adapter, "_metrics", metrics)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "num_classes": 3,
        "partitions": {"train": "train.pt", "val": "val.pt", "test": "test.pt"},
    }))
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(result_path))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_MULTILABEL", "0")
    monkeypatch.setenv("PROGAP_EPOCHS", "1")
    monkeypatch.setenv("PROGAP_BATCH_SIZE", "2")
    monkeypatch.setenv("PROGAP_MAX_DEGREE", "5")
    monkeypatch.setenv("PROGAP_DEPTH", "1")

    adapter.main()

    assert calls == [
        ("setup", 1),
        ("fit", 2),
        ("metrics", 2),
        ("metrics", 3),
    ]
    result = json.loads(result_path.read_text())
    assert result["privacy"]["total"]["epsilon"] == 7.9
    assert result["privacy"]["total"]["delta"] == 5e-4
    assert result["calibration"]["noise_std"] == 1.25


def test_progressive_fit_selects_every_stage_on_distinct_validation_graph():
    adapter = importlib.import_module("inductive_adapter")
    train = adapter._prepare(Data(
        x=torch.ones(3, 2),
        y=torch.tensor([0, 1, 0]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    ))
    validation = adapter._prepare(Data(
        x=torch.full((3, 2), 2.0),
        y=torch.tensor([1, 0, 1]),
        edge_index=torch.tensor([[0, 2], [2, 1]]),
    ))
    fit_validation_markers = []

    class Classifier:
        def __init__(self):
            self.stages = []

        def set_stage(self, stage):
            self.stages.append(stage)

    class Trainer:
        def predict(self, dataloader):
            return dataloader.data.x, torch.zeros(dataloader.data.num_nodes, 2)

        def fit(self, model, train_dataloader, val_dataloader):
            assert train_dataloader == "train-loader"
            fit_validation_markers.append(float(val_dataloader.data.x[0, 0]))
            return {"val/acc": torch.tensor(1.0)}

    method = SimpleNamespace(
        data=train,
        validation=validation,
        num_stages=2,
        classifier=Classifier(),
        trainer=Trainer(),
        nap=lambda embeddings, adjacency: embeddings + 1,
        configure_trainer=Trainer,
        data_loader=lambda phase: "train-loader",
    )
    metrics = adapter.InductiveNodeLevelProGAP.fit(method)
    assert fit_validation_markers == [2.0, 2.0]
    assert method.classifier.stages == [0, 1]
    assert method.data.ready
    assert metrics["val/acc"] == 1


class _ObjectiveModule(torch.nn.Module):
    root_losses = staticmethod(ProgressiveModule.root_losses)

    def __init__(self, logits, multilabel):
        super().__init__()
        self.logits = logits
        self.multilabel = multilabel
        self.current_stage = 0

    def forward(self, xs):
        return xs[-1], self.logits


def test_progressive_categorical_objective_and_prediction():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    module = _ObjectiveModule(logits, multilabel=False)
    data = Data(x0=torch.ones(2, 1), y=torch.tensor([0, 1]), batch_nodes=torch.arange(2))
    loss, metrics = ProgressiveModule.step(module, data, "val")
    _, probabilities = ProgressiveModule.predict(module, data)
    assert torch.allclose(loss, F.cross_entropy(logits, data.y))
    assert metrics["val/acc"] == 100
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_progressive_multilabel_objective_metric_and_prediction():
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    module = _ObjectiveModule(logits, multilabel=True)
    data = Data(x0=torch.ones(2, 1), y=labels, batch_nodes=torch.arange(2))
    loss, metrics = ProgressiveModule.step(module, data, "val")
    _, probabilities = ProgressiveModule.predict(module, data)
    expected = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ).mean(dim=1).mean()
    assert torch.allclose(loss, expected)
    assert metrics["val/micro_f1"] == 50
    assert torch.allclose(probabilities, torch.sigmoid(logits))


def test_manifest_adapter_rejects_multilabel_flag_rank_mismatch(monkeypatch, tmp_path):
    adapter = importlib.import_module("inductive_adapter")
    graph = Data(
        x=torch.ones(3, 2),
        y=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "num_classes": 2,
        "partitions": {"train": "train.pt", "val": "val.pt", "test": "test.pt"},
    }))
    monkeypatch.setattr(adapter, "_load", lambda *_args: graph)
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_MULTILABEL", "0")
    with pytest.raises(ValueError, match="label rank"):
        adapter.main()
