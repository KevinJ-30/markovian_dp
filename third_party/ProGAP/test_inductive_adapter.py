import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from core.methods.progap.edge import EdgeLevelProGAP
from core.methods.progap.node import NodeLevelProGAP
from core.modules.prog import ProgressiveModule, binary_auroc

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
            eval_mask=torch.tensor([True, False, True]),
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
            assert kwargs["monitor"] == "val/acc"
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
        "format": 2,
        "num_classes": 3,
        "binary": False,
        "primary_metric": "accuracy",
        "metric_ignore_label": None,
        "partitions": {"train": "train.pt", "val": "val.pt", "test": "test.pt"},
    }))
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(result_path))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_MULTILABEL", "0")
    monkeypatch.setenv("PROGAP_BINARY", "0")
    monkeypatch.setenv("PROGAP_PRIMARY_METRIC", "accuracy")
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


def test_progressive_fit_restores_global_stage_maxima_and_floor_schedule(monkeypatch, tmp_path):
    adapter = importlib.import_module("inductive_adapter")
    load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False}),
    )
    torch.manual_seed(6)
    nodes = torch.arange(17)
    graph = Data(
        x=torch.randn(17, 3), y=nodes % 2,
        edge_index=torch.stack((nodes, (nodes + 1) % 17)),
    )
    validation = graph.clone()
    validation.eval_mask = nodes % 3 != 0
    method = adapter.InductiveNodeLevelProGAP(
        num_classes=2, epsilon=8.0, delta=1 / 17, depth=2, batch_size=8,
        epochs=10, hidden_dim=64, dropout=0.0, device="cpu", verbose=False,
        eval_chunk_size=3, max_degree=2,
    )
    method.validation = adapter._prepare(validation)
    method.checkpoint_dir = tmp_path
    method.setup(adapter._prepare(graph))
    method.fit()

    assert method.updates_completed == 3 * 10 * (17 // 8)
    assert method.epochs_completed == 30
    assert method.composed_mechanism.get_approxDP(1 / 17) <= 8.0 + 1e-6
    for stage, selected in enumerate(method.stage_states):
        rows = [row for row in method.history if row["stage"] == stage]
        first_max = max(rows, key=lambda row: row["validation_metric"])
        assert selected["epoch"] == first_max["epoch"]
        assert selected["step"] == first_max["step"]
        checkpoint = torch.load(tmp_path / f"stage{stage}_best.pt")
        assert checkpoint["validation"]["score"] == first_max["validation_metric"]
        assert checkpoint["optimizer"]["state"]
    restored = adapter.evaluate_stage(method.classifier, method.validation, chunk_size=17)
    assert restored["score"] == method.best_validation["score"]
    assert restored["loss"] == pytest.approx(method.best_validation["loss"], abs=1e-6)
    final = torch.load(tmp_path / "checkpoint.pt")
    assert final["updates_completed"] == method.updates_completed
    assert len(final["stages"]) == 3
    # Rebuild each held-out stage from the selected final classifier, matching
    # the upstream pipeline while retaining every unscored context node.
    heldout = method.evaluate_partition(validation)
    assert heldout["scored_nodes"] == int(validation.eval_mask.sum())
    assert 0 <= heldout["score"] <= 1


class _ObjectiveModule(torch.nn.Module):
    root_losses = staticmethod(ProgressiveModule.root_losses)

    def __init__(self, logits, multilabel, binary=False):
        super().__init__()
        self.logits = logits
        self.multilabel = multilabel
        self.binary = binary
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


def test_progressive_binary_objective_uses_tie_correct_auroc_and_sigmoid():
    logits = torch.tensor([[0.0], [0.0], [2.0], [-1.0]])
    labels = torch.tensor([0, 1, 1, 0])
    module = _ObjectiveModule(logits, multilabel=False, binary=True)
    data = Data(x0=torch.ones(4, 1), y=labels, batch_nodes=torch.arange(4))

    loss, metrics = ProgressiveModule.step(module, data, "val")
    _, probabilities = ProgressiveModule.predict(module, data)

    assert torch.allclose(
        loss, F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
    )
    assert metrics["val/auroc"] == 87.5
    assert probabilities.shape == (4, 1)
    assert torch.allclose(probabilities, torch.sigmoid(logits))
    assert torch.isnan(binary_auroc(torch.tensor([0.1, 0.2]), torch.ones(2)))


def test_target_metrics_score_only_eval_roots_but_predict_full_graph():
    adapter = importlib.import_module("inductive_adapter")
    graph = Data(
        x=torch.arange(8, dtype=torch.float).reshape(4, 2),
        y=torch.tensor([0, 1, 19, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        eval_mask=torch.tensor([True, True, True, False]),
    )

    class Method:
        classifier = SimpleNamespace(binary=True)
        metric_ignore_label = 19

        def to_device(self, data):
            return data

        def predict(self):
            assert self.data.num_nodes == 4
            assert self.data.adj_t.values().numel() == 4
            return self.data.x, torch.tensor([[0.5], [0.5], [0.99], [0.0]])

    auroc, duplicate = adapter._metrics(Method(), graph)
    assert auroc == 0.5
    assert duplicate == auroc


def test_binary_manifest_uses_one_logit_auroc_and_loads_test_after_fit(
    monkeypatch, tmp_path
):
    adapter = importlib.import_module("inductive_adapter")
    graphs = {
        name: Data(
            x=torch.full((3, 2), float(index)),
            y=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            eval_mask=torch.tensor([True, True, False]),
        )
        for index, name in enumerate(("train", "val", "test"), start=1)
    }
    events = []

    def load(_path, _manifest, name):
        events.append(f"load-{name}")
        return graphs[name]

    class Mechanism:
        params = {"coeff_list": [1, 2]}

        def get_approxDP(self, _delta):
            return 7.9

    class FakeMethod:
        def __init__(self, **kwargs):
            assert kwargs["num_classes"] == 1
            assert kwargs["monitor"] == "val/auroc"
            self.classifier = SimpleNamespace()
            self.composed_mechanism = Mechanism()
            self.effective_delta = 5e-4
            self.noise_scale = 1.25

        def to_device(self, data):
            return data

        def setup(self, data):
            events.append(f"setup-{int(data.x[0, 0])}")

        def fit(self):
            assert "load-test" not in events
            events.append("fit")

    monkeypatch.setattr(adapter, "InductiveNodeLevelProGAP", FakeMethod)
    monkeypatch.setattr(adapter, "_load", load)
    monkeypatch.setattr(
        adapter,
        "_metrics",
        lambda _method, data: (
            (0.75, 0.75) if int(data.x[0, 0]) == 2 else (0.625, 0.625)
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format": 2,
        "num_classes": 2,
        "binary": True,
        "primary_metric": "auroc",
        "metric_ignore_label": None,
        "partitions": {"train": "train.pt", "val": "val.pt", "test": "test.pt"},
    }))
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(result_path))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_MULTILABEL", "0")
    monkeypatch.setenv("PROGAP_BINARY", "1")
    monkeypatch.setenv("PROGAP_PRIMARY_METRIC", "auroc")

    adapter.main()

    assert events.index("fit") < events.index("load-test")
    result = json.loads(result_path.read_text())
    assert result["metric"] == "auroc"
    assert result["validation_auroc"] == 0.75
    assert result["test_auroc"] == 0.625
    assert "validation_accuracy" not in result


def test_manifest_adapter_rejects_multilabel_flag_rank_mismatch(monkeypatch, tmp_path):
    adapter = importlib.import_module("inductive_adapter")
    graph = Data(
        x=torch.ones(3, 2),
        y=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format": 2,
        "num_classes": 2,
        "binary": False,
        "primary_metric": "accuracy",
        "metric_ignore_label": None,
        "partitions": {"train": "train.pt", "val": "val.pt", "test": "test.pt"},
    }))
    monkeypatch.setattr(adapter, "_load", lambda *_args: graph)
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_MULTILABEL", "0")
    monkeypatch.setenv("PROGAP_BINARY", "0")
    monkeypatch.setenv("PROGAP_PRIMARY_METRIC", "accuracy")
    with pytest.raises(ValueError, match="label rank"):
        adapter.main()


def test_environment_knobs_reach_real_constructor_and_adam(monkeypatch):
    adapter = importlib.import_module("inductive_adapter")
    from src.experiments.upstream import _target_environment

    load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False}),
    )
    parameters = {
        "target_epsilon": 8.0, "target_delta": 0.001, "hidden_dim": 7,
        "dropout": 0.25, "optimizer": "adam", "learning_rate": 0.037,
        "weight_decay": 0.12, "max_grad_norm": 0.3, "eval_chunk_size": 2,
    }
    encoded = _target_environment(
        "progap", {"parameters": parameters}, {},
        {"binary": False, "primary_metric": "accuracy", "metric_ignore_label": None},
    )
    for name, value in encoded.items():
        monkeypatch.setenv(name, value)
    method = adapter.InductiveNodeLevelProGAP(
        num_classes=3, epsilon=8.0, delta=0.001, depth=1, device="cpu",
        **adapter._constructor_options(),
    )
    model = method.classifier
    model.eval()
    embeddings, logits = model([torch.ones(5, 4)])
    assert embeddings.shape == (5, 7)
    assert logits.shape == (5, 3)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.037
    assert optimizer.param_groups[0]["weight_decay"] == 0.12
    assert any(isinstance(module, torch.nn.Dropout) and module.p == 0.25
               for module in model.modules())
    assert method.max_grad_norm == 0.3
    assert method.eval_chunk_size == 2


@pytest.mark.parametrize("name,value", [
    ("PROGAP_HIDDEN_DIM", "1.5"), ("PROGAP_EVAL_CHUNK_SIZE", "0"),
    ("PROGAP_LEARNING_RATE", "nan"), ("PROGAP_MAX_GRAD_NORM", "inf"),
    ("PROGAP_DROPOUT", "1"), ("PROGAP_WEIGHT_DECAY", "-1"),
    ("PROGAP_OPTIMIZER", "rmsprop"),
])
def test_constructor_environment_rejects_invalid_numerics(monkeypatch, name, value):
    adapter = importlib.import_module("inductive_adapter")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        adapter._constructor_options()


class _FixedStageLogits(torch.nn.Module):
    root_losses = staticmethod(ProgressiveModule.root_losses)

    def __init__(self, binary=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.current_stage = 0
        self.binary = binary

    def forward(self, xs):
        return xs[0], xs[0] + self.anchor * 0


def test_chunked_binary_metric_is_global_tied_auroc_and_filters_rows():
    adapter = importlib.import_module("inductive_adapter")
    logits = torch.tensor([[0.0], [0.0], [2.0], [1.0], [20.0], [-20.0]])
    graph = Data(
        x=logits, x0=logits, y=torch.tensor([0, 1, 0, 1, 19, 1]),
        eval_mask=torch.tensor([True, True, True, True, True, False]),
    )
    model = _FixedStageLogits(binary=True)
    chunked = adapter.evaluate_stage(model, graph, 2, metric_ignore_label=19)
    full = adapter.evaluate_stage(model, graph, 100, metric_ignore_label=19)
    assert chunked["score"] == full["score"] == 0.375
    assert chunked["loss"] == pytest.approx(full["loss"], abs=1e-7)
    assert chunked["scored_nodes"] == 4
    # A per-chunk AUROC average would be (0.5 + 0) / 2 = 0.25.
    assert chunked["score"] != 0.25


@pytest.mark.parametrize("multilabel", [False, True])
def test_chunked_counts_loss_and_context_embeddings_match_full(multilabel):
    adapter = importlib.import_module("inductive_adapter")
    logits = torch.tensor([[3., -2.], [-1., 2.], [1., 2.], [-2., -1.], [4., 1.]])
    labels = (torch.tensor([[1., 0.], [1., 1.], [0., 1.], [1., 1.], [0., 0.]])
              if multilabel else torch.tensor([0, 1, 0, 1, 1]))
    graph = Data(
        x=logits, x0=logits, y=labels,
        eval_mask=torch.tensor([True, True, False, True, False]),
    )
    model = _FixedStageLogits()
    chunked = adapter.evaluate_stage(model, graph, 2)
    full = adapter.evaluate_stage(model, graph, 100)
    assert chunked["score"] == full["score"]
    assert chunked["loss"] == pytest.approx(full["loss"], abs=1e-7)
    assert chunked["scored_nodes"] == 3
    embeddings = adapter.stage_embeddings(model, graph, 2)
    assert torch.equal(embeddings, logits)  # Includes unscored context nodes.
