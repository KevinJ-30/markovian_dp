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


@pytest.mark.parametrize("regression", [False, True])
def test_progressive_fit_restores_global_stage_maxima_and_floor_schedule(
    monkeypatch, tmp_path, regression
):
    adapter = importlib.import_module("inductive_adapter")
    load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False}),
    )
    torch.manual_seed(6)
    nodes = torch.arange(17)
    graph = Data(
        x=torch.randn(17, 3), y=nodes.float() + 1000 if regression else nodes % 2,
        edge_index=torch.stack((nodes, (nodes + 1) % 17)),
    )
    validation = graph.clone()
    validation.eval_mask = nodes % 3 != 0
    method = adapter.InductiveNodeLevelProGAP(
        num_classes=1 if regression else 2,
        epsilon=8.0, delta=1 / 17, depth=2, batch_size=8,
        epochs=10, hidden_dim=64, dropout=0.0, device="cpu", verbose=False,
        eval_chunk_size=3, max_degree=2,
        monitor="val/r2" if regression else "val/acc",
    )
    method.classifier.regression = regression
    method.validation = adapter._prepare(validation)
    method.checkpoint_dir = tmp_path
    method.setup(adapter._prepare(graph))
    fit_result = method.fit()

    assert method.updates_completed == 3 * 10 * (17 // 8)
    assert method.epochs_completed == 30
    assert method.composed_mechanism.get_approxDP(1 / 17) <= 8.0 + 1e-6
    assert all("loss" not in row and "training_loss" not in row for row in method.history)
    if regression:
        assert all(row["validation_metric"] < 0 for row in method.history)
        assert fit_result["val/r2"] == method.best_validation["score"]
    for stage, selected in enumerate(method.stage_states):
        rows = [row for row in method.history if row["stage"] == stage]
        first_max = max(rows, key=lambda row: row["validation_metric"])
        assert selected["epoch"] == first_max["epoch"]
        assert selected["step"] == first_max["step"]
        checkpoint = torch.load(tmp_path / f"stage{stage}_best.pt")
        assert checkpoint["validation"]["score"] == first_max["validation_metric"]
        assert checkpoint["optimizer"]["state"]
    restored = adapter.evaluate_stage(method.classifier, method.validation, chunk_size=17)
    assert restored["score"] == pytest.approx(method.best_validation["score"])
    # Chunk shapes can change float32 GEMM rounding; MSE scales with the targets.
    assert restored["loss"] == pytest.approx(
        method.best_validation["loss"], rel=1e-7, abs=1e-6
    )
    final = torch.load(tmp_path / "checkpoint.pt")
    assert final["updates_completed"] == method.updates_completed
    assert len(final["stages"]) == 3
    # Rebuild each held-out stage from the selected final classifier, matching
    # the upstream pipeline while retaining every unscored context node.
    heldout = method.evaluate_partition(validation)
    assert heldout["scored_nodes"] == int(validation.eval_mask.sum())
    if regression:
        assert heldout["metric"] == "r2"
        assert heldout["score"] < 0
    else:
        assert 0 <= heldout["score"] <= 1


class _ObjectiveModule(torch.nn.Module):
    root_losses = staticmethod(ProgressiveModule.root_losses)

    def __init__(self, logits, multilabel, binary=False, regression=False):
        super().__init__()
        self.logits = logits
        self.multilabel = multilabel
        self.binary = binary
        self.regression = regression
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


def test_progressive_regression_uses_per_root_squared_error_and_raw_predictions():
    predictions = torch.tensor([[-2.0], [3.0]])
    module = _ObjectiveModule(predictions, multilabel=False, regression=True)
    data = Data(
        x0=torch.ones(2, 1), y=torch.tensor([1.0, 2.0]), batch_nodes=torch.arange(2),
    )
    loss, metrics = ProgressiveModule.step(module, data, "val")
    _, output = ProgressiveModule.predict(module, data)
    assert torch.equal(
        ProgressiveModule.root_losses(predictions, data.y, regression=True),
        torch.tensor([9.0, 1.0]),
    )
    assert loss == 5
    assert metrics["val/r2"] == -19
    assert torch.equal(output, predictions)


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

    def __init__(self, binary=False, regression=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.current_stage = 0
        self.binary = binary
        self.regression = regression

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


def test_chunked_regression_uses_stable_whole_scored_split_r2():
    adapter = importlib.import_module("inductive_adapter")
    offset = 1e12
    labels = torch.tensor(
        [offset, offset + 2, -99, offset + 4, offset + 8, 4e12],
        dtype=torch.float64,
    )
    predictions = torch.tensor(
        [offset + 3, offset - 4, 0, offset, offset + 1, -4e12],
        dtype=torch.float64,
    ).unsqueeze(-1)
    graph = Data(
        x=predictions, x0=predictions, y=labels,
        eval_mask=torch.tensor([True, True, True, True, True, False]),
    )
    model = _FixedStageLogits(regression=True)
    for chunk_size in (1, 2, 3, 100):
        result = adapter.evaluate_stage(model, graph, chunk_size, metric_ignore_label=-99)
        assert result["metric"] == "r2"
        assert result["scored_nodes"] == 4
        assert result["loss"] == pytest.approx(110 / 4)
        assert result["score"] == pytest.approx(1 - 110 / 35)


@pytest.mark.parametrize("prediction,expected", [(7.0, 1.0), (-2.0, 0.0)])
def test_regression_constant_targets_match_sklearn(prediction, expected):
    adapter = importlib.import_module("inductive_adapter")
    predictions = torch.full((4, 1), prediction)
    graph = Data(
        x=predictions, x0=predictions, y=torch.full((4,), 7.0),
        eval_mask=torch.ones(4, dtype=torch.bool),
    )
    result = adapter.evaluate_stage(_FixedStageLogits(regression=True), graph, 1)
    assert result["score"] == expected


def test_regression_fallback_scores_raw_predictions_only_on_selected_roots():
    adapter = importlib.import_module("inductive_adapter")
    graph = Data(
        x=torch.tensor([[-2.0], [4.0], [1000.0]]),
        y=torch.tensor([2.0, 4.0, -1000.0]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        eval_mask=torch.tensor([True, True, False]),
    )

    class Method:
        classifier = SimpleNamespace(regression=True)

        def to_device(self, data):
            return data

        def predict(self):
            return self.data.x, self.data.x

    score, legacy_score = adapter._metrics(Method(), graph)
    assert score == legacy_score == -7.0


@pytest.mark.parametrize("binary,multilabel", [(True, False), (False, True)])
def test_regression_manifest_rejects_conflicting_task_flags(
    monkeypatch, tmp_path, binary, multilabel
):
    adapter = importlib.import_module("inductive_adapter")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format": 2, "num_classes": 1, "binary": binary, "primary_metric": "r2",
        "partitions": {},
    }))
    monkeypatch.setenv("PARTITION_MANIFEST", str(manifest_path))
    monkeypatch.setenv("RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("PROGAP_TARGET_EPSILON", "8")
    monkeypatch.setenv("PROGAP_TARGET_DELTA", "0.0005")
    monkeypatch.setenv("PROGAP_BINARY", str(int(binary)))
    monkeypatch.setenv("PROGAP_MULTILABEL", str(int(multilabel)))
    monkeypatch.setenv("PROGAP_PRIMARY_METRIC", "r2")
    with pytest.raises(ValueError):
        adapter.main()


def test_bootstrap_preserves_sigmoid_ties_and_scored_rows():
    adapter = importlib.import_module("inductive_adapter")
    logits = torch.tensor([[80.0], [90.0], [-100.0]])
    graph = Data(x=logits, x0=logits, y=torch.tensor([1, 0, 19]),
                 eval_mask=torch.ones(3, dtype=torch.bool))
    bootstrap = adapter.BootstrapMetrics(
        "auroc", adapter.BootstrapConfig(n_resamples=100, seed=2),
        metrics=("auroc",))
    result = adapter.evaluate_stage(
        _FixedStageLogits(binary=True), graph, 1, metric_ignore_label=19,
        bootstrap=bootstrap)
    interval = bootstrap.compute()
    assert result["score"] == 0.5  # Raw-logit ranking would incorrectly give 0.
    assert interval["n_observations"] == 2
    assert interval["metrics"]["auroc"]["lower"] == 0.5
    assert interval["metrics"]["auroc"]["upper"] == 0.5
    assert 0 < interval["metrics"]["auroc"]["valid_resamples"] < 100
