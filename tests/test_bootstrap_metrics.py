"""Bootstrap contracts: metric recomputation, node sampling, and test-only use."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models.bootstrap import BootstrapConfig, BootstrapMetrics
from src.models.objectives import _task_metric
from src.models.multilabel_mechanism import _micro_auroc
from src.training.baselines import BaselineConfig, BaselineTrainer


def _interval(values, confidence):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    alpha = (1 - confidence) / 2
    return np.quantile(values, [alpha, 1 - alpha]), len(values)


@pytest.mark.parametrize("task,logits,labels,metric_names", [
    ("accuracy", [[3., 0.], [0., 3.], [0., 3.]], [0, 0, 1], ("accuracy", "macro_f1")),
    ("auroc", [-2., 0., 0., 2.], [0, 0, 1, 1], ("auroc", "accuracy")),
    ("r2", [10., 14., 9.], [10., 11., 12.], ("r2",)),
])
def test_percentiles_match_explicit_resampled_metric_recomputation(task, logits, labels, metric_names):
    logits, labels = torch.tensor(logits), torch.tensor(labels)
    config = BootstrapConfig(confidence_level=.8, n_resamples=91, seed=13)
    calculator = BootstrapMetrics(task, config)
    calculator.update(logits, labels)
    actual = calculator.compute()["metrics"]
    # These rows are distinct and already in lexicographic statistic order.
    # Expand integer node multiplicities and recompute the ordinary metric.
    rng = np.random.default_rng(config.seed)
    expected = {name: [] for name in metric_names}
    for _ in range(config.n_resamples):
        counts = rng.multinomial(len(labels), np.ones(len(labels)) / len(labels))
        indices = torch.tensor(np.repeat(np.arange(len(labels)), counts))
        scores = _task_metric(logits[indices], labels[indices], False,
                              regression=task == "r2", binary=task == "auroc")
        for index, name in enumerate(metric_names):
            expected[name].append(scores[index])
    for name in metric_names:
        bounds, valid = _interval(expected[name], config.confidence_level)
        assert [actual[name]["lower"], actual[name]["upper"]] == pytest.approx(bounds, abs=1e-6)
        assert actual[name]["valid_resamples"] == valid


def test_multilabel_resamples_nodes_not_individual_labels():
    logits = torch.tensor([[2., -1., 0.], [-2., 3., 4.], [1., 2., -3.]])
    labels = torch.tensor([[1., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    config = BootstrapConfig(.9, 123, 9)
    calculator = BootstrapMetrics("micro_f1", config, metrics=("micro_f1", "micro_auroc"))
    calculator.update(logits[:1], labels[:1])
    calculator.update(logits[1:], labels[1:])
    actual = calculator.compute()
    assert actual["n_observations"] == 3
    rng = np.random.default_rng(config.seed)
    f1, auc = [], []
    for _ in range(config.n_resamples):
        counts = rng.multinomial(3, [1/3] * 3)
        index = torch.tensor(np.repeat(np.arange(3), counts))
        f1.append(_task_metric(logits[index], labels[index], True)[0])
        auc.append(_micro_auroc(logits[index], labels[index]))
    for name, scores in (("micro_f1", f1), ("micro_auroc", auc)):
        bounds, valid = _interval(scores, config.confidence_level)
        result = actual["metrics"][name]
        assert [result["lower"], result["upper"]] == pytest.approx(bounds)
        assert result["valid_resamples"] == valid


def test_undefined_auroc_and_empty_test_have_null_bounds():
    for scores, targets in ((torch.ones(4), torch.ones(4)), (torch.empty(0), torch.empty(0))):
        calculator = BootstrapMetrics("auroc", BootstrapConfig(n_resamples=20))
        calculator.update(scores, targets)
        assert calculator.compute()["metrics"]["auroc"] == {
            "lower": None, "upper": None, "valid_resamples": 0}


@pytest.mark.parametrize("perfect", [False, True])
def test_constant_target_r2_retains_finite_convention(perfect):
    labels = torch.full((4,), 1e12, dtype=torch.float64)
    calculator = BootstrapMetrics("r2", BootstrapConfig(n_resamples=20))
    calculator.update(labels + (0 if perfect else 1), labels)
    result = calculator.compute()["metrics"]["r2"]
    assert result == {"lower": float(perfect), "upper": float(perfect), "valid_resamples": 20}


def test_threshold_and_zero_denominator_conventions_are_preserved():
    labels = torch.ones((3, 2))
    for inclusive, expected in ((False, 0.), (True, 1.)):
        calculator = BootstrapMetrics("micro_f1", BootstrapConfig(n_resamples=20),
                                      inclusive_threshold=inclusive)
        calculator.update(torch.zeros_like(labels), labels)
        result = calculator.compute()["metrics"]["micro_f1"]
        assert result["lower"] == result["upper"] == expected
    calculator = BootstrapMetrics("micro_f1", BootstrapConfig(n_resamples=20), zero_division=0.)
    calculator.update(-torch.ones_like(labels), torch.zeros_like(labels))
    assert calculator.compute()["metrics"]["micro_f1"]["lower"] == 0.


def test_chunking_determinism_and_global_rng_isolation():
    logits = torch.tensor([-1., 0., 0., 2., 3.])
    labels = torch.tensor([0, 1, 0, 1, 1])
    config = BootstrapConfig(n_resamples=51, seed=4)
    whole, chunked = BootstrapMetrics("auroc", config), BootstrapMetrics("auroc", config)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    whole.update(logits, labels)
    chunked.update(logits[:2], labels[:2])
    chunked.update(logits[2:], labels[2:])
    assert whole.compute() == chunked.compute()
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


@pytest.mark.parametrize("settings", [
    {"confidence_level": 95}, {"confidence_level": float("nan")},
    {"n_resamples": 1.5}, {"n_resamples": -1}, {"n_resamples": True}, {"seed": -1},
])
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        BootstrapConfig(**settings)


@pytest.mark.parametrize("method", ["mlp", "dp_mlp"])
def test_only_masked_final_test_is_bootstrapped_without_changing_training(monkeypatch, method):
    def partition(label):
        data = Data(x=torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10,
                    y=torch.full((6,), label), edge_index=torch.empty((2, 0), dtype=torch.long))
        return SimpleNamespace(data=data, stats={}, eval_mask=torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool))
    split = SimpleNamespace(train=partition(0), val=partition(1), test=partition(2), num_classes=3)
    observed = []
    original = BootstrapMetrics.update
    def update(self, logits, labels):
        observed.append(labels.clone())
        return original(self, logits, labels)
    monkeypatch.setattr(BootstrapMetrics, "update", update)
    options = dict(method=method, epochs=2, hidden_size=4, layers=1, batch_size=3, seed=7)
    enabled = BaselineTrainer(BaselineConfig(**options, bootstrap_resamples=23)).fit(split)
    enabled_rng = torch.random.get_rng_state().clone()
    disabled = BaselineTrainer(BaselineConfig(**options, bootstrap_resamples=0)).fit(split)
    assert len(observed) == 1
    assert observed[0].tolist() == [2, 2, 2]
    assert enabled["test_confidence_intervals"]["n_observations"] == 3
    assert "test_confidence_intervals" not in disabled
    for key in ("test_accuracy", "test_macro_f1", "validation_accuracy", "validation_macro_f1"):
        assert enabled[key] == disabled[key]
    assert torch.equal(enabled_rng, torch.random.get_rng_state())
