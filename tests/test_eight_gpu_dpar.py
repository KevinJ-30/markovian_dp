"""Uncertainty-bearing DPAR study and maintained callback boundaries."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.models.baselines import DPARMLP
from src.models.objectives import _task_loss
from src.training.dpar import DPARConfig, DPARTrainer, private_ista_ppr


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "results/eight_gpu_domain_graphsaint/dpar/run.py"
    spec = importlib.util.spec_from_file_location("eight_gpu_dpar_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_population_controls_delta_not_ppr_release_count(runner):
    from study_common import resolve_cell
    protocol = dict(id="fixture", dataset="twitch-explicit", split_strategy="domain",
                    split_seed=7, domain_split={"train": ["de"], "val": ["engb"], "test": ["engb"]})
    prepared = {"implementation_hash": "fixture", "protocols": {"fixture": {
        "n_train": 9498, "task": {"binary": True, "num_classes": 2},
        "manifest": "/unused/manifest.json", "split_fingerprint": "fixture",
    }}}
    cell = resolve_cell(protocol, "dpar", 8, phase="smoke", prepared=prepared)
    config = runner.resolve_config(cell)
    assert config.sampled_train_nodes == 1024
    assert config.ppr_num == 70
    assert config.target_delta == 1 / 9498
    assert cell["steps"] == 1
    with pytest.raises(ValueError, match="population"):
        runner.resolve_config({**cell, "sampled_train_nodes": 70})
    prepared["protocols"]["fixture"]["n_train"] = 20001
    larger = resolve_cell(protocol, "dpar", 8, prepared=prepared)
    assert runner.resolve_config(larger).sampled_train_nodes == 1801
    assert larger["steps"] == 10
    assert larger["delta"] == 1 / 20001
    with pytest.raises(ValueError):
        runner.resolve_config({**larger, "steps": 20})
    minibatched = resolve_cell(protocol, "dpar", 8, prepared=prepared,
                               overrides={"batch_size": 60})
    assert runner.resolve_config(minibatched).batch_size == 60
    assert minibatched["steps"] == 20
    assert minibatched["sample_rate"] == 60 / 70
    with pytest.raises(ValueError):
        runner.resolve_config({**minibatched, "sample_rate": 60 / 1801})


def test_ppr_progress_observation_does_not_change_private_release():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    roots = torch.tensor([0, 2])
    config = DPARConfig(dp_ppr=True, ppr_noise=0.04, rho=0.02, topk=2)
    seed = 729
    expected = private_ista_ppr(edges, 4, roots, config, torch.device("cpu"),
                                torch.Generator().manual_seed(seed))
    events = []
    actual = private_ista_ppr(edges, 4, roots, config, torch.device("cpu"),
                              torch.Generator().manual_seed(seed),
                              progress_callback=events.append)
    torch.testing.assert_close(actual.to_dense(), expected.to_dense(), rtol=0, atol=0)
    assert events[-1]["ppr_releases_completed"] == len(roots)
    assert events[-1]["ppr_releases_total"] == len(roots)
    assert [event["ppr_root"] for event in events] == roots.tolist()
    assert all(event["ppr_iterations"] > 0 for event in events)


@pytest.mark.parametrize("task", ["categorical", "multilabel", "binary"])
def test_private_step_returns_preupdate_loss_and_preserves_exact_clipped_adam(task):
    torch.manual_seed(62)
    binary, multilabel = task == "binary", task == "multilabel"
    classes = 1 if binary else 2
    model = DPARMLP(3, classes, hidden=4, layers=2, dropout=0)
    reference = deepcopy(model)
    x = torch.tensor([[1., 2., -1.], [-0.5, 0.8, 1.2], [0.2, -1., 2.]])
    labels = (torch.tensor([[1., 0.], [0., 1.], [1., 1.]]) if multilabel
              else torch.tensor([0, 1, 1]))
    dense_ppr = torch.tensor([[0.7, 0.3, 0.], [0., 1., 0.], [0.2, 0.3, 0.5]])
    roots = torch.tensor([0, 2])
    config = DPARConfig(dp_sgd=True, sgd_noise=0.12, sgd_clip=0.1,
                        binary=binary, multilabel=multilabel)
    actual_adam = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    expected_adam = torch.optim.Adam(reference.parameters(), lr=0.01, weight_decay=0.0001)
    expected_logits = dense_ppr[roots] @ reference(x)
    expected_loss = _task_loss(expected_logits, labels[roots], multilabel, binary=binary).detach()
    parameters = tuple(reference.parameters())
    sums = [torch.zeros_like(parameter) for parameter in parameters]
    for index in range(len(roots)):
        loss = _task_loss(expected_logits[index:index + 1], labels[roots[index:index + 1]],
                          multilabel, binary=binary)
        grads = torch.autograd.grad(loss, parameters, retain_graph=True)
        norm = torch.sqrt(sum(gradient.square().sum() for gradient in grads)).clamp_min(1e-12)
        for total, gradient in zip(sums, grads):
            total.add_(gradient, alpha=min(1., config.sgd_clip / float(norm)))
    generator = torch.Generator().manual_seed(321)
    for parameter, total in zip(parameters, sums):
        parameter.grad = (total + torch.randn(total.shape, generator=generator)
                          * config.sgd_noise) / len(roots)
    expected_adam.step()
    reported = DPARTrainer(config)._private_step(
        model, actual_adam, x, labels, dense_ppr.to_sparse(), roots,
        torch.Generator().manual_seed(321), config,
    )
    assert not reported.requires_grad
    torch.testing.assert_close(reported, expected_loss)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_adam.state[actual][key], expected_adam.state[expected][key])


def test_chunked_evaluation_keeps_unscored_context_and_filters_ignored_labels(runner):
    model = DPARMLP(1, 1, hidden=1, layers=2, dropout=0)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.fill_(1)
    data = Data(x=torch.tensor([[0.], [4.], [1.], [0.]]),
                y=torch.tensor([1, 0, 0, 19]),
                edge_index=torch.tensor([[0], [1]]),
                eval_mask=torch.tensor([True, False, True, True]))
    task = dict(binary=True, multilabel=False, regression=False,
                metric="auroc", primary_metric="auroc", metric_ignore_label=19)
    config = DPARConfig(binary=True, metric_ignore_label=19, inference_steps=1)
    full = runner.evaluate(model, data, task, config, "cpu", 4)
    chunked = runner.evaluate(model, data, task, config, "cpu", 1)
    assert full["score"] == chunked["score"] == 1.0
    data.x[1] = 0
    assert runner.evaluate(model, data, task, config, "cpu", 1)["score"] == 0.0
