"""Numerical boundaries for the study's maintained nonprivate execution paths."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.models.baselines import GraphSAGE, MLP
from src.models.objectives import _task_loss
from src.training.baselines import _LayerwiseNeighborSampler


STUDY = Path(__file__).resolve().parents[1] / "results/eight_gpu_domain_graphsaint"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = _load("eight_gpu_exact_evaluation", STUDY / "evaluation.py")
nonprivate = _load("eight_gpu_nonprivate_runner", STUDY / "nonprivate/run.py")


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield


def _directed_graph():
    # Directed cycles, repeated edges, overlapping roots' neighborhoods, a
    # self-loop and a truly isolated node (8). Orientation cannot be swapped.
    return Data(
        x=torch.randn(9, 4),
        y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        edge_index=torch.tensor([
            [1, 2, 2, 3, 4, 5, 5, 6, 7, 0, 2, 4, 7],
            [0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7],
        ]),
    )


@pytest.mark.parametrize("task", [
    {"binary": False, "multilabel": False},
    {"binary": True, "multilabel": False},
    {"binary": False, "multilabel": True},
])
def test_hierarchical_sampled_logits_loss_and_parameter_gradients(task):
    data = _directed_graph()
    roots = torch.tensor([0, 3, 8])
    sample = _LayerwiseNeighborSampler(data.edge_index, data.num_nodes).sample(
        roots, fanouts=[2, 2], generator=torch.Generator().manual_seed(23),
    )
    outputs = 1 if task["binary"] else 3
    untrimmed = GraphSAGE(4, outputs, hidden=5, layers=2, dropout=0.0)
    hierarchical = deepcopy(untrimmed)
    labels = (torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]])
              if task["multilabel"] else torch.tensor([0, 1, 0])
              if task["binary"] else data.y[roots])
    full_logits = untrimmed.forward_sampled(
        data.x[sample.node_ids], sample.edge_index,
        sample.num_sampled_nodes, sample.num_sampled_edges,
        hierarchical=False,
    )
    trimmed_logits = hierarchical.forward_sampled(
        data.x[sample.node_ids], sample.edge_index,
        sample.num_sampled_nodes, sample.num_sampled_edges,
        hierarchical=True,
    )
    full_loss = _task_loss(full_logits, labels, **task)
    trimmed_loss = _task_loss(trimmed_logits, labels, **task)
    full_loss.backward()
    trimmed_loss.backward()
    torch.testing.assert_close(trimmed_logits, full_logits)
    torch.testing.assert_close(trimmed_loss, full_loss)
    for full_parameter, trimmed_parameter in zip(untrimmed.parameters(), hierarchical.parameters()):
        torch.testing.assert_close(trimmed_parameter.grad, full_parameter.grad)


def test_large_fanout_matches_full_graph_seed_logits_and_gradients():
    data = _directed_graph()
    roots = torch.tensor([0, 4, 8])
    sampled = _LayerwiseNeighborSampler(data.edge_index, data.num_nodes).sample(
        roots, fanouts=[100, 100], generator=torch.Generator().manual_seed(8),
    )
    whole = GraphSAGE(4, 3, hidden=7, layers=2, dropout=0.0)
    local = deepcopy(whole)
    logits_whole = whole(data.x, data.edge_index)[roots]
    logits_local = local.forward_sampled(
        data.x[sampled.node_ids], sampled.edge_index,
        sampled.num_sampled_nodes, sampled.num_sampled_edges,
        hierarchical=True,
    )
    torch.testing.assert_close(logits_local, logits_whole)
    _task_loss(logits_whole, data.y[roots], multilabel=False).backward()
    _task_loss(logits_local, data.y[roots], multilabel=False).backward()
    for actual, expected in zip(local.parameters(), whole.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("chunk_size", [1, 4, 100])
def test_exact_cpu_backed_graphsage_matches_directed_full_forward(chunk_size):
    data = _directed_graph()
    model = GraphSAGE(4, 3, hidden=6, layers=3, dropout=0.5)
    model.eval()
    expected = model(data.x, data.edge_index).detach()
    actual = evaluation.logits_graphsage(model, data, "cpu", chunk_size,
                                        cpu_backed=True)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    # Empty adjacency is not a reason to drop or invent a neighbor for an
    # isolated context node: its output still follows all self linears.
    isolated = data.x[8:9]
    for index, layer in enumerate(model.self_layers):
        isolated = layer(isolated)
        if index + 1 < len(model.self_layers):
            isolated = isolated.relu()
    torch.testing.assert_close(actual[8:9], isolated)


@pytest.mark.parametrize("method", ["mlp", "graphsage"])
def test_shuffled_training_visits_each_root_once_and_keeps_partial_batch(method):
    data = _directed_graph()
    model_type = MLP if method == "mlp" else GraphSAGE
    model = model_type(4, 3, hidden=5, layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sampler = (_LayerwiseNeighborSampler(data.edge_index, data.num_nodes)
               if method == "graphsage" else None)
    generator = torch.Generator().manual_seed(8)
    visited = []
    previous = [parameter.detach().clone() for parameter in model.parameters()]
    for roots in nonprivate.shuffled_batches(data.num_nodes, 4, generator):
        loss, _, _ = nonprivate.training_step(
            model, optimizer, data, roots,
            {"multilabel": False, "binary": False}, torch.device("cpu"),
            sampler=sampler, generator=generator, fanouts=[10, 10],
        )
        assert torch.isfinite(loss)
        visited.append(roots.clone())
    assert [len(batch) for batch in visited] == [4, 4, 1]
    assert torch.equal(torch.cat(visited).sort().values, torch.arange(data.num_nodes))
    assert any(not torch.equal(before, after) for before, after in zip(previous, model.parameters()))


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_global_tied_binary_auroc_excludes_unscored_rows(chunk_size):
    # Per-chunk AUROCs at size2 would average to .5, whereas the global score
    # is .625. The deliberately adversarial unscored rows must not contribute.
    data = Data(
        x=torch.tensor([[-1.], [1.], [1.], [0.], [-100.], [100.]]),
        y=torch.tensor([0, 1, 0, 1, 1, 0]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        eval_mask=torch.tensor([True, True, True, True, False, False]),
    )
    model = MLP(1, 1, hidden=1, layers=1, dropout=0.0)
    with torch.no_grad():
        model.layers[0].weight.fill_(1)
        model.layers[0].bias.zero_()
    model.train()
    result = evaluation.evaluate_mlp(
        model, data, {"binary": True, "multilabel": False, "metric": "auroc"},
        device="cpu", chunk_size=chunk_size,
    )
    assert result["score"] == pytest.approx(0.625)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["scored_nodes"] == 4
    assert result["context_nodes"] == 6
    assert model.training


def test_global_multilabel_f1_uses_global_counts_not_batch_averages():
    logits = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [50., 50.]])
    labels = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    result = evaluation.metrics(
        logits, labels,
        {"multilabel": True, "binary": False, "primary_metric": "micro_f1"},
        eval_mask=torch.tensor([True, True, True, False]),
    )
    assert result["score"] == pytest.approx(0.75)
    assert result["scored_nodes"] == 3
    assert result["context_nodes"] == 4


def test_mag_ignored_and_unscored_nodes_remain_message_passing_context():
    data = Data(
        x=torch.tensor([[0.], [0.], [5.], [1.]]),
        y=torch.tensor([1, 19, 0, 0]),
        edge_index=torch.tensor([[2, 1, 3], [0, 0, 1]]),
        eval_mask=torch.tensor([True, True, False, False]),
    )
    model = GraphSAGE(1, 20, hidden=2, layers=1, dropout=0.0)
    with torch.no_grad():
        model.self_layers[0].weight.zero_()
        model.self_layers[0].bias.zero_()
        model.self_layers[0].bias[0] = 1
        model.neighbor_layers[0].weight.zero_()
        model.neighbor_layers[0].weight[1] = 1
    result = evaluation.evaluate_graphsage(
        model, data,
        {"multilabel": False, "binary": False, "metric_ignore_label": 19,
         "metric": "accuracy"},
        device="cpu", chunk_size=1,
    )
    assert result["score"] == 1.0
    assert result["scored_nodes"] == 1
    assert result["context_nodes"] == 4
    # Removing unscored message-passing context actually changes the answer.
    induced_logits = model(data.x[:2], torch.tensor([[1], [0]]))
    assert induced_logits[0].argmax().item() == 0
    # Ignoring class19 is a scoring convention, not a training loss filter.
    logits = torch.zeros((2, 20), requires_grad=True)
    loss = _task_loss(logits, data.y[:2], multilabel=False)
    loss.backward()
    assert logits.grad[1, 19] < 0


def test_complementary_shared_target_masks_keep_identical_full_context_logits():
    data = _directed_graph()
    val_mask = torch.tensor([True, False, True, False, False, True, False, False, False])
    val = Data(x=data.x, y=data.y, edge_index=data.edge_index, eval_mask=val_mask)
    test = Data(x=data.x, y=data.y, edge_index=data.edge_index, eval_mask=~val_mask)
    model = GraphSAGE(4, 3, hidden=6, layers=2, dropout=0.0).eval()
    val_logits = evaluation.logits_graphsage(model, val, "cpu", chunk_size=2)
    test_logits = evaluation.logits_graphsage(model, test, "cpu", chunk_size=4)
    torch.testing.assert_close(val_logits, test_logits)
    task = {"multilabel": False, "binary": False, "primary_metric": "accuracy"}
    val_result = evaluation.metrics(val_logits, val.y, task, val.eval_mask)
    test_result = evaluation.metrics(test_logits, test.y, task, test.eval_mask)
    assert val_result["scored_nodes"] == 3
    assert test_result["scored_nodes"] == 6
    assert val_result["context_nodes"] == test_result["context_nodes"] == 9
    assert val_result["score"] == pytest.approx(
        float((val_logits[val_mask].argmax(1) == data.y[val_mask]).float().mean()))
    assert test_result["score"] == pytest.approx(
        float((test_logits[~val_mask].argmax(1) == data.y[~val_mask]).float().mean()))
