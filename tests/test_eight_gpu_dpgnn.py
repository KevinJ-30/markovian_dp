"""Requested-delta accounting and exact masked padded inference boundaries."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from src.models.baselines import _OneHopGraphSAGE
from src.models.objectives import _metric_rows, _task_loss, _task_metric
from src.privacy.dpgnn import max_terms_per_node, multiterm_dpsgd_epsilon
from src.processing.sparse_expand import build_adjacency
from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'eight_gpu_dpgnn_runner', ROOT / 'results/eight_gpu_domain_graphsaint/dpgnn/run.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize('steps', [1, 7])
def test_calibration_covers_actual_schedule_at_exact_source_delta(steps):
    n = 2048
    config = DPGNNConfig(num_classes=2, steps=steps, batch_size=1024,
                         noise_multiplier=1.0, max_degree=10, clip=1.7)
    sigma, achieved, calibration = runner.calibrate(n, config, 2.0, 1.0 / n)
    independently_accounted = multiterm_dpsgd_epsilon(
        steps=steps, noise_multiplier=sigma, delta=1.0 / n,
        num_samples=n, batch_size=1024, max_terms=max_terms_per_node(10))
    assert 0 <= independently_accounted <= 2.0
    assert achieved == pytest.approx(independently_accounted, abs=1e-12)
    # The returned upper endpoint must be close to the requested boundary, not
    # calibrated under the maintained fit() delta or the reference's 1/(N+1).
    assert achieved == pytest.approx(2.0, abs=1e-7)
    assert calibration['noise_std_on_sum'] == pytest.approx(2 * 11 * sigma * 1.7)


def test_calibration_rejects_historical_delta_before_accounting():
    config = DPGNNConfig(num_classes=2, steps=2, batch_size=1024, noise_multiplier=1.0)
    with pytest.raises(ValueError, match='exactly 1/N_train'):
        runner.calibrate(2048, config, 2.0, 1.0 / (2048 + 1))


def test_calibration_exhausted_bracket_is_an_error_not_an_unsafe_result(monkeypatch):
    monkeypatch.setattr(runner, 'multiterm_dpsgd_epsilon', lambda **kwargs: 3.0)
    config = DPGNNConfig(num_classes=2, steps=1, batch_size=1024, noise_multiplier=1.0)
    with pytest.raises(RuntimeError, match='sigma=1e6'):
        runner.calibrate(2048, config, 2.0, 1.0 / 2048)


@pytest.mark.parametrize('binary', [False, True])
def test_masked_padded_evaluation_matches_full_prepared_context(binary):
    torch.manual_seed(428)
    partition = SimpleNamespace(
        num_nodes=6, x=torch.randn(6, 3),
        y=torch.tensor([0, 1, 1, 0, 1, 0] if binary else [0, 3, 19, 1, 4, 2]),
        # Cyclic/shared-neighbor context plus an isolated node. Unscored nodes
        # remain required inputs to scored roots zero and three.
        edge_index=torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 2, 3, 0, 3]]),
        eval_mask=torch.tensor([True, False, True, True, False, False]),
    )
    task = dict(binary=binary, multilabel=False,
                metric_ignore_label=None if binary else 19,
                metric='auroc' if binary else 'accuracy')
    config = DPGNNConfig(num_classes=2 if binary else 20, steps=1, batch_size=1,
                         noise_multiplier=1.0, max_degree=10, binary=binary,
                         metric_ignore_label=task['metric_ignore_label'])
    model = _OneHopGraphSAGE(3, 5, 1 if binary else 20).eval()
    prepared = runner.prepare_evaluation(config, partition, seed=3)
    source_x, labels, edge, weights = PartitionedDPGNN(config)._prepared_graph(partition, seed=3)
    with torch.no_grad():
        expected = model(source_x, edge, weights)
    actual = runner.evaluation_logits(model, prepared, device='cpu',
                                      max_padded_nodes=4, chunk_size=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    selected, targets = _metric_rows(expected, labels, eval_mask=partition.eval_mask,
                                     metric_ignore_label=task['metric_ignore_label'])
    expected_score = _task_metric(selected, targets, False, binary=binary)[0]
    expected_loss = float(_task_loss(selected, targets, False, binary=binary))
    result = runner.evaluate(model, prepared, task, device='cpu',
                             max_padded_nodes=4, chunk_size=1)
    assert result['score'] == pytest.approx(expected_score)
    assert result['loss'] == pytest.approx(expected_loss)
    assert result['scored_nodes'] == (3 if binary else 2)
    assert result['context_nodes'] == 6


def test_evaluation_never_truncates_a_large_outgoing_star():
    n = 110
    x = torch.zeros(n, 1)
    x[105] = 100.0  # A truncation at the training cap of 100 would lose this.
    edge = torch.stack((torch.zeros(n - 1, dtype=torch.long), torch.arange(1, n)))
    model = _OneHopGraphSAGE(1, 1, 1).eval()
    with torch.no_grad():
        model.root_encoder.weight.zero_()
        model.root_encoder.bias.zero_()
        model.neighbour_encoder.weight.fill_(1.0)
        model.decoder.weight.fill_(1.0)
        model.decoder.bias.zero_()
    prepared = runner.PreparedEvaluation(
        x=x, labels=torch.zeros(n, dtype=torch.long),
        adjacency=build_adjacency(edge, n, direction='out'), eval_mask=None,
        edge_weights=torch.linspace(0.1, 1.0, n - 1), max_star_nodes=n)
    with torch.no_grad():
        expected = model(x, edge, prepared.edge_weights)
    actual = runner.evaluation_logits(model, prepared, device='cpu',
                                      max_padded_nodes=3, chunk_size=7)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0] > 0.5


def test_binary_ties_are_scored_globally_across_single_row_chunks():
    partition = SimpleNamespace(num_nodes=4, x=torch.ones(4, 1),
                                y=torch.tensor([0, 0, 1, 1]),
                                edge_index=torch.empty((2, 0), dtype=torch.long),
                                eval_mask=torch.ones(4, dtype=torch.bool))
    config = DPGNNConfig(num_classes=2, steps=1, batch_size=1,
                         noise_multiplier=1.0, binary=True)
    model = _OneHopGraphSAGE(1, 2, 1).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = runner.evaluate(
        model, runner.prepare_evaluation(config, partition, seed=3),
        dict(binary=True, multilabel=False, metric='auroc'),
        device='cpu', max_padded_nodes=1, chunk_size=1)
    assert result['score'] == 0.5
    assert result['accuracy'] == 0.5
