"""Behavioral boundaries of the experiment-owned SparseGNN seams."""
from contextlib import contextmanager
import copy
from pathlib import Path
import time

import pytest
import torch
from torch_geometric.data import Data

from results.eight_gpu_domain_graphsaint.sparse.run import TrainingSession, make_mechanism
from results.eight_gpu_domain_graphsaint.sparse.calibration import (
    CONVENTION, cache_identity, calibration_request,
)
from results.eight_gpu_domain_graphsaint.sparse.degree_preprocessing import cap_degrees, dedup_arcs
from results.eight_gpu_domain_graphsaint.sparse.evaluation import score_logits, sparse_logits
from src.models.objectives import _task_metric
from src.processing.sparse_expand import build_adjacency
from src.training.sparse_gnn import train_sparse_gnn
from study_common import ROOT, sha256


def graph(task):
    labels = (torch.tensor([0., 1., 0., 1., 1., 0.]) if task['binary'] else
              torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 1.], [1., 0.], [0., 0.]])
              if task['multilabel'] else torch.tensor([0, 1, 19, 1, 0, 1]))
    return Data(x=torch.tensor([[1., .2, .3], [.1, 1., .5], [.6, .2, 1.],
                                [1., .7, .1], [.3, .9, .4], [.5, .5, .5]]),
                y=labels, edge_index=torch.tensor([[1, 2, 0, 3, 4, 1], [0, 1, 2, 0, 3, 4]]),
                train_mask=torch.ones(6, dtype=torch.bool),
                eval_mask=torch.tensor([True, False, True, True, False, False]))


def task(kind):
    return dict(binary=kind == 'binary', multilabel=kind == 'multilabel',
                num_classes=20 if kind == 'categorical' else 2,
                metric_ignore_label=19 if kind == 'categorical' else None,
                metric={'binary': 'auroc', 'multilabel': 'micro_f1', 'categorical': 'accuracy'}[kind])


def cell(metadata):
    return dict(task=metadata, hidden_size=4, layers=2, dropout=0., aggregation='mean',
                max_private_batch_nodes=1000, learning_rate=.01, betas=[.9, .999],
                adam_eps=1e-8, amsgrad=False, weight_decay=0., steps=2, steps_per_epoch=1,
                cap_seed=20000, seed=0, eval_chunk_size=2)


@pytest.mark.parametrize('kind', ['binary', 'multilabel', 'categorical'])
def test_csr_row_chunks_preserve_logits_context_and_score_masks(kind):
    metadata = task(kind)
    data = graph(metadata)
    torch.manual_seed(21)
    mechanism = make_mechanism(data, cell(metadata), 'cpu')
    mechanism.module.eval()
    with torch.no_grad():
        expected_logits = mechanism.module(data.x, data.edge_index)
    logits, _ = sparse_logits(mechanism, data, metadata, chunk_size=1)
    torch.testing.assert_close(logits, expected_logits, atol=1e-6, rtol=1e-5)
    selected = data.eval_mask.clone()
    if kind == 'categorical':
        selected &= data.y != 19
    expected, secondary = _task_metric(expected_logits[selected], data.y[selected],
                                        multilabel=metadata['multilabel'], binary=metadata['binary'])
    actual = score_logits(logits, data, metadata)
    assert actual['score'] == pytest.approx(expected)
    assert actual['scored_nodes'] == int(selected.sum())
    if kind == 'binary':
        assert actual['binary_accuracy'] == pytest.approx(secondary)
    assert torch.equal(data.edge_index, graph(metadata).edge_index)


def test_unscored_nodes_still_supply_sparse_message_passing_context():
    metadata = task('binary')
    data = graph(metadata)
    mechanism = make_mechanism(data, cell(metadata), 'cpu')
    with torch.no_grad():
        for parameter in mechanism.module.parameters():
            parameter.fill_(.2)
    before, _ = sparse_logits(mechanism, data, metadata, chunk_size=2)
    changed = data.clone()
    changed.x[1].add_(10)  # Node 1 is not scored, but sends an incoming arc to node 0.
    after, _ = sparse_logits(mechanism, changed, metadata, chunk_size=1)
    assert after[0] > before[0]
    assert not data.eval_mask[1]


def test_binary_auroc_is_global_and_tie_correct_after_masking():
    metadata = task('binary')
    data = Data(y=torch.tensor([0., 1., 0., 1., 1.]),
                eval_mask=torch.tensor([True, True, True, True, False]))
    # Each two-row chunk has perfect within-chunk ranking; the global tie makes
    # global AUROC .875, so averaging the two chunk AUROCs would be incorrect.
    result = score_logits(torch.tensor([0., 1., 1., 2., -100.]), data, metadata)
    assert result['score'] == pytest.approx(.875)
    assert result['scored_nodes'] == 4


class MemoryAttempt:
    def __init__(self, config, folder):
        self.cell, self.folder = config, Path(folder)
        self.timing = {name + '_seconds': 0. for name in
                       ('load', 'preprocess', 'calibration', 'train_update', 'validation', 'test', 'checkpoint_io')}
        self.active = None
        self.checkpoints = []

    @contextmanager
    def phase(self, name):
        assert self.active is None
        self.active = name
        before = time.monotonic()
        try:
            yield
        finally:
            self.timing[name + '_seconds'] += time.monotonic()-before
            self.active = None

    def save_checkpoint(self, payload, filename='checkpoint.pt'):
        with self.phase('checkpoint_io'):
            torch.save(payload, self.folder / filename)
        self.checkpoints.append(payload)

    def save_json(self, name, payload):
        pass

    def progress(self, **fields):
        pass


def execute_private(config, folder, *, p1=1., learning_rate=.01):
    torch.manual_seed(112)
    data = graph(config['task'])
    config = dict(config, learning_rate=learning_rate)
    mechanism = make_mechanism(data, config, 'cpu')
    attempt = MemoryAttempt(config, folder)
    session = TrainingSession(attempt, mechanism, data)
    mechanism.study_session = session
    with session.observe_updates():
        train_sparse_gnn(mechanism, data, data, p1=p1, p2=.5, r=2, T=config['steps'],
                         adj=build_adjacency(data.edge_index, data.num_nodes, direction='in'),
                         direction='in', dp=True, clip=.7, sigma=.4, seed=17,
                         track_every=1, checkpoint_callback=session.checkpoint)
    return mechanism, session, attempt


def test_physical_chunks_do_not_change_fixed_noise_adam_updates(tmp_path):
    config = cell(task('binary'))
    (tmp_path / 'chunked').mkdir()
    (tmp_path / 'whole').mkdir()
    chunked, observed, _ = execute_private(dict(config, max_private_batch_nodes=1), tmp_path / 'chunked')
    whole, reference, _ = execute_private(config, tmp_path / 'whole')
    for left, right in zip(chunked.parameters(), whole.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
    assert observed.batch_sizes == reference.batch_sizes == [6, 6]
    assert observed.counters['physical_chunks'] > reference.counters['physical_chunks']
    assert observed.counters['noise_additions'] == observed.counters['optimizer_updates'] == 2


def test_validation_ties_select_first_checkpoint_and_preserve_private_rng(tmp_path):
    mechanism, session, attempt = execute_private(cell(task('binary')), tmp_path, learning_rate=0.)
    assert session.best['step'] == 1
    assert [row['selected'] for row in session.history] == [True, False]
    assert [saved['step'] for saved in attempt.checkpoints] == [1]
    selected = torch.load(tmp_path / 'checkpoint.pt', weights_only=False)
    assert selected['step'] == 1
    assert selected['rng']['sample'].dtype == selected['rng']['noise'].dtype == torch.uint8
    # Both logical updates still execute; checkpoint evaluation does not reset Adam.
    assert {int(state['step']) for state in mechanism.optimizer.state.values()} == {2}


def test_empty_poisson_draw_is_a_real_noise_only_update(tmp_path):
    config = dict(cell(task('binary')), steps=1)
    torch.manual_seed(112)
    original = make_mechanism(graph(config['task']), config, 'cpu')
    initial = [parameter.detach().clone() for parameter in original.parameters()]
    trained, session, _ = execute_private(config, tmp_path, p1=0.)
    assert session.batch_sizes == [0]
    assert session.counters['noise_additions'] == session.counters['optimizer_updates'] == 1
    assert any(not torch.equal(before, after) for before, after in zip(initial, trained.parameters()))
    assert session.history[0]['loss'] == 0.


def test_directed_cap_is_seeded_and_does_not_add_reverse_arcs():
    source = torch.tensor([[i for i in range(24) for j in range(i+1, 24)],
                           [j for i in range(24) for j in range(i+1, 24)]])
    duplicated = torch.cat([source, source[:, :5]], dim=1)
    simple = dedup_arcs(duplicated, 24)
    first = cap_degrees(simple, 24, K_in=10, K_out=10, generator=torch.Generator().manual_seed(20000))
    second = cap_degrees(simple, 24, K_in=10, K_out=10, generator=torch.Generator().manual_seed(20000))
    assert torch.equal(first, second)
    assert int(torch.bincount(first[0], minlength=24).max()) <= 10
    assert int(torch.bincount(first[1], minlength=24).max()) <= 10
    assert bool((first[0] < first[1]).all())
    assert first.size(1) < simple.size(1)


def test_calibration_identity_separates_clip_schedule_grid_population_and_code():
    config = dict(n_train=9498, epsilon=8., delta=1/9498, p1=1024/9498, p2=.5,
                  r=2, K_in=10, K_out=10, steps=100, clip=1., grid=.001, union_safe=False,
                  accountant_hash=sha256(ROOT / 'src/privacy/accounting.py'), accountant_convention=CONVENTION)
    request = calibration_request(config)
    key = cache_identity(request)
    for override in ({'clip': .25}, {'steps': 200}, {'grid': .01}, {'n_train': 10000}):
        assert cache_identity(calibration_request(dict(config, **override))) != key
    changed_code = copy.deepcopy(request)
    changed_code['accountant_hash'] = 'new code revision'
    assert cache_identity(changed_code) != key
    with pytest.raises(ValueError, match='chi1'):
        calibration_request(dict(config, union_safe=True))


def test_large_noise_remains_finite_with_subnormal_path_weights():
    """Backend upgrades must not overflow a valid Gaussian-mixture tail."""
    from src.privacy.accounting import mixture_gaussian_pld, sparsegnn_mixture_weights

    population = 537635
    weights = sparsegnn_mixture_weights(
        1024 / population, .5, 2, 10, 10, union_safe=False)
    epsilons = [
        mixture_gaussian_pld(weights, sigma, .001)
        .self_compose(5260).get_epsilon_for_delta(1 / population)
        for sigma in (16., 32.)
    ]
    assert 0 < epsilons[1] < epsilons[0] < float('inf')
