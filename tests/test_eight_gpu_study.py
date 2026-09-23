"""Scientific boundaries of the approved eight-GPU experiment contract."""
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[1] / 'results/eight_gpu_domain_graphsaint'
sys.path.append(str(STUDY))
import study_common as common


@pytest.fixture
def prepared():
    rows = {}
    for p in common.protocols() + common.protocols(True):
        n = common.EXPECTED_TRAIN.get(p['id'], 9498)
        binary = p['dataset'] == 'twitch-explicit'
        multilabel = p['dataset'] in ('saint-yelp', 'saint-amazon')
        metric = 'auroc' if binary else 'micro_f1' if multilabel else 'accuracy'
        rows[p['id']] = dict(n_train=n, manifest=f'/prepared/{p["id"]}/manifest.json',
            split_fingerprint=p.get('split_fingerprint', p['id'] + '-native'),
            task=dict(binary=binary, multilabel=multilabel, regression=False, primary_metric=metric,
                      num_classes=20 if p['dataset'] == 'mag-countries' else 2,
                      metric_ignore_label=19 if p['dataset'] == 'mag-countries' else None))
    return dict(protocols=rows, implementation_hash='fixture-implementation')


def test_initial_matrix_resolves_all_requested_exposures(prepared):
    matrix = common.expand_matrix(prepared)
    assert len(matrix) == len({common.scientific_key(c) for c in matrix}) == 96
    assert sum(c['epsilon'] is not None for c in matrix) == 80
    assert len({c['protocol'] for c in matrix}) == 8
    assert {c['dataset'] for c in matrix} == {'saint-flickr', 'saint-reddit', 'saint-yelp', 'saint-amazon', 'twitch-explicit', 'facebook100', 'mag-countries'}
    for c in matrix:
        assert c['batch_size'] == 1024
        if c['epsilon'] is not None:
            assert c['delta'] == 1 / c['n_train']
        if c['method'] == 'graphsage':
            assert c['graph_cap'] is None
            assert c['hidden_size'] == 128
            assert c['epochs'] == 100
        if c['method'] == 'progap':
            assert c['steps'] == 3 * 10 * (c['n_train'] // 1024)
        if c['method'] == 'dpar':
            assert c['sampled_train_nodes'] >= 1024
            assert c['steps'] == 10 * c['steps_per_epoch']


def test_dpar_population_floor_is_not_ppr_release_control(prepared):
    de = common.resolve_cell('twitch-de-engb', 'dpar', 2, prepared=prepared)
    amazon = common.resolve_cell('saint-amazon', 'dpar', 2, prepared=prepared)
    assert de['sampled_train_nodes'] == 1024
    assert de['ppr_num'] == 70
    assert amazon['sampled_train_nodes'] == 113038
    assert amazon['delta'] == 1/1255968


def test_phase_membership_reuses_science_but_not_smokes(prepared):
    baseline = common.resolve_cell('twitch-de-engb', 'sparse', 8, prepared=prepared)
    search = common.resolve_cell('twitch-de-engb', 'sparse', 8, phase='search', prepared=prepared)
    assert common.scientific_key(baseline) == common.scientific_key(search)
    for field, value in [('learning_rate', .03), ('clip', .25), ('epochs', 20), ('hidden_size', 128), ('grid', .01)]:
        changed = common.resolve_cell('twitch-de-engb', 'sparse', 8, overrides={field: value}, prepared=prepared)
        assert common.scientific_key(changed) != common.scientific_key(baseline)
    physical = baseline | {'physical_chunk_size': 1, 'eval_chunk_size': 1, 'max_private_batch_nodes': 1}
    assert common.scientific_key(physical) == common.scientific_key(baseline)
    smoke = baseline | {'phase': 'smoke'}
    assert common.scientific_key(smoke) != common.scientific_key(baseline)


def test_schedule_changes_recompute_steps_and_seeded_cap(prepared):
    baseline = common.resolve_cell('mag-default', 'sparse', 2, prepared=prepared)
    child = common.resolve_cell('mag-default', 'sparse', 2, seed=2, overrides={'epochs': 3, 'requested_epochs': 10}, prepared=prepared)
    assert child['steps'] == 3 * baseline['steps_per_epoch']
    assert child['requested_epochs'] == 10
    assert child['cap_seed'] == 20002
    assert child['p1'] == baseline['p1']
    assert child['accountant_hash'] == baseline['accountant_hash']
    assert child['accountant_convention'] == 'repository_path_bound_chi1'
    assert not child['union_safe']


def test_expanded_complements_and_original_requests_are_independent(prepared):
    from src.data.domain_datasets import DOMAIN_REGISTRIES
    original = {common.scientific_key(c) for c in common.expand_matrix(prepared)}
    expanded = common.protocols(True)
    assert [len(p['domain_split']['train']) for p in expanded] == [5, 16, 4]
    for p in expanded:
        split = p['domain_split']
        train, val, test = map(set, (split['train'], split['val'], split['test']))
        assert len(val) == len(test) == 1
        assert not (train & val or train & test or val & test)
        assert train | val | test == set(DOMAIN_REGISTRIES[p['dataset']])
    matrix = common.expand_matrix(prepared, phase='expanded_domains')
    assert len(matrix) == len({common.scientific_key(c) for c in matrix}) == 36
    assert original.isdisjoint(common.scientific_key(c) for c in matrix)
    assert original == {common.scientific_key(c) for c in common.expand_matrix(prepared)}


def test_smoke_matrix_uses_full_populations_and_native_minimum_schedules(prepared):
    smoke = common.expand_matrix(prepared, phase='smoke')
    assert len(smoke) == 28
    assert {c['protocol'] for c in smoke} == {'twitch-de-engb', 'saint-yelp', 'mag-default', 'saint-amazon'}
    for c in smoke:
        assert c['n_train'] == prepared['protocols'][c['protocol']]['n_train']
        assert c['epsilon'] == (None if c['method'] in ('mlp', 'graphsage') else 8)
        if c['method'] == 'progap':
            assert c['steps'] == 3 * (c['n_train'] // 1024)
        elif c['method'] == 'dpar':
            assert c['steps'] == c['steps_per_epoch']
        else:
            assert c['steps'] == 1
