"""Observable scientific and process-lifetime contracts of the initial campaign."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from results.sparsegnn_initial_tuning import study_common as common
from results.sparsegnn_initial_tuning import search


@pytest.fixture
def prepared():
    rows = {}
    for protocol in common.protocols():
        binary = protocol['dataset'] == 'twitch-explicit'
        multilabel = protocol['dataset'] in ('saint-yelp', 'saint-amazon')
        metric = 'auroc' if binary else 'micro_f1' if multilabel else 'accuracy'
        rows[protocol['id']] = dict(n_train=4096, manifest='/fixture/manifest.json',
            split_fingerprint=f"split-{protocol['id']}",
            task=dict(metric=metric, primary_metric=metric, binary=binary,
                      multilabel=multilabel, regression=False, num_classes=2,
                      num_features=8, metric_ignore_label=19 if protocol['dataset']=='mag-countries' else None))
    return dict(protocols=rows, implementation_hash='fixture-implementation')


def completed_rows(cells, score=.6):
    return [dict(cell=c, status='completed', accepted=True, folder=f"/fixture/{common.scientific_key(c)}",
                 result=dict(validation_metric=.5, test_metric=score,
                             scientific_key=common.scientific_key(c))) for c in cells]


def test_exact_grid_and_allbuttwo_domains(prepared):
    cells = search.tuning_cells(prepared)
    assert len(cells) == len({common.scientific_key(c) for c in cells}) == 576
    combinations = set(itertools.product((256,1024),(.5,.1),(.01,.001),(10,20,25)))
    for protocol in common.PROTOCOL_IDS:
        for epsilon in (2.,8.):
            group = [c for c in cells if c['protocol']==protocol and c['epsilon']==epsilon]
            assert {(c['batch_size'],c['p2'],c['learning_rate'],c['epochs']) for c in group} == combinations
            assert all(c['aggregation']=='mean' and c['seed']==0 and c['r']==1 for c in group)
    from src.data.domain_datasets import DOMAIN_REGISTRIES
    expected = {'twitch-allbut2':('engb','es',5), 'facebook100-allbut2':('cornell5','penn94',16),
                'mag-allbut2':('cn','de',4)}
    for p in common.protocols():
        if p['id'] not in expected:
            continue
        val,test,n = expected[p['id']]
        roles = p['domain_split']
        assert roles['val']==[val] and roles['test']==[test]
        assert roles['train']==[d for d in DOMAIN_REGISTRIES[p['dataset']] if d not in (val,test)]
        assert len(roles['train'])==n
        assert set(roles['train']).isdisjoint([val,test])


def test_logical_schedules_and_nonprivate_epoch_override(prepared):
    def cell(method, epochs, batch=256):
        return common.resolve_cell('ogbn-arxiv',method,None if method in ('mlp','graphsage') else 8,
            overrides=dict(batch_size=batch,epochs=epochs),phase='compare',prepared=prepared)
    sparse = cell('sparse',10)
    assert sparse['p1']==.0625 and sparse['steps']==160
    assert cell('sparse',20,1024)['steps']==80
    assert cell('sparse',25)['steps']==400
    progap = cell('progap',25)
    assert progap['steps_per_stage']==400 and progap['steps']==1200
    dpar = cell('dpar',25)
    assert dpar['steps']==25 and dpar['effective_batch_size']==70 and dpar['sample_rate']==1
    for epochs in (10,20,25):
        for method in ('mlp','graphsage'):
            c = cell(method,epochs)
            assert c['epochs']==c['requested_epochs']==100 and c['steps']==1600
    with pytest.raises(ValueError, match='Conflicting derived'):
        common.resolve_cell('ogbn-arxiv','sparse',8,overrides={'batch_size':256,'steps':1},prepared=prepared)


def test_small_evaluation_context_and_batch_admissibility(prepared):
    import torch
    from torch_geometric.data import Data
    from results.sparsegnn_initial_tuning.prepare import _partition_statistics
    data=Data(x=torch.ones(3,2),y=torch.tensor([0,1,0]),edge_index=torch.tensor([[0,1],[1,0]]))
    stats=_partition_statistics(data,torch.arange(3),torch.ones(3,dtype=torch.bool),
        dict(multilabel=False,binary=False,num_classes=2,metric_ignore_label=None),'val')
    assert stats['context_nodes']==stats['scored_nodes']==3
    prepared['protocols']['ogbn-arxiv']['n_train']=512
    cells=[c for c in search.tuning_cells(prepared) if c['protocol']=='ogbn-arxiv']
    assert all('blocked_reason' not in c for c in cells if c['batch_size']==256)
    assert all(c.get('blocked_reason') and 'n_train' not in c for c in cells if c['batch_size']==1024)


def test_test_score_selection_and_failure_exclusion(prepared):
    cells=search.tuning_cells(prepared)
    group=[c for c in cells if c['protocol']=='ogbn-arxiv' and c['epsilon']==2]
    rows=completed_rows(group,.2)
    rows[0]['result'].update(validation_metric=.9,test_metric=.6)
    rows[1]['result'].update(validation_metric=.7,test_metric=.8)
    rows[2].update(status='failed',accepted=False)
    rows[2]['result']['test_metric']=.99
    selection=search.select_winners(rows,cells)
    winner=next(w for w in selection['winners'] if w['protocol']=='ogbn-arxiv' and w['epsilon']==2)
    assert winner['scientific_key']==common.scientific_key(group[1])
    assert winner['status']=='partial_grid' and winner['completed_candidates']==23
    unavailable=next(w for w in selection['winners'] if w['protocol']=='ogbn-products')
    assert unavailable['status']=='unavailable' and unavailable['cell'] is None
    tied=search.select_winners(completed_rows(group,.8),cells)['winners'][0]
    assert (tied['cell']['epochs'],tied['cell']['batch_size'],tied['cell']['learning_rate'],tied['cell']['p2'])==(10,256,.001,.1)


def test_comparison_slots_preserve_nonprivate_reuse(prepared):
    cells=search.tuning_cells(prepared)
    rows=completed_rows(cells,.1)
    for row in rows:
        c=row['cell']
        if c['batch_size']==256 and c['learning_rate']==.001:
            wanted=(c['epsilon']==2 and c['epochs']==10 and c['p2']==.1) or (c['epsilon']==8 and c['epochs']==25 and c['p2']==.5)
            if wanted:
                row['result']['test_metric']=.9
    selected=search.select_winners(rows,cells)
    slots=search.comparison_slots(selected,prepared)
    assert len(slots)==192
    jobs=search.comparison_cells(selected,prepared)
    assert len(jobs)==168  # 144 private slots + 24 reused nonprivate jobs.
    for protocol in common.PROTOCOL_IDS:
        group=[s for s in slots if s['protocol']==protocol]
        for method in ('mlp','graphsage'):
            pair=[s['cell'] for s in group if s['method']==method]
            assert len(pair)==2 and all(c['epochs']==100 for c in pair)
            assert common.scientific_key(pair[0])==common.scientific_key(pair[1])
        sparse=[s['cell'] for s in group if s['method']=='sparse' and s['epsilon_context']==2]
        assert len({common.scientific_key(c) for c in sparse})==2
    smoke=search.smoke_cells(prepared)
    assert len(smoke)==32 and all(c['phase']=='smoke' for c in smoke)
    assert not ({common.scientific_key(c) for c in smoke}&{common.scientific_key(c) for c in cells})


def test_preparation_restores_standard_root_after_error(tmp_path,monkeypatch):
    from results.sparsegnn_initial_tuning.prepare import _loader_root
    monkeypatch.setenv('OGB_DATA_ROOT','original')
    with pytest.raises(RuntimeError,match='loader failed'):
        with _loader_root({'dataset':'ogbn-arxiv'},tmp_path):
            assert os.environ['OGB_DATA_ROOT']==str(tmp_path)
            raise RuntimeError('loader failed')
    assert os.environ['OGB_DATA_ROOT']=='original'


def test_real_deadline_kills_descendant_and_records_timeout(tmp_path,monkeypatch):
    from results.sparsegnn_initial_tuning import queue as scheduler
    from results.sparsegnn_initial_tuning import policy
    script=tmp_path/'sleeper.py'
    child_pid=tmp_path/'child.pid'
    script.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n")
    folder=tmp_path/'attempt'
    folder.mkdir()
    command=[sys.executable,str(script),'--folder',str(folder)]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0', 'OMP_NUM_THREADS':'2',
                   'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                   'PYTHONNOUSERSITE':'1', 'PYTHONPATH':str(tmp_path)}
    monkeypatch.setattr(scheduler,'runner_command',lambda *args:(command,environment))
    monkeypatch.setattr(scheduler,'gpu_snapshot',lambda gpu:{'physical_gpu':gpu,'idle':True})
    monkeypatch.setattr(scheduler,'HardDeadline',lambda:policy.HardDeadline(.5))
    original_terminate=policy.terminate_owned
    monkeypatch.setattr(scheduler,'terminate_owned',lambda p,l:original_terminate(p,l,grace=.2))
    q=scheduler.StudyQueue.__new__(scheduler.StudyQueue)
    q.root=tmp_path; q.revision='test'; q.source_files={}; q.stop=threading.Event()
    q.mutex=threading.RLock(); q.reservations={}; q.assert_frozen=lambda:None
    events=[]
    q.event=lambda event,**fields:events.append((event,fields))
    cell={'method':'sparse','epochs':25,'requested_epochs':25}
    outcome=q.run_process(cell,folder,0,{'estimated_host_bytes':0})
    assert outcome.cancellation['reason']=='hard_runtime_limit'
    exit_info=json.loads((folder/'exit.json').read_text())
    assert exit_info['status']=='timeout' and exit_info['owned_process_exited']
    assert json.loads((folder/'gpu_release.json').read_text())['idle']
    assert any(e=='RUN_TIMEOUT' for e,_ in events)
    assert child_pid.exists()
    pid=int(child_pid.read_text())
    try:
        assert policy.proc_identity(pid)['state']=='Z'
    except FileNotFoundError:
        pass
    launch=json.loads((folder/'launch.json').read_text())
    fake={**launch,'command':launch['observed_command'],'start_ticks':launch['start_ticks']+1}
    with pytest.raises(RuntimeError,match='ownership mismatch'):
        policy.validate_owned_process(launch,identity=lambda _:fake)
    assert policy.HARD_LIMIT==5400


def test_numerical_backend_changes_calibration_identity():
    from results.sparsegnn_initial_tuning.sparse.calibration import cache_identity
    request = {'n_train':4096, 'arguments':{'p1':.0625,'steps':400,'union_safe':False},
               'numerical_versions':{'numpy':'1.26.4','scipy':'1.14.1','dp-accounting':'0.4.3'}}
    changed = {**request, 'numerical_versions':{**request['numerical_versions'],'scipy':'1.15.3'}}
    assert cache_identity(changed) != cache_identity(request)
 

def test_ranking_uses_sage_margin_and_requires_all_comparators():
    from results.sparsegnn_initial_tuning.summarize import rank_datasets
    selected = {'winners':[{'protocol':'ogbn-arxiv','epsilon':e,
                           'status':'complete_grid','completed_candidates':24} for e in (2.,8.)]}
    rows = []
    for epsilon,sparse,best in ((2.,.80,.75),(8.,.85,.83)):
        rows.append(dict(protocol='ogbn-arxiv',epsilon_context=epsilon,method='sparse',
                         aggregation='mean',accepted=True,test_metric=sparse))
        rows.append(dict(protocol='ogbn-arxiv',epsilon_context=epsilon,method='sparse',
                         aggregation='gin',accepted=True,test_metric=.99))
        for method in ('progap','dpar','dpgnn','dpmlp'):
            rows.append(dict(protocol='ogbn-arxiv',epsilon_context=epsilon,method=method,
                             accepted=True,test_metric=best if method=='progap' else .5))
    protocols = [dict(id='ogbn-arxiv',dataset='ogbn-arxiv')]
    ranking = rank_datasets(rows,selected,protocols)
    assert ranking[0]['status']=='definitive' and ranking[0]['suitability']==pytest.approx(.035)
    missing = [r for r in rows if not(r['method']=='dpmlp' and r['epsilon_context']==8)]
    assert rank_datasets(missing,selected,protocols)[0]['status']=='unranked'
    selected['winners'][0]['status']='partial_grid'
    assert rank_datasets(rows,selected,protocols)[0]['status']=='unranked'


def test_direct_entrypoint_preserves_standard_library_queue():
    import subprocess
    for entry in ('prepare.py','summarize.py','verify_complete.py'):
        script=common.HERE/entry
        code=(f"import sys,runpy; sys.path.insert(0,{str(common.HERE)!r}); "
              f"runpy.run_path({str(script)!r},run_name='entrypoint_test'); "
              "from queue import Queue; q=Queue(); q.put('ok'); assert q.get()=='ok'")
        result=subprocess.run([sys.executable,'-c',code],cwd=common.ROOT,
                              capture_output=True,text=True,timeout=30)
        assert result.returncode==0, result.stderr


def test_revised_ranking_requires_complete_shorter_epoch_selection():
    from scripts.sparsegnn_final_reports import rank_datasets, SELECTION_SCOPE
    selected = {'selection_scope': dict(SELECTION_SCOPE),
                'winners': [{'protocol': 'ogbn-arxiv', 'epsilon': epsilon,
                             'status': 'complete_grid', 'completed_candidates': 16,
                             'requested_candidates': 16, 'registered_candidates': 16,
                             'cell': {'epochs': 10}} for epsilon in (2., 8.)]}
    rows = []
    for epsilon, sparse, best in ((2., .80, .75), (8., .85, .83)):
        rows.append(dict(protocol='ogbn-arxiv', epsilon_context=epsilon, method='sparse',
                         aggregation='mean', accepted=True, test_metric=sparse))
        rows.append(dict(protocol='ogbn-arxiv', epsilon_context=epsilon, method='sparse',
                         aggregation='gin', accepted=True, test_metric=.99))
        for method in ('progap', 'dpar', 'dpgnn', 'dpmlp'):
            rows.append(dict(protocol='ogbn-arxiv', epsilon_context=epsilon, method=method,
                             accepted=True, test_metric=best if method=='progap' else .5))
    def result():
        return next(row for row in rank_datasets(rows, selected)
                    if row['protocol'] == 'ogbn-arxiv')
    assert result()['status'] == 'definitive'
    assert result()['suitability'] == pytest.approx(.035)
    selected['winners'][0]['cell']['epochs'] = 25
    assert result()['status'] == 'unranked'
    selected['winners'][0]['cell']['epochs'] = 10
    selected['winners'][0]['completed_candidates'] = 15
    assert result()['status'] == 'unranked'


def test_nonprivate_cancellation_cannot_resolve_private_requests(tmp_path):
    from scripts.sparsegnn_partial_nonprivate import verify_cancelled
    directive = tmp_path / 'nonprivate_stop.json'
    directive.write_text(json.dumps({'reason': 'User cancelled remaining non-private training'}))
    disposition = {'status': 'cancelled_unstarted', 'folder': None,
                   'cancellation_directive_sha256': common.sha256(directive)}
    assert verify_cancelled(tmp_path, {'method': 'graphsage'}, disposition) == []
    assert verify_cancelled(tmp_path, {'method': 'sparse'}, disposition)
    directive.write_text(json.dumps({'reason': 'Changed instruction'}))
    assert verify_cancelled(tmp_path, {'method': 'graphsage'}, disposition)

