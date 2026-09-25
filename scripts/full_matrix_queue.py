#!/usr/bin/env python3
"""Start the fixed experiment grid on eligible GPUs; publish all-run tables."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import full_matrix_records as records
from scripts import full_matrix_runtime as runtime


def _gpu_eligible(snapshot: dict | None) -> bool:
    if not snapshot or snapshot.get('error'):
        return False
    try:
        utilization = float(snapshot['utilization_gpu'])
        free = float(snapshot['memory_free_mib'])
        return math.isfinite(utilization) and 0 <= utilization < 30 and math.isfinite(free) and free > 0
    except (KeyError, TypeError, ValueError):
        return False


def _observe_eligibility(history, uuid, snapshot):
    previous, count = history.get(uuid, (None, 0))
    timestamp = snapshot.get('observed_monotonic') if snapshot else None
    if not _gpu_eligible(snapshot) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        history[uuid] = (timestamp, 0)
        return False
    if timestamp != previous:
        count = count + 1 if previous is None or timestamp > previous else 1
    history[uuid] = (timestamp, count)
    return count >= 2


class CIUnavailable(ValueError):
    """The primary bootstrap interval could not be estimated."""


def _read_completed_output(row: dict, state: dict) -> dict:
    output = Path(state['output'])
    marker = runtime.read_json(output / 'worker_exit.json')
    records._same(marker['status'], 'completed', 'worker commitment status')
    hashes = marker['artifact_sha256']
    records._same(set(hashes), {'config.json', 'result.json', 'result.csv'}, 'committed artifacts')
    for name, digest in hashes.items():
        records._same(runtime.sha256(output / name), digest, f'artifact hash {name}')
    config = runtime.read_json(output / 'config.json')
    result = runtime.read_json(output / 'result.json')
    expected = {key: row[key] for key in ('protocol', 'method', 'lr', 'epochs', 'seed', 'dropout')}
    expected.update(target_epsilon=row['epsilon'], requested_batch_size=row['batch_size'],
                    hidden=row['mlp_hidden'] if row['method'] in ('mlp', 'dp_mlp') else row['gnn_hidden'])
    for key, value in expected.items():
        records._same(config[key], value, f'config {key}')
        records._same(result[key], value, f'result {key}')
    parameters = result['parameters']
    records._same(config['parameters'], parameters, 'config/result parameters')
    if row['method'].startswith('sparse_'):
        records._same(parameters['p2'], row['p2'], 'SparseGNN p2')
        records._same(parameters['r'], row.get('r', 1), 'SparseGNN radius')
        records._same(parameters['K_out'], row.get('K_out', 10), 'SparseGNN outgoing cap')
    metric = config['task']['primary_metric']
    records._same(config['metric'], metric, 'config primary metric')
    records._same(result['metric'], metric, 'result primary metric')
    for key in ('validation_metric', 'test_metric'):
        records._finite(result[key], key)
    records._verify_selection({**row, 'task': config['task']}, result, parameters)
    interval = result['test_confidence_intervals']
    for key, value in records.BOOTSTRAP.items():
        records._same(interval[key], value, f'bootstrap {key}')
    observations = interval['n_observations']
    records._require(type(observations) is int and observations > 0, 'bootstrap has no scored observations')
    primary = interval['metrics'].get(metric)
    if primary is None:
        raise CIUnavailable(f'primary interval unavailable: {metric}')
    records._require(isinstance(primary, dict), 'malformed primary interval')
    valid = primary['valid_resamples']
    records._require(type(valid) is int and 0 <= valid <= 1000, 'invalid bootstrap resample count')
    if valid == 0 or primary['lower'] is None or primary['upper'] is None:
        raise CIUnavailable(f'primary interval unavailable: {metric}')
    lower, upper = records._finite(primary['lower'], 'CI lower'), records._finite(primary['upper'], 'CI upper')
    records._require(lower <= upper, 'reversed primary interval')
    return result


def _validate_completed(row, state):
    try:
        return _read_completed_output(row, state)
    except CIUnavailable as error:
        state.update(status='ci_unavailable', reason=str(error))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError) as error:
        state.update(status='invalid', reason=f'{type(error).__name__}: {error}')
    return {}


def tables(root, rows, states):
    results, projections = [], []
    columns = ['protocol', 'method', 'epsilon', 'lr', 'batch_size', 'effective_batch_size',
               'epochs', 'seed', 'p2', 'hidden', 'dropout', 'metric', 'validation_metric',
               'test_metric', 'ci_lower', 'ci_upper', 'confidence_level', 'selected_epoch',
               'selected_step', 'parameters', 'status', 'attempt', 'reason', 'result_csv']
    if any('r' in row for row in rows):
        columns[9:9] = ['r', 'K_out']
    for row, state in zip(rows, states):
        actual = _validate_completed(row, state) if state['status'] == 'completed' else {}
        if actual and 'r' in row:
            from scripts.sparse_ablation_grid import run_relative_path
            output = Path(state['output']).resolve()
            records._require(output.is_relative_to(root.resolve()), 'ablation output escapes run root')
            published = root / run_relative_path(row)
            records._require(published.parent.resolve().is_relative_to(root.resolve()),
                             'ablation run link parent escapes run root')
            if published.is_symlink():
                records._same(published.resolve(), output, 'accepted ablation run link')
            else:
                records._require(not published.exists(), f'occupied ablation run path: {published}')
                published.parent.mkdir(parents=True, exist_ok=True)
                published.symlink_to(os.path.relpath(output, published.parent), target_is_directory=True)
        result = {key: value for key, value in row.items() if key != 'argv'}
        result.update(actual)
        result.update(status=state['status'], attempt=state['attempt'], reason=state.get('reason'),
                      result_csv=str(Path(state['output']) / 'result.csv') if actual else '')
        results.append(result)
        projection = {key: row.get(key) for key in columns}
        projection['hidden'] = row['mlp_hidden'] if row['method'] in ('mlp', 'dp_mlp') else row['gnn_hidden']
        if actual:
            for key in ('effective_batch_size', 'hidden', 'dropout', 'metric',
                        'validation_metric', 'test_metric', 'parameters'):
                projection[key] = actual[key]
            interval = actual['test_confidence_intervals']
            primary = interval['metrics'][actual['metric']]
            selection = actual['selection']
            projection.update(ci_lower=primary['lower'], ci_upper=primary['upper'],
                              confidence_level=interval['confidence_level'], selected_step=selection['step'],
                              selected_epoch=selection.get('epoch', selection['step'] // actual['parameters']['evaluate_every']))
        projection.update({key: result[key] for key in ('status', 'attempt', 'reason', 'result_csv')})
        projections.append(projection)

    def cell(value):
        return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value

    for name, data, fields in (
        ('results.csv', results, list(dict.fromkeys(key for result in results for key in result))),
        ('summary.csv', projections, columns),
    ):
        temporary = root / f'.{name}.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: cell(value) for key, value in result.items()} for result in data)
        temporary.replace(root / name)
    temporary = root / '.summary.md.tmp'
    with temporary.open('w') as stream:
        stream.write('| ' + ' | '.join(columns) + ' |\n')
        stream.write('| ' + ' | '.join('---' for _ in columns) + ' |\n')
        for projection in projections:
            values = [str(cell(projection[key])) if projection[key] is not None else '' for key in columns]
            stream.write('| ' + ' | '.join(value.replace('|', '&#124;').replace('\r', '&#13;').replace('\n', '<br>')
                                          for value in values) + ' |\n')
    temporary.replace(root / 'summary.md')


def _ablation_rows(root: Path, *, report_only: bool) -> list[dict]:
    from scripts import sparse_ablation_grid as grid

    path = root / 'manifest.json'
    configurations = grid.configurations('ofat')
    expected = [{**config, 'run_dir': grid.run_relative_path(config)}
                for config in configurations]
    if path.exists():
        manifest = runtime.read_json(path)
        records._same(manifest['schema_version'], 1, 'ablation manifest version')
        records._same(manifest['study'], 'ofat', 'ablation study')
        records._same(manifest['configurations'], expected, 'ablation configurations')
        if not report_only:
            records._same(manifest['source_sha256'], grid.source_hashes(),
                          'ablation sources changed; use a fresh output root')
    else:
        if report_only or any(entry.name != 'queue.lock' for entry in root.iterdir()):
            raise ValueError('ablation requires a fresh root or its existing manifest')
        manifest = grid.make_manifest(root, sys.executable, 'cuda')
        runtime.atomic_json(path, manifest)
    rows = []
    for config in configurations:
        command = grid.worker_command(config, root / grid.run_relative_path(config),
                                      manifest['python'], manifest['device'])
        rows.append({**config, 'dropout': 0.5, 'mlp_hidden': 64, 'gnn_hidden': 128,
                     'argv': command})
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out-root', type=Path, required=True)
    cli.add_argument('--gpus', default='auto')
    cli.add_argument('--batch-size', type=int, choices=(256, 1024), default=1024,
                     help='requested batch size for every configuration (default: 1024)')
    cli.add_argument('--ablation-ofat', action='store_true',
                     help='run the fixed 60-cell SparseExpand one-factor study (requires --batch-size 256)')
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--retry-failed', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    args = cli.parse_args()
    if args.ablation_ofat and args.batch_size != 256:
        cli.error('--ablation-ofat requires --batch-size 256')
    root = args.out_root.absolute()
    if args.ablation_ofat:
        from scripts import sparse_ablation_grid as grid
        grid._reject_symlinks(root)
        if root.exists():
            if not (root / 'manifest.json').is_file():
                cli.error('ablation output root is occupied; use a fresh root')
            if runtime.read_json(root / 'manifest.json').get('study') != 'ofat':
                cli.error('output root belongs to a different study')
        else:
            grid._fresh_root(root)
    if args.report_only and not all((root / name).is_file() for name in ('requests.json', 'queue_state.json')):
        cli.error('--report-only requires an existing registry and queue state')
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    with runtime.file_lock(root / 'queue.lock', nonblocking=True):
        rows = (_ablation_rows(root, report_only=args.report_only) if args.ablation_ofat
                else records.enumerate_grid(batch_size=args.batch_size))
        registry = root / 'requests.json'
        if registry.exists():
            if runtime.read_json(registry) != rows:
                raise ValueError('existing queue has a different grid')
        else:
            runtime.atomic_json(registry, rows)
        state_path = root / 'queue_state.json'
        states = runtime.read_json(state_path) if state_path.exists() else [
            {'status': 'pending', 'attempt': 0, 'ooms': 0, 'not_before': 0} for _ in rows]
        if len(states) != len(rows):
            raise ValueError('queue state count differs from registry')
        if args.report_only:
            tables(root, rows, states)
            runtime.atomic_json(state_path, states)
            expected_count = 60 if args.ablation_ofat else 336
            return 0 if len(rows) == expected_count and all(state['status'] == 'completed' for state in states) else 1
        # The first wave spreads loaders across datasets rather than racing one cache.
        groups = [[i for i, row in enumerate(rows) if row['protocol'] == protocol]
                  for protocol in dict.fromkeys(row['protocol'] for row in rows)]
        order = [i for wave in zip(*groups) for i in wave]

        def finish(index, outcome):
            state = states[index]
            status = outcome['status']
            if status in ('completed', 'recovered_committed'):
                output = Path(state['folder']) / 'output'
                state.update(output=str(output))
                _validate_completed(rows[index], state)
                if state['status'] not in ('invalid', 'ci_unavailable'):
                    state.update(status='completed', reason=None)
            elif status == 'oom':
                burst = state['ooms'] + 1
                state.update(status='pending', ooms=burst % 3,
                             not_before=time.time() + (300 if burst >= 3 else 0), reason=outcome.get('reason'))
                order.remove(index)
                order.insert(0, index)
            else:
                state.update(status='pending' if status == 'interrupted' else status,
                             reason=outcome.get('reason'))
            runtime.atomic_json(state_path, states)
            print(json.dumps({'event': 'FINISHED', 'dataset': rows[index]['protocol'],
                              'method': rows[index]['method'], 'attempt': state['attempt'],
                              'status': state['status'], 'reason': state.get('reason')}, sort_keys=True), flush=True)

        for index, state in enumerate(states):
            if state['status'] == 'running':
                folder = Path(state['folder'])
                outcome = runtime.recover_process(folder) if (folder / 'launch.json').exists() else {
                    'status': 'ownership_unverified', 'reason': 'launch identity absent; inspect before retry'}
                finish(index, outcome)
        # Revalidate saved completions before deciding which failed cells may retry.
        tables(root, rows, states)
        if args.retry_failed:
            for state in states:
                if state['status'] not in ('failed', 'timeout', 'invalid', 'ci_unavailable'):
                    continue
                try:
                    evidence = runtime.read_json(Path(state['folder']) / 'exit.json')
                    if evidence.get('owned_process_exited') is not True:
                        raise ValueError('owned process exit not established')
                    if evidence.get('os_exit_observed') is not True or evidence.get('returncode') is None:
                        raise ValueError('clean OS exit evidence absent')
                except (OSError, ValueError, TypeError, KeyError) as error:
                    state.update(status='ownership_unverified', reason=str(error))
                    continue
                state.update(status='pending', reason=None, not_before=0, ooms=0)
        runtime.atomic_json(state_path, states)
        tables(root, rows, states)
        devices = runtime.resolve_gpus(args.gpus)
        uuids = [device['uuid'] for device in devices]
        sampler = runtime.GpuSampler(uuids).start()
        active = {}
        quarantined = set()
        leases = []
        eligibility = {}
        print('QUEUE_READY', flush=True)
        print(json.dumps({'requests': len(rows), 'authorized_gpus': uuids, 'root': str(root)}), flush=True)
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(uuids))) as pool:
                while not stop.is_set() or active:
                    changed = False
                    for uuid, (future, index, lease) in list(active.items()):
                        if not future.done():
                            continue
                        outcome = future.result()
                        finish(index, outcome)
                        if outcome['owned_process_exited']:
                            lease.__exit__(None, None, None)
                            temporary = Path(states[index]['folder']) / 'tmp'
                            if temporary.exists():
                                shutil.rmtree(temporary)
                        else:
                            quarantined.add(uuid)
                            leases.append(lease)
                            stop.set()
                        del active[uuid]
                        changed = True
                    if changed:
                        tables(root, rows, states)
                    if not stop.is_set():
                        for uuid in uuids:
                            if uuid in active or uuid in quarantined:
                                continue
                            observed = sampler.snapshot(uuid)
                            if not _observe_eligibility(eligibility, uuid, observed):
                                continue
                            index = next((i for i in order if states[i]['status'] == 'pending'
                                          and states[i]['not_before'] <= time.time()), None)
                            if index is None:
                                break
                            if runtime.host_available_bytes() < (20 + 2 * (len(active) + 1)) * runtime.GIB:
                                break
                            if shutil.disk_usage(root).free < 50 * runtime.GIB:
                                break
                            if args.ablation_ofat:
                                records._same(grid.source_hashes(),
                                              runtime.read_json(root / 'manifest.json')['source_sha256'],
                                              'ablation sources changed before launch')
                            lease = runtime.file_lock(ROOT / 'results/.full_matrix_gpu_locks' / f'{uuid}.lock', nonblocking=True)
                            try:
                                handle = lease.__enter__()
                            except BlockingIOError:
                                continue
                            eligibility[uuid] = (observed['observed_monotonic'], 0)
                            observed = runtime.gpu_snapshot(uuid)
                            if not _gpu_eligible(observed):
                                lease.__exit__(None, None, None)
                                continue
                            row, state = rows[index], states[index]
                            state['attempt'] += 1
                            folder = root / 'attempts' / f'{index:03d}_{row["protocol"]}_{row["method"]}' / f'attempt_{state["attempt"]}'
                            folder.mkdir(parents=True, exist_ok=False)
                            command = list(row['argv'])
                            command[command.index('--out-dir') + 1] = str(folder / 'output')
                            runtime.atomic_json(folder / 'request.json', row)
                            state.update(status='running', folder=str(folder), gpu_uuid=uuid)
                            runtime.atomic_json(state_path, states)
                            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'OMP_NUM_THREADS': '2',
                                           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                                           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
                            future = pool.submit(runtime.run_process, command, folder, environment, stop,
                                                 gpu=observed, sampler=sampler, lease_fd=handle.fileno(),
                                                 policy={'hard_seconds': None})
                            active[uuid] = (future, index, lease)
                            print(json.dumps({'event': 'STARTED', 'dataset': row['protocol'], 'method': row['method'],
                                              'gpu': uuid, 'attempt': state['attempt'], 'folder': str(folder)}), flush=True)
                    if not active and not any(state['status'] in ('pending', 'running') for state in states):
                        break
                    if stop.is_set() and not active:
                        break
                    time.sleep(1)
        finally:
            stop.set()
            sampler.close()
            tables(root, rows, states)
        return 0 if all(state['status'] == 'completed' for state in states) else 1


if __name__ == '__main__':
    raise SystemExit(main())
