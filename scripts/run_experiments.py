#!/usr/bin/env python3
"""Run JSON-configured graph experiments with ownership-safe GPU packing."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_experiment as worker
from scripts import runner_runtime as runtime

MIB = 1024 ** 2
GIB = 1024 ** 3
POLL_SECONDS = 2.0
EARLY_PROFILE_SECONDS = 30.0
EARLY_PROFILE_GROWTH = 1.10


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def _keys(value, allowed, label):
    _object(value, label)
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown keys {sorted(unknown)}")


def _positive(value, label, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a positive {'integer' if integer else 'number'}")
    if not math.isfinite(value) or value <= 0 or (integer and type(value) is not int):
        raise ValueError(f"{label}: expected a positive {'integer' if integer else 'number'}")
    return value


def _name(value):
    if not isinstance(value, str) or value in {'', '.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('name: expected a nonempty path-safe basename')
    return value


def _gpus(value):
    if isinstance(value, list):
        if not value or any(isinstance(v, bool) or not isinstance(v, (str, int)) for v in value):
            raise ValueError('gpus: expected a nonempty list of indices or UUIDs')
        value = ','.join(map(str, value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError('gpus: expected auto or a nonempty device list')
    return value


def load_config(path: Path) -> dict:
    path = Path(path).absolute()
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    with path.open(encoding='utf-8') as stream:
        config = json.load(stream, parse_constant=invalid_constant)
    _keys(config, {'name', 'gpus', 'defaults', 'grid', 'runs'}, 'config')
    config['name'] = _name(config.get('name', path.stem))
    if 'gpus' in config:
        config['gpus'] = _gpus(config['gpus'])
    # Executables are relative to the config, unlike repository-relative split data.
    scopes = [(config.get('defaults', {}), False), (config.get('grid', {}), True)]
    if isinstance(config.get('runs'), list):
        for block in config['runs']:
            if isinstance(block, dict):
                scopes += [(block.get('parameters', {}), False), (block.get('grid', {}), True)]
    for scope, grid in scopes:
        if not isinstance(scope, dict) or 'progap_python' not in scope:
            continue
        values = scope['progap_python'] if grid else [scope['progap_python']]
        if not isinstance(values, list):
            continue
        resolved = [str((path.parent / v).absolute()) if isinstance(v, str) and '/' in v and not Path(v).is_absolute() else v for v in values]
        scope['progap_python'] = resolved if grid else resolved[0]
    return config


def _scope(parameters, grid, label):
    _object(parameters, f'{label}.parameters')
    _object(grid, f'{label}.grid')
    overlap = set(parameters) & set(grid)
    if overlap:
        raise ValueError(f'{label}: scalar/grid ambiguity for {sorted(overlap)}')
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'{label}.grid.{key}: expected a nonempty list')


def scientific_parameters(parameters):
    return {k: v for k, v in parameters.items() if k != 'progap_python'}


def expand_runs(config: dict) -> list[dict]:
    _keys(config, {'name', 'gpus', 'defaults', 'grid', 'runs'}, 'config')
    defaults, common_grid = config.get('defaults', {}), config.get('grid', {})
    _scope(defaults, common_grid, 'config')
    blocks = config.get('runs')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('runs: expected a nonempty list')
    jobs, seen = [], set()
    for index, block in enumerate(blocks):
        label = f'runs[{index}]'
        _keys(block, {'parameters', 'grid', 'resources'}, label)
        parameters, axes = block.get('parameters', {}), block.get('grid', {})
        _scope(parameters, axes, label)
        resources = block.get('resources', {})
        _keys(resources, {'gpu_memory_mib', 'host_memory_mib'}, f'{label}.resources')
        for key, value in resources.items():
            _positive(value, f'{label}.resources.{key}')
        values, grid = dict(defaults), dict(common_grid)
        for key, value in parameters.items():
            values[key] = value
            grid.pop(key, None)
        for key, choices in axes.items():
            grid[key] = choices
            values.pop(key, None)
        for combination in itertools.product(*grid.values()):
            try:
                normalized = worker.normalize_parameters({**values, **dict(zip(grid, combination))})
            except (TypeError, ValueError) as error:
                raise ValueError(f'{label}: {error}') from error
            identity = json.dumps(scientific_parameters(normalized), sort_keys=True, allow_nan=False)
            if identity in seen:
                raise ValueError(f'{label}: duplicate normalized scientific configuration')
            seen.add(identity)
            ordinal = len(jobs)
            dataset = re.sub(r'[^A-Za-z0-9_.-]', '_', normalized['dataset'])
            method = re.sub(r'[^A-Za-z0-9_.-]', '_', normalized['method'])
            jobs.append({'id': f'{ordinal:04d}_{dataset}_{method}_s{normalized["seed"]}',
                         'run_index': ordinal, 'parameters': normalized, 'resources': dict(resources),
                         'status': 'pending', 'attempt': 0, 'output_dir': None,
                         'log_path': None, 'error': None})
    return jobs


def worker_command(parameters, output, device):
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_experiment.py')]
    for key, value in parameters.items():
        if value is not None:
            command += ['--' + key.replace('_', '-'), json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return command + ['--device', device, '--out-dir', str(output)]


def profile_key(parameters, model=None):
    values = {k: v for k, v in parameters.items() if k not in {'seed', 'lr', 'epsilon', 'epochs'}}
    values['worker_python'] = sys.executable
    if model is not None:
        values['gpu_model'] = model
    return json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)


def gpu_reservation(peak):
    return math.ceil(1.25 * peak) + 512 * MIB


def gpu_headroom(total):
    return max(2 * GIB, math.ceil(.10 * total))


def _fresh(snapshot):
    return bool(snapshot and not snapshot.get('error') and
                time.monotonic() - snapshot.get('observed_monotonic', -math.inf) <= POLL_SECONDS + 2 * runtime.NVIDIA_TIMEOUT)


def _idle(snapshot):
    return bool(_fresh(snapshot) and not snapshot['compute_processes'] and
                snapshot['utilization_gpu'] <= 5 and snapshot['memory_used_mib'] <= 1024)


def read_completed(job):
    folder = Path(job['output_dir'])
    result = runtime.read_json(folder / 'result.json')
    with (folder / 'result.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if result.get('status') != 'completed' or len(rows) != 1 or rows[0].get('status') != 'completed':
        raise ValueError('missing completed scientific result')
    row = rows[0]
    for value in (result, row):
        if value.get('method') != job['parameters']['method'] or value.get('protocol', value.get('dataset')) != job['parameters']['dataset']:
            raise ValueError('scientific result identity does not match the job')
        if int(value['seed']) != job['parameters']['seed'] or not value.get('metric'):
            raise ValueError('scientific result seed/metric is missing or inconsistent')
        for key in ('test_metric', 'validation_metric'):
            if not math.isfinite(float(value[key])):
                raise ValueError(f'nonfinite scientific {key}')
    return result, row


class ExperimentRunner:
    """The controller alone mutates jobs, reservations, profiles, and leases."""

    def __init__(self, root, jobs, args, gpus):
        self.root, self.jobs, self.args = Path(root), jobs, args
        self.gpus = {gpu['uuid']: gpu for gpu in gpus}
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.active, self.leases, self.profiles = {}, {}, {}
        self.live_profiles = {}
        self.exclusive, self.quarantined = set(), set()
        self.idle_history = {}
        self.owned_pids = frozenset()
        self.waiting_reason = None
        self.signal_number = None
        self.sampler = runtime.GpuSampler(self.gpus, poll_seconds=POLL_SECONDS) if self.gpus else None
        self.lock_dir = ROOT / 'results/.gpu_locks'

    def log(self, event, job=None, **fields):
        row = {'timestamp': runtime.utc_now(), 'event': event}
        if job is not None:
            row.update(run_id=job['id'], attempt=job['attempt'], gpu_uuid=job.get('gpu_uuid'),
                       output_dir=job['output_dir'], log_path=job['log_path'])
        row.update(fields)
        runtime.append_jsonl(self.root / 'runner.log', row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def save(self):
        runtime.atomic_json(self.root / 'state.json', {
            'version': 1, 'root': str(self.root), 'jobs': self.jobs,
            'profiles': self.profiles, 'exclusive': sorted(self.exclusive)})
        rows = []
        for job in self.jobs:
            row = dict(job.get('result_row') or job['parameters'])
            row.update(status=job['status'], run_id=job['id'], run_index=job['run_index'],
                       attempt=job['attempt'], gpu_uuid=job.get('gpu_uuid'),
                       returncode=job.get('exit', {}).get('returncode'),
                       wall_seconds=job.get('exit', {}).get('wall_seconds'),
                       output_dir=job['output_dir'], log_path=job['log_path'], error=job['error'])
            rows.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        columns = list(dict.fromkeys(['status', 'dataset', 'protocol', 'method', 'seed', 'metric',
                                     'validation_metric', 'test_metric'] + [k for row in rows for k in row]))
        temporary = self.root / 'results.csv.partial'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.root / 'results.csv')

    def _recover_orphans(self, job):
        attempts = self.root / 'runs' / job['id'] / 'attempts'
        if not attempts.exists():
            return
        for folder in sorted(attempts.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not folder.is_dir() or not folder.name.isdigit() or int(folder.name) <= job['attempt']:
                continue
            job.update(attempt=int(folder.name), output_dir=str(folder / 'output'),
                       log_path=str(folder / 'process.log'), status='interrupted')
            launch_path, exit_path = folder / 'launch.json', folder / 'exit.json'
            if launch_path.exists():
                launch = runtime.read_json(launch_path)
                job['gpu_uuid'] = launch.get('gpu_uuid')
                job['gpu_model'] = (launch.get('gpu') or {}).get('name')
            if not launch_path.exists() and not (folder / 'process.log').exists():
                # run_process opens process.log before spawning. This directory
                # was reserved by the controller, but no launch was attempted.
                record = {'status': 'interrupted', 'returncode': None, 'owned_process_exited': True,
                          'os_exit_observed': False, 'reason': 'controller stopped before worker launch'}
                (folder / 'process.log').write_text(record['reason'] + '\n')
                runtime.atomic_json(exit_path, record)
            else:
                record = runtime.read_json(exit_path) if exit_path.exists() else None
                if not record or not record.get('owned_process_exited'):
                    record = runtime.recover_process(folder)
            if not record.get('owned_process_exited'):
                job.update(status='blocked', error='ownership_unverified', exit=record)
                self._quarantine(job, folder)
                return
            self.log('interrupted', job, reason='recovered attempt absent from saved state')

    def restore(self):
        state = runtime.read_json(self.root / 'state.json')
        if state.get('version') != 1 or state.get('root') != str(self.root):
            raise ValueError('resume requires a version-1 state at its original absolute root')
        old = state['jobs']
        def identity(job):
            return scientific_parameters(job['parameters']), job['resources']
        if [identity(job) for job in old] != [identity(job) for job in self.jobs]:
            raise ValueError('resume configuration differs from the saved scientific jobs/resources')
        self.exclusive = set(state.get('exclusive', []))
        self.profiles = {}
        for previous, planned in zip(old, self.jobs):
            parameters = planned['parameters']
            planned.update(previous)
            planned.pop('result_row', None)
            self._recover_orphans(planned)
            status = planned['status']
            folder = Path(planned['output_dir']).parent if planned.get('output_dir') else None
            if folder is not None:
                exit_path = folder / 'exit.json'
                record = runtime.read_json(exit_path) if exit_path.exists() else None
                clean = record and record.get('owned_process_exited') is True
                if not clean:
                    record = runtime.recover_process(folder)
                    clean = record.get('owned_process_exited') is True
                planned['exit'] = record
                if not clean:
                    planned.update(status='blocked', error='ownership_unverified')
                    self._quarantine(planned, folder)
                    continue
                if record.get('status') == 'completed' and record.get('returncode') == 0:
                    try:
                        result, row = read_completed(planned)
                        planned.update(status='completed', result_row=row, error=None)
                        self._learn(planned, result)
                        continue
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        status = 'failed'
                        planned['error'] = f'invalid completed output: {error}'
                elif status in {'running', 'interrupted', 'blocked'}:
                    status = 'interrupted'
                elif status == 'completed':
                    status = 'failed'
                    planned['error'] = 'completed state has no observed successful exit'
            if status in {'pending', 'running', 'interrupted', 'blocked'} or (status == 'failed' and self.args.retry_failed):
                planned.update(status='pending', error=None)
            else:
                planned['status'] = status
            if planned['status'] == 'pending' and planned['parameters'].get('progap_python') != parameters.get('progap_python'):
                planned['parameters'] = parameters
                self.log('retry', planned, reason='ProGAP interpreter changed for pending/retried execution')
        self.save()

    def _quarantine(self, job, folder):
        uuid = job.get('gpu_uuid')
        if uuid:
            self.quarantined.add(uuid)
            runtime.atomic_json(self.lock_dir / f'{uuid}.quarantine.json', {'attempt': str(folder)})

    def _refresh_live_profiles(self):
        """Use stable live peaks provisionally, never persist unfinished profiles."""
        self.live_profiles = {}
        now = time.monotonic()
        for active in self.active.values():
            gpu, job = active['gpu'], active['job']
            sample = active.get('sample', {})
            snapshot = self.sampler.snapshot(gpu['uuid']) if gpu else None
            observed = sample.get('observed_monotonic', -math.inf)
            gpu_peak = max(sample.get('peak_gpu_memory_mib') or 0,
                           sample.get('owned_gpu_memory_mib') or 0) * MIB
            host_peak = max(sample.get('peak_rss_bytes') or 0,
                            sample.get('tree_rss_bytes') or 0)
            if (not _fresh(snapshot) or not active.get('ownership_known') or
                    profile_key(job['parameters']) in self.exclusive or
                    sample.get('host_error') or not sample.get('tree_rss_bytes') or
                    not sample.get('owned_gpu_memory_mib') or not gpu_peak or not host_peak or
                    now - observed > POLL_SECONDS + 2 * runtime.NVIDIA_TIMEOUT or
                    any(p['pid'] not in self.owned_pids for p in snapshot['compute_processes'])):
                active.pop('warmup', None)
                continue
            warmup = active.get('warmup')
            if warmup is None:
                if snapshot['utilization_gpu'] <= 0:
                    continue
                warmup = active['warmup'] = {
                    'since': observed, 'gpu_bytes': gpu_peak, 'host_bytes': host_peak}
            if (gpu_peak > EARLY_PROFILE_GROWTH * warmup['gpu_bytes'] or
                    host_peak > EARLY_PROFILE_GROWTH * warmup['host_bytes']):
                warmup.update(since=observed, gpu_bytes=gpu_peak, host_bytes=host_peak)
            if observed - warmup['since'] < EARLY_PROFILE_SECONDS:
                continue
            key = profile_key(job['parameters'], gpu['name'])
            previous = self.live_profiles.get(key, {})
            self.live_profiles[key] = {
                'gpu_bytes': max(previous.get('gpu_bytes', 0), gpu_peak),
                'host_bytes': max(previous.get('host_bytes', 0), host_peak)}

    def _peaks(self, job, gpu):
        key = profile_key(job['parameters'], gpu['name'] if gpu else 'cpu')
        profile = self.profiles.get(key, {})
        live = self.live_profiles.get(key, {})
        estimates = job['resources']
        return (max(profile.get('gpu_bytes', 0), live.get('gpu_bytes', 0),
                    estimates.get('gpu_memory_mib', 0) * MIB) or None,
                max(profile.get('host_bytes', 0), live.get('host_bytes', 0),
                    estimates.get('host_memory_mib', 0) * MIB) or 8 * GIB)

    def _solo_gpu_requirement(self, job, gpu):
        profile = self.profiles.get(profile_key(job['parameters'], gpu['name']), {})
        # A provisional live peak may permit sharing, but must never reject a
        # potentially fitting solo job through inflated sharing reservations.
        peak = max(profile.get('gpu_bytes', 0),
                   job['resources'].get('gpu_memory_mib', 0) * MIB)
        if not peak:
            return None
        if profile.get('gpu_bytes', 0) >= peak:
            # A successful whole-run peak already includes the CUDA context.
            # Sharing margins must not disqualify a shape that fits alone.
            return math.ceil(peak)
        return gpu_reservation(peak) + gpu_headroom(gpu['memory_total_mib'] * MIB)

    def _learn(self, job, result):
        record = job['exit']
        if not record.get('owned_process_exited') or record.get('returncode') != 0:
            return
        resources = result.get('resources', {})
        host = record.get('peak_rss_bytes')
        gpu = record.get('peak_gpu_memory_mib')
        gpu = gpu * MIB if gpu is not None and gpu > 0 else None
        parent = resources.get('peak_cuda_allocated_bytes')
        child = resources.get('peak_child_cuda_allocated_bytes')
        if parent and (job['parameters']['method'] != 'progap' or child is not None):
            gpu = max(gpu or 0, parent + (child or 0))
        if host is None or host <= 0:
            return
        key = profile_key(job['parameters'], job.get('gpu_model') or 'cpu')
        old = self.profiles.get(key, {})
        learned = {'host_bytes': max(old.get('host_bytes', 0), host)}
        if gpu and not record.get('telemetry_error'):
            learned['gpu_bytes'] = max(old.get('gpu_bytes', 0), gpu)
        elif old.get('gpu_bytes'):
            learned['gpu_bytes'] = old['gpu_bytes']
        self.profiles[key] = learned

    def _refresh_owned(self):
        pids = set()
        for active in self.active.values():
            try:
                owned = runtime.owned_attempt_pids(active['folder'])
                active['owned_pids'] = owned
                active['ownership_known'] = True
                pids.update(owned)
            except (OSError, RuntimeError, ValueError, KeyError):
                active['ownership_known'] = False
        self.owned_pids = frozenset(pids)

    def _host_fits(self, job, gpu):
        try:
            available = runtime.host_available_bytes()
            with Path('/proc/meminfo').open() as stream:
                total = next(int(line.split()[1]) * 1024 for line in stream if line.startswith('MemTotal:'))
        except (OSError, RuntimeError, ValueError, StopIteration):
            return not self.active
        reserved = 0
        for active in self.active.values():
            sample = active.get('sample', {})
            if (sample.get('tree_rss_bytes') is None or sample.get('host_error') or
                    time.monotonic() - sample.get('observed_monotonic', -math.inf) > POLL_SECONDS + 2 * runtime.NVIDIA_TIMEOUT):
                return False
            _, peak = self._peaks(active['job'], active['gpu'])
            peak = max(peak, sample.get('peak_rss_bytes') or 0)
            reserved += max(0, math.ceil(1.25 * peak) - sample['tree_rss_bytes'])
        _, peak = self._peaks(job, gpu)
        return reserved + math.ceil(1.25 * peak) + max(2 * GIB, math.ceil(.1 * total)) <= available

    def _gpu_fits(self, job, gpu, snapshot):
        if not _fresh(snapshot):
            return False
        active = [a for a in self.active.values() if a['job'].get('gpu_uuid') == gpu['uuid']]
        if self.args.max_jobs_per_gpu is not None and len(active) >= self.args.max_jobs_per_gpu:
            return False
        peak, _ = self._peaks(job, gpu)
        exclusive = profile_key(job['parameters']) in self.exclusive
        if not active:
            required = self._solo_gpu_requirement(job, gpu)
            return _idle(snapshot) and (required is None or required <= snapshot['memory_free_mib'] * MIB)
        if exclusive:
            return False
        if peak is None:
            # Probe one new shape using a 30% peak estimate; active jobs still
            # need measured profiles before any further sharing is admitted.
            peak = .30 * snapshot['memory_total_mib'] * MIB
        owned = set()
        reservation = gpu_reservation(peak)
        for attempt in active:
            previous, _ = self._peaks(attempt['job'], gpu)
            sample = attempt.get('sample', {})
            if (previous is None or not attempt.get('ownership_known') or
                    profile_key(attempt['job']['parameters']) in self.exclusive or
                    sample.get('owned_gpu_memory_mib') is None or
                    time.monotonic() - sample.get('observed_monotonic', -math.inf) > POLL_SECONDS + 2 * runtime.NVIDIA_TIMEOUT):
                return False
            attempt_pids = attempt.get('owned_pids', set())
            current = [p.get('used_memory_mib') for p in snapshot['compute_processes'] if p['pid'] in attempt_pids]
            if any(value is None for value in current):
                return False
            owned.update(attempt_pids)
            reservation += gpu_reservation(max(previous, (sample.get('peak_gpu_memory_mib') or 0) * MIB,
                                               sum(current) * MIB))
        processes = snapshot['compute_processes']
        if any(p['pid'] not in owned or p.get('used_memory_mib') is None for p in processes):
            return False
        total = snapshot['memory_total_mib'] * MIB
        used = sum(p['used_memory_mib'] for p in processes) * MIB
        return reservation + gpu_headroom(total) <= min(total, snapshot['memory_free_mib'] * MIB + used)

    def _lease(self, gpu, snapshot):
        uuid = gpu['uuid']
        if uuid in self.quarantined:
            return False
        if uuid in self.leases:
            lease = self.leases[uuid][1]
            try:
                held = os.fstat(lease.fileno())
                current = (self.lock_dir / f'{uuid}.lock').stat()
                if (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino):
                    return True
            except (OSError, ValueError):
                pass
            self.quarantined.add(uuid)
            return False
        last, count = self.idle_history.get(uuid, (None, 0))
        if not _idle(snapshot):
            self.idle_history[uuid] = (None, 0)
            return False
        observed = snapshot['observed_monotonic']
        if last != observed:
            count = count + 1 if last is not None and observed - last <= POLL_SECONDS + 2 * runtime.NVIDIA_TIMEOUT else 1
            self.idle_history[uuid] = (observed, count)
        if count < 2:
            return False
        context = runtime.file_lock(self.lock_dir / f'{uuid}.lock', nonblocking=True)
        try:
            lease = context.__enter__()
        except BlockingIOError:
            return False
        try:
            quarantine = self.lock_dir / f'{uuid}.quarantine.json'
            if quarantine.exists():
                entry = runtime.read_json(quarantine)
                recovered = runtime.recover_process(entry['attempt'])
                if not recovered.get('owned_process_exited'):
                    self.quarantined.add(uuid)
                    return False
                quarantine.unlink()
            if not _idle(runtime.gpu_snapshot(uuid)):
                return False
            self.leases[uuid] = (context, lease)
            return True
        finally:
            if uuid not in self.leases:
                context.__exit__(None, None, None)

    def _release_unused(self):
        used = {a['job'].get('gpu_uuid') for a in self.active.values()}
        for uuid in list(self.leases):
            if uuid not in used:
                context, _ = self.leases.pop(uuid)
                context.__exit__(None, None, None)
                self.idle_history[uuid] = (None, 0)

    def _launch(self, job, gpu):
        job['attempt'] += 1
        folder = self.root / 'runs' / job['id'] / 'attempts' / str(job['attempt'])
        folder.mkdir(parents=True, exist_ok=False)
        job.update(status='running', output_dir=str(folder / 'output'), log_path=str(folder / 'process.log'),
                   gpu_uuid=gpu['uuid'] if gpu else None, gpu_model=gpu['name'] if gpu else None, error=None)
        job.pop('result_row', None)
        job.pop('exit', None)
        same_gpu = [a for a in self.active.values() if a['job'].get('gpu_uuid') == job['gpu_uuid']]
        for active in same_gpu:
            active['shared'] = True
        active = {'job': job, 'gpu': gpu, 'folder': folder, 'shared': bool(same_gpu)}
        self.log('started', job)
        self.save()
        command = worker_command(job['parameters'], folder / 'output', self.args.device)
        environment = dict(os.environ)
        environment['PYTHONUNBUFFERED'] = '1'
        if gpu:
            environment['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            try:
                value = int(environment.get(key, 2))
            except ValueError:
                value = 2
            environment[key] = str(max(1, min(2, value)))
        lease_fd = self.leases[gpu['uuid']][1].fileno() if gpu else None
        def supervise():
            try:
                record = runtime.run_process(command, folder, environment, self.stop,
                    policy={'hard_seconds': self.args.timeout_seconds, 'poll_seconds': POLL_SECONDS},
                    lease_fd=lease_fd, gpu=gpu, sampler=self.sampler,
                    on_sample=lambda sample: self.events.put(('sample', job['id'], sample)),
                    co_owned_pids=lambda: self.owned_pids)
            except Exception as error:
                # A supervisor exception may have happened after spawn; recovery must prove cleanup.
                try:
                    with (folder / 'process.log').open('a') as log:
                        log.write(f'supervisor {type(error).__name__}: {error}\n')
                except OSError:
                    pass
                try:
                    record = runtime.recover_process(folder)
                except Exception as recovery_error:
                    record = {'status': 'ownership_unverified', 'owned_process_exited': False,
                              'returncode': None, 'recovery_error': str(recovery_error)}
                record.update(reason=f'supervisor {type(error).__name__}: {error}')
            self.events.put(('finished', job['id'], record))
        active['thread'] = threading.Thread(target=supervise, name=f'run-{job["id"]}')
        self.active[job['id']] = active
        try:
            active['thread'].start()
        except BaseException:
            self.active.pop(job['id'])
            raise

    def _finish(self, identifier, record):
        active = self.active.pop(identifier)
        active['thread'].join()
        job = active['job']
        job['exit'] = record
        job['error'] = record.get('reason')
        if not record.get('owned_process_exited'):
            job['status'] = 'blocked'
            self._quarantine(job, active['folder'])
        elif record.get('status') == 'completed' and record.get('returncode') == 0:
            try:
                result, row = read_completed(job)
                job.update(status='completed', result_row=row, error=None)
                self._learn(job, result)
            except (OSError, ValueError, TypeError, KeyError) as error:
                job.update(status='failed', error=f'invalid completed output: {error}')
        elif record.get('status') == 'interrupted':
            job['status'] = 'interrupted'
        else:
            job['status'] = 'failed'
        self.log(job['status'] if job['status'] != 'blocked' else 'failed', job, reason=job['error'])
        if (job['status'] == 'failed' and active['shared'] and
                record.get('error_attribution', {}).get('kind') == 'cuda_oom' and
                not job.get('exclusive_retry_used') and not self.stop.is_set()):
            key = profile_key(job['parameters'])
            self.exclusive.add(key)
            self.profiles.pop(profile_key(job['parameters'], job['gpu_model']), None)
            job.update(status='pending', exclusive_retry_used=True)
            self.log('retry', job, reason='shared CUDA OOM: one exclusive retry')
        self._release_unused()
        self.save()

    def _admit(self):
        pending = [job for job in self.jobs if job['status'] == 'pending']
        if self.args.device == 'cpu':
            if pending and not self.active and self._host_fits(pending[0], None):
                self._launch(pending[0], None)
                return True
            return False
        self._refresh_owned()
        self._refresh_live_profiles()
        changed = False
        for job in pending[:]:
            sizes = [(self._solo_gpu_requirement(job, gpu), gpu['memory_total_mib'] * MIB) for gpu in self.gpus.values()]
            if sizes and all(required is not None and required > total for required, total in sizes):
                job.update(status='failed', error='insufficient_gpu_memory')
                self.log('failed', job, reason=job['error'])
                pending.remove(job)
                changed = True
        counts = {uuid: sum(a['job'].get('gpu_uuid') == uuid for a in self.active.values()) for uuid in self.gpus}
        ordered_gpus = sorted(self.gpus.values(), key=lambda gpu: counts[gpu['uuid']])
        warming_idle = False
        for gpu in ordered_gpus:
            if not pending:
                break
            snapshot = self.sampler.snapshot(gpu['uuid'])
            occupied = any(a['job'].get('gpu_uuid') == gpu['uuid'] for a in self.active.values())
            if not occupied and not _idle(snapshot):
                self.idle_history[gpu['uuid']] = (None, 0)
                continue
            if occupied and warming_idle:
                continue
            candidates = sorted(pending, key=lambda j: (-(self._peaks(j, gpu)[0] or 0), j['run_index'])) if occupied else list(pending)
            for job in candidates:
                if not self._host_fits(job, gpu) or not self._gpu_fits(job, gpu, snapshot):
                    continue
                # Wait for an eligible idle GPU's second observation instead
                # of repeatedly packing a busy GPU while that proof warms up.
                if not self._lease(gpu, snapshot):
                    if (not occupied and gpu['uuid'] not in self.quarantined and
                            self.idle_history.get(gpu['uuid'], (None, 0))[1] == 1):
                        warming_idle = True
                    break
                fresh = runtime.gpu_snapshot(gpu['uuid'])
                if occupied:
                    self._refresh_owned()
                if not self._lease(gpu, fresh) or not self._gpu_fits(job, gpu, fresh):
                    break
                self._launch(job, gpu)
                # Refresh queued observations before choosing the next GPU.
                self._release_unused()
                return True
        self._release_unused()
        if pending and self.quarantined.issuperset(self.gpus):
            for job in pending:
                job.update(status='blocked', error='all authorized GPUs have unverified ownership cleanup')
                self.log('failed', job, reason=job['error'])
            changed = True
        if changed:
            self.save()
        return changed

    def run(self):
        previous = {}
        def request_stop(number, frame):
            self.signal_number = number
            self.stop.set()
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                previous[number] = signal.signal(number, request_stop)
        try:
            if self.sampler:
                self.sampler.start()
            while self.active or (not self.stop.is_set() and any(j['status'] == 'pending' for j in self.jobs)):
                try:
                    event = self.events.get(timeout=.2)
                    while True:
                        kind, identifier, value = event
                        if kind == 'sample' and identifier in self.active:
                            self.active[identifier]['sample'] = value
                        elif kind == 'finished':
                            self._finish(identifier, value)
                        # Admission consumes the newest observations, not an
                        # ever-growing backlog of samples from busy workers.
                        event = self.events.get_nowait()
                except queue.Empty:
                    pass
                if not self.stop.is_set():
                    changed = self._admit()
                    reason = None if changed else 'waiting for idle authorized GPUs or memory reservations' if self.gpus else 'waiting for worker or host memory'
                    if reason != self.waiting_reason:
                        self.waiting_reason = reason
                        if reason and any(j['status'] == 'pending' for j in self.jobs):
                            self.log('waiting', reason=reason)
            self.save()
        finally:
            self.stop.set()
            while self.active:
                kind, identifier, value = self.events.get()
                if kind == 'finished':
                    self._finish(identifier, value)
            self._release_unused()
            if self.sampler:
                self.sampler.close()
            for number, handler in previous.items():
                signal.signal(number, handler)
        if self.signal_number is not None:
            return 128 + self.signal_number
        return 0 if all(job['status'] == 'completed' for job in self.jobs) else 1


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('config', type=Path)
    cli.add_argument('--gpus')
    cli.add_argument('--out-dir', type=Path)
    cli.add_argument('--resume', action='store_true')
    cli.add_argument('--retry-failed', action='store_true')
    cli.add_argument('--max-jobs-per-gpu', type=int)
    cli.add_argument('--timeout-seconds', type=float)
    cli.add_argument('--progap-python')
    cli.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    cli.add_argument('--dry-run', action='store_true')
    return cli


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.max_jobs_per_gpu is not None:
            _positive(args.max_jobs_per_gpu, '--max-jobs-per-gpu', integer=True)
        if args.timeout_seconds is not None:
            _positive(args.timeout_seconds, '--timeout-seconds')
        if args.retry_failed and not args.resume:
            raise ValueError('--retry-failed requires --resume')
        config = load_config(args.config)
        if args.device == 'cpu' and (args.gpus is not None or 'gpus' in config or args.max_jobs_per_gpu is not None):
            raise ValueError('--device cpu rejects GPU selection and concurrency options')
        jobs = expand_runs(config)
        if args.progap_python:
            executable = args.progap_python
            if '/' in executable and not Path(executable).is_absolute():
                executable = str((args.config.absolute().parent / executable).absolute())
            for job in jobs:
                if job['parameters']['method'] == 'progap':
                    job['parameters']['progap_python'] = executable
        root = args.out_dir if args.out_dir is not None else Path('results') / config['name']
        root = Path(os.path.abspath(ROOT / root))
        gpu_spec = _gpus(args.gpus if args.gpus is not None else config.get('gpus', 'auto'))
        if args.dry_run:
            print(json.dumps({'name': config['name'], 'out_dir': str(root), 'count': len(jobs),
                              'device': args.device, 'gpus': gpu_spec, 'max_jobs_per_gpu': args.max_jobs_per_gpu,
                              'timeout_seconds': args.timeout_seconds, 'jobs': jobs}, indent=2, allow_nan=False))
            return 0
        if root.exists() and not args.resume:
            raise ValueError(f'output root exists: {root}; use --resume or a fresh --out-dir')
        if args.resume and not (root / 'state.json').is_file():
            raise ValueError(f'no resumable state at {root}')
        gpus = runtime.resolve_gpus(gpu_spec) if args.device == 'cuda' else []
        if args.device == 'cuda' and not gpus:
            raise ValueError('no authorized GPUs; provide an available allow-list or explicitly use --device cpu')
        if not args.resume:
            root.mkdir(parents=True, exist_ok=False)
        with runtime.file_lock(root / '.runner.lock', nonblocking=True):
            runner = ExperimentRunner(root, jobs, args, gpus)
            if args.resume:
                runner.restore()
            else:
                for job in jobs:
                    runner.log('queued', job)
                runner.save()
            runtime.atomic_json(root / 'experiment.json', {**config, 'execution': {
                'gpus': gpu_spec, 'device': args.device, 'max_jobs_per_gpu': args.max_jobs_per_gpu,
                'timeout_seconds': args.timeout_seconds, 'progap_python': args.progap_python}})
            return runner.run()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
