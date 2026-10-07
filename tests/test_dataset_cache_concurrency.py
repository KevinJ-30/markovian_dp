"""Real loader/process concurrency against small local benchmark artifacts."""
import hashlib
import json
import multiprocessing
from pathlib import Path
import queue
import traceback

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.data import datasets


_EXPECTED = {
    'reddit': ([0, 1, 1, 0], [True, True, False, False]),
    'yelp': ([[1, 0], [0, 1], [1, 1], [0, 0]], [True, True, False, False]),
}


def _fixture(root, name):
    folder = root / name
    folder.mkdir(parents=True)
    adjacency = sp.csr_matrix(
        (np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(4, 4))
    sp.save_npz(folder / 'adj_full.npz', adjacency)
    train = adjacency.copy().tolil()
    train[2:, :] = 0
    sp.save_npz(folder / 'adj_train.npz', train.tocsr())
    np.save(folder / 'feats.npy', np.arange(8).reshape(4, 2).astype(float))
    (folder / 'class_map.json').write_text(json.dumps(dict(enumerate(_EXPECTED[name][0]))))
    (folder / 'role.json').write_text(json.dumps({'tr': [0, 1], 'va': [2], 'te': [3]}))
    return folder / '_labels_cache.pt'


def _load(root, name):
    _, data = datasets._load_graphsaint(name, root=root)
    return data.y.tolist(), data.train_mask.tolist()


def _dataset_worker(root, name, mode, started, checkpoint, release, results):
    try:
        from src.processing import cache

        if mode == 'writer':
            original_save = torch.save

            def paused_save(value, destination, *args, **kwargs):
                own_stream = not hasattr(destination, 'write')
                stream = open(destination, 'wb') if own_stream else destination
                try:
                    stream.write(b'partial cache payload')
                    stream.flush()
                    checkpoint.set()
                    if not release.wait(60):
                        raise TimeoutError('cache writer was not released')
                    stream.seek(0)
                    stream.truncate()
                    return original_save(value, stream, *args, **kwargs)
                finally:
                    if own_stream:
                        stream.close()

            torch.save = paused_save
        elif mode == 'warm':
            original_load = torch.load

            def concurrent_load(path, *args, **kwargs):
                if Path(path).name == '_labels_cache.pt':
                    checkpoint.wait(timeout=60)
                return original_load(path, *args, **kwargs)

            torch.load = concurrent_load

            def no_lock(*args):
                raise AssertionError('warm atomic cache loading acquired flock')
            cache.fcntl.flock = no_lock
        started.set()
        results.put(('ok', _load(root, name)))
    except BaseException:
        results.put(('error', traceback.format_exc()))


def _finish(processes):
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail('dataset worker did not finish')
        assert process.exitcode == 0


def _stop(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)


@pytest.mark.parametrize('name', ['reddit', 'yelp'])
def test_cold_dataset_readers_never_deserialize_partial_cache(tmp_path, name):
    payload = _fixture(tmp_path, name)
    context = multiprocessing.get_context('spawn')
    partial, release = context.Event(), context.Event()
    results = context.Queue()
    writer_started, reader_started = context.Event(), context.Event()
    writer = context.Process(target=_dataset_worker, args=(
        tmp_path, name, 'writer', writer_started, partial, release, results))
    reader = context.Process(target=_dataset_worker, args=(
        tmp_path, name, 'reader', reader_started, None, None, results))
    processes = [writer]
    writer.start()
    try:
        assert partial.wait(60), 'writer did not reach the partial-cache boundary'
        assert not payload.exists(), 'an incomplete label cache became visible'
        reader.start()
        processes.append(reader)
        assert reader_started.wait(60)
        # While publication is blocked, a second loader must neither fail on
        # the partial payload nor publish its own competing cache.
        with pytest.raises(queue.Empty):
            results.get(timeout=0.5)
        release.set()
        observed = [results.get(timeout=60), results.get(timeout=60)]
        assert observed == [('ok', _EXPECTED[name]), ('ok', _EXPECTED[name])]
        _finish(processes)
        assert _load(tmp_path, name) == _EXPECTED[name]
    finally:
        release.set()
        _stop(processes)
        results.close()


@pytest.mark.parametrize('name', ['reddit', 'yelp'])
def test_warm_dataset_deserialization_overlaps_between_processes(tmp_path, name):
    _fixture(tmp_path, name)
    assert _load(tmp_path, name) == _EXPECTED[name]
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(2)
    results = context.Queue()
    started = [context.Event() for _ in range(2)]
    processes = [context.Process(target=_dataset_worker, args=(
        tmp_path, name, 'warm', started[index], barrier, None, results))
        for index in range(2)]
    for process in processes:
        process.start()
    try:
        # Both real deserializers must enter simultaneously; an exclusive
        # warm-load lock would prevent the barrier from completing.
        observed = [results.get(timeout=90), results.get(timeout=90)]
        assert observed == [('ok', _EXPECTED[name]), ('ok', _EXPECTED[name])]
        _finish(processes)
    finally:
        _stop(processes)
        results.close()


@pytest.mark.parametrize('failure', ['lock', 'temporary_file'])
def test_graphsaint_read_only_cache_directory_still_loads_labels(
        tmp_path, monkeypatch, failure):
    payload = _fixture(tmp_path, 'reddit')

    def denied(*args, **kwargs):
        raise PermissionError('read-only dataset directory')

    if failure == 'lock':
        monkeypatch.setattr(datasets, 'cache_creation_lock', denied)
    else:
        monkeypatch.setattr(datasets.tempfile, 'NamedTemporaryFile', denied)
    assert _load(tmp_path, 'reddit') == _EXPECTED['reddit']
    assert not payload.exists()


def _domain_worker(source, destination, checked, writer, started,
                   partial, release, results):
    try:
        from src.data import domain_datasets as domain

        if writer:
            original_copy = domain.shutil.copyfileobj

            def paused_copy(input_stream, output_stream, **kwargs):
                output_stream.write(b'partial download')
                output_stream.flush()
                partial.set()
                if not release.wait(60):
                    raise TimeoutError('domain download was not released')
                output_stream.seek(0)
                output_stream.truncate()
                return original_copy(input_stream, output_stream, **kwargs)

            domain.shutil.copyfileobj = paused_copy
        started.set()
        if checked:
            expected = source.read_bytes()
            path = domain._ensure_mag_file(
                source.as_uri(), destination, len(expected),
                hashlib.md5(expected).hexdigest())
        else:
            path = domain._ensure_file(source.as_uri(), destination)
        results.put(('ok', path.read_bytes()))
    except BaseException:
        results.put(('error', traceback.format_exc()))


@pytest.mark.parametrize('state', ['missing', 'checked_missing', 'checked_invalid'])
def test_domain_acquisition_publishes_once_without_removing_old_file(tmp_path, state):
    source = tmp_path / 'source'
    source.write_bytes(b'complete benchmark artifact')
    destination = tmp_path / 'cache' / 'artifact'
    if state == 'checked_invalid':
        destination.parent.mkdir()
        destination.write_bytes(b'obsolete')
    context = multiprocessing.get_context('spawn')
    partial, release = context.Event(), context.Event()
    results = context.Queue()
    writer_started, reader_started = context.Event(), context.Event()
    writer = context.Process(target=_domain_worker, args=(
        source, destination, state != 'missing', True, writer_started,
        partial, release, results))
    reader = context.Process(target=_domain_worker, args=(
        source, destination, state != 'missing', False, reader_started,
        None, None, results))
    processes = [writer]
    writer.start()
    try:
        assert partial.wait(60)
        if state == 'checked_invalid':
            assert destination.read_bytes() == b'obsolete'
        else:
            assert not destination.exists()
        reader.start()
        processes.append(reader)
        assert reader_started.wait(60)
        with pytest.raises(queue.Empty):
            results.get(timeout=0.5)
        release.set()
        expected = ('ok', source.read_bytes())
        assert [results.get(timeout=60), results.get(timeout=60)] == [expected, expected]
        _finish(processes)
        assert destination.read_bytes() == source.read_bytes()
        assert not list(destination.parent.glob('*.tmp'))
    finally:
        release.set()
        _stop(processes)
        results.close()


def test_warm_domain_acquisition_needs_neither_lock_nor_source(tmp_path, monkeypatch):
    from src.data import domain_datasets as domain
    from src.processing import cache

    destination = tmp_path / 'artifact'
    expected = b'previously downloaded artifact'
    destination.write_bytes(expected)

    def no_lock(*args):
        raise AssertionError('warm domain cache acquisition called flock')

    monkeypatch.setattr(cache.fcntl, 'flock', no_lock)
    missing_url = (tmp_path / 'absent-source').as_uri()
    assert domain._ensure_file(missing_url, destination).read_bytes() == expected
    assert domain._ensure_mag_file(
        missing_url, destination, len(expected),
        hashlib.md5(expected).hexdigest()).read_bytes() == expected
