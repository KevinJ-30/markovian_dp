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


_EXPECTED = ([0, 1, 1, 0], [True, True, False, False])


def _fixture(root, kind):
    folder = root / ('flickr' if kind == 'graphsaint' else 'raw')
    folder.mkdir(parents=True)
    adjacency = sp.csr_matrix(
        (np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(4, 4))
    sp.save_npz(folder / 'adj_full.npz', adjacency)
    if kind == 'graphsaint':
        train = adjacency.copy().tolil()
        train[2:, :] = 0
        sp.save_npz(folder / 'adj_train.npz', train.tocsr())
    np.save(folder / 'feats.npy', np.arange(8).reshape(4, 2).astype(float))
    (folder / 'class_map.json').write_text(json.dumps(dict(enumerate(_EXPECTED[0]))))
    (folder / 'role.json').write_text(json.dumps({'tr': [0, 1], 'va': [2], 'te': [3]}))
    return folder / '_labels_cache.pt' if kind == 'graphsaint' else root / 'processed/data.pt'


def _load(root, kind):
    if kind == 'graphsaint':
        _, data = datasets._load_graphsaint('flickr', root=root)
    else:
        import os
        os.environ['FLICKR_DATA_ROOT'] = str(root)
        _, data = datasets.load_dataset('flickr')
    return data.y.tolist(), data.train_mask.tolist()


def _dataset_worker(root, kind, mode, started, checkpoint, release, results):
    try:
        from src.processing import cache
        from torch_geometric.io import fs

        if mode == 'writer':
            original_save = torch.save if kind == 'graphsaint' else fs.torch_save

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
                    if kind == 'graphsaint':
                        return original_save(value, stream, *args, **kwargs)
                finally:
                    if own_stream:
                        stream.close()
                return original_save(value, destination, *args, **kwargs)

            if kind == 'graphsaint':
                torch.save = paused_save
            else:
                # Only pause the actual dataset payload, not PyG metadata files.
                def save(value, path):
                    if Path(path).name == 'data.pt':
                        return paused_save(value, path)
                    return original_save(value, path)
                fs.torch_save = save
        elif mode == 'warm':
            original_load = torch.load if kind == 'graphsaint' else fs.torch_load
            filename = '_labels_cache.pt' if kind == 'graphsaint' else 'data.pt'

            def concurrent_load(path, *args, **kwargs):
                if Path(path).name == filename:
                    checkpoint.wait(timeout=60)
                return original_load(path, *args, **kwargs)

            if kind == 'graphsaint':
                torch.load = concurrent_load

                def no_lock(*args):
                    raise AssertionError('warm atomic cache loading acquired flock')
                cache.fcntl.flock = no_lock
            else:
                fs.torch_load = concurrent_load
        started.set()
        results.put(('ok', _load(root, kind)))
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


@pytest.mark.parametrize('kind', ['graphsaint', 'flickr'])
def test_cold_dataset_readers_never_deserialize_partial_cache(tmp_path, kind):
    payload = _fixture(tmp_path, kind)
    context = multiprocessing.get_context('spawn')
    partial, release = context.Event(), context.Event()
    results = context.Queue()
    writer_started, reader_started = context.Event(), context.Event()
    writer = context.Process(target=_dataset_worker, args=(
        tmp_path, kind, 'writer', writer_started, partial, release, results))
    reader = context.Process(target=_dataset_worker, args=(
        tmp_path, kind, 'reader', reader_started, None, None, results))
    processes = [writer]
    writer.start()
    try:
        assert partial.wait(60), 'writer did not reach the partial-cache boundary'
        if kind == 'graphsaint':
            assert not payload.exists(), 'an incomplete label cache became visible'
        else:
            assert payload.read_bytes() == b'partial cache payload'
        reader.start()
        processes.append(reader)
        assert reader_started.wait(60)
        # While publication is blocked, a second loader must neither fail on
        # the partial payload nor publish its own competing cache.
        with pytest.raises(queue.Empty):
            results.get(timeout=0.5)
        release.set()
        observed = [results.get(timeout=60), results.get(timeout=60)]
        assert observed == [('ok', _EXPECTED), ('ok', _EXPECTED)]
        _finish(processes)
        assert _load(tmp_path, kind) == _EXPECTED
    finally:
        release.set()
        _stop(processes)
        results.close()


@pytest.mark.parametrize('kind', ['graphsaint', 'flickr'])
def test_warm_dataset_deserialization_overlaps_between_processes(tmp_path, kind):
    _fixture(tmp_path, kind)
    assert _load(tmp_path, kind) == _EXPECTED
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(2)
    results = context.Queue()
    started = [context.Event() for _ in range(2)]
    processes = [context.Process(target=_dataset_worker, args=(
        tmp_path, kind, 'warm', started[index], barrier, None, results))
        for index in range(2)]
    for process in processes:
        process.start()
    try:
        # Both real deserializers must enter simultaneously; an exclusive
        # warm-load lock would prevent the barrier from completing.
        observed = [results.get(timeout=90), results.get(timeout=90)]
        assert observed == [('ok', _EXPECTED), ('ok', _EXPECTED)]
        _finish(processes)
    finally:
        _stop(processes)
        results.close()


@pytest.mark.parametrize('failure', ['lock', 'temporary_file'])
def test_graphsaint_read_only_cache_directory_still_loads_labels(
        tmp_path, monkeypatch, failure):
    payload = _fixture(tmp_path, 'graphsaint')

    def denied(*args, **kwargs):
        raise PermissionError('read-only dataset directory')

    if failure == 'lock':
        monkeypatch.setattr(datasets, 'cache_creation_lock', denied)
    else:
        monkeypatch.setattr(datasets.tempfile, 'NamedTemporaryFile', denied)
    assert _load(tmp_path, 'graphsaint') == _EXPECTED
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
