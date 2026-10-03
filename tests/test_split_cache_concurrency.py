import fcntl
import multiprocessing
from pathlib import Path
import traceback
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from src.processing import splits


_ROLES = ("train", "val", "test")


def _graph():
    nodes = torch.arange(12)
    data = Data(
        x=torch.arange(36, dtype=torch.float32).reshape(12, 3),
        y=nodes % 2,
        edge_index=torch.stack((nodes, (nodes + 1) % 12)),
        num_nodes=12,
    )
    for index, role in enumerate(_ROLES):
        setattr(data, f"{role}_mask", (nodes >= 4 * index) & (nodes < 4 * (index + 1)))
    data.domain_id = (nodes >= 4).long()
    data.domain_names = ["source", "target"]
    data.domain_split = {
        "train": ["source"],
        "val": ["target"],
        "test": ["target"],
        "seed": 4,
        "val_ratio": 0.5,
    }
    data.domain_split_id = "concurrency"
    return data


def _cache_path(root, strategy):
    suffix = "-domain-concurrency" if strategy == "domain" else (
        "-native-seed0" if strategy == "native" else "-seed0"
    )
    return Path(root) / f"race{suffix}.pt"


def _worker(root, strategy, warm, barrier, saves, results):
    try:
        torch.set_num_threads(1)
        cache_path = _cache_path(root, strategy)
        original_exists = Path.exists
        original_save = torch.save
        original_load = torch.load
        original_induce = splits._induce
        first_lookup = True

        def coordinated_exists(path):
            nonlocal first_lookup
            exists = original_exists(path)
            if path == cache_path and first_lookup and not warm:
                first_lookup = False
                assert not exists
                # Both callers observe the same cache miss before either writes.
                barrier.wait(timeout=30)
            return exists

        def counted_save(payload, destination, *args, **kwargs):
            with saves.get_lock():
                saves.value += 1
            assert not original_exists(cache_path)
            original_save(payload, destination, *args, **kwargs)
            # The final path is invisible until the complete payload is published.
            assert not original_exists(cache_path)

        def coordinated_load(*args, **kwargs):
            if warm:
                barrier.wait(timeout=30)
            return original_load(*args, **kwargs)

        def coordinated_induce(*args, **kwargs):
            # Even a cold creator must release its lock before reconstruction.
            barrier.wait(timeout=30)
            return original_induce(*args, **kwargs)

        with (
            patch.object(Path, "exists", coordinated_exists),
            patch.object(torch, "save", counted_save),
            patch.object(torch, "load", coordinated_load),
            patch.object(splits, "_induce", coordinated_induce),
        ):
            split = splits.load_or_create_inductive_split(
                _graph(), "race", root=root, split_strategy=strategy
            )
        results.put({
            "parts": {
                role: {
                    "ids": getattr(split, role).node_ids.tolist(),
                    "x": getattr(split, role).data.x.tolist(),
                    "y": getattr(split, role).data.y.tolist(),
                    "edges": getattr(split, role).data.edge_index.t().tolist(),
                    "eval_mask": getattr(split, role).eval_mask.tolist(),
                }
                for role in _ROLES
            }
        })
    except BaseException:
        results.put({"error": traceback.format_exc()})


def _assert_partitions(parts, strategy):
    data = _graph()
    scored = []
    for role, part in parts.items():
        ids = part["ids"]
        local_ids = {node: index for index, node in enumerate(ids)}
        assert part["x"] == data.x[ids].tolist()
        assert part["y"] == data.y[ids].tolist()
        assert part["edges"] == [
            [local_ids[source], local_ids[target]]
            for source, target in data.edge_index.t().tolist()
            if source in local_ids and target in local_ids
        ]
        scored_ids = [node for node, evaluate in zip(ids, part["eval_mask"]) if evaluate]
        scored.extend(scored_ids)
        if strategy in {"native", "domain"}:
            expected = torch.where(getattr(data, f"{role}_mask"))[0].tolist()
            assert scored_ids == expected
            assert ids == (list(range(4, 12)) if strategy == "domain" and role != "train" else expected)
        else:
            assert all(part["eval_mask"])
            assert len(ids) == (8 if role == "train" else 2)
            assert part["y"].count(0) == part["y"].count(1)
    assert sorted(scored) == list(range(12))


@pytest.mark.parametrize("strategy", ["stratified", "native", "domain"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_concurrent_split_cache_creation_and_independent_reconstruction(tmp_path, strategy, warm):
    if warm:
        splits.load_or_create_inductive_split(
            _graph(), "race", root=tmp_path, split_strategy=strategy
        )
    cache_path = _cache_path(tmp_path, strategy)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    saves = context.Value("i", 0)
    results = context.Queue()
    workers = [
        context.Process(target=_worker, args=(tmp_path, strategy, warm, barrier, saves, results))
        for _ in range(2)
    ]
    # A warm cache must remain usable even while another process holds its lock.
    with cache_path.with_name(cache_path.name + ".lock").open("a+") as lock:
        if warm:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            for worker in workers:
                worker.start()
            outcomes = [results.get(timeout=90) for _ in workers]
            for worker in workers:
                worker.join(timeout=30)
                assert worker.exitcode == 0
            assert all("error" not in outcome for outcome in outcomes), outcomes
            assert outcomes[0]["parts"] == outcomes[1]["parts"]
            _assert_partitions(outcomes[0]["parts"], strategy)
            assert saves.value == (0 if warm else 1)
            payload = torch.load(cache_path, map_location="cpu")
            assert payload["num_nodes"] == 12
            for role, index in payload["indices"].items():
                part = outcomes[0]["parts"][role]
                assert index.tolist() == [
                    node for node, evaluate in zip(part["ids"], part["eval_mask"]) if evaluate
                ]
            assert not list(tmp_path.glob("*.tmp"))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                if worker.pid is not None:
                    worker.join(timeout=30)
            results.close()
            results.join_thread()
            if warm:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
