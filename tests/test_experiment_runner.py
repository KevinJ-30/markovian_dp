"""Configuration and scheduling contracts, with real supervised CPU children.

GPU telemetry is controlled; process launch, ownership, output publication,
resume, logs, and attempt cleanup use the production supervisor.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import run_experiments as runner
from scripts import runner_runtime as runtime


GPU = {"uuid": "GPU-runner-test", "name": "Test 16 GiB", "index": 0,
       "memory_total_mib": 16 * 1024}
DOMAIN_SPLIT = {"train": ["DE", "ENGB"], "val": ["ES"], "test": ["FR"]}


def _config(seeds=(0,), *, resources=None, **parameters):
    block = {"parameters": {"method": "mlp", **parameters}}
    if resources is not None:
        block["resources"] = resources
    return {"name": "runner-test", "defaults": {
        "dataset": "cora-ml", "lr": .01, "epochs": 1, "batch_size": 32,
        "bootstrap_resamples": 0,
    }, "grid": {"seed": list(seeds)}, "runs": [block]}


def _arguments(**overrides):
    return SimpleNamespace(device="cuda", max_jobs_per_gpu=None,
                           timeout_seconds=60.0, retry_failed=False, **overrides)


def _controller(tmp_path, config=None, *, cap=None):
    root = tmp_path / "experiment"
    root.mkdir()
    arguments = _arguments()
    arguments.max_jobs_per_gpu = cap
    controller = runner.ExperimentRunner(
        root, runner.expand_runs(config or _config(range(4))), arguments, [dict(GPU)])
    controller.lock_dir = tmp_path / "gpu-locks"
    return controller


def _snapshot(pids=(), *, utilization=None, gpu=GPU):
    processes = [{"pid": pid, "used_memory_mib": 3072, "gpu_uuid": gpu["uuid"]}
                 for pid in pids]
    used = 3072 * len(processes)
    return {**gpu, "observed_monotonic": time.monotonic(), "error": None,
            "memory_used_mib": used, "memory_free_mib": gpu["memory_total_mib"] - used,
            "utilization_gpu": (100 if processes else 0) if utilization is None else utilization,
            "compute_processes": processes, "idle": not processes}


def _activate(controller, count, *, estimates=True):
    for index, job in enumerate(controller.jobs[:count]):
        if estimates:
            job["resources"] = {"gpu_memory_mib": 3072, "host_memory_mib": 64}
        job.update(status="running", gpu_uuid=GPU["uuid"], gpu_model=GPU["name"])
        pid = 1000 + index
        controller.active[job["id"]] = {
            "job": job, "gpu": GPU, "ownership_known": True, "owned_pids": {pid},
            "sample": {"owned_gpu_memory_mib": 3072, "peak_gpu_memory_mib": 3072,
                       "tree_rss_bytes": 32 * runner.MIB,
                       "peak_rss_bytes": 64 * runner.MIB,
                       "observed_monotonic": time.monotonic()},
        }
    return _snapshot(range(1000, 1000 + count))


def _read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path):
    return json.loads(Path(path).read_text())


def test_block_overrides_replace_scalar_and_axis_without_mutating_input():
    config = _config()
    config["defaults"]["seed"] = 9
    config["defaults"].pop("lr")
    config["defaults"].pop("epochs")
    config["grid"] = {"lr": [.01, .001], "epochs": [1, 2]}
    config["runs"] = [
        {"parameters": {"method": "mlp", "lr": .1}, "grid": {"seed": [2, 3]}},
        {"parameters": {"method": "graphsage", "epochs": 3}, "grid": {"lr": [.02]}},
    ]
    before = copy.deepcopy(config)
    jobs = runner.expand_runs(config)
    actual = [(j["parameters"]["method"], j["parameters"]["lr"],
               j["parameters"]["epochs"], j["parameters"]["seed"]) for j in jobs]
    assert actual == [("mlp", .1, 1, 2), ("mlp", .1, 1, 3),
                      ("mlp", .1, 2, 2), ("mlp", .1, 2, 3),
                      ("graphsage", .02, 3, 9)]
    assert config == before
    assert [job["id"] for job in jobs] == [
        "0000_cora-ml_mlp_s2", "0001_cora-ml_mlp_s3",
        "0002_cora-ml_mlp_s2", "0003_cora-ml_mlp_s3", "0004_cora-ml_graphsage_s9"]


def test_documented_grid_expands_eight_jobs_without_conditional_axes():
    config = {
        "defaults": {"dataset": "saint-amazon", "batch_size": 1024,
                     "epochs": 20, "bootstrap_resamples": 0},
        "grid": {"seed": [0, 1], "lr": [.001]},
        "runs": [{"parameters": {"method": "sparse_sage"},
                  "grid": {"epsilon": [2, 8], "p2": [.1, .5]}}],
    }
    jobs = runner.expand_runs(config)
    assert [(j["parameters"]["seed"], j["parameters"]["epsilon"],
             j["parameters"]["p2"]) for j in jobs] == [
        (0, 2, .1), (0, 2, .5), (0, 8, .1), (0, 8, .5),
        (1, 2, .1), (1, 2, .5), (1, 8, .1), (1, 8, .5)]
    config["runs"].append({"parameters": {"method": "mlp"}})
    nonprivate = runner.expand_runs(config)[8:]
    assert [j["parameters"]["seed"] for j in nonprivate] == [0, 1]
    assert all(not ({"epsilon", "p2", "progap_depth"} & j["parameters"].keys())
               for j in nonprivate)


def test_domain_lists_are_literal_values_and_domain_objects_can_be_axes():
    config = _config((0, 1), dataset="twitch-explicit", domain_split=DOMAIN_SPLIT)
    jobs = runner.expand_runs(config)
    assert [j["parameters"]["seed"] for j in jobs] == [0, 1]
    assert all(j["parameters"]["domain_split"]["train"] == ["DE", "ENGB"] for j in jobs)
    second = {**DOMAIN_SPLIT, "test": ["PTBR"]}
    config["runs"][0]["parameters"].pop("domain_split")
    config["runs"][0]["grid"] = {"domain_split": [DOMAIN_SPLIT, second]}
    assert [(j["parameters"]["seed"], j["parameters"]["domain_split"]["test"])
            for j in runner.expand_runs(config)] == [(0, ["FR"]), (0, ["PTBR"]),
                                                     (1, ["FR"]), (1, ["PTBR"])]


@pytest.mark.parametrize("where", ["common", "block"])
def test_scalar_axis_ambiguity_is_rejected_in_its_own_scope(where):
    config = _config()
    if where == "common":
        config["defaults"]["seed"] = 0
    else:
        config["runs"][0]["parameters"]["seed"] = 0
        config["runs"][0]["grid"] = {"seed": [1]}
    with pytest.raises(ValueError, match="ambiguity.*seed"):
        runner.expand_runs(config)


@pytest.mark.parametrize("change", [
    {"resources": {"gpu_memory_mib": 1024}},
    {"parameters": {"method": "mlp", "dataset": "CORA-ML", "seed": 0}},
])
def test_duplicate_normalized_jobs_are_rejected_even_with_distinct_resources(change):
    config = _config()
    config["runs"].append({"parameters": {"method": "mlp"}, **change})
    with pytest.raises(ValueError, match=r"runs\[1\].*duplicate"):
        runner.expand_runs(config)


@pytest.mark.parametrize("patch,match", [
    ({"surprise": 1}, "unknown"),
    ({"runs": []}, "runs"),
    ({"runs": {}}, "runs"),
    ({"grid": {"seed": []}}, "seed"),
    ({"grid": {"seed": 1}}, "seed"),
    ({"defaults": []}, "parameters"),
    ({"runs": [{"parameters": {"method": "mlp"}, "gpu": 0}]}, "unknown"),
    ({"runs": [{"parameters": {"method": "mlp"}, "resources": {"epochs": 1}}]}, "resources"),
    ({"runs": [{"parameters": {"method": "mlp"}, "resources": {"gpu_memory_mib": 0}}]}, "gpu_memory_mib"),
    ({"runs": [{"parameters": {"method": "mlp"}, "resources": {"host_memory_mib": True}}]}, "host_memory_mib"),
    ({"runs": [{"parameters": {"method": "mlp"}, "resources": {"gpu_memory_mib": float("inf")}}]}, "gpu_memory_mib"),
])
def test_malformed_configuration_identifies_the_invalid_scope(patch, match):
    with pytest.raises(ValueError, match=match):
        runner.expand_runs({**_config(), **patch})


@pytest.mark.parametrize("parameter,value", [
    ("epochs", True), ("seed", False), ("batch_size", 1.5), ("lr", float("nan")),
    ("dropout", float("inf")), ("device", "cpu"), ("out_dir", "elsewhere"),
    ("prepared_protocol", "old"), ("campaign_request", "old"),
    ("progap_depth", 3), ("epsilon", 8), ("p2", .5),
])
def test_invalid_or_inapplicable_science_is_rejected_with_block_and_key(parameter, value):
    with pytest.raises(ValueError, match=rf"runs\[0\].*{parameter}"):
        runner.expand_runs(_config(**{parameter: value}))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_rejected_before_gpu_discovery_or_output(tmp_path, monkeypatch, constant):
    config = tmp_path / "input.json"
    config.write_text(json.dumps(_config()).replace('"lr": 0.01', f'"lr": {constant}'))
    output = tmp_path / "output"
    monkeypatch.setattr(runtime, "resolve_gpus", lambda _: pytest.fail("invalid JSON probed GPUs"))
    assert runner.main([str(config), "--out-dir", str(output)]) == 2
    assert not output.exists()


def test_dry_run_resolves_paths_without_discovery_or_files(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(method="progap", epsilon=8,
                                        progap_python="env/bin/python", split_root="relative-splits")))
    output = tmp_path / "not-created"
    monkeypatch.setattr(runtime, "resolve_gpus", lambda _: pytest.fail("dry-run probed GPUs"))
    monkeypatch.setattr(runner, "worker_command", lambda *a: pytest.fail("dry-run launched work"))
    assert runner.main([str(config), "--out-dir", str(output), "--gpus", "0,1", "--dry-run"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    parameters = resolved["jobs"][0]["parameters"]
    assert parameters["progap_python"] == str(tmp_path / "env/bin/python")
    assert parameters["split_root"] == str(runner.worker.REPO_ROOT / "relative-splits")
    assert resolved["out_dir"] == str(output)
    assert not output.exists()


@pytest.mark.parametrize("options", [
    ["--retry-failed"], ["--timeout-seconds", "nan"], ["--timeout-seconds", "0"],
    ["--max-jobs-per-gpu", "0"], ["--device", "cpu", "--gpus", "0"],
    ["--device", "cpu", "--max-jobs-per-gpu", "2"],
])
def test_invalid_operational_options_do_not_create_outputs(tmp_path, monkeypatch, options):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()))
    output = tmp_path / "output"
    monkeypatch.setattr(runtime, "resolve_gpus", lambda _: pytest.fail("invalid CLI probed GPUs"))
    assert runner.main([str(config), "--out-dir", str(output), *options]) == 2
    assert not output.exists()


def test_empty_gpu_authorization_is_an_error_not_cpu_fallback(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()))
    monkeypatch.setattr(runtime, "resolve_gpus", lambda _: [])
    output = tmp_path / "output"
    assert runner.main([str(config), "--out-dir", str(output)]) == 2
    assert not output.exists()


def test_packing_admits_three_not_four_at_full_owned_utilization(tmp_path):
    controller = _controller(tmp_path)
    candidate = controller.jobs[-1]
    candidate["resources"] = {"gpu_memory_mib": 3072}
    snapshot = _activate(controller, 2)
    assert snapshot["utilization_gpu"] == 100
    assert runner.gpu_reservation(3 * runner.GIB) == 4.25 * runner.GIB
    assert runner.gpu_headroom(16 * runner.GIB) == 2 * runner.GIB
    assert controller._gpu_fits(candidate, GPU, snapshot)
    snapshot = _activate(controller, 3)
    assert not controller._gpu_fits(candidate, GPU, snapshot)


@pytest.mark.parametrize("condition", ["foreign", "stale_snapshot", "query_error",
                                        "missing_process_memory", "missing_owned_memory",
                                        "stale_owned_sample", "unknown_ownership", "higher_peak",
                                        "fresh_peak_growth"])
def test_packing_denies_unsafe_observations(tmp_path, condition):
    controller = _controller(tmp_path)
    candidate = controller.jobs[-1]
    candidate["resources"] = {"gpu_memory_mib": 3072}
    snapshot = _activate(controller, 2)
    first = next(iter(controller.active.values()))
    if condition == "foreign":
        snapshot["compute_processes"].append({"pid": 999999, "used_memory_mib": 1})
    elif condition == "stale_snapshot":
        snapshot["observed_monotonic"] -= 1000
    elif condition == "query_error":
        snapshot["error"] = "query failed"
    elif condition == "missing_process_memory":
        snapshot["compute_processes"][0]["used_memory_mib"] = None
    elif condition == "missing_owned_memory":
        first["sample"]["owned_gpu_memory_mib"] = None
    elif condition == "stale_owned_sample":
        first["sample"]["observed_monotonic"] -= 1000
    elif condition == "unknown_ownership":
        first["ownership_known"] = False
    elif condition == "higher_peak":
        first["sample"]["peak_gpu_memory_mib"] = 6 * 1024
    elif condition == "fresh_peak_growth":
        snapshot["compute_processes"][0]["used_memory_mib"] = 10 * 1024
        snapshot["memory_free_mib"] = 3 * 1024
    assert not controller._gpu_fits(candidate, GPU, snapshot)


def test_user_cap_is_only_an_upper_bound(tmp_path):
    controller = _controller(tmp_path, cap=2)
    candidate = controller.jobs[-1]
    candidate["resources"] = {"gpu_memory_mib": 3072}
    assert not controller._gpu_fits(candidate, GPU, _activate(controller, 2))


def test_idle_gpu_observation_warmup_precedes_busy_gpu_packing(tmp_path, monkeypatch):
    controller = _controller(tmp_path, _config((0, 1), resources={"gpu_memory_mib": 3072}))
    busy = _activate(controller, 1)
    idle_gpu = {**GPU, "uuid": "GPU-runner-idle", "index": 1}
    controller.gpus[idle_gpu["uuid"]] = idle_gpu
    snapshots = {GPU["uuid"]: busy, idle_gpu["uuid"]: _snapshot(gpu=idle_gpu)}
    controller.sampler = SimpleNamespace(snapshot=snapshots.__getitem__)
    monkeypatch.setattr(controller, "_refresh_owned", lambda: None)
    monkeypatch.setattr(runtime, "host_available_bytes", lambda: 1 << 60)
    lease = controller._lease
    monkeypatch.setattr(controller, "_lease",
                        lambda gpu, snapshot: gpu["uuid"] == GPU["uuid"] or lease(gpu, snapshot))
    monkeypatch.setattr(controller, "_launch",
                        lambda *_: pytest.fail("packed a busy GPU before the idle GPU warmed up"))
    assert controller._gpu_fits(controller.jobs[1], GPU, busy)
    assert controller._admit() is False
    assert controller.jobs[1]["status"] == "pending"
    assert controller.idle_history[idle_gpu["uuid"]][1] == 1


def test_unknown_shape_stays_solo_even_if_other_gpu_learns_its_profile(tmp_path):
    controller = _controller(tmp_path)
    candidate = controller.jobs[-1]
    assert controller._gpu_fits(candidate, GPU, _snapshot())
    snapshot = _activate(controller, 1, estimates=False)
    active = next(iter(controller.active.values()))
    active["isolated"] = True
    assert not controller._gpu_fits(candidate, GPU, snapshot)
    completed_elsewhere = controller.jobs[1]
    completed_elsewhere.update(gpu_model=GPU["name"], exit={
        "owned_process_exited": True, "returncode": 0,
        "peak_rss_bytes": 64 * runner.MIB, "peak_gpu_memory_mib": 3072})
    controller._learn(completed_elsewhere, {"resources": {}})
    assert not controller._gpu_fits(candidate, GPU, snapshot)
    controller.active.clear()
    assert controller._gpu_fits(candidate, GPU, _snapshot())


def test_larger_explicit_estimate_retains_solo_safety_margins(tmp_path):
    controller = _controller(tmp_path)
    job = controller.jobs[0]
    controller.profiles[runner.profile_key(job["parameters"], GPU["name"])] = {
        "gpu_bytes": 14 * runner.GIB, "host_bytes": 64 * runner.MIB}
    job["resources"] = {"gpu_memory_mib": 15 * 1024}
    assert not controller._gpu_fits(job, GPU, _snapshot())


@pytest.mark.parametrize("record", [
    {"owned_process_exited": False, "returncode": 0, "peak_rss_bytes": 1024, "peak_gpu_memory_mib": 3072},
    {"owned_process_exited": True, "returncode": 1, "peak_rss_bytes": 1024, "peak_gpu_memory_mib": 3072},
    {"owned_process_exited": True, "returncode": 0, "peak_rss_bytes": None, "peak_gpu_memory_mib": 3072},
    {"owned_process_exited": True, "returncode": 0, "peak_rss_bytes": 1024, "peak_gpu_memory_mib": None},
    {"owned_process_exited": True, "returncode": 0, "peak_rss_bytes": 1024, "peak_gpu_memory_mib": 3072,
     "telemetry_error": "unavailable"},
])
def test_incomplete_peak_or_failed_attempt_never_authorizes_overlap(tmp_path, record):
    controller = _controller(tmp_path)
    first = controller.jobs[0]
    first.update(gpu_model=GPU["name"], exit=record)
    controller._learn(first, {"resources": {}})
    assert controller._peaks(controller.jobs[1], GPU)[0] is None


def test_progap_parent_only_allocator_peak_cannot_authorize_overlap(tmp_path):
    controller = _controller(tmp_path, _config((0, 1), method="progap", epsilon=8))
    first = controller.jobs[0]
    first.update(gpu_model=GPU["name"], exit={"owned_process_exited": True, "returncode": 0,
                                            "peak_rss_bytes": 64 * runner.MIB,
                                            "peak_gpu_memory_mib": None})
    controller._learn(first, {"resources": {"peak_cuda_allocated_bytes": runner.GIB}})
    assert controller._peaks(controller.jobs[1], GPU)[0] is None
    # The runtime's owned-tree peak includes the separately launched backend.
    first["exit"]["peak_gpu_memory_mib"] = 3072
    controller._learn(first, {"resources": {"peak_cuda_allocated_bytes": runner.GIB}})
    assert controller._peaks(controller.jobs[1], GPU)[0] == 3 * runner.GIB
    first["exit"]["peak_gpu_memory_mib"] = None
    controller._learn(first, {"resources": {
        "peak_cuda_allocated_bytes": runner.GIB,
        "peak_child_cuda_allocated_bytes": 3 * runner.GIB}})
    assert controller._peaks(controller.jobs[1], GPU)[0] == 4 * runner.GIB


@pytest.mark.parametrize("field,value", [
    ("batch_size", 64), ("mlp_hidden", 16), ("gnn_hidden", 16), ("p2", .75),
    ("sparse_radius", 2), ("sparse_degree_cap", 20), ("bootstrap_resamples", 100),
    ("split_root", "/another/split"), ("domain_split", {**DOMAIN_SPLIT, "test": ["PTBR"]}),
])
def test_shape_changes_never_reuse_resource_profiles(field, value):
    parameters = runner.expand_runs(_config(method="sparse_sage", epsilon=8, p2=.5,
                                             dataset="twitch-explicit", domain_split=DOMAIN_SPLIT))[0]["parameters"]
    changed = {**parameters, field: value}
    assert runner.profile_key(parameters, "A") != runner.profile_key(changed, "A")


def test_resource_profile_excludes_only_optimization_axes_and_includes_backend():
    parameters = runner.expand_runs(_config(method="progap", epsilon=8))[0]["parameters"]
    changed = {**parameters, "seed": 2, "lr": .1, "epsilon": 2, "epochs": 5}
    assert runner.profile_key(parameters, "A") == runner.profile_key(changed, "A")
    for field, value in (("progap_depth", 2), ("progap_python", "/other/python")):
        assert runner.profile_key(parameters, "A") != runner.profile_key({**parameters, field: value}, "A")
    assert runner.profile_key(parameters, "A") != runner.profile_key(parameters, "B")


def test_idle_lease_needs_distinct_samples_and_locked_recheck(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    first = _snapshot()
    monkeypatch.setattr(runtime, "gpu_snapshot", lambda _: _snapshot((9000,)))
    assert not controller._lease(GPU, first)
    assert not controller._lease(GPU, first)
    assert not controller._lease(GPU, _snapshot())  # busy under the acquired lock
    assert not controller.leases
    monkeypatch.setattr(runtime, "gpu_snapshot", lambda _: _snapshot())
    try:
        controller._lease(GPU, _snapshot())
        assert controller._lease(GPU, _snapshot())
        assert GPU["uuid"] in controller.leases
    finally:
        controller._release_unused()


def test_old_idle_sample_does_not_count_as_recent_pair(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    controller.idle_history[GPU["uuid"]] = (time.monotonic() - 1000, 1)
    monkeypatch.setattr(runtime, "gpu_snapshot", lambda _: _snapshot())
    try:
        assert not controller._lease(GPU, _snapshot())
    finally:
        controller._release_unused()


def test_closed_lease_cannot_authorize_admission(tmp_path, monkeypatch):
    controller = _controller(tmp_path)
    monkeypatch.setattr(runtime, "gpu_snapshot", lambda _: _snapshot())
    controller.idle_history[GPU["uuid"]] = (time.monotonic(), 1)
    try:
        assert controller._lease(GPU, _snapshot())
        controller.leases[GPU["uuid"]][1].close()
        assert not controller._lease(GPU, _snapshot((1000,)))
    finally:
        controller._release_unused()


_CHILD = r'''
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

cli = argparse.ArgumentParser()
cli.add_argument("--out-dir", type=Path, required=True)
cli.add_argument("--parameters", required=True)
cli.add_argument("--scenario", required=True)
cli.add_argument("--duration", type=float, required=True)
cli.add_argument("--cohort", required=True)
a = cli.parse_args()
p = json.loads(a.parameters)
a.out_dir.mkdir(exist_ok=False)
attempt = int(a.out_dir.parent.name)
root = a.out_dir.parents[4]
lifetime = {"pid": os.getpid(), "start": time.monotonic(), "end": None,
            "seed": p["seed"], "attempt": attempt,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
def publish():
    temporary = a.out_dir / "lifetime.partial"
    temporary.write_text(json.dumps(lifetime))
    temporary.replace(a.out_dir / "lifetime.json")
def other_lifetimes():
    for path in root.glob("runs/*/attempts/1/output/lifetime.json"):
        if path != a.out_dir / "lifetime.json":
            yield json.loads(path.read_text())
def wait_for(predicate):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise RuntimeError("test child synchronization timed out")
publish()
wait_for(lambda: (a.out_dir.parent / "memory-observed").exists())
print("child started", p["seed"], attempt, flush=True)
cohort = set(json.loads(a.cohort))
if p["seed"] in cohort:
    wait_for(lambda: cohort - {p["seed"]} <= {
        v["seed"] for v in other_lifetimes()})
if a.scenario.startswith("shared") and attempt == 1:
    if p["seed"] == 0:
        wait_for(lambda: any(v["end"] is not None for v in other_lifetimes()))
    else:
        wait_for(lambda: any(v["end"] is None for v in other_lifetimes()))
time.sleep(a.duration)
fail = ((a.scenario == "failure_once" and attempt == 1) or
        (a.scenario in {"shared_oom", "shared_oom_twice"} and p["seed"] == 1 and
         (attempt == 1 or a.scenario == "shared_oom_twice")) or
        a.scenario in {"cuda_oom", "host_oom"})
if fail:
    kind = "host_oom" if a.scenario == "host_oom" else "cuda_oom"
    if a.scenario != "failure_once":
        (a.out_dir / "worker_error.json").write_text(json.dumps({
            "kind": kind, "exception_class": "OutOfMemoryError", "message": kind}))
    lifetime["end"] = time.monotonic()
    publish()
    print("child failed", kind, flush=True)
    sys.exit(7 if a.scenario == "failure_once" else 86)
if a.scenario != "missing_result":
    result = {"status": "completed", "method": p["method"], "dataset": p["dataset"],
              "protocol": p["dataset"], "seed": p["seed"], "metric": "accuracy",
              "test_metric": .5 + .01 * p["seed"], "validation_metric": .4 + .01 * p["seed"]}
    (a.out_dir / "config.json").write_text(json.dumps(p))
    (a.out_dir / "result.json").write_text(json.dumps(result))
    with (a.out_dir / "result.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
lifetime["end"] = time.monotonic()
publish()
print("child completed", p["seed"], attempt, flush=True)
'''


class _ControlledGpu:
    """Report actual supervised PIDs as 3 GiB workloads on fake GPUs."""

    def __init__(self, root, gpus=(GPU,)):
        self.root = root
        self.gpus = {gpu["uuid"]: gpu for gpu in gpus}

    def start(self):
        return self

    def close(self):
        pass

    def snapshot(self, uuid):
        gpu = self.gpus[uuid]
        pids = []
        for path in self.root.glob("runs/*/attempts/*/launch.json"):
            launch = _read_json(path)
            if launch.get("gpu_uuid") != uuid or not launch.get("spawned") or "pid" not in launch:
                continue
            try:
                identity = runtime.proc_identity(launch["pid"])
            except (FileNotFoundError, ProcessLookupError):
                continue
            if identity["start_ticks"] == launch["start_ticks"] and identity["state"] not in {"Z", "X"}:
                pids.append(launch["pid"])
        return _snapshot(pids, gpu=gpu)


@pytest.fixture
def supervised_children(tmp_path, monkeypatch):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    settings = {"scenario": "success", "duration": .3, "cohort": []}

    def command(parameters, output, device):
        return [sys.executable, "-u", str(script), "--out-dir", str(output),
                "--parameters", json.dumps(parameters), "--scenario", settings["scenario"],
                "--duration", str(settings["duration"]), "--cohort", json.dumps(settings["cohort"])]

    supervise = runtime.run_process
    def observed_process(command, folder, environment, stop, **kwargs):
        callback = kwargs.get("on_sample")
        def observe(sample):
            if (sample.get("tree_rss_bytes") and
                    (kwargs.get("gpu") is None or sample.get("owned_gpu_memory_mib") is not None)):
                (Path(folder) / "memory-observed").touch()
            if callback is not None:
                callback(sample)
        return supervise(command, folder, environment, stop, **{**kwargs, "on_sample": observe})

    monkeypatch.setattr(runtime, "run_process", observed_process)

    monkeypatch.setattr(runner, "worker_command", command)
    monkeypatch.setattr(runner, "POLL_SECONDS", .05)
    monkeypatch.setattr(runtime, "host_available_bytes", lambda: 1 << 60)
    # All cooperative locks/quarantine are private to this test, never results/.
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    return settings


def _run_bounded(controller):
    expired = threading.Event()

    def stop_stalled_controller():
        expired.set()
        controller.stop.set()

    timer = threading.Timer(90, stop_stalled_controller)
    timer.start()
    try:
        result = controller.run()
    finally:
        timer.cancel()
        timer.join()
    assert not expired.is_set(), "scheduler did not drain its bounded test workload"
    return result


def _gpu_controller(tmp_path, monkeypatch, config, cap=None):
    controller = _controller(tmp_path, config, cap=cap)
    telemetry = _ControlledGpu(controller.root)
    controller.sampler = telemetry
    monkeypatch.setattr(runtime, "gpu_snapshot", telemetry.snapshot)
    for job in controller.jobs:
        controller.log("queued", job)
    controller.save()
    return controller


def _lifetimes(controller):
    return sorted((_read_json(path) for path in controller.root.glob(
        "runs/*/attempts/*/output/lifetime.json")), key=lambda row: (row["seed"], row["attempt"]))


def _maximum_overlap(lifetimes):
    events = [(row["start"], 1) for row in lifetimes] + [(row["end"], -1) for row in lifetimes]
    current = maximum = 0
    for _, change in sorted(events):
        current += change
        maximum = max(maximum, current)
    return maximum


@pytest.mark.parametrize("cap,expected", [(None, 3), (2, 2)])
def test_real_supervised_jobs_overlap_but_respect_headroom_and_user_cap(
        tmp_path, monkeypatch, supervised_children, cap, expected):
    supervised_children["cohort"] = list(range(expected))
    config = _config(range(4), resources={"gpu_memory_mib": 3072, "host_memory_mib": 64})
    controller = _gpu_controller(tmp_path, monkeypatch, config, cap)
    assert _run_bounded(controller) == 0
    lifetimes = _lifetimes(controller)
    assert _maximum_overlap(lifetimes) == expected
    assert all(row["cuda_visible_devices"] == GPU["uuid"] for row in lifetimes)
    rows = _read_csv(controller.root / "results.csv")
    assert [(row["status"], row["attempt"]) for row in rows] == [("completed", "1")] * 4
    assert {row["run_id"] for row in rows} == {job["id"] for job in controller.jobs}
    for job in controller.jobs:
        attempt = Path(job["output_dir"]).parent
        assert "child completed" in Path(job["log_path"]).read_text()
        assert _read_json(attempt / "exit.json")["owned_process_exited"] is True
        assert not (attempt / "tmp").exists()
    events = list(map(json.loads, (controller.root / "runner.log").read_text().splitlines()))
    assert [event["run_id"] for event in events if event["event"] == "queued"] == [j["id"] for j in controller.jobs]
    assert {event["run_id"] for event in events if event["event"] == "completed"} == {j["id"] for j in controller.jobs}
    assert all(event["gpu_uuid"] == GPU["uuid"] for event in events if event["event"] == "started")


def test_spreads_real_jobs_across_idle_gpus_before_packing(
        tmp_path, monkeypatch, supervised_children):
    supervised_children["cohort"] = list(range(8))
    config = _config(range(8), resources={"gpu_memory_mib": 3072, "host_memory_mib": 64})
    controller = _controller(tmp_path, config)
    gpus = [{**GPU, "uuid": f"GPU-runner-test-{index}", "index": index} for index in range(4)]
    controller.gpus = {gpu["uuid"]: gpu for gpu in gpus}
    controller.sampler = _ControlledGpu(controller.root, gpus)
    monkeypatch.setattr(runtime, "gpu_snapshot", controller.sampler.snapshot)
    # Keep real OS identity checks, but isolate scans from unrelated host jobs.
    def process_table():
        rows = {}
        for path in controller.root.glob("runs/*/attempts/*/launch.json"):
            launch = _read_json(path)
            if not launch.get("spawned"):
                continue
            try:
                row = runtime.proc_identity(launch["pid"])
            except (FileNotFoundError, ProcessLookupError):
                continue
            if row["state"] not in {"Z", "X"}:
                rows[row["pid"]] = row
        return rows
    monkeypatch.setattr(runtime, "_process_table", process_table)
    assert _run_bounded(controller) == 0
    events = list(map(json.loads, (controller.root / "runner.log").read_text().splitlines()))
    started = [event["gpu_uuid"] for event in events if event["event"] == "started"]
    assert set(started[:4]) == set(controller.gpus)
    lifetimes = _lifetimes(controller)
    assert _maximum_overlap(lifetimes) == 8
    for uuid in controller.gpus:
        assert _maximum_overlap([row for row in lifetimes if row["cuda_visible_devices"] == uuid]) == 2


def test_first_unknown_job_is_isolated_and_completed_profile_enables_overlap(
        tmp_path, monkeypatch, supervised_children):
    supervised_children["cohort"] = [1, 2, 3]
    controller = _gpu_controller(tmp_path, monkeypatch, _config(range(4)))
    assert _run_bounded(controller) == 0
    lifetimes = _lifetimes(controller)
    assert all(row["start"] >= lifetimes[0]["end"] for row in lifetimes[1:])
    assert _maximum_overlap(lifetimes[1:]) == 3
    assert controller.profiles[runner.profile_key(controller.jobs[0]["parameters"], GPU["name"])]["gpu_bytes"] == 3 * runner.GIB


@pytest.mark.parametrize("scenario,final_status", [("shared_oom", "completed"), ("shared_oom_twice", "failed")])
def test_shared_cuda_oom_has_exactly_one_exclusive_retry_and_keeps_both_attempts(
        tmp_path, monkeypatch, supervised_children, scenario, final_status):
    supervised_children.update(scenario=scenario, duration=.2)
    config = _config((0, 1), resources={"gpu_memory_mib": 3072, "host_memory_mib": 64})
    controller = _gpu_controller(tmp_path, monkeypatch, config)
    assert _run_bounded(controller) == (0 if final_status == "completed" else 1)
    first, retried = controller.jobs
    assert first["status"] == "completed"
    assert retried["status"] == final_status
    assert retried["attempt"] == 2 and retried["exclusive_retry_used"]
    assert runner.profile_key(retried["parameters"]) in controller.exclusive
    lifetimes = _lifetimes(controller)
    first_lifetime, failed_lifetime, retry_lifetime = lifetimes
    assert max(first_lifetime["start"], failed_lifetime["start"]) < min(first_lifetime["end"], failed_lifetime["end"])
    assert retry_lifetime["start"] >= max(first_lifetime["end"], failed_lifetime["end"])
    attempts = Path(retried["output_dir"]).parent.parent
    assert sorted(path.name for path in attempts.iterdir()) == ["1", "2"]
    assert _read_json(attempts / "1/exit.json")["error_attribution"]["kind"] == "cuda_oom"
    rows = _read_csv(controller.root / "results.csv")
    assert [(r["run_id"], r["attempt"], r["status"]) for r in rows] == [
        (first["id"], "1", "completed"), (retried["id"], "2", final_status)]
    events = list(map(json.loads, (controller.root / "runner.log").read_text().splitlines()))
    assert [(e["attempt"], e["event"]) for e in events if e.get("run_id") == retried["id"] and
            e["event"] in {"failed", "retry", "completed"}] == [
        (1, "failed"), (1, "retry"), (2, final_status)]


@pytest.mark.parametrize("scenario", ["cuda_oom", "host_oom", "failure_once", "missing_result"])
def test_isolated_failures_are_terminal_without_automatic_retry(
        tmp_path, monkeypatch, supervised_children, scenario):
    supervised_children.update(scenario=scenario, duration=.2)
    controller = _gpu_controller(tmp_path, monkeypatch, _config())
    assert _run_bounded(controller) == 1
    job = controller.jobs[0]
    assert job["status"] == "failed" and job["attempt"] == 1
    assert not job.get("exclusive_retry_used", False)
    assert _read_csv(controller.root / "results.csv")[0]["status"] == "failed"
    if scenario == "missing_result":
        assert "invalid completed output" in job["error"]


def test_successful_near_capacity_profile_runs_solo_instead_of_failing_preflight(
        tmp_path, monkeypatch, supervised_children):
    controller = _gpu_controller(tmp_path, monkeypatch, _config((0, 1)))
    controller.profiles[runner.profile_key(controller.jobs[0]["parameters"], GPU["name"])] = {
        "gpu_bytes": 14 * runner.GIB, "host_bytes": 64 * runner.MIB}
    assert _run_bounded(controller) == 0
    assert [(job["status"], job["attempt"]) for job in controller.jobs] == [
        ("completed", 1), ("completed", 1)]
    assert _maximum_overlap(_lifetimes(controller)) == 1


def test_oversize_known_job_fails_without_blocking_smaller_work(
        tmp_path, monkeypatch, supervised_children):
    config = _config(resources={"gpu_memory_mib": 16 * 1024})
    config["runs"].append({"parameters": {"method": "mlp", "seed": 1},
                           "resources": {"gpu_memory_mib": 3072, "host_memory_mib": 64}})
    controller = _gpu_controller(tmp_path, monkeypatch, config)
    assert _run_bounded(controller) == 1
    assert [(job["status"], job["attempt"]) for job in controller.jobs] == [("failed", 0), ("completed", 1)]
    assert controller.jobs[0]["error"] == "insufficient_gpu_memory"


def test_cli_resume_skips_completed_jobs_and_refuses_occupied_or_changed_roots(
        tmp_path, supervised_children):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config((0, 1))))
    root = tmp_path / "results"
    command = [str(config), "--device", "cpu", "--out-dir", str(root)]
    assert runner.main(command) == 0
    state_before = _read_json(root / "state.json")
    outputs = {path: (path.read_bytes(), path.stat().st_mtime_ns)
               for path in root.glob("runs/*/attempts/*/output/result.*")}
    assert runner.main(command) == 2
    assert runner.main([*command, "--resume"]) == 0
    assert [(j["status"], j["attempt"]) for j in _read_json(root / "state.json")["jobs"]] == [("completed", 1)] * 2
    assert outputs == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}
    assert _maximum_overlap([_read_json(path) for path in root.glob("runs/*/attempts/*/output/lifetime.json")]) == 1
    config.write_text(json.dumps(_config((0, 1), lr=.02)))
    state_bytes = (root / "state.json").read_bytes()
    assert runner.main([*command, "--resume"]) == 2
    assert (root / "state.json").read_bytes() == state_bytes
    assert state_before["root"] == str(root)


def test_failed_jobs_need_explicit_retry_and_retry_keeps_old_logs(tmp_path, supervised_children):
    supervised_children.update(scenario="failure_once", duration=.2)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()))
    root = tmp_path / "results"
    command = [str(config), "--device", "cpu", "--out-dir", str(root)]
    assert runner.main(command) == 1
    failed = _read_json(root / "state.json")["jobs"][0]
    log = Path(failed["log_path"])
    original = log.read_bytes()
    assert runner.main([*command, "--resume"]) == 1
    assert _read_json(root / "state.json")["jobs"][0]["attempt"] == 1
    assert runner.main([*command, "--resume", "--retry-failed"]) == 0
    completed = _read_json(root / "state.json")["jobs"][0]
    assert completed["attempt"] == 2 and completed["status"] == "completed"
    assert log.read_bytes() == original
    rows = _read_csv(root / "results.csv")
    assert [(row["status"], row["attempt"], row["returncode"]) for row in rows] == [("completed", "2", "0")]


def test_resume_rejects_changed_resources_and_relocated_root(tmp_path):
    controller = _controller(tmp_path)
    controller.save()
    controller.jobs[0]["resources"] = {"gpu_memory_mib": 3072}
    with pytest.raises(ValueError, match="configuration differs"):
        controller.restore()
    controller.jobs = runner.expand_runs(_config(range(4)))
    state = _read_json(controller.root / "state.json")
    state["root"] = str(tmp_path / "former-location")
    (controller.root / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="original absolute root"):
        controller.restore()


@pytest.mark.parametrize("observation", ["stale", "missing"])
def test_missing_or_stale_host_rss_prevents_additional_starts(tmp_path, monkeypatch, observation):
    controller = _controller(tmp_path)
    _activate(controller, 1)
    sample = next(iter(controller.active.values()))["sample"]
    if observation == "stale":
        sample["observed_monotonic"] -= 1000
    else:
        sample["tree_rss_bytes"] = None
    monkeypatch.setattr(runtime, "host_available_bytes", lambda: 1 << 60)
    assert not controller._host_fits(controller.jobs[1], GPU)


def test_unknown_host_observation_only_allows_an_isolated_start(tmp_path, monkeypatch):
    controller = _controller(tmp_path)

    def unavailable():
        raise RuntimeError("memory query unavailable")

    monkeypatch.setattr(runtime, "host_available_bytes", unavailable)
    assert controller._host_fits(controller.jobs[0], GPU)
    _activate(controller, 1)
    assert not controller._host_fits(controller.jobs[1], GPU)


def test_resume_recovers_reserved_attempt_without_overwriting_it(tmp_path, supervised_children):
    controller = _controller(tmp_path, _config())
    controller.args.device = "cpu"
    controller.gpus = {}
    controller.sampler = None
    controller.save()
    orphan = controller.root / "runs" / controller.jobs[0]["id"] / "attempts/1"
    orphan.mkdir(parents=True)
    controller.restore()
    assert controller.jobs[0]["status"] == "pending"
    assert controller.jobs[0]["attempt"] == 1
    recovered = _read_json(orphan / "exit.json")
    assert recovered["status"] == "interrupted"
    assert recovered["returncode"] is None and recovered["os_exit_observed"] is False
    original = (orphan / "process.log").read_bytes()
    assert _run_bounded(controller) == 0
    assert controller.jobs[0]["attempt"] == 2
    assert (orphan / "process.log").read_bytes() == original
    assert _read_csv(controller.root / "results.csv")[0]["attempt"] == "2"


def test_resume_blocks_orphan_with_unprovable_ownership(tmp_path):
    controller = _controller(tmp_path, _config())
    controller.save()
    orphan = controller.root / "runs" / controller.jobs[0]["id"] / "attempts/1"
    orphan.mkdir(parents=True)
    (orphan / "process.log").write_text("launch started but identity publication was interrupted\n")
    controller.restore()
    assert controller.jobs[0]["status"] == "blocked"
    assert controller.jobs[0]["error"] == "ownership_unverified"
    assert _read_json(orphan / "exit.json")["owned_process_exited"] is False


@pytest.mark.parametrize("damage", ["missing_result", "unobserved_exit"])
def test_resume_cannot_reuse_cached_success_after_output_or_exit_damage(
        tmp_path, supervised_children, damage):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()))
    root = tmp_path / "results"
    command = [str(config), "--device", "cpu", "--out-dir", str(root)]
    assert runner.main(command) == 0
    job = _read_json(root / "state.json")["jobs"][0]
    output = Path(job["output_dir"])
    if damage == "missing_result":
        (output / "result.csv").unlink()
    else:
        record = _read_json(output.parent / "exit.json")
        record.update(status="interrupted", returncode=None, os_exit_observed=False)
        (output.parent / "exit.json").write_text(json.dumps(record))
    assert runner.main([*command, "--resume"]) == 1
    restored = _read_json(root / "state.json")["jobs"][0]
    assert restored["status"] == "failed" and restored["attempt"] == 1
    assert "result_row" not in restored
    row = _read_csv(root / "results.csv")[0]
    assert row.get("test_metric", "") == "" and row["status"] == "failed"
