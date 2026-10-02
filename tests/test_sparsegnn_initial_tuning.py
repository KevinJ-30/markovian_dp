"""Scientific and process-lifetime contracts of the opportunistic campaign."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from scripts import full_matrix_records as records
from scripts import full_matrix_runtime as runtime
from scripts.full_matrix_run import _protocol


PROTOCOLS = (
    "ogbn-arxiv", "ogbn-products", "saint-reddit", "saint-yelp", "saint-amazon",
    "twitch-allbut2", "facebook100-allbut2", "mag-allbut2",
)
METHODS = (
    "mlp", "graphsage", "gin", "dp_mlp", "progap", "dpar",
    "dp_gnn_sage", "dp_gnn_gin", "sparse_sage", "sparse_gin",
)


def test_exact_grid_and_allbuttwo_domains():
    cells = records.enumerate_grid()
    assert len(cells) == 1584
    assert Counter(row["method"] for row in cells) == {
        method: 48 if method in {"mlp", "graphsage", "gin"} else
        480 if method.startswith("sparse_") else 96 for method in METHODS
    }
    keys = {(row["dataset"], row["method"], row["epsilon"], row["lr"], row["p2"], row["seed"])
            for row in cells}
    assert len(keys) == 1584
    assert Counter(row["dataset"] for row in cells) == {name: 198 for name in PROTOCOLS}
    for row in cells:
        assert row["epochs"] == 20 and row["batch_size"] == 1024 and row["seed"] in (0, 1, 2)
        assert row["degree_bound"] == 10 and row["bootstrap_resamples"] == 0
        assert row["lr"] in (0.01, 0.001)
        assert row["dropout"] == 0.5 and row["mlp_hidden"] == 64 and row["gnn_hidden"] == 128
        assert row["epsilon"] in ((None,) if row["method"] in {"mlp", "graphsage", "gin"} else (2, 8))
        assert row["p2"] in ((0.1, 0.25, 0.5, 0.75, 1.0) if row["method"].startswith("sparse_") else (None,))
    from src.data.domain_datasets import DOMAIN_REGISTRIES
    expected = {"twitch-allbut2": ("engb", "es", 5),
                "facebook100-allbut2": ("cornell5", "penn94", 16),
                "mag-allbut2": ("cn", "de", 4)}
    for protocol, (validation, test, count) in expected.items():
        dataset, roles = _protocol(protocol)
        assert roles["val"] == [validation] and roles["test"] == [test]
        assert roles["train"] == [d for d in DOMAIN_REGISTRIES[dataset]
                                  if d not in (validation, test)]
        assert len(roles["train"]) == count


@pytest.mark.parametrize("method,parameter,legacy", [
    ("graphsage", "max_fanout", 10), ("gin", "max_fanout", 10),
    ("dpar", "topk", 16), ("progap", "max_degree", 5),
    ("dp_gnn_sage", "max_degree", 5), ("dp_gnn_gin", "max_degree", 5),
    ("sparse_sage", "K_out", 10), ("sparse_gin", "K_out", 10),
])
def test_degree_and_bootstrap_registry_preserves_explicit_and_legacy_regimes(method, parameter, legacy):
    row = {"method": method, "epochs": 20, "batch_size": 1024, "seed": 2,
           "dropout": 0.5, "mlp_hidden": 64, "gnn_hidden": 128,
           "lr": 0.01, "epsilon": 8, "p2": 0.25}
    old = records._expected_parameters(row, 4096)
    assert old[parameter] == legacy and old["bootstrap_resamples"] == 1000
    for degree in (7, 10):
        current = records._expected_parameters({**row, "degree_bound": degree, "bootstrap_resamples": 0}, 4096)
        assert current[parameter] == degree and current["bootstrap_resamples"] == 0
        if method == "dpar":
            assert current["ppr_num"] == 70 and current["sampled_train_rate"] == 0.09
        elif method.startswith("dp_gnn_"):
            from src.privacy.dpgnn import max_terms_per_node
            assert current["max_terms"] == max_terms_per_node(degree, 1)
        elif method.startswith("sparse_"):
            assert current["K_in"] == 10 and current["cap_seed"] == 20002


def test_records_best_requires_complete_seed_cohort_and_disabled_ci_is_complete(tmp_path):
    accepted, results, coverage = {}, [], []
    for lr, scores in ((0.01, (0.4, 0.5, 0.6)), (0.001, (0.99,))):
        for seed, score in enumerate(scores):
            key = f"{lr}-{seed}"
            folder = tmp_path / key
            (folder / "output").mkdir(parents=True)
            path = folder / "output/result.csv"
            records._csv_write(path, [{
                "protocol": "fixture", "method": "mlp", "metric": "accuracy",
                "dp": False, "seed": seed, "lr": lr, "test_metric": score,
            }])
            accepted[key] = ({"attempt_dir": str(folder)}, {})
            results.append({
                "request_key": key, "protocol": "fixture", "method": "mlp", "epsilon": None,
                "bootstrap_resamples": 0, "accepted": True, "ci_available": False,
                "result_csv": str(path), "attempt_number": 1, "actual_epsilon": None,
                "actual_delta": None, "test_confidence_intervals": {},
            })
            coverage.append({"request_key": key, "accepted": True, "ci_available": False,
                             "ci_required": False, "status": "completed"})
    evidence = {"accepted": accepted, "results": results, "coverage": coverage,
                "errors": [], "attempts": []}
    summaries, _ = records._summarize_evidence(evidence)
    assert len(summaries) == 1
    assert summaries[0]["value"] == pytest.approx(0.5)
    assert summaries[0]["uncertainty"] == pytest.approx(0.1)
    assert summaries[0]["seeds"] == "0;1;2"
    comparisons = records._comparisons(results, evidence, summaries)
    assert comparisons[0]["selected_request_keys"] == ["0.01-0", "0.01-1", "0.01-2"]
    assert comparisons[0]["test_metric"] == pytest.approx(0.5)
    status = records._coverage_summary(
        {"purpose": "campaign", "expected_requests": 4, "expected_comparisons": 1,
         "interpretation": records.INTERPRETATION},
        evidence, comparisons)
    assert status["fully_executed"] and status["completed_count"] == 4
    assert status["ci_complete_count"] == 0 and status["missing_request_keys"] == []
    for key in ("0.01-1", "0.01-2"):
        accepted.pop(key)
    summaries, _ = records._summarize_evidence(evidence)
    assert summaries == []


def _command(tmp_path, text):
    script = tmp_path / "child.py"
    script.write_text(text)
    folder = tmp_path / "attempt"
    folder.mkdir()
    return [sys.executable, str(script), "--out-dir", str(folder / "output")], folder


def _assert_gone(pid):
    try:
        assert runtime.proc_identity(pid)["state"] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        pass


def test_real_deadline_kills_sigterm_resistant_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    command, folder = _command(tmp_path,
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        policy={**runtime.DEFAULT_POLICY, "hard_seconds": 0.6, "termination_grace_seconds": 0.2})
    assert outcome["status"] == "timeout" and outcome["owned_process_exited"]
    assert not (folder / "output" / "result.csv").exists()
    _assert_gone(int(pid_file.read_text()))
    assert runtime.read_json(folder / "exit.json")["status"] == "timeout"


def test_oom_is_retryable_but_unrelated_failure_is_not(tmp_path):
    oom = tmp_path / "oom"
    oom.mkdir()
    command, folder = _command(oom,
        "import json,sys\nfrom pathlib import Path\n"
        "out=Path(sys.argv[sys.argv.index('--out-dir')+1]);out.mkdir()\n"
        "(out/'worker_error.json').write_text(json.dumps({'kind':'cuda_oom','message':'fixture'}))\n"
        "sys.exit(86)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event())
    assert outcome["status"] == "oom" and outcome["owned_process_exited"]
    failed = tmp_path / "failed"
    failed.mkdir()
    command, folder = _command(failed, "import sys\nsys.exit(7)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event())
    assert outcome["status"] == "failed" and outcome["returncode"] == 7


def test_foreign_workload_does_not_cancel_our_running_child(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    class BusySampler:
        def snapshot(self, uuid):
            return {"uuid": uuid, "idle": False, "memory_used_mib": 40000,
                    "utilization_gpu": 100, "compute_processes": [
                        {"pid": 987654321, "used_memory_mib": 39000, "gpu_uuid": uuid}]}
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        gpu={"uuid": "GPU-fixture"}, sampler=BusySampler(),
        policy={**runtime.DEFAULT_POLICY, "hard_seconds": 5})
    assert outcome["status"] == "completed" and outcome["returncode"] == 0


def test_ownership_mismatch_refuses_signalling_live_process(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(60)\n")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        identity = runtime.proc_identity(process.pid)
        launch = {**identity, "start_ticks": identity["start_ticks"] + 1,
                  "command": command, "observed_command": identity["command"],
                  "out_dir": str(folder / "output"), "folder": str(folder),
                  "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        with pytest.raises(runtime.OwnershipError):
            runtime.validate_owned_process(launch)
        assert process.poll() is None
    finally:
        # This Popen handle is the independently owned test child, not the tampered record.
        process.terminate()
        process.wait(timeout=5)


def test_cooperative_lock_cannot_be_claimed_twice(tmp_path):
    path = tmp_path / "gpu.lock"
    with runtime.file_lock(path, nonblocking=True):
        with pytest.raises((BlockingIOError, OSError, RuntimeError)):
            with runtime.file_lock(path, nonblocking=True):
                pytest.fail("duplicate GPU lease was granted")


def test_empty_visibility_never_expands_authorized_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runtime, "gpu_inventory", lambda: {
        "GPU-free": {"uuid": "GPU-free", "index": 7, "idle": True}})
    assert runtime.resolve_gpus("auto") == []


def test_worker_occupied_output_is_untouched(tmp_path):
    from scripts.full_matrix_run import main
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text('{"evidence":"original"}')
    args = ["--dataset", "fixture", "--method", "mlp", "--lr", ".001",
            "--batch-size", "32", "--epochs", "1", "--device", "cpu",
            "--out-dir", str(output)]
    assert main(args) == 1
    assert (output / "result.json").read_text() == '{"evidence":"original"}'
    assert not (output / "worker_error.json").exists()
    assert not (output / "worker_exit.json").exists()


def test_worker_memory_error_publishes_retryable_failure_not_success(tmp_path, monkeypatch):
    from scripts import full_matrix_run as worker
    monkeypatch.chdir(worker.REPO_ROOT)
    output = tmp_path / "output"
    def exhausted(args):
        raise MemoryError("fixture allocation failure")
    monkeypatch.setattr(worker, "run", exhausted)
    args = ["--dataset", "fixture", "--method", "mlp", "--lr", ".001",
            "--batch-size", "32", "--epochs", "1", "--device", "cpu",
            "--out-dir", str(output)]
    assert worker.main(args) == 86
    assert runtime.read_json(output / "worker_error.json")["kind"] == "host_oom"
    assert not (output / "worker_exit.json").exists()


def test_exited_leader_does_not_leave_owned_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    command, folder = _command(tmp_path,
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(1.3)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        policy={**runtime.DEFAULT_POLICY, "hard_seconds": 5, "termination_grace_seconds": 0.2})
    assert outcome["owned_process_exited"]
    _assert_gone(int(pid_file.read_text()))


def test_oom_bursts_defer_without_losing_other_work_or_retry_eligibility(tmp_path):
    from scripts.full_matrix_campaign import CampaignQueue, _initial_state
    class Clock:
        value = 1000.0
        def time(self):
            return self.value
        def monotonic(self):
            return self.value
    clock = Clock()
    queue = CampaignQueue(tmp_path, device="cpu", purpose="smoke", clock=clock)
    queue.state = {"oom": _initial_state(), "other": _initial_state()}
    queue.order = ["oom", "other"]
    outcome = {"status": "oom", "gpu_uuid": "GPU-first", "finished_epoch": clock.value}
    for attempt in (1, 2):
        queue._transition(queue.state["oom"], outcome, attempt)
        assert queue._ready_key(None) == "oom"
    queue._transition(queue.state["oom"], outcome, 3)
    assert queue._ready_key(None) == "other"
    queue._persist_locked()
    recovered = CampaignQueue(tmp_path, device="cpu", purpose="smoke", clock=clock)
    recovered.state = runtime.read_json(tmp_path / "queue_state.json")
    recovered.order = queue.order
    clock.value = 1299
    assert recovered._ready_key(None) == "other"
    clock.value = 1300
    assert recovered._ready_key(None) == "oom"
    recovered._transition(recovered.state["oom"], {**outcome, "finished_epoch": clock.value}, 4)
    assert recovered._ready_key(None) == "oom"
    recovered._transition(recovered.state["oom"], {"status": "timeout"}, 5)
    assert recovered._ready_key(None) == "other"


def test_new_idle_gpu_requires_two_recent_successful_observations(tmp_path):
    from scripts.full_matrix_campaign import CampaignQueue
    class Sampler:
        row = None
        def snapshot(self, uuid):
            return self.row
    queue = CampaignQueue(tmp_path)
    queue.sampler = Sampler()
    assert not queue._stable_idle("GPU-later")
    queue.sampler.row = {"idle": True, "consecutive_idle": 1,
                         "observed_monotonic": time.monotonic()}
    assert not queue._stable_idle("GPU-later")
    queue.sampler.row["consecutive_idle"] = 2
    assert queue._stable_idle("GPU-later")
    queue.sampler.row["error"] = "NVIDIA query failed"
    assert not queue._stable_idle("GPU-later")
    del queue.sampler.row["error"]
    queue.sampler.row["observed_monotonic"] -= 60
    assert not queue._stable_idle("GPU-later")


def test_recovery_terminates_owned_worker_after_controller_crash(tmp_path):
    worker_script = tmp_path / "sleeper.py"
    worker_script.write_text("import time\ntime.sleep(60)\n")
    folder = tmp_path / "attempt"
    folder.mkdir()
    supervisor_script = tmp_path / "supervisor.py"
    supervisor_script.write_text(
        "import os,sys,threading\nfrom pathlib import Path\n"
        "from scripts.full_matrix_runtime import run_process\n"
        f"folder=Path({str(folder)!r})\n"
        f"run_process([sys.executable,{str(worker_script)!r},'--out-dir',str(folder/'output')],"
        "folder,dict(os.environ),threading.Event())\n")
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    supervisor = subprocess.Popen([sys.executable, str(supervisor_script)],
                                  env=environment, start_new_session=True)
    launch = None
    try:
        until = time.monotonic() + 10
        while not (folder / "launch.json").exists() and time.monotonic() < until:
            time.sleep(0.02)
        launch = runtime.read_json(folder / "launch.json")
        supervisor.kill()
        supervisor.wait(timeout=5)
        recovered = runtime.recover_process(folder, grace=0.2)
        assert recovered["status"] == "interrupted"
        assert recovered["returncode"] is None and recovered["owned_process_exited"]
        _assert_gone(launch["pid"])
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=5)
        if launch is not None:
            runtime.recover_process(folder, grace=0.2)


def test_prepared_roundtrip_preserves_graph_predictions_and_rejects_tampering(tmp_path):
    import torch
    from torch_geometric.nn import SAGEConv
    prepared = records._smoke_prepared(tmp_path, "cuda")["smoke-accuracy"]
    _, split, _, _ = records.load_prepared_protocol(prepared["manifest"])
    expected_x = torch.arange(160 * 8, dtype=torch.float32).reshape(160, 8)[:128] / (160 * 8)
    nodes = torch.arange(128)
    following = (nodes + 1) % 128
    expected_edges = torch.stack([torch.cat([nodes, following]), torch.cat([following, nodes])])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = SAGEConv(8, 2).eval()
        with torch.no_grad():
            expected = model(expected_x, expected_edges)
            restored = model(split.train.data.x, split.train.data.edge_index)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    assert torch.equal(split.test.node_ids, torch.arange(144, 160))
    assert split.test.eval_mask.all()
    payload = Path(prepared["manifest"]).parent / "train.pt"
    with payload.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash|SHA|sha|artifact"):
        records.load_prepared_protocol(prepared["manifest"])


def test_transient_empty_exec_argv_does_not_reject_our_worker(tmp_path, monkeypatch):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    observe = runtime.proc_identity
    initial = True
    def exec_transition(pid):
        nonlocal initial
        row = observe(pid)
        if initial:
            initial = False
            return {**row, "command": []}
        return row
    monkeypatch.setattr(runtime, "proc_identity", exec_transition)
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                  policy={**runtime.DEFAULT_POLICY, "hard_seconds": 5})
    assert outcome["status"] == "completed" and outcome["owned_process_exited"]


def test_direct_queue_requires_distinct_eligible_samples():
    from scripts import full_matrix_queue as queue
    history = {}
    snapshot = {"utilization_gpu": 29.9, "memory_free_mib": 1000,
                "memory_used_mib": 40000, "compute_processes": [{"pid": 987654321}],
                "idle": False, "observed_monotonic": 1.0}
    assert not queue._observe_eligibility(history, "gpu", snapshot)
    assert not queue._observe_eligibility(history, "gpu", snapshot)
    assert queue._observe_eligibility(history, "gpu", {**snapshot, "observed_monotonic": 2.0})
    for bad in (None, {**snapshot, "error": "stale GPU observation"},
                {**snapshot, "error": "query failed"}, {**snapshot, "utilization_gpu": 30},
                {**snapshot, "memory_free_mib": 0}, {**snapshot, "utilization_gpu": float("nan")}):
        assert not queue._observe_eligibility(history, "gpu", bad)
        assert not queue._observe_eligibility(history, "gpu", {**snapshot, "observed_monotonic": 3.0})
        history.clear()


def test_real_child_without_hard_deadline(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                  policy={"hard_seconds": None})
    assert outcome["status"] == "completed" and outcome["returncode"] == 0
    assert outcome["owned_process_exited"]


def _queue_output(tmp_path, row, metric, validation, test):
    output = tmp_path / row["protocol"] / "output"
    output.mkdir(parents=True)
    parameters = {"steps": 40, "evaluate_every": 2, "weight_decay": 0.0005,
                  "bootstrap_resamples": row["bootstrap_resamples"], "knob": "a|b\nc"}
    selection = {"metric": metric, "split": "validation", "early_stopping": False,
                 "epochs_requested": 20, "epochs_completed": 20, "validation_score": validation,
                 "evaluate_every": 2, "step": 6, "epoch": 3, "completed_updates": 40}
    interval = {**records.BOOTSTRAP, "n_observations": 10,
                "metrics": {metric: {"lower": test - 0.123, "upper": test + 0.045,
                                    "valid_resamples": 1000}}}
    if row["bootstrap_resamples"] == 0:
        interval = {}
    identity = {key: row[key] for key in ("protocol", "method", "lr", "epochs", "seed", "dropout")}
    identity.update(target_epsilon=row["epsilon"], requested_batch_size=row["batch_size"],
                    hidden=64, effective_batch_size=1024, metric=metric, parameters=parameters)
    result = {**identity, "validation_metric": validation, "test_metric": test,
              "selection": selection, "completed_epochs": 20, "test_confidence_intervals": interval,
              "native_result": {"selection": selection, "completed_updates": 40}}
    runtime.atomic_json(output / "config.json", {**identity, "task": {"primary_metric": metric}})
    runtime.atomic_json(output / "result.json", result)
    (output / "result.csv").write_text(f"metric,test_metric\n{metric},{test}\n")
    _commit_queue_output(output)
    return {"status": "completed", "attempt": 1, "output": str(output)}


def _commit_queue_output(output):
    runtime.atomic_json(output / "worker_exit.json", {
        "status": "completed", "artifact_sha256": {
            name: runtime.sha256(output / name) for name in ("config.json", "result.json", "result.csv")}})


def test_direct_queue_tables_preserve_every_result_and_raw_evidence(tmp_path):
    import csv
    from scripts import full_matrix_queue as queue
    grid = records.enumerate_grid()
    rows = [{**next(row for row in grid if row["protocol"] == protocol and row["method"] == "mlp"),
             "bootstrap_resamples": 1000}
            for protocol in ("ogbn-arxiv", "saint-yelp", "saint-amazon")]
    states = [_queue_output(tmp_path, rows[0], "accuracy", 0.9, 0.4),
              _queue_output(tmp_path, rows[1], "micro_f1", 0.6, 0.8),
              {"status": "pending", "attempt": 0, "reason": "busy|GPU\nwait"}]
    paths = [Path(state["output"]) / "result.csv" for state in states[:2]]
    hashes = [runtime.sha256(path) for path in paths]
    for _ in range(2):
        queue.tables(tmp_path, rows, states)
        with (tmp_path / "summary.csv").open() as stream:
            exported = list(csv.DictReader(stream))
        assert [row["protocol"] for row in exported] == [row["protocol"] for row in rows]
        for index, metric in enumerate(("accuracy", "micro_f1")):
            actual = runtime.read_json(Path(states[index]["output"]) / "result.json")
            own = exported[index]
            assert own["metric"] == metric
            assert float(own["validation_metric"]) == actual["validation_metric"]
            assert float(own["test_metric"]) == actual["test_metric"]
            assert float(own["ci_lower"]) == actual["test_confidence_intervals"]["metrics"][metric]["lower"]
            assert float(own["ci_upper"]) == actual["test_confidence_intervals"]["metrics"][metric]["upper"]
            assert json.loads(own["parameters"]) == actual["parameters"]
            assert own["selected_epoch"] == "3" and own["selected_step"] == "6"
            assert own["result_csv"] == str(paths[index])
        assert exported[2]["status"] == "pending" and exported[2]["test_metric"] == ""
        assert exported[2]["batch_size"] == "1024" and exported[2]["hidden"] == "64"
        assert [runtime.sha256(path) for path in paths] == hashes
        markdown = (tmp_path / "summary.md").read_text()
        assert "busy&#124;GPU<br>wait" in markdown
    output = Path(states[0]["output"])
    result = runtime.read_json(output / "result.json")
    result["test_confidence_intervals"]["metrics"].clear()
    runtime.atomic_json(output / "result.json", result)
    _commit_queue_output(output)
    queue.tables(tmp_path, rows, states)
    assert states[0]["status"] == "ci_unavailable"
    assert states[1]["status"] == "completed"
    (Path(states[1]["output"]) / "result.csv").write_text("tampered\n")
    queue.tables(tmp_path, rows, states)
    assert states[1]["status"] == "invalid"


def test_direct_queue_accepts_disabled_bootstrap_without_inventing_intervals(tmp_path):
    import csv
    from scripts import full_matrix_queue as queue
    row = records.enumerate_grid()[0]
    state = _queue_output(tmp_path, row, "accuracy", 0.7, 0.6)
    queue.tables(tmp_path, [row], [state])
    assert state["status"] == "completed"
    with (tmp_path / "summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert float(summary["test_metric"]) == 0.6
    assert summary["ci_lower"] == "" and summary["ci_upper"] == ""
    with (tmp_path / "results.csv").open() as stream:
        result = next(csv.DictReader(stream))
    assert json.loads(result["test_confidence_intervals"]) == {}
    queue.tables(tmp_path, [{**row, "bootstrap_resamples": 1000}], [state])
    assert state["status"] != "completed"


def test_direct_queue_busy_locked_probe_never_launches(tmp_path, monkeypatch):
    from scripts import full_matrix_queue as queue
    row = records.enumerate_grid()[0]
    monkeypatch.setattr(queue.records, "enumerate_grid", lambda **_: [row])
    monkeypatch.setattr(queue.records, "PROTOCOLS", (row["protocol"],))
    monkeypatch.setattr(sys, "argv", ["queue", "--out-root", str(tmp_path)])
    monkeypatch.setattr(queue.runtime, "resolve_gpus", lambda _: [{"uuid": "GPU-fixture"}])
    monkeypatch.setattr(queue.runtime, "host_available_bytes", lambda: 100 * runtime.GIB)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100 * runtime.GIB})())
    callbacks = []
    monkeypatch.setattr(queue.signal, "signal", lambda _, callback: callbacks.append(callback))
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    class Sampler:
        count = 0
        def __init__(self, _):
            pass
        def start(self):
            return self
        def close(self):
            pass
        def snapshot(self, _):
            self.count += 1
            return {"observed_monotonic": self.count, "utilization_gpu": 29.9, "memory_free_mib": 1000}
    monkeypatch.setattr(queue.runtime, "GpuSampler", Sampler)
    def busy_probe(_):
        callbacks[0]()
        return {"utilization_gpu": 30, "memory_free_mib": 1000}
    monkeypatch.setattr(queue.runtime, "gpu_snapshot", busy_probe)
    monkeypatch.setattr(queue.runtime, "run_process", lambda *a, **kw: pytest.fail("busy GPU launched"))
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert queue.main() == 1
    assert runtime.read_json(tmp_path / "queue_state.json")[0]["attempt"] == 0


def test_exiting_empty_argv_is_not_a_live_ownership_mismatch(tmp_path, monkeypatch):
    pid = 987654321
    command = ["python", "--out-dir", str(tmp_path / "output")]
    leader = {"pid": pid, "pgid": pid, "session_id": pid, "start_ticks": 123,
              "command": command, "state": "R"}
    launch = {**leader, "folder": str(tmp_path), "observed_command": command,
              "boot_id": runtime._boot_id()}
    empty = {**leader, "command": []}
    monkeypatch.setattr(runtime, "_process_table", lambda: {pid: empty})
    observations = iter([empty, empty, {**empty, "state": "Z"}, {**empty, "state": "Z"}])
    monkeypatch.setattr(runtime, "proc_identity", lambda _: next(observations))
    assert runtime._owned_snapshot(launch, {pid: leader}) == {}
    monkeypatch.setattr(runtime, "proc_identity", lambda _: {**leader, "command": ["foreign"]})
    with pytest.raises(runtime.OwnershipError, match="argv changed"):
        runtime._owned_snapshot(launch, {pid: leader})


@pytest.mark.parametrize("architecture", ["GraphSAGE", "GIN"])
def test_bounded_full_neighbor_inference_preserves_logits(architecture, monkeypatch):
    import torch
    from src.models import baselines
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = getattr(baselines, architecture)(4, 3, hidden=7, layers=2, dropout=0.5).eval()
        features = torch.randn(9, 4)
        edges = torch.tensor([[1, 2, 2, 3, 4, 5, 5, 6, 7, 0],
                              [0, 0, 0, 1, 1, 2, 2, 2, 3, 0]])
        expected = model(features, edges).detach()
        monkeypatch.setattr(baselines, "_MESSAGE_BYTES", 32)
        with torch.no_grad():
            actual = model(features, edges)
            isolated = model(features, torch.empty((2, 0), dtype=torch.long))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[8], isolated[8], rtol=0, atol=0)


def test_progap_preparation_preserves_topology_with_auxiliary_indices():
    if not Path(records.PROGAP_PYTHON).is_file():
        pytest.skip("requires the separately pinned ProGAP environment")
    code = """
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
import inductive_adapter
edges = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 1, 2]])
data = Data(x=torch.arange(9).reshape(3, 3).float(), y=torch.tensor([0, 1, 0]),
            edge_index=edges, train_edge_index=edges.clone(),
            edge_weight=torch.tensor([1., 2., 3., 4., 5.]),
            eval_mask=torch.tensor([True, False, True]))
reference = ToSparseTensor(layout=torch.sparse_csr)(
    Data(edge_index=edges, edge_weight=data.edge_weight, num_nodes=3))
actual = inductive_adapter._prepare(data)
torch.testing.assert_close(actual.adj_t.to_dense(), reference.adj_t.to_dense())
torch.testing.assert_close(actual.x, data.x)
assert torch.equal(actual.eval_mask, data.eval_mask)
assert torch.equal(data.edge_index, edges)
"""
    subprocess.run([records.PROGAP_PYTHON, "-c", code], check=True, timeout=60,
                   cwd=Path(__file__).resolve().parents[1] / "third_party/ProGAP")


def test_normal_exit_allows_inflight_telemetry_to_settle(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                  policy={"hard_seconds": None},
                                  on_sample=lambda _: time.sleep(2.5))
    assert outcome["returncode"] == 0
    assert outcome["status"] == "completed" and outcome["owned_process_exited"]


def _sharing_queue(tmp_path, monkeypatch, *, max_jobs=None):
    from scripts.full_matrix_campaign import CampaignQueue, _initial_state
    options = {} if max_jobs is None else {"max_jobs_per_gpu": max_jobs}
    queue = CampaignQueue(tmp_path, purpose="ideation", **options,
                          host_available=lambda: 100 * 1024 ** 3,
                          disk_available=lambda _: 100 * 1024 ** 3,
                          policy={"poll_seconds": 0.05, "termination_grace_seconds": 0.1})
    queue._lease_root = tmp_path / "leases"
    queue.manifest = {"requests_sha256": "requests", "source_fingerprint": "source",
                      "prepared": {"physics": {"tensor_bytes": 1, "preflight_rss_bytes": 1}}}
    queue.requests = [
        {"request_key": key, "ordinal": index, "protocol": "physics", "method": "sparse_gin",
         "batch_size": 256, "p2": 0.5, "lr": lr, "epsilon": eps, "seed": 0,
         "prepared_fingerprint": "prepared", "prepared_manifest_sha256": "manifest"}
        for index, (key, lr, eps) in enumerate((("first", .01, 2), ("second", .001, 8),
                                                ("third", .001, 2)))]
    queue.by_key = {row["request_key"]: row for row in queue.requests}
    queue.order = list(queue.by_key)
    queue.state = {key: _initial_state() for key in queue.order}
    monkeypatch.setattr(records, "worker_command", lambda request, folder, **kwargs: [
        sys.executable, "-c", "import time; time.sleep(60)", "--out-dir", str(folder / "output")])
    return queue


def _sharing_gpu():
    return {"uuid": "GPU-test", "idle": False, "utilization_gpu": 50,
            "memory_total_mib": 24000, "memory_free_mib": 22000,
            "compute_processes": [{"pid": 123, "used_memory_mib": 2000}]}


def test_measured_shape_admits_overlap_without_idle_and_skips_unknown(tmp_path, monkeypatch):
    queue = _sharing_queue(tmp_path, monkeypatch, max_jobs=2)
    first = queue._claim("GPU-test", {"uuid": "GPU-test", "idle": True})
    queue.by_key["second"]["p2"] = 1.0
    result = first["folder"] / "output"
    result.mkdir()
    runtime.atomic_json(result / "result.json", {"resources": {"peak_cuda_allocated_bytes": 3 * 1024 ** 3}})
    queue._remember_peak(first["request"], first["folder"], {
        "accepted": True, "owned_process_exited": True, "peak_gpu_memory_mib": 4096})
    monkeypatch.setattr(runtime, "owned_attempt_pids", lambda _: {123})
    attempt = queue._claim("GPU-test", _sharing_gpu())
    assert attempt["request"]["request_key"] == "third"
    assert set(queue.active) == {"first", "third"}
    admission = runtime.read_json(attempt["folder"] / "admission.json")["resources"]["gpu_sharing"]
    assert admission["peak_reservation_bytes"] == int(5.5 * 1024 ** 3)
    assert admission["total_peak_reservations_bytes"] == 11 * 1024 ** 3
    assert queue.state["second"]["status"] == "pending"
    assert queue._claim("GPU-test", _sharing_gpu()) is None


@pytest.mark.parametrize("blocker", ["unknown_active", "unknown_candidate", "foreign", "memory",
                                    "utilization", "host", "default_one"])
def test_sharing_refuses_unsafe_overlap(tmp_path, monkeypatch, blocker):
    queue = _sharing_queue(tmp_path, monkeypatch, max_jobs=None if blocker == "default_one" else 2)
    first = queue._claim("GPU-test", {"uuid": "GPU-test", "idle": True})
    queue.order = ["first", "second"]
    queue.shape_peaks[queue._shape(first["request"])] = 4 * 1024 ** 3
    monkeypatch.setattr(runtime, "owned_attempt_pids", lambda _: {123})
    gpu = _sharing_gpu()
    if blocker == "unknown_active":
        first["request"]["p2"] = 0.1
    elif blocker == "unknown_candidate":
        queue.by_key["second"]["p2"] = 0.1
    elif blocker == "foreign":
        gpu["compute_processes"].append({"pid": 999, "used_memory_mib": 1})
    elif blocker == "memory":
        gpu["memory_free_mib"] = 5000
    elif blocker == "utilization":
        gpu["utilization_gpu"] = 99
    elif blocker == "host":
        queue._host_available = lambda: 23 * 1024 ** 3
    assert queue._claim("GPU-test", gpu) is None
    assert set(queue.active) == {"first"}
    assert queue.state["second"]["status"] == "pending"


def test_progap_shape_peak_requires_child_measurement(tmp_path, monkeypatch):
    queue = _sharing_queue(tmp_path, monkeypatch, max_jobs=2)
    request = {**queue.requests[0], "method": "progap"}
    folder = tmp_path / "profile"
    (folder / "output").mkdir(parents=True)
    resources = {"peak_cuda_allocated_bytes": 1024 ** 3}
    runtime.atomic_json(folder / "output/result.json", {"resources": resources})
    outcome = {"accepted": True, "owned_process_exited": True}
    queue._remember_peak(request, folder, outcome)
    assert queue._shape(request) not in queue.shape_peaks
    queue._remember_peak(request, folder, {**outcome, "peak_gpu_memory_mib": 6144})
    assert queue.shape_peaks[queue._shape(request)] == 6 * 1024 ** 3
    resources["peak_child_cuda_allocated_bytes"] = 7 * 1024 ** 3
    runtime.atomic_json(folder / "output/result.json", {"resources": resources})
    queue._remember_peak(request, folder, outcome)
    assert queue.shape_peaks[queue._shape(request)] == 8 * 1024 ** 3


def test_shared_workers_really_overlap_and_stop_cleans_all_trees_and_lease(tmp_path, monkeypatch):
    queue = _sharing_queue(tmp_path, monkeypatch, max_jobs=2)
    queue.order = ["first", "second"]
    queue.shape_peaks[queue._shape(queue.requests[0])] = 1024 ** 3
    script = tmp_path / "worker.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "Path(sys.argv[-1]).parent.joinpath('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n")
    monkeypatch.setattr(records, "worker_command", lambda request, folder, **kwargs: [
        sys.executable, str(script), "--out-dir", str(folder / "output")])
    monkeypatch.setattr(queue, "_check_before_launch", lambda: not queue.stop.is_set())
    monkeypatch.setattr(queue, "_stable_idle", lambda _: True)

    def probe(uuid):
        owned = queue._co_owned_pids(uuid)
        return {**_sharing_gpu(), "idle": not owned,
                "compute_processes": [{"pid": pid, "used_memory_mib": 64} for pid in owned]}

    queue._gpu_probe = probe
    controller = threading.Thread(target=queue._worker, args=("GPU-test",))
    controller.start()
    launches = []
    try:
        until = time.monotonic() + 15
        folders = [tmp_path / "attempts" / key / "attempt_1" for key in queue.order]
        while not all((folder / "child.pid").exists() for folder in folders):
            assert time.monotonic() < until, queue.errors
            time.sleep(0.02)
        launches = [runtime.read_json(folder / "launch.json") for folder in folders]
        children = [int((folder / "child.pid").read_text()) for folder in folders]
        assert len({launch["pid"] for launch in launches}) == 2
        for launch, child in zip(launches, children):
            assert runtime.proc_identity(launch["pid"])["state"] not in {"Z", "X"}
            assert runtime.proc_identity(child)["state"] not in {"Z", "X"}
        with pytest.raises(BlockingIOError):
            with runtime.file_lock(queue._lease_path("GPU-test"), nonblocking=True):
                pytest.fail("lease released while shared workers were live")
    finally:
        queue.request_stop()
        controller.join(timeout=15)
    assert not controller.is_alive()
    assert not queue.active and not queue.reservations and not queue.errors
    for launch, child in zip(launches, children):
        _assert_gone(launch["pid"])
        _assert_gone(child)
        outcome = runtime.read_json(Path(launch["folder"]) / "exit.json")
        assert outcome["owned_process_exited"] and outcome["status"] == "interrupted"
    with runtime.file_lock(queue._lease_path("GPU-test"), nonblocking=True):
        pass


def test_deadline_cleans_owned_worker_even_when_telemetry_callback_is_blocked(tmp_path):
    entered, release = threading.Event(), threading.Event()
    command, folder = _command(tmp_path, "import time\ntime.sleep(60)\n")
    def blocked(_):
        entered.set()
        release.wait(15)
    try:
        outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                      policy={"hard_seconds": 0.5, "termination_grace_seconds": 0.1},
                                      on_sample=blocked)
        assert entered.is_set()
        assert outcome["status"] == "timeout" and outcome["owned_process_exited"]
        _assert_gone(runtime.read_json(folder / "launch.json")["pid"])
    finally:
        release.set()


def test_shared_quarantines_recover_in_any_order_without_erasing_foreign_blocker(tmp_path, monkeypatch):
    queue = _sharing_queue(tmp_path, monkeypatch, max_jobs=2)
    foreign = {"gpu_uuid": "GPU-test", "campaign_root": "/foreign",
               "attempt_dir": "/foreign/attempt", "reason": "unverified"}
    runtime.atomic_json(queue._quarantine_path("GPU-test"), foreign)
    first, second = tmp_path / "first", tmp_path / "second"
    queue._quarantine("GPU-test", first, "unverified")
    queue._quarantine("GPU-test", second, "unverified")
    queue._quarantine("GPU-test", first, "repeat")
    queue._clear_quarantine("GPU-test", first)
    remaining = runtime.read_json(queue._quarantine_path("GPU-test"))
    assert remaining["attempt_dir"] == str(second)
    assert remaining["previous_blocker"] == foreign
    queue._clear_quarantine("GPU-test", second)
    assert runtime.read_json(queue._quarantine_path("GPU-test")) == foreign


def test_ideation_queue_rejects_unapproved_concurrency_and_explicit_physical_gpu(tmp_path):
    from scripts.full_matrix_campaign import CampaignQueue
    with pytest.raises(ValueError, match="at most two"):
        CampaignQueue(tmp_path, purpose="ideation", max_jobs_per_gpu=3)
    queue = CampaignQueue(tmp_path, purpose="ideation", gpus="GPU-unrelated",
                          gpu_resolver=lambda _: [{"uuid": "GPU-unrelated", "index": 0}])
    with pytest.raises(ValueError, match="physical GPU indices"):
        queue._start()
    assert not (tmp_path / "gpu_inventory.json").exists()
    assert not queue.active


@pytest.mark.parametrize("depth_argument,expected_depth", [(None, 1), (2, 2)])
def test_progap_runner_depth_controls_native_stages_and_privacy_composition(
    tmp_path, depth_argument, expected_depth
):
    """Exercise native training: an ignored depth changes both training and privacy."""
    if not Path(records.PROGAP_PYTHON).is_file():
        pytest.skip("requires the separately pinned ProGAP environment")
    from scripts import full_matrix_run as worker

    prepared = records._smoke_prepared(tmp_path / "fixture", "cpu")["smoke-accuracy"]
    _, split, task, _ = records.load_prepared_protocol(Path(prepared["manifest"]))
    command = [
        "--dataset", "smoke-accuracy", "--method", "progap",
        "--lr", "0.01", "--batch-size", "32", "--epochs", "1",
        "--gnn-hidden", "8", "--epsilon", "8", "--device", "cpu",
        "--bootstrap-resamples", "0", "--progap-python", records.PROGAP_PYTHON,
        "--out-dir", str(tmp_path / "output"),
    ]
    if depth_argument is not None:
        command += ["--progap-depth", str(depth_argument)]
    args = worker.parser().parse_args(command)
    worker._check_args(args)
    native, parameters = worker._progap(args, split, task, 32, 1 / 128)
    assert parameters["depth"] == expected_depth
    assert parameters["stages"] == expected_depth + 1
    assert native["completed_updates"] == 4 * (expected_depth + 1)
    assert {stage["stage"] for stage in native["selection"]["stage_selections"]} == set(range(expected_depth + 1))
    privacy = native["privacy"]["total"]
    assert privacy["parameters"]["component_coefficients"] == [expected_depth, expected_depth + 1]
    assert privacy["composition_count"] == 2 * expected_depth + 1
    assert privacy["epsilon"] <= 8 * 1.001
