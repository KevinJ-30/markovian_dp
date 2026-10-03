"""GPU authorization and real process-lifetime safety contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from scripts import runner_runtime as runtime


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


def test_observed_oom_is_distinct_from_unrelated_failure(tmp_path):
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
    with pytest.raises(RuntimeError, match="authorizes no devices"):
        runtime.resolve_gpus("auto")


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


def test_recovery_terminates_owned_worker_after_controller_crash(tmp_path):
    worker_script = tmp_path / "sleeper.py"
    worker_script.write_text("import time\ntime.sleep(60)\n")
    folder = tmp_path / "attempt"
    folder.mkdir()
    supervisor_script = tmp_path / "supervisor.py"
    supervisor_script.write_text(
        "import os,sys,threading\nfrom pathlib import Path\n"
        "from scripts.runner_runtime import run_process\n"
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


@pytest.mark.parametrize("mismatch", [None, "start_ticks", "command", "session_id", "boot_id"])
def test_admission_revalidates_recorded_live_process_identities(tmp_path, mismatch):
    command, folder = _command(tmp_path, "import time\ntime.sleep(60)\n")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        identity = runtime.proc_identity(process.pid)
        launch = {**identity, "command": command, "observed_command": identity["command"],
                  "folder": str(folder), "boot_id": runtime._boot_id(), "spawned": True}
        runtime.atomic_json(folder / "launch.json", launch)
        saved = dict(identity)
        evidence = {"boot_id": launch["boot_id"], "leader_pid": process.pid, "members": [saved]}
        if mismatch == "boot_id":
            evidence["boot_id"] = "another-boot"
        elif mismatch == "command":
            saved["command"] = ["foreign"]
        elif mismatch is not None:
            saved[mismatch] += 1
        runtime.atomic_json(folder / "process_identities.json", evidence)
        if mismatch is None:
            assert runtime.owned_attempt_pids(folder) == {process.pid}
        else:
            with pytest.raises(runtime.OwnershipError):
                runtime.owned_attempt_pids(folder)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


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


def test_real_child_without_hard_deadline(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                  policy={"hard_seconds": None})
    assert outcome["status"] == "completed" and outcome["returncode"] == 0
    assert outcome["owned_process_exited"]


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


def test_normal_exit_allows_inflight_telemetry_to_settle(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(0.3)\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
                                  policy={"hard_seconds": None},
                                  on_sample=lambda _: time.sleep(2.5))
    assert outcome["returncode"] == 0
    assert outcome["status"] == "completed" and outcome["owned_process_exited"]


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


@pytest.mark.parametrize("spec", ["auto", "0", "GPU-first"])
def test_empty_masks_reject_without_querying_cuda(monkeypatch, spec):
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runtime, "gpu_inventory", lambda: pytest.fail("empty authorization queried devices"))
    with pytest.raises(RuntimeError, match="authorizes no devices"):
        runtime.resolve_gpus(spec)


def test_visibility_intersection_preserves_numeric_cuda_mapping(monkeypatch):
    inventory = {f"GPU-{name}": {"uuid": f"GPU-{name}", "index": index}
                 for index, name in enumerate(("first", "second", "third"))}
    monkeypatch.setattr(runtime, "gpu_inventory", lambda: inventory)
    monkeypatch.setattr(runtime, "_cuda_visible_uuids", lambda: ["GPU-third", "GPU-second"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    assert [row["uuid"] for row in runtime.resolve_gpus("auto")] == ["GPU-second"]
    assert runtime.resolve_gpus("1")[0]["uuid"] == "GPU-second"
    with pytest.raises(RuntimeError, match="outside inherited authorization"):
        runtime.resolve_gpus("2")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-first")
    with pytest.raises(RuntimeError, match="no devices"):
        runtime.resolve_gpus("auto")


def test_allocation_without_visibility_rejects_ambiguous_auto(monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(runtime, "gpu_inventory", lambda: {
        "GPU-only": {"uuid": "GPU-only", "index": 0}})
    with pytest.raises(RuntimeError, match="verified explicit allow-list"):
        runtime.resolve_gpus("auto")
    assert runtime.resolve_gpus("0")[0]["uuid"] == "GPU-only"


def test_missing_compute_memory_never_becomes_idle(monkeypatch):
    def query(kind, fields):
        if kind == "gpu":
            return [["GPU-only", "0", "fixture", "16384", "500", "15884", "0", "1"]]
        return [["123", "GPU-only", "N/A"]]
    monkeypatch.setattr(runtime, "_nvidia_query", query)
    snapshot = runtime.gpu_snapshot("GPU-only")
    assert not snapshot["idle"]
    assert snapshot["compute_processes"][0]["used_memory_mib"] is None


def test_transient_gpu_failure_recovers_without_canceling_child(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(1.2)\n")
    samples = []
    class Sampler:
        calls = 0
        def snapshot(self, uuid):
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary query failure")
            return {"uuid": uuid, "compute_processes": [], "idle": True}
    observer = Sampler()
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        gpu={"uuid": "GPU-fixture"}, sampler=observer, on_sample=samples.append,
        policy={"poll_seconds": 0.05})
    assert outcome["status"] == "completed" and outcome["returncode"] == 0
    assert any(sample["gpu"].get("error") for sample in samples)
    assert any(sample["owned_gpu_memory_mib"] == 0 and sample["tree_rss_bytes"] > 0
               for sample in samples if sample["tree_rss_bytes"] is not None)


def test_callback_failure_retries_and_preserves_healthy_child(tmp_path):
    observed = tmp_path / "recovered"
    command, folder = _command(tmp_path,
        "import time\nfrom pathlib import Path\n"
        f"while not Path({str(observed)!r}).exists(): time.sleep(.02)\n")
    samples = []
    def flaky(sample):
        samples.append(sample)
        if len(samples) == 1:
            raise OSError("controller sample delivery failed once")
        if sample["tree_rss_bytes"] is not None:
            observed.touch()
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        policy={"poll_seconds": 0.05, "hard_seconds": 30}, on_sample=flaky)
    assert outcome["status"] == "completed" and outcome["returncode"] == 0
    assert any(sample["observed_monotonic"] > samples[0]["observed_monotonic"]
               and sample["tree_rss_bytes"] is not None for sample in samples[1:])


def test_process_tree_memory_includes_nested_training_child(tmp_path):
    child_file = tmp_path / "nested.pid"
    command, folder = _command(tmp_path,
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"Path({str(child_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(1.2)\n")
    samples = []
    class Sampler:
        def snapshot(self, uuid):
            leader = runtime.read_json(folder / "launch.json")["pid"]
            rows = [{"pid": leader, "used_memory_mib": 100}]
            if child_file.exists():
                rows.append({"pid": int(child_file.read_text()), "used_memory_mib": 300})
            return {"uuid": uuid, "compute_processes": rows}
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event(),
        gpu={"uuid": "GPU-fixture"}, sampler=Sampler(), on_sample=samples.append,
        policy={"poll_seconds": 0.05, "termination_grace_seconds": 0.1})
    assert outcome["status"] == "completed" and outcome["owned_process_exited"]
    assert outcome["peak_gpu_memory_mib"] == 400
    assert any(len(sample["owned_pids"]) == 2 and sample["tree_rss_bytes"] > 0
               and sample["owned_gpu_memory_mib"] == 400 for sample in samples)
    _assert_gone(int(child_file.read_text()))


def test_recovery_never_infers_exit_status_from_completed_outputs(tmp_path):
    command, folder = _command(tmp_path,
        "import json,sys\nfrom pathlib import Path\n"
        "out=Path(sys.argv[-1]);out.mkdir()\n"
        "(out/'result.json').write_text(json.dumps({'status':'completed'}))\n"
        "(out/'result.csv').write_text('test_metric\\n0.5\\n')\n")
    outcome = runtime.run_process(command, folder, dict(os.environ), threading.Event())
    assert outcome["returncode"] == 0
    (folder / "exit.json").unlink()
    recovered = runtime.recover_process(folder, grace=0.1)
    assert recovered["status"] == "interrupted"
    assert recovered["returncode"] is None and not recovered["os_exit_observed"]
    assert recovered["owned_process_exited"]


def test_unverified_recovery_leaves_live_process_and_temporary_files(tmp_path):
    command, folder = _command(tmp_path, "import time\ntime.sleep(60)\n")
    (folder / "tmp").mkdir()
    (folder / "tmp" / "child-data").write_text("still in use")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        identity = runtime.proc_identity(process.pid)
        runtime.atomic_json(folder / "launch.json", {
            **identity, "command": command, "observed_command": identity["command"],
            "folder": str(folder), "boot_id": runtime._boot_id(), "spawned": True,
            "start_ticks": identity["start_ticks"] + 1})
        outcome = runtime.recover_process(folder, grace=0.1)
        assert outcome["status"] == "ownership_unverified"
        assert not outcome["owned_process_exited"] and outcome["returncode"] is None
        assert process.poll() is None
        assert (folder / "tmp" / "child-data").read_text() == "still in use"
    finally:
        process.terminate()
        process.wait(timeout=5)
