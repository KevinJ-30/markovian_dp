"""Uncertain scheduling boundaries, exercised without GPUs or training imports."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

STUDY = Path(__file__).resolve().parents[1] / "results/eight_gpu_domain_graphsaint"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("_eight_queue_common", "study_common.py")
policy = _load("_eight_queue_policy", "policy.py")
search = _load("_eight_queue_search", "search.py")
_saved = {name: sys.modules.get(name) for name in ("study_common", "policy", "search")}
try:
    sys.modules.update(study_common=common, policy=policy, search=search)
    scheduler = _load("_eight_queue_scheduler", "queue.py")
finally:
    for _name, _module in _saved.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


@pytest.fixture
def source():
    protocol = {"id": "fixture", "dataset": "fixture", "split_strategy": "native",
                "split_seed": 0, "domain_split": None}
    prepared = {"implementation_hash": "fixture-revision", "protocols": {"fixture": {
        "n_train": 10240, "task": {"num_classes": 3, "metric": "accuracy"},
        "manifest": "/fixture/manifest.json", "split_fingerprint": "fixed-split"}}}
    return protocol, prepared


def resolved(source, method="dpmlp", epochs=10):
    protocol, prepared = source
    return common.resolve_cell(protocol, method, 8 if method not in ("mlp", "graphsage") else None,
                               overrides={"epochs": epochs}, prepared=prepared)


def queue_fixture(tmp_path, source):
    queue = object.__new__(scheduler.StudyQueue)
    queue.root = tmp_path
    queue.phase = "baseline"
    queue.phase_dir = tmp_path / "phases/baseline"
    queue.phase_dir.mkdir(parents=True)
    queue.prepared = source[1]
    queue.revision = "fixture-revision"
    queue.mutex = threading.RLock()
    queue.stop = threading.Event()
    queue.requests = []
    queue.dispositions = {}
    queue.accepted = {}
    queue.claimed = set()
    queue.attempted = set()
    queue.reservations = {}
    queue.smoke_estimates = {}
    queue.validation = {"protocols": {"fixture": {"valid": True}}}
    queue.events = []
    queue.event = lambda event, **fields: queue.events.append({"event": event, **fields})
    queue.resources = lambda cell: {"admissible": True, "estimated_host_bytes": 1}
    return queue


def test_warmup_is_not_extrapolated_and_three_eligible_windows_cancel():
    clock = policy.SlowPolicy()
    assert not clock.inspect(600, {"phase": "warmup", "updates_completed": 0,
                                  "total_updates": 1000, "train_update_seconds": 500})["cancel"]
    clock = policy.SlowPolicy()
    clock.inspect(120, {"phase": "train_update", "updates_completed": 20,
                        "total_updates": 1000, "train_update_seconds": 100})
    decisions = []
    for index in (1, 2, 3):
        decisions.append(clock.inspect(120 + 30 * index, {
            "phase": "train_update", "updates_completed": 20 + index,
            "total_updates": 1000, "train_update_seconds": 100 + 30 * index}))
    assert [row["cancel"] for row in decisions] == [False, False, True]
    assert decisions[-1]["consecutive_over_budget"] == 3
    assert decisions[-1]["reason"] == "three_projected_runtime_limits"
    assert decisions[-1]["seconds_per_update"] == 30


def test_hard_timeout_during_indivisible_calibration_has_no_epoch_retry():
    decision = policy.SlowPolicy().inspect(10800, {"phase": "calibration", "updates_completed": 0})
    assert decision["cancel"] and not decision["eligible"]
    assert decision["irreducible_seconds"] == 10800
    assert policy.shorter_epochs(100, decision["projected_seconds"], decision["irreducible_seconds"], 0) is None


def test_projection_counter_resets_after_fast_window():
    monitor = policy.SlowPolicy()
    monitor.inspect(120, {"updates_completed": 20, "total_updates": 200, "train_update_seconds": 100})
    assert monitor.inspect(150, {"updates_completed": 21, "total_updates": 200,
                                 "train_update_seconds": 200})["consecutive_over_budget"] == 1
    decision = monitor.inspect(180, {"updates_completed": 200, "total_updates": 200,
                                     "train_update_seconds": 220})
    assert not decision["cancel"] and decision["consecutive_over_budget"] == 0


def test_owned_group_gets_term_grace_then_kill_and_unrelated_pid_never_signaled():
    command = ["python", "-u", "/study/run.py", "--cell", "/study/a/config.json", "--folder", "/study/a"]
    launch = {"pid": 41, "pgid": 41, "start_ticks": 9, "command": command,
              "observed_command": command, "folder": "/study/a"}
    current = {"pid": 41, "pgid": 41, "start_ticks": 9, "command": command}
    signals = []

    class Process:
        def __init__(self):
            self.waits = []

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired(command, timeout)
            return -9

    process = Process()
    used, returncode = policy.terminate_owned(process, launch, identity=lambda _: current,
                                               signal_group=lambda pgid, sig: signals.append((pgid, sig)))
    assert (used, returncode) == ("SIGKILL", -9)
    assert process.waits == [60, 60]
    assert signals == [(41, signal.SIGTERM), (41, signal.SIGKILL)]
    signals.clear()
    with pytest.raises(RuntimeError, match="ownership"):
        policy.terminate_owned(Process(), launch, identity=lambda _: {**current, "start_ticks": 10},
                               signal_group=lambda pgid, sig: signals.append((pgid, sig)))
    assert signals == []


def test_shortened_schedule_changes_scientific_calibration_key_not_logical_batch(source):
    original = resolved(source)
    epochs = policy.shorter_epochs(original["epochs"], 14400, 600, 0)
    child = scheduler.resolve_related(original, source[1], changes={"epochs": epochs,
                                      "retry_of": common.scientific_key(original)})
    assert 1 <= child["epochs"] < original["epochs"]
    assert child["steps"] < original["steps"]
    assert child["delta"] == original["delta"] and child["batch_size"] == 1024
    assert common.scientific_key(child) != common.scientific_key(original)
    assert policy.shorter_epochs(epochs, 14400, 600, 3) is None
    assert policy.shorter_epochs(1, 14400, 600, 0) is None


def test_oom_retries_only_supported_physical_budgets(source):
    cell = resolved(source)
    smaller = policy.smaller_physical_budget(cell, "train_update")
    assert smaller["physical_chunk_size"] == cell["physical_chunk_size"] // 2
    assert common.scientific_key(smaller) == common.scientific_key(cell)
    assert smaller["steps"] == cell["steps"] and smaller["batch_size"] == 1024
    evaluation = policy.smaller_physical_budget(cell, "test")
    assert evaluation["eval_chunk_size"] == cell["eval_chunk_size"] // 2
    assert common.scientific_key(evaluation) == common.scientific_key(cell)
    assert policy.smaller_physical_budget({**cell, "physical_chunk_size": 1}) is None
    assert policy.smaller_physical_budget(resolved(source, "dpar"), "train_update") is None


def test_every_physical_gpu_is_exclusively_bound_before_launch(tmp_path, source):
    cell = resolved(source)
    for gpu in range(8):
        command, environment = scheduler.runner_command(cell, gpu, tmp_path)
        assert environment["CUDA_VISIBLE_DEVICES"] == str(gpu)
        assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == environment["OPENBLAS_NUM_THREADS"] == "2"
        assert command[0] == scheduler.FIRST_PARTY
        assert "--device" not in command
    progap_command, _ = scheduler.runner_command({**cell, "method": "progap", "family": "progap"}, 7, tmp_path)
    assert progap_command[0] == scheduler.PROGAP


def test_restart_reuses_only_verified_artifacts_and_corruption_requeues(tmp_path, source):
    queue = queue_fixture(tmp_path, source)
    cell = resolved(source)
    key = common.scientific_key(cell)
    folder = tmp_path / "accepted_attempt"
    folder.mkdir()
    queue.accepted[key] = {"folder": str(folder)}
    queue.dispositions[key] = {"status": "completed", "resolved_cell": cell, "folder": str(folder)}
    verified = []
    queue.verify = lambda path, config: verified.append(path) or {"accepted": True}
    assert queue.claim(0, [cell]) == (None, None)
    assert verified == [folder] and key in queue.attempted
    queue.attempted.clear()

    def corrupted(*args):
        raise RuntimeError("checkpoint digest mismatch")

    queue.verify = corrupted
    claimed, admission = queue.claim(0, [cell])
    assert claimed == cell and admission["admissible"]
    assert key not in queue.accepted
    assert any(event["event"] == "ACCEPTANCE_REJECTED" for event in queue.events)


def test_slow_children_stop_after_three_reductions_and_keep_original_request(tmp_path, source):
    queue = queue_fixture(tmp_path, source)
    original = resolved(source, epochs=100)
    attempted = []

    def too_slow(cell, folder, gpu, admission):
        attempted.append(cell)
        return scheduler.ProcessOutcome(-15, 180, {
            "projected_seconds": 14400, "irreducible_seconds": 100,
            "elapsed_seconds": 180, "reason": "three_projected_runtime_limits"})

    queue.run_process = too_slow
    queue.process_request(original, 0, {})
    assert len(attempted) == 4
    assert all(left["epochs"] > right["epochs"] for left, right in zip(attempted, attempted[1:]))
    assert len({common.scientific_key(cell) for cell in attempted}) == 4
    disposition = queue.dispositions[common.scientific_key(original)]
    assert disposition["status"] == "runtime_budget_exhausted"
    assert disposition["requested_cell"]["epochs"] == 100
    assert disposition["resolved_cell"]["retry_index"] == 3
    assert len(disposition["lineage"]) == 4


def test_restart_preserves_pending_retry_count_instead_of_retrying_original(tmp_path, source):
    queue = queue_fixture(tmp_path, source)
    original = resolved(source, epochs=100)
    pending = scheduler.resolve_related(original, source[1], changes={"epochs": 12, "retry_index": 3})
    queue.dispositions[common.scientific_key(original)] = {"status": "blocked", "pending_cell": pending,
                                                          "lineage": ["old1", "old2", "old3"]}
    attempted = []

    def too_slow(cell, folder, gpu, admission):
        attempted.append(cell)
        return scheduler.ProcessOutcome(-15, 180, {"projected_seconds": 14400,
            "irreducible_seconds": 100, "elapsed_seconds": 180, "reason": "three_projected_runtime_limits"})

    queue.run_process = too_slow
    queue.process_request(original, 0, {})
    assert [cell["epochs"] for cell in attempted] == [12]
    assert queue.dispositions[common.scientific_key(original)]["status"] == "runtime_budget_exhausted"


def _rows(config, score=0.6, seconds=90):
    return [{"cell": {**config, "protocol": protocol, "epsilon": epsilon, "seed": 0},
             "status": "completed", "test_metric": score, "process_wall_seconds": seconds}
            for protocol, epsilon in sorted(search.EXPECTED)]


def test_search_weights_datasets_equally_and_does_not_shrink_censored_denominator():
    references = {pair: 0.5 for pair in search.EXPECTED}
    rows = _rows(search.BASELINE, score=0.5)
    for row in rows:
        if row["cell"]["protocol"].startswith("twitch-"):
            row["test_metric"] = 0.6
    score = search.shared_score(rows, references, search.BASELINE)
    assert score["weighted_win_fraction"] == pytest.approx(1 / 7)
    rows[0]["status"] = "runtime_budget_exhausted"
    assert search.shared_score(rows, references, search.BASELINE) == {"eligible": False, "reason": "runtime_censored"}
    rows[1]["status"] = "blocked"
    assert search.shared_score(rows, references, search.BASELINE)["reason"] == "blocked"
    assert search.shared_score(rows[:-1], references, search.BASELINE)["reason"] == "incomplete_sweep"


def test_search_mandatory_coverage_then_two_complete_sweeps_without_improvement():
    state = search.new_search_state()
    references = {pair: 0.5 for pair in search.EXPECTED}
    for candidate_id in range(9):
        candidate = search.next_candidate(state)
        assert candidate["id"] == candidate_id
        search.complete_candidate(state, candidate, _rows(candidate["config"]), references)
        assert state["patience"] == 0
    assert [entry["config"] for entry in state["candidates"]][1:] == [
        {**search.BASELINE, **change} for change in search.MANDATORY]
    for patience in (1, 2):
        candidate = search.next_candidate(state)
        search.complete_candidate(state, candidate, _rows(candidate["config"]), references)
        assert state["patience"] == patience
    assert search.next_candidate(state) is None
    assert state["stop_reason"] == "two_nonimproving_completed_sweeps"


def test_significant_improvement_resets_patience_and_fast_cap_needs_every_protocol():
    state = search.new_search_state()
    references = {pair: 0.5 for pair in search.EXPECTED}
    for _ in range(9):
        candidate = search.next_candidate(state)
        search.complete_candidate(state, candidate, _rows(candidate["config"], seconds=30), references)
    assert state["cap"] == 128
    candidate = search.next_candidate(state)
    search.complete_candidate(state, candidate, _rows(candidate["config"], seconds=30), references)
    assert state["patience"] == 1
    improved = search.next_candidate(state)
    search.complete_candidate(state, improved, _rows(improved["config"], score=0.61, seconds=30), references)
    assert state["patience"] == 0 and state["incumbent"] == improved["id"]
    tiny = {**improved["score"], "worst_delta": improved["score"]["worst_delta"] + 0.0001,
            "weighted_mean_delta": improved["score"]["weighted_mean_delta"] + 0.0001}
    assert not search.significant_improvement(tiny, improved["score"])


def test_runtime_shortening_cannot_win_shared_score_but_is_reported_per_protocol():
    state = search.new_search_state()
    candidate = search.next_candidate(state)
    rows = _rows(candidate["config"], score=0.99)
    rows[0]["cell"]["epochs"] = 5
    rows[0]["status"] = "shortened"
    search.complete_candidate(state, candidate, rows, {pair: 0.5 for pair in search.EXPECTED})
    assert candidate["status"] == "runtime_censored" and state["incumbent"] is None
    assert len(state["per_protocol_best"]) == 16


def test_missing_reference_stops_only_after_mandatory_coverage():
    state = search.new_search_state()
    for _ in range(9):
        candidate = search.next_candidate(state)
        search.complete_candidate(state, candidate, _rows(candidate["config"]), {})
    assert state["stopped"] and state["stop_reason"] == "incomplete_runtime_limited_reference"
    assert len(state["candidates"]) == 9


def test_seed_summary_never_pools_different_effective_schedules():
    rows = [{"cell": {"seed": seed}, "status": "completed", "test_metric": score,
             "effective_configuration": configuration}
            for seed, score, configuration in [(0, 0.7, "epochs10"), (1, 0.8, "epochs10"), (2, 0.9, "epochs5")]]
    summaries = search.matching_seed_summary(rows)
    assert summaries["epochs10"]["n"] == 2 and summaries["epochs5"]["n"] == 1
    assert not any(row["complete_three_seed"] for row in summaries.values())
    assert summaries["epochs10"]["mean"] == pytest.approx(0.75)


def test_final_test_is_estimated_once_and_partial_test_time_is_already_paid():
    monitor = policy.SlowPolicy()
    smoke = {"seconds_per_evaluation": 40.0, "test_seconds": 1000.0}
    progress = {"phase": "train_update", "updates_completed": 20, "total_updates": 100,
                "epochs_completed": 1, "evaluations_completed": 1, "total_evaluations": 4,
                "train_update_seconds": 100.0, "validation_seconds": 40.0, "test_seconds": 0.0}
    monitor.inspect(140, progress, smoke)
    decision = monitor.inspect(170, {**progress, "updates_completed": 30,
                                     "train_update_seconds": 110.0}, smoke)
    assert decision["remaining_validation_evaluations"] == 2
    assert decision["remaining_test_seconds"] == decision["irreducible_seconds"] == 1000
    assert decision["projected_seconds"] == pytest.approx(170 + 70 + 2 * 40 + 1000)
    partial = {**progress, "phase": "test", "updates_completed": 100, "epochs_completed": 3,
               "evaluations_completed": 3, "train_update_seconds": 180.0,
               "validation_seconds": 120.0, "test_seconds": 200.0}
    decision = monitor.inspect(1500, partial, smoke)
    assert decision["remaining_validation_evaluations"] == 0
    assert decision["remaining_test_seconds"] == 800
    assert decision["projected_seconds"] == 2300
    final = monitor.inspect(2300, {**partial, "phase": "test_complete", "evaluations_completed": 4,
                                  "test_seconds": 1000.0}, smoke)
    assert final["remaining_test_seconds"] == 0 and final["projected_seconds"] == 2300
    assert final["recent_evaluation_seconds"] == [40.0]


def test_expanded_matrix_is_36_and_transferred_baseline_winner_reuses_keys(tmp_path, source, monkeypatch):
    queue = queue_fixture(tmp_path, source)
    queue.phase = "expanded_domains"
    expanded = [{**source[0], "id": f"expanded-{index}"} for index in range(3)]
    queue.prepared = {**source[1], "protocols": {protocol["id"]: dict(source[1]["protocols"]["fixture"])
                                                for protocol in expanded}}
    monkeypatch.setattr(scheduler, "protocols", lambda expanded=False: globals_expanded)
    globals_expanded = expanded
    queue.require_resolved_phase = lambda phase: {}
    queue.winner = lambda: dict(search.BASELINE)
    captured = []
    queue.run_batch = lambda cells: captured.extend(cells) or True
    assert queue.run_expanded()
    assert len(captured) == len({common.scientific_key(cell) for cell in captured}) == 36
    assert sum(cell["epsilon"] is None for cell in captured) == 6


def test_confirmation_deduplicates_selections_and_keeps_feasible_seed0_schedule(tmp_path, source):
    queue = queue_fixture(tmp_path, source)
    queue.phase = "confirmation"
    cell = resolved(source, "sparse", epochs=5)
    cell["requested_epochs"] = 10
    cell["selection_policy"] = "test_selected_exploratory"
    record = {"status": "shortened", "resolved_cell": cell}
    queue.require_resolved_phase = lambda phase: {"same_selected_cell": record}
    queue.winner = lambda: {**search.BASELINE, "epochs": 5}
    row = {"status": "shortened", "cell": cell, "test_metric": 0.7}
    common.atomic_json(tmp_path / "search_state.json", {"incumbent": 0, "candidates": [{"rows": [row]}],
                                                       "per_protocol_best": {"fixture:8": row}})
    captured = []
    queue.run_batch = lambda cells: captured.extend(cells) or True
    assert queue.run_confirmation()
    assert len(captured) == 2 and {entry["seed"] for entry in captured} == {1, 2}
    assert all(entry["epochs"] == entry["requested_epochs"] == 5 for entry in captured)
    assert {entry["cap_seed"] for entry in captured} == {20001, 20002}
    assert all(entry["selection_policy"] == "test_selected_exploratory" for entry in captured)
