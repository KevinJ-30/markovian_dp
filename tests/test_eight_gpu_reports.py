"""Synthetic evidence boundaries: no GPUs, dataset downloads or training required."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[1] / "results/eight_gpu_domain_graphsaint"


def _module(name):
    # Isolate generic module names from other result workspaces in the same suite.
    path = str(STUDY)
    previous_path = list(sys.path)
    names = ("study_common", "verification_common", "verify_complete", "summarize", "search")
    previous_modules = {key: sys.modules.get(key) for key in names}
    try:
        sys.path[:] = [entry for entry in sys.path if entry != path]
        sys.path.append(path)
        for key in names:
            sys.modules.pop(key, None)
        spec = importlib.util.spec_from_file_location(f"eight_gpu_reports_{name}", STUDY / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path
        for key in names:
            if previous_modules[key] is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous_modules[key]


@pytest.fixture
def modules():
    return _module("summarize"), _module("verification_common"), _module("verify_complete")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _cell(seed=0, epochs=2, phase="baseline"):
    return {"protocol": "fixture", "dataset": "fixture", "split": {"id": "fixture", "domain_split": {
                "train": ["source"], "val": ["target"], "test": ["target"]}},
            "split_fingerprint": "frozen-split", "method": "mlp", "family": "nonprivate", "epsilon": None,
            "delta": None, "seed": seed, "n_train": 2048, "task": {"metric": "accuracy", "metric_ignore_label": None},
            "epochs": epochs, "requested_epochs": epochs, "steps_per_epoch": 2, "steps": 2 * epochs,
            "learning_rate": .01, "clip": 1., "hidden_size": 64, "batch_size": 1024,
            "phase": phase, "selection_policy": "validation_selected_checkpoint",
            "implementation_hash": "fixture-implementation", "accountant_hash": None}


def _stats(role):
    return {"context_nodes": 2048 if role == "train" else 100, "scored_nodes": 2048 if role == "train" else 20 if role == "val" else 80,
            "metric_scored_nodes": 2048 if role == "train" else 20 if role == "val" else 80,
            "edges": 8, "topology_hash": "train-edges" if role == "train" else "shared-target-edges",
            "eval_mask_hash": f"{role}-mask", "node_ids_hash": "source" if role == "train" else "target",
            "metric_ignore_label": None}


def _attempt(root, cell, verifier, *, run="run-complete", score=.7, register=True):
    folder = root / "runs" / run / "attempt_0001"
    folder.mkdir(parents=True)
    partitions = {role: _stats(role) for role in ("train", "val", "test")}
    manifest = root / "prepared/fixture/manifest.json"
    _json(manifest, {"fixture": True})
    _json(root / "prepared.json", {"protocols": {"fixture": {"n_train": 2048, "manifest": str(manifest),
          "split_fingerprint": "frozen-split", "partitions": partitions}}})
    _json(folder / "config.json", cell)
    _json(folder / "admission.json", {"gpu": {"idle": True, "compute_processes": [], "utilization_percent": 0, "memory_used_mib": 0}, "queue_wait_seconds": 2})
    _json(folder / "launch.json", {"run_id": run, "physical_gpu": 7, "logical_device": "cuda:0", "started_utc": "2026-09-23T00:00:00Z"})
    _json(folder / "exit.json", {"returncode": 0, "process_wall_seconds": 12, "owned_process_exited": True, "cancelled_slow": False, "finished_utc": "2026-09-23T00:00:12Z"})
    _json(folder / "gpu_release.json", {"idle": True})
    _json(folder / "sampling.json", {"sampler": "shuffled_without_replacement", "updates_completed": cell["steps"],
          "roots_total": cell["steps"] * 1024, "epochs_completed": cell["epochs"]})
    history = [{"epoch": epoch, "step": epoch * 2, "validation_metric": score, "loss": .5}
               for epoch in range(1, cell["epochs"] + 1)]
    _json(folder / "history.json", history)
    _json(folder / "result.json", {"status": "completed", "metric": "accuracy", "validation_metric": score,
          "test_metric": score, "best_epoch": 1, "best_step": 2, "updates_completed": cell["steps"],
          "epsilon": None, "delta": None, "evaluation": {k: v for k, v in partitions.items() if k != "train"},
          "timing": {**{k: 1. for k in verifier.TIMING_PHASES}, "run_wall_seconds": 10.}})
    (folder / "process.log").write_text("actual fixture process exited\n")
    (folder / "checkpoint.pt").write_bytes(b"opaque checkpoint artifact for hash verification")
    source = root / "fixture_source.py"
    source.write_text("# immutable fixture numerical source\n")
    provenance = {"scientific_key": verifier.scientific_key(cell), "implementation_hash": cell["implementation_hash"],
                  "accountant_hash": cell["accountant_hash"], "environment": {"CUDA_VISIBLE_DEVICES": "7"},
                  "partitions": partitions, "source_sha256": {str(source): verifier.sha256(source)},
                  "prepared_manifest_hash": verifier.sha256(manifest),
                  "artifacts": {name: verifier.sha256(folder / name) for name in (
                      "config.json", "result.json", "history.json", "sampling.json", "checkpoint.pt")}}
    _json(folder / "provenance.json", provenance)
    evidence = verifier.verify_attempt(folder, cell, root=root)
    assert evidence["accepted"], evidence["errors"]
    if register:
        index_path = root / "accepted.json"
        accepted = json.loads(index_path.read_text()) if index_path.exists() else {}
        accepted[verifier.scientific_key(cell)] = {"folder": str(folder)}
        _json(index_path, accepted)
    return folder


def _event(cell, folder, status, run, event="ATTEMPT_COMPLETED"):
    return {"utc": "2026-09-23T00:00:00Z", "event": event, "status": status,
            "phase": cell["phase"], "folder": str(folder), "cell": cell, "run_id": run}


def test_preparing_reports_preserve_authored_prose_and_no_invented_metrics(tmp_path, modules):
    publisher, _, _ = modules
    authored = "# Conclusions\n\n## 2026-09-23 authored interpretation\nNo data yet.\n"
    (tmp_path / "CONCLUSIONS.md").write_text(authored)
    publisher.summarize(tmp_path)
    publisher.summarize(tmp_path)
    text = (tmp_path / "CONCLUSIONS.md").read_text()
    assert text.startswith(authored)
    assert text.count("<!-- BEGIN GENERATED:CONCLUSIONS -->") == 1
    assert json.loads((tmp_path / "comparison.json").read_text())["rows"] == []
    assert "No accepted scientific result" in text
    assert "repository_path_bound_chi1" in text and "test-selected exploratory" in text
    assert (tmp_path / "comparison.csv").read_text().startswith("phase,requested_key,")


def test_publisher_keeps_completed_cancelled_failed_and_shortened_attempts(tmp_path, modules):
    publisher, verifier, _ = modules
    original = _cell()
    child = {**original, "epochs": 1, "steps": 2, "retry_of": verifier.scientific_key(original), "retry_index": 1}
    completed = _attempt(tmp_path, child, verifier, run="shortened", score=.75)
    cancelled = tmp_path / "runs/original/attempt_0001"
    failed = tmp_path / "runs/failed/attempt_0001"
    for path, cell, status in ((cancelled, original, "cancelled_slow"), (failed, _cell(seed=1), "failed")):
        _json(path / "config.json", cell)
        _json(path / "exit.json", {"returncode": -15 if status == "cancelled_slow" else 1,
                                  "process_wall_seconds": 3600 if status == "cancelled_slow" else 60,
                                  "reason": status})
    requests = [original, _cell(seed=1)]
    _json(tmp_path / "phases/baseline/requests.json", requests)
    _json(tmp_path / "phases/baseline/dispositions.json", {
        verifier.scientific_key(original): {"status": "shortened", "requested_cell": original,
                                         "resolved_cell": child, "folder": str(completed)},
        verifier.scientific_key(requests[1]): {"status": "blocked", "folder": str(failed), "reason": "fixture failure"}})
    events = [_event(original, cancelled, "cancelled_slow", "original", "ATTEMPT_CANCELLED"),
              _event(child, completed, "completed", "shortened"),
              _event(requests[1], failed, "failed", "failed", "ATTEMPT_FAILED")]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    publisher.summarize(tmp_path)
    publisher.summarize(tmp_path)
    ledger = (tmp_path / "EXPERIMENTS.md").read_text()
    for run in ("original", "shortened", "failed"):
        assert ledger.count(f"### Attempt `{run}`") == 1
    payload = json.loads((tmp_path / "comparison.json").read_text())
    assert payload["counts"]["attempts"] == 3
    assert payload["counts"]["cumulative_measured_gpu_hours"] == pytest.approx((3600 + 60 + 12) / 3600)
    row = next(r for r in payload["rows"] if r["status"] == "shortened")
    assert (row["requested_epochs"], row["effective_epochs"], row["test_metric"]) == (2, 1, .75)
    assert next(r for r in payload["rows"] if r["status"] == "blocked")["test_metric"] is None
    hyperparameters = (tmp_path / "HYPERPARAMETERS.md").read_text()
    assert f"config-{verifier.scientific_key(original)}" in hyperparameters
    assert f"config-{verifier.scientific_key(child)}" in hyperparameters
    assert "physical_chunk_size" in hyperparameters and "missing / not yet recorded" in hyperparameters


def test_accepted_checkpoint_corruption_stays_rejected_on_reverification(tmp_path, modules):
    _, verifier, _ = modules
    cell = _cell()
    folder = _attempt(tmp_path, cell, verifier)
    (folder / "checkpoint.pt").write_bytes(b"corrupted checkpoint")
    first = verifier.verify_attempt(folder, cell, root=tmp_path)
    second = verifier.verify_attempt(folder, cell, root=tmp_path)
    assert not first["accepted"] and not second["accepted"]
    assert any("checkpoint.pt" in error for error in second["errors"])


def test_changed_context_mask_and_update_count_are_rejected(tmp_path, modules):
    _, verifier, _ = modules
    cell = _cell()
    folder = _attempt(tmp_path, cell, verifier)
    result = json.loads((folder / "result.json").read_text())
    result["evaluation"]["val"]["scored_nodes"] = 100
    result["updates_completed"] += 1
    _json(folder / "result.json", result)
    evidence = verifier.verify_attempt(folder, cell, root=tmp_path)
    assert not evidence["accepted"]
    assert any("logical update" in error for error in evidence["errors"])
    assert any("val evaluation" in error for error in evidence["errors"])


def test_seed_statistics_do_not_pool_later_shortened_schedule_or_duplicate_seed(modules):
    publisher, _, _ = modules
    attempts = []
    for seed, epochs, score in ((0, 2, .5), (1, 2, .7), (2, 1, .99)):
        cell = _cell(seed=seed, epochs=epochs, phase="confirmation" if seed else "baseline")
        attempts.append({"accepted": True, "cell": cell, "scientific_key": publisher.scientific_key(cell),
                         "result": {"test_metric": score}, "run_id": f"seed-{seed}", "folder": f"seed-{seed}"})
    attempts.append(copy.deepcopy(attempts[0]))
    groups = publisher.matched_seed_summary(attempts)
    matched = next(group for group in groups if group["epochs"] == 2)
    assert matched["n"] == 2 and matched["seeds"] == [0, 1]
    assert matched["test_mean"] == pytest.approx(.6)
    assert matched["test_std"] == pytest.approx(2 ** .5 * .1)
    assert not any(group["complete_three_seeds"] for group in groups)


def test_runtime_disposition_requires_owned_exit_and_gpu_release(tmp_path, modules):
    _, verifier, phase_verifier = modules
    cell = _cell(phase="confirmation")
    key = verifier.scientific_key(cell)
    folder = tmp_path / "runs/slow/attempt_0001"
    _json(folder / "exit.json", {"returncode": -15, "process_wall_seconds": 8000, "cancelled_slow": True,
                                "cancellation": {"reason": "projected_timeout"}, "owned_process_exited": True})
    _json(folder / "gpu_release.json", {"idle": True})
    _json(tmp_path / "phases/confirmation/requests.json", [cell])
    disposition = {"status": "runtime_budget_exhausted", "requested_cell": cell,
                   "folder": str(folder), "reason": "projected_timeout"}
    _json(tmp_path / "phases/confirmation/dispositions.json", {key: disposition})
    evidence = phase_verifier.verify_phase("confirmation", root=tmp_path)
    assert evidence["complete"] and not evidence["fully_executed"] and evidence["runtime_limited"]
    (folder / "gpu_release.json").unlink()
    assert not phase_verifier.verify_phase("confirmation", root=tmp_path)["complete"]


def test_extra_disposition_cannot_satisfy_a_missing_request(tmp_path, modules):
    _, verifier, phase_verifier = modules
    cell = _cell(phase="confirmation")
    _json(tmp_path / "phases/confirmation/requests.json", [cell])
    _json(tmp_path / "phases/confirmation/dispositions.json", {"unrequested-key": {"status": "completed"}})
    evidence = phase_verifier.verify_phase("confirmation", root=tmp_path)
    assert not evidence["complete"]
    assert any("unrequested" in error for error in evidence["errors"])
    assert any("unresolved" in error for error in evidence["errors"])


def test_preparation_still_lists_all_96_requested_slots(tmp_path, modules):
    publisher, _, _ = modules
    protocols = [{"id": f"protocol-{index}", "dataset": f"dataset-{index}", "domain_split": None}
                 for index in range(8)]
    _json(tmp_path / "design.json", {"protocols": protocols,
          "methods": ["mlp", "graphsage", "dpmlp", "dpar", "progap", "dpgnn", "sparse"],
          "epsilons": [2, 8], "initial_requested_cells": 96})
    _json(tmp_path / "prepared.json", {"protocols": {}, "blocked": {"protocol-0": "missing raw artifact"}})
    publisher.summarize(tmp_path)
    rows = json.loads((tmp_path / "comparison.json").read_text())["rows"]
    assert len(rows) == 96
    assert sum(row["status"] == "blocked_preparation" for row in rows) == 12
    assert all(row["test_metric"] is None and row["scientific_key"] is None for row in rows)
    assert sum(row["epsilon_target"] is None for row in rows) == 16


def test_state_claim_alone_cannot_publish_a_uniform_winner(modules):
    publisher, _, _ = modules
    state = {"incumbent": 0, "candidates": [{"id": 0, "score": {"eligible": True},
              "config": {}, "rows": []}]}
    summary = publisher.shared_winner_comparisons(state, [], [])
    assert not summary["available"]
    assert summary["denominator"] == 16
    assert summary["weighted_win_fraction"] is None
    assert summary["uniform_improvement"] is None
    assert len(summary["rows"]) == 16
    assert all(row["test_metric"] is None for row in summary["rows"])
