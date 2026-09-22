"""Study-matrix and subprocess fallback invariants."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/graphsaint_p1_01_batch1024_256_k10_r1_r2_eps2_eps8_seed0_20260921"
NORMALIZED_STUDY = ROOT / "results/graphsaint_normalized_batch1024_256_sparse10ep_h64_degree10_progap256_eps2_eps8_seed0_20260922"
sys.path.insert(0, str(STUDY))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


study_common = load_module("matched_study_common", STUDY / "study_common.py")
study_queue = load_module("matched_study_queue", STUDY / "queue.py")
study_summary = load_module("matched_study_summary", STUDY / "summarize.py")
normalized_common = load_module("normalized_study_common", NORMALIZED_STUDY / "study_common.py")
normalized_summary = load_module("normalized_study_summary", NORMALIZED_STUDY / "summarize.py")



def test_design_expands_to_104_unique_scientific_keys():
    design = study_common.load_design()
    cells = study_common.expand_matrix(design)
    keys = [study_common.scientific_key(cell) for cell in cells]
    assert len(cells) == len(set(keys)) == 104
    assert sum(cell["family"] == "sparse" for cell in cells) == 64
    assert sum(cell["family"] in {"dpgnn", "dpmlp", "dpar", "progap"} for cell in cells) == 32
    assert sum(cell["family"] == "nonprivate" for cell in cells) == 8
    assert design["physical_gpus"] == list(range(8))
    assert study_queue.validate_design(list(range(8)))["gpus"] == list(range(8))
    with pytest.raises(ValueError, match="GPUs must be exactly"):
        study_queue.validate_design([1, 2, 3, 4, 5])
    for cell in cells:
        if cell["family"] != "sparse":
            continue
        row = study_common.dataset_map(design)[cell["dataset"]]
        expected = {
            "expected_batch_1024": 1024 / row["training_nodes"],
            "expected_batch_256": 256 / row["training_nodes"],
        }[cell["sampling_mode"]]
        assert cell["p1"] == expected


def fake_child(tmp_path: Path) -> Path:
    path = tmp_path / "fake_calibrator.py"
    path.write_text(
        """import json, pathlib, sys, time
mode, trial = sys.argv[1], pathlib.Path(sys.argv[2])
def emit(value):
    temporary = trial / 'calibration_progress.json.tmp'
    temporary.write_text(json.dumps(value))
    temporary.replace(trial / 'calibration_progress.json')
    print(json.dumps(value), flush=True)
emit({'event':'calibration_start','grid':0.001})
if mode == 'error':
    raise SystemExit(3)
emit({'event':'sigma_evaluation_start','evaluation':1,'sigma':1.0,'grid':0.001})
if mode == 'stall':
    time.sleep(60)
emit({'event':'sigma_evaluation_complete','evaluation':1,'sigma':1.0,'grid':0.001,'epsilon':1.0,'elapsed_seconds':0.0})
emit({'event':'calibration_complete','sigma':1.0,'epsilon':1.0,'evaluations':1,'grid':0.001,'elapsed_seconds':0.0})
"""
    )
    return path


def idle_snapshot(gpu: int, _log: Path):
    return {"captured_at": 0.0, "gpu": gpu, "uuid": "fake", "memory_used_mib": 0,
            "utilization_percent": 0, "compute_processes": [], "idle": True}


def exercise_fallback(monkeypatch, tmp_path: Path, fine_mode: str):
    parent = tmp_path / "attempts"
    script = fake_child(tmp_path)
    cell = {"family": "sparse", "method": "SparseGNN", "dataset": "saint-flickr",
            "epsilon": 2.0, "sampling_mode": "expected_batch_1024", "p1": 1024 / 44625,
            "expected_batch": 1024.0, "p2": .5, "r": 1}
    launched = []
    verified = []

    def command(_cell, _gpu, trial, *, grid=None):
        mode = fine_mode if grid == .001 else "complete"
        launched.append((grid, trial, mode))
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1",
                       "GRAPHSAINT_DATA_ROOT": str(ROOT / "data/graphsaint"),
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        return [sys.executable, "-u", str(script), mode, str(trial)], environment

    def process_runner(command_value, trial, gpu, environment, **kwargs):
        return study_queue.run_process(command_value, trial, gpu, environment,
                                       idle_wait=idle_snapshot, **kwargs)

    monkeypatch.setattr(study_queue, "trial_parent", lambda _cell: parent)
    monkeypatch.setattr(study_queue, "runner_command", command)
    monkeypatch.setattr(study_queue, "verify_and_publish",
                        lambda _cell, trial: verified.append(trial))
    accepted = study_queue.run_sparse_with_fallback(
        cell, 1, stall_seconds=.15, process_runner=process_runner)
    return parent, launched, verified, accepted


def test_progressing_calibration_completes_without_fallback(monkeypatch, tmp_path):
    parent, launched, verified, accepted = exercise_fallback(monkeypatch, tmp_path, "complete")
    assert [grid for grid, _, _ in launched] == [.001]
    assert accepted == parent / "grid_0.001_attempt_1"
    assert verified == [accepted]
    assert not list(parent.glob("grid_0.01_attempt_*"))


def test_stalled_calibration_preserves_fine_attempt_and_retries_once(monkeypatch, tmp_path):
    parent, launched, verified, accepted = exercise_fallback(monkeypatch, tmp_path, "stall")
    assert [grid for grid, _, _ in launched] == [.001, .01]
    fine = parent / "grid_0.001_attempt_1"
    coarse = parent / "grid_0.01_attempt_1"
    assert accepted == coarse and verified == [coarse]
    proof = json.loads((fine / "calibration_stall.json").read_text())
    assert proof["during_calibration"]
    assert proof["last_event"]["event"] == "sigma_evaluation_start"
    assert proof["elapsed_without_progress_seconds"] >= proof["timeout_seconds"]
    assert len(list(parent.glob("grid_0.01_attempt_*"))) == 1


def test_ordinary_calibration_error_never_uses_coarse_grid(monkeypatch, tmp_path):
    parent, launched, verified, accepted = exercise_fallback(monkeypatch, tmp_path, "error")
    assert accepted is None and verified == []
    assert [grid for grid, _, _ in launched] == [.001]
    assert not list(parent.glob("grid_0.01_attempt_*"))
    assert not (parent / "grid_0.001_attempt_1" / "calibration_stall.json").exists()


def synthetic_rows(common, design):
    dataset_details = common.dataset_map(design)
    display_methods = {
        "dpgnn": "DP-GNN", "dpmlp": "DP-MLP", "dpar": "DPAR", "progap": "ProGAP",
        "mlp": "Nonprivate MLP", "graphsage": "Nonprivate GraphSAGE",
    }
    rows = []
    for cell in common.expand_matrix(design):
        sparse = cell["family"] == "sparse"
        validation = .7 if sparse else .6
        test = (.1 if sparse and cell["sampling_mode"] == "expected_batch_1024" else
                .99 if sparse else .6)
        row = {
            "scientific_key": list(common.scientific_key(cell)),
            "family": cell["family"], "dataset": cell["dataset"],
            "method": ("SparseGNN" if sparse else
                       display_methods[cell["method"]] if cell["family"] == "nonprivate"
                       else cell["method"]),
            "target_epsilon": cell["epsilon"], "delta": dataset_details[cell["dataset"]]["delta"],
            "epsilon": cell["epsilon"] if cell["epsilon"] is not None else None,
            "validation": validation, "test": test,
            "metric": dataset_details[cell["dataset"]]["metric"],
            "training_seconds": 1.0, "wall_seconds": 2.0, "seed": 0,
            "train_nodes": dataset_details[cell["dataset"]]["training_nodes"],
            "gpu": 1, "sigma": 1.0, "best_step": 50,
            "trial_path": "synthetic", "config_path": "synthetic/config.json",
            "result_path": "synthetic/result.json", "verification_path": "synthetic/verification.json",
        }
        if sparse:
            schedule = design["sparse"].get("resolved_schedules", {}).get(
                cell["dataset"], {}).get(cell["sampling_mode"], {})
            row.update(sampling_mode=cell["sampling_mode"], p1=cell["p1"],
                       expected_batch=cell["expected_batch"], p2=cell["p2"], r=cell["r"],
                       K=10, hidden=design["sparse"]["hidden"],
                       epochs=design["sparse"].get("epochs"),
                       updates=schedule.get("updates"), accounting_grid=.001)
        rows.append(row)
    return rows


def test_synthetic_report_uses_validation_only_tie_order():
    design = study_common.load_design()
    rows = synthetic_rows(study_common, design)
    verification = {"status": "complete", "accepted": 104, "expected": 104,
                    "fallbacks": [], "family_counts": {}}
    markdown, payload = study_summary.render_report(rows, design, verification)
    assert len(payload["records"]) == 104
    assert len(payload["selected_sparse"]) == 8
    assert all(row["sampling_mode"] == "expected_batch_1024"
               and row["p2"] == .5 and row["r"] == 1
               for row in payload["selected_sparse"])
    assert "## Epsilon 2 headline" in markdown
    assert "## Epsilon 8 headline" in markdown
    assert "PLD grid" in markdown
    assert "## Full run ledger" in markdown


def test_normalized_baseline_only_design_and_report_contract():
    design = normalized_common.load_design()
    cells = normalized_common.expand_matrix(design)
    keys = {normalized_common.scientific_key(cell) for cell in cells}
    assert len(cells) == len(keys) == 40
    assert design["physical_gpus"] == list(range(8))
    assert not design["sparse"]["enabled"]
    assert design["sparse"]["expected_cells"] == 0
    assert not any(cell["family"] == "sparse" for cell in cells)
    assert design["private_baselines"]["dpgnn"]["hidden"] == 64
    assert design["private_baselines"]["dpgnn"]["max_degree"] == 10
    assert design["private_baselines"]["progap"]["batch_size"] == 256
    assert design["private_baselines"]["progap"]["max_degree"] == 10
    assert design["nonprivate"]["dropout"] == 0.1
    assert design["nonprivate"]["root_batch_size"] == 1024
    assert design["nonprivate"]["epochs"] == 100
    assert design["nonprivate"]["graphsage_sampler"] == "exact_root_local_full_neighbor_csr"
    assert design["nonprivate"]["graphsage_full_neighbor"] is True
    assert design["features"]["privacy_treatment"] == "public/fixed preprocessing"
    rows = synthetic_rows(normalized_common, design)
    verification = {"status": "complete", "accepted": 40, "expected": 40,
                    "fallbacks": [], "family_counts": {}}
    markdown, payload = normalized_summary.render_report(rows, design, verification)
    assert len(payload["records"]) == 40
    assert "Normalized Baseline Comparison" in markdown
    assert "SparseGNN is excluded" in markdown
    assert "| SparseGNN |" not in markdown
    assert "upstream batch 256" in markdown
    assert design["features"]["normalization"] in markdown
