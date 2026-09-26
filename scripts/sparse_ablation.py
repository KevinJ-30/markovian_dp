#!/usr/bin/env python3
"""Validate and plot completed SparseExpand ablations without launching training.

Usage: python scripts/sparse_ablation.py --ofat-root ROOT [--depth-root ROOT]
                                      [--out-dir NEW_DIRECTORY]

All 60 epsilon-8, seed-0 OFAT runs must be present and valid. An optional
depth-baseline root adds all 27 DP-GNN/ProGAP runs. Test error bars use each
validation-selected checkpoint's stored 95% node-percentile bootstrap interval:
test-node uncertainty, not training randomness. Depth comparisons use lines;
the other two parameter panels retain grouped SGNN bars. Resource and sampled-
size observations remain in the CSV exports, not the figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.full_matrix_records import BOOTSTRAP, _expected_parameters, _verify_selection
from scripts.sparse_ablation_grid import configurations, run_relative_path

CONFIG_KEYS = ("protocol", "method", "epsilon", "lr", "batch_size", "epochs", "seed", "p2", "r", "K_out")
TASKS = {
    "ogbn-arxiv": ("ogbn-arxiv", "accuracy", False, False, "native"),
    "saint-yelp": ("saint-yelp", "micro_f1", False, True, "native"),
    "twitch-allbut2": ("twitch-explicit", "auroc", True, False, "domain"),
}
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00")
POLICY = {
    "accounting": "Current repository chi=1, union_safe=False policy; no union-graph correction. This analysis does not establish a corrected privacy guarantee.",
    "p2_equals_one": "p2=1 removes Bernoulli edge thinning only: outgoing-degree preprocessing, root sampling, finite radius, and the incoming expansion cap of 20 remain. It is not full-graph training.",
    "degree_caps": "K_out is the outgoing preprocessing cap. K_in=10 remains an accounting input, not an incoming preprocessing cap; incoming preprocessing degree is unrestricted.",
    "checkpoint": "Best validation-primary-metric checkpoint; no test-based selection or configuration ranking.",
    "per_run_interval": "95% node-percentile bootstrap interval, 1000 resamples, bootstrap seed 0.",
    "uncertainty": "Test-node bootstrap uncertainty conditional on the validation-selected checkpoint; not uncertainty from training randomness. There are no independent training-seed replicates or across-run averages.",
    "depth_comparison": "The x coordinate is SGNN expansion radius (fixed two-layer network), DP-GNN message-passing radius, or ProGAP progressive aggregation depth. These are not identical architectures or training schedules.",
    "dpgnn_accounting": "DP-GNN radius-dependent influence is conditional on the fixed sampled topology and node features/labels adjacency; it does not establish a raw-topology node-deletion guarantee.",
    "diagnostics": "Resource and sampled-size observations are research diagnostics, not additional DP releases covered by epsilon.",
    "missing_values": "Empty CSV fields are unavailable, never zero-imputed. Missing CPU CUDA measurements are explicitly marked and not plotted.",
    "resource_scope": "Calibration/training duration excludes data loading. CUDA allocated-memory peak covers backend calibration/training; RSS is runner-process lifetime high-water mark including loading. These are process/allocation measures, not total machine memory.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    require(actual == expected, f"{label}: expected {expected!r}, found {actual!r}")


def finite(value, label, *, minimum=None, maximum=None):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label}: expected a finite number, found {value!r}")
    if minimum is not None:
        require(value >= minimum, f"{label}: {value!r} is below {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{label}: {value!r} exceeds {maximum}")
    return value


def integer(value, label, *, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{label}: expected an integer >= {minimum}, found {value!r}")
    return value


def read_json(path, expected_type=dict):
    def reject_constant(value):
        raise ValueError(f"{path}: non-finite JSON constant {value}")

    def unique_keys(pairs):
        value = {}
        for key, child in pairs:
            require(key not in value, f"{path}: duplicate JSON key {key!r}")
            value[key] = child
        return value

    value = json.loads(path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys)
    require(isinstance(value, expected_type), f"{path}: expected JSON {expected_type.__name__}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def owned_path(root, relative):
    require(isinstance(relative, str) and bool(relative), "artifact path must be a nonempty string")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts,
            f"artifact path escapes root: {relative!r}")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"artifact symlink escapes root: {path}")
    return path


def config_key(config, keys=CONFIG_KEYS):
    return tuple(config[key] for key in keys)


def recorded_path(root, value, original_root):
    """Resolve an owned stored path, including a complete root moved elsewhere."""
    require(isinstance(value, str) and bool(value), "stored output path must be a nonempty string")
    path = Path(value)
    if path.is_absolute():
        if path.is_relative_to(root):
            relative = str(path.relative_to(root))
        else:
            origin = Path(original_root)
            require(origin.is_absolute() and path.is_relative_to(origin),
                    f"stored output path is outside the recorded input root: {value}")
            relative = str(path.relative_to(origin))
    else:
        relative = value
    return owned_path(root, relative)


def read_manifest(root, study):
    path = root / "manifest.json"
    manifest = read_json(path)
    same(manifest["schema_version"], 1, f"{path}: schema_version")
    same(manifest["study"], study, f"{path}: study")
    fixed = {"hidden": 128, "dropout": 0.5, "split_seed": 0,
             "bootstrap_resamples": 1000, "bootstrap_confidence": 0.95,
             "bootstrap_seed": 0}
    if study == "ofat":
        fixed.update(layers=2, K_in=10, chi=1, union_safe=False)
    for name, value in fixed.items():
        same(manifest["fixed"][name], value, f"{path}: fixed {name}")
    entries = manifest["configurations"]
    require(isinstance(entries, list), f"{path}: configurations must be a list")
    expected = configurations(study)
    by_key = {}
    seen_paths = set()
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"{path}: configuration {index} must be an object")
        same(set(entry), set(CONFIG_KEYS) | {"run_dir"}, f"{path}: configuration {index} keys")
        key = config_key(entry)
        require(key not in by_key, f"{path}: duplicated configuration {key}")
        relative = run_relative_path({name: entry[name] for name in CONFIG_KEYS})
        same(entry["run_dir"], relative, f"{path}: canonical run_dir for configuration {index}")
        run_path = owned_path(root, relative).resolve()
        require(run_path not in seen_paths, f"{path}: duplicated run directory {run_path}")
        seen_paths.add(run_path)
        by_key[key] = entry
    expected_keys = {config_key(config) for config in expected}
    missing, extra = expected_keys - by_key.keys(), by_key.keys() - expected_keys
    require(not missing and not extra,
            f"{path}: canonical {study} grid mismatch: expected {len(expected)} configurations, "
            f"found {len(entries)}; missing={len(missing)}, unexpected={len(extra)}; "
            f"first missing={next(iter(sorted(missing)), None)!r}; first unexpected={next(iter(sorted(extra)), None)!r}")
    provenance = manifest.get("provenance", {})
    source_hashes = manifest.get("source_sha256", {})
    require(isinstance(source_hashes, dict) and bool(source_hashes), f"{path}: source hash commitments missing")
    snapshot = owned_path(root, provenance["source_snapshot"])
    hashes = {str(path): sha256(path)}
    for relative, committed in sorted(source_hashes.items()):
        source_path = owned_path(snapshot, relative)
        actual = sha256(source_path)
        same(actual, committed, f"{source_path}: source snapshot SHA256")
        hashes[str(source_path)] = actual
    for required in ("scripts/full_matrix_run.py", "scripts/sparse_ablation_grid.py",
                     "src/training/sparse_gnn.py", "src/processing/sparse_expand.py",
                     "src/processing/graphs.py", "src/privacy/accounting.py"):
        require(required in source_hashes, f"{path}: missing required source commitment {required}")
    if study == "depth-baselines":
        for required in ("src/training/dpgnn.py", "src/processing/dpgnn.py",
                         "src/privacy/dpgnn.py", "third_party/ProGAP/inductive_adapter.py"):
            require(required in source_hashes, f"{path}: missing baseline source commitment {required}")
    invocations = manifest.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == len(entries),
            f"{path}: exact per-run invocation evidence is missing or incomplete")
    invocation_by_path = {}
    for invocation in invocations:
        relative = invocation["run_dir"]
        require(relative not in invocation_by_path, f"{path}: duplicate invocation {relative}")
        argv = invocation["argv"]
        require(isinstance(argv, list) and bool(argv) and all(isinstance(arg, str) for arg in argv),
                f"{path}: invocation argv must be a nonempty string list: {relative}")
        invocation_by_path[relative] = invocation
    same(set(invocation_by_path), {entry["run_dir"] for entry in entries}, f"{path}: invocation membership")
    states_by_key = None
    state_path = root / "queue_state.json"
    if state_path.exists():
        registry_path = root / "requests.json"
        registry = read_json(registry_path, list)
        states = read_json(state_path, list)
        same(len(registry), len(entries), f"{registry_path}: request count")
        same(len(states), len(entries), f"{state_path}: state count")
        states_by_key = {}
        seen_outputs = set()
        for index, (request, state, entry) in enumerate(zip(registry, states, entries)):
            same(config_key(request), config_key(entry), f"{registry_path}: canonical identity at index {index}")
            require(isinstance(state, dict), f"{state_path}: state {index} must be an object")
            if state.get("status") == "completed":
                directory = recorded_path(root, state["output"], provenance["out_root"])
                require(directory.resolve() not in seen_outputs, f"{state_path}: duplicated completed output {directory}")
                seen_outputs.add(directory.resolve())
            states_by_key[config_key(entry)] = state
        hashes[str(registry_path)] = sha256(registry_path)
        hashes[str(state_path)] = sha256(state_path)
    return {
        "root": root, "study": study, "manifest": manifest, "hashes": hashes,
        "entries": [by_key[config_key(config)] for config in expected],
        "invocations": invocation_by_path, "source_sha256": source_hashes,
        "queue_states": states_by_key,
    }


def verify_csv(path, result):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields is not None and len(fields) == len(set(fields)), f"{path}: invalid/duplicate CSV headers")
        rows = list(reader)
    require(len(rows) == 1, f"{path}: expected exactly one final result row, found {len(rows)}")
    row = rows[0]
    require(None not in row and all(value is not None for value in row.values()), f"{path}: malformed CSV row")
    required = {
        "protocol", "dataset", "method", "metric", "target_epsilon", "seed", "lr", "epochs",
        "requested_batch_size", "batch_size", "effective_batch_size", "hidden", "dropout",
        "test_metric", "validation_metric", "parameters", "selection", "test_confidence_intervals",
        "completed_epochs", "status",
    }
    require(required <= row.keys(), f"{path}: missing CSV columns {sorted(required - row.keys())}")
    for name, actual in row.items():
        require(name in result, f"{path}: CSV field {name!r} is absent from result.json")
        expected = result[name]
        if isinstance(expected, (dict, list)):
            actual = json.loads(actual)
        elif expected is None:
            same(actual, "", f"{path}: CSV {name}")
            continue
        elif isinstance(expected, bool):
            same(actual, str(expected), f"{path}: CSV {name}")
            continue
        elif isinstance(expected, (int, float)):
            actual = float(actual)
            finite(actual, f"{path}: CSV {name}")
        same(actual, expected, f"{path}: CSV/result {name}")


def read_run(bundle, expected):
    root, study = bundle["root"], bundle["study"]
    sparse = expected["method"].startswith("sparse_")
    progap = expected["method"] == "progap"
    state = None if bundle["queue_states"] is None else bundle["queue_states"][config_key(expected)]
    if state is None:
        directory = owned_path(root, expected["run_dir"])
    else:
        same(state["status"], "completed", f"queue state for {expected['run_dir']}; reason={state.get('reason')!r}")
        directory = recorded_path(root, state["output"], bundle["manifest"]["provenance"]["out_root"])
    paths = {name: directory / name for name in ("config.json", "result.json", "result.csv", "worker_exit.json")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, "incomplete run; missing " + ", ".join(missing))
    for path in paths.values():
        require(path.resolve().is_relative_to(root.resolve()), f"artifact symlink escapes input root: {path}")
    hashes = {str(path): sha256(path) for path in paths.values()}
    marker = read_json(paths["worker_exit.json"])
    same(marker["status"], "completed", "worker commitment status")
    committed = marker["artifact_sha256"]
    same(set(committed), {"config.json", "result.json", "result.csv"}, "worker committed artifact set")
    for name, digest in committed.items():
        same(hashes[str(paths[name])], digest, f"{paths[name]}: worker commitment SHA256")
    config, result = read_json(paths["config.json"]), read_json(paths["result.json"])
    same(result["status"], "completed", "result status")
    same(config["parameters"], result["parameters"], "config/result parameters")
    dataset, metric, binary, multilabel, strategy = TASKS[expected["protocol"]]
    task = config["task"]
    for name, value in {"primary_metric": metric, "binary": binary, "multilabel": multilabel,
                        "regression": False}.items():
        same(task[name], value, f"task {name}")
    population = integer(config["train_nodes"], "train_nodes", minimum=1)
    batch = min(population, expected["batch_size"])
    interval = population // batch if progap else math.ceil(population / batch)
    stages = expected["r"] + 1 if progap else 1
    steps = stages * expected["epochs"] * interval
    delta = 1 / population
    identity = {
        "protocol": expected["protocol"], "dataset": dataset, "method": expected["method"],
        "target_epsilon": expected["epsilon"], "target_delta": delta,
        "seed": expected["seed"], "lr": expected["lr"], "epochs": expected["epochs"],
        "requested_batch_size": expected["batch_size"], "batch_size": batch,
        "effective_batch_size": batch, "hidden": 128, "dropout": 0.5, "dp": True,
        "metric": metric, "split_seed": 0, "split_strategy": strategy, "split": f"{strategy}:seed0",
        "train_nodes": population, "steps": steps, "weight_decay": 0.0 if progap else 5e-4,
        "architecture": "progap" if progap else "gin" if expected["method"].endswith("gin") else "graphsage",
    }
    for name, value in identity.items():
        same(config[name], value, f"config {name}")
        same(result[name], value, f"result {name}")
    for name in ("domain_split", "domain_split_id", "split_file", "device", "epsilon", "delta"):
        same(config[name], result[name], f"config/result {name}")
    requested = config["requested"]
    requested_values = {
        "dataset": expected["protocol"], "method": expected["method"], "epsilon": expected["epsilon"],
        "lr": expected["lr"], "batch_size": expected["batch_size"], "epochs": expected["epochs"],
        "seed": expected["seed"], "p2": expected["p2"], "gnn_hidden": 128, "dropout": 0.5,
        "bootstrap_resamples": 1000,
    }
    if sparse:
        requested_values.update(sparse_radius=expected["r"], sparse_degree_cap=expected["K_out"])
    else:
        requested_values["progap_depth" if progap else "dpgnn_radius"] = expected["r"]
    for name, value in requested_values.items():
        same(requested[name], value, f"requested {name}")
    parameters, native = result["parameters"], result["native_result"]
    fixed_parameters = {
        "architecture": "gin" if expected["method"] == "sparse_gin" else "mean",
        "hidden": 128, "layers": 2, "lr": expected["lr"], "batch_size": batch,
        "epochs": expected["epochs"], "steps": steps, "evaluate_every": interval,
        "dropout": 0.5, "optimizer": "adam", "weight_decay": 5e-4,
        "p1": batch / population, "p2": expected["p2"], "r": expected["r"],
        "K_in": 10, "K_out": expected["K_out"], "clip": 1.0, "cap_mode": "directed",
        "cap_seed": expected["seed"] + 20_000, "direction": "in", "chi": 1,
        "incoming_sampling_cap": 20,
        "union_safe": False, "accounting_grid": 1e-3, "calibration_rtol": 1e-3,
        "calibration_atol": 1e-6, "bootstrap_confidence": 0.95, "bootstrap_resamples": 1000,
        "bootstrap_seed": 0, "binary": binary, "multilabel": multilabel, "regression": False,
        "metric_ignore_label": task["metric_ignore_label"],
    } if sparse else _expected_parameters(
        {**expected, "dropout": 0.5, "gnn_hidden": 128, "mlp_hidden": 64}, population)
    for name, value in fixed_parameters.items():
        same(parameters[name], value, f"actual parameter {name}")
    if sparse:
        require(integer(parameters["K_out_achieved"], "K_out_achieved") <= expected["K_out"],
                "achieved outgoing degree exceeds preprocessing cap")
        integer(parameters["K_in_achieved"], "K_in_achieved")
    for name in ("test_metric", "validation_metric", f"test_{metric}"):
        finite(result[name], name, minimum=0, maximum=1)
    same(result[f"test_{metric}"], result["test_metric"], "primary score")
    native_score_keys = [f"test_{metric}"] + (["test_accuracy"] if metric != "auroc" else []) + ["test"]
    score_key = next((key for key in native_score_keys if key in native), None)
    require(score_key is not None, "native test primary metric is absent")
    same(native[score_key], result["test_metric"], "native test primary metric")
    _verify_selection({**expected, "task": task}, result, parameters)
    privacy = native["privacy"]["total"] if progap else native["privacy"]
    if sparse:
        same(privacy["accountant"], "src.privacy.accounting.sparsegnn_mixture_weights.chi1", "accountant")
        for name, value in {"p1": batch / population, "p2": expected["p2"], "r": expected["r"],
                            "K_in": 10, "K_out": expected["K_out"], "chi": 1,
                            "union_safe": False, "grid": 1e-3}.items():
            same(privacy["parameters"][name], value, f"accounting parameter {name}")
    elif progap:
        same(privacy["accountant"], "upstream.ProGAP.ComposedNoisyMechanism", "accountant")
        for name, value in {"depth": expected["r"], "max_degree": 5,
                            "batch_size": batch, "train_nodes": population,
                            "component_coefficients": [expected["r"], stages]}.items():
            same(privacy["parameters"][name], value, f"accounting parameter {name}")
    else:
        same(privacy["accountant"], "src.privacy.dpgnn.multiterm_dpsgd_epsilon", "accountant")
        same(privacy["parameters"]["max_terms"], parameters["max_terms"], "accounted sensitivity")
        same(privacy["parameters"]["radius"], expected["r"], "accounted DP-GNN radius")
        same(privacy["parameters"]["max_degree"], 5, "accounted max degree")
        same(privacy["parameters"]["sampling"], "uniform_without_replacement", "accounted sampling")
        same(privacy["parameters"]["opacus_noise_multiplier"],
             2 * parameters["max_terms"] * parameters["noise_multiplier"], "effective noise multiplier")
    same(privacy["composition_count"], expected["r"] + stages if progap else steps,
         "accounted full training schedule")
    same(privacy["sampling_probability"], batch / population, "accounted root sampling probability")
    epsilon = finite(privacy["epsilon"], "actual epsilon", minimum=0, maximum=expected["epsilon"] + 1e-6)
    same(privacy["delta"], delta, "actual delta")
    same(result["epsilon"], epsilon, "result actual epsilon")
    same(result["delta"], delta, "result actual delta")
    sigma = finite(privacy["noise_multiplier"], "noise multiplier", minimum=0)
    require(sigma > 0, "noise multiplier must be positive")
    calibration = native["calibration"]
    calibration_values = (
        {"noise_std": sigma, "achieved_epsilon": epsilon,
         "target_epsilon": expected["epsilon"], "target_delta": delta}
        if progap else
        {"noise_multiplier": sigma, "epsilon" if sparse else "achieved_epsilon": epsilon,
         "target_epsilon": expected["epsilon"], "delta" if sparse else "target_delta": delta})
    for name, value in calibration_values.items():
        same(calibration[name], value, f"noise calibration {name}")
    if not progap:
        same(parameters["sigma" if sparse else "noise_multiplier"], sigma, "training noise multiplier")
    if sparse:
        integer(calibration["evaluations"], "calibration evaluations", minimum=1)
    ci = result["test_confidence_intervals"]
    same(ci, native["test_confidence_intervals"], "native/result confidence intervals")
    for name, value in BOOTSTRAP.items():
        same(ci[name], value, f"bootstrap {name}")
    integer(ci["n_observations"], "bootstrap observation count", minimum=1)
    require(metric in ci["metrics"], f"bootstrap primary interval {metric!r} is unavailable")
    primary = ci["metrics"][metric]
    valid = integer(primary["valid_resamples"], "valid bootstrap resamples", minimum=1)
    require(valid <= BOOTSTRAP["n_resamples"], "valid bootstrap resamples exceed requested resamples")
    lower = finite(primary["lower"], "bootstrap lower", minimum=0, maximum=1)
    upper = finite(primary["upper"], "bootstrap upper", minimum=0, maximum=1)
    require(lower <= upper, "bootstrap interval is reversed")
    duration = finite(result["calibration_and_training_seconds"], "calibration/training seconds", minimum=0)
    resources = result["resources"]
    rss = integer(resources["peak_rss_bytes"], "peak RSS bytes", minimum=1)
    device = result["device"].split(":", 1)[0]
    require(device in {"cpu", "cuda"}, f"unsupported resource device {result['device']!r}")
    cuda = resources["peak_cuda_allocated_bytes"]
    if device == "cpu":
        same(cuda, None, "CPU CUDA memory must be unavailable, not zero")
    else:
        integer(cuda, "peak CUDA allocated bytes", minimum=0 if progap else 1)
    sampling = native.get("sampling_statistics", {
        "count": 0, "mean_nodes": None, "mean_edges": None, "max_nodes": 0, "max_edges": 0})
    count = integer(sampling["count"], "sampled subgraph count")
    for kind in ("nodes", "edges"):
        maximum = integer(sampling[f"max_{kind}"], f"max sampled {kind}")
        mean = sampling[f"mean_{kind}"]
        if count == 0:
            same(mean, None, f"mean sampled {kind} for no samples")
            same(maximum, 0, f"max sampled {kind} for no samples")
        else:
            finite(mean, f"mean sampled {kind}", minimum=1 if kind == "nodes" else 0, maximum=maximum)
    verify_csv(paths["result.csv"], result)
    invocation = bundle["invocations"][expected["run_dir"]]
    actual_argv = invocation["argv"]
    launch_path = None
    if state is not None:
        launch_path = directory.parent / "launch.json"
        require(launch_path.resolve().is_relative_to(root.resolve()), f"launch symlink escapes input root: {launch_path}")
        launch = read_json(launch_path)
        actual_argv = launch["command"]
        hashes[str(launch_path)] = sha256(launch_path)
    require(isinstance(actual_argv, list) and all(isinstance(arg, str) for arg in actual_argv),
            "actual worker command must be a string list")
    command_values = {
        "--dataset": expected["protocol"], "--method": expected["method"], "--lr": expected["lr"],
        "--batch-size": expected["batch_size"], "--epochs": expected["epochs"], "--seed": expected["seed"],
        "--epsilon": expected["epsilon"], "--gnn-hidden": 128, "--dropout": 0.5,
        "--bootstrap-resamples": 1000,
    }
    if sparse:
        command_values.update({"--p2": expected["p2"], "--sparse-radius": expected["r"],
                               "--sparse-degree-cap": expected["K_out"]})
    else:
        command_values["--progap-depth" if progap else "--dpgnn-radius"] = expected["r"]
    for flag, value in command_values.items():
        same(actual_argv.count(flag), 1, f"worker command {flag} occurrence count")
        position = actual_argv.index(flag) + 1
        require(position < len(actual_argv), f"worker command value missing for {flag}")
        actual = actual_argv[position] if isinstance(value, str) else float(actual_argv[position])
        same(actual, value, f"worker command {flag}")
    same(actual_argv.count("--out-dir"), 1, "worker command output directory flag")
    position = actual_argv.index("--out-dir") + 1
    require(position < len(actual_argv), "worker command output directory value missing")
    same(actual_argv[position], requested["out_dir"], "worker command/requested output directory")
    same(recorded_path(root, requested["out_dir"], bundle["manifest"]["provenance"]["out_root"]).resolve(),
         directory.resolve(), "actual worker output directory")
    row = {
        **{name: expected[name] for name in CONFIG_KEYS}, "study": study, "metric": metric,
        "actual_epsilon": epsilon, "delta": delta, "effective_batch_size": batch,
        "hidden": 128, "layers": 2 if sparse else None, "dropout": 0.5,
        "K_in": 10 if sparse else None, "chi": 1 if sparse else None,
        "union_safe": False if sparse else None,
        "device": result["device"], "selected_step": result["selection"]["step"],
        "selected_epoch": result["selection"].get("epoch", result["selection"]["step"] // interval),
        "validation_metric": result["validation_metric"], "test_metric": result["test_metric"],
        "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
        "bootstrap_confidence_level": ci["confidence_level"], "bootstrap_method": ci["method"],
        "bootstrap_n_resamples": ci["n_resamples"], "bootstrap_valid_resamples": valid,
        "bootstrap_seed": ci["seed"], "bootstrap_resampling_unit": ci["resampling_unit"],
        "bootstrap_n_observations": ci["n_observations"],
        "calibration_and_training_seconds": duration, "peak_cuda_allocated_bytes": cuda,
        "peak_rss_bytes": rss, "cuda_memory_status": "unavailable_cpu" if cuda is None else "available",
        "sampling_count": count, "mean_nodes": sampling["mean_nodes"], "mean_edges": sampling["mean_edges"],
        "max_nodes": sampling["max_nodes"], "max_edges": sampling["max_edges"],
        "sampling_status": "available" if count else "unavailable_no_sampled_subgraphs",
        "input_root": str(root), "manifest_path": str(root / "manifest.json"), "run_dir": expected["run_dir"],
        "actual_output_path": str(directory), "queue_attempt": None if state is None else state["attempt"],
        "recorded_output_path": requested["out_dir"], "launch_json_path": str(launch_path) if launch_path else None,
        **{name.replace(".", "_") + "_path": str(path) for name, path in paths.items()},
        **{name.replace(".", "_") + "_sha256": hashes[str(path)] for name, path in paths.items()},
        "invocation_argv": actual_argv, "planned_invocation_argv": invocation["argv"], "parameters": parameters,
        "selection": result["selection"], "test_confidence_intervals": ci, "resources": resources,
    }
    split_signature = {
        "task": task, "train_nodes": population, "split_strategy": strategy,
        "domain_split": config["domain_split"], "domain_split_id": config["domain_split_id"],
        "partitions": config["partitions"], "test_observations": ci["n_observations"],
    }
    return row, hashes, split_signature


def relationship_curves(rows):
    curves = []
    for protocol, method, epsilon in dict.fromkeys((row["protocol"], row["method"], row["epsilon"]) for row in rows):
        subset = [row for row in rows if (row["protocol"], row["method"], row["epsilon"]) == (protocol, method, epsilon)]
        sweeps = (
            ("r", {"p2": 0.5, "K_out": 10}, (1, 2, 3)),
            ("p2", {"r": 1, "K_out": 10}, (0.05, 0.1, 0.25, 0.5, 1.0)),
            ("K_out", {"r": 1, "p2": 0.5}, (5, 10, 20, 40)),
        ) if method.startswith("sparse_") else (("r", {}, (1, 2, 3)),)
        for parameter, fixed, expected_values in sweeps:
            selected = sorted((row for row in subset if all(row[key] == value for key, value in fixed.items())),
                              key=lambda row: row[parameter])
            same(tuple(row[parameter] for row in selected), expected_values,
                 f"OFAT curve membership: {protocol}/{method}/epsilon{epsilon}/{parameter}")
            for row in selected:
                curves.append({"curve_parameter": parameter, "curve_value": row[parameter],
                               "fixed_parameters": fixed, **row})
    return curves


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def draw_figures(curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.size": 18,
                         "axes.labelsize": 22, "axes.titlesize": 21,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    datasets = (
        ("saint-yelp", "Yelp", COLORS[1]),
        ("ogbn-arxiv", "ArXiv", COLORS[0]),
        ("twitch-allbut2", "Twitch", COLORS[2]),
    )
    parameters = (
        ("r", "Expansion / propagation depth"),
        ("p2", "Edge-retention probability $p_2$"),
        ("K_out", r"Outgoing-degree cap $K_{\mathrm{out}}$"),
    )
    outputs = []
    for method, backend in (("sparse_sage", "SAGE"), ("sparse_gin", "GIN")):
        epsilons = {row["epsilon"] for row in curves if row["method"] == method}
        require(len(epsilons) == 1, f"expected one privacy parameter for {backend}")
        figure, axes = plt.subplots(1, 3, figsize=(18.7, 3.6), sharey=True)
        families = [(method, "SGNN", "-")]
        for baseline, label, style in (
            ("progap", "ProGAP", "--"),
            ("dp_gnn_gin" if backend == "GIN" else "dp_gnn_sage", "DP-GNN", ":"),
        ):
            if any(row["method"] == baseline for row in curves):
                families.append((baseline, label, style))
        width = 0.24
        for ax, (parameter, xlabel), panel_label in zip(axes, parameters, ("(a)", "(b)", "(c)")):
            for dataset_index, (protocol, _, color) in enumerate(datasets):
                panel_methods = families if parameter == "r" else [(method, "SGNN", "-")]
                for curve_method, _, style in panel_methods:
                    selected = sorted(
                        (row for row in curves if row["method"] == curve_method
                         and row["protocol"] == protocol and row["curve_parameter"] == parameter),
                        key=lambda row: row["curve_value"])
                    require(bool(selected), f"missing curve: {curve_method}/{protocol}/{parameter}")
                    centers = list(range(len(selected)))
                    if parameter == "r":
                        xs = [row["curve_value"] for row in selected]
                        ax.plot(xs, [row["test_metric"] for row in selected],
                                color=color, linestyle=style, linewidth=3, alpha=1,
                                marker="o", markersize=4, zorder=3)
                    else:
                        xs = [center + (dataset_index - 1) * width for center in centers]
                        ax.bar(xs, [row["test_metric"] for row in selected], width=width,
                               color=color, edgecolor="white", linewidth=0.6, zorder=3)
                        for x, row in zip(xs, selected):
                            # Absolute endpoints need not bracket the empirical score.
                            lower, upper = row["bootstrap_ci_lower"], row["bootstrap_ci_upper"]
                            ax.vlines(x, lower, upper, color="0.15", linewidth=1.3, zorder=4)
                            ax.hlines((lower, upper), x - width * 0.25, x + width * 0.25,
                                      color="0.15", linewidth=1.3, zorder=4)
            if parameter == "r":
                ax.set_xticks((1, 2, 3))
                ax.set_xlim(0.85, 3.15)
            else:
                ax.set_xticks(centers, [f"{row['curve_value']:g}" for row in selected])
                ax.set_xlim(-0.6, len(selected) - 0.4)
            ax.set(xlabel=xlabel, ylim=(0, 1))
            ax.text(0.025, 0.95, panel_label, transform=ax.transAxes,
                    ha="left", va="top", fontsize=18, fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(which="major", color="0.88", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Test metric")
        figure.legend(
            handles=[*[Patch(facecolor=color, label=label) for _, label, color in datasets],
                     *[Line2D([], [], color="0.2", linestyle=style, linewidth=3, alpha=1, label=label)
                       for _, label, style in families]],
            loc="center left", bbox_to_anchor=(0.005, 0.55), borderaxespad=0,
            ncol=1, fontsize=14, frameon=False, handlelength=2.4)
        figure.subplots_adjust(left=0.175, right=0.99, bottom=0.24, top=0.84, wspace=0.12)
        for extension in ("png", "pdf"):
            path = out_dir / f"ablation_{backend.lower()}.{extension}"
            metadata = ({"CreationDate": None, "ModDate": None} if extension == "pdf"
                        else {"Software": "SparseExpand ablation analysis"})
            figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15, metadata=metadata)
            outputs.append(path)
        plt.close(figure)
    return outputs


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--ofat-root", required=True, type=Path)
    cli.add_argument("--depth-root", type=Path,
                     help="complete 27-run DP-GNN/ProGAP depth-baseline study")
    cli.add_argument("--out-dir", type=Path,
                     help="fresh figure/data directory (default: DEPTH_ROOT/figures, else OFAT_ROOT/figures)")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    argument_values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    root = args.ofat_root.expanduser().absolute()
    depth_root = args.depth_root.expanduser().absolute() if args.depth_root else None
    out_dir = args.out_dir.expanduser().absolute() if args.out_dir else (depth_root or root) / "figures"
    argument_values["out_dir"] = str(out_dir)
    try:
        require(not out_dir.exists() and not out_dir.is_symlink(), f"output directory already exists: {out_dir}; refusing overwrite")
        inputs = [(root, "ofat")] + ([(depth_root, "depth-baselines")] if depth_root else [])
        bundles = []
        rows, all_hashes, errors, split_signatures = [], {}, [], {}
        for input_root, study in inputs:
            require(input_root.is_dir(), f"input root is not a directory: {input_root}")
            require(not input_root.resolve().is_relative_to(out_dir.resolve()),
                    f"output directory cannot replace the input root or its ancestors: {input_root}")
            bundle = read_manifest(input_root, study)
            bundles.append(bundle)
            all_hashes.update(bundle["hashes"])
            for entry in bundle["entries"]:
                try:
                    row, hashes, signature = read_run(bundle, entry)
                    if entry["protocol"] in split_signatures:
                        same(signature, split_signatures[entry["protocol"]],
                             "incompatible dataset/task/split evidence across runs")
                    else:
                        split_signatures[entry["protocol"]] = signature
                    rows.append(row)
                    all_hashes.update(hashes)
                except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                    errors.append(f"{input_root / entry['run_dir']}: {type(error).__name__}: {error}")
        if errors:
            details = "\n".join(errors[:20])
            omitted = f"\n... {len(errors) - 20} additional invalid/incomplete runs" if len(errors) > 20 else ""
            raise ValueError(f"strict analysis rejected {len(errors)} invalid/incomplete runs; {len(rows)} valid. "
                             f"No output was written.\n{details}{omitted}")
        same(len(rows), 87 if depth_root else 60, "complete validated run count")
        same(len({config_key(row) for row in rows}), len(rows), "unique configuration count")
        curves = relationship_curves(rows)
        # Resolve plotting dependencies before reserving the fresh output directory.
        versions = {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")}
        import matplotlib
        out_dir.mkdir(parents=True, exist_ok=False)
        write_csv(out_dir / "per_run.csv", rows)
        write_csv(out_dir / "curves.csv", curves)
        figures = draw_figures(curves, out_dir)
        source_paths = [Path(__file__).resolve(), ROOT / "scripts/sparse_ablation_grid.py",
                        ROOT / "scripts/full_matrix_records.py", ROOT / "scripts/full_matrix_runtime.py"]
        provenance = {
            "schema_version": 1, "analysis": "SparseExpand OFAT and depth comparisons" if depth_root else "SparseExpand OFAT",
            "completeness": "strict_complete",
            "arguments": argument_values, "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
            "input_run_count": len(rows), "curve_row_count": len(curves),
            "curve_membership": "The SGNN anchor is referenced in three OFAT curves. Each baseline run appears once in the depth data; ProGAP is shared between backend figures. No averaging.",
            "input_root": str(root), "depth_root": str(depth_root) if depth_root else None,
            "input_sha256": dict(sorted(all_hashes.items())),
            "plot_code_sha256": sha256(Path(__file__)),
            "analysis_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
            "versions": {"python": platform.python_version(), **versions},
            "policy": POLICY, "split_evidence": split_signatures,
            "manifest_provenance": {bundle["study"]: bundle["manifest"]["provenance"] for bundle in bundles},
            "output_sha256": {path.name: sha256(path) for path in [out_dir / "per_run.csv", out_dir / "curves.csv", *figures]},
        }
        with (out_dir / "provenance.json").open("x") as stream:
            json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Validated {len(rows)} runs; {len(curves)} curve points; {len(figures)} PNG/PDF files: {out_dir}")
        print("Metric bars: stored 95% node-bootstrap intervals, not training-randomness uncertainty.")
        print(POLICY["accounting"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, importlib.metadata.PackageNotFoundError) as error:
        cli.exit(2, f"sparse_ablation: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
