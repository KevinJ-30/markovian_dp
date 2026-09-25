#!/usr/bin/env python3
"""Validate and plot completed SparseExpand ablations without launching training.

Usage: python scripts/sparse_ablation.py --ofat-root ROOT [--out-dir NEW_DIRECTORY]

All 60 epsilon-8, seed-0 OFAT runs must be present and valid. Test error bars
use each validation-selected checkpoint's stored 95% node-percentile bootstrap
interval: test-node uncertainty, not training randomness. One comparison chart
per backend groups dataset bars in three horizontal parameter panels. Resource
and sampled-size observations remain in the CSV exports, not the figures.
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
from scripts.full_matrix_records import BOOTSTRAP, _verify_selection
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
    for name, value in {"hidden": 128, "dropout": 0.5, "layers": 2, "K_in": 10,
                        "chi": 1, "union_safe": False, "split_seed": 0,
                        "bootstrap_resamples": 1000, "bootstrap_confidence": 0.95,
                        "bootstrap_seed": 0}.items():
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
    interval = math.ceil(population / batch)
    steps = expected["epochs"] * interval
    delta = 1 / population
    identity = {
        "protocol": expected["protocol"], "dataset": dataset, "method": expected["method"],
        "target_epsilon": expected["epsilon"], "target_delta": delta,
        "seed": expected["seed"], "lr": expected["lr"], "epochs": expected["epochs"],
        "requested_batch_size": expected["batch_size"], "batch_size": batch,
        "effective_batch_size": batch, "hidden": 128, "dropout": 0.5, "dp": True,
        "metric": metric, "split_seed": 0, "split_strategy": strategy, "split": f"{strategy}:seed0",
        "train_nodes": population, "steps": steps, "weight_decay": 5e-4,
        "architecture": "gin" if expected["method"] == "sparse_gin" else "graphsage",
    }
    for name, value in identity.items():
        same(config[name], value, f"config {name}")
        same(result[name], value, f"result {name}")
    for name in ("domain_split", "domain_split_id", "split_file", "device", "epsilon", "delta"):
        same(config[name], result[name], f"config/result {name}")
    requested = config["requested"]
    for name, value in {
        "dataset": expected["protocol"], "method": expected["method"], "epsilon": expected["epsilon"],
        "lr": expected["lr"], "batch_size": expected["batch_size"], "epochs": expected["epochs"],
        "seed": expected["seed"], "p2": expected["p2"], "sparse_radius": expected["r"],
        "sparse_degree_cap": expected["K_out"], "gnn_hidden": 128, "dropout": 0.5,
        "bootstrap_resamples": 1000,
    }.items():
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
    }
    for name, value in fixed_parameters.items():
        same(parameters[name], value, f"actual parameter {name}")
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
    privacy = native["privacy"]
    same(privacy["accountant"], "src.privacy.accounting.sparsegnn_mixture_weights.chi1", "accountant")
    same(privacy["composition_count"], steps, "accounted full training schedule")
    same(privacy["sampling_probability"], batch / population, "accounted root sampling probability")
    for name, value in {"p1": batch / population, "p2": expected["p2"], "r": expected["r"],
                        "K_in": 10, "K_out": expected["K_out"], "chi": 1,
                        "union_safe": False, "grid": 1e-3}.items():
        same(privacy["parameters"][name], value, f"accounting parameter {name}")
    epsilon = finite(privacy["epsilon"], "actual epsilon", minimum=0, maximum=expected["epsilon"] + 1e-6)
    same(privacy["delta"], delta, "actual delta")
    same(result["epsilon"], epsilon, "result actual epsilon")
    same(result["delta"], delta, "result actual delta")
    sigma = finite(parameters["sigma"], "noise multiplier", minimum=0)
    require(sigma > 0, "noise multiplier must be positive")
    same(privacy["noise_multiplier"], sigma, "accounted noise multiplier")
    calibration = native["calibration"]
    for name, value in {"noise_multiplier": sigma, "epsilon": epsilon, "target_epsilon": expected["epsilon"],
                        "delta": delta}.items():
        same(calibration[name], value, f"noise calibration {name}")
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
        integer(cuda, "peak CUDA allocated bytes", minimum=1)
    sampling = native["sampling_statistics"]
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
    for flag, value in {
        "--dataset": expected["protocol"], "--method": expected["method"], "--lr": expected["lr"],
        "--batch-size": expected["batch_size"], "--epochs": expected["epochs"], "--seed": expected["seed"],
        "--epsilon": expected["epsilon"], "--p2": expected["p2"], "--sparse-radius": expected["r"],
        "--sparse-degree-cap": expected["K_out"], "--gnn-hidden": 128, "--dropout": 0.5,
        "--bootstrap-resamples": 1000,
    }.items():
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
        "hidden": 128, "layers": 2, "dropout": 0.5, "K_in": 10, "chi": 1, "union_safe": False,
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
        for parameter, fixed, expected_values in (
            ("r", {"p2": 0.5, "K_out": 10}, (1, 2, 3)),
            ("p2", {"r": 1, "K_out": 10}, (0.05, 0.1, 0.25, 0.5, 1.0)),
            ("K_out", {"r": 1, "p2": 0.5}, (5, 10, 20, 40)),
        ):
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
        ("r", "Expansion depth $r$"),
        ("p2", "Edge-retention probability $p_2$"),
        ("K_out", r"Outgoing-degree cap $K_{\mathrm{out}}$"),
    )
    outputs = []
    for method, backend in (("sparse_sage", "SAGE"), ("sparse_gin", "GIN")):
        epsilons = {row["epsilon"] for row in curves if row["method"] == method}
        require(len(epsilons) == 1, f"expected one privacy parameter for {backend}")
        epsilon = next(iter(epsilons))
        figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
        width = 0.24
        for ax, (parameter, xlabel) in zip(axes, parameters):
            for dataset_index, (protocol, _, color) in enumerate(datasets):
                selected = sorted(
                    (row for row in curves if row["method"] == method
                     and row["protocol"] == protocol and row["curve_parameter"] == parameter),
                    key=lambda row: row["curve_value"])
                centers = list(range(len(selected)))
                xs = [center + (dataset_index - 1) * width for center in centers]
                ax.bar(xs, [row["test_metric"] for row in selected], width=width,
                       color=color, edgecolor="white", linewidth=0.6,
                       zorder=3)
                for x, row in zip(xs, selected):
                    # Draw absolute stored endpoints: percentile intervals need
                    # not bracket the empirical score, even for bar charts.
                    lower, upper = row["bootstrap_ci_lower"], row["bootstrap_ci_upper"]
                    ax.vlines(x, lower, upper, color="0.15", linewidth=1.6, zorder=4)
                    ax.hlines((lower, upper), x - width * 0.25, x + width * 0.25,
                              color="0.15", linewidth=1.6, zorder=4)
            ax.set_xticks(centers, [f"{row['curve_value']:g}" for row in selected])
            ax.set(xlabel=xlabel, ylim=(0, 1),
                   xlim=(-0.6, len(selected) - 0.4))
            ax.set_axisbelow(True)
            ax.grid(which="major", color="0.88", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Test metric")
        header = Patch(facecolor="none", edgecolor="none",
                       label=rf"Base Model: {backend}   Privacy: $\epsilon={epsilon:g}$")
        figure.legend(handles=[header, *[Patch(facecolor=color, label=label)
                                        for _, label, color in datasets]],
                      loc="lower center", bbox_to_anchor=(0.5, 0.80), borderaxespad=0,
                      ncol=4, fontsize=17, frameon=True, fancybox=True,
                      columnspacing=1.4, facecolor="white", edgecolor="0.8", framealpha=0.8)
        figure.subplots_adjust(left=0.06, right=0.985, bottom=0.20, top=0.78, wspace=0.12)
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
    cli.add_argument("--out-dir", type=Path,
                     help="fresh figure/data directory (default: OFAT_ROOT/figures)")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    argument_values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    root = args.ofat_root.expanduser().absolute()
    out_dir = args.out_dir.expanduser().absolute() if args.out_dir else root / "figures"
    argument_values["out_dir"] = str(out_dir)
    try:
        require(not out_dir.exists() and not out_dir.is_symlink(), f"output directory already exists: {out_dir}; refusing overwrite")
        require(root.is_dir(), f"input root is not a directory: {root}")
        require(not root.resolve().is_relative_to(out_dir.resolve()),
                f"output directory cannot replace the input root or its ancestors: {root}")
        bundle = read_manifest(root, "ofat")
        rows, all_hashes, errors, split_signatures = [], dict(bundle["hashes"]), [], {}
        for entry in bundle["entries"]:
            try:
                row, hashes, signature = read_run(bundle, entry)
                if entry["protocol"] in split_signatures:
                    same(signature, split_signatures[entry["protocol"]], "incompatible dataset/task/split evidence across runs")
                else:
                    split_signatures[entry["protocol"]] = signature
                rows.append(row)
                all_hashes.update(hashes)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                errors.append(f"{root / entry['run_dir']}: {type(error).__name__}: {error}")
        if errors:
            details = "\n".join(errors[:20])
            omitted = f"\n... {len(errors) - 20} additional invalid/incomplete runs" if len(errors) > 20 else ""
            raise ValueError(f"strict analysis rejected {len(errors)} invalid/incomplete runs; {len(rows)} valid. "
                             f"No output was written.\n{details}{omitted}")
        same(len(rows), 60, "complete validated run count")
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
            "schema_version": 1, "analysis": "SparseExpand OFAT", "completeness": "strict_complete",
            "arguments": argument_values, "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
            "input_run_count": len(rows), "curve_row_count": len(curves),
            "curve_membership": "The one shared anchor is referenced in each of the three OFAT curves; no additional training run or averaging.",
            "input_root": str(root), "input_sha256": dict(sorted(all_hashes.items())),
            "plot_code_sha256": sha256(Path(__file__)),
            "analysis_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
            "versions": {"python": platform.python_version(), **versions},
            "policy": POLICY, "split_evidence": split_signatures,
            "manifest_provenance": bundle["manifest"]["provenance"],
            "output_sha256": {path.name: sha256(path) for path in [out_dir / "per_run.csv", out_dir / "curves.csv", *figures]},
        }
        with (out_dir / "provenance.json").open("x") as stream:
            json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Validated {len(rows)} runs; {len(curves)} OFAT curve points; {len(figures)} PNG/PDF files: {out_dir}")
        print("Metric bars: stored 95% node-bootstrap intervals, not training-randomness uncertainty.")
        print(POLICY["accounting"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, importlib.metadata.PackageNotFoundError) as error:
        cli.exit(2, f"sparse_ablation: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
