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

Inputs are the root results.csv and its selected output directories, or a
historical summary.csv with result_csv pointers. Moved roots remain readable;
no manifests or source commitments are required. Outputs include per_run.csv,
curves.csv, the existing PNG/PDF panels, and ordinary analysis.json metadata.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys


CONFIG_KEYS = ("protocol", "method", "epsilon", "lr", "batch_size", "epochs", "seed", "p2", "r", "K_out")
TASKS = {
    "ogbn-arxiv": ("ogbn-arxiv", "accuracy", False, False, "native"),
    "saint-yelp": ("saint-yelp", "micro_f1", False, True, "native"),
    "twitch-allbut2": ("twitch-explicit", "auroc", True, False, "domain"),
}
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00")
POLICY = {
    "accounting": "Uses the repository's mixture formula, not an independently established privacy guarantee.",
    "p2_equals_one": "p2=1 removes Bernoulli edge thinning only: outgoing-degree preprocessing, root sampling, finite radius, and the incoming expansion cap of 20 remain. It is not full-graph training.",
    "degree_caps": "K_out is the outgoing preprocessing cap. K_in=10 remains an accounting input, not an incoming preprocessing cap; incoming preprocessing degree is unrestricted.",
    "checkpoint": "Best validation-primary-metric checkpoint; no test-based selection or configuration ranking.",
    "per_run_interval": "Stored 95% node-percentile bootstrap interval; resample counts and seed are retained in the CSV exports.",
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


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields and len(fields) == len(set(fields)), f"{path}: invalid/duplicate CSV headers")
        rows = list(reader)
    require(bool(rows), f"{path}: no result rows")
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"{path}: malformed CSV row")
    return rows


def value(mapping, *keys):
    return next((mapping[key] for key in keys if mapping.get(key) not in (None, "")), None)


def number(value, label):
    require(not isinstance(value, bool), f"{label}: expected a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}: expected a finite number, found {value!r}") from error
    return finite(parsed, label)


def required_members(study):
    """Membership of the retained paper panels, not a training-job generator."""
    sparse = study == "ofat"
    require(sparse or study == "depth-baselines", f"unknown analysis study {study!r}")
    methods = ("sparse_sage", "sparse_gin") if sparse else ("dp_gnn_sage", "dp_gnn_gin", "progap")
    settings = ((1, .5, 10), (2, .5, 10), (3, .5, 10),
                (1, .05, 10), (1, .1, 10), (1, .25, 10), (1, 1., 10),
                (1, .5, 5), (1, .5, 20), (1, .5, 40)) if sparse else (
                    (1, None, None), (2, None, None), (3, None, None))
    return {(protocol, method, 8., .01, 256, 20, 0, p2, radius, cap)
            for protocol in TASKS for method in methods for radius, p2, cap in settings}


def selected_path(root, stored, original_root=None):
    if original_root is None:
        path = Path(stored)
        original_root = root
        if path.is_absolute() and not path.is_relative_to(root):
            # Ordinary historical indexes locate attempts/runs below their saved
            # root. Relocation needs only that layout, never a source manifest.
            anchors = [index for index, part in enumerate(path.parts) if part in {"attempts", "runs"}]
            require(bool(anchors), f"cannot locate moved input root for {stored!r}")
            original_root = Path(*path.parts[:anchors[0]])
    return recorded_path(root, stored, str(original_root))


def settings_from_result(result):
    parameters = result["parameters"]
    method = result["method"]
    sparse = method.startswith("sparse_")
    radius = value(parameters, "r") if sparse else value(
        parameters, "depth" if method == "progap" else "radius")
    fields = {
        "protocol": result["protocol"], "method": method,
        "epsilon": result["target_epsilon"], "lr": result["lr"],
        "batch_size": result["requested_batch_size"], "epochs": result["epochs"],
        "seed": result["seed"], "r": radius,
        "p2": parameters.get("p2") if sparse else None,
        "K_out": parameters.get("K_out") if sparse else None,
    }
    for key in CONFIG_KEYS[2:]:
        if fields[key] is not None:
            fields[key] = number(fields[key], key)
    for key in ("batch_size", "epochs", "seed", "r", "K_out"):
        if fields[key] is not None:
            require(fields[key].is_integer(), f"{key}: expected an integer")
            fields[key] = int(fields[key])
    return fields


def read_run(root, index_path, indexed, study, original_root=None):
    same(indexed.get("status"), "completed", f"{index_path}: indexed run status")
    pointer = value(indexed, "result_csv")
    if pointer is not None:
        result_csv = selected_path(root, pointer, original_root)
        directory = result_csv.parent
    else:
        pointer = value(indexed, "output_dir")
        require(pointer is not None, f"{index_path}: missing output_dir/result_csv pointer")
        directory = selected_path(root, pointer, original_root)
        result_csv = directory / "result.csv"
    paths = {"config_json": directory / "config.json", "result_json": directory / "result.json",
             "result_csv": result_csv}
    for path in paths.values():
        require(path.resolve().is_relative_to(root.resolve()), f"artifact symlink escapes input root: {path}")
        require(path.is_file(), f"incomplete run; missing {path}")
    config, result = read_json(paths["config_json"]), read_json(paths["result_json"])
    same(result.get("status"), "completed", f"{directory}: result status")
    scientific_rows = read_csv(result_csv)
    same(len(scientific_rows), 1, f"{result_csv}: final result row count")
    required = {"protocol", "dataset", "method", "metric", "status", "seed",
                "test_metric", "validation_metric", "parameters", "selection"}
    require(all(value(scientific_rows[0], key) is not None for key in required),
            f"{result_csv}: missing required identity, selection, or metric fields")
    settings = settings_from_result(result)
    protocol, method = settings["protocol"], settings["method"]
    require(protocol in TASKS, f"{directory}: unexpected protocol {protocol!r}")
    dataset, metric, binary, multilabel, strategy = TASKS[protocol]
    same(result["dataset"], dataset, "dataset")
    same(result["metric"], metric, "primary metric")
    same(result["split_strategy"], strategy, "split strategy")
    same(result["split_seed"], 0, "split seed")
    same(result["hidden"], 128, "hidden width")
    same(result["dropout"], .5, "dropout")
    task = config["task"]
    for key, expected in {"primary_metric": metric, "binary": binary, "multilabel": multilabel,
                          "regression": False}.items():
        same(task[key], expected, f"task {key}")
    for key in ("protocol", "dataset", "method", "metric", "seed", "lr", "requested_batch_size",
                "epochs", "parameters", "target_epsilon", "split", "split_seed", "split_strategy",
                "domain_split", "domain_split_id"):
        same(config.get(key), result.get(key), f"config/result {key}")
    for source, label in ((indexed, str(index_path)), (scientific_rows[0], str(result_csv))):
        for key in ("protocol", "dataset", "method", "metric", "status", "split", "domain_split_id"):
            if value(source, key) is not None:
                same(source[key], str(result.get(key, "")), f"{label}: {key}")
        for key in ("seed", "lr", "epochs", "requested_batch_size", "hidden", "dropout",
                    "target_epsilon", "test_metric", "validation_metric"):
            if value(source, key) is not None:
                same(number(source[key], f"{label}: {key}"), result[key], f"{label}: {key}")
        for key in ("r", "p2", "K_out"):
            if value(source, key) is not None:
                same(number(source[key], f"{label}: {key}"), settings[key], f"{label}: {key}")
        for key in ("parameters", "domain_split", "selection"):
            if value(source, key) is not None:
                same(json.loads(source[key]), result.get(key), f"{label}: {key}")
    for key in ("test_metric", "validation_metric"):
        finite(result[key], key, minimum=0, maximum=1)
    if f"test_{metric}" in result:
        same(result[f"test_{metric}"], result["test_metric"], "primary score")
    selection = result["selection"]
    same(selection.get("split"), "validation", "checkpoint selection split")
    same(selection.get("metric"), metric, "checkpoint selection metric")
    same(selection.get("validation_score"), result["validation_metric"], "selected validation score")
    integer(selection["step"], "selected step")
    parameters = result["parameters"]
    sparse = method.startswith("sparse_")
    if sparse:
        same(parameters["K_in"], 10, "incoming accounting cap")
        same(parameters["layers"], 2, "SGNN layers")
    else:
        same(parameters["max_degree"], 5, "baseline degree bound")
    ci = result.get("test_confidence_intervals") or {}
    primary = ci.get("metrics", {}).get(metric) or {}
    needs_ci = sparse and settings["r"] == 1
    lower, upper = primary.get("lower"), primary.get("upper")
    available_ci = lower is not None and upper is not None and primary.get("valid_resamples", 0) > 0
    require(available_ci or not needs_ci, f"{directory}: missing bootstrap CI required for {metric} bar panel")
    if available_ci:
        finite(lower, "bootstrap lower", minimum=0, maximum=1)
        finite(upper, "bootstrap upper", minimum=0, maximum=1)
        require(lower <= upper, "bootstrap interval is reversed")
        for key, expected in {"confidence_level": .95, "method": "percentile",
                              "resampling_unit": "node"}.items():
            same(ci.get(key), expected, f"bootstrap {key}")
    else:
        lower = upper = None
    resources = result.get("resources") or {}
    native = result.get("native_result") or {}
    sampling = native.get("sampling_statistics") or {}
    cuda, rss = resources.get("peak_cuda_allocated_bytes"), resources.get("peak_rss_bytes")
    for key, observation in (("peak_cuda_allocated_bytes", cuda), ("peak_rss_bytes", rss)):
        if observation is not None:
            integer(observation, key)
    duration = result.get("calibration_and_training_seconds")
    if duration is not None:
        finite(duration, "calibration/training seconds", minimum=0)
    count = sampling.get("count")
    if count is not None:
        integer(count, "sampled subgraph count")
    selected_epoch = selection.get("epoch")
    if selected_epoch is None and parameters.get("evaluate_every"):
        selected_epoch = selection["step"] // parameters["evaluate_every"]
    row = {
        **settings, "study": study, "metric": metric,
        "actual_epsilon": result.get("epsilon"), "delta": result.get("delta"),
        "effective_batch_size": result.get("effective_batch_size"),
        "hidden": result["hidden"], "layers": parameters.get("layers"), "dropout": result["dropout"],
        "K_in": parameters.get("K_in"), "device": result.get("device"),
        "selected_step": selection["step"], "selected_epoch": selected_epoch,
        "validation_metric": result["validation_metric"], "test_metric": result["test_metric"],
        "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
        "bootstrap_confidence_level": ci.get("confidence_level"), "bootstrap_method": ci.get("method"),
        "bootstrap_n_resamples": ci.get("n_resamples"), "bootstrap_valid_resamples": primary.get("valid_resamples"),
        "bootstrap_seed": ci.get("seed"), "bootstrap_resampling_unit": ci.get("resampling_unit"),
        "bootstrap_n_observations": ci.get("n_observations"),
        "calibration_and_training_seconds": duration, "peak_cuda_allocated_bytes": cuda,
        "peak_rss_bytes": rss,
        "cuda_memory_status": "available" if cuda is not None else (
            "unavailable_cpu" if str(result.get("device", "")).startswith("cpu") else "unavailable"),
        "sampling_count": count, "mean_nodes": sampling.get("mean_nodes"), "mean_edges": sampling.get("mean_edges"),
        "max_nodes": sampling.get("max_nodes") if count else None,
        "max_edges": sampling.get("max_edges") if count else None,
        "sampling_status": "available" if count else "unavailable_no_sampled_subgraphs",
        "input_root": str(root), "input_csv_path": str(index_path),
        "run_dir": str(directory.relative_to(root)), "actual_output_path": str(directory),
        "attempt": indexed.get("attempt"), "run_id": indexed.get("run_id"),
        "recorded_output_path": pointer,
        **{name + "_path": str(path) for name, path in paths.items()},
        "parameters": parameters, "selection": selection,
        "test_confidence_intervals": ci, "resources": resources,
    }
    signature = {
        "task": task, "train_nodes": config["train_nodes"], "split_strategy": strategy,
        "split": config["split"], "split_seed": config["split_seed"],
        "domain_split": config["domain_split"], "domain_split_id": config["domain_split_id"],
        "partitions": config["partitions"],
    }
    return row, signature


def read_results(root, study):
    root = root.resolve()
    index_path = root / "results.csv"
    if not index_path.is_file():
        index_path = root / "summary.csv"
    require(index_path.is_file(), f"{root}: no results.csv or historical summary.csv")
    original_root = None
    state_path = root / "state.json"
    if state_path.is_file():
        original_root = read_json(state_path).get("root")
    rows, signatures, seen, outputs = [], {}, set(), set()
    for indexed in read_csv(index_path):
        row, signature = read_run(root, index_path, indexed, study, original_root)
        key = config_key(row)
        require(key not in seen, f"{index_path}: ambiguous duplicate scientific settings {key}")
        require(row["actual_output_path"] not in outputs,
                f"{index_path}: repeated selected output {row['actual_output_path']}")
        seen.add(key)
        outputs.add(row["actual_output_path"])
        if row["protocol"] in signatures:
            same(signature, signatures[row["protocol"]], "incompatible dataset/task/split identities across runs")
        else:
            signatures[row["protocol"]] = signature
        rows.append(row)
    expected = required_members(study)
    missing, extra = expected - seen, seen - expected
    require(not missing and not extra,
            f"{index_path}: incomplete {study} curves; missing={len(missing)}, unexpected={len(extra)}")
    return rows, signatures


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
        rows, split_signatures = [], {}
        for input_root, study in inputs:
            require(input_root.is_dir(), f"input root is not a directory: {input_root}")
            require(not input_root.resolve().is_relative_to(out_dir.resolve()),
                    f"output directory cannot replace the input root or its ancestors: {input_root}")
            input_rows, signatures = read_results(input_root, study)
            for protocol, signature in signatures.items():
                if protocol in split_signatures:
                    same(signature, split_signatures[protocol],
                         "incompatible dataset/task/split identities across inputs")
                else:
                    split_signatures[protocol] = signature
            rows.extend(input_rows)
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
        analysis = {
            "schema_version": 1, "analysis": "SparseExpand OFAT and depth comparisons" if depth_root else "SparseExpand OFAT",
            "completeness": "strict_complete",
            "arguments": argument_values, "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
            "input_run_count": len(rows), "curve_row_count": len(curves),
            "curve_membership": "The SGNN anchor is referenced in three OFAT curves. Each baseline run appears once in the depth data; ProGAP is shared between backend figures. No averaging.",
            "input_root": str(root), "depth_root": str(depth_root) if depth_root else None,
            "versions": {"python": platform.python_version(), **versions},
            "policy": POLICY, "split_evidence": split_signatures,
            "outputs": [path.name for path in [out_dir / "per_run.csv", out_dir / "curves.csv", *figures]],
        }
        with (out_dir / "analysis.json").open("x") as stream:
            json.dump(analysis, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Validated {len(rows)} runs; {len(curves)} curve points; {len(figures)} PNG/PDF files: {out_dir}")
        print("Metric bars: stored 95% node-bootstrap intervals, not training-randomness uncertainty.")
        print(POLICY["accounting"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, importlib.metadata.PackageNotFoundError) as error:
        cli.exit(2, f"sparse_ablation: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
