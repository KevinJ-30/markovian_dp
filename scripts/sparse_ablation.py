#!/usr/bin/env python3
"""Plot completed five-seed SparseExpand ablations without launching training.

Usage: python scripts/sparse_ablation.py --ofat-root REPEAT_ROOT
                                      [--epsilon 8] [--out-dir NEW_DIRECTORY]

Requires all 300 runs for the selected epsilon: Products, FB-100, and Arxiv;
SparseSAGE and SparseGIN; ten OFAT settings; training seeds 1–5.
Batch size and learning rate must be frozen within each five-seed cohort.
Seed-0 tuning results are not accepted and no configurations are selected here.

Three line panels show expansion radius, edge retention, and outgoing cap.
Colors denote datasets; solid lines denote SAGE and dotted lines denote GIN.
Error bars are ±1 standard error of the training-seed mean (sample SD / sqrt(5)),
not test-node bootstrap intervals or 95% confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
import platform
import sys


CONFIG_KEYS = ("protocol", "method", "epsilon", "lr", "batch_size", "epochs", "seed", "p2", "r", "K_out")
TASKS = {
    "ogbn-products": ("ogbn-products", "accuracy", False, False, "native"),
    "fb100-year-6": ("facebook100-year", "accuracy", False, False, "domain"),
    "ogbn-arxiv": ("ogbn-arxiv", "accuracy", False, False, "native"),
}
DATASETS = (
    ("ogbn-products", "Products", "#D55E00"),
    ("fb100-year-6", "FB-100", "#0072B2"),
    ("ogbn-arxiv", "Arxiv", "#56B4E9"),
)
METHODS = (("sparse_sage", "SAGE", "-"), ("sparse_gin", "GIN", ":"))
SEEDS = (1, 2, 3, 4, 5)
POINT_KEYS = ("protocol", "method", "epsilon", "r", "p2", "K_out")
SWEEPS = (
    ("r", {"p2": 0.1, "K_out": 5}, (1, 2, 3)),
    ("p2", {"r": 1, "K_out": 5}, (0.05, 0.1, 0.25, 0.5, 1.0)),
    ("K_out", {"r": 1, "p2": 0.1}, (5, 10, 20, 40)),
)
POLICY = {
    "accounting": "Gaussian-mixture substitution accounting with incoming expansion.",
    "p2_equals_one": "p2=1 removes Bernoulli edge thinning only; preprocessing and incoming sampling caps remain.",
    "checkpoint": "Best validation-primary-metric checkpoint; no test-based or repeat-seed configuration selection.",
    "uncertainty": "Mean ±1 standard error across training seeds 1–5: sample SD (ddof=1) / sqrt(5). Not a confidence interval or node-bootstrap interval.",
    "radius": "Radius 1/2 uses two layers; radius 3 uses three. Incoming sampling caps are 20, 10, 5 on successive hops.",
    "metrics": "Products, FB-100, and Arxiv test accuracy share a 0–1 display range; scores are never averaged across datasets.",
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


def required_members(epsilon):
    settings = {(fixed.get("r", x if parameter == "r" else None),
                 fixed.get("p2", x if parameter == "p2" else None),
                 fixed.get("K_out", x if parameter == "K_out" else None))
                for parameter, fixed, values in SWEEPS for x in values}
    return {(protocol, method, epsilon, radius, p2, cap, seed)
            for protocol in TASKS for method, _, _ in METHODS
            for radius, p2, cap in settings for seed in SEEDS}


def settings_from_result(result):
    parameters = result["parameters"]
    method = result["method"]
    radius = parameters["r"]
    fields = {
        "protocol": result["protocol"], "method": method,
        "epsilon": result["target_epsilon"], "lr": result["lr"],
        "batch_size": result["requested_batch_size"], "epochs": result["epochs"],
        "seed": result["seed"], "r": radius,
        "p2": parameters["p2"], "K_out": parameters["K_out"],
    }
    for key in CONFIG_KEYS[2:]:
        if fields[key] is not None:
            fields[key] = number(fields[key], key)
    for key in ("batch_size", "epochs", "seed", "r", "K_out"):
        if fields[key] is not None:
            require(fields[key].is_integer(), f"{key}: expected an integer")
            fields[key] = int(fields[key])
    return fields


def read_run(root, index_path, indexed, original_root=None):
    same(indexed.get("status"), "completed", f"{index_path}: indexed run status")
    pointer = value(indexed, "output_dir")
    require(pointer is not None, f"{index_path}: missing output_dir pointer")
    directory = recorded_path(root, pointer, original_root or root)
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
    require(method in {name for name, _, _ in METHODS}, f"unexpected method {method!r}")
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
                "domain_split", "domain_split_id", "weight_decay"):
        same(config.get(key), result.get(key), f"config/result {key}")
    for source, label in ((indexed, str(index_path)), (scientific_rows[0], str(result_csv))):
        for key in ("protocol", "dataset", "method", "metric", "status", "split", "domain_split_id"):
            if value(source, key) is not None:
                same(source[key], str(result.get(key, "")), f"{label}: {key}")
        for key in ("seed", "lr", "epochs", "requested_batch_size", "hidden", "dropout",
                    "target_epsilon", "test_metric", "validation_metric", "weight_decay"):
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
    same(parameters["K_in"], 10, "incoming accounting cap")
    same(parameters["layers"], 3 if settings["r"] == 3 else 2, "SGNN layers")
    same(parameters["incoming_sampling_caps"], [20, 10, 5][:settings["r"]], "incoming sampling caps")
    same(result["weight_decay"], 0, "weight decay")
    same(parameters["weight_decay"], 0, "optimizer weight decay")
    same(settings["epochs"], 20, "epochs")
    ci = result.get("test_confidence_intervals") or {}
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
        **settings, "metric": metric,
        "actual_epsilon": result.get("epsilon"), "delta": result.get("delta"),
        "effective_batch_size": result.get("effective_batch_size"),
        "hidden": result["hidden"], "layers": parameters.get("layers"), "dropout": result["dropout"],
        "K_in": parameters.get("K_in"), "device": result.get("device"),
        "selected_step": selection["step"], "selected_epoch": selected_epoch,
        "validation_metric": result["validation_metric"], "test_metric": result["test_metric"],
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
        "partition_sizes": {name: {key: partition[key] for key in ("nodes", "evaluated_nodes")}
                            for name, partition in config["partitions"].items()},
    }
    return row, signature


def read_results(root, epsilon):
    root = root.resolve()
    index_path = root / "results.csv"
    require(index_path.is_file(), f"{root}: missing results.csv")
    state_path = root / "state.json"
    original_root = read_json(state_path).get("root") if state_path.is_file() else root
    rows, signatures, seen, outputs = [], {}, set(), set()
    for indexed in read_csv(index_path):
        if number(indexed.get("target_epsilon"), "indexed target epsilon") != epsilon:
            continue
        row, signature = read_run(root, index_path, indexed, original_root)
        key = config_key(row, (*POINT_KEYS, "seed"))
        require(key not in seen, f"{index_path}: duplicate ablation point/seed {key}")
        require(row["actual_output_path"] not in outputs,
                f"{index_path}: repeated selected output {row['actual_output_path']}")
        seen.add(key)
        outputs.add(row["actual_output_path"])
        if row["protocol"] in signatures:
            same(signature, signatures[row["protocol"]], "incompatible dataset/task/split identities across runs")
        else:
            signatures[row["protocol"]] = signature
        rows.append(row)
    expected = required_members(epsilon)
    missing, extra = expected - seen, seen - expected
    require(not missing and not extra,
            f"{index_path}: incomplete repeat curves; missing={len(missing)}, unexpected={len(extra)}")
    return rows, signatures


def aggregate_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[config_key(row, POINT_KEYS)].append(row)
    points = []
    for key, cohort in sorted(groups.items()):
        cohort.sort(key=lambda row: row["seed"])
        same(tuple(row["seed"] for row in cohort), SEEDS, f"training seed cohort {key}")
        reference = cohort[0]
        for row in cohort[1:]:
            for field in ("metric", "lr", "batch_size", "epochs", "hidden", "layers", "dropout"):
                same(row[field], reference[field], f"mixed repeat configurations: {key}/{field}")
            for field in row["parameters"].keys() | reference["parameters"].keys():
                if field not in {"cap_seed", "K_in_achieved", "K_out_achieved", "bootstrap_seed"}:
                    same(row["parameters"].get(field), reference["parameters"].get(field),
                         f"mixed repeat configurations: {key}/{field}")
        scores = [row["test_metric"] for row in cohort]
        sd = statistics.stdev(scores)
        points.append({
            **dict(zip(POINT_KEYS, key)), "metric": reference["metric"],
            "lr": reference["lr"], "batch_size": reference["batch_size"],
            "epochs": reference["epochs"], "layers": reference["layers"],
            "n": len(cohort), "seeds": list(SEEDS),
            "test_mean": statistics.mean(scores), "test_sd": sd,
            "test_se": sd / math.sqrt(len(cohort)),
            "validation_mean": statistics.mean(row["validation_metric"] for row in cohort),
            "run_ids": [row["run_id"] for row in cohort],
        })
    return points


def relationship_curves(points):
    curves = []
    for protocol, method, epsilon in dict.fromkeys((row["protocol"], row["method"], row["epsilon"]) for row in points):
        subset = [row for row in points if (row["protocol"], row["method"], row["epsilon"]) == (protocol, method, epsilon)]
        for parameter, fixed, expected_values in SWEEPS:
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.size": 18,
                         "axes.labelsize": 22, "axes.titlesize": 21,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    epsilon, = {row["epsilon"] for row in curves}
    parameters = (
        ("r", "Expansion radius $r$"),
        ("p2", "Edge-retention probability $p_2$"),
        ("K_out", r"Outgoing-degree cap $K_{\mathrm{out}}$"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.7, 3.6), sharey=True)
    try:
        for ax, (parameter, xlabel), panel in zip(axes, parameters, ("(a)", "(b)", "(c)")):
            for protocol, _, color in DATASETS:
                for method, _, style in METHODS:
                    selected = sorted(
                        (row for row in curves if row["method"] == method
                         and row["protocol"] == protocol and row["curve_parameter"] == parameter),
                        key=lambda row: row["curve_value"])
                    require(bool(selected), f"missing curve: {method}/{protocol}/{parameter}")
                    values = [row["curve_value"] for row in selected]
                    xs = values if parameter == "r" else list(range(len(values)))
                    ax.errorbar(xs, [row["test_mean"] for row in selected],
                                yerr=[row["test_se"] for row in selected],
                                color=color, linestyle=style, linewidth=3.5, alpha=1,
                                marker="o", markersize=5, markerfacecolor="white",
                                markeredgecolor=color, markeredgewidth=1.5, capsize=4,
                                elinewidth=1.6, capthick=1.6, zorder=3)
            ax.set_xticks(xs, [f"{x:g}" for x in values])
            ax.margins(x=0.05)
            ax.set(xlabel=xlabel, ylim=(0, 1))
            ax.set_yticks((0, 0.25, 0.5, 0.75, 1))
            ax.text(0.025, 1.025, panel, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=18, fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(which="major", color="0.88", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Test accuracy")
        figure.legend(
            handles=[*[Patch(facecolor=color, label=label) for _, label, color in DATASETS],
                     *[Line2D([], [], color="0.2", linestyle=style, linewidth=3.5, alpha=1,
                              marker="o", markersize=5, markerfacecolor="white",
                              markeredgewidth=1.5, label=label)
                       for _, label, style in METHODS]],
            loc="center left", bbox_to_anchor=(0.005, 0.55), borderaxespad=0,
            ncol=1, fontsize=14, frameon=False, handlelength=2.4)
        figure.subplots_adjust(left=0.175, right=0.99, bottom=0.28, top=0.94, wspace=0.12)
        outputs = []
        for extension in ("png", "pdf"):
            path = out_dir / f"ablation_se_eps{epsilon:g}.{extension}"
            metadata = ({"CreationDate": None, "ModDate": None} if extension == "pdf"
                        else {"Software": "SparseExpand ablation analysis"})
            figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15, metadata=metadata)
            outputs.append(path)
        return outputs
    finally:
        plt.close(figure)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--ofat-root", required=True, type=Path, help="completed five-seed repeat root")
    cli.add_argument("--epsilon", type=float, default=8, help="target epsilon to plot (default: 8)")
    cli.add_argument("--out-dir", type=Path,
                     help="fresh figure/data directory (default: OFAT_ROOT/figures_eps<EPSILON>)")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    argument_values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    root = args.ofat_root.expanduser().absolute()
    out_dir = args.out_dir.expanduser().absolute() if args.out_dir else root / f"figures_eps{args.epsilon:g}"
    argument_values["out_dir"] = str(out_dir)
    try:
        finite(args.epsilon, "epsilon", minimum=0)
        require(not out_dir.exists() and not out_dir.is_symlink(), f"output directory already exists: {out_dir}; refusing overwrite")
        require(root.is_dir(), f"input root is not a directory: {root}")
        require(not root.resolve().is_relative_to(out_dir.resolve()),
                f"output directory cannot replace the input root or its ancestors: {root}")
        rows, split_signatures = read_results(root, args.epsilon)
        points = aggregate_seeds(rows)
        curves = relationship_curves(points)
        versions = {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")}
        import matplotlib
        out_dir.mkdir(parents=True, exist_ok=False)
        write_csv(out_dir / "per_run.csv", rows)
        write_csv(out_dir / "points.csv", points)
        write_csv(out_dir / "curves.csv", curves)
        figures = draw_figures(curves, out_dir)
        analysis = {
            "schema_version": 2, "analysis": "SparseExpand five-seed OFAT",
            "completeness": "strict_complete",
            "arguments": argument_values, "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
            "input_run_count": len(rows), "point_count": len(points), "curve_row_count": len(curves),
            "curve_membership": "The anchor is referenced in three panels; each point averages exactly seeds 1–5. No configuration reselection.",
            "input_root": str(root),
            "versions": {"python": platform.python_version(), **versions},
            "policy": POLICY, "split_evidence": split_signatures,
            "outputs": [path.name for path in [out_dir / "per_run.csv", out_dir / "points.csv", out_dir / "curves.csv", *figures]],
        }
        with (out_dir / "analysis.json").open("x") as stream:
            json.dump(analysis, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Validated {len(rows)} runs; {len(curves)} curve points; {len(figures)} PNG/PDF files: {out_dir}")
        print(POLICY["uncertainty"])
        print(POLICY["accounting"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, importlib.metadata.PackageNotFoundError) as error:
        cli.exit(2, f"sparse_ablation: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
