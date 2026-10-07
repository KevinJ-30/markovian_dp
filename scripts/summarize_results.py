#!/usr/bin/env python3
"""Summarize completed run_experiment.py CSVs without the training environment.

Use the runner's explicit identities and parameters object; each row is final.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


METRICS = ("accuracy", "auroc", "micro_f1", "r2", "macro_f1")
METHOD_NAMES = {
    "mlp": "MLP", "graphsage": "GraphSAGE", "gin": "GIN", "dp_mlp": "DP-MLP",
    "progap": "ProGAP", "dpar": "DPAR",
    "dp_gnn_sage": "DP-GNN-SAGE", "dp_gnn_gin": "DP-GNN-GIN",
    "sparse_sage": "SparseGNN-SAGE", "sparse_gin": "SparseGNN-GIN",
}
REQUIRED_COLUMNS = {
    "protocol", "method", "dp", "target_epsilon", "target_delta", "metric",
    "seed", "parameters", "status", "test_metric", "validation_metric",
    "split", "domain_split", "domain_split_id",
}
# The runner's parameters object is the hyperparameter identity. Replicate
# seeds and calibration diagnostics must not split a multi-seed configuration.
IGNORED_PARAMETERS = {
    "seed", "cap_seed", "bootstrap_seed", "method", "target_epsilon",
    "target_delta", "delta", "K_in_achieved", "K_out_achieved",
    "accounting_grid", "calibration_rtol", "calibration_atol", "sigma", "noise_multiplier",
}
OUTPUT_COLUMNS = [
    "group", "dataset", "method", "privacy", "epsilon", "delta", "split",
    "metric", "config_id", "configuration", "n", "seeds", "value", "validation_value",
    "uncertainty", "uncertainty_type", "ci_lower", "ci_upper",
    "confidence_level", "display", "selection", "sources",
]


class SummaryError(ValueError):
    pass


class Diagnostics:
    def __init__(self):
        self.warnings = []
        self._seen = set()

    def warn(self, message):
        if message not in self._seen:
            self._seen.add(message)
            self.warnings.append(message)
            print(f"warning: {message}", file=sys.stderr)


def populated(value):
    return value is not None and str(value).strip() != ""


def canonical(value):
    """Comparable text for numeric settings, seeds, and raw-column filters."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text.lower()
    if not number:
        return "0"
    # Decimal's normalized scientific spelling also avoids huge zero-filled strings.
    return str(number.normalize())


def flag(value, location):
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise SummaryError(f"{location}: expected a boolean, got {value!r}")


def finite(value, location):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SummaryError(f"{location}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise SummaryError(f"{location}: expected a finite number, got {value!r}")
    return number


def object_json(value, location):
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise SummaryError(f"{location}: malformed JSON: {exc}") from exc
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise SummaryError(f"{location}: JSON must be an object")
    return result


def privacy_fields(row, location):
    if not flag(row["dp"], f"{location}: dp"):
        return "non-private", "non-private", ""
    epsilon, delta = row["target_epsilon"], row["target_delta"]
    if finite(epsilon, f"{location}: target_epsilon") <= 0:
        raise SummaryError(f"{location}: target_epsilon must be positive")
    if not 0 < finite(delta, f"{location}: target_delta") < 1:
        raise SummaryError(f"{location}: target_delta must be in (0, 1)")
    return "private", canonical(epsilon), canonical(delta)


def metric_fields(row, requested, location):
    primary = row["metric"]
    if primary not in METRICS:
        raise SummaryError(f"{location}: unsupported primary metric {primary!r}")
    metric = primary if requested == "auto" else requested
    score = row.get(f"test_{metric}", "")
    if not populated(score) and metric == primary:
        score = row["test_metric"]
    return metric, score


def configuration(row, location):
    parameters = object_json(row["parameters"], f"{location}: parameters")
    settings = {key: canonical(value) for key, value in parameters.items()
                if key not in IGNORED_PARAMETERS and populated(value)}
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identifier = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    description = "; ".join(f"{key}={value}" for key, value in sorted(settings.items())) or "(unspecified)"
    return identifier, description


@dataclass
class Run:
    group: str
    source: str
    line: int
    raw: dict
    dataset: str
    method: str
    privacy: str
    epsilon: str
    delta: str
    split: str
    metric: str
    score: str
    seed: str
    config_id: str
    configuration: str
    validation_score: str
    run_index: int | None

    @property
    def location(self):
        return f"{self.source}:{self.line}"

    def identity(self):
        return (self.group, self.dataset, self.method, self.privacy, self.epsilon,
                self.delta, self.split, self.metric)


def make_run(row, group, source, line, requested):
    location = f"{source}:{line}"
    dataset = row["protocol"]
    if not dataset:
        raise SummaryError(f"{location}: missing protocol")
    if row["method"] not in METHOD_NAMES:
        raise SummaryError(f"{location}: unsupported method {row['method']!r}")
    method = METHOD_NAMES[row["method"]]
    privacy, epsilon, delta = privacy_fields(row, location)
    metric, score = metric_fields(row, requested, location)
    config_id, description = configuration(row, location)
    split = "; ".join(f"{key}={row[key]}" for key in ("domain_split", "domain_split_id", "split")
                      if populated(row[key]))
    seed = row["seed"]
    run_index = None
    if populated(row.get("run_index")):
        index = finite(row["run_index"], f"{location}: run_index")
        if index < 0 or not index.is_integer():
            raise SummaryError(f"{location}: run_index must be a nonnegative integer")
        run_index = int(index)
    return Run(group, str(source), line, row, dataset, method, privacy, epsilon, delta,
               split, metric, score, canonical(seed) if seed else "", config_id, description,
               row["validation_metric"] if metric == row["metric"] else "", run_index)


def expand_patterns(patterns, base, label):
    files = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        if not os.path.isabs(expanded):
            expanded = str(base / expanded)
        matches = [Path(path).resolve() for path in glob.glob(expanded, recursive=True) if Path(path).is_file()]
        if not matches:
            raise SummaryError(f"{label}: file pattern {pattern!r} matched no files (relative to {base})")
        files.update(matches)
    return sorted(files)


def groups_from_args(args):
    groups = []
    if args.files:
        groups.append(("default", expand_patterns(args.files, Path.cwd(), "group 'default'"), {}))
    if args.groups:
        path = Path(args.groups).expanduser().resolve()
        try:
            spec = object_json(path.read_text(encoding="utf-8"), str(path))
        except OSError as exc:
            raise SummaryError(f"cannot read group configuration {path}: {exc}") from exc
        if set(spec) != {"groups"} or not isinstance(spec["groups"], list):
            raise SummaryError(f"{path}: expected {{\"groups\": [{{\"name\": ..., \"files\": [...], \"where\": {{...}}}}]}}")
        for index, entry in enumerate(spec["groups"]):
            label = f"{path}: groups[{index}]"
            if not isinstance(entry, dict) or set(entry) - {"name", "files", "where"}:
                raise SummaryError(f"{label}: expected only name, files, and optional where")
            name, patterns, where = entry.get("name"), entry.get("files"), entry.get("where", {})
            if not isinstance(name, str) or not name.strip():
                raise SummaryError(f"{label}: name must be a nonempty string")
            if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) or not p for p in patterns):
                raise SummaryError(f"{label}: files must be a nonempty list of path/glob strings")
            if not isinstance(where, dict):
                raise SummaryError(f"{label}: where must be an object of raw CSV column filters")
            for key, values in where.items():
                options = values if isinstance(values, list) else [values]
                if not options or any(isinstance(value, (dict, list)) or value is None for value in options):
                    raise SummaryError(f"{label}: filter {key!r} requires a scalar or nonempty list of scalars")
            groups.append((name.strip(), expand_patterns(patterns, path.parent, label), where))
    if not groups:
        raise SummaryError("provide at least one CSV path/glob or a nonempty --groups configuration")
    names = [group[0] for group in groups]
    if len(names) != len(set(names)):
        raise SummaryError("group names must be unique; positional files reserve the name 'default'")
    return groups


def guard_outputs(prefix, groups, group_config):
    prefix = Path(prefix).expanduser()
    outputs = [Path(str(prefix) + extension) for extension in (".csv", ".md")]
    inputs = {path for _, files, _ in groups for path in files}
    if group_config:
        inputs.add(Path(group_config).expanduser().resolve())
    for index, output in enumerate(outputs):
        for other in list(inputs) + outputs[:index]:
            if output.resolve() == other.resolve() or (
                output.exists() and other.exists() and os.path.samefile(output, other)
            ):
                raise SummaryError(f"output path collision: {output} would overwrite {other}")
        if output.exists() and not output.is_file():
            raise SummaryError(f"output path is not a regular file: {output}")
    return outputs


def read_group(name, files, where, requested, diagnostics):
    runs = []
    matched = 0
    filters = {key: {canonical(value) for value in (values if isinstance(values, list) else [values])}
               for key, values in where.items()}
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle, strict=True)
                header = reader.fieldnames
                if not header or any(not field or not field.strip() for field in header):
                    raise SummaryError(f"{path}: missing or empty CSV column names")
                if len(header) != len(set(header)):
                    raise SummaryError(f"{path}: duplicate CSV column names")
                missing = REQUIRED_COLUMNS - set(header)
                if missing:
                    raise SummaryError(f"{path}: missing runner CSV columns: {', '.join(sorted(missing))}")
                missing = set(filters) - set(header)
                if missing:
                    raise SummaryError(f"group {name!r}, {path}: unknown filter columns: {', '.join(sorted(missing))}")
                for row in reader:
                    location = f"{path}:{reader.line_num}"
                    if None in row or any(value is None for value in row.values()):
                        raise SummaryError(f"{location}: malformed CSV row (column count differs from header)")
                    row = {key: value.strip() for key, value in row.items()}
                    if any(canonical(row[key]) not in values for key, values in filters.items()):
                        continue
                    matched += 1
                    if row["status"] != "completed":
                        diagnostics.warn(f"{location}: skipping outcome with status {row['status']!r}")
                        continue
                    runs.append(make_run(row, name, path, reader.line_num, requested))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise SummaryError(f"cannot read CSV {path}: {exc}") from exc
    if not matched:
        raise SummaryError(f"group {name!r}: no CSV rows matched the configured filters")
    return runs


def usable_score(run, diagnostics):
    try:
        value = float(run.score)
    except (TypeError, ValueError, OverflowError):
        value = math.nan
    if not math.isfinite(value):
        diagnostics.warn(f"{run.location}: absent/nonfinite usable test score for {run.metric}; skipping final run")
        return None
    return value


def bootstrap_interval(run, diagnostics):
    text = run.raw.get("test_confidence_intervals", "")
    payload = object_json(text, f"{run.location}: test_confidence_intervals") if populated(text) else {}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise SummaryError(f"{run.location}: CI metrics must be an object")
    interval = metrics.get(run.metric)
    if interval is None:
        diagnostics.warn(f"{run.location}: missing stored bootstrap CI for {run.metric}; retaining point estimate with N/A uncertainty")
        return None
    if not isinstance(interval, dict):
        raise SummaryError(f"{run.location}: bootstrap CI for {run.metric} must be an object")
    valid = interval.get("valid_resamples")
    if valid is not None and (not isinstance(valid, int) or isinstance(valid, bool) or valid < 0):
        raise SummaryError(f"{run.location}: valid_resamples must be a nonnegative integer")
    if valid == 0 or interval.get("lower") is None or interval.get("upper") is None:
        diagnostics.warn(f"{run.location}: no usable stored bootstrap CI for {run.metric}; retaining point estimate with N/A uncertainty")
        return None
    lower = finite(interval["lower"], f"{run.location}: CI lower")
    upper = finite(interval["upper"], f"{run.location}: CI upper")
    if lower > upper:
        raise SummaryError(f"{run.location}: bootstrap CI lower bound exceeds upper bound")
    level = payload.get("confidence_level")
    if level is not None:
        level = finite(level, f"{run.location}: CI confidence_level")
        if not 0 < level <= 1:
            raise SummaryError(f"{run.location}: CI confidence_level must be in (0, 1]")
    return lower, upper, level


def output_row(run, value, n, seeds, sources, validation_value=None):
    return {
        "group": run.group, "dataset": run.dataset, "method": run.method,
        "privacy": run.privacy, "epsilon": run.epsilon, "delta": run.delta,
        "split": run.split, "metric": run.metric, "config_id": run.config_id,
        "configuration": run.configuration, "n": n, "seeds": ";".join(seeds),
        "value": value, "validation_value": validation_value if validation_value is not None else "",
        "_run_index": run.run_index,
        "uncertainty": "N/A", "uncertainty_type": "none",
        "ci_lower": "", "ci_upper": "", "confidence_level": "",
        "selection": "all", "sources": ";".join(sorted(sources)),
    }


def summarize(runs, seed_mode, diagnostics, require_validation=False):
    scored = []
    for run in runs:
        value = usable_score(run, diagnostics)
        if value is None:
            continue
        try:
            validation = finite(run.validation_score, f"{run.location}: validation score for {run.metric}")
        except SummaryError:
            if require_validation:
                raise
            validation = None
        scored.append((run, value, validation))
    if not seed_mode:
        rows = []
        for run, value, validation in scored:
            result = output_row(run, value, 1, [run.seed] if run.seed else [], {run.source}, validation)
            interval = bootstrap_interval(run, diagnostics)
            if interval is not None:
                lower, upper, level = interval
                result.update(uncertainty=max(abs(value - lower), abs(upper - value)),
                              uncertainty_type="stored_bootstrap_ci", ci_lower=lower,
                              ci_upper=upper, confidence_level=level if level is not None else "")
            rows.append(result)
        return rows
    cells = defaultdict(dict)
    sources = defaultdict(set)
    indices = defaultdict(list)
    for run, value, validation in scored:
        if not run.seed:
            raise SummaryError(f"{run.location}: --seed requires a seed ID for every usable final run")
        key = run.identity() + (run.config_id,)
        prior = cells[key].get(run.seed)
        if prior is not None and (prior[1] != value or (
            prior[2] is not None and validation is not None and prior[2] != validation
        )):
            raise SummaryError(f"conflicting results for seed {run.seed}, group {run.group!r}, "
                               f"{run.dataset}/{run.method}, configuration {run.config_id}: "
                               f"{prior[0].location} versus {run.location}; use separate groups")
        if prior is None or validation is not None:
            cells[key][run.seed] = (run, value, validation)
        if run.run_index is not None:
            indices[key].append(run.run_index)
        sources[key].add(run.source)
    rows = []
    for key, seeds in cells.items():
        ordered = sorted(seeds)
        run = seeds[ordered[0]][0]
        values = [seeds[seed][1] for seed in ordered]
        validation_values = [seeds[seed][2] for seed in ordered]
        validation = (statistics.mean(validation_values)
                      if all(value is not None for value in validation_values) else None)
        result = output_row(run, statistics.mean(values), len(values), ordered, sources[key], validation)
        result["_run_index"] = min(indices[key], default=None)
        result["uncertainty_type"] = "sample_sd"
        if len(values) > 1:
            result["uncertainty"] = statistics.stdev(values)
        else:
            diagnostics.warn(f"group {run.group!r}, {run.dataset}/{run.method}, configuration {run.config_id}: "
                             "only one unique seed; sample SD is N/A")
        rows.append(result)
    return rows


def ordering(row):
    return tuple(str(row[key]) for key in ("group", "dataset", "method", "privacy", "epsilon", "delta",
                                         "split", "metric", "config_id", "seeds", "sources")) + (
        json.dumps(row, sort_keys=True, ensure_ascii=False),)


def select_best(rows, diagnostics):
    diagnostics.warn("--best selects on TEST performance, not validation; reported results are subject to test-selection bias")
    chosen = {}
    for row in sorted(rows, key=ordering):
        key = tuple(row[field] for field in ("group", "dataset", "method", "privacy", "epsilon", "delta", "split", "metric"))
        if key not in chosen or row["value"] > chosen[key]["value"]:
            row["selection"] = "best_test"
            chosen[key] = row
    return list(chosen.values())


def select_best_validation(rows):
    chosen = {}
    candidates = sorted(rows, key=lambda row: (
        row["_run_index"] is None, row["_run_index"] if row["_run_index"] is not None else 0,
        ordering(row),
    ))
    for row in candidates:
        score = finite(row["validation_value"], "best-validation: validation score")
        key = tuple(row[field] for field in (
            "group", "dataset", "method", "privacy", "epsilon", "delta", "split", "metric"))
        if key not in chosen or score > chosen[key]["validation_value"]:
            row["selection"] = "best_validation"
            chosen[key] = row
    return list(chosen.values())


def markdown_cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def number_display(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def export(rows, outputs, args, diagnostics):
    rows.sort(key=ordering)
    for row in rows:
        row["display"] = f"{number_display(row['value'])} ± {number_display(row['uncertainty'])}"
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in OUTPUT_COLUMNS} for row in rows)
    headers = ["Group", "Dataset / split", "Method", "ε", "δ", "Metric",
               "Test result", "n", "Seeds", "Configuration"]
    if args.bootstrap:
        headers.insert(7, "CI level")
    lines = ["# Result summary", "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        dataset = row["dataset"] + (f" ({row['split']})" if row["split"] else "")
        radius = number_display(row["uncertainty"])
        display = f"${number_display(row['value'])} \\pm {radius}$" if radius != "N/A" else f"${number_display(row['value'])} \\pm \\mathrm{{N/A}}$"
        cells = [row["group"], dataset, row["method"], row["epsilon"], row["delta"] or "—",
                 row["metric"], None, row["n"], row["seeds"] or "—", row["configuration"]]
        if args.bootstrap:
            level = row["confidence_level"]
            cells.insert(7, f"{100 * level:g}%" if isinstance(level, (int, float)) else "N/A")
        escaped = [display if value is None else markdown_cell(value) for value in cells]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(["", "## Interpretation", ""])
    if args.seed:
        lines.append("- Values are means across unique seeds within one configuration; $\\pm$ is sample standard deviation (ddof=1), not a confidence interval. One seed has N/A SD.")
        if args.best:
            lines.append("- `--best` selects the configuration with the highest mean TEST score; its own mean, SD, and seeds are retained. Configurations are never pooled.")
    else:
        lines.append("- Values are the original final-run TEST point estimates. Stored bootstrap CIs are not recomputed or averaged across seeds.")
        lines.append("- $\\pm$ uses the conservative symmetric envelope radius `max(abs(value - lower), abs(upper - value))`. Original (possibly asymmetric) CI endpoints and confidence levels are preserved in the CSV; the point estimate is never replaced by the CI midpoint.")
        if args.best:
            lines.append("- `--best` selects the highest TEST-scoring final run across configurations and seeds, retaining exactly that run's stored CI.")
    lines.append("- Scores remain on their input scale. Different datasets, methods, privacy budgets, splits, metrics, and named groups remain separate.")
    if args.best:
        lines.append("- **TEST-selection bias:** these best-test summaries are optimistically selected and are not validation-selected or unbiased comparisons.")
    if args.best_validation:
        lines.append("- `--best-validation` ranks by validation performance and retains the winner's TEST statistics. In seed mode validation and test means use the same unique seeds; ties use the lowest run index when available, otherwise deterministic input ordering.")
    if diagnostics.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend("- " + markdown_cell(warning) for warning in diagnostics.warnings)
    outputs[1].write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    cli = argparse.ArgumentParser(
        description="Summarize final result CSVs by dataset, method, privacy budget, split, and named hyperparameter regime.",
        epilog="Bootstrap mode reads stored CIs only. Seed mode averages unique seeds within each configuration and uses sample SD (ddof=1). --best deliberately selects on TEST, not validation: bootstrap chooses one final run and its CI; seed mode chooses the configuration with highest mean test score and retains its mean/SD/seeds. Both selections incur test-selection bias. No intermediate checkpoint is selected by test score.",
    )
    cli.add_argument("files", nargs="*", metavar="CSV_OR_GLOB", help="CSV paths or quoted globs (including recursive **); assigned to group 'default'")
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true", help="report final-run point estimates with their stored bootstrap CIs")
    mode.add_argument("--seed", action="store_true", help="mean ± sample SD over unique seed IDs, separately per configuration")
    selection = cli.add_mutually_exclusive_group()
    selection.add_argument("--best", action="store_true", help="select highest TEST score (bootstrap) or configuration mean TEST score (seed); selection bias applies")
    selection.add_argument("--best-validation", action="store_true", help="select by validation score using the same unique seeds as test aggregation; retain the winner's test statistics")
    cli.add_argument("--groups", metavar="FILE.json", help='JSON: {"groups":[{"name":"regime","files":["*.csv"],"where":{"lr":[0.01]}}]}; paths relative to JSON')
    cli.add_argument("--out", required=True, metavar="PREFIX", help="write PREFIX.csv and PREFIX.md")
    cli.add_argument("--metric", choices=("auto",) + METRICS, default="auto", help="auto uses the runner's metric field; an override reads the corresponding test_<metric> column")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    diagnostics = Diagnostics()
    # Python/platform C long limits differ; large CI/config JSON cells are legitimate.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    try:
        groups = groups_from_args(args)
        outputs = guard_outputs(args.out, groups, args.groups)
        rows = []
        for name, files, where in groups:
            runs = read_group(name, files, where, args.metric, diagnostics)
            results = summarize(runs, args.seed, diagnostics,
                                require_validation=args.best_validation)
            if not results:
                raise SummaryError(f"group {name!r}: no usable final results; check status, metric, and warnings")
            rows.extend(results)
        if args.best:
            rows = select_best(rows, diagnostics)
        elif args.best_validation:
            rows = select_best_validation(rows)
        export(rows, outputs, args, diagnostics)
    except (SummaryError, OSError) as exc:
        cli.error(str(exc))
    print(f"Wrote {len(rows)} summary rows to {outputs[0]} and {outputs[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
