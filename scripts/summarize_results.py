#!/usr/bin/env python3
"""Summarize final CSV results without importing the training environment."""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


METRICS = ("accuracy", "auroc", "micro_f1", "r2", "macro_f1")
METRIC_COLUMNS = {
    "accuracy": ("test_accuracy",),
    "auroc": ("test_auroc", "test_auc"),
    "micro_f1": ("test_micro_f1", "test_f1_micro"),
    "r2": ("test_r2",),
    "macro_f1": ("test_macro_f1", "test_f1_macro"),
}
ALIASES = {
    "lr": "lr", "learning_rate": "lr",
    "hidden": "hidden", "hidden_size": "hidden", "hidden_dim": "hidden",
    "latent_size": "hidden", "L": "layers", "layers": "layers",
    "num_layers": "layers", "T": "steps", "steps": "steps",
    "epochs": "epochs", "epochs_per_stage": "epochs",
    "depth": "depth", "stages": "stages",
    "sigma": "sigma", "noise_multiplier": "sigma",
    "clip": "clip", "clip_norm": "clip", "max_grad_norm": "clip",
}
CONFIG_COLUMNS = set(ALIASES) | {
    "batch_size", "p1", "p2", "r", "dropout", "weight_decay", "optimizer",
    "momentum", "K_in", "K_out", "cap_mode", "direction", "fanouts",
    "fanout", "activation", "heads", "num_heads", "normalize", "residual",
    "batch_norm", "layer_norm", "sampling_rate", "sample_rate", "q",
    "train_batch_size", "eval_batch_size", "max_degree", "degree_bound",
    "num_neighbors", "aggregation", "aggr", "architecture", "legacy_shells",
    "epsilon_split", "epsilon_agg", "epsilon_train", "delta_agg", "delta_train",
    "aggregation_steps", "num_stages", "patience", "min_delta",
    "selection_metric", "selection_evaluate_every", "evaluate_every",
}
IGNORED_PARAMETERS = {
    "seed", "seeds", "cap_seed", "bootstrap_seed", "step", "epoch",
    "dataset", "protocol", "method", "model", "family", "dp", "private",
    "is_private", "is_dp", "metric", "primary_metric", "domain_split",
    "domain_split_id", "split", "split_id", "target_epsilon", "epsilon_target",
    "epsilon", "calibrated_epsilon", "actual_epsilon", "epsilon_estimate",
    "epsilon_context", "target_delta", "delta", "status", "accepted",
    "selection", "test_confidence_intervals", "K_in_achieved", "K_out_achieved",
    "noise_std", "noise_variance", "calibration_evaluations", "accounting_grid",
    "calibration_rtol", "calibration_atol", "output", "out", "output_dir",
    "device", "num_workers", "completed_epochs", "updates_completed",
}
OUTPUT_COLUMNS = [
    "group", "dataset", "method", "privacy", "epsilon", "delta", "split",
    "metric", "config_id", "configuration", "n", "seeds", "value",
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


def first(row, *keys):
    return next((row[k] for k in keys if populated(row.get(k))), "")


def canonical(value):
    """Comparable text for numeric aliases, seeds, and raw-column filters."""
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
    if text in {"true", "1", "yes", "y", "private", "dp"}:
        return True
    if text in {"false", "0", "no", "n", "nonprivate", "non-private"}:
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


def compact(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def architecture(row):
    text = compact(first(row, "aggregation", "aggr", "architecture"))
    if "gin" in text:
        return "GIN"
    if "gcn" in text:
        return "GCN"
    if text in {"", "mean", "sage", "graphsage"}:
        return "SAGE"
    return first(row, "aggregation", "aggr", "architecture").upper()


def method_name(row, location):
    explicit = first(row, "method", "model", "family")
    if not explicit:
        raise SummaryError(f"{location}: missing method/model/family identity")
    name = compact(explicit)
    family = compact(row.get("family", ""))
    arch = architecture(row)
    generic = name in {"gnn", "binarygnn", "multilabelgnn", "regressiongnn"}
    if name in {"dpgnn", "dpgnnsage", "dpgnngraphsage", "dpgnngin", "dpgnngcn", "dpgraphsage", "dpgin", "dpgcn"} or (
        family in {"dpgnn", "pnpignn"} and (generic or name in {"graphsage", "sage", "gin", "gcn"})
    ):
        suffix = "GIN" if "gin" in name else "GCN" if "gcn" in name else "SAGE" if "sage" in name else arch
        return f"DP-GNN-{suffix}"
    if name in {"sparse", "sparsegnn", "sparsesage", "sparsegin", "sparsegcn",
                "sparsegnnsage", "sparsegnngraphsage", "sparsegnngin", "sparsegnngcn"} or generic:
        suffix = "GIN" if "gin" in name else "GCN" if "gcn" in name else "SAGE" if "sage" in name else arch
        return f"SparseGNN-{suffix}"
    if name in {"dpmlp", "privatemlp"}:
        return "DP-MLP"
    if name == "dpar":
        return "DPAR"
    if name == "progap":
        return "ProGAP"
    if name in {"graphsage", "sage", "nonprivategraphsage", "nonprivatesage"}:
        return "GraphSAGE"
    if name in {"gin", "nonprivategin"}:
        return "GIN"
    if name in {"mlp", "nonprivatemlp"}:
        return "DP-MLP" if family == "dpmlp" else "MLP"
    return explicit.strip()


def privacy_fields(row, method, location):
    private = None
    for key in ("dp", "private", "is_private", "is_dp"):
        if populated(row.get(key)):
            private = flag(row[key], f"{location}: {key}")
            break
    family = compact(row.get("family", ""))
    if private is None:
        if family in {"nonprivate", "reference", "nonprivatereference", "mlp", "graphsage", "gin"}:
            private = False
        elif family in {"dp", "private", "dpmlp", "dpgnn", "dpar", "progap"}:
            private = True
    epsilon = first(row, "target_epsilon", "epsilon_target", "epsilon",
                    "calibrated_epsilon", "actual_epsilon", "epsilon_estimate")
    if private is None:
        raw_name = compact(first(row, "method", "model"))
        if raw_name.startswith("nonprivate") or method in {"GraphSAGE", "GIN", "MLP"}:
            private = False
        elif method.startswith(("DP-", "SparseGNN-")) or method in {"DPAR", "ProGAP"}:
            private = True
        elif epsilon:
            private = True
    if private is False:
        return "non-private", "non-private", "", ""
    target = first(row, "target_epsilon", "epsilon_target")
    if epsilon:
        if finite(epsilon, f"{location}: epsilon") < 0:
            raise SummaryError(f"{location}: epsilon must be nonnegative")
        epsilon = canonical(epsilon)
    else:
        epsilon = "unknown"
    delta = first(row, "target_delta", "delta")
    if delta:
        if not 0 <= finite(delta, f"{location}: delta") < 1:
            raise SummaryError(f"{location}: delta must be in [0, 1)")
        delta = canonical(delta)
    return "private" if private else "unknown", epsilon, delta, canonical(target) if target else ""


def metric_name(value):
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {"acc": "accuracy", "auc": "auroc", "roc_auc": "auroc", "f1_micro": "micro_f1",
            "f1_macro": "macro_f1", "r²": "r2", "r_squared": "r2"}.get(name, name)


def metric_fields(row, requested, location):
    primary = metric_name(first(row, "metric", "primary_metric"))
    if not primary:
        primary = "accuracy"
        # Infer the primary independently of --metric: selecting AUROC must
        # never relabel an accuracy-only test_acc/test_metric value.
        for candidate in ("accuracy", "r2", "auroc", "micro_f1", "macro_f1"):
            keys = METRIC_COLUMNS[candidate]
            if candidate == "accuracy":
                keys += ("test_acc", "test_metric")
            if first(row, *keys):
                primary = candidate
                break
    metric = primary if requested == "auto" else requested
    if metric not in METRICS:
        raise SummaryError(f"{location}: unsupported primary metric {metric!r}; use --metric")
    keys = METRIC_COLUMNS[metric]
    if primary == metric:
        keys += ("test_acc", "test_metric")
    return metric, first(row, *keys)


def configuration(row, target, location):
    settings = {}

    def add(key, value):
        if not populated(value) or value is None:
            return
        name = ALIASES.get(key, key)
        if name == "sigma" and target:
            return  # Calibration changes by seed, not the nominal experiment.
        normalized = canonical(value)
        if name in settings and settings[name] != normalized:
            raise SummaryError(f"{location}: conflicting configuration aliases for {name}: "
                               f"{settings[name]!r} versus {normalized!r}")
        settings[name] = normalized

    for key in sorted(CONFIG_COLUMNS):
        # Architecture is already part of normalized method identity.
        if key not in {"aggregation", "aggr", "architecture"}:
            add(key, row.get(key))
    for field in ("parameters", "config"):
        if not populated(row.get(field)):
            continue
        for key, value in object_json(row[field], f"{location}: {field}").items():
            if key in IGNORED_PARAMETERS or key in {"aggregation", "aggr", "architecture"}:
                continue
            if key.startswith(("train_", "val_", "test_")) and key not in CONFIG_COLUMNS:
                continue
            if any(token in key for token in ("seconds", "timing", "wall_time", "elapsed", "achieved")):
                continue
            add(key, value)
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identifier = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    description = "; ".join(f"{key}={value}" for key, value in sorted(settings.items())) or "(unspecified)"
    return settings, identifier, description


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
    target: str
    split: str
    metric: str
    score: str
    seed: str
    settings: dict
    config_id: str
    configuration: str
    selection: dict

    @property
    def location(self):
        return f"{self.source}:{self.line}"

    def identity(self):
        return (self.group, self.dataset, self.method, self.privacy, self.epsilon,
                self.delta, self.split, self.metric)


def accepted(row, location):
    if populated(row.get("accepted")) and not flag(row["accepted"], f"{location}: accepted"):
        return False
    status = compact(row.get("status", ""))
    return status in {"", "ok", "success", "successful", "succeeded", "complete", "completed",
                      "finished", "accepted", "done", "pass", "passed"}


def make_run(row, group, source, line, requested):
    location = f"{source}:{line}"
    dataset = first(row, "protocol", "dataset").strip()
    if not dataset:
        raise SummaryError(f"{location}: missing dataset/protocol identity")
    method = method_name(row, location)
    privacy, epsilon, delta, target = privacy_fields(row, method, location)
    metric, score = metric_fields(row, requested, location)
    settings, config_id, description = configuration(row, target, location)
    split = "; ".join(f"{key}={row[key]}" for key in ("domain_split", "domain_split_id", "split", "split_id")
                      if populated(row.get(key)))
    selection = object_json(row["selection"], f"{location}: selection") if populated(row.get("selection")) else {}
    seed = first(row, "seed", "random_seed", "run_seed")
    return Run(group, str(source), line, row, dataset, method, privacy, epsilon, delta,
               target, split, metric, score, canonical(seed) if seed else "", settings,
               config_id, description, selection)


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
                    if not accepted(row, location):
                        diagnostics.warn(f"{location}: skipping unsuccessful/unaccepted outcome")
                        continue
                    runs.append(make_run(row, name, path, reader.line_num, requested))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise SummaryError(f"cannot read CSV {path}: {exc}") from exc
    if not matched:
        raise SummaryError(f"group {name!r}: no CSV rows matched the configured filters")
    return runs


def progress(run, key):
    value = run.raw.get(key)
    return finite(value, f"{run.location}: {key}") if populated(value) else None


def final_runs(runs, diagnostics):
    logical = defaultdict(list)
    for run in runs:
        # Fixed-noise tracking rows may contain achieved epsilon at each step;
        # their nominal noise/configuration, not interim accounting, identifies a run.
        evolving = any(populated(run.raw.get(key)) for key in ("step", "epoch")) and "sigma" in run.settings
        epsilon = run.target or ("fixed-noise" if evolving else run.epsilon)
        key = (run.source, run.dataset, run.method, run.privacy, epsilon,
               run.delta, run.split, run.metric, run.config_id, run.seed)
        logical[key].append(run)
    finals = []
    for candidates in logical.values():
        selected = [run for run in candidates if run.selection]
        if selected:
            finals.extend(selected)
            continue
        key = "step" if any(populated(run.raw.get("step")) for run in candidates) else "epoch"
        tracked = [(run, progress(run, key)) for run in candidates]
        if all(value is None for _, value in tracked):
            finals.extend(candidates)  # Ordinary one-row-per-seed exports need no tracking metadata.
            continue
        budget_key = "steps" if key == "step" else "epochs"
        budget = candidates[0].settings.get(budget_key)
        if budget is not None:
            expected = finite(budget, f"{candidates[0].location}: {budget_key}")
            final = [run for run, value in tracked if value == expected]
            if not final:
                diagnostics.warn(f"{candidates[0].location}: incomplete run: no final {key}={expected:g}; skipping run")
                continue
        else:
            last = max(value for _, value in tracked if value is not None)
            final = [run for run, value in tracked if value == last]
        finals.extend(final)
    return finals


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
    entries = [entry for name, entry in metrics.items() if metric_name(name) == run.metric]
    if len(entries) > 1:
        raise SummaryError(f"{run.location}: ambiguous bootstrap CI aliases for {run.metric}")
    interval = entries[0] if entries else None
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


def output_row(run, value, n, seeds, sources):
    return {
        "group": run.group, "dataset": run.dataset, "method": run.method,
        "privacy": run.privacy, "epsilon": run.epsilon, "delta": run.delta,
        "split": run.split, "metric": run.metric, "config_id": run.config_id,
        "configuration": run.configuration, "n": n, "seeds": ";".join(seeds),
        "value": value, "uncertainty": "N/A", "uncertainty_type": "none",
        "ci_lower": "", "ci_upper": "", "confidence_level": "",
        "selection": "all", "sources": ";".join(sorted(sources)),
    }


def summarize(runs, seed_mode, diagnostics):
    scored = [(run, value) for run in runs if (value := usable_score(run, diagnostics)) is not None]
    if not seed_mode:
        rows = []
        for run, value in scored:
            result = output_row(run, value, 1, [run.seed] if run.seed else [], {run.source})
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
    for run, value in scored:
        if not run.seed:
            raise SummaryError(f"{run.location}: --seed requires a seed ID for every usable final run")
        key = run.identity() + (run.config_id,)
        prior = cells[key].get(run.seed)
        if prior is not None and prior[1] != value:
            raise SummaryError(f"conflicting results for seed {run.seed}, group {run.group!r}, "
                               f"{run.dataset}/{run.method}, configuration {run.config_id}: "
                               f"{prior[0].location} versus {run.location}; use separate groups")
        cells[key][run.seed] = (run, value)
        sources[key].add(run.source)
    rows = []
    for key, seeds in cells.items():
        ordered = sorted(seeds)
        run = seeds[ordered[0]][0]
        values = [seeds[seed][1] for seed in ordered]
        result = output_row(run, statistics.mean(values), len(values), ordered, sources[key])
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
        if row["privacy"] != "non-private" and row["epsilon"] == "unknown":
            diagnostics.warn(f"group {row['group']!r}, {row['dataset']}/{row['method']}: unknown private epsilon; "
                             "--best keeps different configurations separate because privacy budgets are not comparable")
            key += (row["config_id"],)
        if key not in chosen or row["value"] > chosen[key]["value"]:
            row["selection"] = "best_test"
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
        writer.writerows(rows)
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
    lines.append("- Scores remain on their input scale. Different datasets, methods, privacy budgets, splits, metrics, and named groups remain separate. `unknown` epsilon never means non-private.")
    if args.best:
        lines.append("- **TEST-selection bias:** these best-test summaries are optimistically selected and are not validation-selected or unbiased comparisons.")
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
    cli.add_argument("--best", action="store_true", help="select highest TEST score (bootstrap) or configuration mean TEST score (seed); selection bias applies")
    cli.add_argument("--groups", metavar="FILE.json", help='JSON: {"groups":[{"name":"regime","files":["*.csv"],"where":{"lr":[0.01]}}]}; paths relative to JSON')
    cli.add_argument("--out", required=True, metavar="PREFIX", help="write PREFIX.csv and PREFIX.md")
    cli.add_argument("--metric", choices=("auto",) + METRICS, default="auto", help="auto respects row metric/primary_metric, otherwise infers populated test columns")
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
            results = summarize(final_runs(runs, diagnostics), args.seed, diagnostics)
            if not results:
                raise SummaryError(f"group {name!r}: no usable final results; check outcome status, final step/budget, metric, and warnings")
            rows.extend(results)
        if args.best:
            rows = select_best(rows, diagnostics)
        export(rows, outputs, args, diagnostics)
    except (SummaryError, OSError) as exc:
        cli.error(str(exc))
    print(f"Wrote {len(rows)} summary rows to {outputs[0]} and {outputs[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
