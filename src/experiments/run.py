"""Configuration-driven runner for graph-disjoint inductive experiments.

Example:
    python -m src.experiments.run --config configs/inductive_smoke.json
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import torch

from src.data.datasets import load_dataset
from src.training.baselines import BaselineConfig, BaselineTrainer
from src.training.dpar import DPARConfig, DPARTrainer
from src.processing.graphs import preprocess_inductive_split
from src.processing.splits import load_or_create_inductive_split


# Methods that accept a continuous target.  Regression is a loss/head change,
# not a mechanism change -- every one of these bounds sensitivity by clipping a
# per-example gradient, and clipping is downstream of the loss, so epsilon is
# identical to the classification run at the same hyperparameters.
#
# `progap` is the one exclusion, and it is a code-provenance decision rather
# than a privacy one: upstream fuses its loss and its metric inside
# ProgressiveModule.step (third_party/ProGAP/core/modules/prog.py:112-130), so a
# regression head there means editing vendored code and giving up the
# "unmodified upstream" property that is ProGAP's remaining faithfulness claim.
_REGRESSION_METHODS = frozenset(
    {"mlp", "dp_mlp", "graphsage", "dpar", "dp_gnn", "heterpoisson"})


def _dataclass_config(cls: type, values: dict[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unexpected = set(values) - allowed
    if unexpected:
        raise ValueError(f"unknown {cls.__name__} setting(s): {sorted(unexpected)}")
    return cls(**values)


def _device(name: str) -> str:
    if name == "auto":
        # GPUs 0–3 are reserved. Some portable/container deployments expose a
        # CUDA runtime without making physical device 4 usable, so auto falls
        # back to CPU instead of selecting a forbidden ordinal.
        if torch.cuda.is_available() and torch.cuda.device_count() > 4:
            try:
                torch.cuda.get_device_properties(4)
                return "cuda:4"
            except RuntimeError:
                pass
        return "cpu"
    if name.startswith("cuda:") and int(name.split(":", 1)[1]) < 4:
        raise ValueError("experiments must not use reserved GPUs cuda:0 through cuda:3")
    return name


def _resolve_task_metadata(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve task flags once, preferring authoritative dataset metadata."""
    task_type = str(getattr(dataset, "task_type", "")).upper()
    declares_task = any(
        token in task_type
        for token in ("BINARY", "MULTICLASS", "MULTI_CLASS", "MULTILABEL",
                      "MULTI_LABEL", "REGRESSION")
    )
    declares_multilabel = hasattr(dataset, "multilabel")
    dataset_multilabel = bool(getattr(dataset, "multilabel", False))
    inferred = {
        "binary": "BINARY" in task_type,
        "multilabel": dataset_multilabel
        or "MULTILABEL" in task_type or "MULTI_LABEL" in task_type,
        "regression": "REGRESSION" in task_type,
    }
    authoritative = declares_task or declares_multilabel
    resolved = {}
    for name in ("binary", "multilabel", "regression"):
        if authoritative:
            resolved[name] = inferred[name]
            if name in config and bool(config[name]) != resolved[name]:
                raise ValueError(
                    f"configured {name}={bool(config[name])} conflicts with "
                    f"dataset task type {task_type or 'multilabel metadata'}")
        else:
            resolved[name] = bool(config.get(name, False))
    if sum(resolved.values()) > 1:
        raise ValueError("a target cannot be binary, multilabel, and/or regression")

    expected_metric = (
        "auroc" if resolved["binary"]
        else "micro_f1" if resolved["multilabel"]
        else "mae" if resolved["regression"]
        else "accuracy"
    )
    primary_metric = str(getattr(dataset, "primary_metric", expected_metric)).lower()
    if primary_metric != expected_metric:
        raise ValueError(
            f"dataset primary_metric {primary_metric!r} conflicts with its "
            f"resolved {expected_metric!r} task")
    has_ignore_metadata = hasattr(dataset, "metric_ignore_label")
    metric_ignore_label = getattr(dataset, "metric_ignore_label", None)
    if "metric_ignore_label" in config:
        configured_ignore = config["metric_ignore_label"]
        if has_ignore_metadata and configured_ignore != metric_ignore_label:
            raise ValueError(
                "configured metric_ignore_label conflicts with dataset metadata")
        metric_ignore_label = configured_ignore
    if (
        metric_ignore_label is not None
        and (not isinstance(metric_ignore_label, int)
             or isinstance(metric_ignore_label, bool))
    ):
        raise ValueError("metric_ignore_label must be an integer or null")
    return {
        **resolved,
        "primary_metric": primary_metric,
        "metric_ignore_label": metric_ignore_label,
    }


def _check_task_options(
    options: dict[str, Any], task: dict[str, Any],
) -> None:
    for name in ("binary", "multilabel", "regression", "metric_ignore_label"):
        if name in options and options[name] != task[name]:
            raise ValueError(
                f"parameters.{name}={options[name]!r} conflicts with "
                f"resolved dataset value {task[name]!r}")


def run(config: dict[str, Any]) -> dict[str, Any]:
    required = {"dataset", "method"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"experiment config is missing {sorted(missing)}")
    seed = int(config.get("seed", 0))
    device = _device(str(config.get("device", "auto")))

    load_options: dict[str, Any] = {"device": "cpu"}
    if "domain_split" in config:
        load_options["domain_split"] = config["domain_split"]
    dataset, data = load_dataset(config["dataset"], **load_options)
    task = _resolve_task_metadata(dataset, config)
    raw_options = config.get("parameters", {})
    if not isinstance(raw_options, dict):
        raise ValueError("parameters must be a configuration mapping")
    options = dict(raw_options)
    options.setdefault("seed", seed)
    _check_task_options(options, task)
    method = config["method"]
    if task["regression"] and method not in _REGRESSION_METHODS:
        raise ValueError(
            f"method {method!r} does not support regression; only "
            f"{sorted(_REGRESSION_METHODS)} accept it. ProGAP fuses its loss "
            f"and metric inside upstream's ProgressiveModule.step, so a "
            f"regression head there means editing vendored code")

    domain_dataset = bool(
        getattr(dataset, "domain_dataset", getattr(data, "domain_dataset", False)))
    if domain_dataset and method == "heterpoisson":
        raise ValueError(
            "heterpoisson does not support domain datasets; use SparseGNN, a "
            "first-party method, DP-GNN, or ProGAP"
        )
    requested_strategy = config.get("split_strategy")
    if domain_dataset:
        if requested_strategy not in {None, "domain"}:
            raise ValueError("domain datasets require split_strategy='domain'")
        split_strategy = "domain"
    else:
        split_strategy = str(requested_strategy or "stratified")
        if split_strategy == "domain":
            raise ValueError("split_strategy='domain' requires a domain dataset")
        if split_strategy not in {"stratified", "native"}:
            raise ValueError("split_strategy must be 'stratified' or 'native'")

    domain_split = getattr(
        dataset, "domain_split", getattr(data, "domain_split", None))
    domain_split_id = getattr(
        dataset, "domain_split_id", getattr(data, "domain_split_id", None))
    split = load_or_create_inductive_split(
        data,
        config["dataset"],
        root=config.get("split_root", "data/inductive_splits"),
        seed=seed,
        split_strategy=split_strategy,
        multilabel=task["multilabel"],
        regression=task["regression"],
        primary_metric=task["primary_metric"],
        binary=task["binary"],
        metric_ignore_label=task["metric_ignore_label"],
        domain_split=domain_split,
        domain_split_id=domain_split_id,
    )
    split = preprocess_inductive_split(split)

    first_party = {"dpar", "mlp", "dp_mlp", "graphsage"}
    if method in first_party:
        for name in ("multilabel", "regression", "binary", "metric_ignore_label"):
            options[name] = task[name]
    elif method == "progap":
        # ProGAP consumes these at its adapter boundary. HeterPoisson retains
        # its existing task surface and is deliberately not changed here.
        options["multilabel"] = task["multilabel"]
        options["binary"] = task["binary"]
        options["metric_ignore_label"] = task["metric_ignore_label"]
    if method in _REGRESSION_METHODS:
        options.setdefault("regression", task["regression"])

    if method == "dpar":
        result = DPARTrainer(
            _dataclass_config(DPARConfig, options), device=device).fit(split)
    elif method in {"mlp", "dp_mlp", "graphsage"}:
        options["method"] = method
        result = BaselineTrainer(
            _dataclass_config(BaselineConfig, options), device=device).fit(split)
    elif method == "dp_gnn":
        from tempfile import TemporaryDirectory
        from .dpgnn_adapter import run_partitioned
        from .upstream import export_partitions

        # Manifest format 2 is the single task-metadata source for DP-GNN.
        for name in ("binary", "multilabel", "metric_ignore_label"):
            options.pop(name, None)
        with TemporaryDirectory(prefix="dp-gnn-partitions-") as temporary:
            manifest = export_partitions(split, temporary)
            result = run_partitioned(
                manifest, Path(temporary) / "result.json", **options)
    else:
        from .upstream import UpstreamBaseline
        result = UpstreamBaseline(
            method, {**config, "parameters": options}).run(split)

    result.update({
        "dataset": config["dataset"], "seed": seed, "device": device,
        "split_strategy": split_strategy,
        "split_file": str(split.path),
        "primary_metric": task["primary_metric"],
        "partitions": {
            name: getattr(split, name).stats
            for name in ("train", "val", "test")
        },
    })
    if domain_dataset:
        result["domain_split"] = getattr(split, "domain_split", domain_split)
        result["domain_split_id"] = getattr(
            split, "domain_split_id", domain_split_id)
    if device.startswith("cuda"):
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(
            torch.device(device))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", help="override the configured dataset")
    parser.add_argument("--method", help="override the configured method")
    parser.add_argument(
        "--out", type=Path,
        help="default: results/inductive/<dataset>/<method>[-<domain-split-id>].json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset:
        config["dataset"] = args.dataset
    if args.method:
        config["method"] = args.method
    device = _device(str(config.get("device", "auto")))
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    result = run(config)
    filename = f"{config['method']}.json"
    if result.get("domain_split_id"):
        filename = f"{config['method']}-{result['domain_split_id']}.json"
    output = (
        args.out
        or Path("results/inductive") / config["dataset"].lower() / filename
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {"output": str(output)}
    if "validation_auroc" in result:
        summary.update({
            "validation_auroc": result.get("validation_auroc"),
            "test_auroc": result.get("test_auroc"),
        })
    else:
        summary.update({
            "validation_accuracy": result.get("validation_accuracy"),
            "test_accuracy": result.get("test_accuracy"),
        })
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
