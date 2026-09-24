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
from src.models.bootstrap import BootstrapConfig
from src.training.baselines import BaselineConfig, BaselineTrainer
from src.training.dpar import DPARConfig, DPARTrainer
from src.processing.graphs import preprocess_inductive_split
from src.processing.splits import load_or_create_inductive_split


# Every supported method also accepts continuous targets through its task head
# and objective; the privacy mechanisms and accountants remain method-owned.
_SUPPORTED_METHODS = frozenset(
    {"mlp", "dp_mlp", "graphsage", "gin", "dpar", "dp_gnn", "progap"})


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
        else "r2" if resolved["regression"]
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
    method = config["method"]
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported method {method!r}; expected {sorted(_SUPPORTED_METHODS)}")
    bootstrap = BootstrapConfig(
        confidence_level=config.get("bootstrap_confidence", 0.95),
        n_resamples=config.get("bootstrap_resamples", 1000),
        seed=config.get("bootstrap_seed", 0),
    )
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
    if method != "progap":
        options.update(
            bootstrap_confidence=bootstrap.confidence_level,
            bootstrap_resamples=bootstrap.n_resamples,
            bootstrap_seed=bootstrap.seed,
        )
    _check_task_options(options, task)

    domain_dataset = bool(
        getattr(dataset, "domain_dataset", getattr(data, "domain_dataset", False)))
    requested_strategy = config.get("split_strategy")
    if domain_dataset:
        if requested_strategy not in {None, "domain"}:
            raise ValueError("domain datasets require split_strategy='domain'")
        split_strategy = "domain"
    else:
        dataset_strategy = getattr(dataset, "split_strategy", None)
        if (dataset_strategy is not None and requested_strategy is not None
                and requested_strategy != dataset_strategy):
            raise ValueError(
                f"configured split_strategy={requested_strategy!r} conflicts with "
                f"dataset split_strategy={dataset_strategy!r}")
        split_strategy = str(dataset_strategy or requested_strategy or "stratified")
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

    first_party = {"dpar", "mlp", "dp_mlp", "graphsage", "gin"}
    if method in first_party:
        for name in ("multilabel", "regression", "binary", "metric_ignore_label"):
            options[name] = task[name]
    elif method == "progap":
        # Task metadata travels in the partition manifest; the bridge takes the
        # seed from the top-level config, not ProGAP's hyperparameter namespace.
        options["multilabel"] = task["multilabel"]
        for name in ("seed", "binary", "regression", "metric_ignore_label"):
            options.pop(name, None)
    if method != "progap":
        options.setdefault("regression", task["regression"])

    if method == "dpar":
        result = DPARTrainer(
            _dataclass_config(DPARConfig, options), device=device).fit(split)
    elif method in {"mlp", "dp_mlp", "graphsage", "gin"}:
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
    parser.add_argument(
        "--method", choices=sorted(_SUPPORTED_METHODS),
        help="override the configured method")
    parser.add_argument(
        "--out", type=Path,
        help="default: results/inductive/<dataset>/<method>[-<domain-split-id>].json")
    parser.add_argument(
        "--bootstrap-confidence", type=float,
        help="bootstrap confidence fraction (default: 0.95)")
    parser.add_argument(
        "--bootstrap-resamples", type=int,
        help="number of test bootstrap resamples; 0 disables (default: 1000)")
    parser.add_argument(
        "--bootstrap-seed", type=int,
        help="local test bootstrap seed (default: 0)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset:
        config["dataset"] = args.dataset
    if args.method:
        config["method"] = args.method
    for name in ("bootstrap_confidence", "bootstrap_resamples", "bootstrap_seed"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    BootstrapConfig(
        confidence_level=config.get("bootstrap_confidence", 0.95),
        n_resamples=config.get("bootstrap_resamples", 1000),
        seed=config.get("bootstrap_seed", 0),
    )
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
    if "test_confidence_intervals" in result:
        summary["test_confidence_intervals"] = result["test_confidence_intervals"]
    if "selection" in result:
        summary["selection"] = result["selection"]
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
