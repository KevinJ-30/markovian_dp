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


def run(config: dict[str, Any]) -> dict[str, Any]:
    required = {"dataset", "method"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"experiment config is missing {sorted(missing)}")
    split_strategy = str(config.get("split_strategy", "stratified"))
    if split_strategy not in {"stratified", "native"}:
        raise ValueError("split_strategy must be 'stratified' or 'native'")
    # One top-level flag controls both split construction (a multi-hot target
    # has no single per-node class to build num_classes or a stratified split
    # from) and the trainer's loss/metric (softmax cross-entropy vs per-label
    # BCE).  Single source of truth: a config that sets this only once still
    # gets a consistent split AND a consistent loss, rather than needing the
    # same flag repeated in "parameters" and risking the two disagreeing.
    multilabel = bool(config.get("multilabel", False))
    # regression (e.g. RelBench's item-ltv): see _REGRESSION_METHODS above for
    # which methods accept it and why ProGAP does not.
    regression = bool(config.get("regression", False))
    if multilabel and regression:
        raise ValueError("a target cannot be both multilabel and regression")
    # Fail loudly rather than deep inside the trainer: a config with
    # method="progap" and regression=true would otherwise reach upstream's
    # cross_entropy and die on float targets -- exactly the opaque crash the
    # sparse-side guard in src/experiments/sparse.py was added to prevent.
    if regression and config["method"] not in _REGRESSION_METHODS:
        raise ValueError(
            f"method {config['method']!r} does not support regression; only "
            f"{sorted(_REGRESSION_METHODS)} accept it. ProGAP fuses its loss "
            f"and metric inside upstream's ProgressiveModule.step, so a "
            f"regression head there means editing vendored code")
    seed = int(config.get("seed", 0))
    device = _device(str(config.get("device", "auto")))
    # Dataset loading and split creation run on CPU. Private preprocessing and
    # training only receive ``split.train`` after graph-disjoint partitioning.
    _, data = load_dataset(config["dataset"], device="cpu")
    split = load_or_create_inductive_split(
        data,
        config["dataset"],
        root=config.get("split_root", "data/inductive_splits"),
        seed=seed,
        split_strategy=split_strategy,
        multilabel=multilabel,
        regression=regression,
    )
    method = config["method"]
    options = dict(config.get("parameters", {}))
    options.setdefault("seed", seed)
    if method in {"dpar", "mlp", "dp_mlp", "graphsage"}:
        options.setdefault("multilabel", multilabel)
    if method in _REGRESSION_METHODS:
        options.setdefault("regression", regression)
    if method == "dpar":
        result = DPARTrainer(_dataclass_config(DPARConfig, options), device=device).fit(split)
    elif method in {"mlp", "dp_mlp", "graphsage"}:
        options["method"] = method
        result = BaselineTrainer(_dataclass_config(BaselineConfig, options), device=device).fit(split)
    elif method == "dp_gnn":
        from tempfile import TemporaryDirectory
        from .dpgnn_adapter import run_partitioned
        from .upstream import export_partitions

        with TemporaryDirectory(prefix="dp-gnn-partitions-") as temporary:
            manifest = export_partitions(split, temporary)
            result = run_partitioned(manifest, Path(temporary) / "result.json", **options)
    else:
        from .upstream import UpstreamBaseline
        # Hand the RESOLVED options down, not the raw config block: the
        # top-level `regression` flag has to reach the subprocess adapter, and
        # requiring it to be repeated inside "parameters" is exactly the
        # two-places-to-set-one-thing bug the multilabel flag was consolidated
        # to avoid.  `options` is a copy, so the caller's config is untouched.
        result = UpstreamBaseline(method, {**config, "parameters": options}).run(split)
    result.update({
        "dataset": config["dataset"], "seed": seed, "device": device,
        "split_strategy": split_strategy,
        "split_file": str(split.path),
        "partitions": {name: getattr(split, name).stats for name in ("train", "val", "test")},
    })
    if device.startswith("cuda"):
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(torch.device(device))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", help="override the configured dataset")
    parser.add_argument("--method", help="override the configured method")
    parser.add_argument("--out", type=Path, help="default: results/inductive/<dataset>/<method>.json")
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
    output = args.out or Path("results/inductive") / config["dataset"].lower() / f"{config['method']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "validation_accuracy": result.get("validation_accuracy"),
                      "test_accuracy": result.get("test_accuracy")}, sort_keys=True))


if __name__ == "__main__":
    main()
