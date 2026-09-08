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

from src.datasets import load_dataset
from .baselines import BaselineConfig, BaselineTrainer
from .dpar import DPARConfig, DPARTrainer
from .inductive import load_or_create_inductive_split


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
    )
    method = config["method"]
    options = dict(config.get("parameters", {}))
    options.setdefault("seed", seed)
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
        result = UpstreamBaseline(method, config).run(split)
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
