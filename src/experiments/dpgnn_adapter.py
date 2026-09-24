"""First-party partition adapter for DP-GNN.

The previous adapter imported Google's checked-out ``differentially_private_gnns``
tree at runtime. The implementation now lives in :mod:`src.training.dpgnn`;
this module retains the public manifest/result contract.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


def _load_partitions(manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    if payload.get("format") != 2:
        raise ValueError("DP-GNN requires partition manifest format 2")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "val", "test"}:
        raise ValueError("partition manifest must contain train, val, and test partitions")
    num_classes = payload.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("partition manifest must record a positive num_classes")
    primary_metric = payload.get("primary_metric")
    if primary_metric not in {"accuracy", "micro_f1", "auroc", "r2"}:
        raise ValueError("partition manifest has an unsupported primary_metric")
    binary = payload.get("binary")
    if not isinstance(binary, bool):
        raise ValueError("partition manifest must record boolean binary metadata")
    if binary != (primary_metric == "auroc"):
        raise ValueError("partition manifest binary and primary_metric metadata conflict")
    metric_ignore_label = payload.get("metric_ignore_label")
    if (
        metric_ignore_label is not None
        and (not isinstance(metric_ignore_label, int)
             or isinstance(metric_ignore_label, bool))
    ):
        raise ValueError("metric_ignore_label must be an integer or null")
    domain_split = payload.get("domain_split")
    if domain_split is not None and not isinstance(domain_split, dict):
        raise ValueError("domain_split must be an object or null")
    domain_split_id = payload.get("domain_split_id")
    if domain_split_id is not None and (
        not isinstance(domain_split_id, str) or not domain_split_id
    ):
        raise ValueError("domain_split_id must be a nonempty string or null")
    data = {}
    for name, filename in partitions.items():
        stored = torch.load(
            manifest.parent / filename, map_location="cpu", weights_only=False)
        partition = stored["data"]
        if not hasattr(partition, "eval_mask") and "eval_mask" in stored:
            partition.eval_mask = stored["eval_mask"]
        eval_mask = getattr(partition, "eval_mask", None)
        if (
            not isinstance(eval_mask, torch.Tensor)
            or eval_mask.dtype != torch.bool
            or eval_mask.ndim != 1
            or eval_mask.numel() != int(partition.num_nodes)
        ):
            raise ValueError(f"{name} partition must carry a local boolean eval_mask")
        data[name] = partition
    return data, {
        "num_classes": num_classes,
        "primary_metric": primary_metric,
        "binary": binary,
        "metric_ignore_label": metric_ignore_label,
        "domain_split": domain_split,
        "domain_split_id": domain_split_id,
    }


def run_partitioned(manifest: str | Path, result_path: str | Path, *, steps: int = 1,
                    batch_size: int = 32, noise_multiplier: float = 2.0,
                    evaluate_every: int = 0, seed: int = 0,
                    clip: float = 1.0, regression: bool = False,
                    max_private_batch_nodes: int = 8192,
                    architecture: str = "graphsage", dropout: float = 0.5,
                    bootstrap_confidence: float = 0.95,
                    bootstrap_resamples: int = 1000,
                    bootstrap_seed: int = 0) -> dict[str, Any]:
    """Train first-party DP-GNN on train.pt and evaluate val.pt/test.pt.

    The manifest defines graph-disjoint partitions. Validation selects the best
    primary-metric checkpoint at ``evaluate_every`` updates (zero means one
    expected epoch), including the final update. Training always completes all
    ``steps``; test metrics and bootstrap intervals use the selected checkpoint.
    """
    manifest = Path(manifest)
    data, task = _load_partitions(manifest)
    manifest_regression = task["primary_metric"] == "r2"
    if regression and not manifest_regression:
        raise ValueError("regression argument conflicts with partition task metadata")
    trainer = PartitionedDPGNN(
        DPGNNConfig(
            num_classes=task["num_classes"], steps=steps, batch_size=batch_size,
            noise_multiplier=noise_multiplier, evaluate_every=evaluate_every,
            seed=seed, clip=clip, regression=manifest_regression,
            multilabel=task["primary_metric"] == "micro_f1",
            binary=task["binary"],
            metric_ignore_label=task["metric_ignore_label"],
            max_private_batch_nodes=max_private_batch_nodes,
            architecture=architecture, dropout=dropout,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    )
    trained = trainer.fit(data["train"], data["val"], data["test"])
    metric = trained["metric"]
    if metric != task["primary_metric"]:
        raise ValueError("DP-GNN result metric conflicts with partition task metadata")
    result = {
        "method": "dp_gnn",
        "metric": metric,
        "parameters": asdict(trainer.config),
        "selection": trained["selection"],
        "privacy": {
            "epsilon": trained["epsilon"],
            "delta": trained["delta"],
            "accountant": "first_party.dpgnn.multiterm_rdp",
            "noise_multiplier": noise_multiplier,
            "composition_count": steps,
        },
        "implementation": {
            "architecture": trained["architecture"],
            "source": "src.training.dpgnn",
            "algorithm": [
                "reverse-edge bounded-degree sampling",
                "one-hop per-root gradients with uniform without-replacement root batches",
                "Opacus global per-root clipping and isotropic Gaussian DP-Adam",
                "multi-term hypergeometric RDP accounting",
            ],
        },
    }
    if task["binary"]:
        result.update({
            "validation_auroc": trained["validation_auroc"],
            "test_auroc": trained["test_auroc"],
        })
    else:
        # Retain the established subprocess result envelope for nonbinary
        # callers, even when the value is macro-F1 or R².
        result.update({
            "validation_accuracy": trained[f"validation_{metric}"],
            "test_accuracy": trained[f"test_{metric}"],
        })
    if "test_confidence_intervals" in trained:
        result["test_confidence_intervals"] = trained["test_confidence_intervals"]
    if task["domain_split"] is not None:
        result["domain_split"] = task["domain_split"]
        result["domain_split_id"] = task["domain_split_id"]
    Path(result_path).write_text(json.dumps(result, indent=2) + "\n")
    return result
