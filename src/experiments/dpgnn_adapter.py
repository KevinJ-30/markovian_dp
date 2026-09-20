"""First-party partition adapter for DP-GNN.

The previous adapter imported Google's checked-out ``differentially_private_gnns``
tree at runtime. The implementation now lives in :mod:`src.training.dpgnn`;
this module retains the public manifest/result contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


def _load_partitions(manifest: Path) -> tuple[dict[str, Any], int]:
    payload = json.loads(manifest.read_text())
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "val", "test"}:
        raise ValueError("partition manifest must contain train, val, and test partitions")
    num_classes = payload.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("partition manifest must record a positive num_classes")
    data = {
        name: torch.load(manifest.parent / filename, map_location="cpu", weights_only=False)["data"]
        for name, filename in partitions.items()
    }
    return data, num_classes


def run_partitioned(manifest: str | Path, result_path: str | Path, *, steps: int = 1,
                    batch_size: int = 32, noise_multiplier: float = 2.0,
                    evaluate_every: int = 50, seed: int = 0,
                    clip: float = 1.0, regression: bool = False,
                    max_private_batch_nodes: int = 8192) -> dict[str, Any]:
    """Train first-party DP-GNN on train.pt and evaluate val.pt/test.pt.

    The manifest is the graph-disjoint boundary: no validation or test graph is
    read before fitting completes. ``evaluate_every`` remains accepted for API
    compatibility; the first-party trainer returns final-partition metrics.
    """
    manifest = Path(manifest)
    data, num_classes = _load_partitions(manifest)
    trainer = PartitionedDPGNN(
        DPGNNConfig(
            num_classes=num_classes, steps=steps, batch_size=batch_size,
            noise_multiplier=noise_multiplier, evaluate_every=evaluate_every,
            seed=seed, clip=clip, regression=regression,
            max_private_batch_nodes=max_private_batch_nodes,
        ),
    )
    trained = trainer.fit(data["train"], data["val"], data["test"])
    # Result keys are metric-named.  Outer keys stay fixed: upstream.py
    # requires validation_accuracy/test_accuracy, reused for whatever the
    # metric is.
    metric = trained["metric"]
    result = {
        "method": "dp_gnn",
        "metric": metric,
        "validation_accuracy": trained[f"validation_{metric}"],
        "test_accuracy": trained[f"test_{metric}"],
        "privacy": {
            "epsilon": trained["epsilon"],
            "delta": trained["delta"],
            "accountant": "first_party.dpgnn.multiterm_rdp",
            "noise_multiplier": noise_multiplier,
            "composition_count": steps,
        },
        "implementation": {
            "source": "src.training.dpgnn",
            "algorithm": [
                "reverse-edge bounded-degree sampling",
                "one-hop per-root gradients with uniform without-replacement root batches",
                "Opacus global per-root clipping and isotropic Gaussian DP-Adam",
                "multi-term hypergeometric RDP accounting",
            ],
        },
    }
    Path(result_path).write_text(json.dumps(result, indent=2) + "\n")
    return result
