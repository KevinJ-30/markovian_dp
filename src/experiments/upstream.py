"""Safe bridge for retained upstream baseline implementations.

Each upstream project has its own CLI and accountant. The bridge exports the
same graph-disjoint partitions, invokes a caller-specified upstream command,
and requires one normalized JSON result rather than scraping terminal output.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from typing import Any

import torch


UPSTREAM_METHODS = {
    "dp_gnn": {
        "repository": "https://github.com/google-research/google-research/tree/master/differentially_private_gnns",
        "revision": "4fde028f6017e16aefcbc2b6d3f77f70b9f6b421",
        "local_source": "../google-research/differentially_private_gnns",
    },
    "progap": {
        "repository": "https://github.com/sisaman/ProGAP",
        "revision": "3ccad59e29e49949b8f0984381a6e6e5d5257cdf",
        "local_source": "third_party/ProGAP",
    },
    "sagd": {
        "repository": "https://zenodo.org/records/18401643",
        "revision": "zenodo-18401643",
        "local_source": None,
    },
    "heterpoisson": {
        "repository": "https://github.com/zihangxiang/PNPiGNNs",
        "revision": "9a06332147532d0cd163b484c95d4e347ff1c285",
        "local_source": "third_party/PNPiGNNs/Preserving_Node_level_Privacy_in_Graph_Neural_Networks",
    },
}


def export_partitions(split: Any, destination: str | Path) -> Path:
    """Export CPU PyG partitions for an upstream adapter without cross edges."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"format": 1, "num_classes": split.num_classes, "partitions": {}}
    for name in ("train", "val", "test"):
        partition = getattr(split, name)
        path = destination / f"{name}.pt"
        torch.save({"data": partition.data.cpu(), "node_ids": partition.node_ids.cpu(),
                    "statistics": dict(partition.stats)}, path)
        manifest["partitions"][name] = path.name
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination / "manifest.json"


def _finite_positive(value: Any, name: str) -> float:
    """Return a finite, strictly positive numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not value > 0 or not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _target_environment(method: str, config: dict[str, Any], configured_env: dict[str, Any]) -> dict[str, str]:
    """Encode method-owned privacy controls without ambient-environment aliases."""
    if method not in {"progap", "heterpoisson"}:
        return {}
    prefix = method.upper()
    prohibited = {f"{prefix}_TARGET_EPSILON", f"{prefix}_TARGET_DELTA"}
    if method == "progap":
        prohibited.add("PROGAP_EPSILON")
    conflicting = prohibited & set(configured_env)
    if conflicting:
        raise ValueError(
            f"{method} privacy targets must be configured in parameters, not environment: "
            f"{sorted(conflicting)}"
        )
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a configuration mapping")
    epsilon = _finite_positive(parameters.get("target_epsilon"), "parameters.target_epsilon")
    delta = _finite_positive(parameters.get("target_delta"), "parameters.target_delta")
    if delta >= 1:
        raise ValueError("parameters.target_delta must be less than one")
    encoded = {
        f"{prefix}_TARGET_EPSILON": str(epsilon),
        f"{prefix}_TARGET_DELTA": str(delta),
        f"{prefix}_SEED": str(int(config.get("seed", 0))),
    }
    if method == "progap":
        optional = {
            "epochs": "PROGAP_EPOCHS",
            "batch_size": "PROGAP_BATCH_SIZE",
            "max_degree": "PROGAP_MAX_DEGREE",
            "depth": "PROGAP_DEPTH",
        }
        for parameter, environment in optional.items():
            if parameter in parameters:
                encoded[environment] = str(parameters[parameter])
        return encoded
    required = {
        "epochs": "HETERPOISSON_EPOCHS",
        "expected_batchsize": "HETERPOISSON_EXPECTED_BATCHSIZE",
        "K": "HETERPOISSON_K",
        "num_neighbors": "HETERPOISSON_NUM_NEIGHBORS",
        "clip_norm": "HETERPOISSON_CLIP_NORM",
        "learning_rate": "HETERPOISSON_LEARNING_RATE",
        "degree_bound": "HETERPOISSON_DEGREE_BOUND",
    }
    for parameter, environment in required.items():
        if parameter not in parameters:
            raise ValueError(f"heterpoisson requires parameters.{parameter}")
        encoded[environment] = str(parameters[parameter])
    return encoded


class UpstreamBaseline:
    """Run an upstream implementation via an explicit partition-aware adapter.

    ``command`` must write ``result.json`` to the provided ``RESULT_PATH``. It
    is deliberately explicit: none of the upstream CLIs can safely consume a
    concatenated graph without reintroducing held-out graph data into private
    preprocessing. A shell is never used; command values are argv tokens.
    """

    def __init__(self, method: str, config: dict[str, Any]):
        if method not in UPSTREAM_METHODS:
            raise ValueError(f"unsupported upstream method {method!r}; expected {sorted(UPSTREAM_METHODS)}")
        self.method = method
        self.config = config

    def run(self, split: Any) -> dict[str, Any]:
        command = self.config.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError(
                f"{self.method} requires a partition-aware command argv list. It receives "
                "PARTITION_MANIFEST and must write RESULT_PATH JSON; unsafe monolithic upstream "
                "CLIs are intentionally not run against held-out graphs."
            )
        source = Path(self.config.get("source_dir") or UPSTREAM_METHODS[self.method]["local_source"] or ".")
        if not source.exists():
            raise FileNotFoundError(f"upstream source directory not found: {source}")
        with tempfile.TemporaryDirectory(prefix=f"{self.method}-partitions-") as temporary:
            manifest = export_partitions(split, temporary)
            result_path = Path(temporary) / "result.json"
            configured_env = self.config.get("environment", {})
            if not isinstance(configured_env, dict) or not all(
                    isinstance(key, str) and isinstance(value, (str, int, float))
                    for key, value in configured_env.items()):
                raise ValueError("environment must be a string-keyed configuration mapping")
            env = {
                **os.environ,
                **{key: str(value) for key, value in configured_env.items()},
                **_target_environment(self.method, self.config, configured_env),
                "PARTITION_MANIFEST": str(manifest.resolve()),
                "RESULT_PATH": str(result_path.resolve()),
                "PYTHON": sys.executable,
            }
            subprocess.run(command, cwd=source, env=env, check=True)
            if not result_path.exists():
                raise RuntimeError(f"{self.method} adapter did not write {result_path}")
            result = json.loads(result_path.read_text())
        required = {"validation_accuracy", "test_accuracy", "privacy"}
        missing = required - set(result)
        if self.method in {"progap", "heterpoisson"}:
            normalized = {
                "validation_accuracy", "validation_macro_f1",
                "test_accuracy", "test_macro_f1", "privacy", "calibration",
            }
            absent = normalized - set(result)
            if absent:
                raise ValueError(f"{self.method} result omits normalized fields {sorted(absent)}")
            total = result["privacy"].get("total") if isinstance(result["privacy"], dict) else None
            total_fields = {
                "epsilon", "delta", "accountant", "noise_multiplier",
                "sampling_probability", "composition_count", "parameters",
            }
            if not isinstance(total, dict) or total_fields - set(total):
                raise ValueError(f"{self.method} result omits normalized privacy.total fields")
            calibration_fields = {"target_epsilon", "target_delta", "achieved_epsilon", "noise_std"}
            if not isinstance(result["calibration"], dict) or calibration_fields - set(result["calibration"]):
                raise ValueError(f"{self.method} result omits normalized calibration fields")
        if missing:
            raise ValueError(f"{self.method} result omits standardized fields {sorted(missing)}")
        result["method"] = self.method
        result["upstream"] = UPSTREAM_METHODS[self.method]
        return result
