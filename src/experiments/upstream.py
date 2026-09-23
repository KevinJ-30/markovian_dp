"""Safe bridge for retained upstream baseline implementations.

Each upstream project has its own CLI and accountant. The bridge exports the
same graph-disjoint partitions, invokes a caller-specified upstream command,
and requires one normalized JSON result rather than scraping terminal output.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from typing import Any

import torch


UPSTREAM_METHODS = {

    "progap": {
        "repository": "https://github.com/sisaman/ProGAP",
        "revision": "3ccad59e29e49949b8f0984381a6e6e5d5257cdf",
        "local_source": "third_party/ProGAP",
    },

    "heterpoisson": {
        "repository": "https://github.com/zihangxiang/PNPiGNNs",
        "revision": "9a06332147532d0cd163b484c95d4e347ff1c285",
        "local_source": "third_party/PNPiGNNs/Preserving_Node_level_Privacy_in_Graph_Neural_Networks",
    },
}

_PROGAP_TASK_ENVIRONMENT = frozenset({
    "PROGAP_BINARY",
    "PROGAP_PRIMARY_METRIC",
    "PROGAP_METRIC_IGNORE_LABEL",
})


def _task_metadata(split: Any) -> dict[str, Any]:
    """Return the resolved task contract carried by an inductive split."""
    primary_metric = getattr(split, "primary_metric", "accuracy")
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError("split.primary_metric must be a nonempty string")
    binary = getattr(split, "binary", False)
    if not isinstance(binary, bool):
        raise ValueError("split.binary must be boolean")
    metric_ignore_label = getattr(split, "metric_ignore_label", None)
    if (
        metric_ignore_label is not None
        and (isinstance(metric_ignore_label, bool) or not isinstance(metric_ignore_label, int))
    ):
        raise ValueError("split.metric_ignore_label must be an integer or null")
    domain_split = getattr(split, "domain_split", None)
    if domain_split is not None:
        domain_split = dict(domain_split)
    domain_split_id = getattr(split, "domain_split_id", None)
    if domain_split_id is not None and not isinstance(domain_split_id, str):
        raise ValueError("split.domain_split_id must be a string or null")
    return {
        "primary_metric": primary_metric,
        "binary": binary,
        "metric_ignore_label": metric_ignore_label,
        "domain_split": domain_split,
        "domain_split_id": domain_split_id,
    }


def _export_data(partition: Any) -> Any:
    """Copy one partition to CPU and attach its local scoring mask."""
    data = partition.data.clone().cpu()
    eval_mask = getattr(partition, "eval_mask", getattr(data, "eval_mask", None))
    if eval_mask is None:
        eval_mask = torch.ones(int(data.num_nodes), dtype=torch.bool)
    if (
        not isinstance(eval_mask, torch.Tensor)
        or eval_mask.dtype != torch.bool
        or eval_mask.ndim != 1
        or eval_mask.numel() != int(data.num_nodes)
    ):
        raise ValueError("partition eval_mask must be a local boolean node mask")
    data.eval_mask = eval_mask.detach().cpu().clone()
    return data




def export_partitions(split: Any, destination: str | Path) -> Path:
    """Export CPU PyG partitions and their resolved task contract."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 2,
        "num_classes": int(split.num_classes),
        **_task_metadata(split),
        "partitions": {},
    }
    for name in ("train", "val", "test"):
        partition = getattr(split, name)
        path = destination / f"{name}.pt"
        torch.save({
            "data": _export_data(partition),
            "node_ids": partition.node_ids.detach().cpu(),
            "statistics": dict(partition.stats),
        }, path)
        manifest["partitions"][name] = path.name
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination / "manifest.json"


def _finite_positive(value: Any, name: str) -> float:
    """Return a finite, strictly positive numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not value > 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _target_environment(
    method: str,
    config: dict[str, Any],
    configured_env: dict[str, Any],
    task_metadata: dict[str, Any],
) -> dict[str, str]:
    """Encode method-owned controls without ambient-environment aliases."""
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
    if method == "progap":
        conflicting_task = _PROGAP_TASK_ENVIRONMENT & set(configured_env)
        if conflicting_task:
            raise ValueError(
                "progap task settings are resolved from the dataset, not environment: "
                f"{sorted(conflicting_task)}"
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
        supported = {
            "target_epsilon", "target_delta", "epochs", "batch_size", "max_degree",
            "depth", "multilabel", "hidden_dim", "dropout", "optimizer",
            "learning_rate", "weight_decay", "max_grad_norm", "eval_chunk_size",
        }
        unknown = set(parameters) - supported
        if unknown:
            raise ValueError(f"Unsupported ProGAP parameters: {sorted(unknown)}")
        optional = {
            "epochs": "PROGAP_EPOCHS",
            "batch_size": "PROGAP_BATCH_SIZE",
            "max_degree": "PROGAP_MAX_DEGREE",
            "depth": "PROGAP_DEPTH",
        }
        for parameter, environment in optional.items():
            if parameter in parameters:
                encoded[environment] = str(parameters[parameter])
        positive = {
            "hidden_dim": "PROGAP_HIDDEN_DIM",
            "learning_rate": "PROGAP_LEARNING_RATE",
            "max_grad_norm": "PROGAP_MAX_GRAD_NORM",
            "eval_chunk_size": "PROGAP_EVAL_CHUNK_SIZE",
        }
        for parameter, environment in positive.items():
            if parameter in parameters:
                value = _finite_positive(parameters[parameter], f"parameters.{parameter}")
                if parameter in {"hidden_dim", "eval_chunk_size"}:
                    if not value.is_integer():
                        raise ValueError(f"parameters.{parameter} must be an integer")
                    value = int(value)
                encoded[environment] = str(value)
        for parameter, environment in {
            "dropout": "PROGAP_DROPOUT",
            "weight_decay": "PROGAP_WEIGHT_DECAY",
        }.items():
            if parameter not in parameters:
                continue
            value = parameters[parameter]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"parameters.{parameter} must be a finite nonnegative number")
            value = float(value)
            if not math.isfinite(value) or value < 0 or (parameter == "dropout" and value >= 1):
                raise ValueError(f"parameters.{parameter} is outside its supported range")
            encoded[environment] = str(value)
        if "optimizer" in parameters:
            optimizer = parameters["optimizer"]
            if not isinstance(optimizer, str) or optimizer not in {"adam", "sgd"}:
                raise ValueError("parameters.optimizer must be 'adam' or 'sgd'")
            encoded["PROGAP_OPTIMIZER"] = optimizer
        encoded["PROGAP_MULTILABEL"] = "1" if parameters.get("multilabel", False) else "0"
        encoded["PROGAP_BINARY"] = "1" if task_metadata["binary"] else "0"
        encoded["PROGAP_PRIMARY_METRIC"] = task_metadata["primary_metric"]
        if task_metadata["metric_ignore_label"] is not None:
            encoded["PROGAP_METRIC_IGNORE_LABEL"] = str(task_metadata["metric_ignore_label"])
        return encoded
    if "degree_bound" in parameters:
        raise ValueError(
            "parameters.degree_bound is retired; the HeterPoisson bound is derived from the training population"
        )
    if "HETERPOISSON_DEGREE_BOUND" in configured_env or "HETERPOISSON_DEGREE_BOUND" in os.environ:
        raise ValueError(
            "HETERPOISSON_DEGREE_BOUND is retired; the bound is derived from the training population"
        )
    required = {
        "epochs": "HETERPOISSON_EPOCHS",
        "expected_batchsize": "HETERPOISSON_EXPECTED_BATCHSIZE",
        "K": "HETERPOISSON_K",
        "num_neighbors": "HETERPOISSON_NUM_NEIGHBORS",
        "clip_norm": "HETERPOISSON_CLIP_NORM",
        "learning_rate": "HETERPOISSON_LEARNING_RATE",
    }
    for parameter, environment in required.items():
        if parameter not in parameters:
            raise ValueError(f"heterpoisson requires parameters.{parameter}")
        encoded[environment] = str(parameters[parameter])
    # Swaps criterion and head width only; not seen by get_std_node_dp.
    if parameters.get("regression", False):
        encoded["HETERPOISSON_REGRESSION"] = "1"
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
        task_metadata = _task_metadata(split)
        with tempfile.TemporaryDirectory(prefix=f"{self.method}-partitions-") as temporary:
            manifest = export_partitions(split, temporary)
            result_path = Path(temporary) / "result.json"
            configured_env = self.config.get("environment", {})
            if not isinstance(configured_env, dict) or not all(
                    isinstance(key, str) and isinstance(value, (str, int, float))
                    for key, value in configured_env.items()):
                raise ValueError("environment must be a string-keyed configuration mapping")
            target_environment = _target_environment(
                self.method, self.config, configured_env, task_metadata
            )
            inherited_environment = os.environ
            if self.method == "progap":
                inherited_environment = {
                    key: value for key, value in os.environ.items()
                    if key not in _PROGAP_TASK_ENVIRONMENT
                }
            env = {
                **inherited_environment,
                **{key: str(value) for key, value in configured_env.items()},
                **target_environment,
                "PARTITION_MANIFEST": str(manifest.resolve()),
                "RESULT_PATH": str(result_path.resolve()),
                "PYTHON": sys.executable,
            }
            subprocess.run(command, cwd=source, env=env, check=True)
            if not result_path.exists():
                raise RuntimeError(f"{self.method} adapter did not write {result_path}")
            result = json.loads(result_path.read_text())
        binary_result = self.method == "progap" and task_metadata["binary"]
        if binary_result:
            metric = task_metadata["primary_metric"]
            required = {
                "metric", f"validation_{metric}", f"test_{metric}", "privacy",
            }
            if result.get("metric") != metric:
                raise ValueError(
                    f"{self.method} binary result must report metric {metric!r}"
                )
            legacy_metric_fields = {"validation_accuracy", "test_accuracy"} & set(result)
            if legacy_metric_fields:
                raise ValueError(
                    f"{self.method} binary result must not store {metric} in accuracy fields "
                    f"{sorted(legacy_metric_fields)}"
                )
        else:
            required = {"validation_accuracy", "test_accuracy", "privacy"}
        missing = required - set(result)
        if self.method in {"progap", "heterpoisson"}:
            if binary_result:
                normalized = {
                    "metric",
                    f"validation_{task_metadata['primary_metric']}",
                    f"test_{task_metadata['primary_metric']}",
                    "privacy",
                    "calibration",
                }
            else:
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
