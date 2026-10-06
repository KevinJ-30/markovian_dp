"""Resolve authoritative dataset task metadata for experiment workers."""

from __future__ import annotations

from typing import Any


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
