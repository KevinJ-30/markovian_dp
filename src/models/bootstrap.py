"""Node-wise percentile intervals conditional on fixed test predictions.

Repeated node types are collapsed before multinomial resampling. This is exactly
sampling N nodes with replacement, without allocating a resamples-by-N array or
sorting AUROC scores again for every replicate. No training RNG is consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class BootstrapConfig:
    confidence_level: float = 0.95
    n_resamples: int = 1000
    seed: int = 0

    def __post_init__(self):
        if (isinstance(self.confidence_level, bool)
                or not isinstance(self.confidence_level, Real)
                or not 0 < self.confidence_level < 1):
            raise ValueError("bootstrap confidence must be a finite fraction in (0, 1)")
        if (isinstance(self.n_resamples, bool)
                or not isinstance(self.n_resamples, Integral)
                or self.n_resamples < 0):
            raise ValueError("bootstrap resamples must be a nonnegative integer")
        if (isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
                or self.seed < 0):
            raise ValueError("bootstrap seed must be a nonnegative integer")


_METRICS = {
    "accuracy": ("accuracy", "macro_f1"),
    "auroc": ("auroc", "accuracy"),
    "micro_f1": ("micro_f1",),
    "r2": ("r2",),
}


class BootstrapMetrics:
    """Accumulate already-masked test logits as compact CPU node statistics.

    ``inclusive_threshold`` preserves evaluators using >= 0 rather than > 0.
    ``zero_division`` preserves the evaluator's undefined micro-F1 convention.
    Undefined bootstrap draws are excluded per metric and counted, not redrawn.
    An undefined original score always yields null interval endpoints.
    """

    def __init__(self, task: str, config: BootstrapConfig, *,
                 metrics: Sequence[str] | None = None,
                 inclusive_threshold: bool = False,
                 zero_division: float = float("nan")):
        if task not in _METRICS:
            raise ValueError(f"unsupported bootstrap task: {task!r}")
        self.task = task
        self.config = config
        self.metrics = tuple(_METRICS[task] if metrics is None else metrics)
        supported = _METRICS[task] + (("micro_auroc",) if task == "micro_f1" else ())
        if not self.metrics or len(set(self.metrics)) != len(self.metrics) or any(
                metric not in supported for metric in self.metrics):
            raise ValueError(f"unsupported bootstrap metrics for {task}: {self.metrics}")
        self.inclusive_threshold = inclusive_threshold
        self.zero_division = zero_division
        self._chunks: list[np.ndarray] = []
        self._auc_chunks: list[tuple[np.ndarray, np.ndarray]] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("bootstrap predictions and targets must have equal row counts")
        if not self.config.n_resamples or not labels.shape[0]:
            return
        if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
            raise ValueError("bootstrap predictions and targets must be finite")
        if self.task == "accuracy":
            rows = torch.stack((labels.reshape(-1).long(), logits.argmax(-1)), dim=1)
        elif self.task == "micro_f1":
            positive = logits >= 0 if self.inclusive_threshold else logits > 0
            actual = labels.bool()
            rows = torch.stack(((positive & actual).sum(1), positive.sum(1), actual.sum(1)), dim=1)
            if "micro_auroc" in self.metrics:
                self._auc_chunks.append((logits.detach().cpu().numpy(), actual.cpu().numpy()))
        elif self.task == "auroc":
            scores, targets = logits.reshape(-1), labels.reshape(-1)
            if not ((targets == 0) | (targets == 1)).all():
                raise ValueError("bootstrap AUROC targets must be binary")
            rows = torch.stack((scores.double(), targets.double()), dim=1)
        else:
            # Keep targets, not a global SST: every resample has its own mean.
            targets = labels.reshape(-1).double()
            residual = logits.reshape(-1).double() - targets
            rows = torch.stack((targets, residual.square()), dim=1)
        self._chunks.append(rows.detach().cpu().numpy())

    def _micro_auroc(self):
        scores = np.concatenate([scores for scores, _ in self._auc_chunks])
        targets = np.concatenate([targets for _, targets in self._auc_chunks])
        n_labels = scores.shape[1]
        flat = scores.reshape(-1)
        order = np.argsort(flat, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(flat[order])) + 1]
        positive = targets.reshape(-1)[order]
        nodes = order // n_labels

        def statistic(weights):
            # All label entries from one node share the SAME bootstrap weight.
            entry_weights = weights[nodes]
            pos = np.add.reduceat(entry_weights * positive, starts)
            neg = np.add.reduceat(entry_weights * ~positive, starts)
            pairs = float(pos.sum()) * float(neg.sum())
            favorable = (pos * (np.cumsum(neg) - 0.5 * neg)).sum()
            return float(favorable / pairs) if pairs else float("nan")
        return statistic

    def _statistic(self, rows: np.ndarray):
        """Build a weighted metric reducer once, outside the resampling loop."""
        if self.task == "accuracy":
            classes, inverse = np.unique(rows[:, :2], return_inverse=True)
            actual, predicted = inverse.reshape(-1, 2).T
            correct = actual == predicted

            def statistic(weights):
                result = {"accuracy": float(weights[correct].sum() / weights.sum())}
                if "macro_f1" in self.metrics:
                    truth = np.bincount(actual, weights=weights, minlength=len(classes))
                    prediction = np.bincount(predicted, weights=weights, minlength=len(classes))
                    tp = np.bincount(actual[correct], weights=weights[correct], minlength=len(classes))
                    present = truth > 0
                    result["macro_f1"] = float((2 * tp[present] / (truth + prediction)[present]).mean())
                return result
        elif self.task == "micro_f1":
            auc = self._micro_auroc() if "micro_auroc" in self.metrics else None
            def statistic(weights):
                tp, predicted, actual = (rows * weights[:, None]).sum(0)
                denominator = predicted + actual
                result = {"micro_f1": float(2 * tp / denominator) if denominator else self.zero_division}
                if auc is not None:
                    result["micro_auroc"] = auc(weights)
                return result
        elif self.task == "auroc":
            # Rows are lexicographically sorted by np.unique: contiguous ties.
            starts = np.r_[0, np.flatnonzero(np.diff(rows[:, 0])) + 1]
            positive = rows[:, 1] == 1
            predicted = rows[:, 0] >= 0 if self.inclusive_threshold else rows[:, 0] > 0
            correct = predicted == positive

            def statistic(weights):
                pos = np.add.reduceat(weights * positive, starts)
                neg = np.add.reduceat(weights * ~positive, starts)
                pairs = float(pos.sum()) * float(neg.sum())
                favorable = (pos * (np.cumsum(neg) - 0.5 * neg)).sum()
                return {"auroc": float(favorable / pairs) if pairs else float("nan"),
                        "accuracy": float(weights[correct].sum() / weights.sum())}
        else:
            # Offset before variance reduction to retain large-offset precision.
            target = rows[:, 0] - rows[0, 0]
            residual_square = rows[:, 1]

            def statistic(weights):
                n = weights.sum()
                if n < 2:
                    return {"r2": float("nan")}
                mean = (weights * target).sum() / n
                sst = (weights * (target - mean) ** 2).sum()
                sse = (weights * residual_square).sum()
                return {"r2": float(1 - sse / sst) if sst else float(sse == 0)}
        return statistic

    def compute(self) -> dict:
        config = self.config
        n = sum(chunk.shape[0] for chunk in self._chunks)
        values = {metric: [] for metric in self.metrics}
        original = {metric: float("nan") for metric in self.metrics}
        if n and config.n_resamples:
            if "micro_auroc" in self.metrics:
                rows = np.concatenate(self._chunks)
                counts = np.ones(n, dtype=np.int64)
            else:
                # Identical sufficient statistics represent exchangeable node draws.
                rows, counts = np.unique(np.concatenate(self._chunks), axis=0, return_counts=True)
            statistic = self._statistic(rows)
            original = statistic(counts)
            rng = np.random.default_rng(config.seed)
            probabilities = counts / n
            for _ in range(config.n_resamples):
                scores = statistic(rng.multinomial(n, probabilities))
                for metric in self.metrics:
                    if np.isfinite(scores[metric]):
                        values[metric].append(scores[metric])
        alpha = (1 - config.confidence_level) / 2
        intervals = {}
        for metric, samples in values.items():
            lower = upper = None
            if samples and np.isfinite(original[metric]):
                lower, upper = map(float, np.quantile(samples, [alpha, 1 - alpha]))
            intervals[metric] = {"lower": lower, "upper": upper,
                                 "valid_resamples": len(samples)}
        return {"method": "percentile", "confidence_level": config.confidence_level,
                "n_resamples": config.n_resamples, "seed": config.seed,
                "resampling_unit": "node", "n_observations": n, "metrics": intervals}
