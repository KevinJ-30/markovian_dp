"""Task losses, metrics, and label-only reference predictors."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _binary_auroc(labels: Tensor, scores: Tensor) -> float:
    """Tie-correct rank AUROC, or NaN when either class is absent."""
    labels = torch.as_tensor(labels).detach().view(-1).cpu()
    scores = torch.as_tensor(scores).detach().view(-1).to(
        device="cpu", dtype=torch.float64)
    if labels.numel() != scores.numel():
        raise ValueError("binary AUROC labels and scores must have equal length")
    positive = labels == 1
    negative = labels == 0
    num_positive = int(positive.sum())
    num_negative = int(negative.sum())
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(
        1, scores.numel() + 1, dtype=torch.float64)
    _, inverse, counts = torch.unique(
        scores, return_inverse=True, return_counts=True)
    rank_sums = torch.zeros(
        counts.numel(), dtype=torch.float64).scatter_add_(0, inverse, ranks)
    tied_ranks = (rank_sums / counts)[inverse]
    return float(
        (tied_ranks[positive].sum()
         - num_positive * (num_positive + 1) / 2)
        / (num_positive * num_negative)
    )


def _metric_rows(
    logits: Tensor,
    labels: Tensor,
    *,
    eval_mask: Tensor | None = None,
    metric_ignore_label: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Select scored rows without changing the graph used for the forward pass."""
    if logits.size(0) != labels.size(0):
        raise ValueError("metric logits and labels must have equal leading dimensions")
    if eval_mask is None:
        selected = torch.ones(labels.size(0), dtype=torch.bool, device=labels.device)
    else:
        selected = torch.as_tensor(eval_mask, device=labels.device)
        if selected.dtype != torch.bool or selected.ndim != 1:
            raise ValueError("eval_mask must be a one-dimensional boolean tensor")
        if selected.numel() != labels.size(0):
            raise ValueError("eval_mask length must match the partition")
    if metric_ignore_label is not None:
        if labels.ndim != 1:
            raise ValueError("metric_ignore_label requires one-dimensional class labels")
        selected = selected & (labels != metric_ignore_label)
    return logits[selected], labels[selected]

def _accuracy_and_macro_f1(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    if labels.numel() == 0:
        return float("nan"), float("nan")
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean())
    f1s = []
    for label in torch.unique(labels):
        positive = predictions == label
        truth = labels == label
        denom = 2 * (positive & truth).sum() + (positive & ~truth).sum() + (~positive & truth).sum()
        f1s.append(float(2 * (positive & truth).sum() / denom) if denom else 0.0)
    return accuracy, sum(f1s) / len(f1s)


def _multilabel_micro_f1(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    """Micro-F1 over every (node, label) pair, thresholding logits at 0.

    Returned in both slots of the (accuracy, macro_f1) tuple both trainers
    already unpack: neither "accuracy" nor "macro-F1" means the same thing for
    a multi-hot target that it does for a single class index.  Micro-F1 is the
    metric src.models.multilabel_mechanism already reports for this exact
    label shape (PPI's 121 binary functional labels), kept identical here so a
    baseline and the SparseGNN mechanism are judged the same way.
    """
    predictions = (logits > 0).float()
    labels = labels.float()
    tp = float((predictions * labels).sum())
    fp = float((predictions * (1 - labels)).sum())
    fn = float(((1 - predictions) * labels).sum())
    denom = 2 * tp + fp + fn
    micro_f1 = 2 * tp / denom if denom > 0 else float("nan")
    return micro_f1, micro_f1


def _regression_mae(preds: Tensor, target: Tensor) -> tuple[float, float]:
    """MAE, returned in both slots of the (metric, secondary) tuple every
    trainer already unpacks -- matches src.models.regression_mechanism's
    metric, so a baseline and the SparseGNN mechanism are judged the same way.
    """
    mae = float((preds.view(-1) - target.view(-1).float()).abs().mean())
    return mae, mae


def _task_loss(
    logits: Tensor,
    target: Tensor,
    multilabel: bool,
    regression: bool = False,
    binary: bool = False,
) -> Tensor:
    """Return the loss for one resolved task."""
    if sum((bool(multilabel), bool(regression), bool(binary))) > 1:
        raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
    if regression:
        return F.mse_loss(logits.view(-1), target.view(-1).float())
    if binary:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), target.view(-1).float())
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, target.float())
    return F.cross_entropy(logits, target)


def _task_metric(
    logits: Tensor,
    labels: Tensor,
    multilabel: bool,
    regression: bool = False,
    binary: bool = False,
) -> tuple[float, float]:
    if sum((bool(multilabel), bool(regression), bool(binary))) > 1:
        raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
    if binary:
        scores = logits.view(-1)
        targets = labels.view(-1)
        auroc = _binary_auroc(targets, scores)
        accuracy = (
            float(((scores > 0).to(targets.dtype) == targets).float().mean())
            if targets.numel() else float("nan")
        )
        return auroc, accuracy
    if regression:
        return _regression_mae(logits, labels)
    return (_multilabel_micro_f1(logits, labels) if multilabel
            else _accuracy_and_macro_f1(logits, labels))


def trivial_baseline(data, metric):
    """Score of the best label-only predictor under the dataset's own metric.

      accuracy  -> most frequent training class, evaluated on test
      micro_f1  -> predict every label positive: 2p/(1+p) at positive rate p
      auroc     -> 0.5 by definition
      mae       -> MAE of "always predict the train mean" on test, i.e.
                   mean(|y_test - mean(y_train)|), in the SCALED space the
                   targets are stored in -- train-std units, matching
                   _regression_mae and RegressionGNNMechanism.evaluate.
                   Multiply by data.target_std for the label's own units.

                   NOTE: targets are scaled but NOT centred
                   (data.relbench.load_relbench divides by the train std and
                   leaves the mean alone), so the train mean is NOT 0 here.
                   An earlier version computed mean(|y_test|), the MAE of the
                   ALL-ZERO predictor -- a much weaker bar on the non-negative
                   heavy-tailed targets RelBench regression uses (LTV, sales),
                   so "beats trivial" was too easy to clear.

    Recorded in the CSV as the floor every result must clear (for mae, the
    ceiling every result must undercut -- see experiments.sparse._HIGHER_IS_BETTER).  Note
    micro_f1's floor is high but has no ranking ability (its AUROC is 0.5), so
    a model below it may still be learning — compare AUROC too.
    """
    import torch as _t
    if metric == "auroc":
        return 0.5
    y, te = data.y, data.test_mask
    if metric == "micro_f1":
        p = float(y[te].float().mean())
        return 2 * p / (1 + p) if p > 0 else float("nan")
    if metric == "mae":
        train_mean = float(y[data.train_mask].view(-1).float().mean())
        return float((y[te].view(-1).float() - train_mean).abs().mean())
    tr_counts = _t.bincount(y[data.train_mask].view(-1))
    majority = int(tr_counts.argmax())
    return float((y[te].view(-1) == majority).float().mean())
