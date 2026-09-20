"""Task losses, metrics, and label-only reference predictors."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

def _accuracy_and_macro_f1(logits: Tensor, labels: Tensor) -> tuple[float, float]:
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


def _task_loss(logits: Tensor, target: Tensor, multilabel: bool,
               regression: bool = False) -> Tensor:
    """The loss each config's label shape needs -- not a change to either
    method's private mechanism.

    DPAR's ISTA/PPR/propagation and the plain MLP/GraphSAGE clip-and-noise loop
    both operate on whatever gradient this loss produces; neither looks at the
    loss's type.  Swapping softmax cross-entropy for per-label BCE (or MSE)
    changes what task is being fit, not what either method does with the
    resulting gradient.
    """
    if regression:
        return F.mse_loss(logits.view(-1), target.view(-1).float())
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, target.float())
    return F.cross_entropy(logits, target)


def _task_metric(logits: Tensor, labels: Tensor, multilabel: bool,
                 regression: bool = False) -> tuple[float, float]:
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
