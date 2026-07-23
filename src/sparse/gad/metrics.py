"""
GADBench evaluation metrics (anomaly = positive class).

  AUROC  — area under ROC curve (overall ranking; insensitive to top-k).
  AUPRC  — area under precision-recall curve = average precision (balances the two).
  Rec@K  — recall among the top-K highest-scored, with K = #anomalies in the eval set
           (equals precision@K and F1@K at that K).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _as_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a)


def auroc(y_true, scores):
    return float(roc_auc_score(_as_numpy(y_true), _as_numpy(scores)))


def auprc(y_true, scores):
    return float(average_precision_score(_as_numpy(y_true), _as_numpy(scores)))


def rec_at_k(y_true, scores, k=None):
    """Recall of true anomalies among the top-k highest-scored samples.

    k defaults to the number of positives (anomalies) in y_true, matching GADBench.
    """
    y = _as_numpy(y_true).astype(int)
    s = _as_numpy(scores).astype(float)
    n_pos = int(y.sum())
    if k is None:
        k = n_pos
    if k <= 0 or n_pos == 0:
        return 0.0
    topk = np.argsort(-s)[:k]
    return float(y[topk].sum()) / float(n_pos)
