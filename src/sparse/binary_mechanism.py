"""
BinaryGNNMechanism: the base mechanism g0 for binary entity classification.

Same shape as `GNNMechanism` — an L-layer GCN on each root's sparsified
subgraph, read off at the root — but with a single output logit, BCE loss and
AUROC as the reported metric.

AUROC rather than accuracy because RelBench's binary entity tasks are heavily
imbalanced (rel-f1/driver-top3 has a ~17-20% positive rate, so a constant
predictor scores ~0.82 accuracy and tells you nothing).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import build_conv_stack


class _BinaryGNN(nn.Module):
    """L-layer message-passing stack emitting one logit per node."""

    def __init__(self, in_channels, hidden_channels, dropout=0.5,
                 num_layers=2, aggr='mean'):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [1]
        self.convs = build_conv_stack(dims, aggr=aggr)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.view(-1)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC; nan when a split has only one class."""
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so tied scores do not create spurious ordering.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class BinaryGNNMechanism(BaseMechanism):
    """Per-root GCN binary classifier.

    Same constructor signature as the other mechanisms so run.py can swap them
    freely; `num_classes` is accepted and ignored (always one logit).
    """

    metric_name = "auroc"

    def __init__(self, data, num_features, num_classes=2, *, hidden=64,
                 num_layers=2, dropout=0.5, aggr='mean', device=None):
        module = _BinaryGNN(num_features, hidden, dropout=dropout,
                            num_layers=num_layers, aggr=aggr)
        super().__init__(module, device=device)
        self.data = data
        self._train_mask = data.train_mask

    def subgraph_loss(self, subgraph) -> torch.Tensor:
        root = subgraph.root
        if not bool(self._train_mask[root]):
            return self.zero_loss()

        x = self.data.x[subgraph.nodes.to(self.device)]
        edge_index = subgraph.edge_index.to(self.device)
        out = self.module(x, edge_index)          # [n_v]
        root_logit = out[0:1]
        root_y = self.data.y[root].view(1).float()
        return F.binary_cross_entropy_with_logits(root_logit, root_y)

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        # Default (eval_edge_index unset): data.edge_index, which for RelBench
        # is the test-cutoff graph, so held-out rows keep real neighbourhoods.
        scores = self.module(data.x, self.eval_edges(data)).cpu().numpy()
        y = data.y.cpu().numpy()
        metrics = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask").cpu().numpy()
            metrics[split] = (_auroc(y[mask], scores[mask]) if mask.any()
                              else float("nan"))
        return metrics
