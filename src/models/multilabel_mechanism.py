"""
MultiLabelGNNMechanism: the base mechanism g0 for multilabel node classification.

Identical in structure to `GNNMechanism` — an L-layer GCN run on each root's
sparsified subgraph, reading off the root's output — but the label of a root is
a 0/1 vector rather than a class index, so the per-root loss is
`binary_cross_entropy_with_logits` and the reported metric is micro-F1.

Use this mechanism for multilabel datasets such as GraphSAINT PPI-large
(121 binary labels per node).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import PaddedGNNStack, build_conv_stack


class _MultiLabelGNN(nn.Module):
    """L-layer message-passing stack returning raw logits (labels are not 1-of-K)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2, aggr='mean'):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = build_conv_stack(dims, aggr=aggr)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def _micro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Micro-averaged F1 over all (node, label) pairs, the standard PPI metric."""
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float('nan')


def _micro_auroc(scores: torch.Tensor, target: torch.Tensor) -> float:
    """Micro-averaged AUROC over all (node, label) pairs.

    Reported alongside micro-F1 because micro-F1 is DEGENERATE on PPI: the
    all-positive predictor scores 0.4608 — above everything DP training reaches
    — while having no ranking ability at all (AUROC 0.5).  F1 reads a fixed
    logit>0 threshold, and DP noise decalibrates that threshold far more than
    it damages the ranking, so micro-F1 understates a private model.  AUROC is
    threshold-free and therefore the honest comparison at low epsilon.
    """
    import numpy as np
    s = scores.reshape(-1).detach().cpu().numpy().astype(np.float64)
    t = target.reshape(-1).detach().cpu().numpy()
    pos = t == 1
    n_pos, n_neg = int(pos.sum()), int((t == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within ties, matching binary_mechanism._auroc.  Without
    # this a constant predictor scores != 0.5 (measured 0.4988 on a
    # PPI-shaped target), which is where README's "AUROC 0.4955" floor came
    # from.  float64 because the flattened (node, label) pool is large.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class MultiLabelGNNMechanism(BaseMechanism):
    """Per-root GCN multilabel base mechanism.

    Same constructor signature as GNNMechanism so run.py can swap them freely.
    `num_classes` is the number of label columns.
    """

    metric_name = "micro_f1"

    def __init__(self, data, num_features, num_classes, *, hidden=64,
                 num_layers=2, dropout=0.5, aggr='mean', device=None):
        module = _MultiLabelGNN(num_features, hidden, num_classes,
                            dropout=dropout, num_layers=num_layers, aggr=aggr)
        super().__init__(module, device=device)
        self.data = data
        self._train_mask = data.train_mask

    def subgraph_loss(self, subgraph) -> torch.Tensor:
        root = subgraph.root
        # Roots outside the training split contribute nothing to the objective.
        if not bool(self._train_mask[root]):
            return self.zero_loss()

        x = self.data.x[subgraph.nodes.to(self.device)]
        edge_index = subgraph.edge_index.to(self.device)
        out = self.module(x, edge_index)          # [n_v, num_labels]
        # Local index 0 is the root by RootedSubgraph convention.
        root_logits = out[0:1]
        root_y = self.data.y[root].view(1, -1).float()
        return F.binary_cross_entropy_with_logits(root_logits, root_y)

    def build_private_module(self) -> nn.Module:
        return PaddedGNNStack(
            self.module.convs, dropout=self.module.dropout).to(self.device)

    def private_losses(self, private_module: nn.Module, batch) -> torch.Tensor:
        logits = private_module(
            batch.features, batch.edge_index, batch.edge_mask, batch.node_mask)
        rows = torch.arange(batch.batch_size, device=self.device)
        root_logits = logits[rows, batch.root_index]
        losses = F.binary_cross_entropy_with_logits(
            root_logits, batch.labels.float(), reduction="none").mean(dim=-1)
        return losses * batch.loss_mask.to(losses.dtype)

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        logits = self.module(data.x, self.eval_edges(data))
        pred = (logits > 0).float()
        metrics = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            n = int(mask.sum().item())
            metrics[split] = (_micro_f1(pred[mask], data.y[mask].float())
                              if n else float("nan"))
            metrics[f"{split}_auroc"] = (
                _micro_auroc(logits[mask], data.y[mask]) if n else float("nan"))
        return metrics
