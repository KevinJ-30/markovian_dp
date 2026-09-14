"""
MLPMechanism: a graph-blind base mechanism g0 for the Stage-0 baseline.

The MLP ignores all edges — it classifies each root node from its own features
alone.  It slots into the exact same SparseGNN engine (per-root loss, DP
clip+noise) so the baseline is measured through the identical training loop as
the GNN; only the model differs.  Because edges are irrelevant, run it with
``--r 0`` (no expansion): each sampled root's subgraph is then just the root
itself, and ``subgraph_loss`` uses ``data.x[root]``.

THIS is the graph-blind arm, not ``--model gnn --r 0``.  At r=0 a GNN's
aggregation is identically zero, so its neighbour weight receives no gradient
and never trains — but ``evaluate`` still runs message passing over a real
graph, multiplying real neighbour means by that untrained (under DP, pure
noise) weight.  An MLP has no neighbour weight at all, so it is blind at
training AND at evaluation, and scores identically on any graph.

Handles single-label and MULTILABEL targets.  Multilabel support exists so that
multilabel datasets (PPI, Yelp, AmazonProducts) have a genuinely blind
comparator; without it the only r=0 option was the GNN mechanism above, which
is the trap this docstring opens with.

Evaluation is a standard full-batch forward pass on the node features, using the
same metric as the corresponding GNN mechanism (accuracy for single-label,
micro-F1 + micro-AUROC for multilabel), so the numbers are directly comparable.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism


class _MLP(nn.Module):
    """Plain feed-forward net: (Linear-ReLU-Dropout) x (L-1) -> Linear.

    Returns log-softmax for single-label and RAW LOGITS for multilabel, matching
    GNNMechanism and MultiLabelGNNMechanism respectively (the multilabel loss
    fuses the sigmoid via binary_cross_entropy_with_logits).
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2, multilabel=False):
        super().__init__()
        self.dropout = dropout
        self.multilabel = multilabel
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.lins = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x if self.multilabel else F.log_softmax(x, dim=1)


class MLPMechanism(BaseMechanism):
    """Graph-blind per-root MLP node-classification base mechanism.

    Same constructor signature as GNNMechanism so run.py can swap them freely.
    `multilabel` is inferred from the shape of `data.y` when not given.
    """

    def __init__(self, data, num_features, num_classes, *, hidden=64,
                 num_layers=2, dropout=0.5, device=None, multilabel=None):
        if multilabel is None:
            multilabel = data.y.dim() > 1 and data.y.size(-1) > 1
        module = _MLP(num_features, hidden, num_classes,
                      dropout=dropout, num_layers=num_layers,
                      multilabel=multilabel)
        super().__init__(module, device=device)
        self.data = data
        self.multilabel = bool(multilabel)
        self.metric_name = 'micro_f1' if self.multilabel else 'accuracy'
        self._train_mask = data.train_mask

    def subgraph_loss(self, subgraph) -> torch.Tensor:
        root = subgraph.root
        # Roots without a training label contribute nothing to the objective.
        if not bool(self._train_mask[root]):
            return self.zero_loss()

        x = self.data.x[root:root + 1].to(self.device)   # [1, num_features]
        out = self.module(x)                              # [1, num_classes]
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(
                out[0], self.data.y[root].float())
        return F.nll_loss(out, self.data.y[root].view(1))

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        out = self.module(data.x)
        if self.multilabel:
            # Same metrics and thresholds as MultiLabelGNNMechanism so the blind
            # arm and the GNN arm are scored identically.
            from .multilabel_mechanism import _micro_f1, _micro_auroc
            pred = (out > 0).float()
            accs = {}
            for split in ("train", "val", "test"):
                mask = getattr(data, f"{split}_mask")
                if not int(mask.sum().item()):
                    accs[split] = float("nan")
                    accs[f"{split}_auroc"] = float("nan")
                    continue
                accs[split] = _micro_f1(pred[mask], data.y[mask].float())
                accs[f"{split}_auroc"] = _micro_auroc(out[mask],
                                                      data.y[mask].float())
            return accs
        pred = out.argmax(dim=1)
        accs = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            n = int(mask.sum().item())
            accs[split] = (float((pred[mask] == data.y[mask]).sum().item()) / n
                           if n else float("nan"))
        return accs
