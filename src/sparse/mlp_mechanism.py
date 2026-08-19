"""
MLPMechanism: a graph-blind base mechanism g0 for the Stage-0 baseline.

The MLP ignores all edges — it classifies each root node from its own features
alone.  It slots into the exact same SparseGNN engine (per-root loss, DP
clip+noise) so the baseline is measured through the identical training loop as
the GNN; only the model differs.  Because edges are irrelevant, run it with
``--r 0`` (no expansion): each sampled root's subgraph is then just the root
itself, and ``subgraph_loss`` uses ``data.x[root]``.

Evaluation is a standard full-batch forward pass on the node features, reporting
train/val/test accuracy on whatever masks the data carries (identical protocol
to GNNMechanism.evaluate, so the numbers are directly comparable).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism


class _MLP(nn.Module):
    """Plain feed-forward classifier: (Linear-ReLU-Dropout) x (L-1) -> Linear."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
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
        return F.log_softmax(x, dim=1)


class MLPMechanism(BaseMechanism):
    """Graph-blind per-root MLP node-classification base mechanism.

    Same constructor signature as GNNMechanism so run.py can swap them freely.
    """

    def __init__(self, data, num_features, num_classes, *, hidden=64,
                 num_layers=2, dropout=0.5, device=None):
        module = _MLP(num_features, hidden, num_classes,
                      dropout=dropout, num_layers=num_layers)
        super().__init__(module, device=device)
        self.data = data
        self._train_mask = data.train_mask

    def subgraph_loss(self, subgraph) -> torch.Tensor:
        root = subgraph.root
        # Roots without a training label contribute nothing to the objective.
        if not bool(self._train_mask[root]):
            return self.zero_loss()

        x = self.data.x[root:root + 1].to(self.device)   # [1, num_features]
        out = self.module(x)                              # [1, num_classes]
        root_y = self.data.y[root].view(1)
        return F.nll_loss(out, root_y)

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        out = self.module(data.x)
        pred = out.argmax(dim=1)
        accs = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            n = int(mask.sum().item())
            accs[split] = (float((pred[mask] == data.y[mask]).sum().item()) / n
                           if n else float("nan"))
        return accs
