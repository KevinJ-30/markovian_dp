"""Scalar node regression with MSE training and R² evaluation.

R² uses each evaluated split's own mean as its baseline, is higher-is-better,
and is invariant to a shared affine transformation of predictions and targets.
The label-only reference predictor instead predicts the training mean.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import PaddedGNNStack, build_conv_stack
from .objectives import _regression_r2


class _RegressionGNN(nn.Module):
    """L-layer message-passing stack emitting one unbounded value per node."""

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


class RegressionGNNMechanism(BaseMechanism):
    """Per-root GCN regressor.

    Same constructor signature as the other mechanisms so run.py can swap them
    freely; `num_classes` is accepted and ignored (always one output).
    """

    metric_name = "r2"

    def __init__(self, data, num_features, num_classes=1, *, hidden=64,
                num_layers=2, dropout=0.5, aggr='mean', device=None):
        module = _RegressionGNN(num_features, hidden, dropout=dropout,
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
        root_pred = out[0:1]
        root_y = self.data.y[root].view(1).float()
        return F.mse_loss(root_pred, root_y)

    def build_private_module(self) -> nn.Module:
        return PaddedGNNStack(
            self.module.convs, dropout=self.module.dropout).to(self.device)

    def private_losses(self, private_module: nn.Module, batch) -> torch.Tensor:
        predictions = private_module(
            batch.features, batch.edge_index, batch.edge_mask, batch.node_mask)
        rows = torch.arange(batch.batch_size, device=self.device)
        root_predictions = predictions[rows, batch.root_index, 0]
        losses = F.mse_loss(
            root_predictions, batch.labels.float().view(-1), reduction="none")
        return losses * batch.loss_mask.to(losses.dtype)

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        pred = self.module(data.x, self.eval_edges(data))
        target = data.y.float()
        metrics = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            metrics[split] = _regression_r2(pred[mask], target[mask])
        return metrics
