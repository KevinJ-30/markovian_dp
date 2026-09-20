"""RegressionGNNMechanism: the base mechanism g0 for node/entity regression.

Same shape as `GNNMechanism` — an L-layer GCN on each root's sparsified
subgraph, read off at the root — but with a single unbounded output, MSE loss,
and MAE as the reported metric.

Targets are expected SCALED but NOT centred: `load_relbench` divides by the
train-split std and leaves the mean alone.  MAE and RMSE are reported in that
scaled space -- i.e. in train-std units -- matching `objectives._regression_mae`,
which the DPAR/DP-GNN/MLP baselines use.  Multiply by `data.target_std` to
recover the label's original units.  (R^2 is scale-invariant either way.)

Because the target is not centred, "predict the train mean" is NOT "predict 0"
— `objectives.py`'s trivial_baseline subtracts mean(y_train) explicitly — and this
head, which emits an unbounded scalar with no output transform, has to learn
that intercept itself.  Under per-root clipping at C the intercept competes for
gradient budget with the signal, so a large uncentred target is worth noticing.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import PaddedGNNStack, build_conv_stack


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

    metric_name = "mae"

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
            n = int(mask.sum().item())
            if not n:
                metrics[split] = float("nan")
                metrics[f"{split}_rmse"] = float("nan")
                metrics[f"{split}_r2"] = float("nan")
                continue
            residual = pred[mask] - target[mask]
            metrics[split] = float(residual.abs().mean())
            metrics[f"{split}_rmse"] = float(residual.pow(2).mean().sqrt())
            y_true = target[mask]
            ss_tot = (y_true - y_true.mean()).pow(2).sum()
            metrics[f"{split}_r2"] = (
                float(1.0 - residual.pow(2).sum() / ss_tot)
                if ss_tot > 0 else float("nan"))
        return metrics
