"""
GNNMechanism: the base mechanism g0 instantiated as a GNN node classifier.

For a rooted sparsified subgraph H = (V_v, E_v, F|_{V_v}), g0 runs an L-layer
GCN forward pass on H and returns the negative log-likelihood at the ROOT node
(local index 0) against its label.  This is the "node classification per-root"
choice: each sampled root contributes a single supervised loss term computed on
its own sparsified neighborhood, exactly matching G(y) = sum_v g0(y_v).

Evaluation is standard full-graph transductive inference on the (unsparsified)
graph, reporting train/val/test accuracy on the Planetoid masks.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import build_conv_stack


class _NodeGNN(nn.Module):
    """L-layer message-passing stack; see layers.build_conv_stack for the
    aggregator choice (SAGE-mean by default, GCN with aggr='gcn')."""

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
        return F.log_softmax(x, dim=1)


class GNNMechanism(BaseMechanism):
    """Per-root GCN node-classification base mechanism.

    Args:
        data:      PyG Data with x, y, and train/val/test masks (on `device`).
        hidden:    hidden width.
        num_layers: number of GCN layers L (== max SparseExpand distance r for a
                    faithful receptive field, though not enforced).
        dropout:   dropout probability.
        device:    torch device.
    """

    def __init__(self, data, num_features, num_classes, *, hidden=64,
                 num_layers=2, dropout=0.5, aggr='mean', device=None):
        module = _NodeGNN(num_features, hidden, num_classes,
                      dropout=dropout, num_layers=num_layers, aggr=aggr)
        super().__init__(module, device=device)
        self.data = data
        # Precompute which nodes carry a training label (only these produce a
        # non-zero g0 loss when sampled as roots).
        self._train_mask = data.train_mask

    def subgraph_loss(self, subgraph) -> torch.Tensor:
        root = subgraph.root
        # Roots without a training label contribute nothing to the objective.
        if not bool(self._train_mask[root]):
            return self.zero_loss()

        x = self.data.x[subgraph.nodes.to(self.device)]
        edge_index = subgraph.edge_index.to(self.device)
        out = self.module(x, edge_index)          # [n_v, num_classes]
        # Local index 0 is the root by RootedSubgraph convention.
        root_logits = out[0:1]
        root_y = self.data.y[root].view(1)
        return F.nll_loss(root_logits, root_y)

    @torch.no_grad()
    def evaluate(self, data=None) -> Dict[str, float]:
        data = data or self.data
        self.eval_mode()
        out = self.module(data.x, self.eval_edges(data))
        pred = out.argmax(dim=1)
        accs = {}
        for split in ("train", "val", "test"):
            mask = getattr(data, f"{split}_mask")
            n = int(mask.sum().item())
            accs[split] = (float((pred[mask] == data.y[mask]).sum().item()) / n
                           if n else float("nan"))
        return accs
