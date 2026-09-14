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

from typing import Dict, Iterable, List, Sequence

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
                 num_layers=2, dropout=0.5, aggr='mean', device=None,
                 max_batched_subgraph_nodes: int = 8192):
        module = _NodeGNN(num_features, hidden, num_classes,
                          dropout=dropout, num_layers=num_layers, aggr=aggr)
        super().__init__(module, device=device)
        if max_batched_subgraph_nodes <= 0:
            raise ValueError("max_batched_subgraph_nodes must be positive")
        self.max_batched_subgraph_nodes = int(max_batched_subgraph_nodes)
        self.data = data
        # Root ids originate in CPU SparseExpand. Keeping the lookup on CPU
        # avoids synchronizing CUDA once per sampled root.
        self._train_mask = data.train_mask.cpu()

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

    def _is_supervised(self, subgraph) -> bool:
        return bool(self._train_mask[subgraph.root])

    def _batched_loss_chunk(self, subgraphs: Sequence) -> List[torch.Tensor]:
        """Evaluate supervised disconnected components in one GNN forward."""
        losses = [self.zero_loss() for _ in subgraphs]
        supervised = [(i, subgraph) for i, subgraph in enumerate(subgraphs)
                      if self._is_supervised(subgraph)]
        if not supervised:
            return losses

        offsets_list = []
        offset = 0
        for _, subgraph in supervised:
            offsets_list.append(offset)
            offset += subgraph.num_nodes
        offsets = torch.tensor(offsets_list, dtype=torch.long,
                               device=self.device)
        nodes = torch.cat([subgraph.nodes for _, subgraph in supervised]).to(
            self.device)
        edge_parts = []
        offset = 0
        for _, subgraph in supervised:
            if subgraph.num_edges:
                edge_parts.append(subgraph.edge_index + offset)
            offset += subgraph.num_nodes
        edge_index = (torch.cat(edge_parts, dim=1).to(self.device)
                      if edge_parts else torch.zeros(
                          (2, 0), dtype=torch.long, device=self.device))

        out = self.module(self.data.x[nodes], edge_index)
        roots = torch.tensor([subgraph.root for _, subgraph in supervised],
                             dtype=torch.long, device=self.device)
        labels = self.data.y[roots].view(-1)
        root_losses = F.nll_loss(out[offsets], labels, reduction='none')
        for (i, _), loss in zip(supervised, root_losses.unbind()):
            losses[i] = loss
        return losses

    def iter_subgraph_loss_batches(self, subgraphs: Sequence
                                   ) -> Iterable[List[torch.Tensor]]:
        """Yield bounded batches without retaining prior forward graphs."""
        chunk = []
        chunk_nodes = 0
        for subgraph in subgraphs:
            n_nodes = subgraph.num_nodes
            if n_nodes > self.max_batched_subgraph_nodes:
                if chunk:
                    yield self._batched_loss_chunk(chunk)
                    chunk, chunk_nodes = [], 0
                yield [self.subgraph_loss(subgraph)]
                continue
            if chunk and chunk_nodes + n_nodes > self.max_batched_subgraph_nodes:
                yield self._batched_loss_chunk(chunk)
                chunk, chunk_nodes = [], 0
            chunk.append(subgraph)
            chunk_nodes += n_nodes
        if chunk:
            yield self._batched_loss_chunk(chunk)

    def subgraph_losses(self, subgraphs: Sequence) -> List[torch.Tensor]:
        """Return one loss per input root using disconnected GNN batches."""
        return [loss for batch in self.iter_subgraph_loss_batches(subgraphs)
                for loss in batch]

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
