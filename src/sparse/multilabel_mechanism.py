"""
MultiLabelGNNMechanism: the base mechanism g0 for multilabel node classification.

Identical in structure to `GNNMechanism` — an L-layer GCN run on each root's
sparsified subgraph, reading off the root's output — but the label of a root is
a 0/1 vector rather than a class index, so the per-root loss is
`binary_cross_entropy_with_logits` and the reported metric is micro-F1.

This is what PPI needs (121 binary labels per node).  PPI is also the cleanest
inductive setting in the suite: its 24 graphs are disconnected and already
partitioned 20/2/2, so a training root's expansion can never reach a val/test
node regardless of r.
"""

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_mechanism import BaseMechanism
from .layers import build_conv_stack


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

    Same constructor signature as GNNMechanism and MLPMechanism so run.py can
    swap them freely.  `num_classes` is the number of label columns.
    """

    metric_name = "micro_f1"

    def __init__(self, data, num_features, num_classes, *, hidden=64,
                 num_layers=2, dropout=0.5, aggr='mean', device=None,
                 max_batched_subgraph_nodes: int = 8192):
        module = _MultiLabelGNN(num_features, hidden, num_classes,
                            dropout=dropout, num_layers=num_layers, aggr=aggr)
        super().__init__(module, device=device)
        if max_batched_subgraph_nodes <= 0:
            raise ValueError("max_batched_subgraph_nodes must be positive")
        self.max_batched_subgraph_nodes = int(max_batched_subgraph_nodes)
        self.data = data
        # Root ids come from CPU SparseExpand; keep the lookup on CPU so a
        # sampled root does not force a CUDA sync.
        self._train_mask = data.train_mask.cpu()

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

    def _is_supervised(self, subgraph) -> bool:
        return bool(self._train_mask[subgraph.root])

    def _batched_loss_chunk(self, subgraphs: Sequence) -> List[torch.Tensor]:
        """One forward over several DISCONNECTED rooted subgraphs.

        The components share no edges, so message passing cannot cross between
        them and each root's gradient is identical to what a per-root forward
        would give -- asserted in tests/test_sparse_batching.py.  Mirrors
        GNNMechanism._batched_loss_chunk; without it this mechanism ran one
        forward per root, which is what made Yelp/Amazon too slow to train to
        convergence.
        """
        losses = [self.zero_loss() for _ in subgraphs]
        supervised = [(i, sg) for i, sg in enumerate(subgraphs)
                      if self._is_supervised(sg)]
        if not supervised:
            return losses

        offsets_list, offset = [], 0
        for _, sg in supervised:
            offsets_list.append(offset)
            offset += sg.num_nodes
        offsets = torch.tensor(offsets_list, dtype=torch.long, device=self.device)
        nodes = torch.cat([sg.nodes for _, sg in supervised]).to(self.device)

        edge_parts, offset = [], 0
        for _, sg in supervised:
            if sg.num_edges:
                edge_parts.append(sg.edge_index + offset)
            offset += sg.num_nodes
        edge_index = (torch.cat(edge_parts, dim=1).to(self.device)
                      if edge_parts else
                      torch.zeros((2, 0), dtype=torch.long, device=self.device))

        out = self.module(self.data.x[nodes], edge_index)
        roots = torch.tensor([sg.root for _, sg in supervised],
                             dtype=torch.long, device=self.device)
        labels = self.data.y[roots].float()
        # reduction='none' then mean over labels only, so each root's loss
        # matches subgraph_loss's default mean-over-labels exactly.
        per_root = F.binary_cross_entropy_with_logits(
            out[offsets], labels, reduction='none').mean(dim=1)
        for (i, _), loss in zip(supervised, per_root.unbind()):
            losses[i] = loss
        return losses

    def iter_subgraph_loss_batches(self, subgraphs: Sequence
                                   ) -> Iterable[List[torch.Tensor]]:
        """Yield ONE loss at a time, deliberately unbatched.

        This is the DP path: `_step_dp` needs a separate gradient per root in
        order to clip per root, so it calls autograd.grad once per loss.  If
        those losses share one batched forward, every backward has to traverse
        the whole batch's graph -- O(n^2) instead of O(n).  Measured on 512 Yelp
        roots (mean subgraph 7.36 nodes, hidden 512): 0.64s unbatched against
        8.36s batched, a 13x penalty.

        Batching is still the right thing for the NON-DP path, which sums the
        losses and takes a single backward; see subgraph_losses below.
        """
        for sg in subgraphs:
            yield [self.subgraph_loss(sg)]

    def subgraph_losses(self, subgraphs: Sequence) -> List[torch.Tensor]:
        """One loss per root, using batched forwards over disconnected chunks.

        Used by the NON-DP path, which sums these and takes a single backward,
        so the batched forward is a straight win.  Do not route the DP path
        here -- see iter_subgraph_loss_batches.
        """
        chunk, chunk_nodes, out = [], 0, []
        for sg in subgraphs:
            n = sg.num_nodes
            if n > self.max_batched_subgraph_nodes:
                if chunk:
                    out.extend(self._batched_loss_chunk(chunk)); chunk, chunk_nodes = [], 0
                out.append(self.subgraph_loss(sg))
                continue
            if chunk and chunk_nodes + n > self.max_batched_subgraph_nodes:
                out.extend(self._batched_loss_chunk(chunk)); chunk, chunk_nodes = [], 0
            chunk.append(sg); chunk_nodes += n
        if chunk:
            out.extend(self._batched_loss_chunk(chunk))
        return out

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
