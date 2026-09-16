"""
Shared message-passing stack for the GNN base mechanisms.

The aggregator decides whether g0 on a rooted subgraph equals full-graph
inference at the root, which is what makes "train sparsified, evaluate on the
full graph" exact rather than approximate:

    aggr='mean'  weights an arc by 1/|in-neighbours of the target|, which
                 SparseExpand always materializes in full, so the rooted
                 computation is EXACT.
    aggr='gcn'   symmetric normalization needs the SOURCE degree, which is
                 wrong for subgraph boundary nodes.  Error grows with density
                 (~0.3% on capped ogbn-arxiv, 150-400% on uncapped PPI).

Both are valid mechanisms for the privacy analysis, which only needs g0 to be
a function of the rooted subgraph with ||g0||_2 <= C.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

VALID_AGGR = ("mean", "gcn")


def build_conv_stack(dims: List[int], aggr: str = "mean") -> nn.ModuleList:
    """Message-passing layers mapping dims[0] -> dims[1] -> ... -> dims[-1]."""
    if aggr not in VALID_AGGR:
        raise ValueError(f"aggr must be one of {VALID_AGGR}, got {aggr!r}")
    if aggr == "mean":
        return nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1], aggr="mean")
            for i in range(len(dims) - 1)
        ])
    return nn.ModuleList([
        GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True)
        for i in range(len(dims) - 1)
    ])


def _linear_view(weight: nn.Parameter, bias: "nn.Parameter | None") -> nn.Linear:
    """Torch Linear sharing existing parameter objects for Opacus hooks."""
    out_features, in_features = weight.shape
    linear = nn.Linear(in_features, out_features, bias=bias is not None)
    linear.weight = weight
    if bias is not None:
        linear.bias = bias
    return linear


def _masked_aggregate(x, edge_index, edge_mask, *, weights=None):
    """Sum source features into destinations independently for every graph."""
    batch_size, _, feature_dim = x.shape
    edge_count = edge_index.size(-1)
    if edge_count == 0:
        return torch.zeros_like(x)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    safe_src = src.masked_fill(~edge_mask, 0)
    safe_dst = dst.masked_fill(~edge_mask, 0)
    gathered = x.gather(
        1, safe_src.unsqueeze(-1).expand(batch_size, edge_count, feature_dim))
    scale = edge_mask.to(x.dtype) if weights is None else weights * edge_mask.to(x.dtype)
    gathered = gathered * scale.unsqueeze(-1)
    out = torch.zeros_like(x).scatter_add(
        1, safe_dst.unsqueeze(-1).expand(
            batch_size, edge_count, feature_dim),
        gathered)
    return out


class PaddedSAGEConv(nn.Module):
    """Batch-first SAGE-mean using the parameters of an existing SAGEConv."""

    def __init__(self, conv: SAGEConv):
        super().__init__()
        self.neighbor_linear = _linear_view(conv.lin_l.weight, conv.lin_l.bias)
        self.root_linear = _linear_view(conv.lin_r.weight, conv.lin_r.bias)

    def forward(self, x, edge_index, edge_mask, node_mask):
        batch_size, node_count, _ = x.shape
        summed = _masked_aggregate(x, edge_index, edge_mask)
        degree = torch.zeros(
            (batch_size, node_count), dtype=x.dtype, device=x.device)
        if edge_index.size(-1):
            dst = edge_index[:, 1].masked_fill(~edge_mask, 0)
            degree = degree.scatter_add(1, dst, edge_mask.to(x.dtype))
        mean = summed / degree.clamp_min(1).unsqueeze(-1)
        out = self.neighbor_linear(mean) + self.root_linear(x)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class PaddedGCNConv(nn.Module):
    """Batch-first GCN normalization using an existing GCNConv's parameters."""

    def __init__(self, conv: GCNConv):
        super().__init__()
        self.linear = _linear_view(conv.lin.weight, conv.bias)

    def forward(self, x, edge_index, edge_mask, node_mask):
        batch_size, node_count, feature_dim = x.shape
        edge_count = edge_index.size(-1)
        if edge_count:
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            non_loop = edge_mask & (src != dst)
            safe_dst = dst.masked_fill(~non_loop, 0)
        else:
            src = edge_index[:, 0]
            dst = edge_index[:, 1]
            non_loop = edge_mask
            safe_dst = dst

        # PyG add_remaining_self_loops removes all existing loops and appends
        # exactly one unit-weight loop for every real node.
        degree = node_mask.to(x.dtype)
        if edge_count:
            degree = degree.scatter_add(1, safe_dst, non_loop.to(x.dtype))
        inv_sqrt = degree.pow(-0.5)
        inv_sqrt = inv_sqrt.masked_fill(~torch.isfinite(inv_sqrt), 0.0)

        if edge_count:
            safe_src = src.masked_fill(~non_loop, 0)
            weights = inv_sqrt.gather(1, safe_src) * inv_sqrt.gather(1, safe_dst)
            neighbors = _masked_aggregate(
                x, edge_index, non_loop, weights=weights)
        else:
            neighbors = torch.zeros(
                (batch_size, node_count, feature_dim),
                dtype=x.dtype, device=x.device)
        aggregate = neighbors + x * inv_sqrt.square().unsqueeze(-1)
        out = self.linear(aggregate)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class PaddedGNNStack(nn.Module):
    """Private batch-first view of an existing PyG convolution stack."""

    def __init__(self, convs: nn.ModuleList, *, dropout: float):
        super().__init__()
        adapters = []
        for conv in convs:
            if isinstance(conv, SAGEConv):
                adapters.append(PaddedSAGEConv(conv))
            elif isinstance(conv, GCNConv):
                adapters.append(PaddedGCNConv(conv))
            else:
                raise TypeError(f"unsupported convolution type {type(conv).__name__}")
        self.convs = nn.ModuleList(adapters)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_mask, node_mask):
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_mask, node_mask)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x * node_mask.unsqueeze(-1).to(x.dtype)
        return x
