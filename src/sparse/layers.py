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

import torch.nn as nn
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
