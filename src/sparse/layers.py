"""
Shared message-passing stack for the GNN base mechanisms.

The choice of aggregator is not cosmetic here — it decides whether the base
mechanism g0 computed on a rooted subgraph agrees with full-graph inference at
the root, which is what makes "train on sparsified subgraphs, evaluate on the
full graph" an honest protocol.

    aggr='mean'  GraphSAGE mean aggregation.  Edge w->u is weighted by
                 1/|in-neighbours of u|, which depends only on u's
                 in-neighbourhood.  Every node that acts as a TARGET inside a
                 rooted subgraph has its complete in-neighbourhood present, by
                 construction of SparseExpand, so the root's output on the
                 subgraph equals full-graph inference EXACTLY (verified to
                 0.00e+00 relative error on PPI).

    aggr='gcn'   Symmetric normalization, deg(w)^-1/2 deg(u)^-1/2.  This needs
                 the degree of the SOURCE, and boundary nodes of a rooted
                 subgraph are sources whose in-edges were never expanded, so
                 their degree reads as 1 instead of their true degree.  The
                 rooted computation is then only an approximation of full-graph
                 inference, and the error grows with density:

                     ogbn-arxiv, capped K=5   ~0.3%   (inside seed noise)
                     ogbn-arxiv, uncapped     ~0.7%
                     PPI, uncapped            150-400%  (fatal)

Both are valid mechanisms as far as the privacy analysis is concerned —
Assumption 3.2 / 6.3 only requires g0 to be a function of the rooted subgraph
with ||g0||_2 <= C — so 'gcn' is kept for reproducing earlier results.  'mean'
is the default because it makes the evaluation protocol exact rather than
approximate.
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
