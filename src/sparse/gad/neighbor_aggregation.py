"""
Parameter-free multi-hop neighbor aggregation (GADBench XGB-Graph feature builder).

    h^(0)_v = x_v
    h^(l)_v = Aggregate{ h^(l-1)_u : u in Neighbor(v) }     # mean / sum / max, NO params
    features(v) = [ h^0_v || h^1_v || ... || h^L_v ]        # dim (L+1) * d

Two builders that share the same output contract ([N, (L+1)*d]):

  * `aggregate_features` (global): fast, uses one edge set + scatter reductions. Combine with
    `sparsify_edges_bernoulli` to aggregate on a globally edge-sparsified graph — the default,
    scalable "pass in a sparsified graph" route.
  * `aggregate_features_expand` (per-root fidelity): runs `sparse_expand` (the paper's S2
    composite-subsampling mechanism, src/sparse/sparse_expand.py) from each root and aggregates
    within that root's sampled subgraph. Matches the DP mechanism exactly but is slow on dense
    graphs (per-node BFS); intended for spot-checks / small node subsets, not full dense sweeps.
"""

import torch
from torch_geometric.utils import scatter

from ..sparse_expand import sparse_expand

_VALID_AGGR = ("mean", "sum", "max", "min")


def sparsify_edges_bernoulli(edge_index, p2, generator=None):
    """Global S2: keep each edge independently with probability p2.

    Args:
        edge_index: LongTensor [2, E].
        p2:         edge-keep probability.
        generator:  optional torch.Generator for reproducibility.

    Returns:
        LongTensor [2, E'] of retained edges.
    """
    if p2 >= 1.0:
        return edge_index
    if p2 <= 0.0:
        return edge_index[:, :0]
    E = edge_index.size(1)
    keep = torch.rand(E, generator=generator) < p2
    return edge_index[:, keep]


def aggregate_features(x, edge_index, num_layers, aggr="mean"):
    """Global parameter-free L-hop aggregation; returns [N, (L+1)*d].

    For each directed edge (src, dst) in `edge_index`, node `dst` aggregates the feature of
    `src`. Tolokers/Questions store undirected edges as both arcs, so this is symmetric
    neighbor aggregation. Nodes with no incoming edges get a zero aggregate at that layer.
    """
    if aggr not in _VALID_AGGR:
        raise ValueError(f"aggr must be one of {_VALID_AGGR}, got {aggr!r}")
    N = x.size(0)
    src, dst = edge_index[0], edge_index[1]
    outs = [x]
    h = x
    for _ in range(num_layers):
        h = scatter(h[src], dst, dim=0, dim_size=N, reduce=aggr)
        outs.append(h)
    return torch.cat(outs, dim=1)


def aggregate_features_expand(x, adj, nodes, p2, r, aggr="mean", generator=None):
    """Per-root fidelity aggregation via SparseExpand; returns [len(nodes), (r+1)*d].

    For each root v in `nodes`, run `sparse_expand(adj, v, p2, r, direction='in')` and aggregate
    within the sampled rooted subgraph, reading off the root's concatenated features. At p2=1 and
    r=L this equals `aggregate_features` for the same nodes.

    Both builders aggregate along INCOMING edges: `aggregate_features` scatters src -> dst, so a
    node absorbs its in-neighbours, and in-expansion is what collects exactly those nodes (see
    src/sparse/sparse_expand.py on the v35 orientation fix). On the undirected GADBench graphs the
    two orientations coincide; on a directed graph only this one matches.

    Args:
        x:        [N, d] feature matrix (original node indexing).
        adj:      in-adjacency from build_adjacency(..., direction='in').
        nodes:    iterable of root node ids to compute features for.
        p2, r:    SparseExpand edge-keep probability and depth.
        aggr:     aggregation reduction.
        generator: optional torch.Generator.
    """
    if aggr not in _VALID_AGGR:
        raise ValueError(f"aggr must be one of {_VALID_AGGR}, got {aggr!r}")
    d = x.size(1)
    feats = torch.zeros((len(nodes), (r + 1) * d), dtype=x.dtype)
    for i, v in enumerate(nodes):
        sub = sparse_expand(adj, int(v), p2, r, generator=generator, direction='in')
        n_sub = sub.num_nodes
        h = x[sub.nodes]                      # [n_sub, d], local index 0 == root
        ei = sub.edge_index                   # local arcs (src -> dst), dst closer to the root
        outs = [h[0]]                          # root's h^0
        for _ in range(r):
            if ei.size(1) == 0:
                agg = torch.zeros_like(h)
            else:
                # Same convention as aggregate_features: dst absorbs src.
                agg = scatter(h[ei[0]], ei[1], dim=0, dim_size=n_sub, reduce=aggr)
            h = agg
            outs.append(h[0])                  # root's h^(l)
        feats[i] = torch.cat(outs, dim=0)
    return feats
