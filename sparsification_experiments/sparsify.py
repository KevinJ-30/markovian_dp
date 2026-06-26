"""
Out-degree sparsification for node-level DP-GNN.

Rationale: the sensitivity bound
    Delta = C * (1 + D + D^2 + ... + D^L)
comes from bounding |R_out(u)|, the set of nodes u can INFLUENCE over L
message-passing hops (u itself + every node reachable via L outgoing arcs).
An out-degree cap D bounds |R_out(u)| <= 1 + D + D^2 + ... + D^L.

An in-degree cap does NOT bound |R_out(u)|: a star-center node that receives
all edges has in-degree >= E but out-degree unbounded, so its influence set
is uncapped and Delta would be invalid.

Two sparsifiers are provided:
  sparsify_by_outdegree  -- the correct default: caps out-degree <= D.
  sparsify_symmetric     -- Daigavane-style: caps undirected degree <= D via
                            greedy edge selection (for comparison only; also
                            bounds |R_out| because undirected degree >= out-degree).
"""

import torch


def _materialize_and_dedup(edge_index):
    """Return directed edges with both arcs materialized and self-loops removed."""
    src, dst = edge_index[0], edge_index[1]
    bi_src = torch.cat([src, dst])
    bi_dst = torch.cat([dst, src])
    pairs = torch.unique(torch.stack([bi_src, bi_dst], dim=1), dim=0)
    bi_src, bi_dst = pairs[:, 0], pairs[:, 1]
    no_self = bi_src != bi_dst
    return bi_src[no_self], bi_dst[no_self]


def _make_gen(seed):
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def sparsify_by_outdegree(edge_index, num_nodes, D, seed=None):
    """
    Cap out-degree of every node to D by randomly keeping at most D outgoing arcs.

    Steps:
      1. Materialize both directed arcs (u->v and v->u); deduplicate; drop self-loops.
      2. For each source node u with out-degree > D, keep a uniform-random subset
         of exactly D outgoing arcs. `seed` controls this randomness.

    This is the correct sparsifier for the node-level DP sensitivity bound.
    Out-degree D => influence set |R_out(u)| <= 1+D+...+D^L => Delta bounded.

    Returns:
        sparse_edge_index: [2, E'] long tensor with max out-degree D per node.
    """
    gen = _make_gen(seed)
    bi_src, bi_dst = _materialize_and_dedup(edge_index)

    # Sort by SOURCE so we can iterate per-source-node
    order = torch.argsort(bi_src, stable=True)
    bi_src = bi_src[order]
    bi_dst = bi_dst[order]

    counts = torch.bincount(bi_src, minlength=num_nodes)
    keep_mask = torch.ones(bi_src.size(0), dtype=torch.bool)

    ptr = 0
    for u in range(num_nodes):
        c = int(counts[u].item())
        if c > D:
            perm = torch.randperm(c, generator=gen) if gen else torch.randperm(c)
            keep_mask[ptr + perm[D:]] = False
        ptr += c

    return torch.stack([bi_src[keep_mask], bi_dst[keep_mask]], dim=0)


def sparsify_symmetric(edge_index, num_nodes, D, seed=None):
    """
    Cap undirected degree of every node to D (Daigavane-style, for comparison).

    Greedily selects a maximal subset of undirected edges such that every node's
    undirected degree stays <= D.  Edges are visited in random order; both arcs
    u->v and v->u are kept or dropped together.

    This also bounds |R_out(u)| because out-degree <= undirected-degree <= D, so
    the same Delta formula holds and Delta = 2C*(1+D+...+D^L) is UNCHANGED
    relative to sparsify_by_outdegree — only utility (accuracy) changes.
    Use as a comparison baseline against sparsify_by_outdegree.

    Returns:
        sparse_edge_index: [2, E'] long tensor with max undirected degree D per node.
    """
    gen = _make_gen(seed)
    bi_src, bi_dst = _materialize_and_dedup(edge_index)

    # Canonical undirected pairs: u < v
    u_idx = torch.minimum(bi_src, bi_dst)
    v_idx = torch.maximum(bi_src, bi_dst)
    pairs = torch.unique(torch.stack([u_idx, v_idx], dim=1), dim=0)
    n_edges = pairs.size(0)

    perm = torch.randperm(n_edges, generator=gen) if gen else torch.randperm(n_edges)
    pairs = pairs[perm]

    degree = torch.zeros(num_nodes, dtype=torch.long)
    kept_u, kept_v = [], []
    for i in range(n_edges):
        u = int(pairs[i, 0].item())
        v = int(pairs[i, 1].item())
        if degree[u] < D and degree[v] < D:
            degree[u] += 1
            degree[v] += 1
            kept_u.append(u)
            kept_v.append(v)

    if not kept_u:
        return torch.zeros((2, 0), dtype=torch.long)

    u_t = torch.tensor(kept_u, dtype=torch.long)
    v_t = torch.tensor(kept_v, dtype=torch.long)
    return torch.stack([torch.cat([u_t, v_t]), torch.cat([v_t, u_t])], dim=0)


def node_sensitivity(C, D, L, adjacency='add_remove'):
    """
    L2 sensitivity of the summed per-node gradient w.r.t. one node operation.

    With per-node gradient clipping at norm C and an L-layer directed GNN whose
    out-degree is bounded by D, node u can influence at most

        |R_out(u)| <= 1 + D + D^2 + ... + D^L

    nodes through message passing (u's own loss term plus L-hop forward
    reachability). Clipping each per-node gradient to C, the L2 sensitivity of
    the gradient SUM under REMOVAL is:

        Delta_remove = C * (1 + D + D^2 + ... + D^L)

    For ADD/REMOVE adjacency (inserting or deleting a node), each of the affected
    (1+D+...+D^L) per-node clipped gradients can change by up to 2C across
    neighboring datasets (from at most +C in one dataset to at most -C in the
    other, by the triangle inequality). The conservative bound is therefore:

        Delta_add_remove = 2 * C * (1 + D + D^2 + ... + D^L)

    # NOTE: 'add_remove' is the default because it is the only configuration
    # that gives a valid node-DP guarantee for both insertions and deletions.
    # 'remove' is the one-sided bound — only correct if the threat model is
    # restricted to node removal (e.g., membership inference, not reconstruction).
    #
    # NOTE: Delta is IDENTICAL for sparsify_symmetric and sparsify_by_outdegree
    # at the same D.  Both ensure out-degree <= D, so |R_out(u)| is bounded
    # by the same geometric sum.  Noise scaling, accountants, and epsilon numbers
    # are therefore unchanged when switching sparsifiers — only utility differs.
    #
    # Reference: Daigavane et al. 2021 "Node-Level Differentially Private
    # Graph Neural Networks", Theorem 1 (with the out-degree / fan-out
    # interpretation, which is equivalent to their in-degree bound on
    # undirected degree-bounded graphs but strictly correct on directed graphs).

    Args:
        C:          float, per-node gradient clipping norm
        D:          int, out-degree bound (from sparsify_by_outdegree)
        L:          int, number of GNN layers
        adjacency:  'add_remove' (default) or 'remove'

    Returns:
        Delta: float
    """
    geometric_sum = sum(D ** k for k in range(L + 1))
    Delta = float(C) * geometric_sum
    if adjacency == 'add_remove':
        Delta *= 2.0
    return Delta
