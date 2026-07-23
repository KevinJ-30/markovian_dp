"""
Algorithm 2: SparseExpand.

    SparseExpand(G, v, p2, r) -> rooted sparsified subgraph G_v = (V_v, E_v, F|_{V_v})

A randomized breadth-first expansion from a root vertex v.  Starting from the
frontier Q_0 = {v}, at each of r levels every examined outgoing edge (u, w) is
retained independently with probability p2 (Bernoulli(p2)).  A newly reached
vertex w is added to the vertex set and to the next frontier.  This exactly
mirrors the paper's pseudocode (Algorithm 2), including the detail that an edge
is added to E_v BEFORE the "w already visited" check (line 9 precedes line 10),
so E_v may contain edges pointing into already-discovered vertices.

Notation follows the paper: p2 is the edge-sampling probability, r the maximum
distance (number of expansion levels).

Graph orientation.  We expand along OUTGOING edges (u, w) in E.  Planetoid
citation graphs (CiteSeer/Cora/PubMed) store each undirected edge as both
directed arcs, so out-neighbors coincide with neighbors and the expansion is
symmetric.  On a genuinely directed graph, expansion follows arc direction, as
the privacy analysis (out-degree bound K_out) assumes.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class RootedSubgraph:
    """A rooted sparsified subgraph produced by SparseExpand.

    Attributes:
        root:       original-graph node id of the root vertex v.
        nodes:      LongTensor [n_v] of original node ids in V_v.  By convention
                    nodes[0] == root, so local index 0 always denotes the root.
        edge_index: LongTensor [2, E_v] of retained edges in LOCAL indices
                    (0 .. n_v-1), i.e. remapped through `nodes`.  Features for
                    the subgraph are obtained as x[nodes].
    """

    root: int
    nodes: torch.Tensor
    edge_index: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


def build_out_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """Build a per-node list of outgoing neighbors from an edge_index.

    Returns a Python list `adj` of length num_nodes where `adj[u]` is a 1-D
    LongTensor of the destination vertices w for every arc (u, w) in E.  Built
    once per graph and reused across all SparseExpand calls.

    Self-loops are kept if present in edge_index (SparseExpand handles them
    naturally: a retained self-loop adds an edge but never a new vertex).
    """
    edge_index = edge_index.cpu()
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(src, stable=True)
    src_sorted = src[order]
    dst_sorted = dst[order]
    counts = torch.bincount(src_sorted, minlength=num_nodes)
    adj: List[torch.Tensor] = []
    ptr = 0
    for u in range(num_nodes):
        c = int(counts[u].item())
        adj.append(dst_sorted[ptr:ptr + c].clone())
        ptr += c
    return adj


def _bernoulli_keep(n: int, p2: float, generator) -> torch.Tensor:
    """Return a bool mask [n] of independent Bernoulli(p2) keep decisions."""
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    if p2 >= 1.0:
        return torch.ones(n, dtype=torch.bool)
    if p2 <= 0.0:
        return torch.zeros(n, dtype=torch.bool)
    u = torch.rand(n, generator=generator)
    return u < p2


def sparse_expand(
    adj: Sequence[torch.Tensor],
    root: int,
    p2: float,
    r: int,
    generator: torch.Generator = None,
) -> RootedSubgraph:
    """Algorithm 2: randomized rooted expansion.

    Args:
        adj:       out-adjacency from `build_out_adjacency`.
        root:      root vertex v (original node id).
        p2:        edge-sampling probability (Bernoulli per examined arc).
        r:         maximum distance / number of expansion levels.
        generator: optional torch.Generator for reproducible sampling.

    Returns:
        RootedSubgraph with local-indexed edges (see RootedSubgraph docstring).
    """
    # V_v <- {v};  E_v <- empty;  Q_0 <- {v}
    visited = {root: 0}          # original id -> local index
    nodes_order = [root]
    edges_local: List[List[int]] = []   # [local_src, local_dst] pairs
    frontier = [root]

    for _ell in range(r):
        next_frontier: List[int] = []
        for u in frontier:
            out = adj[u]
            keep = _bernoulli_keep(int(out.numel()), p2, generator)
            if not bool(keep.any()):
                continue
            kept_dst = out[keep].tolist()
            u_local = visited[u]
            for w in kept_dst:
                # Line 9: add the edge regardless of whether w is new.
                if w not in visited:
                    visited[w] = len(nodes_order)
                    nodes_order.append(w)
                    next_frontier.append(w)
                edges_local.append([u_local, visited[w]])
        frontier = next_frontier
        if not frontier:
            break

    nodes = torch.tensor(nodes_order, dtype=torch.long)
    if edges_local:
        edge_index = torch.tensor(edges_local, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return RootedSubgraph(root=root, nodes=nodes, edge_index=edge_index)


def cap_degrees(
    edge_index: torch.Tensor,
    num_nodes: int,
    K_in: "int | None" = None,
    K_out: "int | None" = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Enforce Assumption 3.1 / Theorem 4's degree bounds by random arc removal.

    For every node with in-degree > K_in, keep a uniformly-random subset of
    exactly K_in incoming arcs; then likewise cap out-degrees at K_out.  Both
    passes only REMOVE arcs, so the out-cap cannot re-violate the in-cap.  The
    capped graph is fixed once before training (one-time preprocessing).

    Caveat (same as Daigavane et al. 2021): under node insertion/removal the
    capping randomness at a surviving node can depend on the inserted node's
    arcs, so strictly the accounting applies to the capped graph as given.  We
    follow the standard practice of capping once and accounting with (K_in,
    K_out) on the result.

    Returns a new edge_index [2, E'] (original node ids, arbitrary order).
    """
    ei = edge_index.cpu()

    def _cap(ei: torch.Tensor, row: int, bound: int) -> torch.Tensor:
        # Keep at most `bound` uniformly-random arcs per distinct ei[row] value.
        n_edges = ei.size(1)
        if n_edges == 0:
            return ei
        # Shuffle, then stable-sort by key: within each key group the arcs
        # appear in uniformly-random order.
        shuffle = torch.argsort(torch.rand(n_edges, generator=generator))
        order = shuffle[torch.argsort(ei[row][shuffle], stable=True)]
        sorted_keys = ei[row][order]
        # Rank of each arc within its key group (0, 1, 2, ... per group).
        change = torch.ones(n_edges, dtype=torch.bool)
        change[1:] = sorted_keys[1:] != sorted_keys[:-1]
        idx = torch.arange(n_edges)
        group_id = torch.cumsum(change.to(torch.long), 0) - 1
        rank = idx - idx[change][group_id]
        return ei[:, order[rank < bound]]

    if K_in is not None:
        ei = _cap(ei, row=1, bound=K_in)
    if K_out is not None:
        ei = _cap(ei, row=0, bound=K_out)
    return ei


def max_degrees(edge_index: torch.Tensor, num_nodes: int):
    """Return (max_in_degree, max_out_degree) of edge_index."""
    out_deg = torch.bincount(edge_index[0].cpu(), minlength=num_nodes)
    in_deg = torch.bincount(edge_index[1].cpu(), minlength=num_nodes)
    return int(in_deg.max()), int(out_deg.max())


def sample_roots(num_nodes: int, p1: float, generator: torch.Generator = None,
                 candidate_nodes: torch.Tensor = None) -> torch.Tensor:
    """Poisson (independent-Bernoulli) root sampling from Algorithm 1, line 3.

    V_root = { v : B_v = 1 },  B_v ~ Bernoulli(p1) independently.

    Args:
        num_nodes:       total number of nodes (used when candidate_nodes is None).
        p1:              root-sampling probability.
        generator:       optional torch.Generator.
        candidate_nodes: optional LongTensor restricting the pool of eligible
                         roots (e.g. training nodes).  If None, all nodes are
                         eligible.

    Returns:
        LongTensor of selected root node ids.
    """
    pool = (torch.arange(num_nodes) if candidate_nodes is None
            else candidate_nodes.cpu())
    n = int(pool.numel())
    if p1 >= 1.0:
        return pool.clone()
    keep = torch.rand(n, generator=generator) < p1
    return pool[keep]
