"""
SparseExpand: randomized breadth-first expansion from a root vertex.

    SparseExpand(G, v, p2, r) -> rooted sparsified subgraph G_v = (V_v, E_v, F|_{V_v})

Starting from the frontier Q_0 = {v}, at each of r levels every examined edge is
retained independently with probability p2 (Bernoulli(p2)).  A newly reached
vertex is added to the vertex set and to the next frontier.  This mirrors the
paper's pseudocode, including the detail that an edge is added to E_v BEFORE the
"already visited" check, so E_v may contain edges pointing into
already-discovered vertices.

Notation follows the paper: p2 is the edge-sampling probability, r the maximum
distance (number of expansion levels).


ORIENTATION (manuscript v35, Sections 5-6)
------------------------------------------
Earlier versions of this module expanded along OUTGOING edges (u, w), matching
Algorithm 2/4.  That orientation is *backwards* for message passing.  A GNN that
aggregates from in-neighbours computes the root's representation from vertices
that admit a directed path TO the root, so with out-expansion the root receives
nothing but its own self-loop and the "GNN" silently degenerates to a
graph-blind MLP.  Section 5 of v35 states this explicitly and Section 6 develops
the corrected procedure, `SparseExpand_in` (Algorithm 5): traverse each INCOMING
edge (w, u) from u to w, but retain the original orientation (w, u) in the
returned subgraph.

`direction='in'` (the default) implements Algorithm 5; `direction='out'` keeps
the legacy Algorithm 2/4 behaviour for the orientation ablation.

The direction also selects which degree bound governs the accounting shells:

    direction='in'   n_d = K_out^d   (Theorem 6.4, Eq. 44)
    direction='out'  n_d = K_in^d    (Theorem 1/2)

while q_d is driven by K = min(K_in, K_out) in both cases.  See
`src/sparse/accounting.py`.

Undirected graphs (Planetoid citation graphs, Flickr, PPI, Reddit) store each
edge as both arcs, so in- and out-expansion coincide there; the distinction
bites on genuinely directed graphs such as ogbn-arxiv (max in-degree 3015 vs
max out-degree 221) and on RelBench foreign-key graphs.
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


def build_adjacency(edge_index: torch.Tensor, num_nodes: int,
                    direction: str = 'in') -> List[torch.Tensor]:
    """Build the per-node neighbour lists that SparseExpand traverses.

    Args:
        edge_index: LongTensor [2, E] of arcs (edge_index[0] = source,
                    edge_index[1] = target).
        num_nodes:  number of vertices.
        direction:  'in'  -> adj[u] lists the sources w of arcs (w, u)
                             (Algorithm 5, SparseExpand_in);
                    'out' -> adj[u] lists the targets w of arcs (u, w)
                             (legacy Algorithm 2/4).

    Returns a Python list `adj` of length num_nodes of 1-D LongTensors.  Built
    once per graph and reused across all SparseExpand calls.

    IMPORTANT: `adj` and the `direction` passed to `sparse_expand` must match —
    the adjacency decides *who* is traversed, `direction` decides how the
    resulting arc is oriented in the returned subgraph.

    Self-loops are kept if present in edge_index (SparseExpand handles them
    naturally: a retained self-loop adds an edge but never a new vertex).
    """
    if direction not in ('in', 'out'):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    edge_index = edge_index.cpu()
    # Group by the endpoint we expand FROM: for in-expansion a root u collects
    # the arcs whose target is u, so we key on edge_index[1].
    key_row, val_row = (1, 0) if direction == 'in' else (0, 1)
    key, val = edge_index[key_row], edge_index[val_row]
    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    val_sorted = val[order]
    counts = torch.bincount(key_sorted, minlength=num_nodes)
    adj: List[torch.Tensor] = []
    ptr = 0
    for u in range(num_nodes):
        c = int(counts[u].item())
        adj.append(val_sorted[ptr:ptr + c].clone())
        ptr += c
    return adj


def build_out_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    """Out-adjacency: adj[u] lists the targets w of every arc (u, w).

    Thin alias for `build_adjacency(..., direction='out')`, kept for the GAD
    pipeline (`src/sparse/gad/`) which aggregates over forward neighbourhoods.
    """
    return build_adjacency(edge_index, num_nodes, direction='out')


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
    direction: str = 'in',
) -> RootedSubgraph:
    """SparseExpand: randomized rooted expansion (Algorithm 5 / Algorithm 2).

    Args:
        adj:       neighbour lists from `build_adjacency(..., direction)` — must
                   have been built with the SAME `direction` passed here.
        root:      root vertex v (original node id).
        p2:        edge-sampling probability (Bernoulli per examined arc).
        r:         maximum distance / number of expansion levels.
        generator: optional torch.Generator for reproducible sampling.
        direction: 'in'  -> Algorithm 5: traverse incoming arcs (w, u) and record
                            them with their original orientation, so messages
                            flow toward the root;
                   'out' -> legacy Algorithm 2/4: traverse outgoing arcs (u, w).

    Returns:
        RootedSubgraph with local-indexed edges (see RootedSubgraph docstring).
    """
    if direction not in ('in', 'out'):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    expand_in = direction == 'in'
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
                # Add the edge regardless of whether w is new (Alg 5 line 8
                # precedes the membership test on line 9).
                if w not in visited:
                    visited[w] = len(nodes_order)
                    nodes_order.append(w)
                    next_frontier.append(w)
                # 'in': the traversed arc is (w, u), and Algorithm 5 retains
                # that original orientation, so w is the source and u the
                # target — messages flow toward the root.
                edges_local.append([visited[w], u_local] if expand_in
                                   else [u_local, visited[w]])
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
    """Enforce the degree bounds (Assumption 3.1 / 6.2) by random arc removal.

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


def edge_set_is_symmetric(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """True iff every arc (u, v) has its reverse (v, u) present.

    Undirected PyG datasets (Planetoid, Flickr, Reddit, PPI) store each edge as
    both arcs, so this detects them.  Assumes no parallel arcs (deduplicate
    first); comparison is by sorted arc encodings, O(E log E).
    """
    ei = edge_index.cpu()
    u, v = ei[0].to(torch.long), ei[1].to(torch.long)
    fwd = torch.sort(u * num_nodes + v).values
    rev = torch.sort(v * num_nodes + u).values
    return bool(torch.equal(fwd, rev))


def cap_degrees_undirected(
    edge_index: torch.Tensor,
    num_nodes: int,
    K: int,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Cap the UNDIRECTED degree at K; returns a symmetric arc set.

    `cap_degrees` treats the two arcs of an undirected edge independently, so
    on an undirected graph it silently destroys symmetry (measured: only ~1/3
    of surviving arcs keep their reverse at K=5).  This variant collapses arc
    pairs to undirected edges, keeps a random subset with every endpoint's
    degree <= K (greedy over a uniformly-random edge order, so some nodes may
    end below K — any subgraph satisfies the degree-bound assumption), and
    re-emits both arcs.  The result has max in-degree = max out-degree <= K
    and is symmetric, so in- and out-expansion coincide on it.

    A self-loop consumes one unit of its node's capacity and is emitted as a
    single arc.  Parallel arcs are collapsed before capping.
    """
    ei = edge_index.cpu()
    u, v = ei[0].to(torch.long), ei[1].to(torch.long)
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    key = torch.unique(a * num_nodes + b)
    a = (key // num_nodes).tolist()
    b = (key % num_nodes).tolist()
    m = len(a)

    perm = torch.randperm(m, generator=generator).tolist()
    deg = [0] * num_nodes
    keep_a, keep_b = [], []
    for i in perm:
        ai, bi = a[i], b[i]
        if ai == bi:
            if deg[ai] < K:
                deg[ai] += 1
                keep_a.append(ai)
                keep_b.append(bi)
        elif deg[ai] < K and deg[bi] < K:
            deg[ai] += 1
            deg[bi] += 1
            keep_a.append(ai)
            keep_b.append(bi)

    ka = torch.tensor(keep_a, dtype=torch.long)
    kb = torch.tensor(keep_b, dtype=torch.long)
    loops = ka == kb
    src = torch.cat([ka[~loops], kb[~loops], ka[loops]])
    dst = torch.cat([kb[~loops], ka[~loops], ka[loops]])
    return torch.stack([src, dst])


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
