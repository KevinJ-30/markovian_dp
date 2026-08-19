"""
SparseExpand: randomized breadth-first expansion from a root vertex.

    SparseExpand(G, v, p2, r) -> rooted sparsified subgraph (V_v, E_v, F|_{V_v})

From frontier Q_0 = {v}, each of r levels retains every examined arc
independently with probability p2.  Following the paper's pseudocode, an arc
joins E_v before the "already visited" test, so E_v may contain arcs into
already-discovered vertices.

`direction='in'` (default) is Algorithm 5: traverse incoming arcs (w, u) but
keep their original orientation, so messages flow toward the root — what a
message-passing GNN needs.  `direction='out'` is the legacy Algorithm 2/4,
retained for the orientation ablation.  The direction also selects the
accounting shell size: n_d = K_out^d for 'in' (Eq. 44), K_in^d for 'out'.

In- and out-expansion coincide on undirected graphs, which store both arcs;
they differ on directed ones (ogbn-arxiv, RelBench foreign-key graphs).
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
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


def dedup_arcs(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Remove parallel arcs (the accounting assumes a simple graph).

    Encodes each arc as a single int64 key rather than calling
    `torch.unique(..., dim=1)`, whose row-wise sort needs several copies of the
    [2, E] tensor: on Reddit (114.6M arcs, 1.8 GB) that alone exhausts a 16 GB
    allocation.  When there are no duplicates — the common case — the original
    tensor is returned without rebuilding it.
    """
    ei = edge_index.cpu()
    key = ei[0].to(torch.long) * num_nodes + ei[1].to(torch.long)
    uniq = torch.unique(key)
    if uniq.numel() == key.numel():
        return ei
    del key
    return torch.stack([uniq // num_nodes, uniq % num_nodes])


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

    The greedy pass iterates over numpy arrays, not Python lists: on a dense
    graph (Reddit has ~57M undirected edges) `.tolist()` materializes billions
    of boxed ints and exhausts memory.
    """
    ei = edge_index.cpu()
    u, v = ei[0].to(torch.long), ei[1].to(torch.long)
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    key = torch.unique(a * num_nodes + b)
    a_np = (key // num_nodes).numpy()
    b_np = (key % num_nodes).numpy()
    m = a_np.shape[0]

    perm = torch.randperm(m, generator=generator).numpy()
    deg = np.zeros(num_nodes, dtype=np.int32)
    keep = np.zeros(m, dtype=bool)
    for i in perm:
        ai = a_np[i]
        bi = b_np[i]
        if ai == bi:
            if deg[ai] < K:
                deg[ai] += 1
                keep[i] = True
        elif deg[ai] < K and deg[bi] < K:
            deg[ai] += 1
            deg[bi] += 1
            keep[i] = True

    ka = torch.from_numpy(a_np[keep]).to(torch.long)
    kb = torch.from_numpy(b_np[keep]).to(torch.long)
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
