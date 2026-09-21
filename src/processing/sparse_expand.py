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
from typing import List

import torch


@dataclass(frozen=True)
class SparseAdjacency:
    """Compact CPU CSR neighbour storage for SparseExpand."""

    rowptr: torch.Tensor
    col: torch.Tensor
    direction: str

    def neighbors(self, node: int) -> torch.Tensor:
        start = int(self.rowptr[node])
        end = int(self.rowptr[node + 1])
        return self.col[start:end]

    def __len__(self) -> int:
        return int(self.rowptr.numel() - 1)


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
                    direction: str = 'in') -> SparseAdjacency:
    """Build compact CPU CSR neighbours for SparseExpand."""
    if direction not in ('in', 'out'):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    edge_index = edge_index.cpu()
    key_row, val_row = (1, 0) if direction == 'in' else (0, 1)
    key, val = edge_index[key_row], edge_index[val_row]
    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    col = val[order]
    counts = torch.bincount(key_sorted, minlength=num_nodes)
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long)
    rowptr[0] = 0
    rowptr[1:] = counts.cumsum(0)
    return SparseAdjacency(rowptr=rowptr, col=col, direction=direction)


def build_out_adjacency(edge_index: torch.Tensor, num_nodes: int) -> SparseAdjacency:
    """Out-adjacency for SparseExpand and GAD."""
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
    adj: SparseAdjacency,
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
    if adj.direction != direction:
        raise ValueError(
            f"adjacency direction {adj.direction!r} does not match "
            f"expansion direction {direction!r}")
    expand_in = direction == 'in'
    # V_v <- {v};  E_v <- empty;  Q_0 <- {v}
    visited = {root: 0}          # original id -> local index
    nodes_order = [root]
    edges_local: List[List[int]] = []   # [local_src, local_dst] pairs
    frontier = [root]

    for _ell in range(r):
        next_frontier: List[int] = []
        for u in frontier:
            out = adj.neighbors(u)
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
