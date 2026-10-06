"""
SparseExpand: randomized breadth-first expansion from a root vertex.

    SparseExpand(G, v, p2, r) -> rooted sparsified subgraph (V_v, E_v, F|_{V_v})

From frontier Q_0 = {v}, each of r levels samples arcs with probability p2.
Incoming expansion caps retained arcs per expanded node at 20, 10, then 5
(5 for later levels): draw the Bernoulli count, cap it, then choose that many
arcs uniformly without replacement.
An arc joins E_v before the "already visited" test, so E_v may contain arcs into
already-discovered vertices.

Capped Algorithm 5 traverses incoming arcs (w, u) and keeps their original
orientation, so messages flow toward the root. Its accounting shell size is
n_d = K_out^d (Eq. 44).
"""

import math
from dataclasses import dataclass
from typing import List

import torch

INCOMING_EDGE_CAPS = (20, 10, 5)


@dataclass(frozen=True)
class SparseAdjacency:
    """Compact CPU CSR neighbour storage shared by SparseExpand and DP-GNN."""

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
    """Build incoming CSR neighbours for SparseExpand, or outgoing for DP-GNN."""
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


def _capped_incoming_positions(
    degrees: torch.Tensor, p2: float, cap: int, generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample capped Bernoulli subsets using O(frontier size * cap) storage."""
    if p2 <= 0.0:
        counts = torch.zeros_like(degrees)
    elif p2 >= 1.0:
        counts = degrees.clamp(max=cap)
    else:
        populations = degrees.to(torch.float64)
        counts = torch.binomial(
            populations, torch.full_like(populations, p2), generator=generator,
        ).to(torch.long).clamp(max=cap)
    width = int(counts.max())
    slots = torch.arange(width)
    positions = slots.expand(*degrees.shape, width).clone()
    partial = (counts > 0) & (counts < degrees)
    # Floyd's algorithm: a uniform subset without a degree-sized permutation.
    for slot in range(width):
        active = partial & (counts > slot)
        active_count = int(active.sum())
        if not active_count:
            break
        upper = degrees[active] - counts[active] + slot
        candidate = (
            torch.rand(active_count, dtype=torch.float64, generator=generator)
            * (upper + 1)
        ).to(torch.long)
        duplicate = (positions[active][:, :slot] == candidate[:, None]).any(dim=1)
        positions[..., slot][active] = torch.where(duplicate, upper, candidate)
    retained = slots < counts.unsqueeze(-1)
    # Keep CSR order, with inactive slots sorted after every real position.
    positions = torch.where(retained, positions, degrees.unsqueeze(-1)).sort(dim=-1).values
    return positions, retained


def sparse_expand(
    adj: SparseAdjacency,
    root: int,
    p2: float,
    r: int,
    generator: torch.Generator = None,
) -> RootedSubgraph:
    """SparseExpand: randomized incoming rooted expansion (Algorithm 5).

    Args:
        adj:       incoming neighbour lists from `build_adjacency`.
        root:      root vertex v (original node id).
        p2:        Bernoulli edge probability before the per-hop incoming cap.
        r:         maximum distance / number of expansion levels.
        generator: optional torch.Generator for reproducible sampling.

    Returns:
        RootedSubgraph with local-indexed edges (see RootedSubgraph docstring).
    """
    if adj.direction != 'in':
        raise ValueError("SparseExpand requires incoming adjacency")
    # V_v <- {v};  E_v <- empty;  Q_0 <- {v}
    visited = {root: 0}          # original id -> local index
    nodes_order = [root]
    edges_local: List[List[int]] = []   # [local_src, local_dst] pairs
    frontier = [root]

    for _ell in range(r):
        cap = INCOMING_EDGE_CAPS[min(_ell, len(INCOMING_EDGE_CAPS) - 1)]
        next_frontier: List[int] = []
        for u in frontier:
            neighbors = adj.neighbors(u)
            if neighbors.numel() > cap:
                positions, retained = _capped_incoming_positions(
                    torch.tensor([neighbors.numel()]), p2, cap, generator)
                kept_neighbors = neighbors[positions[retained]].tolist()
            else:
                keep = _bernoulli_keep(int(neighbors.numel()), p2, generator)
                kept_neighbors = neighbors[keep].tolist()
            u_local = visited[u]
            for w in kept_neighbors:
                # Add the edge regardless of whether w is new (Alg 5 line 8
                # precedes the membership test on line 9).
                if w not in visited:
                    visited[w] = len(nodes_order)
                    nodes_order.append(w)
                    next_frontier.append(w)
                # Retain the original arc (w, u), so messages flow toward the root.
                edges_local.append([visited[w], u_local])
        frontier = next_frontier
        if not frontier:
            break

    nodes = torch.tensor(nodes_order, dtype=torch.long)
    if edges_local:
        edge_index = torch.tensor(edges_local, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return RootedSubgraph(root=root, nodes=nodes, edge_index=edge_index)


def batch_sparse_expand(
    adj: SparseAdjacency,
    roots: torch.Tensor,
    p2: float,
    r: int,
    generator: torch.Generator = None,
) -> List[RootedSubgraph]:
    """Vectorized SparseExpand for an ordered batch of CPU roots.

    Expansion is level-synchronous across roots. Incoming neighborhoods retain
    at most 20/10/5 arcs per expanded node by hop (5 thereafter), without
    allocating degree-sized candidates.
    Segmented composite keys preserve first-discovery order and map repeated
    discoveries to one local node.
    """
    if adj.direction != 'in':
        raise ValueError("SparseExpand requires incoming adjacency")
    if roots.ndim != 1:
        raise ValueError("roots must be a one-dimensional tensor")
    if roots.device.type != 'cpu':
        raise ValueError("roots must be on CPU")

    roots = roots.detach().to(dtype=torch.long)
    batch_size = int(roots.numel())
    if batch_size == 0:
        return []

    num_graph_nodes = len(adj)
    if bool(((roots < 0) | (roots >= num_graph_nodes)).any()):
        raise IndexError("root index out of range")

    root_rows = torch.arange(batch_size, dtype=torch.long).unsqueeze(1)
    nodes = roots.unsqueeze(1).clone()
    node_counts = torch.ones(batch_size, dtype=torch.long)
    frontier_nodes = nodes.clone()
    frontier_local = torch.zeros_like(frontier_nodes)
    frontier_counts = node_counts.clone()
    edge_index = torch.zeros((batch_size, 2, 0), dtype=torch.long)
    edge_counts = torch.zeros(batch_size, dtype=torch.long)

    for _ell in range(r):
        cap = INCOMING_EDGE_CAPS[min(_ell, len(INCOMING_EDGE_CAPS) - 1)]
        frontier_width = frontier_nodes.size(1)
        frontier_slots = torch.arange(frontier_width, dtype=torch.long)
        frontier_mask = frontier_slots.unsqueeze(0) < frontier_counts.unsqueeze(1)
        safe_frontier = frontier_nodes.masked_fill(~frontier_mask, 0)

        starts = adj.rowptr[safe_frontier]
        degrees = (adj.rowptr[safe_frontier + 1] - starts).masked_fill(
            ~frontier_mask, 0)
        max_degree = int(degrees.max())
        if max_degree == 0:
            break

        if max_degree > cap:
            neighbor_positions, retained = _capped_incoming_positions(
                degrees, p2, cap, generator)
            width = neighbor_positions.size(-1)
            safe_offsets = (
                starts.unsqueeze(-1) + neighbor_positions
            ).masked_fill(~retained, 0)
            retained = retained.reshape(batch_size, -1)
        else:
            width = max_degree
            neighbor_slots = torch.arange(width, dtype=torch.long)
            candidate_mask = (
                neighbor_slots.view(1, 1, -1) < degrees.unsqueeze(-1))
            safe_offsets = (
                starts.unsqueeze(-1) + neighbor_slots.view(1, 1, -1)
            ).masked_fill(~candidate_mask, 0)
            candidate_mask = candidate_mask.reshape(batch_size, -1)
            if p2 >= 1.0:
                retained = candidate_mask
            elif p2 <= 0.0:
                break
            else:
                retained = torch.zeros_like(candidate_mask)
                draws = torch.rand(
                    int(candidate_mask.sum()), generator=generator)
                retained[candidate_mask] = draws < p2
        if not bool(retained.any()):
            break
        candidate_nodes = adj.col[safe_offsets].reshape(batch_size, -1)
        parent_local = frontier_local.unsqueeze(-1).expand(
            -1, -1, width).reshape(batch_size, -1)

        candidate_width = candidate_nodes.size(1)
        candidate_rows = root_rows.expand(-1, candidate_width)
        retained_linear = torch.nonzero(
            retained.reshape(-1), as_tuple=False).flatten()
        retained_batch = candidate_rows.reshape(-1)[retained_linear]
        retained_nodes = candidate_nodes.reshape(-1)[retained_linear]
        retained_parents = parent_local.reshape(-1)[retained_linear]
        retained_keys = retained_batch * num_graph_nodes + retained_nodes

        unique_keys, inverse = torch.unique(
            retained_keys, sorted=True, return_inverse=True)
        retained_positions = torch.arange(
            retained_keys.numel(), dtype=torch.long)
        first_positions = torch.full(
            (unique_keys.numel(),),
            retained_keys.numel(),
            dtype=torch.long,
        )
        first_positions.scatter_reduce_(
            0, inverse, retained_positions, reduce='amin', include_self=True)

        node_width = nodes.size(1)
        node_slots = torch.arange(node_width, dtype=torch.long)
        node_mask = node_slots.unsqueeze(0) < node_counts.unsqueeze(1)
        existing_batch = root_rows.expand(-1, node_width)[node_mask]
        existing_keys = (
            existing_batch * num_graph_nodes + nodes[node_mask])
        existing_local = node_slots.unsqueeze(0).expand(
            batch_size, -1)[node_mask]
        existing_keys, existing_order = torch.sort(existing_keys)
        existing_local = existing_local[existing_order]

        lookup = torch.searchsorted(existing_keys, unique_keys)
        safe_lookup = lookup.clamp(max=existing_keys.numel() - 1)
        already_visited = (
            (lookup < existing_keys.numel())
            & (existing_keys[safe_lookup] == unique_keys)
        )

        first_for_retained = (
            retained_positions == first_positions[inverse])
        new_first_retained = (
            first_for_retained & ~already_visited[inverse])
        new_first_mask = torch.zeros_like(retained)
        new_first_mask.reshape(-1)[
            retained_linear[new_first_retained]] = True
        new_ranks = new_first_mask.cumsum(dim=1) - 1
        new_local_dense = node_counts.unsqueeze(1) + new_ranks

        first_dense_positions = retained_linear[first_positions]
        local_for_unique = existing_local[safe_lookup].clone()
        local_for_unique[~already_visited] = (
            new_local_dense.reshape(-1)[
                first_dense_positions[~already_visited]])
        retained_neighbor_local = local_for_unique[inverse]

        retained_ranks = retained.cumsum(dim=1) - 1
        retained_ranks = retained_ranks.reshape(-1)[retained_linear]
        added_edge_counts = torch.bincount(
            retained_batch, minlength=batch_size)
        next_edge_counts = edge_counts + added_edge_counts
        next_edge_width = int(next_edge_counts.max())
        next_edge_index = torch.zeros(
            (batch_size, 2, next_edge_width), dtype=torch.long)
        next_edge_index[:, :, :edge_index.size(2)] = edge_index
        edge_positions = edge_counts[retained_batch] + retained_ranks
        next_edge_index[
            retained_batch, 0, edge_positions] = retained_neighbor_local
        next_edge_index[
            retained_batch, 1, edge_positions] = retained_parents
        edge_index = next_edge_index
        edge_counts = next_edge_counts

        if not bool(new_first_retained.any()):
            break

        new_batch = retained_batch[new_first_retained]
        new_global = retained_nodes[new_first_retained]
        new_local = retained_neighbor_local[new_first_retained]
        new_rank = new_ranks.reshape(-1)[
            retained_linear[new_first_retained]]
        added_node_counts = torch.bincount(
            new_batch, minlength=batch_size)
        next_node_counts = node_counts + added_node_counts
        next_node_width = int(next_node_counts.max())
        next_nodes = torch.zeros(
            (batch_size, next_node_width), dtype=torch.long)
        next_nodes[:, :nodes.size(1)] = nodes
        next_nodes[new_batch, new_local] = new_global
        nodes = next_nodes
        node_counts = next_node_counts

        next_frontier_width = int(added_node_counts.max())
        frontier_nodes = torch.zeros(
            (batch_size, next_frontier_width), dtype=torch.long)
        frontier_local = torch.zeros_like(frontier_nodes)
        frontier_nodes[new_batch, new_rank] = new_global
        frontier_local[new_batch, new_rank] = new_local
        frontier_counts = added_node_counts

    subgraphs = []
    for batch_idx in range(batch_size):
        num_nodes = int(node_counts[batch_idx])
        num_edges = int(edge_counts[batch_idx])
        subgraphs.append(RootedSubgraph(
            root=int(roots[batch_idx]),
            nodes=nodes[batch_idx, :num_nodes].clone(),
            edge_index=edge_index[
                batch_idx, :, :num_edges].clone().contiguous(),
        ))
    return subgraphs




def _bernoulli_positions(
    population_size: int,
    probability: float,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample Bernoulli successes in order without scanning the population."""
    if population_size <= 0 or probability <= 0.0:
        return torch.empty(0, dtype=torch.long)
    if probability >= 1.0:
        return torch.arange(population_size, dtype=torch.long)

    log_failure = math.log1p(-probability)
    cursor = -1
    sampled = []
    while cursor < population_size - 1:
        remaining = population_size - cursor - 1
        expected = remaining * probability
        draws = max(
            16,
            math.ceil(expected + 8.0 * math.sqrt(expected * (1.0 - probability)) + 8.0),
        )
        uniforms = torch.rand(draws, dtype=torch.float64, generator=generator)
        gaps = torch.floor(torch.log1p(-uniforms) / log_failure).to(torch.long)
        positions = cursor + torch.cumsum(gaps + 1, dim=0)
        valid = positions < population_size
        if bool(valid.any()):
            sampled.append(positions[valid])
        if not bool(valid.all()):
            break
        cursor = int(positions[-1])

    return torch.cat(sampled) if sampled else torch.empty(0, dtype=torch.long)


def sample_roots(num_nodes: int, p1: float, generator: torch.Generator = None,
                 candidate_nodes: torch.Tensor = None) -> torch.Tensor:
    """Sample a fresh independent-Bernoulli root set without an O(n) scan.

    Geometric gaps are exactly the inter-arrival law of independent Bernoulli
    successes. Candidate nodes retain their input order, so every eligible node
    is independently selected with probability ``p1`` on every call.
    """
    if candidate_nodes is None:
        positions = _bernoulli_positions(num_nodes, p1, generator)
        return positions

    pool = candidate_nodes.detach().to(device="cpu", dtype=torch.long)
    positions = _bernoulli_positions(int(pool.numel()), p1, generator)
    return pool[positions]
