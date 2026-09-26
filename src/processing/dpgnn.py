"""DP-GNN degree sampling and complete root-local physical batches."""

from typing import Any, Iterable

import torch

from .padded import PaddedRootedBatch
from .sparse_expand import SparseAdjacency


def sample_training_edges(data: Any, *, max_degree: int, seed: int) -> torch.Tensor:
    """Sample incoming arc multiplicities, then deduplicate and drop overflow."""
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    num_nodes = int(data.num_nodes)
    loops = torch.arange(num_nodes, dtype=torch.long, device="cpu")
    self_edges = torch.stack((loops, loops))
    if num_nodes == 0 or data.edge_index.numel() == 0:
        return self_edges

    original = data.edge_index.detach().to(device="cpu", dtype=torch.long)
    edges = torch.cat((original, original.flip(0)), dim=1)
    order = torch.argsort(edges[1], stable=True)
    senders, receivers = edges[:, order]
    incoming_counts = torch.bincount(receivers, minlength=num_nodes)
    probabilities = max_degree / (2.0 * incoming_counts[receivers])
    generator = torch.Generator().manual_seed(seed)
    selected = torch.rand(receivers.numel(), generator=generator, device="cpu") <= probabilities

    # Multiplicities affect independent draws, but overflow counts unique senders.
    keys = torch.unique(receivers[selected] * num_nodes + senders[selected], sorted=True)
    sampled_receivers = torch.div(keys, num_nodes, rounding_mode="floor")
    retained_counts = torch.bincount(sampled_receivers, minlength=num_nodes)
    keep = retained_counts[sampled_receivers] <= max_degree
    sampled_edges = torch.stack((keys[keep] % num_nodes, sampled_receivers[keep]))
    # Sampled self-arcs remain distinct from the explicit self-loop vector.
    return torch.cat((self_edges, sampled_edges), dim=1)


def sample_dpgnn_roots(
    num_nodes: int, batch_size: int, *, generator: torch.Generator,
) -> torch.Tensor:
    """Draw a fresh uniform fixed-size root subset without replacement on CPU."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if batch_size < 1 or batch_size > num_nodes:
        raise ValueError("batch_size must be between 1 and num_nodes")
    return torch.randperm(num_nodes, generator=generator, device="cpu")[:batch_size]


def iter_dpgnn_batches(
    roots: torch.Tensor, *,
    adjacency: SparseAdjacency,
    x: torch.Tensor,
    y: torch.Tensor,
    max_padded_nodes: int,
    device: torch.device,
    radius: int = 1,
) -> Iterable[PaddedRootedBatch]:
    """Gather complete outgoing neighborhoods without a semantic size cap.

    ``max_padded_nodes`` controls physical chunks, not receptive fields: an
    oversized root is yielded intact on its own. Radius one retains the
    vectorized star representation; deeper batches also carry local arcs.
    """
    if type(radius) is not int or radius < 1:
        raise ValueError("radius must be a positive integer")
    if max_padded_nodes < 1:
        raise ValueError("max_padded_nodes must be positive")
    if radius > 1:
        yield from _iter_multihop_batches(
            roots, adjacency=adjacency, x=x, y=y, radius=radius,
            max_padded_nodes=max_padded_nodes, device=device)
        return

    roots = roots.detach().to(device="cpu", dtype=torch.long)
    starts = adjacency.rowptr[roots]
    neighbor_counts = adjacency.rowptr[roots + 1] - starts
    star_sizes = (neighbor_counts + 1).tolist()
    chunk_start = 0
    while chunk_start < len(star_sizes):
        chunk_end = chunk_start + 1
        max_nodes = star_sizes[chunk_start]
        while chunk_end < len(star_sizes):
            next_max_nodes = max(max_nodes, star_sizes[chunk_end])
            if (chunk_end - chunk_start + 1) * next_max_nodes > max_padded_nodes:
                break
            max_nodes = next_max_nodes
            chunk_end += 1

        chunk_count = chunk_end - chunk_start
        counts = neighbor_counts[chunk_start:chunk_end]
        slots = torch.arange(max_nodes, dtype=torch.long, device="cpu")
        node_mask = slots.unsqueeze(0) <= counts.unsqueeze(1)
        node_ids = torch.zeros((chunk_count, max_nodes), dtype=torch.long, device="cpu")
        node_ids[:, 0] = roots[chunk_start:chunk_end]
        if max_nodes > 1 and adjacency.col.numel():
            neighbor_mask = node_mask[:, 1:]
            offsets = starts[chunk_start:chunk_end, None] + slots[None, :-1]
            safe_offsets = offsets.masked_fill(~neighbor_mask, 0)
            node_ids[:, 1:] = adjacency.col[safe_offsets].masked_fill(~neighbor_mask, 0)

        node_ids = node_ids.to(device)
        node_mask = node_mask.to(device)
        chunk_roots = node_ids[:, 0]
        features = x[node_ids]
        features.masked_fill_(~node_mask.unsqueeze(-1), 0)
        edge_index = torch.zeros(
            (chunk_count, 2, max_nodes), dtype=torch.long, device=device)
        edge_index[:, 1] = slots.to(device).unsqueeze(0).expand(
            chunk_count, -1).masked_fill(~node_mask, 0)
        yield PaddedRootedBatch(
            roots=chunk_roots,
            node_ids=node_ids,
            features=features,
            node_mask=node_mask,
            edge_index=edge_index,
            edge_mask=node_mask,
            root_index=torch.zeros(chunk_count, dtype=torch.long, device=device),
            labels=y[chunk_roots],
            loss_mask=torch.ones(chunk_count, dtype=torch.bool, device=device),
        )
        chunk_start = chunk_end


def _iter_multihop_batches(
    roots: torch.Tensor, *, adjacency: SparseAdjacency, x: torch.Tensor,
    y: torch.Tensor, radius: int, max_padded_nodes: int, device: torch.device,
) -> Iterable[PaddedRootedBatch]:
    """BFS deduplicates vertices while keeping every dependency arc.

    Only vertices at distance < radius need outgoing arcs: the final layer
    at the root cannot depend on an update at the outer boundary.
    """
    rowptr, col = adjacency.rowptr.numpy(), adjacency.col.numpy()

    def neighborhood(root: int):
        nodes, indices, edges = [root], {root: 0}, []
        frontier = [root]
        for _ in range(radius):
            following = []
            for node in frontier:
                sender = indices[node]
                for neighbor in col[rowptr[node]:rowptr[node + 1]].tolist():
                    if neighbor not in indices:
                        indices[neighbor] = len(nodes)
                        nodes.append(neighbor)
                        following.append(neighbor)
                    edges.append((sender, indices[neighbor]))
            frontier = following
            if not frontier:
                break
        return nodes, edges

    def pack(graphs):
        count = len(graphs)
        width = max(len(nodes) for nodes, _ in graphs)
        edge_width = max(len(edges) for _, edges in graphs)
        ids = torch.zeros((count, width), dtype=torch.long)
        mask = torch.zeros((count, width), dtype=torch.bool)
        edges = torch.zeros((count, 2, edge_width), dtype=torch.long)
        edge_mask = torch.zeros((count, edge_width), dtype=torch.bool)
        for index, (nodes, arcs) in enumerate(graphs):
            ids[index, :len(nodes)] = torch.tensor(nodes, dtype=torch.long)
            mask[index, :len(nodes)] = True
            if arcs:
                edges[index, :, :len(arcs)] = torch.tensor(arcs, dtype=torch.long).t()
                edge_mask[index, :len(arcs)] = True
        ids, mask = ids.to(device), mask.to(device)
        features = x[ids].masked_fill(~mask.unsqueeze(-1), 0)
        return PaddedRootedBatch(
            roots=ids[:, 0], node_ids=ids, features=features, node_mask=mask,
            edge_index=edges.to(device), edge_mask=edge_mask.to(device),
            root_index=torch.zeros(count, dtype=torch.long, device=device),
            labels=y[ids[:, 0]],
            loss_mask=torch.ones(count, dtype=torch.bool, device=device))

    pending, width = [], 0
    for root in roots.detach().cpu().tolist():
        graph = neighborhood(root)
        next_width = max(width, len(graph[0]))
        if pending and (len(pending) + 1) * next_width > max_padded_nodes:
            yield pack(pending)
            pending, width = [], 0
        pending.append(graph)
        width = max(width, len(graph[0]))
    if pending:
        yield pack(pending)
