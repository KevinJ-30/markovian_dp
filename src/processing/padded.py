"""Lossless batch-first padding for sampled rooted subgraphs.

Padding is a representation detail only: real node order and every retained local
edge are copied unchanged.  Boolean masks keep padded slots out of message
passing and losses.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .sparse_expand import RootedSubgraph


@dataclass(frozen=True)
class PaddedRootedBatch:
    """A root-first padded batch; the leading dimension is the privacy unit."""

    roots: torch.Tensor
    node_ids: torch.Tensor
    features: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_mask: torch.Tensor
    root_index: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.roots.numel())


def pad_rooted_subgraphs(
    subgraphs: Sequence[RootedSubgraph],
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    device: torch.device,
) -> PaddedRootedBatch:
    """Copy rooted subgraphs into padded tensors without changing topology.

    An empty logical batch is represented by one fully masked sentinel.  Its
    loss is differentiable zero, allowing the private optimizer to perform the
    noise-only update required by the analyzed mechanism.
    """

    device = torch.device(device)
    if not subgraphs:
        feature_shape = tuple(x.shape[1:])
        label_shape = tuple(y.shape[1:])
        return PaddedRootedBatch(
            roots=torch.full((1,), -1, dtype=torch.long, device=device),
            node_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
            features=torch.zeros((1, 1, *feature_shape), dtype=x.dtype, device=device),
            node_mask=torch.zeros((1, 1), dtype=torch.bool, device=device),
            edge_index=torch.zeros((1, 2, 0), dtype=torch.long, device=device),
            edge_mask=torch.zeros((1, 0), dtype=torch.bool, device=device),
            root_index=torch.zeros((1,), dtype=torch.long, device=device),
            labels=torch.zeros((1, *label_shape), dtype=y.dtype, device=device),
            loss_mask=torch.zeros((1,), dtype=torch.bool, device=device),
        )

    batch_size = len(subgraphs)
    nodes = [
        subgraph.nodes.detach().to(device="cpu", dtype=torch.long)
        for subgraph in subgraphs
    ]
    edges = [
        subgraph.edge_index.detach().to(device="cpu", dtype=torch.long)
        for subgraph in subgraphs
    ]
    node_counts = torch.tensor([part.numel() for part in nodes], dtype=torch.long)
    edge_counts = torch.tensor([part.size(1) for part in edges], dtype=torch.long)
    if bool((node_counts < 1).any()):
        raise ValueError("each RootedSubgraph must contain its root")

    max_nodes = int(node_counts.max())
    max_edges = int(edge_counts.max())
    roots = torch.tensor([subgraph.root for subgraph in subgraphs], dtype=torch.long)
    node_mask = (
        torch.arange(max_nodes, dtype=torch.long).unsqueeze(0)
        < node_counts.unsqueeze(1)
    )
    node_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_ids[node_mask] = torch.cat(nodes)
    if bool((node_ids[:, 0] != roots).any()):
        raise ValueError("each RootedSubgraph must store its root at local index 0")

    edge_mask = (
        torch.arange(max_edges, dtype=torch.long).unsqueeze(0)
        < edge_counts.unsqueeze(1)
    )
    edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long)
    nonempty_edges = [part for part in edges if part.numel()]
    if nonempty_edges:
        flat_edges = torch.cat(nonempty_edges, dim=1)
        edge_index[:, 0][edge_mask] = flat_edges[0]
        edge_index[:, 1][edge_mask] = flat_edges[1]
        limits = node_counts.unsqueeze(1)
        invalid = edge_mask & (
            (edge_index[:, 0] < 0)
            | (edge_index[:, 1] < 0)
            | (edge_index[:, 0] >= limits)
            | (edge_index[:, 1] >= limits)
        )
        if bool(invalid.any()):
            raise ValueError("RootedSubgraph edge_index contains an invalid local node")

    roots = roots.to(device)
    node_ids = node_ids.to(device)
    node_mask = node_mask.to(device)
    edge_index = edge_index.to(device)
    edge_mask = edge_mask.to(device)
    features = x.to(device)[node_ids]
    features = features * node_mask.unsqueeze(-1).to(features.dtype)
    labels = y.to(device)[roots]
    supervised = train_mask.to(device)[roots].to(dtype=torch.bool)
    return PaddedRootedBatch(
        roots=roots,
        node_ids=node_ids,
        features=features,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        root_index=torch.zeros(batch_size, dtype=torch.long, device=device),
        labels=labels,
        loss_mask=supervised,
    )


def iter_padded_root_batches(
    subgraphs: Sequence[RootedSubgraph],
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    device: torch.device,
    max_padded_nodes: int,
) -> Iterable[PaddedRootedBatch]:
    """Yield ordered physical chunks bounded by padded node slots."""

    if max_padded_nodes <= 0:
        raise ValueError("max_padded_nodes must be positive")
    if not subgraphs:
        yield pad_rooted_subgraphs(
            (), x=x, y=y, train_mask=train_mask, device=device)
        return

    chunk = []
    chunk_max_nodes = 0
    for subgraph in subgraphs:
        next_max_nodes = max(chunk_max_nodes, subgraph.num_nodes)
        next_slots = (len(chunk) + 1) * next_max_nodes
        if chunk and next_slots > max_padded_nodes:
            yield pad_rooted_subgraphs(
                chunk, x=x, y=y, train_mask=train_mask, device=device)
            chunk = []
            chunk_max_nodes = 0
        chunk.append(subgraph)
        chunk_max_nodes = max(chunk_max_nodes, subgraph.num_nodes)
    if chunk:
        yield pad_rooted_subgraphs(
            chunk, x=x, y=y, train_mask=train_mask, device=device)
