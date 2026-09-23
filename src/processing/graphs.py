"""Shared structural preprocessing for train-time graph inputs."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np
import torch

from src.processing.splits import (
    GraphPartition,
    InductiveSplit,
    graph_statistics,
)


DegreeCapMode = Literal["directed", "undirected"]


def _validate_degree_bound(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _deduplicate(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return edge_index
    key = edge_index[0] * num_nodes + edge_index[1]
    unique = torch.unique(key)
    if unique.numel() == key.numel():
        return edge_index
    return torch.stack((unique // num_nodes, unique % num_nodes))


def _make_bidirectional(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return edge_index
    keys = edge_index[0] * num_nodes + edge_index[1]
    reverse_keys = edge_index[1] * num_nodes + edge_index[0]
    keys = torch.unique(torch.cat((keys, reverse_keys)))
    return torch.stack((keys // num_nodes, keys % num_nodes))




def _cap_directed(
    edge_index: torch.Tensor,
    *,
    max_in_degree: int | None,
    max_out_degree: int | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    def cap_row(edges: torch.Tensor, row: int, bound: int) -> torch.Tensor:
        count = edges.size(1)
        if count == 0:
            return edges
        shuffle = torch.argsort(torch.rand(count, generator=generator))
        order = shuffle[torch.argsort(edges[row, shuffle], stable=True)]
        sorted_keys = edges[row, order]
        change = torch.ones(count, dtype=torch.bool)
        change[1:] = sorted_keys[1:] != sorted_keys[:-1]
        index = torch.arange(count)
        group_id = torch.cumsum(change.to(torch.long), dim=0) - 1
        rank = index - index[change][group_id]
        return edges[:, order[rank < bound]]

    result = edge_index
    if max_in_degree is not None:
        result = cap_row(result, row=1, bound=max_in_degree)
    if max_out_degree is not None:
        result = cap_row(result, row=0, bound=max_out_degree)
    return result


def _cap_undirected(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    bound: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return edge_index
    source, target = edge_index
    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    keys = torch.unique(low * num_nodes + high)
    low_array = (keys // num_nodes).numpy()
    high_array = (keys % num_nodes).numpy()
    permutation = torch.randperm(keys.numel(), generator=generator).numpy()
    degrees = np.zeros(num_nodes, dtype=np.int32)
    keep = np.zeros(keys.numel(), dtype=bool)
    for index in permutation:
        u = low_array[index]
        v = high_array[index]
        if degrees[u] < bound and degrees[v] < bound:
            degrees[u] += 1
            degrees[v] += 1
            keep[index] = True
    kept_low = torch.from_numpy(low_array[keep]).to(torch.long)
    kept_high = torch.from_numpy(high_array[keep]).to(torch.long)
    return torch.stack(
        (
            torch.cat((kept_low, kept_high)),
            torch.cat((kept_high, kept_low)),
        )
    )


def preprocess_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    max_in_degree: int | None = None,
    max_out_degree: int | None = None,
    degree_cap_mode: DegreeCapMode = "directed",
    make_bidirectional: bool = True,
    add_self_loops: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Remove loops, symmetrize and deduplicate arcs, cap, then add exact loops."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    num_nodes = int(num_nodes)
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a tensor with shape [2, E]")
    if edge_index.dtype == torch.bool or edge_index.dtype.is_floating_point:
        raise ValueError("edge_index must contain integer node indices")
    max_in_degree = _validate_degree_bound("max_in_degree", max_in_degree)
    max_out_degree = _validate_degree_bound("max_out_degree", max_out_degree)
    if degree_cap_mode not in ("directed", "undirected"):
        raise ValueError("degree_cap_mode must be 'directed' or 'undirected'")
    if degree_cap_mode == "undirected" and (
        max_in_degree is None
        or max_out_degree is None
        or max_in_degree != max_out_degree
    ):
        raise ValueError(
            "undirected degree capping requires equal max_in_degree and max_out_degree"
        )

    result = edge_index.detach().cpu().to(torch.long).clone()
    if result.numel() and (
        bool(torch.any(result < 0)) or bool(torch.any(result >= num_nodes))
    ):
        raise ValueError("edge_index contains endpoints outside [0, num_nodes)")
    result = result[:, result[0] != result[1]]
    result = (
        _make_bidirectional(result, num_nodes)
        if make_bidirectional
        else _deduplicate(result, num_nodes)
    )
    if degree_cap_mode == "directed":
        result = _cap_directed(
            result,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            generator=generator,
        )
    else:
        result = _cap_undirected(
            result,
            num_nodes=num_nodes,
            bound=max_in_degree,
            generator=generator,
        )
    if add_self_loops:
        nodes = torch.arange(num_nodes)
        result = torch.cat((result, torch.stack((nodes, nodes))), dim=1)
    return result


def max_degrees(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, int]:
    """Return maximum nonnegative in- and out-degree, or zeros for no nodes."""
    if num_nodes == 0:
        return 0, 0
    edges = edge_index.detach().cpu().to(torch.long)
    out_degree = torch.bincount(edges[0], minlength=num_nodes)
    in_degree = torch.bincount(edges[1], minlength=num_nodes)
    return int(in_degree.max()), int(out_degree.max())


def preprocess_graph(
    data: Any,
    *,
    max_in_degree: int | None = None,
    max_out_degree: int | None = None,
    degree_cap_mode: DegreeCapMode = "directed",
    make_bidirectional: bool = True,
    add_self_loops: bool = False,
    generator: torch.Generator | None = None,
) -> Any:
    """Clone a PyG-like graph and structurally preprocess its edge tensor."""
    result = data.clone()
    edge_device = data.edge_index.device
    num_nodes = int(data.num_nodes if data.num_nodes is not None else data.x.size(0))
    result.edge_index = preprocess_edges(
        data.edge_index,
        num_nodes,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        degree_cap_mode=degree_cap_mode,
        make_bidirectional=make_bidirectional,
        add_self_loops=add_self_loops,
        generator=generator,
    ).to(edge_device)
    return result


def preprocess_inductive_split(
    split: InductiveSplit,
    *,
    max_in_degree: int | None = None,
    max_out_degree: int | None = None,
    degree_cap_mode: DegreeCapMode = "directed",
    make_bidirectional: bool = True,
    add_self_loops: bool = False,
    generator: torch.Generator | None = None,
) -> InductiveSplit:
    """Preprocess each disjoint partition and rebuild graph statistics."""
    partitions: dict[str, GraphPartition] = {}
    for name in ("train", "val", "test"):
        partition = getattr(split, name)
        data = preprocess_graph(
            partition.data,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            degree_cap_mode=degree_cap_mode,
            make_bidirectional=make_bidirectional,
            add_self_loops=add_self_loops,
            generator=generator,
        )
        partitions[name] = GraphPartition(
            data=data,
            node_ids=partition.node_ids,
            stats=graph_statistics(data, partition.eval_mask),
            eval_mask=partition.eval_mask,
        )
    return InductiveSplit(
        **partitions,
        masks=split.masks,
        num_classes=split.num_classes,
        path=split.path,
        primary_metric=split.primary_metric,
        binary=split.binary,
        metric_ignore_label=split.metric_ignore_label,
        domain_split=split.domain_split,
        domain_split_id=split.domain_split_id,
    )


def make_training_graph(test_graph):
    """Return the graph visible to training, separate from the test graph."""
    train_graph = test_graph.clone()
    if hasattr(test_graph, "train_edge_index"):
        train_graph.edge_index = test_graph.train_edge_index
    else:
        is_train = test_graph.train_mask
        edge_index = test_graph.edge_index
        train_graph.edge_index = edge_index[
            :, is_train[edge_index[0]] & is_train[edge_index[1]]
        ]
    return train_graph
