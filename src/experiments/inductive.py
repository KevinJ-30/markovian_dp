"""Deterministic, saved graph-disjoint splits for inductive node classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphPartition:
    """One induced graph and its original node identifiers."""

    data: Any
    node_ids: torch.Tensor
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class InductiveSplit:
    """Three disjoint induced graphs with the source graph's class space."""

    train: GraphPartition
    val: GraphPartition
    test: GraphPartition
    masks: Mapping[str, torch.Tensor]
    num_classes: int
    path: Path

    def to(self, device: torch.device | str) -> "InductiveSplit":
        return InductiveSplit(
            **{
                name: GraphPartition(part.data.to(device), part.node_ids.to(device), part.stats)
                for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
            },
            masks={name: mask.to(device) for name, mask in self.masks.items()},
            num_classes=self.num_classes,
            path=self.path,
        )


def _num_nodes(data: Any) -> int:
    if getattr(data, "num_nodes", None) is not None:
        return int(data.num_nodes)
    return int(data.x.size(0))


def _num_classes(data: Any) -> int:
    """Return the categorical class space of the full source graph."""
    labels = data.y.detach().cpu().reshape(-1)
    if labels.dtype.is_floating_point:
        raise ValueError("inductive node classification requires categorical labels")
    return int(labels.max()) + 1


def _split_indices(labels: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    """Return a deterministic 60/20/20 stratified split.

    Classes with fewer than five examples are shuffled deterministically and use
    the closest feasible allocation. This is deliberately a split-time policy,
    not a training-time operation; trainers receive only their train partition.
    """
    labels = labels.detach().cpu().reshape(-1)
    if labels.dtype.is_floating_point:
        raise ValueError("inductive node classification requires categorical labels")
    generator = torch.Generator().manual_seed(seed)
    result: dict[str, list[torch.Tensor]] = {name: [] for name in _SPLITS}
    for label in torch.unique(labels, sorted=True):
        members = torch.where(labels == label)[0]
        members = members[torch.randperm(members.numel(), generator=generator)]
        n = members.numel()
        train_n = int(round(n * 0.60))
        val_n = int(round(n * 0.20))
        # Preserve every non-empty class in train when possible. Validation/test
        # are allowed to be empty only for genuinely tiny classes.
        if n >= 3:
            train_n = min(max(train_n, 1), n - 2)
            val_n = min(max(val_n, 1), n - train_n - 1)
        else:
            train_n = max(1, train_n)
            val_n = min(val_n, n - train_n)
        result["train"].append(members[:train_n])
        result["val"].append(members[train_n:train_n + val_n])
        result["test"].append(members[train_n + val_n:])
    return {name: torch.cat(parts).sort().values for name, parts in result.items()}


def _native_split_indices(data: Any, num_nodes: int) -> dict[str, torch.Tensor]:
    """Return validated benchmark-provided global partition indices."""
    masks: dict[str, torch.Tensor] = {}
    for name in _SPLITS:
        attribute = f"{name}_mask"
        if not hasattr(data, attribute):
            raise ValueError(f"native inductive split requires {attribute}")
        mask = getattr(data, attribute)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError(f"native inductive split {attribute} must be a one-dimensional boolean tensor")
        if mask.numel() != num_nodes:
            raise ValueError(
                f"native inductive split {attribute} has {mask.numel()} nodes, dataset has {num_nodes}"
            )
        masks[name] = mask.detach().cpu()
    memberships = sum(mask.to(torch.long) for mask in masks.values())
    if torch.any(memberships > 1):
        raise ValueError("native inductive split masks overlap")
    if not torch.all(memberships == 1):
        raise ValueError("native inductive split masks must cover every node exactly once")
    return {name: torch.where(masks[name])[0] for name in _SPLITS}


def _validated_indices(
    num_nodes: int, indices: Mapping[str, torch.Tensor], source: str
) -> dict[str, torch.Tensor]:
    """Validate saved global indices before reconstructing partition masks."""
    normalized: dict[str, torch.Tensor] = {}
    membership = torch.zeros(num_nodes, dtype=torch.bool)
    for name in _SPLITS:
        if name not in indices:
            raise ValueError(f"{source} is missing {name} indices")
        index = indices[name]
        if not isinstance(index, torch.Tensor) or index.ndim != 1:
            raise ValueError(f"{source} {name} indices must be one-dimensional tensors")
        if index.dtype.is_floating_point or index.dtype == torch.bool:
            raise ValueError(f"{source} {name} indices must be integer tensors")
        index = index.detach().cpu().to(torch.long)
        if torch.any(index < 0) or torch.any(index >= num_nodes):
            raise ValueError(f"{source} {name} indices are outside the dataset")
        if torch.any(membership[index]):
            raise ValueError(f"{source} indices overlap")
        membership[index] = True
        normalized[name] = index
    if not torch.all(membership):
        raise ValueError(f"{source} indices must cover every node exactly once")
    return normalized


def _masks_from_indices(num_nodes: int, indices: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    masks = {}
    for name in _SPLITS:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices[name]] = True
        masks[name] = mask
    if sum(int(mask.sum()) for mask in masks.values()) != num_nodes:
        raise RuntimeError("inductive split does not cover every node")
    return masks


def _induce(data: Any, mask: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Return a relabelled induced graph without importing PyG at module import."""
    from torch_geometric.utils import subgraph

    node_ids = torch.where(mask)[0]
    edge_index, _ = subgraph(mask, data.edge_index.cpu(), relabel_nodes=True,
                             num_nodes=_num_nodes(data))
    # Data.clone preserves non-tensor metadata while avoiding mutation of a
    # shared loaded object. Remove masks: a partition is entirely one split.
    partition = data.clone()
    partition.x = data.x.cpu()[node_ids]
    partition.y = data.y.cpu()[node_ids]
    partition.edge_index = edge_index
    partition.num_nodes = int(node_ids.numel())
    for name in _SPLITS:
        if hasattr(partition, f"{name}_mask"):
            delattr(partition, f"{name}_mask")
    return partition, node_ids


def graph_statistics(data: Any) -> dict[str, Any]:
    """Comparable directed-edge statistics for one already-induced partition."""
    n = _num_nodes(data)
    edge_index = data.edge_index.cpu()
    degree = torch.bincount(edge_index[0], minlength=n) if n else torch.empty(0, dtype=torch.long)
    labels = data.y.detach().cpu().reshape(-1).to(torch.long)
    classes, counts = torch.unique(labels, sorted=True, return_counts=True)
    isolated = int((degree == 0).sum())
    return {
        "nodes": n,
        "edges": int(edge_index.size(1)),
        "average_degree": float(degree.float().mean()) if n else 0.0,
        "maximum_degree": int(degree.max()) if n else 0,
        "isolated_nodes": isolated,
        "isolated_fraction": isolated / n if n else 0.0,
        "class_distribution": {str(int(label)): int(count) for label, count in zip(classes, counts)},
    }


def load_or_create_inductive_split(
    data: Any,
    dataset: str,
    root: str | Path = "data/inductive_splits",
    seed: int = 0,
    split_strategy: Literal["stratified", "native"] = "stratified",
) -> InductiveSplit:
    """Load or atomically define a saved graph-disjoint partition.

    Stratified splits use the repository's deterministic 60/20/20 policy.
    Native splits retain validated benchmark-provided masks. The saved payload
    contains global indices only; partitions are reconstructed from the current
    dataset so stale serialized PyG objects never become a compatibility boundary.
    """
    if split_strategy not in {"stratified", "native"}:
        raise ValueError("split_strategy must be 'stratified' or 'native'")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    suffix = "-native" if split_strategy == "native" else ""
    path = root / f"{dataset.lower()}{suffix}-seed{seed}.pt"
    n = _num_nodes(data)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or "num_nodes" not in payload:
            raise ValueError(f"saved split {path} is corrupt")
        if payload["num_nodes"] != n:
            raise ValueError(f"saved split {path} has {payload['num_nodes']} nodes, dataset has {n}")
        if split_strategy == "native" and payload.get("split_strategy") != "native":
            raise ValueError(f"saved native split {path} has a mismatched strategy")
        saved_indices = payload.get("indices")
        if not isinstance(saved_indices, Mapping):
            raise ValueError(f"saved split {path} is missing indices")
        indices = _validated_indices(n, saved_indices, f"saved split {path}")
    else:
        indices = (
            _native_split_indices(data, n)
            if split_strategy == "native"
            else _split_indices(data.y, seed)
        )
        payload: dict[str, Any] = {"num_nodes": n, "seed": seed, "indices": indices}
        if split_strategy == "native":
            payload["split_strategy"] = "native"
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    masks = _masks_from_indices(n, indices)
    partitions = {}
    for name in _SPLITS:
        partition, node_ids = _induce(data, masks[name])
        partitions[name] = GraphPartition(partition, node_ids, graph_statistics(partition))
    return InductiveSplit(
        **partitions,
        masks=masks,
        num_classes=_num_classes(data),
        path=path,
    )
