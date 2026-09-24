"""Deterministic, saved graph-disjoint splits for inductive node classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphPartition:
    """One induced graph, its original node identifiers, and scored nodes."""

    data: Any
    node_ids: torch.Tensor
    stats: Mapping[str, Any]
    eval_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        num_nodes = int(
            self.data.num_nodes
            if getattr(self.data, "num_nodes", None) is not None
            else self.data.x.size(0)
        )
        eval_mask = self.eval_mask
        if eval_mask is None:
            eval_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.node_ids.device)
            object.__setattr__(self, "eval_mask", eval_mask)
        if (
            not isinstance(eval_mask, torch.Tensor)
            or eval_mask.dtype != torch.bool
            or eval_mask.ndim != 1
            or eval_mask.numel() != num_nodes
        ):
            raise ValueError(
                "partition eval_mask must be a one-dimensional boolean tensor "
                f"with {num_nodes} entries"
            )


@dataclass(frozen=True)
class InductiveSplit:
    """Three inductive graph views with a shared task and class space."""

    train: GraphPartition
    val: GraphPartition
    test: GraphPartition
    masks: Mapping[str, torch.Tensor]
    num_classes: int
    path: Path
    primary_metric: str = "accuracy"
    binary: bool = False
    metric_ignore_label: int | None = None
    domain_split: Mapping[str, Any] | None = None
    domain_split_id: str | None = None

    def to(self, device: torch.device | str) -> "InductiveSplit":
        return InductiveSplit(
            **{
                name: GraphPartition(
                    part.data.to(device),
                    part.node_ids.to(device),
                    part.stats,
                    part.eval_mask.to(device),
                )
                for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
            },
            masks={name: mask.to(device) for name, mask in self.masks.items()},
            num_classes=self.num_classes,
            path=self.path,
            primary_metric=self.primary_metric,
            binary=self.binary,
            metric_ignore_label=self.metric_ignore_label,
            domain_split=self.domain_split,
            domain_split_id=self.domain_split_id,
        )


def _num_nodes(data: Any) -> int:
    if getattr(data, "num_nodes", None) is not None:
        return int(data.num_nodes)
    return int(data.x.size(0))


def _num_classes(data: Any, multilabel: bool = False, regression: bool = False) -> int:
    """Return the label space of the full source graph.

    Single-label (default): the categorical class count, from an integer
    class-index target.  Multilabel (e.g. PPI-large's 121 binary functional
    labels per node): the number of label columns, from a 2-D 0/1 target --
    there is no shared "class index" across nodes to take a max over.
    Regression: a single continuous output, always 1.
    """
    if multilabel:
        if data.y.dim() != 2:
            raise ValueError(
                "multilabel targets must be 2-D (num_nodes, num_labels), got "
                f"shape {tuple(data.y.shape)}")
        return int(data.y.size(1))
    if regression:
        return 1
    labels = data.y.detach().cpu().reshape(-1)
    if labels.dtype.is_floating_point:
        raise ValueError("inductive node classification requires categorical "
                         "labels (pass multilabel=True for a 0/1 label matrix, "
                         "or regression=True for a continuous target)")
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


def _native_split_indices(
    data: Any, num_nodes: int, source: str = "native inductive split"
) -> dict[str, torch.Tensor]:
    """Return validated benchmark-provided global partition indices."""
    masks: dict[str, torch.Tensor] = {}
    for name in _SPLITS:
        attribute = f"{name}_mask"
        if not hasattr(data, attribute):
            raise ValueError(f"{source} requires {attribute}")
        mask = getattr(data, attribute)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError(f"{source} {attribute} must be a one-dimensional boolean tensor")
        if mask.numel() != num_nodes:
            raise ValueError(
                f"{source} {attribute} has {mask.numel()} nodes, dataset has {num_nodes}"
            )
        masks[name] = mask.detach().cpu()
    memberships = sum(mask.to(torch.long) for mask in masks.values())
    if torch.any(memberships > 1):
        raise ValueError(f"{source} masks overlap")
    if not torch.all(memberships == 1):
        raise ValueError(f"{source} masks must cover every node exactly once")
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
        if index.unique().numel() != index.numel():
            raise ValueError(f"{source} {name} indices contain duplicates")
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

    mask = mask.detach().cpu()
    node_ids = torch.where(mask)[0]
    edge_index, _ = subgraph(mask, data.edge_index.cpu(), relabel_nodes=True,
                             num_nodes=_num_nodes(data))
    # Data.clone preserves graph-level metadata while avoiding mutation of the
    # shared loaded object. Split masks are represented by GraphPartition.
    partition = data.clone()
    partition.x = data.x.cpu()[node_ids]
    partition.y = data.y.cpu()[node_ids]
    partition.edge_index = edge_index
    partition.num_nodes = int(node_ids.numel())
    if hasattr(data, "domain_id"):
        partition.domain_id = data.domain_id.detach().cpu()[node_ids]
    for name in _SPLITS:
        if hasattr(partition, f"{name}_mask"):
            delattr(partition, f"{name}_mask")
    return partition, node_ids


def graph_statistics(
    data: Any, eval_mask: torch.Tensor | None = None
) -> dict[str, Any]:
    """Comparable directed-edge statistics for one already-induced partition.

    Multilabel targets (2-D) have no single "class" to distribute over, so
    their entry reports per-label positive rate instead. Regression targets
    (1-D float) have no classes at all -- truncating them to int and counting
    "classes" would silently report garbage bins rather than erroring, so
    they get a mean/std entry instead. Both detected from `data.y`'s shape/
    dtype directly, since this is a read-only diagnostic rather than a
    training-behavior switch.
    """
    n = _num_nodes(data)
    if eval_mask is None:
        evaluated_nodes = n
    elif (
        not isinstance(eval_mask, torch.Tensor)
        or eval_mask.dtype != torch.bool
        or eval_mask.ndim != 1
        or eval_mask.numel() != n
    ):
        raise ValueError(
            "eval_mask must be a one-dimensional boolean tensor "
            f"with {n} entries"
        )
    else:
        evaluated_nodes = int(eval_mask.sum())
    edge_index = data.edge_index.cpu()
    degree = torch.bincount(edge_index[0], minlength=n) if n else torch.empty(0, dtype=torch.long)
    isolated = int((degree == 0).sum())
    stats = {
        "nodes": n,
        "evaluated_nodes": evaluated_nodes,
        "edges": int(edge_index.size(1)),
        "average_degree": float(degree.float().mean()) if n else 0.0,
        "maximum_degree": int(degree.max()) if n else 0,
        "isolated_nodes": isolated,
        "isolated_fraction": isolated / n if n else 0.0,
    }
    y = data.y.detach().cpu()
    if y.dim() == 2:
        rates = y.float().mean(dim=0) if n else torch.zeros(y.size(1))
        stats["label_positive_rate"] = {str(i): float(r) for i, r in enumerate(rates)}
    elif y.dtype.is_floating_point:
        labels = y.reshape(-1).float()
        stats["target_mean"] = float(labels.mean()) if n else float("nan")
        stats["target_std"] = float(labels.std()) if n else float("nan")
    else:
        labels = y.reshape(-1).to(torch.long)
        classes, counts = torch.unique(labels, sorted=True, return_counts=True)
        stats["class_distribution"] = {str(int(label)): int(count)
                                       for label, count in zip(classes, counts)}
    return stats


def _normalized_domain_split(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("domain split strategy requires a domain_split mapping")
    allowed = {*_SPLITS, "seed", "val_ratio"}
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"domain_split has unexpected keys: {sorted(unexpected)}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"domain_split is missing keys: {sorted(missing)}")

    normalized = dict(value)
    for name in _SPLITS:
        domains = value[name]
        if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence):
            raise ValueError(f"domain_split[{name!r}] must be a sequence of domain names")
        domains = list(domains)
        if not domains or any(not isinstance(domain, str) or not domain for domain in domains):
            raise ValueError(f"domain_split[{name!r}] must contain nonempty domain names")
        if len(domains) != len(set(domains)):
            raise ValueError(f"domain_split[{name!r}] contains duplicate domains")
        normalized[name] = domains

    if set(normalized["train"]) & (set(normalized["val"]) | set(normalized["test"])):
        raise ValueError("training domains must be disjoint from validation and test domains")
    split_seed = normalized["seed"]
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("domain_split seed must be an integer")
    ratio = normalized["val_ratio"]
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio < 1:
        raise ValueError("domain_split val_ratio must be strictly between zero and one")
    normalized["seed"] = int(split_seed)
    normalized["val_ratio"] = float(ratio)
    return normalized


def _domain_layout(
    data: Any,
    num_nodes: int,
    domain_split: Mapping[str, Any] | None,
    domain_split_id: str | None,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    data_split = getattr(data, "domain_split", None)
    normalized = _normalized_domain_split(
        data_split if domain_split is None else domain_split
    )
    if data_split is not None and normalized != _normalized_domain_split(data_split):
        raise ValueError("requested domain_split does not match dataset domain_split metadata")

    data_split_id = getattr(data, "domain_split_id", None)
    resolved_id = data_split_id if domain_split_id is None else domain_split_id
    if (
        not isinstance(resolved_id, str)
        or not resolved_id
        or "/" in resolved_id
        or "\\" in resolved_id
        or resolved_id in {".", ".."}
    ):
        raise ValueError("domain split strategy requires a path-safe nonempty domain_split_id")
    if data_split_id is not None and resolved_id != data_split_id:
        raise ValueError("requested domain_split_id does not match dataset metadata")

    domain_names = getattr(data, "domain_names", None)
    if (
        isinstance(domain_names, (str, bytes))
        or not isinstance(domain_names, Sequence)
        or not domain_names
        or any(not isinstance(name, str) or not name for name in domain_names)
        or len(domain_names) != len(set(domain_names))
    ):
        raise ValueError("domain dataset requires unique ordered domain_names")
    domain_names = list(domain_names)
    selected = set().union(*(set(normalized[name]) for name in _SPLITS))
    unknown = selected - set(domain_names)
    if unknown:
        raise ValueError(f"domain_split contains unknown domains: {sorted(unknown)}")
    name_to_id = {name: index for index, name in enumerate(domain_names)}
    for role in _SPLITS:
        if normalized[role] != sorted(normalized[role], key=name_to_id.__getitem__):
            raise ValueError(f"domain_split[{role!r}] is not in canonical domain order")

    domain_id = getattr(data, "domain_id", None)
    if (
        not isinstance(domain_id, torch.Tensor)
        or domain_id.dtype != torch.long
        or domain_id.ndim != 1
        or domain_id.numel() != num_nodes
    ):
        raise ValueError(
            "domain dataset domain_id must be a one-dimensional long tensor "
            f"with {num_nodes} entries"
        )
    domain_id = domain_id.detach().cpu()
    if torch.any(domain_id < 0) or torch.any(domain_id >= len(domain_names)):
        raise ValueError("domain dataset domain_id contains an unknown domain index")
    present = {domain_names[index] for index in torch.unique(domain_id).tolist()}
    if present != selected:
        raise ValueError(
            "domain dataset nodes must contain exactly the domains selected by domain_split"
        )
    indices = _native_split_indices(data, num_nodes, source="domain split")
    masks = _masks_from_indices(num_nodes, indices)
    contexts: dict[str, torch.Tensor] = {}
    for name in _SPLITS:
        ids = torch.tensor(
            [name_to_id[domain] for domain in normalized[name]], dtype=torch.long
        )
        context_mask = torch.isin(domain_id, ids)
        contexts[name] = torch.where(context_mask)[0]
        if contexts[name].numel() == 0:
            raise ValueError(f"domain {name} context is empty")
        if indices[name].numel() == 0:
            raise ValueError(f"domain {name} evaluation mask is empty")
        if torch.any(~context_mask[indices[name]]):
            raise ValueError(f"{name}_mask contains nodes outside its selected domains")
    if not torch.equal(contexts["train"], indices["train"]):
        raise ValueError("training mask must contain every node in the training domains")
    return normalized, resolved_id, masks, contexts


def _validated_context_indices(
    num_nodes: int, value: Any, source: str
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} is missing context indices")
    contexts: dict[str, torch.Tensor] = {}
    for name in _SPLITS:
        index = value.get(name)
        if not isinstance(index, torch.Tensor) or index.ndim != 1:
            raise ValueError(f"{source} {name} context indices must be one-dimensional tensors")
        if index.dtype.is_floating_point or index.dtype == torch.bool:
            raise ValueError(f"{source} {name} context indices must be integer tensors")
        index = index.detach().cpu().to(torch.long)
        if torch.any(index < 0) or torch.any(index >= num_nodes):
            raise ValueError(f"{source} {name} context indices are outside the dataset")
        if index.unique().numel() != index.numel():
            raise ValueError(f"{source} {name} context indices contain duplicates")
        contexts[name] = index
    return contexts


def load_or_create_inductive_split(
    data: Any,
    dataset: str,
    root: str | Path = "data/inductive_splits",
    seed: int = 0,
    split_strategy: Literal["stratified", "native", "domain"] = "stratified",
    multilabel: bool = False,
    regression: bool = False,
    primary_metric: str = "accuracy",
    binary: bool = False,
    metric_ignore_label: int | None = None,
    domain_split: Mapping[str, Any] | None = None,
    domain_split_id: str | None = None,
) -> InductiveSplit:
    """Load or atomically define a saved inductive partition.

    Stratified and native strategies induce three disjoint graphs. The domain
    strategy induces a source-only training graph and complete role-specific
    target-domain contexts. A domain selected for both validation and test is
    intentionally present in both target graphs; their local ``eval_mask``
    values remain complementary.
    """
    if split_strategy not in {"stratified", "native", "domain"}:
        raise ValueError("split_strategy must be 'stratified', 'native', or 'domain'")
    if multilabel and regression:
        raise ValueError("a target cannot be both multilabel and regression")
    if (multilabel or regression) and split_strategy != "native":
        raise ValueError(
            "multilabel/regression targets require split_strategy='native' -- "
            "stratified/domain splitting has no single per-node class to balance on"
        )
    if not isinstance(primary_metric, str) or not primary_metric:
        raise ValueError("primary_metric must be a nonempty string")
    if not isinstance(binary, bool):
        raise ValueError("binary must be a boolean")
    if (
        metric_ignore_label is not None
        and (not isinstance(metric_ignore_label, int) or isinstance(metric_ignore_label, bool))
    ):
        raise ValueError("metric_ignore_label must be an integer or None")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = _num_nodes(data)
    normalized_domain_split: Mapping[str, Any] | None = None
    resolved_domain_split_id: str | None = None

    if split_strategy == "domain":
        (
            normalized_domain_split,
            resolved_domain_split_id,
            current_masks,
            current_contexts,
        ) = _domain_layout(data, n, domain_split, domain_split_id)
        path = root / f"{dataset.lower()}-domain-{resolved_domain_split_id}.pt"
        current_indices = {
            name: torch.where(current_masks[name])[0] for name in _SPLITS
        }
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping) or payload.get("num_nodes") != n:
                raise ValueError(f"saved domain split {path} is corrupt or has the wrong node count")
            if payload.get("split_strategy") != "domain":
                raise ValueError(f"saved domain split {path} has a mismatched strategy")
            if payload.get("domain_split_id") != resolved_domain_split_id:
                raise ValueError(f"saved domain split {path} has a mismatched domain_split_id")
            try:
                saved_domain_split = _normalized_domain_split(payload.get("domain_split"))
            except ValueError as error:
                raise ValueError(f"saved domain split {path} has invalid configuration") from error
            if saved_domain_split != normalized_domain_split:
                raise ValueError(f"saved domain split {path} has a mismatched configuration")
            indices = _validated_indices(
                n, payload.get("indices", {}), f"saved domain split {path}"
            )
            contexts = _validated_context_indices(
                n, payload.get("context_indices"), f"saved domain split {path}"
            )
            saved_masks = payload.get("masks")
            if not isinstance(saved_masks, Mapping):
                raise ValueError(f"saved domain split {path} is missing global masks")
            for name in _SPLITS:
                mask = saved_masks.get(name)
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or mask.ndim != 1
                    or mask.numel() != n
                ):
                    raise ValueError(f"saved domain split {path} has an invalid {name} mask")
                if (
                    not torch.equal(indices[name], current_indices[name])
                    or not torch.equal(mask.cpu(), current_masks[name])
                    or not torch.equal(contexts[name], current_contexts[name])
                ):
                    raise ValueError(
                        f"saved domain split {path} does not match current dataset metadata"
                    )
        else:
            indices = current_indices
            contexts = current_contexts
            payload = {
                "num_nodes": n,
                "seed": normalized_domain_split["seed"],
                "split_strategy": "domain",
                "indices": indices,
                "masks": current_masks,
                "context_indices": contexts,
                "domain_split": normalized_domain_split,
                "domain_split_id": resolved_domain_split_id,
            }
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        masks = _masks_from_indices(n, indices)
    else:
        suffix = "-native" if split_strategy == "native" else ""
        path = root / f"{dataset.lower()}{suffix}-seed{seed}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping) or "num_nodes" not in payload:
                raise ValueError(f"saved split {path} is corrupt")
            if payload["num_nodes"] != n:
                raise ValueError(
                    f"saved split {path} has {payload['num_nodes']} nodes, dataset has {n}"
                )
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
            payload = {"num_nodes": n, "seed": seed, "indices": indices}
            if split_strategy == "native":
                payload["split_strategy"] = "native"
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        masks = _masks_from_indices(n, indices)
        contexts = indices

    partitions: dict[str, GraphPartition] = {}
    for name in _SPLITS:
        context_mask = torch.zeros(n, dtype=torch.bool)
        context_mask[contexts[name]] = True
        partition, node_ids = _induce(data, context_mask)
        eval_mask = masks[name][node_ids]
        partitions[name] = GraphPartition(
            partition,
            node_ids,
            graph_statistics(partition, eval_mask),
            eval_mask,
        )
    return InductiveSplit(
        **partitions,
        masks=masks,
        num_classes=_num_classes(data, multilabel=multilabel, regression=regression),
        path=path,
        primary_metric=primary_metric,
        binary=binary,
        metric_ignore_label=metric_ignore_label,
        domain_split=normalized_domain_split,
        domain_split_id=resolved_domain_split_id,
    )
