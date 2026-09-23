"""Domain-disjoint node-classification dataset loaders.

The raw artifacts are kept in provenance-specific caches and converted into one
PyG graph whose domains are disconnected components.  Split masks describe the
requested source/target-domain protocol; validation and test may share a target
component, in which case only their score masks are split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

TWITCH_REVISION = "af14a88470d30b1dadd3803d911dfc1064bcf172"
TWITCH_RAW_URL = (
    "https://raw.githubusercontent.com/CUAI/Non-Homophily-Benchmarks/"
    f"{TWITCH_REVISION}/data/twitch"
)
TWITCH_DOMAINS = ("de", "engb", "es", "fr", "ptbr", "ru", "tw")
TWITCH_NUM_FEATURES = 3170

FB100_REVISION = "9a92bf1e84f73b7b24dd745eb14f13e4d1979769"
FB100_RAW_URL = (
    "https://raw.githubusercontent.com/sisaman/pyg-datasets/"
    f"{FB100_REVISION}/datasets/facebook100"
)
FB100_FILES = {
    "penn94": "Penn94.mat",
    "amherst41": "Amherst41.mat",
    "cornell5": "Cornell5.mat",
    "johns-hopkins55": "JohnsHopkins55.mat",
    "reed98": "Reed98.mat",
    "caltech36": "Caltech36.mat",
    "berkeley13": "Berkeley13.mat",
    "brown11": "Brown11.mat",
    "columbia2": "Columbia2.mat",
    "yale4": "Yale4.mat",
    "virginia63": "Virginia63.mat",
    "texas80": "Texas80.mat",
    "bingham82": "Bingham82.mat",
    "duke14": "Duke14.mat",
    "princeton12": "Princeton12.mat",
    "washu32": "WashU32.mat",
    "brandeis99": "Brandeis99.mat",
    "carnegie49": "Carnegie49.mat",
}
FB100_DOMAINS = tuple(FB100_FILES)

MAG_RECORD = "10681285"
MAG_RAW_URL = f"https://zenodo.org/api/records/{MAG_RECORD}/files"
# Byte sizes and MD5 digests published by the immutable Zenodo record.
MAG_ARTIFACTS = {
    "us": ("US_labels_20.pt", 80171299, "677b46f78e5fb946b2d9d2e4f76418fb"),
    "cn": ("CN_labels_20.pt", 57592355, "3e09b899d12d5801f39bf9cd187edcad"),
    "de": ("DE_labels_20.pt", 24421347, "3e3830bd6102db954f1b0163761aebc3"),
    "fr": ("FR_labels_20.pt", 16484579, "a2387bdff7841edb395f23d224b0b1c5"),
    "ru": ("RU_labels_20.pt", 18167331, "3c86cf9b3b2052d31a433d5422a7ec5f"),
    "jp": ("JP_labels_20.pt", 20962979, "7910d054a972897fc2466f177cb9fed4"),
}
MAG_DOMAINS = tuple(MAG_ARTIFACTS)

DOMAIN_DATASET_NAMES = ("twitch-explicit", "facebook100", "mag-countries")
DOMAIN_REGISTRIES = {
    "twitch-explicit": TWITCH_DOMAINS,
    "facebook100": FB100_DOMAINS,
    "mag-countries": MAG_DOMAINS,
}
DOMAIN_DEFAULTS = {
    "twitch-explicit": {
        "train": ["de"],
        "val": ["engb"],
        "test": ["es", "fr", "ptbr", "ru", "tw"],
        "seed": 0,
        "val_ratio": 0.2,
    },
    "facebook100": {
        "train": ["johns-hopkins55", "caltech36", "amherst41"],
        "val": ["cornell5", "yale4"],
        "test": ["penn94", "brown11", "texas80"],
        "seed": 0,
        "val_ratio": 0.2,
    },
    "mag-countries": {
        "train": ["us"],
        "val": ["cn"],
        "test": ["cn"],
        "seed": 0,
        "val_ratio": 0.2,
    },
}
DOMAIN_TASKS = {
    "twitch-explicit": {
        "num_classes": 2,
        "task_type": "BINARY",
        "primary_metric": "auroc",
        "metric_ignore_label": None,
    },
    "facebook100": {
        "num_classes": 2,
        "task_type": "MULTICLASS",
        "primary_metric": "accuracy",
        "metric_ignore_label": None,
    },
    "mag-countries": {
        "num_classes": 20,
        "task_type": "MULTICLASS",
        "primary_metric": "accuracy",
        "metric_ignore_label": 19,
    },
}

_ROLE_NAMES = ("train", "val", "test")
_SPLIT_KEYS = frozenset((*_ROLE_NAMES, "seed", "val_ratio"))


def normalize_domain_split(
    dataset_name: str, domain_split: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    """Validate and canonicalize a domain split, returning it and its ID."""
    name = str(dataset_name).lower()
    if name not in DOMAIN_REGISTRIES:
        raise ValueError(f"Unknown domain dataset '{dataset_name}'")
    if domain_split is None:
        supplied: dict[str, Any] = {}
    elif isinstance(domain_split, Mapping):
        supplied = dict(domain_split)
    else:
        raise TypeError("domain_split must be a mapping or None")

    unknown_keys = sorted(set(supplied) - _SPLIT_KEYS)
    if unknown_keys:
        raise ValueError(f"domain_split has unknown keys: {unknown_keys}")

    supplied_roles = [role for role in _ROLE_NAMES if role in supplied]
    if supplied_roles and len(supplied_roles) != len(_ROLE_NAMES):
        missing = [role for role in _ROLE_NAMES if role not in supplied]
        raise ValueError(
            "domain_split must provide train, val, and test together; "
            f"missing {missing}"
        )

    defaults = DOMAIN_DEFAULTS[name]
    registry = DOMAIN_REGISTRIES[name]
    registry_set = set(registry)
    normalized: dict[str, Any] = {}
    for role in _ROLE_NAMES:
        raw = supplied.get(role, defaults[role])
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(f"domain_split['{role}'] must be a sequence of domains")
        values = list(raw)
        if not values:
            raise ValueError(f"domain_split['{role}'] must not be empty")
        if any(not isinstance(value, str) for value in values):
            raise TypeError(f"domain_split['{role}'] entries must be strings")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(
                f"domain_split['{role}'] contains duplicate domains: "
                f"{sorted(duplicates)}"
            )
        unknown = sorted(set(values) - registry_set)
        if unknown:
            raise ValueError(
                f"Unknown {name} domain(s) in '{role}': {unknown}; "
                f"expected canonical names from {list(registry)}"
            )
        selected = set(values)
        normalized[role] = [domain for domain in registry if domain in selected]

    train = set(normalized["train"])
    held_out = set(normalized["val"]) | set(normalized["test"])
    overlap = sorted(train & held_out)
    if overlap:
        raise ValueError(
            "Training domains must be disjoint from validation and test domains; "
            f"overlap: {overlap}"
        )

    seed = supplied.get("seed", defaults["seed"])
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("domain_split['seed'] must be an integer")
    ratio = supplied.get("val_ratio", defaults["val_ratio"])
    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError("domain_split['val_ratio'] must be a real number")
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("domain_split['val_ratio'] must satisfy 0 < val_ratio < 1")
    normalized["seed"] = int(seed)
    normalized["val_ratio"] = ratio

    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    split_id = hashlib.sha256(f"{name}:{serialized}".encode("utf-8")).hexdigest()
    return normalized, split_id


def _check_file(
    path: Path, *, expected_size: int | None = None, expected_md5: str | None = None
) -> None:
    if not path.is_file():
        raise ValueError(f"Expected a regular file at {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(
            f"Size mismatch for {path}: expected {expected_size} bytes, "
            f"found {path.stat().st_size}"
        )
    if expected_md5 is not None:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {path}: expected {expected_md5}, found {actual}"
            )


def _download_atomic(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
) -> Path:
    """Download to a same-directory temporary file and atomically install it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        _check_file(
            temporary, expected_size=expected_size, expected_md5=expected_md5
        )
        os.replace(temporary, destination)
        temporary = None
        return destination
    except (OSError, urllib.error.URLError, ValueError) as error:
        raise RuntimeError(
            f"Could not acquire {url} into cache root {destination.parent}: {error}"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _ensure_file(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    if destination.exists():
        raise ValueError(f"Dataset cache path is not a regular file: {destination}")
    return _download_atomic(url, destination)


def _ensure_mag_file(
    url: str, destination: Path, expected_size: int, expected_md5: str
) -> Path:
    if destination.exists():
        try:
            _check_file(
                destination,
                expected_size=expected_size,
                expected_md5=expected_md5,
            )
            return destination
        except ValueError:
            if destination.is_file():
                destination.unlink()
            else:
                raise
    return _download_atomic(
        url,
        destination,
        expected_size=expected_size,
        expected_md5=expected_md5,
    )


def _parse_bool(value: str, *, path: Path, row_number: int) -> int:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(
        f"Invalid mature label {value!r} in {path} at CSV row {row_number}"
    )


def _load_unique_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _parse_twitch_domain(root: Path, domain: str) -> Data:
    upper = domain.upper()
    folder = root / upper
    base = f"musae_{upper}"
    paths = {
        kind: folder / f"{base}_{kind}.{extension}"
        for kind, extension in (
            ("target", "csv"),
            ("edges", "csv"),
            ("features", "json"),
        )
    }
    for path in paths.values():
        url = f"{TWITCH_RAW_URL}/{upper}/{path.name}"
        _ensure_file(url, path)

    node_to_local: dict[int, int] = {}
    identities: dict[int, tuple[int, int]] = {}
    source_to_graph: dict[int, int] = {}
    labels: list[int] = []
    with paths["target"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "new_id", "mature"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Invalid Twitch target schema in {paths['target']}: "
                f"required columns are {sorted(required)}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                graph_id = int(row["new_id"])
                source_id = int(row["id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid Twitch node ID in {paths['target']} at CSV row "
                    f"{row_number}"
                ) from error
            label = _parse_bool(
                row["mature"], path=paths["target"], row_number=row_number
            )
            if (
                source_id in source_to_graph
                and source_to_graph[source_id] != graph_id
            ):
                raise ValueError(
                    f"Duplicate Twitch source ID {source_id} maps to multiple "
                    f"graph IDs in {paths['target']}"
                )
            identity = (source_id, label)
            if graph_id in node_to_local:
                # The pinned FR file repeats two otherwise identical graph-node
                # records.  Collapse those source duplicates, but never merge
                # conflicting identities or labels.
                if identities[graph_id] != identity:
                    raise ValueError(
                        f"Conflicting duplicate Twitch node ID {graph_id} in "
                        f"{paths['target']}"
                    )
                continue
            node_to_local[graph_id] = len(labels)
            identities[graph_id] = identity
            source_to_graph[source_id] = graph_id
            labels.append(label)
    if not labels:
        raise ValueError(f"Twitch target file has no nodes: {paths['target']}")

    sources: list[int] = []
    targets: list[int] = []
    with paths["edges"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"from", "to"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Invalid Twitch edge schema in {paths['edges']}: "
                "required columns are ['from', 'to']"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                raw_source = int(row["from"])
                raw_target = int(row["to"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid Twitch edge ID in {paths['edges']} at CSV row "
                    f"{row_number}"
                ) from error
            unknown = [
                node_id
                for node_id in (raw_source, raw_target)
                if node_id not in node_to_local
            ]
            if unknown:
                raise ValueError(
                    f"Unknown Twitch node ID(s) {unknown} in {paths['edges']} "
                    f"at CSV row {row_number}"
                )
            sources.append(node_to_local[raw_source])
            targets.append(node_to_local[raw_target])

    feature_rows = _load_unique_json_object(paths["features"])
    x = torch.zeros((len(labels), TWITCH_NUM_FEATURES), dtype=torch.float32)
    seen_feature_nodes: set[int] = set()
    for raw_node, raw_features in feature_rows.items():
        try:
            node_id = int(raw_node)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid Twitch feature node ID {raw_node!r} in {paths['features']}"
            ) from error
        if node_id not in node_to_local:
            raise ValueError(
                f"Unknown Twitch feature node ID {node_id} in {paths['features']}"
            )
        if node_id in seen_feature_nodes:
            raise ValueError(
                f"Duplicate Twitch feature node ID {node_id} in {paths['features']}"
            )
        seen_feature_nodes.add(node_id)
        if isinstance(raw_features, (str, bytes)) or not isinstance(
            raw_features, Sequence
        ):
            raise ValueError(
                f"Features for Twitch node {node_id} must be a list of indices"
            )
        if any(
            isinstance(feature, bool) or not isinstance(feature, Integral)
            for feature in raw_features
        ):
            raise ValueError(f"Invalid feature index for Twitch node {node_id}")
        # The pinned files contain repeated indices for some nodes. They encode
        # the same binary feature and are therefore idempotent, not conflicting.
        features = sorted({int(feature) for feature in raw_features})
        invalid = [
            feature
            for feature in features
            if feature < 0 or feature >= TWITCH_NUM_FEATURES
        ]
        if invalid:
            raise ValueError(
                f"Out-of-range feature indices for Twitch node {node_id}: {invalid}"
            )
        if features:
            x[node_to_local[node_id], torch.tensor(features, dtype=torch.long)] = 1.0
    missing_features = sorted(set(node_to_local) - seen_feature_nodes)
    if missing_features:
        raise ValueError(
            f"Missing Twitch feature rows for node IDs {missing_features[:10]}"
        )

    if sources:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        num_nodes=len(labels),
    )


def _validated_fb100_matrix(path: Path) -> tuple[Any, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise ImportError("facebook100 requires scipy") from error

    raw = loadmat(path)
    if "A" not in raw or "local_info" not in raw:
        raise ValueError(f"FB-100 matrix {path} must contain A and local_info")
    adjacency = raw["A"]
    info = np.asarray(raw["local_info"])
    if info.ndim != 2 or info.shape[1] < 7:
        raise ValueError(
            f"FB-100 local_info in {path} must have at least seven columns"
        )
    if adjacency.ndim != 2 or adjacency.shape != (info.shape[0], info.shape[0]):
        raise ValueError(
            f"FB-100 adjacency in {path} must be square and match local_info rows"
        )
    if not np.issubdtype(info.dtype, np.number) or not np.isfinite(info).all():
        raise ValueError(f"FB-100 local_info in {path} must be finite numeric data")
    if not np.equal(info, np.floor(info)).all():
        raise ValueError(f"FB-100 local_info in {path} must contain integer categories")
    return adjacency, info.astype(np.int64, copy=False)


def _fb100_edge_index(adjacency: Any, path: Path) -> torch.Tensor:
    try:
        from scipy import sparse
    except ImportError as error:
        raise ImportError("facebook100 requires scipy") from error

    if sparse.issparse(adjacency):
        coo = adjacency.tocoo(copy=False)
        if (
            not np.issubdtype(coo.data.dtype, np.number)
            or not np.isfinite(coo.data).all()
        ):
            raise ValueError(
                f"FB-100 adjacency in {path} must be finite numeric data"
            )
        rows = np.asarray(coo.row, dtype=np.int64)
        columns = np.asarray(coo.col, dtype=np.int64)
    else:
        dense = np.asarray(adjacency)
        if not np.issubdtype(dense.dtype, np.number) or not np.isfinite(dense).all():
            raise ValueError(f"FB-100 adjacency in {path} must be finite numeric data")
        rows, columns = np.nonzero(dense)
    if rows.size:
        return torch.from_numpy(np.stack((rows, columns))).long()
    return torch.empty((2, 0), dtype=torch.long)


def _load_fb100_domains(root: Path, selected: Sequence[str]) -> dict[str, Data]:
    # The categorical feature vocabulary is fitted over the complete 18-school
    # benchmark, so every matrix must be present and validated before any
    # selected school is transformed.
    matrices: dict[str, tuple[Any, np.ndarray]] = {}
    for filename in FB100_FILES.values():
        path = root / filename
        _ensure_file(f"{FB100_RAW_URL}/{filename}", path)
    selected_set = set(selected)
    for domain, filename in FB100_FILES.items():
        adjacency, info = _validated_fb100_matrix(root / filename)
        # Only selected domains need their (potentially large) adjacency after
        # schema validation; every school's metadata remains for the vocabulary.
        matrices[domain] = (
            adjacency if domain in selected_set else None,
            info,
        )

    feature_columns = (0, 2, 3, 4, 5, 6)
    categories: list[np.ndarray] = []
    offsets: list[int] = []
    total_features = 0
    for column in feature_columns:
        values = np.unique(
            np.concatenate([info[:, column] for _, info in matrices.values()])
        ).astype(np.int64, copy=False)
        if values.size == 0:
            raise ValueError(f"FB-100 feature column {column} has no categories")
        categories.append(values)
        # Match sklearn.preprocessing.label_binarize, which GraphOOD uses:
        # binary columns occupy one positive-class column; multiclass columns
        # have one column per class (including raw missing-value category 0).
        width = 1 if values.size <= 2 else int(values.size)
        offsets.append(total_features)
        total_features += width

    result: dict[str, Data] = {}
    for domain in selected:
        adjacency, info = matrices[domain]
        x = torch.zeros((info.shape[0], total_features), dtype=torch.float32)
        for column, values, offset in zip(
            feature_columns, categories, offsets
        ):
            raw_values = info[:, column]
            if values.size == 2:
                rows = np.flatnonzero(raw_values == values[1])
                if rows.size:
                    x[torch.from_numpy(rows), offset] = 1.0
            elif values.size > 2:
                encoded = np.searchsorted(values, raw_values)
                rows = np.arange(info.shape[0], dtype=np.int64)
                x[
                    torch.from_numpy(rows),
                    torch.from_numpy(encoded.astype(np.int64, copy=False) + offset),
                ] = 1.0
        # This intentionally matches the selected GraphOOD behavior: missing
        # raw gender 0 maps to class 0 and every positive value maps to class 1.
        y = torch.from_numpy((info[:, 1] > 0).astype(np.int64, copy=False))
        path = root / FB100_FILES[domain]
        result[domain] = Data(
            x=x,
            edge_index=_fb100_edge_index(adjacency, path),
            y=y,
            num_nodes=info.shape[0],
        )
    return result


def _load_mag_domain(root: Path, domain: str) -> Data:
    filename, expected_size, expected_md5 = MAG_ARTIFACTS[domain]
    path = root / filename
    url = f"{MAG_RAW_URL}/{filename}/content"
    _ensure_mag_file(url, path, expected_size, expected_md5)
    # The PyTorch pickle is trusted only after both published integrity checks.
    graph = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graph, Data):
        raise ValueError(f"MAG artifact {path} must contain a PyG Data object")
    if not torch.is_tensor(graph.x) or graph.x.ndim != 2:
        raise ValueError(f"MAG artifact {path} must contain a two-dimensional x")
    if graph.x.size(1) == 0:
        raise ValueError(f"MAG artifact {path} must contain at least one feature")
    if not torch.is_tensor(graph.y):
        raise ValueError(f"MAG artifact {path} must contain tensor labels")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if graph.y.dtype not in integer_dtypes:
        raise ValueError(f"MAG labels in {path} must contain integer class IDs")
    y = graph.y.reshape(-1).to(dtype=torch.long, device="cpu")
    if y.numel() != graph.x.size(0):
        raise ValueError(f"MAG artifact {path} has inconsistent x and y lengths")
    if y.numel() == 0 or bool((y < 0).any()) or bool((y >= 20).any()):
        raise ValueError(f"MAG artifact {path} labels must be in [0, 19]")
    if not torch.is_tensor(graph.edge_index) or graph.edge_index.ndim != 2:
        raise ValueError(f"MAG artifact {path} must contain edge_index")
    if graph.edge_index.size(0) != 2:
        raise ValueError(f"MAG edge_index in {path} must have shape [2, E]")
    if graph.edge_index.dtype not in integer_dtypes:
        raise ValueError(f"MAG edge_index in {path} must contain integer node IDs")
    edge_index = graph.edge_index.to(dtype=torch.long, device="cpu").contiguous()
    if edge_index.numel() and (
        bool((edge_index < 0).any()) or bool((edge_index >= y.numel()).any())
    ):
        raise ValueError(f"MAG edge_index in {path} contains unknown node IDs")
    edge_index = to_undirected(edge_index, num_nodes=y.numel())
    x = graph.x.detach().to(device="cpu")
    if x.layout != torch.strided:
        x = x.to_dense()
    if not (x.is_floating_point() or x.is_complex()):
        x = x.float()
    if x.is_complex() or not bool(torch.isfinite(x).all()):
        raise ValueError(f"MAG features in {path} must be finite real values")
    return Data(x=x.contiguous(), edge_index=edge_index, y=y, num_nodes=y.numel())


def _shared_domain_masks(
    labels: torch.Tensor, domain: str, seed: int, val_ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    val_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    test_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    for label in torch.unique(labels, sorted=True).tolist():
        indices = torch.nonzero(labels == label, as_tuple=False).reshape(-1)
        material = f"{seed}:{domain}:{int(label)}".encode("utf-8")
        generator_seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big"
        ) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(generator_seed)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        val_count = math.floor(val_ratio * indices.numel())
        if indices.numel() >= 2:
            val_count = min(max(val_count, 1), indices.numel() - 1)
        val_mask[indices[:val_count]] = True
        test_mask[indices[val_count:]] = True
    return val_mask, test_mask


def _assemble_domains(
    dataset_name: str,
    normalized: dict[str, Any],
    split_id: str,
    graphs: Mapping[str, Data],
) -> tuple[Data, dict[str, Any]]:
    registry = DOMAIN_REGISTRIES[dataset_name]
    selected_set = {
        domain for role in _ROLE_NAMES for domain in normalized[role]
    }
    selected = [domain for domain in registry if domain in selected_set]
    if set(graphs) != set(selected):
        raise ValueError(
            f"Loaded domains {sorted(graphs)} do not match requested domains "
            f"{sorted(selected)}"
        )

    feature_sizes = {int(graphs[domain].x.size(1)) for domain in selected}
    if len(feature_sizes) != 1:
        raise ValueError(
            f"All {dataset_name} domains must have the same feature dimension; "
            f"found {sorted(feature_sizes)}"
        )
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    domain_ids: list[torch.Tensor] = []
    train_masks: list[torch.Tensor] = []
    val_masks: list[torch.Tensor] = []
    test_masks: list[torch.Tensor] = []
    offset = 0
    train_domains = set(normalized["train"])
    val_domains = set(normalized["val"])
    test_domains = set(normalized["test"])

    for domain_id, domain in enumerate(selected):
        graph = graphs[domain]
        if graph.x.ndim != 2 or graph.y.reshape(-1).numel() != graph.x.size(0):
            raise ValueError(f"Invalid node tensors for {dataset_name} domain {domain}")
        y = graph.y.reshape(-1).long().cpu()
        edge_index = graph.edge_index.long().cpu()
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Invalid edge_index for {dataset_name} domain {domain}")
        if edge_index.numel() and (
            bool((edge_index < 0).any())
            or bool((edge_index >= graph.x.size(0)).any())
        ):
            raise ValueError(f"Unknown edge endpoint in {dataset_name} domain {domain}")
        num_nodes = y.numel()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if domain in train_domains:
            train_mask.fill_(True)
        elif domain in val_domains and domain in test_domains:
            val_mask, test_mask = _shared_domain_masks(
                y, domain, normalized["seed"], normalized["val_ratio"]
            )
        elif domain in val_domains:
            val_mask.fill_(True)
        elif domain in test_domains:
            test_mask.fill_(True)
        else:
            raise ValueError(f"Requested domain {domain} has no split role")

        xs.append(graph.x.cpu())
        ys.append(y)
        edge_indices.append(edge_index + offset)
        domain_ids.append(torch.full((num_nodes,), domain_id, dtype=torch.long))
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
        offset += num_nodes

    data = Data(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        y=torch.cat(ys, dim=0),
        train_mask=torch.cat(train_masks),
        val_mask=torch.cat(val_masks),
        test_mask=torch.cat(test_masks),
        domain_id=torch.cat(domain_ids),
        num_nodes=offset,
    )
    membership = (
        data.train_mask.to(torch.uint8)
        + data.val_mask.to(torch.uint8)
        + data.test_mask.to(torch.uint8)
    )
    if not bool((membership == 1).all()):
        raise ValueError("Domain split masks must be disjoint and exhaustive")
    for role in _ROLE_NAMES:
        if not bool(getattr(data, f"{role}_mask").any()):
            raise ValueError(
                f"Domain split produced an empty {role} mask; choose more target nodes "
                "or a different val_ratio"
            )

    task = DOMAIN_TASKS[dataset_name]
    data.domain_names = list(selected)
    data.domain_split = {
        role: list(normalized[role]) for role in _ROLE_NAMES
    }
    data.domain_split["seed"] = normalized["seed"]
    data.domain_split["val_ratio"] = normalized["val_ratio"]
    data.domain_split_id = split_id
    data.domain_dataset = True
    data.task_type = task["task_type"]
    data.primary_metric = task["primary_metric"]
    if task["metric_ignore_label"] is not None:
        data.metric_ignore_label = task["metric_ignore_label"]

    metadata = {
        "num_features": next(iter(feature_sizes)),
        "num_classes": task["num_classes"],
        "domain_dataset": True,
        "domain_split": data.domain_split,
        "domain_split_id": split_id,
        "task_type": task["task_type"],
        "primary_metric": task["primary_metric"],
        "metric_ignore_label": task["metric_ignore_label"],
    }
    return data, metadata


def load_domain_dataset(
    dataset_name: str,
    domain_split: Mapping[str, Any] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
) -> tuple[Data, dict[str, Any]]:
    """Load a supported domain dataset and return ``(data, metadata)``."""
    name = str(dataset_name).lower()
    normalized, split_id = normalize_domain_split(name, domain_split)
    selected_set = {
        domain for role in _ROLE_NAMES for domain in normalized[role]
    }
    selected = [
        domain for domain in DOMAIN_REGISTRIES[name] if domain in selected_set
    ]

    if name == "twitch-explicit":
        cache = Path(
            root
            if root is not None
            else os.environ.get("GRAPHOOD_TWITCH_DATA_ROOT", "data/graphood/twitch")
        )
        graphs = {domain: _parse_twitch_domain(cache, domain) for domain in selected}
    elif name == "facebook100":
        cache = Path(
            root
            if root is not None
            else os.environ.get(
                "GRAPHOOD_FB100_DATA_ROOT", "data/graphood/facebook100"
            )
        )
        graphs = _load_fb100_domains(cache, selected)
    elif name == "mag-countries":
        cache = Path(
            root
            if root is not None
            else os.environ.get("PAIR_ALIGN_MAG_DATA_ROOT", "data/pair_align_mag")
        )
        graphs = {domain: _load_mag_domain(cache, domain) for domain in selected}
    else:  # normalize_domain_split already rejects this; keep narrowing explicit.
        raise ValueError(f"Unknown domain dataset '{dataset_name}'")
    return _assemble_domains(name, normalized, split_id, graphs)


__all__ = [
    "DOMAIN_DATASET_NAMES",
    "DOMAIN_DEFAULTS",
    "DOMAIN_REGISTRIES",
    "FB100_DOMAINS",
    "FB100_FILES",
    "MAG_ARTIFACTS",
    "MAG_DOMAINS",
    "TWITCH_DOMAINS",
    "load_domain_dataset",
    "normalize_domain_split",
]
