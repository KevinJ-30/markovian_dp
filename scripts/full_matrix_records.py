"""Frozen inputs and evidence-derived reports for the full-matrix campaign.

No training libraries are imported until a preparation/load operation needs them.
The manifest and attempt artifacts, never the mutable queue projection, are the
scientific authority. GPU admission and process ownership live in runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).absolute().parents[1]
MAIN_PYTHON = "/usr/scratch/asaha92/envs/graph_subsampling/bin/python"
PROGAP_PYTHON = "/usr/scratch/asaha92/envs/progap/bin/python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.full_matrix_runtime import (
    DEFAULT_POLICY, atomic_json, file_lock, json_hash, read_json, sha256, utc_now,
)

PROTOCOLS = (
    "ogbn-arxiv", "ogbn-products", "saint-reddit", "saint-yelp", "saint-amazon",
    "twitch-allbut2", "facebook100-allbut2", "mag-allbut2",
)
METHODS = (
    "mlp", "graphsage", "gin", "dp_mlp", "progap", "dpar", "dp_gnn_sage",
    "dp_gnn_gin", "sparse_sage", "sparse_gin",
)
NONPRIVATE = frozenset(("mlp", "graphsage", "gin"))
BOOTSTRAP = {"method": "percentile", "confidence_level": 0.95,
             "n_resamples": 1000, "seed": 0, "resampling_unit": "node"}
BINDING_FIELDS = frozenset(("campaign_manifest_path", "campaign_manifest_sha256", "attempt_number"))
KEY_EXCLUDED = frozenset(("request_key", "ordinal", "argv", "prepared_manifest", "prepared_manifest_sha256"))
INTERPRETATION = {
    "configuration_selection": "highest FINAL TEST score; TEST-selection bias applies",
    "checkpoint_selection": "first strict validation-primary-metric maximum; no early stopping",
    "uncertainty": "node bootstrap conditional on fixed predictions, not seed variance or a test-selection correction",
    "privacy": "per-run epsilon is not a composed privacy guarantee for the sweep or retries",
    "gpu_ownership": "flock and NVIDIA idle observations are advisory, not atomic reservations against unrelated users",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str) -> float:
    _require(not isinstance(value, bool), f"{name}: boolean is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: expected finite number") from error
    _require(math.isfinite(result), f"{name}: expected finite number")
    return result


def _same(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, float):
        _require(math.isclose(_finite(actual, name), expected, rel_tol=1e-10, abs_tol=1e-12),
                 f"{name}: {actual!r} != {expected!r}")
    else:
        _require(actual == expected, f"{name}: {actual!r} != {expected!r}")


def _owned(path: Path, root: Path) -> Path:
    path, root = Path(path).absolute(), Path(root).absolute()
    _require(path.is_relative_to(root), f"artifact outside owning root: {path}")
    _require(path.resolve().is_relative_to(root.resolve()), f"artifact symlink escapes root: {path}")
    _require(not any(parent.is_symlink() for parent in (path, *path.parents)
                     if parent.is_relative_to(root)), f"symlink is not immutable evidence: {path}")
    return path


def _request_key(request: dict) -> str:
    return json_hash({key: value for key, value in request.items()
                      if key not in KEY_EXCLUDED and key not in BINDING_FIELDS})


def _grid_identity(row: dict) -> tuple:
    return tuple(row[key] for key in ("protocol", "method", "lr", "epsilon", "p2"))


def _validate_grid(rows: list[dict], batch_size: int = 1024) -> None:
    _same(len(rows), 336, "campaign request count")
    _same(Counter(row["protocol"] for row in rows), Counter({p: 42 for p in PROTOCOLS}),
          "protocol counts")
    _same(len({_grid_identity(row) for row in rows}), 336, "unique configurations")
    for row in rows:
        method = row["method"]
        _require(method in METHODS, f"unexpected method {method}")
        for key, value in {"batch_size": batch_size, "epochs": 20, "seed": 0,
                           "dropout": 0.5, "mlp_hidden": 64, "gnn_hidden": 128}.items():
            _same(row[key], value, key)
        _require(row["lr"] in (0.01, 0.001), "unexpected learning rate")
        _require(row["epsilon"] is None if method in NONPRIVATE else row["epsilon"] in (2, 8),
                 "unexpected privacy dimension")
        _require(row["p2"] in (0.5, 0.1) if method.startswith("sparse_") else row["p2"] is None,
                 "unexpected p2 dimension")
    expected = {method: (16 if method in NONPRIVATE else 64 if method.startswith("sparse_") else 32)
                for method in METHODS}
    _same(Counter(row["method"] for row in rows), Counter(expected), "method counts")


def enumerate_grid(batch_size: int = 1024) -> list[dict]:
    """Use the maintained shell as the sole Cartesian-product expander; no writes."""
    from scripts.full_matrix_run import parser

    _require(type(batch_size) is int and batch_size in (256, 1024), "unsupported grid batch size")
    placeholder = REPO_ROOT / "__full_matrix_dry_run_only__"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHON": MAIN_PYTHON,
                   "PROGAP_PYTHON": PROGAP_PYTHON, "DEVICE": "cuda", "OUT_ROOT": str(placeholder),
                   "DATASETS": " ".join(PROTOCOLS), "METHODS": " ".join(METHODS),
                   "EPSILONS": "2 8", "SEEDS": "0", "LEARNING_RATES": "0.01 0.001",
                   "BATCH_SIZES": str(batch_size), "EPOCHS": "20", "P2_VALUES": "0.5 0.1",
                   "BOOTSTRAP_RESAMPLES": "1000"}
    expanded = subprocess.run(["bash", str(REPO_ROOT / "scripts/full_matrix.sh"), "--dry-run"],
                              cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
                              check=True, timeout=30)
    rows, summaries, counts = [], 0, 0
    for line in expanded.stdout.splitlines():
        if line == "Dry run: 336 training runs; no experiments executed.":
            counts += 1
            continue
        command = shlex.split(line)
        if command == [MAIN_PYTHON, str(REPO_ROOT / "scripts/summarize_results.py"),
                       str(placeholder / "runs/**/result.csv"), "--bootstrap", "--best", "--out",
                       str(placeholder / "summary")]:
            summaries += 1
            continue
        _require(len(command) > 2 and command[:2] ==
                 [MAIN_PYTHON, str(REPO_ROOT / "scripts/full_matrix_run.py")],
                 f"unexpected shell dry-run output: {line!r}")
        args = parser().parse_args(command[2:])
        _same(args.device, "cuda", "grid device")
        _same(args.bootstrap_resamples, 1000, "bootstrap resamples")
        _same(args.progap_python, PROGAP_PYTHON if args.method == "progap" else None,
              "ProGAP interpreter")
        rows.append({"protocol": args.dataset, "method": args.method, "lr": args.lr,
                     "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed,
                     "dropout": args.dropout, "mlp_hidden": args.mlp_hidden,
                     "gnn_hidden": args.gnn_hidden, "epsilon": args.epsilon, "p2": args.p2,
                     "argv": command})
        rows[-1]["dataset"] = args.dataset
    _same((summaries, counts), (1, 1), "shell footer")
    _validate_grid(rows, batch_size)
    return rows


def source_hashes() -> dict[str, str]:
    paths = [REPO_ROOT / "scripts" / name for name in (
        "full_matrix_campaign.py", "full_matrix_records.py", "full_matrix_runtime.py",
        "full_matrix_run.py", "full_matrix.sh", "summarize_results.py")]
    paths += list((REPO_ROOT / "src").rglob("*.py"))
    paths += list((REPO_ROOT / "third_party/ProGAP").rglob("*.py"))
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sorted(paths)}


def check_sources(root: Path, manifest: dict | None = None) -> None:
    root = Path(root).absolute()
    manifest = manifest or read_json(root / "campaign_manifest.json")
    hashes = source_hashes()
    _same(hashes, manifest["source_sha256"], "source identity changed")
    _same(json_hash(hashes), manifest["source_fingerprint"], "source fingerprint")
    for relative, digest in hashes.items():
        snapshot = _owned(root / "source_snapshot" / relative, root / "source_snapshot")
        _same(sha256(snapshot), digest, f"source snapshot {relative}")


_PACKAGE_PROBE = '''import importlib, importlib.metadata as m, json, os, platform, sys
names = ["torch", "torch-geometric", "pyg-lib", "numpy", "scipy", "opacus", "dp-accounting", "scikit-learn", "ogb", "pytest"]
versions = {name: m.version(name) for name in names}
for name in ["torch", "torch_geometric", "pyg_lib", "numpy", "scipy", "opacus", "dp_accounting", "sklearn", "ogb"]:
    importlib.import_module(name)
import torch
installed = sorted((d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"])
print(json.dumps({"executable": os.path.abspath(sys.executable), "python": platform.python_version(), "implementation": platform.python_implementation(), "required": versions, "installed": installed, "torch_cuda_runtime": torch.version.cuda}, sort_keys=True))
'''


def package_versions() -> dict:
    result = {}
    for label, executable in (("main", MAIN_PYTHON), ("progap", PROGAP_PYTHON)):
        _require(Path(executable).is_file(), f"missing {label} interpreter: {executable}")
        process = subprocess.run([executable, "-c", _PACKAGE_PROBE], cwd=REPO_ROOT,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
                                      "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                                      "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                 capture_output=True, text=True, timeout=120)
        _require(process.returncode == 0, f"{label} dependency preflight failed: {process.stderr}")
        result[label] = json.loads(process.stdout.strip().splitlines()[-1])
        _same(result[label]["executable"], executable, f"{label} interpreter identity")
    return result


def _cache_identity(protocol: str) -> dict:
    variable, default = (
        ("OGB_DATA_ROOT", f"data/{protocol}") if protocol.startswith("ogbn-") else
        ("GRAPHSAINT_DATA_ROOT", "data/graphsaint") if protocol.startswith("saint-") else
        ("GRAPHOOD_TWITCH_DATA_ROOT", "data/graphood/twitch") if protocol == "twitch-allbut2" else
        ("GRAPHOOD_FB100_DATA_ROOT", "data/graphood/facebook100") if protocol == "facebook100-allbut2" else
        ("PAIR_ALIGN_MAG_DATA_ROOT", "data/pair_align_mag"))
    path = Path(os.environ.get(variable, default)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    _require(path.is_dir(), f"dataset cache missing: {variable}={path}")
    return {"environment_variable": variable, "environment_value": os.environ.get(variable),
            "resolved_path": str(path.resolve())}


def _validate_split(split: Any, task: dict, strategy: str) -> int:
    import torch

    _require(int(split.train.data.num_nodes) > 1, "N_train must exceed one")
    _require(int(split.num_classes) > 0, "invalid shared task head")
    _require(sum(bool(task[name]) for name in ("binary", "multilabel", "regression")) <= 1,
             "task modes conflict")
    _same(split.primary_metric, task["primary_metric"], "split primary metric")
    _same(split.binary, task["binary"], "split binary task")
    tensor_bytes, identifiers = 0, []
    for name in ("train", "val", "test"):
        part = getattr(split, name)
        data, count = part.data, int(part.data.num_nodes)
        mask = part.eval_mask
        _require(count > 0 and mask.dtype == torch.bool and tuple(mask.shape) == (count,),
                 f"invalid {name} scoring mask")
        _require(part.node_ids.numel() == count and torch.unique(part.node_ids).numel() == count,
                 f"invalid {name} node IDs")
        identifiers.append(part.node_ids.cpu())
        _require(data.x.ndim == 2 and data.x.shape[0] == count and data.y.shape[0] == count,
                 f"invalid {name} feature/label population")
        _require(data.edge_index.ndim == 2 and data.edge_index.shape[0] == 2,
                 f"invalid {name} edge index")
        if data.edge_index.numel():
            _require(int(data.edge_index.min()) >= 0 and int(data.edge_index.max()) < count,
                     f"{name} edge escapes graph partition")
        labels = data.y[mask]
        if task["metric_ignore_label"] is not None and labels.ndim == 1:
            labels = labels[labels != task["metric_ignore_label"]]
        _require(labels.shape[0] > 0, f"empty scored {name} partition")
        if task["binary"]:
            _same(int(split.num_classes), 2, "binary shared class head")
            _require(bool(((data.y == 0) | (data.y == 1)).all()), "invalid binary labels")
            if name != "train":
                _same(torch.unique(labels).numel(), 2, f"{name} AUROC requires both classes")
        elif task["multilabel"]:
            _require(data.y.ndim == 2 and data.y.shape[1] == split.num_classes,
                     "multilabel head mismatch")
            _require(bool(((data.y == 0) | (data.y == 1)).all()), "invalid multilabel labels")
        elif not task["regression"]:
            _require(data.y.ndim == 1 and int(data.y.min()) >= 0 and
                     int(data.y.max()) < split.num_classes, "categorical head mismatch")
        tensor_bytes += sum(value.numel() * value.element_size()
                            for key, value in data if isinstance(value, torch.Tensor) and key != "eval_mask")
        tensor_bytes += part.eval_mask.numel() * part.eval_mask.element_size()
        tensor_bytes += part.node_ids.numel() * part.node_ids.element_size()
    # Native and explicit domain splits both assign disjoint global IDs.
    combined = torch.cat(identifiers)
    _same(torch.unique(combined).numel(), combined.numel(), "partition ID overlap")
    if strategy == "domain":
        roles = split.domain_split
        _require(isinstance(roles, dict), "domain split roles are missing")
        domains = [domain for role in ("train", "val", "test") for domain in roles[role]]
        _same(len(domains), len(set(domains)), "domain-role overlap")
    for mask in split.masks.values():
        tensor_bytes += mask.numel() * mask.element_size()
    return int(tensor_bytes)


def _prepared_fingerprint(manifest: dict) -> str:
    return json_hash({key: manifest[key] for key in (
        "artifact_sha256", "protocol", "dataset", "task", "split_strategy", "split_seed",
        "original_population", "num_classes", "preprocessing", "domain_split", "domain_split_id",
        "scored_populations")})


def _prepared_row(path: Path, manifest: dict | None = None) -> dict:
    path = Path(path).absolute()
    manifest = manifest or read_json(path)
    return {"manifest": str(path), "manifest_sha256": sha256(path),
            "fingerprint": manifest["fingerprint"], "n_train": manifest["original_population"],
            "dataset": manifest["dataset"], "task": manifest["task"],
            "tensor_bytes": manifest["tensor_bytes"],
            "preflight_rss_bytes": manifest["preflight_rss_bytes"]}


def _publish_prepared(protocol: str, root: Path, dataset: str, split: Any, task: dict,
                      strategy: str, *, started: float, cache: dict | None) -> dict:
    import torch
    from src.experiments.upstream import export_partitions

    root = Path(root).absolute()
    _require("/" not in protocol and protocol not in (".", ".."), "invalid protocol path")
    destination = root / "prepared" / protocol
    _require(not destination.exists(), f"prepared protocol already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_bytes = _validate_split(split, task, strategy)
    staging = Path(tempfile.mkdtemp(prefix=f".{protocol}.", dir=destination.parent))
    exported = export_partitions(split, staging)
    metadata = {field.name: getattr(split, field.name) for field in fields(split)
                if field.name not in ("train", "val", "test")}
    metadata["path"] = str(metadata["path"])
    metadata["masks"] = {name: value.detach().cpu() for name, value in metadata["masks"].items()}
    torch.save(metadata, staging / "split_metadata.pt")
    manifest = read_json(exported)
    original = Path(split.path)
    original_hash = sha256(original) if original.is_file() else None
    _require(cache is None or original_hash is not None, f"split cache is absent: {original}")
    manifest.update(
        protocol=protocol, dataset=dataset, task=task, split_strategy=strategy, split_seed=0,
        original_split_cache={"path": str(original.absolute()), "sha256": original_hash},
        dataset_cache=cache, original_population=int(split.train.data.num_nodes),
        tensor_bytes=tensor_bytes, preflight_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        preparation_seconds=time.monotonic() - started, prepared_utc=utc_now(),
        split_metadata="split_metadata.pt", preprocessing="worker._load_split/preprocess_inductive_split:once" if cache else "deterministic_fixture:no_preprocessing",
        ignored_training_label=task["metric_ignore_label"],
        artifact_sha256={name: sha256(staging / name)
                         for name in ("train.pt", "val.pt", "test.pt", "split_metadata.pt")},
    )
    manifest["scored_populations"] = {}
    for name in ("train", "val", "test"):
        part = getattr(split, name)
        labels = part.data.y[part.eval_mask]
        if task["metric_ignore_label"] is not None and labels.ndim == 1:
            labels = labels[labels != task["metric_ignore_label"]]
        manifest["scored_populations"][name] = int(labels.shape[0])
    manifest["fingerprint"] = _prepared_fingerprint(manifest)
    atomic_json(exported, manifest)
    # Hash-check before publication; deserialization uses only the published copy.
    _verify_prepared(exported)
    staging.rename(destination)
    return _prepared_row(destination / "manifest.json", manifest)


def prepare_protocol(protocol: str, root: Path) -> dict:
    """One CPU child, one raw load, one immutable format-2 export."""
    _require(protocol in PROTOCOLS, f"unknown campaign protocol: {protocol}")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "preparation requires explicitly hidden CUDA")
    from scripts.full_matrix_run import _load_split

    started = time.monotonic()
    cache = _cache_identity(protocol)
    dataset, split, task, strategy = _load_split(protocol, Path(root) / "prepared/split_cache")
    return _publish_prepared(protocol, Path(root), dataset, split, task, strategy,
                             started=started, cache=cache)


def _verify_prepared(path: Path) -> dict:
    path = Path(path).absolute()
    if path.is_dir():
        path /= "manifest.json"
    manifest = read_json(path)
    _same(manifest["format"], 2, "prepared format")
    _same(manifest["partitions"], {name: f"{name}.pt" for name in ("train", "val", "test")},
          "prepared partition paths")
    _same(manifest["split_metadata"], "split_metadata.pt", "split metadata path")
    _same(set(manifest["artifact_sha256"]), {"train.pt", "val.pt", "test.pt", "split_metadata.pt"},
          "prepared artifact inventory")
    for relative, digest in manifest["artifact_sha256"].items():
        _same(sha256(_owned(path.parent / relative, path.parent)), digest, f"prepared artifact {relative}")
    _same(_prepared_fingerprint(manifest), manifest["fingerprint"], "prepared fingerprint")
    return manifest


def load_prepared_protocol(path: Path) -> tuple:
    """Hash every owned payload before torch.load; never invoke raw loaders."""
    import torch
    from src.processing.splits import GraphPartition, InductiveSplit

    path = Path(path).absolute()
    if path.is_dir():
        path /= "manifest.json"
    manifest = _verify_prepared(path)
    metadata = torch.load(path.parent / manifest["split_metadata"], map_location="cpu", weights_only=False)
    metadata["path"] = Path(metadata["path"])
    parts = {}
    for name in ("train", "val", "test"):
        payload = torch.load(path.parent / manifest["partitions"][name], map_location="cpu", weights_only=False)
        data = payload["data"]
        parts[name] = GraphPartition(data, payload["node_ids"], payload["statistics"], data.eval_mask)
    split = InductiveSplit(**parts, **metadata)
    _same(split.num_classes, manifest["num_classes"], "prepared shared class head")
    _same(int(split.train.data.num_nodes), manifest["original_population"], "prepared original population")
    _same(_validate_split(split, manifest["task"], manifest["split_strategy"]), manifest["tensor_bytes"],
          "prepared tensor bytes")
    return manifest["dataset"], split, manifest["task"], manifest["split_strategy"]


def _expected_parameters(row: dict, n: int) -> dict:
    method, epochs = row["method"], row["epochs"]
    batch = min(row["batch_size"], n)
    interval = math.ceil(n / batch)
    common = {"optimizer": "adam", "weight_decay": 0.0 if method == "progap" else 5e-4,
              "dropout": row["dropout"], "batch_size": batch, "epochs": epochs,
              "bootstrap_confidence": 0.95, "bootstrap_resamples": 1000, "bootstrap_seed": 0}
    if method in NONPRIVATE or method == "dp_mlp":
        common.update(hidden_size=row["mlp_hidden"] if method in ("mlp", "dp_mlp") else row["gnn_hidden"],
                      layers=2, learning_rate=row["lr"], steps=epochs * interval, evaluate_every=interval)
        if method in ("graphsage", "gin"):
            common.update(graphsage_sampling="hierarchical", max_fanout=10)
        if method == "dp_mlp":
            common.update(clip=1.0, delta=1.0 / n)
    elif method == "dpar":
        common.update(hidden_size=row["gnn_hidden"], layers=2, learning_rate=row["lr"],
                      ppr_num=70, sampled_train_rate=0.09, sampled_train_nodes=None,
                      alpha=0.25, rho=1e-4, ista_epsilon=1e-4, topk=16, ppr_clip=0.01,
                      sgd_clip=1.0, inference_steps=2, target_delta=1.0 / n,
                      target_epsilon=row["epsilon"], dp_ppr=True, dp_sgd=True)
    elif method == "progap":
        common.update(hidden_dim=row["gnn_hidden"], learning_rate=row["lr"], depth=2, stages=3,
                      epochs_total=3 * epochs, max_degree=5, max_grad_norm=1.0,
                      steps=3 * epochs * (n // batch), evaluate_every=n // batch,
                      accounted_sgd_steps_per_stage=epochs * n // batch, base_layers=1, head_layers=1,
                      activation="selu", jk="cat", batch_norm=True, layerwise=False,
                      eval_chunk_size=16384, target_delta=1.0 / n, target_epsilon=row["epsilon"],
                      normalization="upstream_ModuleValidator.fix", epoch_schedule="native_drop_last_per_stage")
    elif method.startswith("dp_gnn_"):
        common.update(latent_size=row["gnn_hidden"], learning_rate=row["lr"], max_degree=5,
                      clip=1.0, max_subgraph_nodes=100, max_private_batch_nodes=8192,
                      architecture="gin" if method.endswith("gin") else "graphsage", delta=1.0 / n,
                      steps=epochs * interval, evaluate_every=interval, max_terms=min(6, n))
    else:
        common.update(hidden=row["gnn_hidden"], layers=2, lr=row["lr"],
                      architecture="gin" if method.endswith("gin") else "mean",
                      p1=batch / n, p2=row["p2"], r=1, clip=1.0, K_in=10, K_out=10,
                      cap_mode="directed", cap_seed=20000, direction="in", chi=1, union_safe=False,
                      accounting_grid=1e-3, calibration_rtol=1e-3, calibration_atol=1e-6,
                      max_private_batch_nodes=8192, steps=epochs * interval, evaluate_every=interval)
    return common


def _make_request(row: dict, prepared: dict, source_fingerprint: str, ordinal: int) -> dict:
    n = int(prepared["n_train"])
    _require(n > 1, "N_train must exceed one")
    method = row["method"]
    request = {**row, "ordinal": ordinal, "dataset": prepared["dataset"],
               "hidden": row["mlp_hidden"] if method in ("mlp", "dp_mlp") else row["gnn_hidden"],
               "weight_decay": 0.0 if method == "progap" else 5e-4,
               "delta": None if method in NONPRIVATE else 1.0 / n, "n_train": n,
               "prepared_manifest": prepared["manifest"],
               "prepared_manifest_sha256": prepared["manifest_sha256"],
               "prepared_fingerprint": prepared["fingerprint"], "source_fingerprint": source_fingerprint,
               "task": prepared["task"], "expected_parameters": _expected_parameters(row, n),
               "bootstrap": BOOTSTRAP, "split_seed": 0,
               "epoch_semantics": "per_stage_native_drop_last" if method == "progap" else
               "released_ppr_root_pass" if method == "dpar" else "training_population_expected_pass"}
    request["request_key"] = _request_key(request)
    return request


def build_requests(prepared: dict) -> list[dict]:
    _same(set(prepared), set(PROTOCOLS), "prepared campaign protocols")
    hashes = {row.get("source_fingerprint") for row in prepared.values()} - {None}
    _require(len(hashes) <= 1, "mixed preparation source fingerprints")
    fingerprint = next(iter(hashes)) if hashes else json_hash(source_hashes())
    grouped = defaultdict(list)
    for row in enumerate_grid():
        grouped[row["protocol"]].append(row)
    # Preserve the shell's order within each protocol while interleaving datasets.
    ordered = [grouped[protocol][index] for index in range(42) for protocol in PROTOCOLS]
    requests = [_make_request(row, prepared[row["protocol"]], fingerprint, ordinal)
                for ordinal, row in enumerate(ordered)]
    _validate_grid(requests)
    _same(len({request["request_key"] for request in requests}), 336, "unique request keys")
    return requests


def _smoke_prepared(root: Path, device: str) -> dict:
    import torch
    from torch_geometric.data import Data
    from src.processing.splits import GraphPartition, InductiveSplit, graph_statistics

    tasks = [("smoke-accuracy", False, False, "accuracy")]
    if device == "cpu":
        tasks += [("smoke-binary", True, False, "auroc"), ("smoke-multilabel", False, True, "micro_f1")]
    result = {}
    x = torch.arange(160 * 8, dtype=torch.float32).reshape(160, 8) / (160 * 8)
    labels = torch.arange(160) % 2
    for protocol, binary, multilabel, metric in tasks:
        started = time.monotonic()
        y = torch.stack([labels, 1 - labels], dim=1) if multilabel else labels
        parts, masks = {}, {}
        for name, lo, hi in (("train", 0, 128), ("val", 128, 144), ("test", 144, 160)):
            count = hi - lo
            source = torch.arange(count)
            target = (source + 1) % count
            edges = torch.stack([torch.cat([source, target]), torch.cat([target, source])])
            data = Data(x=x[lo:hi].clone(), y=y[lo:hi].clone(), edge_index=edges, num_nodes=count,
                        eval_mask=torch.ones(count, dtype=torch.bool))
            parts[name] = GraphPartition(data, torch.arange(lo, hi), graph_statistics(data), data.eval_mask)
            masks[name] = torch.zeros(160, dtype=torch.bool)
            masks[name][lo:hi] = True
        task = {"binary": binary, "multilabel": multilabel, "regression": False,
                "metric_ignore_label": None, "primary_metric": metric}
        split = InductiveSplit(**parts, masks=masks, num_classes=2,
                               path=root / "prepared" / protocol / "fixture_split", primary_metric=metric,
                               binary=binary)
        result[protocol] = _publish_prepared(protocol, root, protocol, split, task, "fixture",
                                            started=started, cache=None)
    return result


def _smoke_requests(prepared: dict, fingerprint: str) -> list[dict]:
    requests = []
    for protocol, part in prepared.items():
        methods = METHODS if protocol == "smoke-accuracy" else (
            "sparse_sage", "sparse_gin", "dp_gnn_sage", "dp_gnn_gin", "progap")
        for method in methods:
            row = {"protocol": protocol, "method": method, "lr": 0.001, "batch_size": 32,
                   "epochs": 1, "seed": 0, "dropout": 0.5, "mlp_hidden": 64, "gnn_hidden": 128,
                   "epsilon": None if method in NONPRIVATE else 8.0,
                   "p2": 0.5 if method.startswith("sparse_") else None}
            command = [MAIN_PYTHON, str(REPO_ROOT / "scripts/full_matrix_run.py")]
            for key in ("protocol", "method", "lr", "batch_size", "epochs", "seed", "dropout",
                        "mlp_hidden", "gnn_hidden", "epsilon", "p2"):
                if row[key] is not None:
                    command += ["--dataset" if key == "protocol" else "--" + key.replace("_", "-"), str(row[key])]
            command += ["--device", "cuda", "--out-dir", str(REPO_ROOT / "__smoke_only__"),
                        "--bootstrap-resamples", "1000"]
            if method == "progap":
                command += ["--progap-python", PROGAP_PYTHON]
            row["argv"] = command
            requests.append(_make_request(row, part, fingerprint, len(requests)))
    return requests


def _comparison_slots(requests: list[dict]) -> list[dict]:
    grouped = {}
    for request in requests:
        key = (request["protocol"], request["method"], request["epsilon"])
        if key not in grouped:
            grouped[key] = {"protocol": key[0], "method": key[1], "target_epsilon": key[2],
                            "privacy": "non-private" if key[2] is None else "private", "request_keys": []}
        grouped[key]["request_keys"].append(request["request_key"])
    return list(grouped.values())


def prepare_campaign(root: Path, *, purpose: str = "campaign", device: str = "cuda") -> dict:
    _require(purpose in ("campaign", "smoke"), "unknown manifest purpose")
    _require(device in ("cpu", "cuda"), "unknown device")
    _require(purpose != "campaign" or device == "cuda", "production campaign requires CUDA")
    root = Path(root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    with file_lock(root / "campaign.lock", nonblocking=True):
        hashes = source_hashes()
        fingerprint = json_hash(hashes)
        for relative, digest in hashes.items():
            destination = root / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, destination)
            _same(sha256(destination), digest, f"source changed during snapshot: {relative}")
        packages = package_versions()
        if purpose == "campaign":
            prepared = {}
            for protocol in PROTOCOLS:
                _same(source_hashes(), hashes, "source changed before preparation")
                log = root / "preparation_logs" / f"{protocol}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [MAIN_PYTHON, str(REPO_ROOT / "scripts/full_matrix_campaign.py"),
                           "_prepare-protocol", "--out-root", str(root), "--dataset", protocol]
                with log.open("x") as stream:
                    process = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT),
                                                  "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                                  "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
                _require(process.returncode == 0, f"preparation failed for {protocol}; inspect {log}")
                prepared[protocol] = _prepared_row(root / "prepared" / protocol / "manifest.json")
                _verify_prepared(Path(prepared[protocol]["manifest"]))
                _same(source_hashes(), hashes, "source changed during preparation")
            requests = build_requests({p: {**row, "source_fingerprint": fingerprint} for p, row in prepared.items()})
        else:
            prepared = _smoke_prepared(root, device)
            requests = _smoke_requests(prepared, fingerprint)
        _same(source_hashes(), hashes, "source changed during preparation")
        _same(package_versions(), packages, "package identity changed during preparation")
        expected = 336 if purpose == "campaign" else 20 if device == "cpu" else 10
        _same(len(requests), expected, "purpose-specific request count")
        atomic_json(root / "requests.json", requests)
        atomic_json(root / "execution_policy.json", dict(DEFAULT_POLICY))
        manifest = {"format": 1, "purpose": purpose, "device": device, "created_utc": utc_now(),
                    "expected_requests": expected, "expected_comparisons": len(_comparison_slots(requests)),
                    "protocols": list(prepared), "requests_sha256": sha256(root / "requests.json"),
                    "execution_policy_sha256": sha256(root / "execution_policy.json"),
                    "source_sha256": hashes, "source_fingerprint": fingerprint,
                    "package_versions": packages, "prepared": prepared, "interpretation": INTERPRETATION}
        if purpose == "campaign":
            _same(manifest["expected_comparisons"], 136, "comparison registry")
        # Published last: a partial preparation cannot be mistaken for a sealed campaign.
        atomic_json(root / "campaign_manifest.json", manifest)
    return manifest


def load_campaign(root: Path, *, check_sources: bool = True, check_prepared: bool = False) -> tuple:
    root = Path(root).expanduser().absolute()
    manifest = read_json(_owned(root / "campaign_manifest.json", root))
    _same(manifest["format"], 1, "campaign format")
    _require(manifest["purpose"] in ("campaign", "smoke"), "unknown manifest purpose")
    _require(manifest["device"] in ("cpu", "cuda"), "unknown manifest device")
    expected = 336 if manifest["purpose"] == "campaign" else 20 if manifest["device"] == "cpu" else 10
    _same(manifest["expected_requests"], expected, "purpose-specific request count")
    _same(sha256(root / "requests.json"), manifest["requests_sha256"], "requests hash")
    _same(sha256(root / "execution_policy.json"), manifest["execution_policy_sha256"], "execution policy hash")
    _same(read_json(root / "execution_policy.json"), dict(DEFAULT_POLICY), "sealed execution policy")
    requests = read_json(root / "requests.json")
    _same(len(requests), expected, "request registry size")
    _same(len({row["request_key"] for row in requests}), expected, "unique request keys")
    _same([row["ordinal"] for row in requests], list(range(expected)), "request ordinals")
    _same(len(_comparison_slots(requests)), manifest["expected_comparisons"], "comparison registry size")
    if manifest["purpose"] == "campaign":
        _validate_grid(requests)
        _same(manifest["protocols"], list(PROTOCOLS), "campaign protocols")
        _same(manifest["expected_comparisons"], 136, "campaign comparison cells")
        _same(manifest["device"], "cuda", "campaign device")
    else:
        _same(manifest["protocols"], ["smoke-accuracy", "smoke-binary", "smoke-multilabel"]
              if manifest["device"] == "cpu" else ["smoke-accuracy"], "smoke protocol registry")
        _same(requests, _smoke_requests(manifest["prepared"], manifest["source_fingerprint"]),
              "smoke fixture registry")
    _same(set(manifest["prepared"]), set(manifest["protocols"]), "prepared protocol registry")
    _same(json_hash(manifest["source_sha256"]), manifest["source_fingerprint"], "source fingerprint")
    for request in requests:
        _same(_request_key(request), request["request_key"], "scientific request key")
        _same(request["source_fingerprint"], manifest["source_fingerprint"], "request source")
        part = manifest["prepared"][request["protocol"]]
        for key, prepared_key in (("prepared_manifest", "manifest"), ("prepared_manifest_sha256", "manifest_sha256"),
                                  ("prepared_fingerprint", "fingerprint"), ("n_train", "n_train"), ("task", "task"),
                                  ("dataset", "dataset")):
            _same(request[key], part[prepared_key], f"request {key}")
        _owned(Path(request["prepared_manifest"]), root / "prepared")
        _same(request["delta"], None if request["method"] in NONPRIVATE else 1.0 / request["n_train"], "request delta")
        _same(request["weight_decay"], 0.0 if request["method"] == "progap" else 5e-4, "request decay")
        _same(request["expected_parameters"], _expected_parameters(request, request["n_train"]), "expected parameters")
        _same(request["bootstrap"], BOOTSTRAP, "request bootstrap")
    for protocol, row in manifest["prepared"].items():
        path = _owned(Path(row["manifest"]), root / "prepared")
        _same(sha256(path), row["manifest_sha256"], f"prepared manifest {protocol}")
        metadata = _verify_prepared(path) if check_prepared else read_json(path)
        _same(_prepared_row(path, metadata), row, f"prepared metadata {protocol}")
    if check_sources:
        globals()["check_sources"](root, manifest)
    if check_prepared:
        _same(package_versions(), manifest["package_versions"], "installed package identity")
    return manifest, requests


def validate_worker_request(args: argparse.Namespace) -> dict:
    """Validate immutable membership, every argv field and data hashes before loading."""
    request_path = Path(args.campaign_request).absolute()
    binding = read_json(request_path)
    manifest_path = Path(binding["campaign_manifest_path"]).absolute()
    root = manifest_path.parent
    _same(manifest_path.name, "campaign_manifest.json", "campaign manifest filename")
    _same(sha256(manifest_path), binding["campaign_manifest_sha256"], "bound campaign manifest")
    manifest, requests = load_campaign(root, check_sources=True, check_prepared=False)
    registered = next((row for row in requests if row["request_key"] == binding["request_key"]), None)
    _require(registered is not None, "request is not registered")
    _same({key: value for key, value in binding.items() if key not in BINDING_FIELDS}, registered,
          "bound immutable request")
    number = binding["attempt_number"]
    _require(isinstance(number, int) and not isinstance(number, bool) and number >= 1, "invalid attempt number")
    attempt = root / "attempts" / binding["request_key"] / f"attempt_{number}"
    _same(request_path, _owned(attempt / "request.json", root), "attempt request path")
    _same(Path(args.out_dir).absolute(), attempt / "output", "worker output path")
    _same(Path(args.prepared_protocol).absolute(), Path(registered["prepared_manifest"]), "prepared worker argument")
    for key in ("method", "lr", "batch_size", "epochs", "seed", "dropout", "mlp_hidden", "gnn_hidden", "epsilon", "p2"):
        _same(getattr(args, key), registered[key], f"worker argument {key}")
    _same(args.dataset, registered["protocol"], "worker protocol")
    _same(args.bootstrap_resamples, 1000, "worker bootstrap")
    _same(str(args.device).split(":")[0], manifest["device"], "worker device")
    _same(getattr(args, "progap_python", None), PROGAP_PYTHON if registered["method"] == "progap" else None,
          "worker ProGAP interpreter")
    prepared = read_json(Path(registered["prepared_manifest"]))
    _same(_prepared_fingerprint(prepared), registered["prepared_fingerprint"], "prepared request fingerprint")
    for field, name in (("protocol", "protocol"), ("task", "task"), ("fingerprint", "prepared_fingerprint"),
                        ("original_population", "n_train")):
        _same(prepared[field], registered[name], f"prepared request {field}")
    return binding


def worker_command(request: dict, attempt_dir: Path, *, device: str = "cuda") -> list[str]:
    """Keep the frozen shell argv; replace runtime paths/device only."""
    command = list(request["argv"])
    _same(command[:2], [MAIN_PYTHON, str(REPO_ROOT / "scripts/full_matrix_run.py")], "worker executable")
    _require(device in ("cpu", "cuda"), "worker device must be explicit cpu or cuda")
    for option, value in (("--out-dir", str(Path(attempt_dir).absolute() / "output")), ("--device", device)):
        _same(command.count(option), 1, f"frozen {option} count")
        command[command.index(option) + 1] = value
    _require("--campaign-request" not in command and "--prepared-protocol" not in command,
             "frozen argv already contains attempt paths")
    command += ["--prepared-protocol", request["prepared_manifest"],
                "--campaign-request", str(Path(attempt_dir).absolute() / "request.json")]
    return command


def _verify_selection(request: dict, result: dict, parameters: dict) -> None:
    selection, native = result["selection"], result["native_result"]
    method, epochs = request["method"], request["epochs"]
    metric = request["task"]["primary_metric"]
    _same(selection["metric"], metric, "checkpoint metric")
    _same(selection["split"], "validation", "checkpoint split")
    _same(selection["early_stopping"], False, "early stopping")
    _same(selection["epochs_requested"], epochs, "selection requested epochs")
    _same(selection["epochs_completed"], epochs, "selection completed epochs")
    _same(selection["validation_score"], result["validation_metric"], "selected validation score")
    _same(result["completed_epochs"], epochs, "completed epochs")
    interval, steps = parameters["evaluate_every"], parameters["steps"]
    _same(selection["evaluate_every"], interval, "selection cadence")
    selected_step = selection["step"]
    _require(isinstance(selected_step, int) and not isinstance(selected_step, bool)
             and 0 <= selected_step <= steps, "selected step out of schedule")
    if method in NONPRIVATE or method in ("dp_mlp", "dpar"):
        epoch = selection["epoch"]
        _require(isinstance(epoch, int) and 1 <= epoch <= epochs, "selected epoch out of schedule")
        updates = native["completed_updates"]
        _same(selection["completed_updates"], updates, "actual optimizer update count")
        _require(isinstance(updates, int) and selected_step <= updates <= steps, "invalid executed updates")
        if method == "dp_mlp":
            _same(native["completed_steps"], steps, "DP-MLP logical Poisson draw schedule")
            _same(selection["completed_steps"], steps, "selection logical draw schedule")
        else:
            _same(updates, steps, "full optimizer update schedule")
            _same(selected_step, epoch * interval, "selected epoch step")
    elif method == "progap":
        _same(selection["stages"], 3, "ProGAP stage count")
        _same(selection["completed_stage_epochs"], 3 * epochs, "ProGAP completed stage epochs")
        _same(selection["completed_updates"], steps, "ProGAP completed updates")
        _same(native["completed_updates"], steps, "ProGAP native completed updates")
        _same(native["epochs_completed"], 3 * epochs, "ProGAP native completed epochs")
        stages = selection["stage_selections"]
        _same(len(stages), 3, "ProGAP checkpoint count")
        for stage, checkpoint in enumerate(stages):
            _same(checkpoint["stage"], stage, "ProGAP stage index")
            _same(checkpoint["epochs_completed"], epochs, "ProGAP stage full schedule")
            _same(checkpoint["completed_updates"], epochs * interval, "ProGAP stage updates")
            _same(checkpoint["metric"], metric, "ProGAP stage metric")
            _finite(checkpoint["validation_score"], "ProGAP stage validation")
            epoch = checkpoint["epoch"]
            _require(isinstance(epoch, int) and 1 <= epoch <= epochs, "ProGAP selected epoch")
            _same(checkpoint["stage_step"], epoch * interval, "ProGAP selected stage step")
            _same(checkpoint["step"], (stage * epochs + epoch) * interval, "ProGAP selected global step")
        for key in ("stage", "epoch", "stage_step", "step", "validation_score"):
            _same(selection[key], stages[-1][key], f"ProGAP final stage {key}")
    else:
        _require(1 <= selected_step <= steps and selected_step % interval == 0,
                 "selected checkpoint is not on expected-epoch cadence")
    # The native checkpoint is exported independently of the worker's annotations.
    for key, value in native["selection"].items():
        _same(selection[key], value, f"native checkpoint {key}")


def _verify_science(request: dict, config: dict, result: dict, prepared: dict) -> bool:
    method, n, epochs = request["method"], request["n_train"], request["epochs"]
    batch, metric = min(n, request["batch_size"]), request["task"]["primary_metric"]
    expected_identity = {
        "protocol": request["protocol"], "dataset": request["dataset"], "method": method,
        "target_epsilon": request["epsilon"], "target_delta": request["delta"],
        "train_nodes": n, "requested_batch_size": request["batch_size"], "batch_size": batch,
        "hidden": request["hidden"], "weight_decay": request["weight_decay"], "dropout": request["dropout"],
        "epochs": epochs, "lr": request["lr"], "seed": request["seed"], "split_seed": 0,
        "dp": method not in NONPRIVATE, "metric": metric,
        "split_strategy": prepared["split_strategy"], "split": f"{prepared['split_strategy']}:seed0",
        "domain_split": prepared["domain_split"], "domain_split_id": prepared["domain_split_id"],
    }
    for key, value in expected_identity.items():
        _same(config[key], value, f"config {key}")
        _same(result[key], value, f"result {key}")
    _same(config["task"], request["task"], "config task")
    _same(config["epoch_semantics"], request["epoch_semantics"], "epoch semantics")
    _same(config["parameters"], result["parameters"], "config/result parameters")
    parameters, native = result["parameters"], result["native_result"]
    for key, expected in request["expected_parameters"].items():
        _same(parameters[key], expected, f"actual parameter {key}")
    if "optimizer_betas" in parameters:
        _same(parameters["optimizer_betas"], [0.9, 0.999], "Adam betas")
    if "optimizer_epsilon" in parameters:
        _same(parameters["optimizer_epsilon"], 1e-8, "Adam epsilon")
    steps = parameters["steps"]
    _same(result["steps"], steps, "result update schedule")
    _same(config["steps"], steps, "config update schedule")
    _require(isinstance(steps, int) and steps > 0, "invalid full schedule")
    for key in ("test_metric", "validation_metric", f"test_{metric}"):
        _finite(result[key], key)
    _same(result[f"test_{metric}"], result["test_metric"], "primary score")
    if method == "dpar":
        privacy = native["privacy"]
        roots = privacy["ppr"]["composition_count"]
        _same(roots, min(70, math.ceil(0.09 * n)), "native released PPR roots")
        _same(parameters["epoch_population"], roots, "released-root epoch population")
        _same(parameters["effective_batch_size"], min(batch, roots), "DPAR effective batch")
        _same(steps, epochs * math.ceil(roots / batch), "DPAR root-pass schedule")
        _same(parameters["evaluate_every"], math.ceil(roots / batch), "DPAR validation cadence")
        _same(privacy["training"]["composition_count"], steps, "DPAR accounted updates")
        _same(native["sampled_train_graph"]["nodes"], math.ceil(0.09 * n), "DPAR sampled population")
    _same(result["effective_batch_size"], parameters.get("effective_batch_size", batch), "effective batch")
    if method not in NONPRIVATE:
        from scripts.full_matrix_run import _privacy_pair

        epsilon, delta = _privacy_pair(native, True, request["epsilon"], request["delta"])
        _same(result["epsilon"], epsilon, "actual epsilon")
        _same(config["epsilon"], epsilon, "config actual epsilon")
        _same(result["delta"], delta, "actual delta")
        _same(config["delta"], delta, "config actual delta")
        _require(isinstance(native["calibration"], dict) and bool(native["calibration"]),
                 "native noise calibration evidence missing")
        if method in ("dp_mlp", "dp_gnn_sage", "dp_gnn_gin", "sparse_sage", "sparse_gin"):
            privacy = native["privacy"]
            _same(privacy["composition_count"], steps, "accounted steps")
            _same(privacy["sampling_probability"], batch / n, "accounted sampling probability")
            _require(_finite(privacy["noise_multiplier"], "noise multiplier") > 0, "noise must be positive")
        if method.startswith("dp_gnn_"):
            private = native["privacy"]["parameters"]
            _same(private["max_terms"], min(6, n), "effective DP-GNN sensitivity terms")
            _same(private["opacus_noise_multiplier"], 2 * min(6, n) * parameters["noise_multiplier"],
                  "sensitivity-normalized Opacus multiplier")
        if method == "progap":
            privacy = native["privacy"]["total"]
            _same(privacy["parameters"]["component_coefficients"], [2, 3], "native ProGAP NAP/SGD composition")
            _same(privacy["composition_count"], 5, "native ProGAP component composition count")
            _same(privacy["sampling_probability"], batch / n, "ProGAP accounted sample rate")
            _same(privacy["parameters"]["effective_delta"], request["delta"], "ProGAP effective delta")
            _require(_finite(privacy["noise_multiplier"], "ProGAP noise") > 0, "ProGAP noise must be positive")
    else:
        _same(result["epsilon"], None, "nonprivate epsilon")
        _same(result["delta"], None, "nonprivate delta")
    _verify_selection(request, result, parameters)
    interval = result["test_confidence_intervals"]
    _same(interval, native["test_confidence_intervals"], "native confidence intervals")
    for key, value in BOOTSTRAP.items():
        _same(interval[key], value, f"bootstrap {key}")
    _require(isinstance(interval["n_observations"], int) and interval["n_observations"] > 0,
             "bootstrap has no scored observations")
    _same(interval["n_observations"], prepared["scored_populations"]["test"], "bootstrap scored population")
    primary = interval["metrics"].get(metric)
    if primary is None:
        return False
    _require(isinstance(primary, dict), "malformed primary confidence interval")
    valid = primary["valid_resamples"]
    _require(isinstance(valid, int) and not isinstance(valid, bool) and 0 <= valid <= 1000,
             "invalid bootstrap valid-resample count")
    if valid == 0 or primary["lower"] is None or primary["upper"] is None:
        return False
    lower, upper = _finite(primary["lower"], "CI lower"), _finite(primary["upper"], "CI upper")
    _require(lower <= upper, "reversed primary confidence interval")
    return True


def _verify_attempt(root: Path, request: dict, attempt_dir: Path, manifest: dict) -> dict:
    attempt_dir = _owned(Path(attempt_dir).absolute(), root / "attempts")
    binding = read_json(attempt_dir / "request.json")
    number = binding["attempt_number"]
    _same(attempt_dir, root / "attempts" / request["request_key"] / f"attempt_{number}", "attempt directory")
    _same({key: value for key, value in binding.items() if key not in BINDING_FIELDS}, request,
          "attempt request membership")
    _same(binding["campaign_manifest_path"], str(root / "campaign_manifest.json"), "attempt manifest path")
    manifest_hash = sha256(root / "campaign_manifest.json")
    _same(binding["campaign_manifest_sha256"], manifest_hash, "attempt manifest hash")
    output = attempt_dir / "output"
    marker = read_json(_owned(output / "worker_exit.json", attempt_dir))
    _same(marker["status"], "completed", "worker commitment status")
    _same(marker["request_key"], request["request_key"], "worker commitment request")
    _same(marker["campaign_manifest_sha256"], manifest_hash, "worker commitment manifest")
    _require(isinstance(marker["completed_utc"], str) and bool(marker["completed_utc"]), "missing completion timestamp")
    _same(set(marker["artifact_sha256"]), {"config.json", "result.json", "result.csv"}, "committed artifact inventory")
    _require(not (output / "worker_error.json").exists(), "successful output also has worker error evidence")
    for name, digest in marker["artifact_sha256"].items():
        _same(sha256(_owned(output / name, output)), digest, f"committed {name}")
    config, result = read_json(output / "config.json"), read_json(output / "result.json")
    _same(result["status"], "completed", "result status")
    identity = {key: binding[key] for key in
                ("request_key", "campaign_manifest_sha256", "prepared_fingerprint", "source_fingerprint", "attempt_number")}
    for key, value in identity.items():
        _same(config[key], value, f"config provenance {key}")
        _same(result[key], value, f"result provenance {key}")
    _same(config["source_fingerprint"], manifest["source_fingerprint"], "manifest source identity")
    prepared_path = Path(request["prepared_manifest"])
    _same(sha256(prepared_path), request["prepared_manifest_sha256"], "prepared manifest hash")
    prepared = read_json(prepared_path)
    _same(prepared["fingerprint"], request["prepared_fingerprint"], "prepared fingerprint")
    _same(_prepared_fingerprint(prepared), request["prepared_fingerprint"], "prepared content identity")
    csv.field_size_limit(sys.maxsize)
    with (output / "result.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        rows = list(reader)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                 "malformed result CSV header")
    _same(len(rows), 1, "result CSV row count")
    csv_row = rows[0]
    _require(None not in csv_row and all(value is not None for value in csv_row.values()), "malformed result CSV")
    required_csv = set(identity) | {
        "status", "test_metric", "validation_metric", "parameters", "selection", "test_confidence_intervals",
        "completed_epochs", "method", "protocol", "metric", "epsilon", "delta", "target_epsilon",
        "requested_batch_size", "weight_decay", "steps", "epochs", "lr", "seed", "hidden",
    }
    _require(required_csv <= set(csv_row), "result CSV omits required fields")
    for key, text in csv_row.items():
        _require(key in result, f"unregistered CSV field: {key}")
        value = result[key]
        if isinstance(value, (dict, list)):
            _same(json.loads(text), value, f"CSV {key}")
        elif value is None:
            _same(text, "", f"CSV {key}")
        elif isinstance(value, bool):
            _same(text.lower(), str(value).lower(), f"CSV {key}")
        elif isinstance(value, (int, float)):
            _same(_finite(text, key), value, f"CSV {key}")
        else:
            _same(text, str(value), f"CSV {key}")
    ci_available = _verify_science(request, config, result, prepared)
    return {"accepted": True, "status": "completed" if ci_available else "ci_unavailable",
            "reason": None if ci_available else "required primary node-bootstrap CI is unavailable",
            "request_key": request["request_key"], "attempt_number": number,
            "result": result, "ci_available": ci_available}


def verify_attempt(root: Path, request: dict, attempt_dir: Path) -> dict:
    """Check immutable scientific commitment; the controller checks OS exit/cleanup."""
    root = Path(root).absolute()
    try:
        manifest, requests = load_campaign(root, check_sources=False, check_prepared=False)
        registered = next((row for row in requests if row["request_key"] == request["request_key"]), None)
        _same(request, registered, "registered attempt request")
        return _verify_attempt(root, request, attempt_dir, manifest)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, OverflowError) as error:
        name = Path(attempt_dir).name
        number = int(name.removeprefix("attempt_")) if name.removeprefix("attempt_").isdigit() else None
        return {"accepted": False, "status": "invalid", "reason": str(error),
                "request_key": request.get("request_key"), "attempt_number": number, "ci_available": False}


def _csv_write(path: Path, rows: list[dict], *, columns: list[str] | None = None) -> None:
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _read_optional(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _collect_evidence(root: Path, manifest: dict, requests: list[dict]) -> dict:
    registry = {request["request_key"]: request for request in requests}
    queue = _read_optional(root / "queue_state.json")
    errors, attempts, timing, accepted = [], [], [], {}
    by_request = defaultdict(list)
    folders = []
    attempts_root = root / "attempts"
    if attempts_root.exists():
        for request_dir in sorted(attempts_root.iterdir()):
            if not request_dir.is_dir() or request_dir.name not in registry:
                errors.append(f"unregistered attempt evidence: {request_dir}")
                unknown = sorted(request_dir.iterdir()) if request_dir.is_dir() else [request_dir]
                for folder in unknown:
                    row = {"request_key": request_dir.name, "attempt_number": None,
                           "attempt_dir": str(folder), "status": "invalid", "accepted": False,
                           "ci_available": False, "reason": "unregistered attempt evidence"}
                    attempts.append(row)
                    timing.append({**row, "wall_seconds": None})
                continue
            for folder in request_dir.iterdir():
                suffix = folder.name.removeprefix("attempt_")
                if not folder.is_dir() or not folder.name.startswith("attempt_") or not suffix.isdigit():
                    errors.append(f"invalid attempt entry: {folder}")
                    row = {"request_key": request_dir.name, "attempt_number": None,
                           "attempt_dir": str(folder), "status": "invalid", "accepted": False,
                           "ci_available": False, "reason": "invalid attempt entry"}
                    attempts.append(row)
                    timing.append({**row, "wall_seconds": None})
                    continue
                folders.append((registry[request_dir.name]["ordinal"], int(suffix), folder))
    for _, number, folder in sorted(folders):
        key = folder.parent.name
        request = registry[key]
        row = {"request_key": key, "attempt_number": number, "protocol": request["protocol"],
               "method": request["method"], "attempt_dir": str(folder), "accepted": False}
        try:
            binding = read_json(folder / "request.json")
            _same(binding["attempt_number"], number, "attempt number binding")
            _same({k: v for k, v in binding.items() if k not in BINDING_FIELDS}, request, "attempt immutable request")
            _same(binding["campaign_manifest_sha256"], sha256(root / "campaign_manifest.json"), "attempt manifest binding")
            launch = _read_optional(folder / "launch.json")
            exit_record = _read_optional(folder / "exit.json")
            status = exit_record.get("status", "running" if launch else "interrupted")
            row.update(status=status, reason=exit_record.get("reason"), started_utc=exit_record.get("started_utc"),
                       ended_utc=exit_record.get("ended_utc"), wall_seconds=exit_record.get("wall_seconds"),
                       gpu_uuid=exit_record.get("gpu_uuid", launch.get("gpu_uuid")),
                       returncode=exit_record.get("returncode"), owned_process_exited=exit_record.get("owned_process_exited", False),
                       retry_state=exit_record.get("retry_state"), retry_eligible=status in ("oom", "interrupted"),
                       admission=_read_optional(folder / "admission.json"))
            if status == "invalid":
                errors.append(f"{folder}: {row['reason'] or 'invalid terminal evidence'}")
            complete = status in ("completed", "recovered_committed")
            if complete:
                _same(exit_record.get("owned_process_exited"), True, "accepted attempt cleanup proof")
                _require(exit_record.get("returncode") == 0 or
                         (status == "recovered_committed" and exit_record.get("returncode") is None),
                         "accepted attempt lacks successful exit/recovery proof")
                verification = _verify_attempt(root, request, folder, manifest)
                row.update(accepted=True, ci_available=verification["ci_available"],
                           verification_status=verification["status"], reason=verification["reason"])
                _require(key not in accepted, "multiple accepted attempts for one request")
                accepted[key] = (row, verification["result"])
            elif exit_record.get("accepted"):
                raise ValueError("non-successful exit incorrectly marked accepted")
            row["evidence_sha256"] = {
                str(path.relative_to(folder)): sha256(path)
                for path in (folder / "request.json", folder / "launch.json", folder / "exit.json",
                             folder / "admission.json", folder / "output/worker_exit.json",
                             folder / "output/worker_error.json", folder / "output/config.json",
                             folder / "output/result.json", folder / "output/result.csv")
                if path.is_file()
            }
            result = (accepted[key][1] if key in accepted and accepted[key][0] is row
                      else _read_optional(folder / "output/result.json"))
            native = result.get("native_result", {})
            timing.append({
                "request_key": key, "attempt_number": number, "status": status,
                "wall_seconds": exit_record.get("wall_seconds"), "started_utc": row["started_utc"],
                "ended_utc": row["ended_utc"], "loading_seconds": result.get("loading_seconds"),
                "worker_wall_seconds": result.get("worker_wall_seconds"),
                "calibration_and_training_seconds": result.get("calibration_and_training_seconds"),
                "native_timing": result.get("native_timing"),
                "native_preprocessing_seconds": native.get("preprocessing_seconds"),
                "native_training_seconds": native.get("training_seconds"),
                "sampled_peak_owned_rss_bytes": exit_record.get("peak_rss_bytes"),
                "sampled_peak_gpu_memory_mib": exit_record.get("peak_gpu_memory_mib"),
                "gpu_memory_scope": "GPU-wide sample; not exclusively attributable during contention",
                "contention": exit_record.get("contention_observed"),
                "resource_samples_path": str(folder / "resources.jsonl"),
            })
        except (OSError, ValueError, KeyError, TypeError, csv.Error, OverflowError) as error:
            row.update(status="invalid", accepted=False, ci_available=False, reason=str(error), retry_eligible=False)
            errors.append(f"{folder}: {error}")
            if key in accepted and accepted[key][0] is row:
                accepted.pop(key)
            timing.append({"request_key": key, "attempt_number": number, "status": "invalid",
                           "wall_seconds": None, "reason": str(error)})
        attempts.append(row)
        by_request[key].append(row)
    results, coverage = [], []
    for request in requests:
        key = request["request_key"]
        history = by_request[key]
        latest = history[-1] if history else {}
        projection = queue.get(key, {})
        accepted_pair = accepted.get(key)
        if accepted_pair:
            attempt, native = accepted_pair
            ci = native["test_confidence_intervals"]
            primary = ci["metrics"].get(request["task"]["primary_metric"], {})
            disposition = "completed" if attempt["ci_available"] else "ci_unavailable"
            metrics = {
                "test_metric": native["test_metric"], "validation_metric": native["validation_metric"],
                "ci_lower": primary.get("lower"), "ci_upper": primary.get("upper"),
                "ci_valid_resamples": primary.get("valid_resamples"), "test_confidence_intervals": ci,
                "selection": native["selection"], "actual_epsilon": native["epsilon"], "actual_delta": native["delta"],
                "actual_steps": native["steps"], "effective_batch_size": native["effective_batch_size"],
                "parameters": native["parameters"], "result_csv": str(Path(attempt["attempt_dir"]) / "output/result.csv"),
                "native_result_json": str(Path(attempt["attempt_dir"]) / "output/result.json"),
                "attempt_number": attempt["attempt_number"], "ci_available": attempt["ci_available"],
            }
            reason = attempt["reason"]
        else:
            disposition = latest.get("status", projection.get("status", "pending"))
            if disposition == "oom":
                disposition = "retry_deferred" if projection.get("not_before", 0) > time.time() else "pending"
            elif disposition == "running" and projection.get("status") == "interrupted":
                disposition = "interrupted"
            reason = latest.get("reason") or projection.get("reason")
            metrics = {name: None for name in (
                "test_metric", "validation_metric", "ci_lower", "ci_upper", "ci_valid_resamples",
                "test_confidence_intervals", "selection", "actual_epsilon", "actual_delta", "actual_steps",
                "effective_batch_size", "parameters", "result_csv", "native_result_json", "attempt_number")}
            metrics["ci_available"] = False
        results.append({**request, "status": disposition, "accepted": bool(accepted_pair),
                        "reason": reason, **metrics})
        coverage.append({"request_key": key, "ordinal": request["ordinal"], "protocol": request["protocol"],
                         "method": request["method"], "epsilon": request["epsilon"], "status": disposition,
                         "accepted": bool(accepted_pair), "ci_available": metrics["ci_available"],
                         "attempt_count": len(history), "attempt_number": metrics["attempt_number"],
                         "retry_eligible": not bool(accepted_pair) and disposition in ("pending", "running", "interrupted", "retry_pending", "retry_deferred"),
                         "prompt_ooms": projection.get("prompt_ooms", 0), "not_before": projection.get("not_before", 0),
                         "last_gpu": projection.get("last_gpu"), "reason": reason})
    return {"attempts": attempts, "timing": timing, "results": results, "coverage": coverage,
            "accepted": accepted, "errors": errors}


def _summarize_evidence(evidence: dict) -> tuple[list[dict], Any]:
    from scripts import summarize_results as summary

    diagnostics = summary.Diagnostics()
    accepted = evidence["accepted"]
    paths = [Path(attempt["attempt_dir"]) / "output/result.csv" for attempt, _ in accepted.values()]
    if not paths:
        return [], diagnostics
    runs = summary.read_group("default", sorted(paths), {}, "auto", diagnostics)
    finals = summary.final_runs(runs, diagnostics)
    rows = summary.summarize(finals, False, diagnostics)
    return summary.select_best(rows, diagnostics), diagnostics


def _comparisons(requests: list[dict], evidence: dict, summary_rows: list[dict]) -> list[dict]:
    result_by_key = {row["request_key"]: row for row in evidence["results"]}
    winners = {row["sources"]: row for row in summary_rows}
    comparisons = []
    for slot in _comparison_slots(requests):
        candidates = [result_by_key[key] for key in slot["request_keys"]]
        accepted = [row for row in candidates if row["accepted"]]
        complete = len(accepted) == len(candidates) and all(row["ci_available"] for row in accepted)
        selected = [row for row in accepted if row["result_csv"] in winners]
        _require(len(selected) <= 1, f"ambiguous summary mapping for {slot}")
        _require(not accepted or len(selected) == 1, f"summary omitted accepted comparison {slot}")
        chosen = selected[0] if selected else None
        comparison = {**slot, "expected_candidate_count": len(candidates), "accepted_count": len(accepted),
                      "ci_complete_count": sum(row["ci_available"] for row in accepted),
                      "status": "complete_grid" if complete else "partial_grid" if accepted else "unavailable",
                      "selection": "best_test", "test_selection_bias": True,
                      "selected_request_key": chosen["request_key"] if chosen else None,
                      "selected_attempt_number": chosen["attempt_number"] if chosen else None,
                      "missing_request_keys": [row["request_key"] for row in candidates
                                               if not row["accepted"] or not row["ci_available"]]}
        for key in ("test_metric", "ci_lower", "ci_upper", "ci_valid_resamples", "test_confidence_intervals",
                    "actual_epsilon", "actual_delta", "result_csv"):
            comparison[key] = chosen[key] if chosen else None
        comparison["summary_config_id"] = winners[chosen["result_csv"]]["config_id"] if chosen else None
        comparisons.append(comparison)
    return comparisons


def _coverage_summary(manifest: dict, evidence: dict, comparisons: list[dict]) -> dict:
    accepted = sum(row["accepted"] for row in evidence["coverage"])
    ci_complete = sum(row["accepted"] and row["ci_available"] for row in evidence["coverage"])
    complete_cells = sum(row["status"] == "complete_grid" for row in comparisons)
    return {
        "purpose": manifest["purpose"], "expected_requests": manifest["expected_requests"],
        "accepted_count": accepted, "ci_complete_count": ci_complete,
        "expected_comparisons": manifest["expected_comparisons"], "comparison_complete_count": complete_cells,
        "fully_executed": ci_complete == manifest["expected_requests"]
        and complete_cells == manifest["expected_comparisons"] and not evidence["errors"],
        "attempt_count": len(evidence["attempts"]), "dispositions": dict(Counter(row["status"] for row in evidence["coverage"])),
        "missing_request_keys": [row["request_key"] for row in evidence["coverage"]
                                 if not row["accepted"] or not row["ci_available"]],
        "missing_comparison_cells": [
            {"protocol": row["protocol"], "method": row["method"], "epsilon": row["target_epsilon"],
             "status": row["status"]} for row in comparisons if row["status"] != "complete_grid"],
        "errors": evidence["errors"], "invalid_count": sum(row["status"] == "invalid" for row in evidence["attempts"]),
        "corrupt": bool(evidence["errors"]), "interpretation": INTERPRETATION,
    }


def _event_watermark(root: Path) -> dict:
    path = root / "events.jsonl"
    if not path.exists():
        return {"complete_lines": 0, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    # Appenders lock this same short-lived event boundary; only complete lines count.
    with file_lock(path, shared=True):
        data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    data = data[:end]
    return {"complete_lines": data.count(b"\n"), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_reports(root: Path) -> dict:
    """Build outside report.lock, then publish one hash-bound generation atomically.

    report.build.lock serializes independent report processes, without holding the
    reader lock during result verification, CSV parsing, or summary formatting.
    The scheduler invokes this only in its dedicated reporting thread.
    """
    from scripts import summarize_results as summary

    root = Path(root).absolute()
    with file_lock(root / "report.build.lock"):
        manifest, requests = load_campaign(root, check_sources=False, check_prepared=False)
        watermark = _event_watermark(root)
        evidence = _collect_evidence(root, manifest, requests)
        summary_rows, diagnostics = _summarize_evidence(evidence)
        comparisons = _comparisons(requests, evidence, summary_rows)
        status = _coverage_summary(manifest, evidence, comparisons)
        staging = Path(tempfile.mkdtemp(prefix=".report.", dir=root))
        try:
            tables = {key: evidence[key] for key in ("attempts", "timing", "results", "coverage")}
            tables["comparison"] = comparisons
            for name, rows in tables.items():
                atomic_json(staging / f"{name}.json", rows)
                _csv_write(staging / f"{name}.csv", rows,
                           columns=None if rows else ["request_key", "attempt_number", "status"])
            outputs = (staging / "summary.csv", staging / "summary.md")
            args = argparse.Namespace(bootstrap=True, seed=False, best=True)
            summary.export(summary_rows, outputs, args, diagnostics)
            with outputs[1].open("a", encoding="utf-8") as stream:
                if not summary_rows:
                    stream.write("\nNo accepted results: the comparison is unavailable.\n")
                stream.write(f"\nAccepted requests: {status['accepted_count']}/{status['expected_requests']}; "
                             f"CI-complete: {status['ci_complete_count']}; complete comparison cells: "
                             f"{status['comparison_complete_count']}/{status['expected_comparisons']}.\n")
                stream.write("\n" + "\n".join(f"- {value}" for value in INTERPRETATION.values()) + "\n")
                if status["missing_comparison_cells"]:
                    stream.write("\nUnavailable and partial cells are retained explicitly in comparison.csv.\n")
                if status["errors"]:
                    stream.write(f"\nEvidence errors: {len(status['errors'])}; see coverage_summary.json.\n")
            atomic_json(staging / "coverage_summary.json", status)
            files = sorted(path.name for path in staging.iterdir())
            hashes = {name: sha256(staging / name) for name in files}
            with file_lock(root / "report.lock"):
                previous = _read_optional(root / "report_generation.json")
                generation = {"format": 1, "generation": previous.get("generation", 0) + 1,
                              "generated_utc": utc_now(), "input_event_watermark": watermark,
                              "files_sha256": hashes, "coverage": status}
                for name in files:
                    os.replace(staging / name, root / name)
                atomic_json(root / "report_generation.json", generation)
            return status
        finally:
            shutil.rmtree(staging)


def read_reports(root: Path) -> dict:
    """Read one complete generation or reject a torn/crashed publication."""
    root = Path(root).absolute()
    with file_lock(root / "report.lock", shared=True):
        generation = read_json(root / "report_generation.json")
        result = {"generation": generation}
        for name, digest in generation["files_sha256"].items():
            path = _owned(root / name, root)
            _same(sha256(path), digest, f"report generation file {name}")
            result[name] = read_json(path) if name.endswith(".json") else path.read_text(encoding="utf-8")
        _same(read_json(root / "report_generation.json"), generation, "report generation changed while reading")
        return result


def verify_campaign(root: Path) -> dict:
    """Read-only full integrity audit; incomplete is distinct from corrupt."""
    root = Path(root).absolute()
    try:
        manifest, requests = load_campaign(root, check_sources=True, check_prepared=True)
        evidence = _collect_evidence(root, manifest, requests)
        summaries, _ = _summarize_evidence(evidence)
        comparisons = _comparisons(requests, evidence, summaries)
        status = _coverage_summary(manifest, evidence, comparisons)
        if (root / "report_generation.json").exists():
            read_reports(root)
        from scripts.full_matrix_runtime import group_members

        blockers = []
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for attempt in evidence["attempts"]:
            folder = Path(attempt["attempt_dir"])
            launch = _read_optional(folder / "launch.json")
            if launch.get("boot_id") != boot_id or launch.get("pgid") is None:
                continue
            persisted = _read_optional(folder / "process_identities.json")
            identities = persisted.get("members", launch.get("observed_members", []))
            if "pid" in launch:
                identities = [*identities, launch]
            for current in group_members(int(launch["pgid"])):
                if current.get("state") in ("Z", "X"):
                    continue
                if any(all(current.get(key) == recorded.get(key) for key in
                           ("pid", "start_ticks", "pgid", "session_id"))
                       for recorded in identities):
                    blockers.append(f"owned process remains for {attempt['attempt_dir']}: PID {current['pid']}")
        status["process_blockers"] = blockers
        status["fully_executed"] = status["fully_executed"] and not blockers
        return status
    except (OSError, ValueError, KeyError, TypeError, csv.Error, OverflowError, subprocess.SubprocessError) as error:
        return {"fully_executed": False, "accepted_count": 0, "expected_requests": None,
                "comparison_complete_count": 0, "invalid_count": 1, "corrupt": True, "errors": [str(error)]}
