#!/usr/bin/env python3
"""Train one full-matrix cell and publish a normalized final-result CSV.

The common split is seed 0, independently of the model seed. Epochs are full
training-population passes (expected passes for root-sampling mechanisms).
ProGAP retains its native drop-last schedule and trains three stages, each for
--epochs epochs. Every backend selects using validation, without early stopping.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "mlp", "graphsage", "gin", "dp_mlp", "progap", "dpar",
    "dp_gnn_sage", "dp_gnn_gin", "sparse_sage", "sparse_gin",
)
NONPRIVATE = {"mlp", "graphsage", "gin"}
NATIVE_PROTOCOLS = {
    "ogbn-arxiv", "ogbn-products", "reddit", "facebook", "flickr",
    "saint-reddit", "saint-yelp", "saint-flickr", "saint-amazon", "ppi-large",
}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", required=True, help="dataset name or *-allbut2 protocol")
    cli.add_argument("--method", required=True, choices=METHODS)
    cli.add_argument("--lr", required=True, type=float)
    cli.add_argument("--batch-size", required=True, type=int)
    cli.add_argument("--epochs", required=True, type=int)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--dropout", type=float, default=0.5)
    cli.add_argument("--mlp-hidden", type=int, default=64)
    cli.add_argument("--gnn-hidden", type=int, default=128)
    cli.add_argument("--device", default="cuda", help="explicit torch device; never falls back to CPU")
    cli.add_argument("--out-dir", required=True, type=Path, help="new, unoccupied per-cell directory")
    cli.add_argument("--epsilon", type=float, help="required for private methods; rejected otherwise")
    cli.add_argument("--p2", type=float, help="required only for SparseGNN")
    cli.add_argument("--sparse-radius", type=int, default=1,
                     help="SparseExpand radius (positive integer; SparseGNN only)")
    cli.add_argument("--sparse-degree-cap", type=int, default=10,
                     help="SparseGNN preprocessing outgoing-degree cap; not the fixed incoming sampling cap")
    cli.add_argument("--progap-python", help="ProGAP interpreter (default: this Python executable)")
    cli.add_argument("--bootstrap-resamples", type=int, default=1000,
                     help="final-test node bootstrap resamples at 95%% confidence; 0 disables")
    cli.add_argument("--split-root", type=Path, default=REPO_ROOT / "data" / "inductive_splits",
                     help="common seed-0 split cache (default: repository data/inductive_splits)")
    cli.add_argument("--prepared-protocol", type=Path,
                     help="immutable prepared partition manifest for a campaign")
    cli.add_argument("--campaign-request", type=Path,
                     help="hash-bound per-attempt campaign request")
    return cli


def _check_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    for name in ("batch_size", "epochs", "mlp_hidden", "gnn_hidden"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("sparse_radius", "sparse_degree_cap"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    sparse_override = args.sparse_radius != 1 or args.sparse_degree_cap != 10
    if sparse_override and not args.method.startswith("sparse_"):
        raise ValueError("--sparse-radius and --sparse-degree-cap overrides are supported only by SparseGNN")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be finite and in [0, 1)")
    if args.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be nonnegative")
    if args.method in NONPRIVATE:
        if args.epsilon is not None:
            raise ValueError("--epsilon is not applicable to a non-private method")
    elif args.epsilon is None or not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("private methods require a finite positive --epsilon")
    if args.method.startswith("sparse_"):
        if args.p2 is None or not math.isfinite(args.p2) or not 0 < args.p2 <= 1:
            raise ValueError("SparseGNN requires --p2 in (0, 1]")
    elif args.p2 is not None:
        raise ValueError("--p2 is supported only by SparseGNN")
    if args.progap_python is not None and args.method != "progap":
        raise ValueError("--progap-python is supported only by ProGAP")
    if (args.prepared_protocol is None) != (args.campaign_request is None):
        raise ValueError("--prepared-protocol and --campaign-request must be supplied together")
    if args.prepared_protocol is not None:
        if sparse_override:
            raise ValueError("prepared campaigns do not support SparseGNN radius or degree-cap overrides")
        args.prepared_protocol = args.prepared_protocol.expanduser().resolve()
        args.campaign_request = args.campaign_request.expanduser().resolve()
    args.dataset = args.dataset.lower()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.split_root = args.split_root.expanduser().resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"output directory is occupied: {args.out_dir}; no overwrite or resume")
    if args.method == "progap":
        executable = shutil.which(args.progap_python or sys.executable)
        if executable is None:
            raise FileNotFoundError(f"ProGAP Python executable not found: {args.progap_python}")
        # Preserve venv symlinks: resolving them changes Python's environment.
        args.progap_python = str(Path(executable).absolute())


def _protocol(name: str) -> tuple[str, dict[str, Any] | None]:
    if name not in {"twitch-allbut2", "facebook100-allbut2", "mag-allbut2"}:
        return name, None
    from src.data.domain_datasets import FB100_DOMAINS, MAG_DOMAINS, TWITCH_DOMAINS

    dataset, domains, validation, test = {
        "twitch-allbut2": ("twitch-explicit", TWITCH_DOMAINS, "engb", "es"),
        "facebook100-allbut2": ("facebook100", FB100_DOMAINS, "cornell5", "penn94"),
        "mag-allbut2": ("mag-countries", MAG_DOMAINS, "cn", "de"),
    }[name]
    return dataset, {
        "train": [domain for domain in domains if domain not in {validation, test}],
        "val": [validation], "test": [test], "seed": 0, "val_ratio": 0.2,
    }


def _load_split(protocol: str, split_root: Path):
    from src.data.datasets import load_dataset
    from src.experiments.run import _resolve_task_metadata
    from src.processing.graphs import preprocess_inductive_split
    from src.processing.splits import load_or_create_inductive_split

    dataset_name, domain_options = _protocol(protocol)
    try:
        dataset, data = load_dataset(dataset_name, device="cpu", domain_split=domain_options)
    except (OSError, ImportError, RuntimeError) as error:
        raise RuntimeError(
            f"could not load protocol {protocol!r} ({dataset_name}) on CPU: {error}; "
            "install the dataset dependencies and prepare its configured dataset cache"
        ) from error
    task = _resolve_task_metadata(dataset, {})
    is_domain = bool(getattr(dataset, "domain_dataset", False))
    strategy = "domain" if is_domain else (
        getattr(dataset, "split_strategy", None) or
        ("native" if protocol in NATIVE_PROTOCOLS or dataset_name.startswith("graphsaint:")
         or task["multilabel"] or task["regression"] else "stratified")
    )
    split = load_or_create_inductive_split(
        data, dataset_name, root=split_root, seed=0,
        split_strategy=strategy, **task,
        domain_split=getattr(dataset, "domain_split", None),
        domain_split_id=getattr(dataset, "domain_split_id", None),
    )
    split = preprocess_inductive_split(split)
    for name in ("train", "val", "test"):
        part = getattr(split, name)
        if int(part.data.num_nodes) < 1:
            raise ValueError(f"{protocol}: the {name} partition is empty")
        # The direct DP-GNN API reads masks from Data rather than GraphPartition.
        part.data.eval_mask = part.eval_mask
    return dataset_name, split, task, strategy


def _bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    return {"bootstrap_confidence": 0.95, "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": 0}


def _task_options(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in ("binary", "multilabel", "regression", "metric_ignore_label")}


def _baseline(args, split, task, batch, delta):
    from src.training.baselines import BaselineConfig, BaselineTrainer

    options = {
        "method": args.method, "hidden_size": args.mlp_hidden if args.method in {"mlp", "dp_mlp"} else args.gnn_hidden,
        "learning_rate": args.lr, "batch_size": batch, "epochs": args.epochs,
        "weight_decay": 5e-4,
        "dropout": args.dropout, "seed": args.seed, **_task_options(task), **_bootstrap(args),
    }
    calibration = None
    if args.method == "dp_mlp":
        from src.privacy.accountants import DPMLPAccountant
        calibration = dict(DPMLPAccountant().calibrate(
            args.epsilon, delta, sample_rate=batch / int(split.train.data.num_nodes),
            steps=args.epochs * math.ceil(int(split.train.data.num_nodes) / batch),
        ))
        options.update(calibration, delta=delta)
    config = BaselineConfig(**options)
    result = BaselineTrainer(config, device=args.device).fit(split)
    if calibration is not None:
        result["calibration"] = {**calibration, "target_epsilon": args.epsilon,
                                 "target_delta": delta, "achieved_epsilon": result["privacy"]["epsilon"]}
    parameters = asdict(config)
    if args.method != "dp_mlp":
        for name in ("noise_multiplier", "clip", "delta"):
            parameters.pop(name)
    if args.method not in {"graphsage", "gin"}:
        parameters.pop("graphsage_sampling")
        parameters.pop("max_fanout")
    parameters["optimizer"] = "adam"
    return result, parameters


def _dpar(args, split, task, batch, delta):
    from src.training.dpar import DPARConfig, DPARTrainer

    config = DPARConfig(
        target_epsilon=args.epsilon, target_delta=delta, dp_ppr=True, dp_sgd=True,
        hidden_size=args.gnn_hidden, learning_rate=args.lr, batch_size=batch,
        epochs=args.epochs, dropout=args.dropout, seed=args.seed,
        weight_decay=5e-4,
        **_task_options(task), **_bootstrap(args),
    )
    result = DPARTrainer(config, device=args.device).fit(split)
    # The trainer replaces both noises and split deltas after native calibration.
    parameters = {**result["config"], "optimizer": "adam"}
    privacy = result["privacy"]
    roots = privacy["ppr"]["composition_count"]
    parameters.update(
        epoch_population=roots,
        effective_batch_size=min(config.batch_size, roots),
        steps=privacy["training"]["composition_count"],
        evaluate_every=privacy["training"]["composition_count"] // config.epochs,
    )
    return result, parameters


def _dpgnn_noise(*, target_epsilon, delta, population, batch, steps, max_terms):
    from src.privacy.dpgnn import multiterm_dpsgd_epsilon

    evaluations = 0

    def epsilon_at(sigma):
        nonlocal evaluations
        evaluations += 1
        value = multiterm_dpsgd_epsilon(
            steps=steps, noise_multiplier=sigma, delta=delta,
            num_samples=population, batch_size=batch, max_terms=max_terms,
        )
        if math.isnan(value):
            raise RuntimeError(f"DP-GNN accountant returned NaN at noise_multiplier={sigma}")
        return value

    low, high = 0.0, 1.0
    achieved = epsilon_at(high)
    while achieved > target_epsilon:
        low = high
        if high >= 1e6:
            raise RuntimeError(
                "DP-GNN could not bracket target epsilon with its native finite-order "
                "multi-term accountant at noise_multiplier <= 1e6"
            )
        high = min(2 * high, 1e6)
        achieved = epsilon_at(high)
    while high - low > max(1e-6, 1e-3 * high):
        midpoint = (low + high) / 2
        value = epsilon_at(midpoint)
        if value > target_epsilon:
            low = midpoint
        else:
            high, achieved = midpoint, value
    return {"noise_multiplier": high, "achieved_epsilon": achieved,
            "target_epsilon": target_epsilon, "target_delta": delta,
            "evaluations": evaluations, "sigma_rtol": 1e-3, "sigma_atol": 1e-6}


def _dpgnn(args, split, task, batch, delta):
    from src.privacy.dpgnn import max_terms_per_node
    from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN

    population = int(split.train.data.num_nodes)
    interval = math.ceil(population / batch)
    steps = args.epochs * interval
    max_degree = 5
    max_terms = min(max_terms_per_node(max_degree), population)
    calibration = _dpgnn_noise(
        target_epsilon=args.epsilon, delta=delta, population=population,
        batch=batch, steps=steps, max_terms=max_terms,
    )
    config = DPGNNConfig(
        num_classes=split.num_classes, architecture="gin" if args.method.endswith("gin") else "graphsage",
        latent_size=args.gnn_hidden, learning_rate=args.lr, dropout=args.dropout,
        batch_size=batch, steps=steps, evaluate_every=interval,
        weight_decay=5e-4, delta=delta,
        noise_multiplier=calibration["noise_multiplier"], max_degree=max_degree,
        seed=args.seed, **_task_options(task), **_bootstrap(args),
    )
    result = PartitionedDPGNN(config, device=args.device).fit(
        split.train.data, split.val.data, split.test.data)
    result.pop("model", None)
    result["calibration"] = calibration
    result["privacy"] = {
        "epsilon": result["epsilon"], "delta": result["delta"],
        "accountant": "src.privacy.dpgnn.multiterm_dpsgd_epsilon",
        "noise_multiplier": config.noise_multiplier,
        "sampling_probability": batch / population, "composition_count": steps,
        "parameters": {"max_terms": max_terms, "max_degree": max_degree,
                       "sampling": "uniform_without_replacement",
                       "opacus_noise_multiplier": 2 * max_terms * config.noise_multiplier,
                       "noise_std": 2 * max_terms * config.noise_multiplier * config.clip},
    }
    return result, {**asdict(config), "epochs": args.epochs, "optimizer": "adam",
                    "max_terms": max_terms}


def _sparse_evaluation_graph(split, device):
    import torch
    from torch_geometric.data import Data

    # Validation and test are disconnected components, never a fitting graph.
    # No training-graph copy is needed merely to report a held-out metric.
    val, test = split.val, split.test
    nv, nt = int(val.data.num_nodes), int(test.data.num_nodes)
    return Data(
        x=torch.cat((val.data.x, test.data.x)).to(device),
        y=torch.cat((val.data.y, test.data.y)).to(device),
        edge_index=torch.cat((val.data.edge_index, test.data.edge_index + nv), dim=1).to(device),
        num_nodes=nv + nt,
        train_mask=torch.zeros(nv + nt, dtype=torch.bool, device=device),
        val_mask=torch.cat((val.eval_mask, torch.zeros(nt, dtype=torch.bool))).to(device),
        test_mask=torch.cat((torch.zeros(nv, dtype=torch.bool), test.eval_mask)).to(device),
    )


def _sparse(args, split, task, batch, delta):
    import torch
    from torch_geometric.data import Data
    from src.models.bootstrap import BootstrapConfig
    from src.privacy.accounting import calibrate_sparsegnn_noise
    from src.processing.graphs import max_degrees, preprocess_edges
    from src.processing.sparse_expand import MAX_INCOMING_EDGES, build_adjacency
    from src.training.sparse_gnn import train_sparse_gnn

    if task["binary"]:
        from src.models.binary_mechanism import BinaryGNNMechanism as Mechanism
    elif task["multilabel"]:
        from src.models.multilabel_mechanism import MultiLabelGNNMechanism as Mechanism
    elif task["regression"]:
        from src.models.regression_mechanism import RegressionGNNMechanism as Mechanism
    else:
        from src.models.gnn_mechanism import GNNMechanism as Mechanism
    population = int(split.train.data.num_nodes)
    p1 = batch / population
    interval = math.ceil(population / batch)
    steps = args.epochs * interval
    aggr = "gin" if args.method.endswith("gin") else "mean"
    generator = torch.Generator().manual_seed(args.seed + 20_000)
    train_edges = preprocess_edges(
        split.train.data.edge_index, population,
        max_in_degree=10, max_out_degree=args.sparse_degree_cap,
        degree_cap_mode="directed", add_self_loops=False, generator=generator,
    )
    achieved_degrees = max_degrees(train_edges, population)
    adjacency = build_adjacency(train_edges, population, direction="in")
    train = Data(
        x=split.train.data.x.to(args.device), y=split.train.data.y.to(args.device),
        edge_index=train_edges, num_nodes=population,
        train_mask=torch.ones(population, dtype=torch.bool, device=args.device),
    )
    calibration = calibrate_sparsegnn_noise(
        target_epsilon=args.epsilon, target_delta=delta, p1=p1, p2=args.p2,
        r=args.sparse_radius, K_in=10, K_out=args.sparse_degree_cap, steps=steps, clip=1.0,
        grid=1e-3, sigma_rtol=1e-3, sigma_atol=1e-6, union_safe=False,
    )
    extra = {} if any(task[name] for name in ("binary", "multilabel", "regression")) else {
        "metric_ignore_label": task["metric_ignore_label"]}
    torch.manual_seed(args.seed)
    mechanism = Mechanism(
        train, int(train.x.size(1)), split.num_classes,
        hidden=args.gnn_hidden, num_layers=2, dropout=args.dropout,
        aggr=aggr, device=torch.device(args.device), **extra,
    )
    weight_decay = 5e-4
    mechanism.build_optimizer(lr=args.lr, weight_decay=weight_decay, kind="adam")
    evaluation = _sparse_evaluation_graph(split, args.device)
    result = train_sparse_gnn(
        mechanism, train, evaluation, p1=p1, p2=args.p2, r=args.sparse_radius, T=steps,
        adj=adjacency, direction="in", dp=True, clip=1.0,
        sigma=calibration.noise_multiplier, seed=args.seed, eval_every=interval,
        track_every=0, bootstrap=BootstrapConfig(
            confidence_level=0.95, n_resamples=args.bootstrap_resamples, seed=0),
    )
    result["calibration"] = calibration.as_dict()
    result["privacy"] = {
        "epsilon": calibration.epsilon, "delta": calibration.delta,
        "accountant": "src.privacy.accounting.sparsegnn_mixture_weights.chi1",
        "noise_multiplier": calibration.noise_multiplier,
        "sampling_probability": p1, "composition_count": steps,
        "parameters": {"p1": p1, "p2": args.p2, "r": args.sparse_radius,
                       "K_in": 10, "K_out": args.sparse_degree_cap,
                       "chi": 1, "union_safe": False, "grid": 1e-3,
                       "qualification": "Retains the repository's current chi=1 accounting policy; no union-graph correction."},
    }
    parameters = {
        "architecture": aggr, "hidden": args.gnn_hidden, "layers": 2,
        "lr": args.lr, "batch_size": batch, "epochs": args.epochs, "steps": steps,
        "dropout": args.dropout, "optimizer": "adam", "weight_decay": weight_decay,
        "p1": p1, "p2": args.p2, "r": args.sparse_radius, "clip": 1.0,
        "sigma": calibration.noise_multiplier, "K_in": 10, "K_out": args.sparse_degree_cap,
        "cap_mode": "directed", "cap_seed": args.seed + 20_000, "direction": "in",
        "cap_semantics": "outgoing arcs capped; incoming degree unrestricted",
        "incoming_sampling_cap": MAX_INCOMING_EDGES,
        "K_in_achieved": achieved_degrees[0], "K_out_achieved": achieved_degrees[1],
        "chi": 1, "union_safe": False, "accounting_grid": 1e-3,
        "calibration_rtol": 1e-3, "calibration_atol": 1e-6,
        "evaluate_every": interval, "max_private_batch_nodes": mechanism.max_private_batch_nodes,
        **_task_options(task), **_bootstrap(args),
    }
    result["train_metric_evaluated"] = False
    return result, parameters


def _progap(args, split, task, batch, delta):
    from src.experiments.upstream import UpstreamBaseline

    source = REPO_ROOT / "third_party" / "ProGAP"
    if not (source / "inductive_adapter.py").is_file():
        raise FileNotFoundError(f"ProGAP partition adapter is absent: {source}")
    options = {
        "target_epsilon": args.epsilon, "target_delta": delta,
        "epochs": args.epochs, "batch_size": batch, "hidden_dim": args.gnn_hidden,
        "learning_rate": args.lr, "dropout": args.dropout, "multilabel": task["multilabel"],
        "depth": 2, "max_degree": 5, "max_grad_norm": 1.0,
        "optimizer": "adam", "weight_decay": 0.0, "eval_chunk_size": 16384,
    }
    config = {
        "source_dir": str(source), "command": [args.progap_python, "inductive_adapter.py"],
        "environment": {"PROGAP_DEVICE": args.device, "PROGAP_VERBOSE": "0"},
        "parameters": options, "seed": args.seed, **_bootstrap(args),
    }
    result = UpstreamBaseline("progap", config).run(split)
    population = int(split.train.data.num_nodes)
    interval = population // batch
    parameters = {
        **options, "stages": 3, "epochs_total": 3 * args.epochs,
        "steps": 3 * args.epochs * interval, "evaluate_every": interval,
        "epoch_schedule": "native_drop_last_per_stage",
        "accounted_sgd_steps_per_stage": args.epochs * population // batch,
        "base_layers": 1, "head_layers": 1, "activation": "selu", "jk": "cat",
        "batch_norm": True, "layerwise": False,
        "normalization": "upstream_ModuleValidator.fix",
        "progap_python": args.progap_python, **_bootstrap(args),
    }
    return result, parameters


def _json_value(value: Any) -> Any:
    """Keep native JSON metadata, omitting model/tensor/opaque runtime objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            try:
                result[str(key)] = _json_value(child)
            except TypeError:
                continue
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    # Native NumPy scalar metadata can occur in privacy accountants.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(f"nonserializable runtime value: {type(value).__name__}")


def _metric_value(result, metric, *, validation=False):
    prefix = "validation" if validation else "test"
    keys = [f"{prefix}_{metric}"]
    if metric != "auroc":
        # Existing first-party trainers and ProGAP use this legacy container for
        # regression R2/multilabel micro-F1. The resolved dataset names the value.
        keys.append(f"{prefix}_accuracy")
    keys.append("val" if validation else "test")
    for key in keys:
        if key in result:
            return result[key]
    raise ValueError(f"backend result is missing the resolved {prefix} {metric} score")


def _privacy_pair(result, private, target, delta):
    if not private:
        return None, None
    privacy = result.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("private backend omitted its actual accounting result")
    total = privacy.get("total", privacy)
    if not isinstance(total, dict):
        raise ValueError("private backend omitted its composed accounting result")
    actual_epsilon = float(total["epsilon"])
    actual_delta = float(total["delta"])
    if not math.isfinite(actual_epsilon) or actual_epsilon < 0 or actual_epsilon > target + 1e-6:
        raise RuntimeError(f"actual epsilon {actual_epsilon} exceeds or invalidates target {target}")
    if not math.isclose(actual_delta, delta, rel_tol=1e-10, abs_tol=0.0):
        raise RuntimeError(f"backend delta {actual_delta} differs from common delta {delta}")
    return actual_epsilon, actual_delta


def run(args: argparse.Namespace) -> dict[str, Any]:
    worker_started = time.perf_counter()
    binding = None
    if args.campaign_request is not None:
        from scripts.full_matrix_records import validate_worker_request
        binding = validate_worker_request(args)
    import torch

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this campaign supports explicit cpu or cuda devices only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested device {args.device} is unavailable; CPU fallback is disabled")
        if device.index is not None:
            torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    args.device = str(device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_started = time.perf_counter()
    if binding is not None:
        from scripts.full_matrix_records import load_prepared_protocol
        dataset, split, task, strategy = load_prepared_protocol(args.prepared_protocol)
    else:
        from scripts.full_matrix_runtime import file_lock, json_hash
        with file_lock(args.split_root / f".load-{json_hash(args.dataset)[:16]}.lock"):
            dataset, split, task, strategy = _load_split(args.dataset, args.split_root)
    loading_seconds = time.perf_counter() - load_started
    population = int(split.train.data.num_nodes)
    batch = min(args.batch_size, population)
    delta = 1.0 / population
    interval = math.ceil(population / batch)
    if device.type == "cuda":
        # The allocator peak covers the whole backend, including calibration,
        # training, checkpoint selection and final evaluation, but not loading.
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.method in {"mlp", "dp_mlp", "graphsage", "gin"}:
        native, parameters = _baseline(args, split, task, batch, delta)
    elif args.method == "dpar":
        native, parameters = _dpar(args, split, task, batch, delta)
    elif args.method.startswith("dp_gnn_"):
        native, parameters = _dpgnn(args, split, task, batch, delta)
    elif args.method.startswith("sparse_"):
        native, parameters = _sparse(args, split, task, batch, delta)
    else:
        native, parameters = _progap(args, split, task, batch, delta)
    duration = time.perf_counter() - started
    resources = {
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
        "peak_cuda_allocated_scope": (
            "runner-process PyTorch allocator on the selected device from backend start "
            "through final evaluation; excludes child processes and non-PyTorch allocations"),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                          * (1 if sys.platform == "darwin" else 1024),
        "peak_rss_scope": (
            "runner-process lifetime high-water RSS through backend completion, including "
            "data loading; excludes child processes"),
    }
    private = args.method not in NONPRIVATE
    actual_epsilon, actual_delta = _privacy_pair(native, private, args.epsilon, delta)
    metric = task["primary_metric"]
    test = _metric_value(native, metric)
    if "selection" not in native:
        raise ValueError(f"{args.method}: backend omitted validation checkpoint selection")
    selection = {**native["selection"], "split": "validation", "early_stopping": False,
                 "epochs_requested": args.epochs}
    # Report the score that selected this checkpoint, not a re-evaluation whose
    # GPU reduction order can perturb ranking metrics at floating-point ties.
    validation = float(selection["validation_score"])
    selection.setdefault("epochs_completed", args.epochs)
    if args.method == "progap":
        selection.update(stages=3, epochs_per_stage=args.epochs, final_test_stage=2)
    steps = parameters.get("steps", args.epochs * interval)
    parameters.setdefault("steps", steps)
    parameters.setdefault("evaluate_every", interval)
    identity = {
        "dataset": dataset, "protocol": args.dataset, "method": args.method,
        "architecture": "mlp" if args.method in {"mlp", "dp_mlp"} else
                        "gin" if args.method.endswith("gin") else
                        "graphsage" if args.method in {"graphsage", "dp_gnn_sage", "sparse_sage"} else args.method,
        "dp": private, "target_epsilon": args.epsilon, "target_delta": delta if private else None,
        "epsilon": actual_epsilon, "delta": actual_delta, "metric": metric,
        "seed": args.seed, "lr": args.lr, "batch_size": batch,
        "requested_batch_size": args.batch_size, "epochs": args.epochs,
        "effective_batch_size": parameters.get("effective_batch_size", batch),
        "hidden": args.mlp_hidden if args.method in {"mlp", "dp_mlp"} else args.gnn_hidden,
        "dropout": args.dropout, "device": args.device, "steps": steps,
        "split": f"{strategy}:seed0", "split_strategy": strategy, "split_seed": 0,
        "split_file": str(split.path), "domain_split": split.domain_split,
        "domain_split_id": split.domain_split_id, "train_nodes": population,
    }
    identity["weight_decay"] = parameters["weight_decay"]
    if binding is not None:
        identity.update({key: binding[key] for key in (
            "request_key", "campaign_manifest_sha256", "prepared_fingerprint",
            "source_fingerprint", "attempt_number")})
    config = {**identity, "parameters": parameters, "task": task,
              "requested": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "epoch_semantics": (
                  "per_stage_native_drop_last" if args.method == "progap" else
                  "released_ppr_root_pass" if args.method == "dpar" else
                  "training_population_expected_pass"),
              "partitions": {name: getattr(split, name).stats for name in ("train", "val", "test")}}
    result = {
        "native_result": native, **identity, "parameters": parameters,
        "test_metric": test, f"test_{metric}": test, "validation_metric": validation,
        "selection": selection, "test_confidence_intervals": native.get("test_confidence_intervals", {}),
        "status": "completed", "completed_epochs": args.epochs,
        "calibration_and_training_seconds": duration,
        "loading_seconds": loading_seconds,
        "worker_wall_seconds": time.perf_counter() - worker_started,
        "native_timing": native.get("timing"),
        "resources": resources,
    }
    # Only one canonical hyperparameter mapping enters the summary parser. The
    # native config stays nested in result.json rather than conflicting aliases.
    row = {**identity, "status": "completed", "test_metric": test,
           f"test_{metric}": test, "validation_metric": validation,
           "parameters": parameters, "selection": selection,
           "test_confidence_intervals": native.get("test_confidence_intervals", {}),
           "completed_epochs": args.epochs}
    config, result, row = map(_json_value, (config, result, row))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary = args.out_dir / "result.csv.partial"
    with temporary.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                         if isinstance(value, (dict, list)) else value for key, value in row.items()})
    temporary.replace(args.out_dir / "result.csv")
    return {"output": str(args.out_dir / "result.csv"), "metric": metric,
            "test_metric": result["test_metric"], "epsilon": actual_epsilon, "delta": actual_delta,
            "request_key": identity.get("request_key"),
            "campaign_manifest_sha256": identity.get("campaign_manifest_sha256")}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    created = False
    try:
        _check_args(args)
        # mkdir is exclusive even if another worker races the initial guard.
        args.out_dir.mkdir(parents=True, exist_ok=False)
        created = True
        os.chdir(REPO_ROOT)
        summary = run(args)
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        from scripts.full_matrix_runtime import atomic_json, sha256, utc_now
        atomic_json(args.out_dir / "worker_exit.json", {
            "status": "completed", "request_key": summary["request_key"],
            "campaign_manifest_sha256": summary["campaign_manifest_sha256"],
            "completed_utc": utc_now(),
            "artifact_sha256": {name: sha256(args.out_dir / name)
                               for name in ("config.json", "result.json", "result.csv")},
        })
    except Exception as error:
        import traceback
        traceback.print_exc()
        cuda_oom = type(error).__name__ == "OutOfMemoryError"
        host_oom = isinstance(error, MemoryError)
        kind = "cuda_oom" if cuda_oom else "host_oom" if host_oom else "runtime_error"
        if created:
            try:
                from scripts.full_matrix_runtime import atomic_json
                atomic_json(args.out_dir / "worker_error.json", {
                    "kind": kind, "exception_class": type(error).__name__,
                    "message": str(error),
                })
            except OSError as record_error:
                print(f"Unable to publish worker error: {record_error}", file=sys.stderr)
        return 86 if cuda_oom or host_oom else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
