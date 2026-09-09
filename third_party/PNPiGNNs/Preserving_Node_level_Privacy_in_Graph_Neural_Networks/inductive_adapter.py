"""Run HeterPoisson on graph-disjoint manifest partitions only."""
import json
import math
import os
from argparse import Namespace
from pathlib import Path
import random

import numpy as np
import torch

import datasets.model as model_module
import train_scheduler
from privacy import sampling


def _load(manifest, name):
    entry = json.loads(Path(manifest).read_text())["partitions"][name]
    return torch.load(Path(manifest).parent / entry, map_location="cpu", weights_only=False)["data"]


def _target_pair():
    try:
        epsilon = float(os.environ["HETERPOISSON_TARGET_EPSILON"])
        delta = float(os.environ["HETERPOISSON_TARGET_DELTA"])
    except KeyError as error:
        raise ValueError("HETERPOISSON_TARGET_EPSILON and HETERPOISSON_TARGET_DELTA are required") from error
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("HeterPoisson target epsilon must be positive and target delta must be in (0, 1)")
    return epsilon, delta


def _positive_int(name):
    value = int(os.environ[name])
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name):
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _normalize(train, *held_out):
    mean = train.x.mean(dim=0, keepdim=True)
    std = train.x.std(dim=0, keepdim=True).clamp_min(1e-12)
    for data in (train, *held_out):
        data.x = (data.x - mean) / std
    return train, *held_out


def _sampler(data, *, name, mode, args, cache_dir):
    return sampling.subgraph_sampler(
        K=args.K,
        num_neighbors=args.num_neighbors,
        neighbor_num_constrain_for_training_for_memory=args.num_neighbors,
        out_degree_inverse=sampling.compute_out_degree_inverse(
            data.edge_index, data.num_nodes, cache_dir / f"{name}-degree-inverse.pt",
        ),
        graph_data=data,
        graph_data_name=name,
        mask=torch.ones(data.num_nodes, dtype=torch.bool),
        setting="inductive",
        dataset_mode=mode,
        device=args.device,
        args=args,
        cache_file_path=cache_dir / name,
    )


def main():
    manifest = Path(os.environ["PARTITION_MANIFEST"])
    result_path = Path(os.environ["RESULT_PATH"])
    epsilon, delta = _target_pair()
    epochs = _positive_int("HETERPOISSON_EPOCHS")
    expected_batchsize = _positive_int("HETERPOISSON_EXPECTED_BATCHSIZE")
    K = _positive_int("HETERPOISSON_K")
    num_neighbors = _positive_int("HETERPOISSON_NUM_NEIGHBORS")
    degree_bound = _positive_int("HETERPOISSON_DEGREE_BOUND")
    clip_norm = _positive_float("HETERPOISSON_CLIP_NORM")
    learning_rate = _positive_float("HETERPOISSON_LEARNING_RATE")
    seed = int(os.environ.get("HETERPOISSON_SEED", "0"))
    device = os.environ.get("HETERPOISSON_DEVICE", "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train, val, test = _normalize(_load(manifest, "train"), _load(manifest, "val"), _load(manifest, "test"))
    if expected_batchsize > train.num_nodes:
        raise ValueError("expected_batchsize must not exceed train node count")
    q = expected_batchsize / train.num_nodes
    if q * num_neighbors / degree_bound > 1:
        raise ValueError("expected_batchsize / train_nodes * num_neighbors / degree_bound must not exceed one")
    rounds_per_epoch = math.ceil(train.num_nodes / expected_batchsize)
    steps = epochs * rounds_per_epoch
    cache_dir = manifest.parent / "heterpoisson-neighbor-cache"
    args = Namespace(
        expected_batchsize=expected_batchsize,
        epoch=epochs,
        K=K,
        num_neighbors=num_neighbors,
        num_classes=int(train.y.max()) + 1,
        priv_epsilon=epsilon,
        C=clip_norm,
        lr=learning_rate,
        seed=seed,
        graph_setting="inductive",
        dataset="manifest",
        log_dir="logs",
        cache_dir=str(cache_dir),
        device=device,
    )
    train_loader = sampling.get_subgraphs_loader(
        _sampler(train, name="train", mode="train", args=args, cache_dir=cache_dir),
        expected_batchsize, worker_num=0, dataset_mode="train",
    )
    val_loader = sampling.get_subgraphs_loader(
        _sampler(val, name="val", mode="val", args=args, cache_dir=cache_dir),
        expected_batchsize, worker_num=0, dataset_mode="val",
    )
    test_loader = sampling.get_subgraphs_loader(
        _sampler(test, name="test", mode="test", args=args, cache_dir=cache_dir),
        expected_batchsize, worker_num=0, dataset_mode="test",
    )
    model = model_module.G_net(K=K, feat_dim=train.x.shape[1], num_classes=args.num_classes, hidden_channels=128).to(device)
    scheduler = train_scheduler.trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        loaders=[train_loader, val_loader, test_loader],
        device=device,
        criterion=model_module.criterion,
        args=args,
        target_delta=delta,
        degree_bound=degree_bound,
        steps=steps,
    )
    validation_metrics, test_metrics = scheduler.run()
    achieved_epsilon = float(scheduler.achieved_epsilon)
    if achieved_epsilon > epsilon:
        raise RuntimeError(f"HeterPoisson calibration exceeded target epsilon: {achieved_epsilon} > {epsilon}")
    result = {
        "validation_accuracy": float(validation_metrics.hit_accuracy),
        "validation_macro_f1": float(validation_metrics.mean_f1_s),
        "test_accuracy": float(test_metrics.hit_accuracy),
        "test_macro_f1": float(test_metrics.mean_f1_s),
        "privacy": {"total": {
            "epsilon": achieved_epsilon,
            "delta": delta,
            "accountant": "pnpignns.heterpoisson",
            "noise_multiplier": float(scheduler.std),
            "sampling_probability": q,
            "composition_count": steps,
            "parameters": {
                "sigma": float(scheduler.std), "q": q, "steps": steps,
                "rounds_per_epoch": rounds_per_epoch, "degree_bound": degree_bound,
                "num_neighbors": num_neighbors, "clip_norm": clip_norm,
                "train_nodes": train.num_nodes,
            },
        }},
        "calibration": {
            "target_epsilon": epsilon, "target_delta": delta,
            "achieved_epsilon": achieved_epsilon, "noise_std": float(scheduler.std),
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
