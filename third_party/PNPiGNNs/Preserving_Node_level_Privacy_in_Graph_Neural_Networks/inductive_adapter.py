"""Run HeterPoisson on graph-disjoint manifest partitions only."""
import json
import math
import os
from argparse import Namespace
from copy import deepcopy
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


def _flag(name):
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


class _MeanSquaredError(torch.nn.Module):
    """MSE that flattens both sides before reducing.

    train_scheduler calls the criterion two ways: ``criterion(pred, target)``
    with pred [1, C] / target [1] in the per-example gradient path
    (compute_loss), and pred [B, C] / target [B] in the batched path
    (manual_forward).  With a single output C == 1, torch.nn.MSELoss would
    BROADCAST [B, 1] against [B] into a [B, B] matrix and silently return the
    mean of every cross-pair -- a wrong loss, not an error.  Flattening first
    keeps both call sites correct.  This is injected as `criterion`, which
    upstream already accepts as a constructor argument, so no upstream file is
    modified to train on a continuous target.
    """

    def forward(self, prediction, target):
        return torch.nn.functional.mse_loss(
            prediction.reshape(-1), target.reshape(-1).to(prediction.dtype))


@torch.no_grad()
def _mean_absolute_error(scheduler, loader):
    """MAE over a loader, matching upstream's center-node convention.

    train_scheduler.one_epoch scores the ROOT only -- ``out[:, 0, :]`` against
    ``targets[:, 0]`` -- so evaluation has to do the same, or the sampled
    neighbours would be scored as if they were prediction targets.

    G_net.forward standardizes over its whole input tensor, and upstream
    applies it under vmap (one rooted subgraph at a time), so a single batched
    forward would share statistics across the batch and change the model's
    output.  Evaluate per sample to keep it numerically identical.
    """
    model = scheduler.model
    model.eval()
    absolute_error, count = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        x, targets = batch
        x, targets = x.to(scheduler.device), targets.to(scheduler.device)
        predictions = torch.stack([model(sample) for sample in x])[:, 0, 0]
        truth = targets[:, 0].reshape(-1).to(predictions.dtype)
        absolute_error += float((predictions - truth).abs().sum())
        count += int(truth.numel())
    return absolute_error / max(count, 1)


def _train_regression(scheduler, epochs):
    """Train for `epochs` and select the checkpoint on validation MAE.

    Upstream's trainer.run() selects on ``hit_accuracy``, which a one-output
    regressor cannot produce: argmax over a [B, 1] logit is identically 0, so
    hit_accuracy is a constant and ``> best_validation_accuracy`` fires only on
    the first epoch -- run() would silently return the EPOCH-ZERO weights.  So
    drive one_epoch directly and keep the best state here instead.

    The privacy accounting is untouched: trainer.__init__ fixes `std` and
    `achieved_epsilon` from `steps` before any training happens, and this runs
    the same args.epoch epochs over the same train_loader, so the composition
    count is identical to the classification path.
    """
    best_state, best_validation = None, float("inf")
    for epoch in range(epochs):
        scheduler.epoch = epoch
        scheduler.one_epoch(train_or_val=train_scheduler.Phase.TRAIN,
                            loader=scheduler.train_loader)
        validation = _mean_absolute_error(scheduler, scheduler.val_loader)
        if validation < best_validation:
            best_validation = validation
            best_state = deepcopy(scheduler.model.state_dict())
    if best_state is not None:
        scheduler.model.load_state_dict(best_state)
        # Mirror run()'s restore: the vmapped worker parameters are a separate
        # copy and must be re-synced, or evaluation would use the last epoch's
        # weights rather than the selected ones.
        for p_model, p_worker in zip(scheduler.model.parameters(),
                                     scheduler.worker_param_func):
            p_worker.copy_(p_model.data)
    return best_validation, _mean_absolute_error(scheduler, scheduler.test_loader)


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
        neighbor_num_constrain_for_training_for_memory=500,
        out_degree_inverse=sampling.compute_in_degree_inverse(
            data.edge_index, data.num_nodes, cache_dir / f"{name}-in-degree-inverse.pt",
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
    if "HETERPOISSON_DEGREE_BOUND" in os.environ:
        raise ValueError(
            "HETERPOISSON_DEGREE_BOUND is retired; the bound is derived from the training population"
        )
    manifest = Path(os.environ["PARTITION_MANIFEST"])
    result_path = Path(os.environ["RESULT_PATH"])
    epsilon, delta = _target_pair()
    epochs = _positive_int("HETERPOISSON_EPOCHS")
    expected_batchsize = _positive_int("HETERPOISSON_EXPECTED_BATCHSIZE")
    K = _positive_int("HETERPOISSON_K")
    num_neighbors = _positive_int("HETERPOISSON_NUM_NEIGHBORS")
    clip_norm = _positive_float("HETERPOISSON_CLIP_NORM")
    learning_rate = _positive_float("HETERPOISSON_LEARNING_RATE")
    seed = int(os.environ.get("HETERPOISSON_SEED", "0"))
    device = os.environ.get("HETERPOISSON_DEVICE", "cpu")
    regression = _flag("HETERPOISSON_REGRESSION")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train, val, test = _normalize(_load(manifest, "train"), _load(manifest, "val"), _load(manifest, "test"))
    if expected_batchsize > train.num_nodes:
        raise ValueError("expected_batchsize must not exceed train node count")
    degree_bound = int(train.num_nodes)
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
        # A continuous target is one unbounded output, not a class count.
        # num_classes reaches only G_net's final Linear; the privacy
        # calibration (get_std_node_dp) is driven by q, steps, degree bound and
        # num_neighbors, so this does not move epsilon.
        num_classes=1 if regression else int(train.y.max()) + 1,
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
        criterion=_MeanSquaredError() if regression else model_module.criterion,
        args=args,
        target_delta=delta,
        degree_bound=degree_bound,
        steps=steps,
    )
    if regression:
        # MAE in both slots, matching src.models.objectives._regression_mae:
        # neither "accuracy" nor "macro-F1" means anything for a continuous
        # target, and reporting one number twice is better than inventing a
        # second one.
        validation_score, test_score = _train_regression(scheduler, epochs)
        validation_secondary, test_secondary = validation_score, test_score
    else:
        validation_metrics, test_metrics = scheduler.run()
        validation_score = float(validation_metrics.hit_accuracy)
        test_score = float(test_metrics.hit_accuracy)
        validation_secondary = float(validation_metrics.mean_f1_s)
        test_secondary = float(test_metrics.mean_f1_s)
    achieved_epsilon = float(scheduler.achieved_epsilon)
    if achieved_epsilon > epsilon:
        raise RuntimeError(f"HeterPoisson calibration exceeded target epsilon: {achieved_epsilon} > {epsilon}")
    # The accuracy/macro_f1 slots carry whatever metric the task defines --
    # MAE for regression, where LOWER is better.  `metric` names it; the slot
    # reuse matches how the multilabel and RelBench paths already report.
    result = {
        "metric": "mae" if regression else "accuracy",
        "validation_accuracy": validation_score,
        "validation_macro_f1": validation_secondary,
        "test_accuracy": test_score,
        "test_macro_f1": test_secondary,
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
