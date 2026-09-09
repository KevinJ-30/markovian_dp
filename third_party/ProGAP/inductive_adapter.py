"""Run released ProGAP on graph-disjoint partitions.

Only the data boundary is new: core/methods/progap, NAP, NoisySGD, and their
privacy calibration are imported unchanged from this checkout.
"""
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from core.methods.progap.node import NodeLevelProGAP


def _load(manifest, name):
    file = json.loads(Path(manifest).read_text())["partitions"][name]
    return torch.load(Path(manifest).parent / file, map_location="cpu", weights_only=False)["data"]


def _prepare(data):
    data = ToSparseTensor(layout=torch.sparse_csr)(data)
    all_nodes = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = all_nodes
    data.val_mask = all_nodes.clone()
    data.test_mask = all_nodes.clone()
    return data


def _metrics(method, data):
    # Do not call NodeLevelProGAP.setup here: it would recalibrate the private
    # training mechanism from a held-out graph. Prediction stages reuse the
    # trained classifier and upstream NAP/pipeline implementation only.
    data = _prepare(data)
    method.data = method.to_device(Data(**data.to_dict()))
    method.data.ready = False
    prediction = method.predict()[0].argmax(dim=-1).cpu()
    target = data.y.cpu()
    accuracy = float((prediction == target).float().mean())
    f1_scores = []
    for label in torch.unique(target):
        predicted = prediction == label
        actual = target == label
        denominator = int(predicted.sum() + actual.sum())
        f1_scores.append(0.0 if denominator == 0 else 2.0 * int((predicted & actual).sum()) / denominator)
    return accuracy, float(sum(f1_scores) / len(f1_scores))


def _target_pair():
    try:
        epsilon = float(os.environ["PROGAP_TARGET_EPSILON"])
        delta = float(os.environ["PROGAP_TARGET_DELTA"])
    except KeyError as error:
        raise ValueError("PROGAP_TARGET_EPSILON and PROGAP_TARGET_DELTA are required") from error
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("ProGAP target epsilon must be positive and target delta must be in (0, 1)")
    return epsilon, delta


def _positive_int(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _seed():
    seed = int(os.environ.get("PROGAP_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    manifest = os.environ["PARTITION_MANIFEST"]
    result_path = Path(os.environ["RESULT_PATH"])
    epsilon, delta = _target_pair()
    _seed()
    epochs = _positive_int("PROGAP_EPOCHS", 1)
    batch_size = _positive_int("PROGAP_BATCH_SIZE", 32)
    max_degree = _positive_int("PROGAP_MAX_DEGREE", 5)
    depth = _positive_int("PROGAP_DEPTH", 1)
    verbose = os.environ.get("PROGAP_VERBOSE", "0") == "1"
    device = os.environ.get("PROGAP_DEVICE", "cpu")
    train = _prepare(_load(manifest, "train"))
    # Opacus 1.1.3 clones modules with torch.load; newer PyTorch defaults to
    # weights-only deserialization. This is an in-process compatibility bridge
    # for a freshly constructed, trusted module—not an algorithm change.
    load = torch.load
    torch.load = lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False})
    try:
        method = NodeLevelProGAP(
            num_classes=int(train.y.max()) + 1,
            epsilon=epsilon,
            delta=delta,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            verbose=verbose,
            max_degree=max_degree,
            depth=depth,
        )
    finally:
        torch.load = load
    method.run(train)
    achieved_epsilon = float(method.composed_mechanism.get_approxDP(method.effective_delta))
    if achieved_epsilon > epsilon:
        raise RuntimeError(
            f"ProGAP calibration exceeded target epsilon: {achieved_epsilon} > {epsilon}"
        )
    validation_accuracy, validation_macro_f1 = _metrics(method, _load(manifest, "val"))
    test_accuracy, test_macro_f1 = _metrics(method, _load(manifest, "test"))
    coefficients = list(method.composed_mechanism.params["coeff_list"])
    result = {
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "privacy": {
            "total": {
                "epsilon": achieved_epsilon,
                "delta": delta,
                "accountant": "upstream.ProGAP.ComposedNoisyMechanism",
                "noise_multiplier": method.noise_scale,
                "sampling_probability": batch_size / train.num_nodes,
                "composition_count": int(sum(coefficients)),
                "parameters": {
                    "depth": depth,
                    "max_degree": max_degree,
                    "batch_size": batch_size,
                    "train_nodes": train.num_nodes,
                    "component_coefficients": coefficients,
                    "effective_delta": method.effective_delta,
                },
            },
        },
        "calibration": {
            "target_epsilon": epsilon,
            "target_delta": delta,
            "achieved_epsilon": achieved_epsilon,
            "noise_std": method.noise_scale,
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")




if __name__ == "__main__":
    main()
