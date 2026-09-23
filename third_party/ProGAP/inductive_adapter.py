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

from core import console
from core.data.loader.node import NodeDataLoader
from core.methods.progap.node import NodeLevelProGAP
from core.modules.prog import binary_auroc

def _load(manifest_path, manifest, name):
    file = manifest["partitions"][name]
    return torch.load(Path(manifest_path).parent / file, map_location="cpu", weights_only=False)["data"]


def _prepare(data):
    data = ToSparseTensor(layout=torch.sparse_csr)(data)
    all_nodes = torch.ones(data.num_nodes, dtype=torch.bool)
    eval_mask = getattr(data, "eval_mask", all_nodes)
    if (
        not torch.is_tensor(eval_mask)
        or eval_mask.dtype != torch.bool
        or eval_mask.ndim != 1
        or eval_mask.numel() != data.num_nodes
    ):
        raise ValueError("partition eval_mask must be a boolean vector over all nodes")
    data.eval_mask = eval_mask.clone()
    data.train_mask = all_nodes
    data.val_mask = all_nodes.clone()
    data.test_mask = all_nodes.clone()
    return data


def _score_mask(data, metric_ignore_label=None):
    mask = data.eval_mask.clone()
    if metric_ignore_label is not None:
        if data.y.ndim != 1:
            raise ValueError("metric_ignore_label requires one-dimensional labels")
        mask &= data.y != metric_ignore_label
    if not bool(mask.any()):
        raise ValueError("partition has no nodes selected for evaluation")
    return mask


class InductiveNodeLevelProGAP(NodeLevelProGAP):
    """Select each progressive stage on a graph-disjoint validation partition."""

    def fit(self):
        self.data.x0 = self.data.x
        self.validation.x0 = self.validation.x
        metrics = {}
        for stage in range(self.num_stages):
            if stage:
                for graph in (self.data, self.validation):
                    embeddings, _ = self.trainer.predict(
                        dataloader=NodeDataLoader(graph, batch_size="full", shuffle=False)
                    )
                    graph[f"x{stage}"] = self.nap(embeddings, graph.adj_t)

            self.classifier.set_stage(stage)
            console.info(f"Fitting stage {stage + 1} of {self.num_stages}")
            self.trainer = self.configure_trainer()
            metrics = self.trainer.fit(
                model=self.classifier,
                train_dataloader=self.data_loader("train"),
                val_dataloader=NodeDataLoader(
                    self.validation,
                    subset=_score_mask(
                        self.validation, getattr(self, "metric_ignore_label", None)
                    ),
                    batch_size="full",
                    shuffle=False,
                ),
            )

        self.data.ready = True
        return metrics


def _metrics(method, data):
    # Do not call NodeLevelProGAP.setup here: it would recalibrate the private
    # training mechanism from a held-out graph. Prediction stages reuse the
    # trained classifier and upstream NAP/pipeline implementation only.
    data = _prepare(data)
    method.data = method.to_device(Data(**data.to_dict()))
    method.data.ready = False
    probabilities = method.predict()[1].cpu()
    target = data.y.cpu()
    mask = _score_mask(data, getattr(method, "metric_ignore_label", None)).cpu()
    probabilities = probabilities[mask]
    target = target[mask]
    if getattr(method.classifier, "binary", False):
        scores = probabilities.squeeze(-1)
        if scores.ndim != 1:
            raise ValueError("binary ProGAP predictions must have one output per node")
        auroc = float(binary_auroc(scores, target))
        return auroc, auroc
    if target.ndim == 2:
        positive = probabilities >= 0.5
        actual = target.bool()
        denominator = int(positive.sum() + actual.sum())
        micro_f1 = (
            0.0
            if denominator == 0
            else 2.0 * int((positive & actual).sum()) / denominator
        )
        return micro_f1, micro_f1

    prediction = probabilities.argmax(dim=-1)
    accuracy = float((prediction == target).float().mean())
    f1_scores = []
    for label in torch.unique(target):
        predicted = prediction == label
        actual = target == label
        denominator = int(predicted.sum() + actual.sum())
        f1_scores.append(
            0.0
            if denominator == 0
            else 2.0 * int((predicted & actual).sum()) / denominator
        )
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


def _multilabel():
    value = os.environ.get("PROGAP_MULTILABEL")
    if value not in {"0", "1"}:
        raise ValueError("PROGAP_MULTILABEL must be exactly '0' or '1'")
    return value == "1"


def _seed():
    seed = int(os.environ.get("PROGAP_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _flag(name):
    value = os.environ.get(name)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be exactly '0' or '1'")
    return value == "1"


def _task_metadata(manifest):
    binary = _flag("PROGAP_BINARY")
    manifest_binary = bool(manifest.get("binary", False))
    if int(manifest.get("format", 1)) >= 2 and binary != manifest_binary:
        raise ValueError("PROGAP_BINARY does not match the partition manifest")

    manifest_metric = manifest.get("primary_metric")
    primary_metric = os.environ.get("PROGAP_PRIMARY_METRIC", manifest_metric)
    if primary_metric is None:
        primary_metric = "auroc" if binary else "accuracy"
    if manifest_metric is not None and primary_metric != manifest_metric:
        raise ValueError("PROGAP_PRIMARY_METRIC does not match the partition manifest")
    if binary and primary_metric != "auroc":
        raise ValueError("binary ProGAP requires primary metric 'auroc'")

    manifest_ignore = manifest.get("metric_ignore_label")
    configured_ignore = os.environ.get("PROGAP_METRIC_IGNORE_LABEL")
    metric_ignore_label = (
        int(configured_ignore) if configured_ignore is not None else manifest_ignore
    )
    if manifest_ignore is not None and metric_ignore_label != int(manifest_ignore):
        raise ValueError(
            "PROGAP_METRIC_IGNORE_LABEL does not match the partition manifest"
        )
    return binary, primary_metric, metric_ignore_label
def main():
    manifest_path = os.environ["PARTITION_MANIFEST"]
    manifest = json.loads(Path(manifest_path).read_text())
    result_path = Path(os.environ["RESULT_PATH"])
    epsilon, delta = _target_pair()
    multilabel = _multilabel()
    binary, primary_metric, metric_ignore_label = _task_metadata(manifest)
    if binary and multilabel:
        raise ValueError("PROGAP_BINARY and PROGAP_MULTILABEL cannot both be enabled")
    _seed()
    epochs = _positive_int("PROGAP_EPOCHS", 1)
    batch_size = _positive_int("PROGAP_BATCH_SIZE", 32)
    max_degree = _positive_int("PROGAP_MAX_DEGREE", 5)


    depth = _positive_int("PROGAP_DEPTH", 1)
    verbose = os.environ.get("PROGAP_VERBOSE", "0") == "1"
    device = os.environ.get("PROGAP_DEVICE", "cpu")
    train = _prepare(_load(manifest_path, manifest, "train"))
    validation = _prepare(_load(manifest_path, manifest, "val"))
    if multilabel != (train.y.ndim == 2):
        raise ValueError("PROGAP_MULTILABEL does not match the training label rank")
    if binary and train.y.ndim != 1:
        raise ValueError("binary ProGAP requires one-dimensional training labels")
    # Opacus 1.1.3 clones modules with torch.load; newer PyTorch defaults to
    # weights-only deserialization. This is an in-process compatibility bridge
    # for a freshly constructed, trusted module—not an algorithm change.
    load = torch.load
    torch.load = lambda *args, **kwargs: load(*args, **{**kwargs, "weights_only": False})
    try:
        method = InductiveNodeLevelProGAP(
            num_classes=1 if binary else int(manifest["num_classes"]),
            epsilon=epsilon,
            delta=delta,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            verbose=verbose,
            max_degree=max_degree,
            depth=depth,
            monitor=(
                "val/auroc"
                if binary
                else "val/micro_f1" if multilabel else "val/acc"
            ),
        )
    finally:
        torch.load = load
    method.classifier.multilabel = multilabel
    method.classifier.binary = binary
    method.metric_ignore_label = metric_ignore_label
    method.validation = method.to_device(Data(**validation.to_dict()))
    method.setup(train)
    method.fit()
    achieved_epsilon = float(method.composed_mechanism.get_approxDP(method.effective_delta))
    if achieved_epsilon > epsilon:
        raise RuntimeError(
            f"ProGAP calibration exceeded target epsilon: {achieved_epsilon} > {epsilon}"
        )
    validation_primary, validation_macro_f1 = _metrics(
        method, _load(manifest_path, manifest, "val")
    )
    test_primary, test_macro_f1 = _metrics(
        method, _load(manifest_path, manifest, "test")
    )
    coefficients = list(method.composed_mechanism.params["coeff_list"])
    result = {
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
    if binary:
        result.update({
            "metric": "auroc",
            "validation_auroc": validation_primary,
            "test_auroc": test_primary,
        })
    else:
        result.update({
            "validation_accuracy": validation_primary,
            "validation_macro_f1": validation_macro_f1,
            "test_accuracy": test_primary,
            "test_macro_f1": test_macro_f1,
        })
    result_path.write_text(json.dumps(result, indent=2) + "\n")




if __name__ == "__main__":
    main()
