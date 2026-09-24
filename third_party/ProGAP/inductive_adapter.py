"""Graph-disjoint ProGAP boundary with exact, chunked global evaluation.

The classifier objective and predictions adapt to categorical, binary, multilabel,
or scalar regression tasks. NAP, private optimization and composed calibration
remain unchanged. Evaluation chunks only nodewise work, never graph context.
"""
import json
from contextlib import nullcontext
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.bootstrap import BootstrapConfig, BootstrapMetrics

from core import console
from core.data.loader.node import NodeDataLoader
from core.methods.progap.node import NodeLevelProGAP
from core.data.transforms.bound_degree import BoundOutDegree
from core.methods.progap.base import ProGAP
from core.modules.prog import RegressionR2, binary_auroc

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


@torch.no_grad()
def stage_embeddings(model, data, chunk_size):
    """Retain every context embedding without an edge-by-feature expansion."""
    model.eval()
    device = next(model.parameters()).device
    output = None
    for start in range(0, data.num_nodes, chunk_size):
        stop = min(start + chunk_size, data.num_nodes)
        xs = [data[f"x{i}"][start:stop].to(device)
              for i in range(model.current_stage + 1)]
        embeddings, _ = model(xs)
        if output is None:
            output = torch.empty(
                (data.num_nodes, embeddings.shape[1]),
                dtype=embeddings.dtype, device=data.x.device,
            )
        output[start:stop].copy_(embeddings)
    if output is None:
        raise ValueError("ProGAP requires a nonempty context graph")
    return output


@torch.no_grad()
def evaluate_stage(
    model, data, chunk_size=16384, metric_ignore_label=None, *, bootstrap=None,
):
    """Reduce a whole-split score, never an average of chunk-level metrics."""
    model.eval()
    device = next(model.parameters()).device
    mask = _score_mask(data, metric_ignore_label)
    binary = getattr(model, "binary", False)
    regression = getattr(model, "regression", False)
    multilabel = data.y.ndim == 2
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    predicted_counts = actual_counts = true_counts = None
    scores, targets = [], []
    count = 0
    regression_metric = RegressionR2() if regression else None
    if regression and data.y.ndim != 1:
        raise ValueError("regression ProGAP requires one-dimensional targets")
    for start in range(0, data.num_nodes, chunk_size):
        stop = min(start + chunk_size, data.num_nodes)
        selected = mask[start:stop]
        if not bool(selected.any()):
            continue
        xs = [data[f"x{i}"][start:stop][selected].to(device)
              for i in range(model.current_stage + 1)]
        logits = model(xs)[1]
        labels = data.y[start:stop][selected].to(device)
        count += labels.shape[0]
        if regression:
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError("regression ProGAP requires exactly one output")
            regression_metric.update(logits, labels)
        elif binary:
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ValueError("binary ProGAP requires exactly one logit")
            logits = logits.squeeze(-1)
            totals[0] += F.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction="sum"
            ).double()
            totals[1] += ((logits >= 0) == labels.bool()).sum()
            scores.append(logits.sigmoid().cpu())
            targets.append(labels.cpu())
        elif multilabel:
            totals[0] += model.root_losses(logits, labels, regression=regression).double().sum()
            positive, actual = logits >= 0, labels.bool()
            totals[1] += (positive & actual).sum()
            totals[2] += positive.sum()
            totals[3] += actual.sum()
        else:
            totals[0] += model.root_losses(logits, labels, regression=regression).double().sum()
            prediction = logits.argmax(dim=-1)
            correct = prediction == labels
            totals[1] += correct.sum()
            if predicted_counts is None:
                predicted_counts = torch.zeros(logits.shape[1], dtype=torch.long, device=device)
                actual_counts = torch.zeros_like(predicted_counts)
                true_counts = torch.zeros_like(predicted_counts)
            predicted_counts += torch.bincount(prediction, minlength=logits.shape[1])
            actual_counts += torch.bincount(labels, minlength=logits.shape[1])
            true_counts += torch.bincount(labels[correct], minlength=logits.shape[1])
        if bootstrap is not None:
            # Binary output publishes AUROC only, using these sigmoid-score ties.
            bootstrap.update(
                scores[-1] if binary else logits,
                targets[-1] if binary else labels,
            )
    if regression:
        totals[0] = regression_metric.residual_sum
    values = totals.cpu().tolist()
    result = {"loss": values[0] / count, "scored_nodes": count}
    if regression:
        result.update(metric="r2", score=float(regression_metric.compute()))
    elif binary:
        result.update(metric="auroc", score=float(binary_auroc(torch.cat(scores), torch.cat(targets))),
                      accuracy=values[1] / count)
    elif multilabel:
        denominator = values[2] + values[3]
        result.update(metric="micro_f1", score=2 * values[1] / denominator if denominator else 0.0)
    else:
        present = actual_counts > 0
        macro = (2 * true_counts[present].double()
                 / (predicted_counts[present] + actual_counts[present])).mean()
        result.update(metric="accuracy", score=values[1] / count, macro_f1=float(macro))
    if not math.isfinite(result["score"]) or not math.isfinite(result["loss"]):
        raise ValueError("ProGAP evaluation produced a nonfinite metric or loss")
    return result

def _cpu_copy(value):
    if isinstance(value, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)):
        return deepcopy(value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return deepcopy(value)


def _load_selected_state(model, state):
    # Earlier checkpoints contain genuinely unused lazy stages. Once those stages
    # are initialized, loading an UninitializedParameter into them is invalid.
    initialized = {
        key: value for key, value in state.items()
        if not isinstance(value, (torch.nn.parameter.UninitializedParameter,
                                  torch.nn.parameter.UninitializedBuffer))
    }
    model.load_state_dict(initialized, strict=False)


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


class InductiveNodeLevelProGAP(NodeLevelProGAP):
    """Select each stage using one global held-out metric and first strict max."""

    def __init__(self, *args, eval_chunk_size=16384, **kwargs):
        if isinstance(eval_chunk_size, bool) or int(eval_chunk_size) != eval_chunk_size or eval_chunk_size <= 0:
            raise ValueError("eval_chunk_size must be a positive integer")
        self.eval_chunk_size = int(eval_chunk_size)
        self.attempt = None
        self.history = []
        self.stage_states = []
        self.updates_completed = 0
        self.roots_total = 0
        self.batch_min = None
        self.batch_max = 0
        self.empty_draws = 0
        self.epochs_completed = 0
        self.evaluations_completed = 0
        self.timing = {name: 0.0 for name in (
            "train_update_seconds", "validation_seconds", "checkpoint_io_seconds"
        )}
        super().__init__(*args, **kwargs)

    def _phase(self, name):
        return self.attempt.phase(name) if self.attempt is not None else nullcontext()


    def setup(self, data):
        with self._phase("preprocess"):
            data = BoundOutDegree(self.max_degree)(data)
            ProGAP.setup(self, data)
            num_train_nodes = int(self.data.train_mask.sum())
        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self._progress("calibration")
            with self._phase("calibration"):
                self.calibrate()

    def _sync(self):
        if self.trainer.device.type == "cuda":
            torch.cuda.synchronize(self.trainer.device)

    def _progress(self, phase):
        timing = dict(self.timing)
        if phase == "train_update" and getattr(self, "_train_started", None) is not None:
            timing["train_update_seconds"] += time.monotonic() - self._train_started
        fields = {
            "phase": phase, "updates_completed": self.updates_completed,
            "total_updates": self.num_stages * self.trainer.epochs * (self.num_train_nodes // self.batch_size),
            "epochs_completed": self.epochs_completed,
            "evaluations_completed": self.evaluations_completed,
            "total_evaluations": self.num_stages * self.trainer.epochs + 1,
            "roots_total": self.roots_total, **timing,
        }
        if self.attempt is not None:
            self.attempt.progress(**fields)

    def _checkpoint(self, payload, filename):
        if self.attempt is not None:
            self.attempt.save_checkpoint(payload, filename=filename)
        elif getattr(self, "checkpoint_dir", None) is not None:
            destination = Path(self.checkpoint_dir) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)

    def fit(self):
        self.data.x0 = self.data.x
        self.validation.x0 = self.validation.x
        self.stage_states = []
        for stage in range(self.num_stages):
            self._progress("stage_preprocess")
            if stage:
                # NAP still receives every context row and the original adjacency.
                # Validation can remain CPU-backed; only classifier chunks move.
                for graph in (self.data, self.validation):
                    with self._phase("preprocess"), torch.no_grad():
                        embeddings = stage_embeddings(self.classifier, graph, self.eval_chunk_size)
                        graph[f"x{stage}"] = self.nap(embeddings, graph.adj_t)
                        del embeddings
            self.classifier.set_stage(stage)
            console.info(f"Fitting stage {stage + 1} of {self.num_stages}")
            self.trainer = self.configure_trainer()
            model = self.classifier.to(self.trainer.device)
            self.trainer.model = model
            # Initialize this stage's lazy layers before optimizer/grad sampling.
            with self._phase("preprocess"), torch.no_grad():
                model.eval()
                model([self.data[f"x{i}"][:2] for i in range(stage + 1)])
            optimizer = model.configure_optimizers()
            self.trainer.optimizer = optimizer
            loader = self.data_loader("train")
            if len(loader) == 0:
                raise ValueError("ProGAP drop-last schedule has no updates")
            best = None
            for epoch in range(1, self.trainer.epochs + 1):
                self._sync()
                started = time.monotonic()
                epoch_roots = epoch_updates = 0
                self._train_started = started
                with self._phase("train_update"):
                    model.train()
                    for batch in loader:
                        count = batch.batch_nodes.numel()
                        optimizer.zero_grad(set_to_none=True)
                        if count:
                            xs = [batch[f"x{i}"][batch.batch_nodes] for i in range(stage + 1)]
                            logits = model(xs)[1]
                            labels = batch.y[batch.batch_nodes]
                            if getattr(model, "binary", False):
                                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
                            else:
                                loss = model.root_losses(
                                    logits, labels, regression=model.regression
                                ).mean()
                        else:
                            # Exactly one private noise/Adam update, even on an empty draw.
                            logits = model([self.data[f"x{i}"][:2] for i in range(stage + 1)])[1]
                            loss = logits.sum() * 0
                        loss.backward()
                        optimizer.step()
                        epoch_roots += count
                        epoch_updates += 1
                        self.updates_completed += 1
                        self.roots_total += count
                        self.empty_draws += int(count == 0)
                        self.batch_min = count if self.batch_min is None else min(self.batch_min, count)
                        self.batch_max = max(self.batch_max, count)
                        if self.updates_completed % 20 == 0:
                            # No device synchronization solely for per-update telemetry.
                            self._progress("train_update")
                self._sync()
                self.timing["train_update_seconds"] += time.monotonic() - started
                self._train_started = None
                if epoch_updates != len(loader):
                    raise RuntimeError("ProGAP executed an unexpected number of logical updates")
                started = time.monotonic()
                with self._phase("validation"):
                    validation = evaluate_stage(
                        model, self.validation, self.eval_chunk_size,
                        getattr(self, "metric_ignore_label", None),
                    )
                self._sync()
                self.timing["validation_seconds"] += time.monotonic() - started
                self.evaluations_completed += 1
                self.epochs_completed += 1
                if best is None or validation["score"] > best["validation"]["score"]:
                    started = time.monotonic()
                    with self._phase("checkpoint_io"):
                        best = {
                            "model": _cpu_copy(model.state_dict()),
                            "optimizer": _cpu_copy(optimizer.state_dict()),
                            "rng": _rng_state(), "stage": stage, "epoch": epoch,
                            "step": self.updates_completed, "stage_step": epoch * len(loader),
                            "validation": validation,
                        }
                    self._checkpoint(best, f"stage{stage}_best.pt")
                    self.timing["checkpoint_io_seconds"] += time.monotonic() - started
                row = {
                    "stage": stage, "epoch": epoch, "stage_epochs_completed": self.epochs_completed,
                    "updates_completed": self.updates_completed, "epoch_updates": epoch_updates,
                    "roots_total": self.roots_total, "epoch_roots": epoch_roots,
                    "step": self.updates_completed,
                    "validation_metric": validation["score"], "validation_loss": validation["loss"],
                    "best_epoch": best["epoch"], **self.timing,
                }
                self.history.append(row)
                if self.attempt is not None:
                    self.attempt.save_json("history.json", self.history)
                self._progress("epoch_complete")
            _load_selected_state(model, best["model"])
            self.stage_states.append(best)
        self.data.ready = True
        self.best_validation = self.stage_states[-1]["validation"]
        self._checkpoint({
            **self.stage_states[-1], "stages": self.stage_states,
            "updates_completed": self.updates_completed,
            "epochs_completed": self.epochs_completed,
        }, "checkpoint.pt")
        return {
            self.trainer.monitor: self.best_validation["score"] * (
                1 if self.classifier.regression else 100
            ),
            "epoch": self.stage_states[-1]["epoch"],
        }

    def evaluate_partition(self, data, *, bootstrap=None):
        """Run the upstream final-classifier NAP pipeline with full CPU context."""
        graph = _prepare(Data(**data.to_dict())).cpu()
        graph.x0 = graph.x
        model = self.classifier
        final_state = self.stage_states[-1]["model"]
        # As in ProGAP.pipeline(fit=False), the selected final classifier is
        # shared by every prediction stage; archived stage checkpoints are not
        # an ensemble and must not replace the upstream inference convention.
        _load_selected_state(model, final_state)
        try:
            for stage in range(self.num_stages):
                # Do not call the private set_stage wrapper again: this is inference,
                # not another gradient-sampling registration or privacy calibration.
                model.current_stage = stage
                if stage + 1 < self.num_stages:
                    embeddings = stage_embeddings(model, graph, self.eval_chunk_size)
                    with torch.no_grad():
                        graph[f"x{stage + 1}"] = self.nap(embeddings, graph.adj_t)
                    del embeddings
            return evaluate_stage(
                model, graph, self.eval_chunk_size, getattr(self, "metric_ignore_label", None),
                bootstrap=bootstrap,
            )
        finally:
            model.current_stage = self.num_stages - 1


def _metrics(method, data, *, bootstrap=None):
    if hasattr(method, "evaluate_partition"):
        result = (
            method.evaluate_partition(data)
            if bootstrap is None
            else method.evaluate_partition(data, bootstrap=bootstrap)
        )
        return result["score"], result.get("macro_f1", result["score"])
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
    if bootstrap is not None:
        # The prediction API returns probabilities, not the stage's raw logits.
        bootstrap.update(
            probabilities - 0.5 if target.ndim == 2 else probabilities, target,
        )
    if getattr(method.classifier, "regression", False):
        metric = RegressionR2()
        metric.update(probabilities, target)
        score = float(metric.compute())
        if not math.isfinite(score):
            raise ValueError("ProGAP evaluation produced a nonfinite regression metric")
        return score, score
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


def _constructor_options():
    """Validate optional controls while preserving upstream constructor defaults."""
    options = {}
    for key, environment in {
        "hidden_dim": "PROGAP_HIDDEN_DIM",
        "eval_chunk_size": "PROGAP_EVAL_CHUNK_SIZE",
    }.items():
        if environment in os.environ:
            options[key] = _positive_int(environment, 1)
    for key, environment in {
        "learning_rate": "PROGAP_LEARNING_RATE",
        "max_grad_norm": "PROGAP_MAX_GRAD_NORM",
        "dropout": "PROGAP_DROPOUT",
        "weight_decay": "PROGAP_WEIGHT_DECAY",
    }.items():
        if environment not in os.environ:
            continue
        value = float(os.environ[environment])
        lower_valid = value > 0 if key in {"learning_rate", "max_grad_norm"} else value >= 0
        if not math.isfinite(value) or not lower_valid or (key == "dropout" and value >= 1):
            raise ValueError(f"{environment} is outside its supported finite range")
        options[key] = value
    if "PROGAP_OPTIMIZER" in os.environ:
        optimizer = os.environ["PROGAP_OPTIMIZER"]
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("PROGAP_OPTIMIZER must be 'sgd' or 'adam'")
        options["optimizer"] = optimizer
    return options


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
    if primary_metric not in {"accuracy", "micro_f1", "auroc", "r2"}:
        raise ValueError(f"unsupported ProGAP primary metric: {primary_metric!r}")
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
    bootstrap_config = BootstrapConfig(
        confidence_level=float(os.environ.get("PROGAP_BOOTSTRAP_CONFIDENCE", "0.95")),
        n_resamples=int(os.environ.get("PROGAP_BOOTSTRAP_RESAMPLES", "1000")),
        seed=int(os.environ.get("PROGAP_BOOTSTRAP_SEED", "0")),
    )
    manifest_path = os.environ["PARTITION_MANIFEST"]
    manifest = json.loads(Path(manifest_path).read_text())
    result_path = Path(os.environ["RESULT_PATH"])
    epsilon, delta = _target_pair()
    multilabel = _multilabel()
    binary, primary_metric, metric_ignore_label = _task_metadata(manifest)
    regression = primary_metric == "r2"
    if regression and (binary or multilabel):
        raise ValueError("regression ProGAP cannot be binary or multilabel")
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
            num_classes=1 if binary or regression else int(manifest["num_classes"]),
            epsilon=epsilon,
            delta=delta,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            verbose=verbose,
            max_degree=max_degree,
            depth=depth,
            monitor=(
                "val/r2" if regression else "val/auroc" if binary
                else "val/micro_f1" if multilabel else "val/acc"
            ),
            **_constructor_options(),
        )
    finally:
        torch.load = load
    method.classifier.multilabel = multilabel
    method.classifier.binary = binary
    method.classifier.regression = regression
    method.metric_ignore_label = metric_ignore_label
    method.validation = Data(**validation.to_dict())
    method.checkpoint_dir = result_path.parent
    method.setup(train)
    method.fit()
    achieved_epsilon = float(method.composed_mechanism.get_approxDP(method.effective_delta))
    if not math.isfinite(achieved_epsilon) or achieved_epsilon > epsilon + 1e-6:
        raise RuntimeError(
            f"ProGAP calibration exceeded target epsilon: {achieved_epsilon} > {epsilon}"
        )
    if hasattr(method, "best_validation"):
        validation_primary = method.best_validation["score"]
        validation_macro_f1 = method.best_validation.get("macro_f1", validation_primary)
    else:
        validation_primary, validation_macro_f1 = _metrics(
            method, _load(manifest_path, manifest, "val")
        )
    bootstrap = None
    if bootstrap_config.n_resamples:
        bootstrap = BootstrapMetrics(
            "r2" if regression else "auroc" if binary else "micro_f1" if multilabel else "accuracy",
            bootstrap_config,
            metrics=("auroc",) if binary else None,
            inclusive_threshold=True,
            zero_division=0.0,
        )
    test_primary, test_macro_f1 = _metrics(
        method, _load(manifest_path, manifest, "test"), bootstrap=bootstrap,
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
    if regression:
        result["metric"] = "r2"
    if bootstrap is not None:
        result["test_confidence_intervals"] = bootstrap.compute()
    result_path.write_text(json.dumps(result, indent=2) + "\n")




if __name__ == "__main__":
    main()
