"""First-party DP-GNN training for graph-disjoint experiment partitions.

DP-GNN's bounded-incoming-degree sampler supplies complete rooted neighborhoods.
Opacus globally clips per-root gradients and adds isotropic Gaussian noise before
Adam updates. Uniform without-replacement root batches match the separate
multi-term hypergeometric RDP accountant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isnan
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

from src.models.baselines import (
    _OneHopGCN, _OneHopGIN, _OneHopGraphSAGE,
    _PaddedOneHopGCN, _PaddedOneHopGIN, _PaddedOneHopGraphSAGE,
)
from src.models.dpgnn import _MultiHopDPGNN, _PaddedMultiHopDPGNN
from src.models.bootstrap import BootstrapConfig, BootstrapMetrics
from src.models.objectives import _metric_rows, _task_loss, _task_metric
from src.processing.dpgnn import (
    iter_dpgnn_batches, sample_dpgnn_roots, sample_training_edges,
)
from src.processing.padded import PaddedRootedBatch
from src.processing.sparse_expand import build_adjacency
from src.privacy.dpgnn import max_terms_per_node, multiterm_dpsgd_epsilon


@dataclass(frozen=True)
class DPGNNConfig:
    num_classes: int
    steps: int
    batch_size: int
    noise_multiplier: float
    evaluate_every: int = 0
    seed: int = 0
    max_degree: int = 5
    radius: int = 1
    latent_size: int = 100
    learning_rate: float = 3e-3
    clip: float = 1.0
    max_private_batch_nodes: int = 8192
    multilabel: bool = False
    # Loss/metric only; epsilon is unchanged.
    regression: bool = False
    binary: bool = False
    metric_ignore_label: int | None = None
    architecture: str = "graphsage"
    dropout: float = 0.5
    bootstrap_confidence: float = 0.95
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    weight_decay: float = 0.0
    delta: float | None = None

    def __post_init__(self) -> None:
        BootstrapConfig(
            confidence_level=self.bootstrap_confidence,
            n_resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed,
        )


def _inverse_degree_weights(edge_index: torch.Tensor, num_nodes: int,
                            device: torch.device) -> torch.Tensor:
    senders = edge_index[0].to(device)
    degree = torch.bincount(senders, minlength=num_nodes).to(
        device=device, dtype=torch.float32).clamp_min_(1.0)
    return degree[senders].reciprocal()


class PartitionedDPGNN:
    """Train DP-GNN solely on a graph-disjoint training partition."""

    def __init__(self, config: DPGNNConfig, device: str | torch.device = "cpu"):
        if sum((config.multilabel, config.regression, config.binary)) > 1:
            raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
        if config.steps < 1 or config.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if config.evaluate_every < 0:
            raise ValueError("evaluate_every must be nonnegative")
        if not isfinite(config.noise_multiplier) or config.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive and finite")
        if not isfinite(config.clip) or config.clip <= 0.0:
            raise ValueError("clip must be positive and finite")
        if not isfinite(config.dropout) or not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isfinite(config.weight_decay) or config.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if config.delta is not None and (
                not isfinite(config.delta) or not 0 < config.delta < 1):
            raise ValueError("delta must be finite and in (0, 1)")
        if config.max_degree < 1:
            raise ValueError("max_degree must be positive")
        max_terms_per_node(config.max_degree, config.radius)
        if config.radius > 1 and config.architecture == "gcn":
            raise ValueError("multi-hop DP-GNN supports graphsage and gin")
        if config.max_private_batch_nodes < 1:
            raise ValueError("max_private_batch_nodes must be positive")
        if config.architecture not in {"gcn", "gin", "graphsage"}:
            raise ValueError("architecture must be 'gcn', 'gin', or 'graphsage'")
        self.config = config
        self.device = torch.device(device)

    def _prepared_graph(self, data: Any, *, seed: int):
        edge_index = sample_training_edges(
            data, max_degree=self.config.max_degree, seed=seed).to(self.device)
        weights = _inverse_degree_weights(edge_index, int(data.num_nodes), self.device)
        label_dtype = (
            torch.float32
            if (self.config.multilabel or self.config.regression or self.config.binary)
            else torch.long
        )
        return data.x.to(self.device), data.y.to(self.device, dtype=label_dtype), edge_index, weights

    def _private_step(
        self, model: GradSampleModule, optimizer: DPOptimizer,
        batches: Iterable[PaddedRootedBatch],
    ) -> None:
        """Accumulate physical chunks into one clipped, noisy Adam update."""
        optimizer.zero_grad()
        batches = iter(batches)
        current = next(batches)
        while True:
            following = next(batches, None)
            logits = (
                model(current.features, current.node_mask,
                      current.edge_index, current.edge_mask)
                if self.config.radius > 1 else
                model(current.features, current.node_mask)
            )
            if self.config.regression:
                # One mean over roots, so Opacus keeps the per-sample axis.
                _task_loss(
                    logits, current.labels, multilabel=False,
                    regression=True).backward()
            elif self.config.binary:
                _task_loss(
                    logits, current.labels, multilabel=False,
                    binary=True).backward()
            elif self.config.multilabel:
                # BCE averages classes within each root, then roots. Opacus
                # retains the leading root axis for globally clipped samples.
                _task_loss(logits, current.labels, multilabel=True).backward()
            else:
                losses = F.cross_entropy(logits, current.labels, reduction="none")
                losses.mean().backward()
            if following is not None:
                optimizer.signal_skip_step(True)
            optimizer.step()
            if following is None:
                break
            # A skipped step retains summed_grad but clears the physical samples.
            optimizer.zero_grad()
            current = following

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module, data: Any, *, seed: int,
        bootstrap: BootstrapMetrics | None = None,
    ) -> float:
        model.eval()
        x, labels, edge_index, weights = self._prepared_graph(data, seed=seed)
        predictions = model(x, edge_index, weights)
        predictions, labels = _metric_rows(
            predictions,
            labels,
            eval_mask=getattr(data, "eval_mask", None),
            metric_ignore_label=self.config.metric_ignore_label,
        )
        if bootstrap is not None:
            bootstrap.update(predictions, labels)
        return _task_metric(
            predictions, labels, self.config.multilabel,
            regression=self.config.regression, binary=self.config.binary)[0]

    def fit(self, train: Any, val: Any, test: Any) -> dict[str, Any]:
        num_nodes = int(train.num_nodes)
        if num_nodes < 1:
            raise ValueError("training partition must contain at least one node")
        if self.config.batch_size > num_nodes:
            raise ValueError("batch_size must not exceed the number of training nodes")
        max_terms = min(
            max_terms_per_node(self.config.max_degree, self.config.radius), num_nodes)
        torch.manual_seed(self.config.seed)
        # Training only needs CPU CSR; do not copy full-graph edges to the GPU.
        edge_index = sample_training_edges(
            train, max_degree=self.config.max_degree, seed=self.config.seed + 1)
        adjacency = build_adjacency(
            edge_index[:, edge_index[0] != edge_index[1]], num_nodes, direction="out")
        label_dtype = (
            torch.float32
            if (self.config.multilabel or self.config.regression or self.config.binary)
            else torch.long
        )
        x, labels = train.x.to(self.device), train.y.to(self.device, dtype=label_dtype)
        outputs = 1 if (self.config.binary or self.config.regression) else self.config.num_classes
        if self.config.radius > 1:
            model = _MultiHopDPGNN(
                x.size(1), self.config.latent_size, outputs,
                radius=self.config.radius, architecture=self.config.architecture,
                dropout=self.config.dropout).to(self.device)
            private_model = _PaddedMultiHopDPGNN(model)
        elif self.config.architecture == "graphsage":
            model = _OneHopGraphSAGE(
                x.size(1), self.config.latent_size, outputs,
                dropout=self.config.dropout).to(self.device)
            private_model = _PaddedOneHopGraphSAGE(model)
        elif self.config.architecture == "gin":
            model = _OneHopGIN(
                x.size(1), self.config.latent_size, outputs,
                dropout=self.config.dropout).to(self.device)
            private_model = _PaddedOneHopGIN(model)
        else:
            model = _OneHopGCN(
                x.size(1), self.config.latent_size, outputs,
                dropout=self.config.dropout).to(self.device)
            private_model = _PaddedOneHopGCN(model)
        adam = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        private_module = GradSampleModule(
            private_model, batch_first=True, loss_reduction="mean", strict=True)
        root_generator = torch.Generator().manual_seed(self.config.seed + 2)
        noise_generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 10_000)
        # lambda is sensitivity-normalized, unlike Opacus's clip-normalized multiplier.
        optimizer = DPOptimizer(
            optimizer=adam, noise_multiplier=2 * max_terms * self.config.noise_multiplier,
            max_grad_norm=self.config.clip, expected_batch_size=self.config.batch_size,
            loss_reduction="mean", generator=noise_generator, secure_mode=False)
        evaluate_every = self.config.evaluate_every or (
            (num_nodes + self.config.batch_size - 1) // self.config.batch_size)
        best_state = None
        best_validation = float("nan")
        best_step = 0
        private_module.train()
        for step in range(1, self.config.steps + 1):
            roots = sample_dpgnn_roots(
                num_nodes, self.config.batch_size, generator=root_generator)
            batches = iter_dpgnn_batches(
                roots, adjacency=adjacency, x=x, y=labels,
                radius=self.config.radius,
                max_padded_nodes=self.config.max_private_batch_nodes, device=self.device)
            self._private_step(private_module, optimizer, batches)
            if step % evaluate_every == 0 or step == self.config.steps:
                validation = self.evaluate(model, val, seed=self.config.seed + 3)
                if best_state is None or (
                    not isnan(validation)
                    and (isnan(best_validation) or validation > best_validation)
                ):
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    best_validation = validation
                    best_step = step
                # Full-graph evaluation shares the private view's layers.
                # Restore their training mode before the next Opacus update.
                private_module.train()
        private_module.to_standard_module()
        assert best_state is not None
        model.load_state_dict(best_state)
        delta = self.config.delta if self.config.delta is not None else 1.0 / (10 * num_nodes)
        metric = ("r2" if self.config.regression
                  else "auroc" if self.config.binary
                  else "micro_f1" if self.config.multilabel
                  else "accuracy")
        bootstrap = (
            BootstrapMetrics(
                metric,
                BootstrapConfig(
                    confidence_level=self.config.bootstrap_confidence,
                    n_resamples=self.config.bootstrap_resamples,
                    seed=self.config.bootstrap_seed,
                ),
                metrics=(metric,),
            )
            if self.config.bootstrap_resamples else None
        )
        result = {
            "model": model,
            "architecture": self.config.architecture,
            "radius": self.config.radius,
            "max_terms": max_terms,
            "privacy_scope": "fixed_sampled_topology_node_features_and_labels",
            # Keys are metric-named; publish the name so callers can resolve.
            "metric": metric,
            "selection": {
                "metric": metric,
                "step": best_step,
                "validation_score": None if isnan(best_validation) else best_validation,
                "evaluate_every": evaluate_every,
            },
            f"validation_{metric}": best_validation,
            f"test_{metric}": self.evaluate(
                model, test, seed=self.config.seed + 4, bootstrap=bootstrap),
            "epsilon": multiterm_dpsgd_epsilon(
                steps=self.config.steps, noise_multiplier=self.config.noise_multiplier,
                delta=delta, num_samples=num_nodes,
                batch_size=self.config.batch_size,
                max_terms=max_terms),
            "delta": delta,
        }
        if bootstrap is not None:
            result["test_confidence_intervals"] = bootstrap.compute()
        return result
