"""First-party DP-GNN training for graph-disjoint experiment partitions.

DP-GNN's bounded-degree graph sampler supplies fixed one-hop stars. Opacus
globally clips per-root gradients and adds isotropic Gaussian noise before
Adam updates. Uniform without-replacement root batches match the separate
multi-term hypergeometric RDP accountant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

from src.models.baselines import _OneHopGCN, _PaddedOneHopGCN
from src.models.objectives import _multilabel_micro_f1, _task_loss
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
    evaluate_every: int = 50
    seed: int = 0
    max_degree: int = 5
    latent_size: int = 100
    learning_rate: float = 3e-3
    clip: float = 1.0
    max_subgraph_nodes: int = 100
    max_private_batch_nodes: int = 8192
    multilabel: bool = False


def _inverse_degree_weights(edge_index: torch.Tensor, num_nodes: int,
                            device: torch.device) -> torch.Tensor:
    senders = edge_index[0].to(device)
    degree = torch.bincount(senders, minlength=num_nodes).to(
        device=device, dtype=torch.float32).clamp_min_(1.0)
    return degree[senders].reciprocal()


class PartitionedDPGNN:
    """Train DP-GNN solely on a graph-disjoint training partition."""

    def __init__(self, config: DPGNNConfig, device: str | torch.device = "cpu"):
        if config.steps < 1 or config.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if not isfinite(config.noise_multiplier) or config.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive and finite")
        if not isfinite(config.clip) or config.clip <= 0.0:
            raise ValueError("clip must be positive and finite")
        if config.max_degree < 1:
            raise ValueError("max_degree must be positive")
        if config.max_subgraph_nodes < 1:
            raise ValueError("max_subgraph_nodes must be positive")
        if config.max_private_batch_nodes < 1:
            raise ValueError("max_private_batch_nodes must be positive")
        self.config = config
        self.device = torch.device(device)

    def _prepared_graph(self, data: Any, *, seed: int):
        edge_index = sample_training_edges(
            data, max_degree=self.config.max_degree, seed=seed).to(self.device)
        weights = _inverse_degree_weights(edge_index, int(data.num_nodes), self.device)
        label_dtype = torch.float32 if self.config.multilabel else torch.long
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
            logits = model(current.features, current.node_mask)
            if self.config.multilabel:
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
    def evaluate(self, model: _OneHopGCN, data: Any, *, seed: int) -> float:
        model.eval()
        x, labels, edge_index, weights = self._prepared_graph(data, seed=seed)
        if self.config.multilabel:
            return _multilabel_micro_f1(model(x, edge_index, weights), labels)[0]
        return float((model(x, edge_index, weights).argmax(dim=-1) == labels).float().mean())

    def fit(self, train: Any, val: Any, test: Any) -> dict[str, Any]:
        num_nodes = int(train.num_nodes)
        if num_nodes < 1:
            raise ValueError("training partition must contain at least one node")
        if self.config.batch_size > num_nodes:
            raise ValueError("batch_size must not exceed the number of training nodes")
        max_terms = min(max_terms_per_node(self.config.max_degree), num_nodes)
        torch.manual_seed(self.config.seed)
        # Training only needs CPU CSR; do not copy full-graph edges to the GPU.
        edge_index = sample_training_edges(
            train, max_degree=self.config.max_degree, seed=self.config.seed + 1)
        adjacency = build_adjacency(
            edge_index[:, edge_index[0] != edge_index[1]], num_nodes, direction="out")
        label_dtype = torch.float32 if self.config.multilabel else torch.long
        x, labels = train.x.to(self.device), train.y.to(self.device, dtype=label_dtype)
        model = _OneHopGCN(x.size(1), self.config.latent_size,
                            self.config.num_classes).to(self.device)
        adam = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        private_module = GradSampleModule(
            _PaddedOneHopGCN(model), batch_first=True, loss_reduction="mean", strict=True)
        root_generator = torch.Generator().manual_seed(self.config.seed + 2)
        noise_generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 10_000)
        # lambda is sensitivity-normalized, unlike Opacus's clip-normalized multiplier.
        optimizer = DPOptimizer(
            optimizer=adam, noise_multiplier=2 * max_terms * self.config.noise_multiplier,
            max_grad_norm=self.config.clip, expected_batch_size=self.config.batch_size,
            loss_reduction="mean", generator=noise_generator, secure_mode=False)
        private_module.train()
        for _ in range(self.config.steps):
            roots = sample_dpgnn_roots(
                num_nodes, self.config.batch_size, generator=root_generator)
            batches = iter_dpgnn_batches(
                roots, adjacency=adjacency, x=x, y=labels,
                max_subgraph_nodes=self.config.max_subgraph_nodes,
                max_padded_nodes=self.config.max_private_batch_nodes, device=self.device)
            self._private_step(private_module, optimizer, batches)
        private_module.to_standard_module()
        delta = 1.0 / (10 * num_nodes)
        metric = "micro_f1" if self.config.multilabel else "accuracy"
        return {
            "model": model,
            f"validation_{metric}": self.evaluate(model, val, seed=self.config.seed + 3),
            f"test_{metric}": self.evaluate(model, test, seed=self.config.seed + 4),
            "epsilon": multiterm_dpsgd_epsilon(
                steps=self.config.steps, noise_multiplier=self.config.noise_multiplier,
                delta=delta, num_samples=num_nodes,
                batch_size=self.config.batch_size,
                max_terms=max_terms),
            "delta": delta,
        }
