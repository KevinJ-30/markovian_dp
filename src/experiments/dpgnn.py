"""First-party DP-GNN training for graph-disjoint experiment partitions.

This ports the DP-GNN GCN path used by the former Google Research adapter:
reverse-edge preprocessing, bounded-degree Bernoulli sampling, inverse-degree
normalization, one-hop per-root gradients, per-parameter clipping, Gaussian
noise, DP-Adam, and the multi-term RDP accountant.  It deliberately depends
only on this repository's PyTorch/scientific Python stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import scipy.special
import scipy.stats
import torch
from torch import nn
import torch.nn.functional as F


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
    clip_percentile: float = 75.0
    max_subgraph_nodes: int = 100


class _OneHopGCN(nn.Module):
    """The DP-GNN one-hop GCN convention: receivers send to senders."""

    def __init__(self, inputs: int, hidden: int, classes: int):
        super().__init__()
        self.encoder = nn.Linear(inputs, hidden)
        self.core = nn.Linear(hidden, hidden)
        self.decoder = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.encoder(x))
        aggregated = torch.zeros_like(x)
        if edge_index.numel():
            senders, receivers = edge_index
            aggregated.index_add_(0, senders, x[receivers] * edge_weight[:, None])
        x = aggregated + torch.tanh(self.core(aggregated))
        return self.decoder(x)


def max_terms_per_node(max_degree: int) -> int:
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    return max_degree + 1


def base_sensitivity(max_degree: int) -> float:
    return float(2 * max_terms_per_node(max_degree))


def multiterm_dpsgd_epsilon(*, steps: int, noise_multiplier: float,
                             delta: float, num_samples: int,
                             batch_size: int, max_terms: int) -> float:
    """Port of DP-GNN's hypergeometric multi-term RDP accountant."""
    if steps < 1 or num_samples < 1 or batch_size < 1:
        raise ValueError("steps, num_samples, and batch_size must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if noise_multiplier < 1e-20:
        return float("inf")
    from dp_accounting import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant, compute_epsilon

    batch_size = min(batch_size, num_samples)
    max_terms = min(max_terms, num_samples)
    terms = np.arange(max_terms + 1)
    terms_logprobs = scipy.stats.hypergeom(
        num_samples, max_terms, batch_size).logpmf(terms)
    orders = np.arange(1, 10, 0.1)[1:]
    accountant = RdpAccountant(orders)
    accountant.compose(GaussianDpEvent(noise_multiplier))
    unamplified = np.asarray(accountant._rdp)  # DP-Accounting has no public RDP accessor.
    amplified = []
    for order, rdp in zip(orders, unamplified):
        beta = rdp * (order - 1)
        log_factors = beta * np.square(terms / max_terms)
        amplified.append(scipy.special.logsumexp(terms_logprobs + log_factors) /
                         (order - 1))
    amplified = np.asarray(amplified)
    if not np.all(unamplified * (batch_size / num_samples) ** 2 <= amplified + 1e-6):
        raise ValueError("DP-GNN multi-term RDP lower bound was violated")
    return float(compute_epsilon(orders, amplified * steps, delta)[0])


def _sample_training_edges(data: Any, *, max_degree: int, seed: int) -> torch.Tensor:
    """Port DP-GNN's train-node incoming-edge Bernoulli sampler on CPU."""
    num_nodes = int(data.num_nodes)
    original = data.edge_index.detach().cpu().to(torch.long)
    # input_pipeline.add_reverse_edges(), retaining multiplicities.
    edge_index = torch.cat((original, original.flip(0)), dim=1)
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    for sender, receiver in edge_index.t().tolist():
        incoming[receiver].append(sender)
    generator = torch.Generator().manual_seed(seed)
    kept: list[tuple[int, int]] = []
    for receiver, senders in enumerate(incoming):
        if not senders:
            continue
        probability = max_degree / (2.0 * len(senders))
        sampled = [sender for sender in senders
                   if bool(torch.rand((), generator=generator) <= probability)]
        unique = sorted(set(sampled))
        if len(unique) <= max_degree:
            kept.extend((sender, receiver) for sender in unique)
    if kept:
        sampled_edges = torch.tensor(kept, dtype=torch.long).t().contiguous()
    else:
        sampled_edges = torch.empty((2, 0), dtype=torch.long)
    loops = torch.arange(num_nodes, dtype=torch.long)
    return torch.cat((torch.stack((loops, loops)), sampled_edges), dim=1)


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
        if config.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive")
        self.config = config
        self.device = torch.device(device)

    def _prepared_graph(self, data: Any, *, seed: int):
        edge_index = _sample_training_edges(
            data, max_degree=self.config.max_degree, seed=seed).to(self.device)
        weights = _inverse_degree_weights(edge_index, int(data.num_nodes), self.device)
        return data.x.to(self.device), data.y.to(self.device).long(), edge_index, weights

    def _root_graph(self, x: torch.Tensor, edge_index: torch.Tensor, root: int):
        neighbours = edge_index[1, edge_index[0] == root]
        neighbours = torch.unique(neighbours, sorted=True)
        neighbours = torch.cat((torch.tensor([root], device=self.device),
                                neighbours[neighbours != root]))
        neighbours = neighbours[:self.config.max_subgraph_nodes]
        local_receivers = torch.arange(neighbours.numel(), device=self.device)
        local_edges = torch.stack((torch.zeros_like(local_receivers), local_receivers))
        weights = torch.full((neighbours.numel(),), 1.0 / neighbours.numel(),
                             device=self.device)
        return x[neighbours], local_edges, weights

    def _loss_for_root(self, model: _OneHopGCN, x: torch.Tensor,
                       edge_index: torch.Tensor, labels: torch.Tensor, root: int):
        sub_x, sub_edges, sub_weights = self._root_graph(x, edge_index, root)
        return F.cross_entropy(model(sub_x, sub_edges, sub_weights)[0:1], labels[root:root + 1])

    def _clip_thresholds(self, model: _OneHopGCN, x: torch.Tensor,
                         edge_index: torch.Tensor, labels: torch.Tensor,
                         roots: torch.Tensor) -> list[torch.Tensor]:
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        thresholds: list[list[torch.Tensor]] = [[] for _ in parameters]
        for root in roots.tolist():
            grads = torch.autograd.grad(
                self._loss_for_root(model, x, edge_index, labels, root), parameters,
                allow_unused=True)
            for values, grad, parameter in zip(thresholds, grads, parameters):
                values.append((torch.zeros_like(parameter) if grad is None else grad).norm())
        return [torch.quantile(torch.stack(values), self.config.clip_percentile / 100.0)
                .clamp_min(1e-12) for values in thresholds]

    def _private_step(self, model: _OneHopGCN, optimizer: torch.optim.Optimizer,
                      x: torch.Tensor, edge_index: torch.Tensor,
                      labels: torch.Tensor, roots: torch.Tensor,
                      thresholds: list[torch.Tensor], generator: torch.Generator) -> None:
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        batch_size = int(roots.numel())
        accumulated = [torch.zeros_like(parameter) for parameter in parameters]
        for root in roots.tolist():
            grads = torch.autograd.grad(
                self._loss_for_root(model, x, edge_index, labels, root), parameters,
                allow_unused=True)
            for accumulator, grad, parameter, threshold in zip(accumulated, grads, parameters, thresholds):
                grad = (torch.zeros_like(parameter) if grad is None else grad) / batch_size
                accumulator.add_(grad / (grad.norm() / threshold).clamp_min(1.0))
        optimizer.zero_grad(set_to_none=True)
        sensitivity = base_sensitivity(self.config.max_degree)
        for parameter, total, threshold in zip(parameters, accumulated, thresholds):
            parameter.grad = total + torch.randn(
                total.shape, dtype=total.dtype, device=total.device,
                generator=generator) * (threshold * sensitivity * self.config.noise_multiplier)
        optimizer.step()

    @torch.no_grad()
    def evaluate(self, model: _OneHopGCN, data: Any, *, seed: int) -> float:
        model.eval()
        x, labels, edge_index, weights = self._prepared_graph(data, seed=seed)
        return float((model(x, edge_index, weights).argmax(dim=-1) == labels).float().mean())

    def fit(self, train: Any, val: Any, test: Any) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        x, labels, edge_index, _ = self._prepared_graph(train, seed=self.config.seed + 1)
        model = _OneHopGCN(x.size(1), self.config.latent_size,
                            self.config.num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 2)
        roots = torch.arange(int(train.num_nodes), device=self.device)
        estimate = roots[:min(self.config.batch_size, roots.numel())]
        thresholds = self._clip_thresholds(model, x, edge_index, labels, estimate)
        model.train()
        for _ in range(self.config.steps):
            batch = roots[torch.randint(roots.numel(), (self.config.batch_size,),
                                        generator=generator, device=self.device)]
            self._private_step(model, optimizer, x, edge_index, labels, batch,
                               thresholds, generator)
        delta = 1.0 / (10 * int(train.num_nodes))
        return {
            "model": model,
            "validation_accuracy": self.evaluate(model, val, seed=self.config.seed + 3),
            "test_accuracy": self.evaluate(model, test, seed=self.config.seed + 4),
            "epsilon": multiterm_dpsgd_epsilon(
                steps=self.config.steps, noise_multiplier=self.config.noise_multiplier,
                delta=delta, num_samples=int(train.num_nodes),
                batch_size=self.config.batch_size,
                max_terms=max_terms_per_node(self.config.max_degree)),
            "delta": delta,
        }
