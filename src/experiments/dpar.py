"""PyTorch implementation of the released DPAR training structure.

DPAR's TensorFlow model is a decoupled MLP: private approximate PPR weights
aggregate per-neighbour logits during training, followed by power-iteration
propagation for inference.  This module ports that structure without requiring
TensorFlow 1.x or changing the upstream privacy formulas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .inductive import _induce, graph_statistics
from .privacy import DPARAccountant, PrivacyResult, calibrate_dpar_noise


@dataclass(frozen=True)
class DPARConfig:
    alpha: float = 0.25
    rho: float = 1e-4
    ista_epsilon: float = 1e-4
    topk: int = 16
    sampled_train_nodes: int | None = None
    dp_ppr: bool = False
    ppr_noise: float = 0.0067
    ppr_clip: float = 0.01
    ppr_delta: float = 1e-4
    ppr_column_clip: float = 1.0
    dp_sgd: bool = False
    sgd_noise: float = 0.95
    sgd_clip: float = 1.0
    target_epsilon: float | None = None
    target_delta: float | None = None
    calibration_rtol: float = 1e-3
    calibration_atol: float = 1e-6
    calibration_max_noise_multiplier: float = 1e6
    sgd_delta: float = 1e-3
    batch_size: int = 60
    hidden_size: int = 32
    layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    inference_steps: int = 2
    seed: int = 0


class DPARMLP(nn.Module):
    """The released DPAR ``W1 ... Wo`` MLP, expressed with torch modules."""

    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 2:
            raise ValueError("DPAR requires at least two MLP layers")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))
        return self.layers[-1](F.dropout(x, p=self.dropout, training=self.training))


def _coalesced_adjacency(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    raw = torch.sparse_coo_tensor(edge_index.to(device), values, (num_nodes, num_nodes),
                                  device=device, check_invariants=False).coalesce()
    return torch.sparse_coo_tensor(
        raw.indices(), torch.ones(raw._nnz(), dtype=torch.float32, device=device),
        raw.shape, device=device, check_invariants=False,
    ).coalesce()


@torch.no_grad()
def private_ista_ppr(
    edge_index: Tensor,
    num_nodes: int,
    config: DPARConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Compute a top-k PPR row for every sampled training-graph node.

    Dense PPR vectors are L2-clipped and noised before top-k selection when
    ``dp_ppr`` is enabled. The resulting sparse columns are only scaled down
    when their L1 mass exceeds the Algorithm-2 bound.
    """
    if config.topk < 1:
        raise ValueError("topk must be positive")
    if not math.isfinite(config.ppr_column_clip) or config.ppr_column_clip <= 0.0:
        raise ValueError("ppr_column_clip must be finite and positive")
    adjacency = _coalesced_adjacency(edge_index, num_nodes, device)
    degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
    out_degree = degrees
    inverse_degree = degrees.clamp_min(1e-12).reciprocal()
    rows: list[Tensor] = []
    cols: list[Tensor] = []
    values: list[Tensor] = []
    # The upstream stopping rule has no iteration cap. A high cap turns genuine
    # non-convergence into a clear error instead of silently releasing a partial PPR.
    max_iterations = max(10_000, num_nodes * 10)
    for root in range(num_nodes):
        p = torch.zeros(num_nodes, device=device)
        delta = torch.zeros(num_nodes, device=device)
        delta[root] = -config.alpha * inverse_degree[root]
        for _ in range(max_iterations):
            if delta.abs().amax() <= (1.0 + config.ista_epsilon) * config.rho * config.alpha:
                break
            active = (p - delta >= config.rho * config.alpha) & (out_degree > 0)
            if not bool(active.any()):
                break
            delta_p = torch.zeros_like(delta)
            delta_p[active] = -(delta[active] + config.rho * config.alpha)
            message = torch.sparse.mm(adjacency, (delta_p / out_degree.clamp_min(1.0)).unsqueeze(1)).squeeze(1)
            updated = delta.clone()
            inv_out = out_degree.clamp_min(1.0).reciprocal()
            updated[active] = (
                (1.0 - inv_out[active]) * delta[active]
                - config.rho * config.alpha * inv_out[active]
                - 0.5 * (1.0 - config.alpha) * delta_p[active] * inv_out[active]
                - 0.5 * (1.0 - config.alpha) * message[active] * inv_out[active]
            )
            neighbour = (~active) & (message != 0) & (out_degree > 0)
            updated[neighbour] = delta[neighbour] - 0.5 * (1.0 - config.alpha) * message[neighbour] * inv_out[neighbour]
            p += delta_p
            delta = updated
        else:
            raise RuntimeError(f"DPAR ISTA did not converge for root {root} after {max_iterations} iterations")
        if config.dp_ppr:
            p = p * min(1.0, config.ppr_clip / float(p.norm().clamp_min(1e-12)))
            p = p + torch.randn(num_nodes, device=device, generator=generator) * config.ppr_noise
        nonzero = torch.where(p != 0)[0]
        if nonzero.numel():
            chosen = nonzero[torch.topk(p[nonzero], k=min(config.topk, nonzero.numel())).indices]
            rows.append(torch.full_like(chosen, root))
            cols.append(chosen)
            values.append(p[chosen])
    if rows:
        ppr = torch.sparse_coo_tensor(
            torch.stack((torch.cat(rows), torch.cat(cols))), torch.cat(values),
            (num_nodes, num_nodes), device=device, check_invariants=False,
        ).coalesce()
        column_norm = torch.zeros(num_nodes, device=device)
        column_norm.scatter_add_(0, ppr.indices()[1], ppr.values().abs())
        scale = (config.ppr_column_clip / column_norm.clamp_min(1e-12)).clamp(max=1.0)
        return torch.sparse_coo_tensor(
            ppr.indices(), ppr.values() * scale[ppr.indices()[1]], ppr.shape,
            device=device, check_invariants=False,
        ).coalesce()
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.float32, device=device), (num_nodes, num_nodes),
        device=device, check_invariants=False,
    ).coalesce()


def _select_ppr_rows(ppr: Tensor, roots: Tensor) -> Tensor:
    """Extract selected rows without materializing an N×N dense PPR matrix."""
    ppr = ppr.coalesce()
    global_to_local = torch.full((ppr.size(0),), -1, dtype=torch.long, device=roots.device)
    global_to_local[roots] = torch.arange(roots.numel(), device=roots.device)
    local_rows = global_to_local[ppr.indices()[0]]
    keep = local_rows >= 0
    return torch.sparse_coo_tensor(
        torch.stack((local_rows[keep], ppr.indices()[1, keep])),
        ppr.values()[keep], (roots.numel(), ppr.size(1)), device=roots.device,
        check_invariants=False,
    ).coalesce()

@torch.no_grad()
def propagate_logits(logits: Tensor, edge_index: Tensor, alpha: float, steps: int) -> Tensor:
    """Released DPAR's row-normalized power-iteration inference."""
    adjacency = _coalesced_adjacency(edge_index, logits.size(0), logits.device)
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1e-12)
    local = logits
    propagated = logits.clone()
    for _ in range(steps):
        propagated = (1.0 - alpha) * torch.sparse.mm(adjacency, propagated) / degree[:, None] + alpha * local
    return propagated


def _accuracy_and_macro_f1(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean())
    f1s = []
    for label in torch.unique(labels):
        positive = predictions == label
        truth = labels == label
        denom = 2 * (positive & truth).sum() + (positive & ~truth).sum() + (~positive & truth).sum()
        f1s.append(float(2 * (positive & truth).sum() / denom) if denom else 0.0)
    return accuracy, sum(f1s) / len(f1s)


def _sample_train_partition(partition: Any, requested_nodes: int | None,
                            generator: torch.Generator, device: torch.device) -> tuple[Any, dict[str, Any]]:
    """Sample the outer DPAR training graph and return its reproducible statistics."""
    train_data = partition.data
    train_nodes = int(train_data.num_nodes)
    if train_nodes <= 0:
        raise ValueError("DPAR requires a non-empty train partition")
    if requested_nodes is None:
        sampled_nodes = train_nodes
    else:
        if requested_nodes <= 0:
            raise ValueError("sampled_train_nodes must be positive")
        sampled_nodes = min(requested_nodes, train_nodes)
    if sampled_nodes == train_nodes:
        return train_data.to(device), graph_statistics(train_data)
    chosen = torch.randperm(train_nodes, device=device, generator=generator)[:sampled_nodes].cpu()
    mask = torch.zeros(train_nodes, dtype=torch.bool)
    mask[chosen] = True
    sampled_data, _ = _induce(train_data, mask)
    return sampled_data.to(device), graph_statistics(sampled_data)


class DPARTrainer:
    """DPAR trainer operating solely on a graph-disjoint train partition."""

    def __init__(self, config: DPARConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def _evaluate(self, model: DPARMLP, partition: Any) -> tuple[float, float]:
        data = partition.data.to(self.device)
        model.eval()
        with torch.no_grad():
            logits = propagate_logits(model(data.x), data.edge_index, self.config.alpha, self.config.inference_steps)
        return _accuracy_and_macro_f1(logits, data.y)

    def fit(self, split: Any) -> dict[str, Any]:
        """Train, choose by validation accuracy, and evaluate the held-out graph."""
        torch.manual_seed(self.config.seed)
        full_train_data = split.train.data
        model = DPARMLP(full_train_data.x.size(1), split.num_classes,
                         self.config.hidden_size, self.config.layers, self.config.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        sampling_generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)
        train_data, sampled_train_graph = _sample_train_partition(
            split.train, self.config.sampled_train_nodes, sampling_generator, self.device,
        )
        sampled_nodes = int(train_data.num_nodes)
        target_mode = self.config.target_epsilon is not None or self.config.target_delta is not None
        if target_mode and (self.config.target_epsilon is None or self.config.target_delta is None):
            raise ValueError("target_epsilon and target_delta must be provided together")
        calibration = None
        effective_config = self.config
        if target_mode:
            steps = self.config.epochs * math.ceil(sampled_nodes / self.config.batch_size)
            calibration = calibrate_dpar_noise(
                target_epsilon=self.config.target_epsilon, target_delta=self.config.target_delta,
                train_nodes=int(full_train_data.num_nodes), ppr_releases=sampled_nodes,
                ppr_clip=self.config.ppr_clip, sgd_clip=self.config.sgd_clip,
                batch_size=self.config.batch_size, steps=steps,
                sigma_rtol=self.config.calibration_rtol, sigma_atol=self.config.calibration_atol,
                max_noise_multiplier=self.config.calibration_max_noise_multiplier,
            )
            effective_config = replace(
                self.config, dp_ppr=True, dp_sgd=True,
                ppr_noise=calibration.ppr_noise_std,
                ppr_delta=calibration.ppr_delta_per_release,
                sgd_noise=calibration.sgd_noise_std, sgd_delta=calibration.sgd_delta,
            )
        preprocessing_start = time.perf_counter()
        ppr = private_ista_ppr(train_data.edge_index, sampled_nodes, effective_config,
                               self.device, sampling_generator)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        best_state, best_val = None, float("-inf")
        training_start = time.perf_counter()
        for _ in range(effective_config.epochs):
            model.train()
            permutation = torch.randperm(sampled_nodes, device=self.device, generator=sampling_generator)
            for root_indices in permutation.split(effective_config.batch_size):
                if effective_config.dp_sgd:
                    self._private_step(model, optimizer, train_data.x, train_data.y, ppr, root_indices,
                                       sampling_generator, effective_config)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    logits = torch.sparse.mm(_select_ppr_rows(ppr, root_indices), model(train_data.x))
                    F.cross_entropy(logits, train_data.y[root_indices]).backward()
                    optimizer.step()
            val_accuracy, _ = self._evaluate(model, split.val)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        training_seconds = time.perf_counter() - training_start
        assert best_state is not None
        model.load_state_dict(best_state)
        val_accuracy, val_f1 = self._evaluate(model, split.val)
        test_accuracy, test_f1 = self._evaluate(model, split.test)
        privacy = self._privacy(int(full_train_data.num_nodes), sampled_nodes, effective_config)
        result = {
            "method": "dpar", "config": asdict(effective_config), "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1, "test_accuracy": test_accuracy, "test_macro_f1": test_f1,
            "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
            "privacy": privacy, "train_graph": split.train.stats, "sampled_train_graph": sampled_train_graph,
        }
        if calibration is not None:
            result["calibration"] = calibration.as_dict()
        return result

    def _private_step(self, model: DPARMLP, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor,
                      ppr: Tensor, roots: Tensor, generator: torch.Generator, config: DPARConfig) -> None:
        """Microbatch=example DP-Adam update matching upstream DPAR's setting."""
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        clipped = [torch.zeros_like(parameter) for parameter in parameters]
        logits = model(x)
        root_logits = torch.sparse.mm(_select_ppr_rows(ppr, roots), logits)
        for row, target in zip(root_logits, y[roots]):
            gradients = torch.autograd.grad(F.cross_entropy(row.unsqueeze(0), target.unsqueeze(0)), parameters,
                                            retain_graph=True)
            norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients)).clamp_min(1e-12)
            scale = min(1.0, config.sgd_clip / float(norm))
            for accumulator, gradient in zip(clipped, gradients):
                accumulator.add_(gradient, alpha=scale)
        optimizer.zero_grad(set_to_none=True)
        for parameter, accumulator in zip(parameters, clipped):
            parameter.grad = (
                accumulator
                + torch.randn(accumulator.shape, dtype=accumulator.dtype, device=accumulator.device,
                              generator=generator) * config.sgd_noise
            ) / len(roots)
        optimizer.step()

    def _privacy(self, train_nodes: int, ppr_releases: int, config: DPARConfig) -> dict[str, Any]:
        accountant = DPARAccountant()
        amplification_rate = ppr_releases / train_nodes
        ppr = accountant.account(
            ppr_releases=ppr_releases, amplification_rate=amplification_rate,
            delta=config.ppr_delta, ppr_clip=config.ppr_clip if config.dp_ppr else None,
            ppr_noise=config.ppr_noise if config.dp_ppr else None, topk=config.topk,
        )
        if config.dp_sgd:
            steps = config.epochs * math.ceil(ppr_releases / config.batch_size)
            sgd = accountant.account_training(
                noise_multiplier=config.sgd_noise / config.sgd_clip,
                sample_rate=min(config.batch_size / ppr_releases, 1.0), steps=steps,
                delta=config.sgd_delta, amplification_rate=amplification_rate,
            )
        else:
            sgd = None
        total = None
        if ppr.epsilon is not None and sgd is not None and sgd.epsilon is not None:
            total = PrivacyResult(
                epsilon=ppr.epsilon + sgd.epsilon, delta=ppr.delta + sgd.delta,
                accountant="dpar.paper_theorem2_composition",
                parameters={"amplification_rate": amplification_rate},
            )
        return {
            "ppr": ppr.as_dict(), "training": None if sgd is None else sgd.as_dict(),
            "total": None if total is None else total.as_dict(),
        }
