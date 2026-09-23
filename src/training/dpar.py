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
from torch import Tensor

from src.processing.graphs import preprocess_edges
from src.processing.splits import _induce, graph_statistics
from src.privacy.accountants import DPARAccountant, PrivacyResult, calibrate_dpar_noise

from src.models.baselines import DPARMLP

from src.models.objectives import _metric_rows, _task_loss, _task_metric


@dataclass(frozen=True)
class DPARConfig:
    alpha: float = 0.25
    rho: float = 1e-4
    ista_epsilon: float = 1e-4
    topk: int = 16
    sampled_train_rate: float | None = 0.09
    sampled_train_nodes: int | None = None
    ppr_num: int = 70
    dp_ppr: bool = False
    ppr_noise: float = 0.0067
    ppr_clip: float = 0.01
    ppr_delta: float = 1e-4
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
    multilabel: bool = False
    # Loss/metric only; epsilon is unchanged.
    regression: bool = False
    binary: bool = False
    metric_ignore_label: int | None = None

    def __post_init__(self) -> None:
        if sum((self.multilabel, self.regression, self.binary)) > 1:
            raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
        selectors = (self.sampled_train_rate is not None, self.sampled_train_nodes is not None)
        if sum(selectors) != 1:
            raise ValueError(
                "exactly one of sampled_train_rate and sampled_train_nodes must be set"
            )
        if self.sampled_train_rate is not None:
            rate = self.sampled_train_rate
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not 0.0 < float(rate) <= 1.0
            ):
                raise ValueError("sampled_train_rate must be finite and lie in (0, 1]")
        if self.sampled_train_nodes is not None and (
            isinstance(self.sampled_train_nodes, bool)
            or not isinstance(self.sampled_train_nodes, int)
            or self.sampled_train_nodes <= 0
        ):
            raise ValueError("sampled_train_nodes must be a positive integer")
        if isinstance(self.ppr_num, bool) or not isinstance(self.ppr_num, int) or self.ppr_num <= 0:
            raise ValueError("ppr_num must be a positive integer")


def _dpar_adjacency(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    """Return a binary adjacency with exactly one effective self-loop per node."""
    indices = preprocess_edges(
        edge_index, num_nodes, add_self_loops=True
    ).to(device)
    values = torch.ones(indices.size(1), dtype=torch.float32, device=device)
    raw = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=device, check_invariants=False
    ).coalesce()
    return torch.sparse_coo_tensor(
        raw.indices(),
        torch.ones(raw._nnz(), dtype=torch.float32, device=device),
        raw.shape,
        device=device,
        check_invariants=False,
    ).coalesce()


@torch.no_grad()
def private_ista_ppr(
    edge_index: Tensor,
    num_nodes: int,
    ppr_roots: Tensor,
    config: DPARConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Release top-k PPR rows for selected roots and identity rows for the rest."""
    if config.topk < 1:
        raise ValueError("topk must be positive")
    roots = ppr_roots.to(device=device, dtype=torch.long)
    if roots.ndim != 1 or roots.numel() == 0:
        raise ValueError("ppr_roots must be a non-empty one-dimensional tensor")
    if bool(torch.any(roots < 0)) or bool(torch.any(roots >= num_nodes)):
        raise ValueError("ppr_roots must index the sampled training graph")
    if torch.unique(roots).numel() != roots.numel():
        raise ValueError("ppr_roots must be unique")

    adjacency = _dpar_adjacency(edge_index, num_nodes, device)
    transpose = adjacency.transpose(0, 1).coalesce()
    out_degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    inverse_degree = out_degree.clamp_min(1e-12).reciprocal()
    rows: list[Tensor] = []
    cols: list[Tensor] = []
    values: list[Tensor] = []
    # The released stopping rule has no iteration cap. A high cap turns genuine
    # non-convergence into a clear error instead of silently releasing partial PPR.
    max_iterations = max(10_000, num_nodes * 10)
    for root in roots.tolist():
        p = torch.zeros(num_nodes, device=device)
        residual = torch.zeros(num_nodes, device=device)
        residual[root] = -config.alpha * inverse_degree[root]
        for _ in range(max_iterations):
            if residual.abs().amax() <= (1.0 + config.ista_epsilon) * config.rho * config.alpha:
                break
            active = p - residual >= config.rho * config.alpha
            if not bool(active.any()):
                break
            delta_pk = torch.zeros_like(residual)
            delta_pk[active] = -(residual[active] + config.rho * config.alpha)
            p[active] = p[active] + delta_pk[active]

            # This alias is intentional: the released recurrence assigns
            # delta_fp_new = delta_fp_old before updating active coordinates.
            next_residual = residual
            message = torch.sparse.mm(
                adjacency, (delta_pk / out_degree).unsqueeze(1)
            ).squeeze(1)
            next_residual[active] = (
                (1.0 - inverse_degree[active]) * residual[active]
                - config.rho * config.alpha * inverse_degree[active]
                - 0.5 * (1.0 - config.alpha) * delta_pk[active] * inverse_degree[active]
                - 0.5 * (1.0 - config.alpha) * message[active] * inverse_degree[active]
            )
            outgoing = torch.sparse.mm(
                transpose, active.to(torch.float32).unsqueeze(1)
            ).squeeze(1) > 0
            neighbours = outgoing & ~active
            next_residual[neighbours] = (
                residual[neighbours]
                - 0.5
                * (1.0 - config.alpha)
                * message[neighbours]
                * inverse_degree[neighbours]
            )
            residual = next_residual
        else:
            raise RuntimeError(
                f"DPAR ISTA did not converge for root {root} "
                f"after {max_iterations} iterations"
            )
        if config.dp_ppr:
            p = p * min(1.0, config.ppr_clip / float(p.norm().clamp_min(1e-12)))
            p = p + torch.randn(
                num_nodes, device=device, generator=generator
            ) * config.ppr_noise
        nonzero = torch.where(p != 0)[0]
        if nonzero.numel():
            chosen = nonzero[
                torch.topk(p[nonzero], k=min(config.topk, nonzero.numel())).indices
            ]
            rows.append(torch.full_like(chosen, root))
            cols.append(chosen)
            values.append(p[chosen])

    released = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    released[roots] = True
    identity_rows = torch.where(~released)[0]
    if identity_rows.numel():
        rows.append(identity_rows)
        cols.append(identity_rows)
        values.append(torch.ones(identity_rows.numel(), device=device))

    if not rows:
        raise RuntimeError("DPAR PPR preprocessing produced no entries")
    ppr = torch.sparse_coo_tensor(
        torch.stack((torch.cat(rows), torch.cat(cols))),
        torch.cat(values),
        (num_nodes, num_nodes),
        device=device,
        check_invariants=False,
    ).coalesce()
    column_norm = torch.zeros(num_nodes, device=device)
    column_norm.scatter_add_(0, ppr.indices()[1], ppr.values().abs())
    normalized = ppr.values() / column_norm[ppr.indices()[1]]
    return torch.sparse_coo_tensor(
        ppr.indices(),
        normalized,
        ppr.shape,
        device=device,
        check_invariants=False,
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
    adjacency = _dpar_adjacency(edge_index, logits.size(0), logits.device)
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1e-12)
    local = logits
    propagated = logits.clone()
    for _ in range(steps):
        propagated = (
            (1.0 - alpha) * torch.sparse.mm(adjacency, propagated) / degree[:, None]
            + alpha * local
        )
    return propagated


def _sample_train_partition(
    partition: Any,
    config: DPARConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Any, Tensor, dict[str, Any]]:
    """Sample the outer graph and identify the independently bounded PPR roots."""
    train_data = partition.data
    train_nodes = int(train_data.num_nodes)
    if train_nodes <= 0:
        raise ValueError("DPAR requires a non-empty train partition")
    if config.sampled_train_rate is not None:
        sampled_nodes = math.ceil(float(config.sampled_train_rate) * train_nodes)
    else:
        assert config.sampled_train_nodes is not None
        sampled_nodes = min(config.sampled_train_nodes, train_nodes)

    chosen = torch.randperm(
        train_nodes, device=device, generator=generator
    )[:sampled_nodes].cpu()
    mask = torch.zeros(train_nodes, dtype=torch.bool)
    mask[chosen] = True
    sampled_data, node_ids = _induce(train_data, mask)
    global_to_local = torch.full((train_nodes,), -1, dtype=torch.long)
    global_to_local[node_ids] = torch.arange(sampled_nodes)
    release_count = min(config.ppr_num, sampled_nodes)
    ppr_roots = global_to_local[chosen[:release_count]].to(device)
    sampled_data = sampled_data.to(device)
    return sampled_data, ppr_roots, graph_statistics(sampled_data)


class DPARTrainer:
    """DPAR trainer operating solely on a graph-disjoint train partition."""

    def __init__(self, config: DPARConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def _evaluate(self, model: DPARMLP, partition: Any) -> tuple[float, float]:
        data = partition.data.to(self.device)
        model.eval()
        with torch.no_grad():
            logits = propagate_logits(
                model(data.x), data.edge_index, self.config.alpha,
                self.config.inference_steps)
        logits, labels = _metric_rows(
            logits,
            data.y,
            eval_mask=getattr(partition, "eval_mask", None),
            metric_ignore_label=self.config.metric_ignore_label,
        )
        return _task_metric(
            logits, labels, self.config.multilabel,
            regression=self.config.regression, binary=self.config.binary)

    def fit(self, split: Any) -> dict[str, Any]:
        """Train, choose on the validation metric, and evaluate the held-out graph."""
        torch.manual_seed(self.config.seed)
        full_train_data = split.train.data
        outputs = 1 if (self.config.binary or self.config.regression) else split.num_classes
        model = DPARMLP(
            full_train_data.x.size(1), outputs, self.config.hidden_size,
            self.config.layers, self.config.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        sampling_generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)
        train_data, ppr_roots, sampled_train_graph = _sample_train_partition(
            split.train, self.config, sampling_generator, self.device,
        )
        sampled_nodes = int(train_data.num_nodes)
        ppr_releases = int(ppr_roots.numel())
        target_mode = self.config.target_epsilon is not None or self.config.target_delta is not None
        if target_mode and (self.config.target_epsilon is None or self.config.target_delta is None):
            raise ValueError("target_epsilon and target_delta must be provided together")
        calibration = None
        effective_config = self.config
        if target_mode:
            steps = self.config.epochs * math.ceil(sampled_nodes / self.config.batch_size)
            calibration = calibrate_dpar_noise(
                target_epsilon=self.config.target_epsilon, target_delta=self.config.target_delta,
                train_nodes=int(full_train_data.num_nodes), sampled_train_nodes=sampled_nodes,
                ppr_releases=ppr_releases, ppr_clip=self.config.ppr_clip,
                sgd_clip=self.config.sgd_clip, batch_size=self.config.batch_size, steps=steps,
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
        ppr = private_ista_ppr(
            train_data.edge_index, sampled_nodes, ppr_roots, effective_config,
            self.device, sampling_generator,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        # MAE is lower-is-better; a bare `>` would keep the worst checkpoint.
        lower_is_better = bool(effective_config.regression)
        best_state = None
        best_val = float("inf") if lower_is_better else float("-inf")
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
                    _task_loss(
                        logits, train_data.y[root_indices],
                        effective_config.multilabel,
                        regression=effective_config.regression,
                        binary=effective_config.binary).backward()
                    optimizer.step()
            val_metric, _ = self._evaluate(model, split.val)
            improved = (
                val_metric < best_val if lower_is_better else val_metric > best_val
            )
            if not math.isnan(val_metric) and improved:
                best_val = val_metric
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            elif best_state is None:
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        training_seconds = time.perf_counter() - training_start
        assert best_state is not None
        model.load_state_dict(best_state)
        validation, validation_secondary = self._evaluate(model, split.val)
        test, test_secondary = self._evaluate(model, split.test)
        privacy = self._privacy(
            int(full_train_data.num_nodes), sampled_nodes, ppr_releases, effective_config
        )
        result = {
            "method": "dpar", "config": asdict(effective_config),
            "preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds, "privacy": privacy,
            "train_graph": split.train.stats,
            "sampled_train_graph": sampled_train_graph,
        }
        if effective_config.binary:
            result.update({
                "metric": "auroc",
                "validation_auroc": validation,
                "validation_binary_accuracy": validation_secondary,
                "test_auroc": test,
                "test_binary_accuracy": test_secondary,
            })
        else:
            result.update({
                "validation_accuracy": validation,
                "validation_macro_f1": validation_secondary,
                "test_accuracy": test,
                "test_macro_f1": test_secondary,
            })
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
            gradients = torch.autograd.grad(
                _task_loss(
                    row.unsqueeze(0), target.unsqueeze(0), config.multilabel,
                    regression=config.regression, binary=config.binary),
                parameters, retain_graph=True)
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

    def _privacy(
        self,
        train_nodes: int,
        sampled_train_nodes: int,
        ppr_releases: int,
        config: DPARConfig,
    ) -> dict[str, Any]:
        accountant = DPARAccountant()
        amplification_rate = sampled_train_nodes / train_nodes
        ppr = accountant.account(
            ppr_releases=ppr_releases, amplification_rate=amplification_rate,
            delta=config.ppr_delta, ppr_clip=config.ppr_clip if config.dp_ppr else None,
            ppr_noise=config.ppr_noise if config.dp_ppr else None, topk=config.topk,
        )
        if config.dp_sgd:
            steps = config.epochs * math.ceil(sampled_train_nodes / config.batch_size)
            effective_batch_size = min(config.batch_size, sampled_train_nodes)
            sgd = accountant.account_training(
                noise_multiplier=config.sgd_noise / config.sgd_clip,
                sample_rate=effective_batch_size / sampled_train_nodes, steps=steps,
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
