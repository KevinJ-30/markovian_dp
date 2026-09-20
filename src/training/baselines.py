"""Portable non-private and DP-SGD baselines for the common inductive split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from src.models.objectives import _task_loss, _task_metric
from src.privacy.accountants import DPMLPAccountant

from src.models.baselines import MLP, GraphSAGE

@dataclass(frozen=True)
class BaselineConfig:
    method: str
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.5
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    epochs: int = 100
    batch_size: int = 256
    noise_multiplier: float = 1.0
    clip: float = 1.0
    delta: float = 1e-5
    seed: int = 0
    multilabel: bool = False
    regression: bool = False


class BaselineTrainer:
    """Train/evaluate graph partitions without exposing held-out graphs to fitting."""

    def __init__(self, config: BaselineConfig, device: str | torch.device = "cpu"):
        if config.method not in {"mlp", "graphsage", "dp_mlp"}:
            raise ValueError(f"unsupported portable baseline {config.method!r}")
        self.config = config
        self.device = torch.device(device)

    def _model(self, train_data: Any, num_classes: int) -> nn.Module:
        factory = GraphSAGE if self.config.method == "graphsage" else MLP
        return factory(train_data.x.size(1), num_classes, self.config.hidden_size,
                       self.config.layers, self.config.dropout).to(self.device)

    def _forward(self, model: nn.Module, data: Any) -> Tensor:
        return model(data.x, getattr(data, "edge_index", None))

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, partition: Any) -> tuple[float, float]:
        data = partition.data.to(self.device)
        model.eval()
        return _task_metric(self._forward(model, data), data.y, self.config.multilabel,
                           regression=self.config.regression)

    def fit(self, split: Any) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        train = split.train.data.to(self.device)
        model = self._model(train, split.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)
        steps_per_epoch = math.ceil(train.num_nodes / self.config.batch_size)
        # MAE is lower-is-better; a bare `>` would keep the worst checkpoint.
        lower_is_better = bool(self.config.regression)
        best_state = None
        best_val = float("inf") if lower_is_better else float("-inf")
        started = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            if self.config.method == "dp_mlp":
                for _ in range(steps_per_epoch):
                    self._private_step(model, optimizer, train, generator)
            else:
                # Same step budget as dp_mlp.  One full-batch step per epoch
                # meant the non-private ceiling trained ~1000x less than it.
                for _ in range(steps_per_epoch):
                    self._step(model, optimizer, train, generator)
            validation, _ = self._evaluate(model, split.val)
            improved = (validation < best_val) if lower_is_better else (validation > best_val)
            if improved:
                best_val = validation
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        training_seconds = time.perf_counter() - started
        assert best_state is not None
        model.load_state_dict(best_state)
        validation, val_f1 = self._evaluate(model, split.val)
        test, test_f1 = self._evaluate(model, split.test)
        privacy = None
        if self.config.method == "dp_mlp":
            privacy = DPMLPAccountant().account(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=min(self.config.batch_size / train.num_nodes, 1.0),
                steps=self.config.epochs * steps_per_epoch, delta=self.config.delta,
            ).as_dict()
        return {
            "method": self.config.method, "config": asdict(self.config),
            "validation_accuracy": validation, "validation_macro_f1": val_f1,
            "test_accuracy": test, "test_macro_f1": test_f1,
            "preprocessing_seconds": 0.0, "training_seconds": training_seconds,
            "privacy": privacy, "train_graph": split.train.stats,
        }

    def _step(self, model: nn.Module, optimizer: torch.optim.Optimizer, data: Any,
              generator: torch.Generator) -> None:
        """One non-private minibatch step.

        GraphSAGE needs the graph, so it keeps the full-graph forward and reads
        the loss off the sampled rows; the MLP forwards only those rows.
        """
        selected = torch.randint(int(data.num_nodes), (self.config.batch_size,),
                                 device=self.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        if self.config.method == "graphsage":
            out = self._forward(model, data)[selected]
        else:
            out = model(data.x[selected])
        _task_loss(out, data.y[selected], self.config.multilabel,
                   regression=self.config.regression).backward()
        optimizer.step()

    def _per_sample_grads(self, model: nn.Module, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Per-example gradients in one vmapped pass (leading dim = sample)."""
        params = {name: p.detach() for name, p in model.named_parameters()}
        buffers = {name: b.detach() for name, b in model.named_buffers()}

        def loss_of_one(p, b, xi, yi):
            out = functional_call(model, (p, b), (xi.unsqueeze(0),))
            return _task_loss(out, yi.unsqueeze(0), self.config.multilabel,
                              regression=self.config.regression)

        # Each example needs its own dropout mask.
        return vmap(grad(loss_of_one), in_dims=(None, None, 0, 0),
                    randomness='different')(params, buffers, x, y)

    def _private_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, data: Any,
                      generator: torch.Generator) -> None:
        """Poisson-sampled per-example DP-SGD for the feature-only MLP.

        Forwards only the sampled rows, and takes all per-example gradients in
        one vmapped pass instead of a Python loop of autograd.grad calls over a
        full-graph forward.
        """
        sample_rate = min(self.config.batch_size / data.num_nodes, 1.0)
        selected = torch.where(torch.rand(data.num_nodes, device=self.device, generator=generator) < sample_rate)[0]
        if not selected.numel():
            return
        grads = self._per_sample_grads(model, data.x[selected], data.y[selected])
        flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)
        scale = (self.config.clip / flat.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
        named = dict(model.named_parameters())
        optimizer.zero_grad(set_to_none=True)
        for name, per_sample in grads.items():
            accumulator = torch.einsum('i,i...->...', scale, per_sample)
            noise = torch.randn(accumulator.shape, dtype=accumulator.dtype, device=accumulator.device,
                                generator=generator) * (self.config.noise_multiplier * self.config.clip)
            named[name].grad = (accumulator + noise) / selected.numel()
        optimizer.step()
