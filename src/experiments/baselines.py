"""Portable non-private and DP-SGD baselines for the common inductive split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dpar import _accuracy_and_macro_f1
from .privacy import DPMLPAccountant


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


class MLP(nn.Module):
    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GraphSAGE(nn.Module):
    """Mean-aggregating GraphSAGE without a PyG runtime dependency."""

    def __init__(self, inputs: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        widths = [inputs] + [hidden] * (layers - 1) + [classes]
        self.self_layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.neighbor_layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    @staticmethod
    def _mean_neighbors(x: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        sums = torch.zeros_like(x)
        sums.index_add_(0, target, x[source])
        degree = torch.bincount(target, minlength=x.size(0)).to(x.dtype).clamp_min_(1)
        return sums / degree[:, None]

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        if edge_index is None:
            raise ValueError("GraphSAGE requires edge_index")
        for index, (self_layer, neighbor_layer) in enumerate(zip(self.self_layers, self.neighbor_layers)):
            x = self_layer(x) + neighbor_layer(self._mean_neighbors(x, edge_index))
            if index < len(self.self_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


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
        return _accuracy_and_macro_f1(self._forward(model, data), data.y)

    def fit(self, split: Any) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        train = split.train.data.to(self.device)
        model = self._model(train, split.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)
        steps_per_epoch = math.ceil(train.num_nodes / self.config.batch_size)
        best_state, best_val = None, float("-inf")
        started = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            if self.config.method == "dp_mlp":
                for _ in range(steps_per_epoch):
                    self._private_step(model, optimizer, train, generator)
            else:
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(self._forward(model, train), train.y).backward()
                optimizer.step()
            validation, _ = self._evaluate(model, split.val)
            if validation > best_val:
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

    def _private_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, data: Any,
                      generator: torch.Generator) -> None:
        """Poisson-sampled per-example DP-SGD for the feature-only MLP."""
        sample_rate = min(self.config.batch_size / data.num_nodes, 1.0)
        selected = torch.where(torch.rand(data.num_nodes, device=self.device, generator=generator) < sample_rate)[0]
        if not selected.numel():
            return
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        clipped = [torch.zeros_like(parameter) for parameter in parameters]
        logits = model(data.x)
        for row, target in zip(logits[selected], data.y[selected]):
            gradients = torch.autograd.grad(F.cross_entropy(row[None], target[None]), parameters,
                                            retain_graph=True)
            norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients)).clamp_min(1e-12)
            scale = min(1.0, self.config.clip / float(norm))
            for accumulator, gradient in zip(clipped, gradients):
                accumulator.add_(gradient, alpha=scale)
        optimizer.zero_grad(set_to_none=True)
        for parameter, accumulator in zip(parameters, clipped):
            noise = torch.randn(accumulator.shape, dtype=accumulator.dtype, device=accumulator.device,
                                generator=generator) * (self.config.noise_multiplier * self.config.clip)
            parameter.grad = (accumulator + noise) / selected.numel()
        optimizer.step()
