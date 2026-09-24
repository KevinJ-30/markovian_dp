"""Portable non-private and DP-SGD baselines for the common inductive split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from src.models.bootstrap import BootstrapConfig, BootstrapMetrics
from src.models.objectives import _metric_rows, _task_loss, _task_metric
from src.privacy.accountants import DPMLPAccountant

from src.models.baselines import GIN, MLP, GraphSAGE

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
    binary: bool = False
    metric_ignore_label: int | None = None
    graphsage_sampling: str = "hierarchical"
    max_fanout: int = 10
    bootstrap_confidence: float = 0.95
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        BootstrapConfig(
            confidence_level=self.bootstrap_confidence,
            n_resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed,
        )


@dataclass(frozen=True)
class _SampledNeighborhood:
    node_ids: Tensor
    edge_index: Tensor
    num_sampled_nodes: list[int]
    num_sampled_edges: list[int]


class _LayerwiseNeighborSampler:
    """Compiled fixed-fanout sampling with seed-first BFS node ordering."""

    def __init__(self, edge_index: Tensor, num_nodes: int):
        edges = edge_index.detach().cpu().to(torch.long)
        if edges.ndim != 2 or edges.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        order = torch.argsort(edges[1], stable=True)
        self.neighbors = edges[0, order]
        counts = torch.bincount(edges[1], minlength=num_nodes)
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.num_nodes = int(num_nodes)

    def sample(
        self,
        roots: Tensor,
        *,
        fanouts: list[int],
        generator: torch.Generator,
    ) -> _SampledNeighborhood:
        roots = roots.detach().cpu().to(torch.long).view(-1)
        if roots.numel() == 0:
            raise ValueError("roots must be nonempty")
        if torch.unique(roots).numel() != roots.numel():
            raise ValueError("roots must be unique")
        if bool(torch.any(roots < 0)) or bool(torch.any(roots >= self.num_nodes)):
            raise ValueError("roots contain an invalid node index")
        if any(fanout <= 0 for fanout in fanouts):
            raise ValueError("fanouts must be positive")

        from pyg_lib.sampler import neighbor_sample

        # pyg-lib uses the global CPU RNG; isolate it from model/dropout draws.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            source, target, node_ids, _, node_counts, edge_counts = neighbor_sample(
                self.offsets, self.neighbors, roots, fanouts,
                csc=True, replace=False, directed=True, return_edge_id=False,
            )
            generator.set_state(torch.get_rng_state())

        return _SampledNeighborhood(
            node_ids=node_ids,
            edge_index=torch.stack((source, target)),
            num_sampled_nodes=node_counts,
            num_sampled_edges=edge_counts,
        )


class BaselineTrainer:
    """Train/evaluate graph partitions without exposing held-out graphs to fitting."""

    def __init__(self, config: BaselineConfig, device: str | torch.device = "cpu"):
        if config.method not in {"mlp", "graphsage", "gin", "dp_mlp"}:
            raise ValueError(f"unsupported portable baseline {config.method!r}")
        if sum((config.multilabel, config.regression, config.binary)) > 1:
            raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
        if config.method in {"graphsage", "gin"}:
            if config.graphsage_sampling not in {"neighbor", "hierarchical"}:
                raise ValueError(
                    "graphsage_sampling must be 'neighbor' or 'hierarchical'"
                )
            if (
                isinstance(config.max_fanout, bool)
                or not isinstance(config.max_fanout, int)
                or config.max_fanout <= 0
            ):
                raise ValueError("max_fanout must be a positive integer")
        self.config = config
        self.device = torch.device(device)

    def _model(self, train_data: Any, num_classes: int) -> nn.Module:
        if self.config.method == "graphsage":
            factory = GraphSAGE
        elif self.config.method == "gin":
            factory = GIN
        else:
            factory = MLP
        outputs = 1 if (self.config.binary or self.config.regression) else num_classes
        return factory(train_data.x.size(1), outputs, self.config.hidden_size,
                       self.config.layers, self.config.dropout).to(self.device)

    def _forward(self, model: nn.Module, data: Any) -> Tensor:
        return model(data.x, getattr(data, "edge_index", None))

    @torch.no_grad()
    def _evaluate(
        self, model: nn.Module, partition: Any, *,
        bootstrap: BootstrapMetrics | None = None,
    ) -> tuple[float, float]:
        data = partition.data.to(self.device)
        model.eval()
        logits = self._forward(model, data)
        logits, labels = _metric_rows(
            logits,
            data.y,
            eval_mask=getattr(partition, "eval_mask", None),
            metric_ignore_label=self.config.metric_ignore_label,
        )
        if bootstrap is not None:
            bootstrap.update(logits, labels)
        return _task_metric(
            logits, labels, self.config.multilabel,
            regression=self.config.regression, binary=self.config.binary)

    def fit(self, split: Any) -> dict[str, Any]:
        torch.manual_seed(self.config.seed)
        train_cpu = split.train.data
        train = (
            train_cpu.to(self.device, "x", "y")
            if self.config.method in {"graphsage", "gin"}
            else train_cpu.to(self.device)
        )
        model = self._model(train_cpu, split.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)
        sampling_generator = torch.Generator().manual_seed(self.config.seed + 1)
        steps_per_epoch = math.ceil(train.num_nodes / self.config.batch_size)
        sampler = (
            _LayerwiseNeighborSampler(train.edge_index, int(train.num_nodes))
            if self.config.method in {"graphsage", "gin"}
            else None
        )
        best_state = None
        best_val = float("-inf")
        started = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            if self.config.method in {"graphsage", "gin"}:
                assert sampler is not None
                self._sampled_gnn_epoch(
                    model, optimizer, train, sampler, sampling_generator
                )
            elif self.config.method == "dp_mlp":
                for _ in range(steps_per_epoch):
                    self._private_step(model, optimizer, train, generator)
            else:
                for _ in range(steps_per_epoch):
                    self._step(model, optimizer, train, generator)
            validation, _ = self._evaluate(model, split.val)
            if not math.isnan(validation) and validation > best_val:
                best_val = validation
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            elif best_state is None:
                # Preserve an explicit NaN metric for an unscorable validation
                # partition while still returning a trained model.
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        training_seconds = time.perf_counter() - started
        assert best_state is not None
        model.load_state_dict(best_state)
        validation, val_f1 = self._evaluate(model, split.val)
        bootstrap = (
            BootstrapMetrics(
                "r2" if self.config.regression else
                "auroc" if self.config.binary else
                "micro_f1" if self.config.multilabel else "accuracy",
                BootstrapConfig(
                    confidence_level=self.config.bootstrap_confidence,
                    n_resamples=self.config.bootstrap_resamples,
                    seed=self.config.bootstrap_seed,
                ),
            )
            if self.config.bootstrap_resamples else None
        )
        test, test_f1 = self._evaluate(model, split.test, bootstrap=bootstrap)
        privacy = None
        if self.config.method == "dp_mlp":
            privacy = DPMLPAccountant().account(
                noise_multiplier=self.config.noise_multiplier,
                sample_rate=min(self.config.batch_size / train.num_nodes, 1.0),
                steps=self.config.epochs * steps_per_epoch, delta=self.config.delta,
            ).as_dict()
        result = {
            "method": self.config.method, "config": asdict(self.config),
            "preprocessing_seconds": 0.0, "training_seconds": training_seconds,
            "privacy": privacy, "train_graph": split.train.stats,
        }
        if self.config.binary:
            result.update({
                "metric": "auroc",
                "validation_auroc": validation,
                "validation_binary_accuracy": val_f1,
                "test_auroc": test,
                "test_binary_accuracy": test_f1,
            })
        else:
            result.update({
                "validation_accuracy": validation, "validation_macro_f1": val_f1,
                "test_accuracy": test, "test_macro_f1": test_f1,
            })
        if self.config.regression:
            result["metric"] = "r2"
        if bootstrap is not None:
            result["test_confidence_intervals"] = bootstrap.compute()
        return result

    def _sampled_gnn_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: Any,
        sampler: _LayerwiseNeighborSampler,
        generator: torch.Generator,
    ) -> None:
        if not isinstance(model, (GraphSAGE, GIN)):
            raise TypeError("neighbor sampling requires a GraphSAGE or GIN model")
        roots = torch.randperm(int(data.num_nodes), generator=generator)
        fanouts = [self.config.max_fanout] * self.config.layers
        hierarchical = self.config.graphsage_sampling == "hierarchical"
        for start in range(0, roots.numel(), self.config.batch_size):
            sampled = sampler.sample(
                roots[start:start + self.config.batch_size],
                fanouts=fanouts,
                generator=generator,
            )
            node_ids = sampled.node_ids.to(self.device)
            x = data.x[node_ids]
            labels = data.y[node_ids[:sampled.num_sampled_nodes[0]]]
            edge_index = sampled.edge_index.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_sampled(
                x,
                edge_index,
                sampled.num_sampled_nodes,
                sampled.num_sampled_edges,
                hierarchical=hierarchical,
            )
            _task_loss(
                logits, labels, self.config.multilabel,
                regression=self.config.regression,
                binary=self.config.binary,
            ).backward()
            optimizer.step()

    def _step(self, model: nn.Module, optimizer: torch.optim.Optimizer, data: Any,
              generator: torch.Generator) -> None:
        """One non-private feature minibatch step."""
        selected = torch.randint(int(data.num_nodes), (self.config.batch_size,),
                                 device=self.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x[selected])
        _task_loss(
            out, data.y[selected], self.config.multilabel,
            regression=self.config.regression, binary=self.config.binary).backward()
        optimizer.step()

    def _per_sample_grads(self, model: nn.Module, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Per-example gradients in one vmapped pass (leading dim = sample)."""
        params = {name: p.detach() for name, p in model.named_parameters()}
        buffers = {name: b.detach() for name, b in model.named_buffers()}

        def loss_of_one(p, b, xi, yi):
            out = functional_call(model, (p, b), (xi.unsqueeze(0),))
            return _task_loss(
                out, yi.unsqueeze(0), self.config.multilabel,
                regression=self.config.regression, binary=self.config.binary)

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
