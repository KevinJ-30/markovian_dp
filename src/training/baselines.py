"""Portable non-private and DP-SGD baselines for the common inductive split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from src.models.objectives import _metric_rows, _task_loss, _task_metric
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
    binary: bool = False
    metric_ignore_label: int | None = None
    graphsage_sampling: str = "hierarchical"
    max_fanout: int = 10


@dataclass(frozen=True)
class _SampledNeighborhood:
    node_ids: Tensor
    edge_index: Tensor
    num_sampled_nodes: list[int]
    num_sampled_edges: list[int]


class _LayerwiseNeighborSampler:
    """Uniform fixed-fanout sampling with seed-first BFS node ordering."""

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

        node_ids = roots.tolist()
        local_index = {node: index for index, node in enumerate(node_ids)}
        frontier = list(node_ids)
        node_counts = [len(node_ids)]
        edge_counts: list[int] = []
        edge_groups: list[Tensor] = []

        for fanout in fanouts:
            next_frontier: list[int] = []
            local_sources: list[int] = []
            local_targets: list[int] = []
            for target_node in frontier:
                start = int(self.offsets[target_node])
                stop = int(self.offsets[target_node + 1])
                degree = stop - start
                if degree == 0:
                    continue
                if degree <= fanout:
                    selected = self.neighbors[start:stop]
                else:
                    selected = self.neighbors[
                        start + torch.randperm(degree, generator=generator)[:fanout]
                    ]
                target_index = local_index[target_node]
                for source_node in selected.tolist():
                    source_index = local_index.get(source_node)
                    if source_index is None:
                        source_index = len(node_ids)
                        local_index[source_node] = source_index
                        node_ids.append(source_node)
                        next_frontier.append(source_node)
                    local_sources.append(source_index)
                    local_targets.append(target_index)
            edge_count = len(local_sources)
            edge_counts.append(edge_count)
            edge_groups.append(
                torch.tensor([local_sources, local_targets], dtype=torch.long)
                if edge_count
                else torch.empty((2, 0), dtype=torch.long)
            )
            node_counts.append(len(next_frontier))
            frontier = next_frontier

        return _SampledNeighborhood(
            node_ids=torch.tensor(node_ids, dtype=torch.long),
            edge_index=torch.cat(edge_groups, dim=1),
            num_sampled_nodes=node_counts,
            num_sampled_edges=edge_counts,
        )


class BaselineTrainer:
    """Train/evaluate graph partitions without exposing held-out graphs to fitting."""

    def __init__(self, config: BaselineConfig, device: str | torch.device = "cpu"):
        if config.method not in {"mlp", "graphsage", "dp_mlp"}:
            raise ValueError(f"unsupported portable baseline {config.method!r}")
        if sum((config.multilabel, config.regression, config.binary)) > 1:
            raise ValueError("binary, multilabel, and regression tasks are mutually exclusive")
        if config.method == "graphsage":
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
        factory = GraphSAGE if self.config.method == "graphsage" else MLP
        outputs = 1 if (self.config.binary or self.config.regression) else num_classes
        return factory(train_data.x.size(1), outputs, self.config.hidden_size,
                       self.config.layers, self.config.dropout).to(self.device)

    def _forward(self, model: nn.Module, data: Any) -> Tensor:
        return model(data.x, getattr(data, "edge_index", None))

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, partition: Any) -> tuple[float, float]:
        data = partition.data.to(self.device)
        model.eval()
        logits = self._forward(model, data)
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
        torch.manual_seed(self.config.seed)
        train_cpu = split.train.data
        train = (
            train_cpu
            if self.config.method == "graphsage"
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
            if self.config.method == "graphsage"
            else None
        )
        best_state = None
        best_val = float("-inf")
        started = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            if self.config.method == "graphsage":
                assert sampler is not None
                self._graphsage_epoch(
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
        test, test_f1 = self._evaluate(model, split.test)
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
        return result

    def _graphsage_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: Any,
        sampler: _LayerwiseNeighborSampler,
        generator: torch.Generator,
    ) -> None:
        if not isinstance(model, GraphSAGE):
            raise TypeError("GraphSAGE sampling requires a GraphSAGE model")
        roots = torch.randperm(int(data.num_nodes), generator=generator)
        fanouts = [self.config.max_fanout] * self.config.layers
        hierarchical = self.config.graphsage_sampling == "hierarchical"
        for start in range(0, roots.numel(), self.config.batch_size):
            sampled = sampler.sample(
                roots[start:start + self.config.batch_size],
                fanouts=fanouts,
                generator=generator,
            )
            node_ids = sampled.node_ids
            x = data.x[node_ids].to(self.device)
            labels = data.y[node_ids[:sampled.num_sampled_nodes[0]]].to(
                self.device
            )
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
