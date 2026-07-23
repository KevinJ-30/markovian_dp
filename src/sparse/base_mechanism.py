"""
BaseMechanism: the model-agnostic base mechanism g0 for SparseGNN.

Assumption 3.2 of the paper factors every learning update through a per-subgraph
gradient function g0:  G(y) = sum_v g0(y_v), with ||g0(H)||_2 <= C.  A GNN node
classifier is one instantiation of g0; a non-GNN graph-anomaly detector is
another.  Everything specific to the choice of g0 lives behind this interface,
so the SparseGNN engine (sparse_gnn.py) and the DP clip/noise machinery are
shared across models.

A concrete mechanism must supply:
    * an nn.Module (or parameter list) via `parameters()`
    * `subgraph_loss(subgraph)` -> scalar tensor   (the per-subgraph g0 loss)
    * `evaluate(data)` -> dict of metrics

The base class provides the shared optimizer, gradient flattening / clipping,
and Gaussian-noise helpers used by the DP path in the engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class BaseMechanism(ABC):
    """Abstract base mechanism g0 consumed by the SparseGNN engine."""

    def __init__(self, module: torch.nn.Module, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.module = module.to(self.device)
        self.optimizer = None

    # ── parameters / optimizer ────────────────────────────────────────────────

    def parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def build_optimizer(self, lr: float, weight_decay: float = 0.0,
                        kind: str = "adam") -> torch.optim.Optimizer:
        if kind == "adam":
            self.optimizer = torch.optim.Adam(
                self.module.parameters(), lr=lr, weight_decay=weight_decay)
        elif kind == "sgd":
            self.optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unknown optimizer kind '{kind}'")
        return self.optimizer

    def train_mode(self):
        self.module.train()

    def eval_mode(self):
        self.module.eval()

    # ── model-specific pieces (implemented by subclasses) ─────────────────────

    @abstractmethod
    def subgraph_loss(self, subgraph) -> torch.Tensor:
        """g0 loss for a single RootedSubgraph.

        Returns a scalar tensor whose gradient w.r.t. `parameters()` is the
        subgraph's contribution g0(H).  Subgraphs that carry no supervision
        (e.g. an unlabeled root) should return a zero scalar that still
        participates in autograd (see `zero_loss`).
        """
        ...

    @abstractmethod
    def evaluate(self, data) -> Dict[str, float]:
        """Return a dict of evaluation metrics (e.g. train/val/test accuracy)."""
        ...

    def zero_loss(self) -> torch.Tensor:
        """A differentiable zero, for subgraphs with no supervision signal."""
        return torch.zeros((), device=self.device)

    # ── shared DP helpers (used by the engine's DP path) ──────────────────────

    def clip_flat_grad(self, grads: List[torch.Tensor], C: float) -> List[torch.Tensor]:
        """Clip a per-subgraph gradient list to global L2 norm C (in place-safe).

        Returns a new list of tensors scaled by min(1, C / ||g||_2), matching
        the per-example clipping in Assumption 3.2 (||g0(H)||_2 <= C).
        """
        total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
        norm = float(total_sq.sqrt())
        coef = min(1.0, C / (norm + 1e-12))
        return [g * coef for g in grads]

    def gaussian_noise_like(self, grads: List[torch.Tensor], sigma: float,
                            C: float, generator: torch.Generator = None
                            ) -> List[torch.Tensor]:
        """Draw N(0, (sigma*C)^2 I) noise shaped like `grads` (Alg adds noise).

        The Gaussian base mechanism of Assumption 3.2 has covariance sigma^2 C^2 I.
        """
        std = sigma * C
        return [torch.randn(g.shape, generator=generator, device=g.device) * std
                for g in grads]
