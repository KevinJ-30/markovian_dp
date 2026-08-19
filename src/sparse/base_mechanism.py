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

    #: What the train/val/test numbers from `evaluate` mean.  Recorded in the
    #: results CSV so a sweep over datasets with different targets (accuracy,
    #: micro-F1, AUROC) stays self-describing.
    metric_name: str = "accuracy"

    def __init__(self, module: torch.nn.Module, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.module = module.to(self.device)
        self.optimizer = None
        #: Optional edge_index override for `evaluate`.  None means evaluate on
        #: data.edge_index (the full graph).  run.py sets this to the actual
        #: training graph (inductive-filtered, deduplicated, degree-capped)
        #: when --eval_graph train is passed, so utility can be measured on the
        #: same graph the model was trained on.
        self.eval_edge_index = None

    #: Above this many (arc x feature) elements, full-graph evaluation switches
    #: from an edge_index to a CSR adjacency.  Message passing over an
    #: edge_index gathers x[edge_index[0]], materializing an [E, F] tensor: on
    #: Reddit that is 114.6M x 602 x 4B = 276 GB.  A CSR adjacency fuses the
    #: gather and scatter, and PyG returns identical values either way.
    _DENSE_MESSAGE_BUDGET = 250_000_000

    def eval_edges(self, data):
        """The adjacency `evaluate` should use (see `eval_edge_index`).

        Returns an edge_index normally, or a CSR adjacency when the dense
        message tensor would be too large to allocate.
        """
        ei = (self.eval_edge_index if self.eval_edge_index is not None
              else data.edge_index)
        if not hasattr(data, 'x') or data.x is None:
            return ei
        if ei.size(1) * data.x.size(1) <= self._DENSE_MESSAGE_BUDGET:
            return ei

        cached = getattr(self, '_eval_adj_cache', None)
        if cached is not None and cached[0] is ei:
            return cached[1]
        from torch_geometric.utils import to_torch_csr_tensor
        # PyG expects adj_t[target, source]; edge_index is (source, target).
        n = int(data.num_nodes)
        adj_t = to_torch_csr_tensor(ei.flip(0), size=(n, n))
        self._eval_adj_cache = (ei, adj_t)
        return adj_t

    # ── parameters / optimizer ────────────────────────────────────────────────

    def parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def build_optimizer(self, lr: float, weight_decay: float = 0.0,
                        kind: str = "adam",
                        momentum: float = 0.0) -> torch.optim.Optimizer:
        if kind == "adam":
            self.optimizer = torch.optim.Adam(
                self.module.parameters(), lr=lr, weight_decay=weight_decay)
        elif kind == "sgd":
            # Momentum is a data-independent function of past (already noised)
            # updates, i.e. post-processing — no privacy cost.  With DP noise
            # it acts as an averaging filter over ~1/(1-momentum) steps.
            self.optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, weight_decay=weight_decay,
                momentum=momentum)
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

        The draw happens on CPU and is then moved to the gradient's device:
        `generator` is a CPU torch.Generator, and torch.randn requires the
        generator's device to match the output device, so drawing directly on
        CUDA would raise.  CPU draws also make the noise stream identical
        across CPU and GPU runs for a given seed.
        """
        std = sigma * C
        return [(torch.randn(g.shape, generator=generator) * std).to(g.device)
                for g in grads]
