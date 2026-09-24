"""
BaseMechanism: the model-agnostic base mechanism g0 for SparseGNN.

Assumption 3.2 of the paper factors every learning update through a per-subgraph
gradient function g0: G(y) = sum_v g0(y_v), with ||g0(H)||_2 <= C. Concrete
node-classification mechanisms implement this interface, while SparseGNN and
the DP clip/noise machinery remain shared.

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

from .bootstrap import BootstrapMetrics


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
        # Physical padding budget for the private root-first path.  It bounds
        # B * N_max per forward chunk; logical DP batches may span chunks.
        self.max_private_batch_nodes = 8192

    #: Above this many (arc x feature) elements, full-graph evaluation switches
    #: from an edge_index to a CSR adjacency.  Message passing over an
    #: edge_index gathers x[edge_index[0]], materializing an [E, F] tensor: on
    #: Reddit that is 114.6M x 602 x 4B = 276 GB.  A CSR adjacency fuses the
    #: gather and scatter, and PyG returns identical values either way.
    _DENSE_MESSAGE_BUDGET = 250_000_000

    def eval_edges(self, data):
        """Return evaluation edges, using CSR when dense messages are too large."""
        ei = data.edge_index
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

    def subgraph_losses(self, subgraphs) -> List[torch.Tensor]:
        """Return per-root losses in the supplied order."""
        return [self.subgraph_loss(subgraph) for subgraph in subgraphs]

    def iter_subgraph_loss_batches(self, subgraphs):
        """Yield loss batches without retaining a mechanism-specific contract."""
        if subgraphs:
            yield self.subgraph_losses(subgraphs)

    def build_private_module(self) -> torch.nn.Module:
        """Return a batch-first module sharing this mechanism's parameters."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement private padded training")

    def private_losses(self, private_module: torch.nn.Module, batch) -> torch.Tensor:
        """Return one scalar loss per padded rooted-subgraph sample."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement private padded training")

    @abstractmethod
    def evaluate(self, data, *, splits=("train", "val", "test"),
                 bootstrap: BootstrapMetrics = None) -> Dict[str, float]:
        """Score only requested splits; optionally accumulate test nodes for CIs."""
        ...

    def zero_loss(self) -> torch.Tensor:
        """A differentiable zero for subgraphs with no supervision signal."""
        params = self.parameters()
        if params:
            return params[0].sum() * 0.0
        return torch.zeros((), device=self.device, requires_grad=True)

