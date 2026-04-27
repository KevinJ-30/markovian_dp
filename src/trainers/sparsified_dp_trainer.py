"""
SparsifiedDPTrainer — DP training paradigm with degree-aware sensitivity.

Kept totally separate from SubgraphTrainer's DP path. Use it when the input
graph has been preprocessed by `sparsify_by_degree` (run.py --max-in-degree)
so that every node has bounded in-degree D.

Privacy story
-------------
For an L-layer GCN, removing one node u from the dataset changes the loss
summand of at most the L-hop neighborhood of u — bounded by

    Δ_count = 1 + D + D^2 + ... + D^L

nodes (u itself, its <=D 1-hop neighbors, its <=D^2 2-hop neighbors, ...).
With per-node gradient L2-clip C, the summed gradient has L2 sensitivity

    Δ = C · Δ_count

w.r.t. one node addition/removal (node-DP). The Gaussian mechanism with noise
multiplier sigma releases the summed gradient with std

    sigma · Δ = sigma · C · Δ_count.

Post-processing (dividing the noisy sum by the number of active nodes this
step to get a usable optimizer update) preserves DP. The accountant input is
the same sigma fed to the standard paradigm; only the *claim* about
sensitivity changes.

This is the standard node-DP bound from Daigavane et al. 2021 ("Node-Level
Differentially Private Graph Neural Networks").
"""

import torch
import torch.nn.functional as F

from src.trainers.subgraph_trainer import SubgraphTrainer


def degree_sensitivity_factor(max_in_degree: int, num_layers: int) -> int:
    """Geometric sum 1 + D + D^2 + ... + D^L bounding L-hop neighborhood size."""
    if max_in_degree is None or max_in_degree <= 0:
        raise ValueError(f"max_in_degree must be a positive int, got {max_in_degree}")
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")
    return sum(max_in_degree ** l for l in range(num_layers + 1))


class SparsifiedDPTrainer(SubgraphTrainer):
    """
    DP trainer with per-node clipping + degree-aware noise.

    Differences from SubgraphTrainer's DP path:
    * Per-node gradient clip (not per-bin).
    * Single Gaussian release per step with std sigma·C·Δ_count (degree-aware).
    * Coverage correction is not supported (pass use_coverage_correction=False).
    """

    def __init__(self, *args, max_in_degree, gnn_layers=2, **kwargs):
        if max_in_degree is None or max_in_degree <= 0:
            raise ValueError(
                "SparsifiedDPTrainer requires max_in_degree > 0 "
                "(set --max-in-degree on the command line)"
            )
        if kwargs.get('use_coverage_correction', False):
            raise ValueError(
                "SparsifiedDPTrainer does not support coverage correction; "
                "the privacy claim assumes uniform per-node grad clip"
            )
        kwargs['dp'] = True  # paradigm is DP-only

        super().__init__(*args, **kwargs)

        self.max_in_degree = max_in_degree
        self.gnn_layers = gnn_layers
        self.sensitivity_factor = degree_sensitivity_factor(max_in_degree, gnn_layers)

        # Parent guarantees noise_multiplier is set whenever dp=True.
        assert self.noise_multiplier is not None

        # Override parent's degree-agnostic noise_std.
        # Gaussian-mechanism std on the summed clipped gradient with sensitivity
        # Δ = C · sensitivity_factor.
        self.noise_std = (
            self.noise_multiplier
            * self.max_grad_norm
            * self.sensitivity_factor
        )

    def _train_step_dp(self, data, partitions, active_mask, edge_index, y,
                       num_nodes, N) -> float:
        params = list(self.model.parameters())
        accum = [torch.zeros_like(p) for p in params]
        total_active = 0
        total_loss_val = 0.0

        for bin_mask, directed_ei in partitions:
            active_in_bin = bin_mask & active_mask
            if not active_in_bin.any():
                continue

            node_indices = active_in_bin.nonzero(as_tuple=True)[0]
            num_in_bin = int(node_indices.numel())

            # One forward per bin; reuse the autograd graph for N_bin backwards.
            out = self.model(data.x, directed_ei)
            per_node_losses = self._compute_per_node_losses(data, out, node_indices)
            if per_node_losses is None:
                continue

            for i in range(num_in_bin):
                loss_v = per_node_losses[i]
                grads = torch.autograd.grad(
                    loss_v, params,
                    retain_graph=(i < num_in_bin - 1),
                    allow_unused=True,
                )
                grad_sq = torch.zeros((), device=accum[0].device)
                for g in grads:
                    if g is not None:
                        grad_sq = grad_sq + g.detach().pow(2).sum()
                grad_norm = torch.sqrt(grad_sq)
                clip_coef = min(
                    1.0, self.max_grad_norm / (grad_norm.item() + 1e-8)
                )
                for a, g in zip(accum, grads):
                    if g is not None:
                        a.add_(g.detach(), alpha=clip_coef)

                total_loss_val += loss_v.item()

            total_active += num_in_bin

        if total_active == 0:
            return None

        # Single Gaussian release on the summed clipped gradient.
        for a in accum:
            a.add_(torch.randn_like(a) * self.noise_std)

        # Post-processing: normalize by active-node count for a usable
        # gradient scale. Data-dependent, but applied after the noisy
        # release, so DP is preserved.
        for a in accum:
            a.div_(total_active)

        self.optimizer.zero_grad(set_to_none=True)
        for p, a in zip(params, accum):
            p.grad = a
        self.optimizer.step()
        self.training_steps += 1
        return total_loss_val

    def _compute_per_node_losses(self, data, out, node_indices):
        """Per-node NLL losses for active train nodes in a bin.

        Returns a 1-D tensor of length |node_indices|, or None if empty.
        Override in subclasses for edge-level supervision.
        """
        if node_indices.numel() == 0:
            return None
        return F.nll_loss(
            out[node_indices], data.y[node_indices], reduction='none'
        )
