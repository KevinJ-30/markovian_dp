"""
Subgraph trainer — trains on random subgraph partitions with optional coverage correction.
Accepts a pluggable SubgraphAlgorithm for partitioning.
"""

import math

import torch
import torch.nn.functional as F

from src.utils import compute_full_degrees


class SubgraphTrainer:
    def __init__(self, model, optimizer, num_bins, algorithm,
                 use_coverage_correction=True,
                 use_epoch_assignment=False,
                 poisson_subsampling=False,
                 single_phase_poisson=False,
                 q=1.0,
                 q_epoch=1.0,
                 q_step=1.0,
                 steps_per_epoch=10,
                 device='cpu',
                 dp=False,
                 max_grad_norm=1.0,
                 epsilon=None,
                 delta=1e-5,
                 noise_multiplier=None):
        self.model = model
        self.optimizer = optimizer
        self.num_bins = num_bins
        self.algorithm = algorithm
        self.use_coverage_correction = use_coverage_correction
        self.use_epoch_assignment = use_epoch_assignment
        self.poisson_subsampling = poisson_subsampling
        self.single_phase_poisson = single_phase_poisson
        self.q = q
        self.q_epoch = q_epoch
        self.q_step = q_step
        self.steps_per_epoch = steps_per_epoch
        self.device = device
        self.dp = dp
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.training_steps = 0

        modes_on = sum([self.poisson_subsampling,
                        self.use_epoch_assignment,
                        self.single_phase_poisson])
        if modes_on > 1:
            raise ValueError(
                "poisson_subsampling, use_epoch_assignment, single_phase_poisson "
                "are mutually exclusive"
            )

        if self.dp:
            if noise_multiplier is not None and epsilon is not None:
                raise ValueError("Provide noise_multiplier or epsilon, not both")
            if noise_multiplier is not None:
                self.noise_multiplier = noise_multiplier
                self.noise_std = noise_multiplier * self.max_grad_norm / math.sqrt(self.num_bins) #this is the change to divide by the number of bins
            elif epsilon is not None:
                # Standard Gaussian mechanism (same as Opacus):
                # sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon
                # Applied per-bin before averaging.
                C = self.max_grad_norm
                self.noise_std = C * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
                self.noise_multiplier = self.noise_std / self.max_grad_norm
            else:
                raise ValueError("dp=True requires either noise_multiplier or epsilon")

    def train_epoch(self, data) -> list:
        """
        Runs one outer iteration (steps_per_epoch steps). Returns list of per-step losses.

        Four modes (mutually exclusive):
        - single_phase_poisson: each of steps_per_epoch steps draws a fresh
            independent Bernoulli(q) sample over the full train set. Matches
            the standard DP-SGD / Opacus assumption of independent Poisson
            subsampling per step.
        - poisson_subsampling: Two-phase Poisson subsampling.
            Phase 1: each train node included independently with prob q_epoch.
            Phase 2: for each of steps_per_epoch steps, each epoch-node
            included independently with prob q_step.
            Effective per-step rate = q_epoch * q_step, but steps within an
            outer iteration share the epoch pool (not independent).
        - use_epoch_assignment: deterministic chunking into steps_per_epoch.
        - default: 1 step with all training nodes.
        """
        train_indices = data.train_mask.nonzero(as_tuple=True)[0]

        if self.single_phase_poisson:
            losses = []
            for _ in range(self.steps_per_epoch):
                step_mask = torch.bernoulli(
                    torch.full((len(train_indices),), self.q, device=self.device)
                ).bool()
                step_nodes = train_indices[step_mask]
                if len(step_nodes) > 0:
                    loss = self._train_step(data, step_nodes)
                    if loss is not None:
                        losses.append(loss)
            return losses

        if self.poisson_subsampling:
            # Phase 1: Poisson subsample for this epoch
            epoch_mask = torch.bernoulli(
                torch.full((len(train_indices),), self.q_epoch, device=self.device)
            ).bool()
            epoch_nodes = train_indices[epoch_mask]

            # Phase 2: per-step Poisson subsample from epoch pool
            losses = []
            for _ in range(self.steps_per_epoch):
                step_mask = torch.bernoulli(
                    torch.full((len(epoch_nodes),), self.q_step, device=self.device)
                ).bool()
                step_nodes = epoch_nodes[step_mask]
                if len(step_nodes) > 0:
                    loss = self._train_step(data, step_nodes)
                    if loss is not None:
                        losses.append(loss)
            return losses

        elif self.use_epoch_assignment:
            perm = torch.randperm(len(train_indices), device=self.device)
            chunks = torch.chunk(train_indices[perm], self.steps_per_epoch)
        else:
            chunks = [train_indices]

        losses = []
        for active_nodes in chunks:
            loss = self._train_step(data, active_nodes)
            if loss is not None:
                losses.append(loss)
        return losses

    def _train_step(self, data, active_train_nodes) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        num_nodes = data.x.size(0)
        edge_index = data.edge_index
        y = data.y
        N = self.num_bins

        active_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        active_mask[active_train_nodes] = True

        partitions = self.algorithm.partition(edge_index, num_nodes, N, self.device)

        if self.dp:
            return self._train_step_dp(data, partitions, active_mask, edge_index, y, num_nodes, N)
        else:
            return self._train_step_standard(data, partitions, active_mask, edge_index, y, num_nodes, N)

    def _train_step_standard(self, data, partitions, active_mask, edge_index, y, num_nodes, N) -> float:
        if self.use_coverage_correction:
            full_degree = compute_full_degrees(edge_index, num_nodes)

        total_loss = torch.tensor(0.0, device=self.device)
        any_loss = False

        for bin_mask, directed_ei in partitions:
            out = self.model(data.x, directed_ei)

            active_in_bin = bin_mask & active_mask
            if not active_in_bin.any():
                continue

            loss_i = F.nll_loss(out[active_in_bin], y[active_in_bin], reduction='sum')

            if self.use_coverage_correction:
                c_i = self._compute_coverage_correction(bin_mask, edge_index, full_degree, num_nodes)
            else:
                c_i = 1.0

            total_loss = total_loss + c_i * loss_i / N
            any_loss = True

        if not any_loss:
            return None

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def _train_step_dp(self, data, partitions, active_mask, edge_index, y, num_nodes, N) -> float:
        if self.use_coverage_correction:
            full_degree = compute_full_degrees(edge_index, num_nodes)

        bin_grads = []
        total_loss_val = 0.0

        for bin_mask, directed_ei in partitions:
            out = self.model(data.x, directed_ei)

            active_in_bin = bin_mask & active_mask
            if not active_in_bin.any():
                continue

            loss_i = F.nll_loss(out[active_in_bin], y[active_in_bin], reduction='sum')

            if self.use_coverage_correction:
                c_i = self._compute_coverage_correction(bin_mask, edge_index, full_degree, num_nodes)
            else:
                c_i = 1.0

            scaled_loss = c_i * loss_i / N
            total_loss_val += scaled_loss.item()

            self.optimizer.zero_grad()
            scaled_loss.backward()

            # Save a copy of gradients for this bin
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().clone())
                else:
                    grads.append(torch.zeros_like(p))
            bin_grads.append(grads)

        if not bin_grads:
            return None

        # Clip and noise each bin's gradient vector independently
        for grads in bin_grads:
            total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
            clip_coef = min(1.0, self.max_grad_norm / (total_norm.item() + 1e-8))
            if clip_coef < 1.0:
                for g in grads:
                    g.mul_(clip_coef)
            for g in grads:
                g.add_(torch.randn_like(g) * self.noise_std)

        # Average noised gradients
        avg_grads = []
        for param_idx in range(len(bin_grads[0])):
            stacked = torch.stack([bg[param_idx] for bg in bin_grads])
            avg_grads.append(stacked.mean(dim=0))

        # Set gradients and step
        self.optimizer.zero_grad()
        for p, avg_g in zip(self.model.parameters(), avg_grads):
            p.grad = avg_g
        self.optimizer.step()
        self.training_steps += 1

        return total_loss_val

    def _compute_coverage_correction(self, bin_mask, edge_index, full_degree, num_nodes) -> float:
        """
        c_i = mean_{v in B_i} (d_v^(i) / d_v)
        """
        src, dst = edge_index[0], edge_index[1]
        src_in_bin = bin_mask[src]
        dst_of_bin_edges = dst[src_in_bin]
        in_degree_i = torch.bincount(dst_of_bin_edges, minlength=num_nodes).float()

        bin_nodes = bin_mask.nonzero(as_tuple=True)[0]
        d_v = full_degree[bin_nodes]
        d_v_i = in_degree_i[bin_nodes]

        ratio = torch.where(d_v > 0, d_v_i / d_v, torch.zeros_like(d_v_i))
        return ratio.mean().item()

    @torch.no_grad()
    def evaluate(self, data) -> tuple:
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        return train_acc, test_acc
