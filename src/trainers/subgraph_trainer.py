"""
Subgraph trainer — trains on random subgraph partitions with optional coverage correction.
Accepts a pluggable SubgraphAlgorithm for partitioning.
"""

import torch
import torch.nn.functional as F

from src.utils import compute_full_degrees


class SubgraphTrainer:
    def __init__(self, model, optimizer, num_bins, algorithm,
                 use_coverage_correction=True,
                 use_epoch_assignment=False,
                 steps_per_epoch=10,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.num_bins = num_bins
        self.algorithm = algorithm
        self.use_coverage_correction = use_coverage_correction
        self.use_epoch_assignment = use_epoch_assignment
        self.steps_per_epoch = steps_per_epoch
        self.device = device

    def train_epoch(self, data) -> list:
        """
        Runs one full epoch. Returns list of per-step losses.
        If use_epoch_assignment: steps_per_epoch steps, each ~|train|/T active nodes.
        Else: 1 step with all training nodes.
        """
        train_indices = data.train_mask.nonzero(as_tuple=True)[0]

        if self.use_epoch_assignment:
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

        if self.use_coverage_correction:
            full_degree = compute_full_degrees(edge_index, num_nodes)

        partitions = self.algorithm.partition(edge_index, num_nodes, N, self.device)

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
