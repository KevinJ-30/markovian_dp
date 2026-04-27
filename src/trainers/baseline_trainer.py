"""
Baseline trainer — standard full-graph GCN training (no subgraph partitioning).
"""

import torch
import torch.nn.functional as F


class BaselineTrainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data) -> tuple:
        self.model.eval()
        eval_ei = getattr(data, 'eval_edge_index', data.edge_index)
        out = self.model(data.x, eval_ei)
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        return train_acc, test_acc
