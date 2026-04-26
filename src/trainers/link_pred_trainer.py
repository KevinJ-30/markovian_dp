"""
Link-prediction trainer for the subgraph DP framework.

Subclasses SubgraphTrainer to reuse the per-bin clip+noise+average DP
plumbing. Only the per-bin loss and final evaluation differ:

- `_compute_bin_loss` is overridden to compute BCE on positive edges whose
  both endpoints fall in the current bin's active node set, plus an equal
  number of in-bin random negatives.
- `evaluate` is overridden to compute Hits@K (default K=50) on the
  validation and test edge splits via OGB's Evaluator.

The DP unit remains node-level (consistent with node-DP for the existing
algorithms): one node's removal can change only the gradients of bins it
participates in. Edge supervision feeds into the same per-bin gradient,
which is then clipped and noised exactly as in the node-classification path.
"""

import torch
import torch.nn.functional as F

from src.trainers.subgraph_trainer import SubgraphTrainer


class LinkPredTrainer(SubgraphTrainer):
    def __init__(self, *args, eval_hits_k=50, neg_ratio=1, **kwargs):
        # Coverage correction does not generalize cleanly to edge supervision;
        # force it off so the base class skips that path.
        kwargs['use_coverage_correction'] = False
        super().__init__(*args, **kwargs)
        self.eval_hits_k = eval_hits_k
        self.neg_ratio = neg_ratio

    def _compute_bin_loss(self, data, out, bin_mask, active_mask):
        """
        BCE-with-logits on in-bin training edges.

        Positives: training pos edges where both endpoints are in
            `bin_mask & active_mask`.
        Negatives: uniform random pairs sampled from the same in-bin node
            set, matching positive count (or `neg_ratio` * positives).
        """
        eligible = bin_mask & active_mask
        bin_nodes = eligible.nonzero(as_tuple=True)[0]
        if bin_nodes.numel() < 2:
            return None

        pos = data.train_pos_edge  # [2, E]
        pos_in_bin_mask = eligible[pos[0]] & eligible[pos[1]]
        if not pos_in_bin_mask.any():
            return None
        pos_pairs = pos[:, pos_in_bin_mask]
        n_pos = pos_pairs.size(1)

        # In-bin negatives: sample random (src, dst) from bin_nodes.
        n_neg = n_pos * self.neg_ratio
        idx_src = torch.randint(0, bin_nodes.size(0), (n_neg,), device=self.device)
        idx_dst = torch.randint(0, bin_nodes.size(0), (n_neg,), device=self.device)
        neg_pairs = torch.stack([bin_nodes[idx_src], bin_nodes[idx_dst]], dim=0)

        pos_logits = self.model.score(out, pos_pairs)
        neg_logits = self.model.score(out, neg_pairs)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        targets = torch.cat([
            torch.ones(n_pos, device=self.device),
            torch.zeros(n_neg, device=self.device),
        ], dim=0)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='sum')

    @torch.no_grad()
    def evaluate(self, data) -> tuple:
        """
        Returns (train_hits, test_hits) at self.eval_hits_k. Uses OGB Evaluator
        if available; falls back to a simple Hits@K computation otherwise.
        """
        self.model.eval()
        z = self.model(data.x, data.edge_index)

        def _score(pairs):
            return self.model.score(z, pairs)

        # Train accuracy isn't a standard metric for link prediction; we
        # report Hits@K on validation as the "train" column proxy and on
        # test for the test column. This keeps the (train_acc, test_acc)
        # signature compatible with the node-classification trainer.
        valid_pos = _score(data.valid_pos_edge)
        valid_neg = _score(data.valid_neg_edge)
        test_pos = _score(data.test_pos_edge)
        test_neg = _score(data.test_neg_edge)

        try:
            from ogb.linkproppred import Evaluator
            evaluator = Evaluator(name='ogbl-collab')
            evaluator.K = self.eval_hits_k
            valid_hits = evaluator.eval({
                'y_pred_pos': valid_pos,
                'y_pred_neg': valid_neg,
            })[f'hits@{self.eval_hits_k}']
            test_hits = evaluator.eval({
                'y_pred_pos': test_pos,
                'y_pred_neg': test_neg,
            })[f'hits@{self.eval_hits_k}']
        except Exception:
            valid_hits = _hits_at_k(valid_pos, valid_neg, self.eval_hits_k)
            test_hits = _hits_at_k(test_pos, test_neg, self.eval_hits_k)

        return float(valid_hits), float(test_hits)


def _hits_at_k(pos_scores, neg_scores, k) -> float:
    """Fraction of positive edges with score above the k-th highest negative."""
    if neg_scores.numel() < k:
        return float((pos_scores > neg_scores.max()).float().mean().item())
    kth = torch.topk(neg_scores, k=k).values[-1]
    return float((pos_scores > kth).float().mean().item())
