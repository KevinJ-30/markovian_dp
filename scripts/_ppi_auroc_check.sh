#!/bin/zsh
# Is the DP model actually learning, or is micro-F1 hiding it?
#
# PPI's "trivial baseline" of 0.4608 micro-F1 is the ALL-ONES predictor: perfect
# recall, no precision, and AUROC exactly 0.5 — it has zero ranking ability and
# costs zero privacy.  A DP model can therefore sit below 0.4608 micro-F1 while
# still ranking labels far better than chance.  This measures both metrics on
# the same trained models.
set -e
cd "$(dirname "$0")/.."
/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python -u - <<'EOF'
import numpy as np, torch
from src.datasets import load_dataset
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism, _micro_f1
from src.sparse.sparse_expand import (build_adjacency, cap_degrees_undirected,
                                      sample_roots, sparse_expand)
from src.sparse.sparse_gnn import _step_dp, _step_nondp

ds, data = load_dataset('ppi')
ei = torch.unique(data.edge_index.cpu(), dim=1)
ei = cap_degrees_undirected(ei, int(data.num_nodes), 5,
                            generator=torch.Generator().manual_seed(12345))
adj = build_adjacency(ei, int(data.num_nodes), direction='in')
train_nodes = torch.where(data.train_mask)[0]

def auroc_multilabel(scores, y):
    """Micro-averaged AUROC over all (node,label) pairs."""
    s, t = scores.ravel(), y.ravel()
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = int(t.sum()), int((1 - t).sum())
    if npos == 0 or nneg == 0: return float('nan')
    return (ranks[t == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def report(tag, mech):
    mech.eval_mode()
    with torch.no_grad():
        logits = mech.module(data.x, data.edge_index)
    te = data.test_mask
    f1 = _micro_f1((logits[te] > 0).float(), data.y[te].float())
    au = auroc_multilabel(logits[te].cpu().numpy(), data.y[te].cpu().numpy())
    print(f"{tag:<34} micro_f1={f1:.4f}   AUROC={au:.4f}", flush=True)

def train(dp, T, p1, sigma, lr, seed=0):
    mech = MultiLabelGNNMechanism(data, ds.num_features, ds.num_classes,
                                  hidden=64, num_layers=2, dropout=0.0)
    mech.build_optimizer(lr=lr, weight_decay=0.0,
                         kind='sgd' if dp else 'adam', momentum=0.0)
    sg = torch.Generator().manual_seed(seed)
    ng = torch.Generator().manual_seed(seed + 10_000)
    B = p1 * train_nodes.numel()
    for t in range(T):
        roots = sample_roots(int(data.num_nodes), p1, generator=sg,
                             candidate_nodes=train_nodes)
        subs = [sparse_expand(adj, int(v), 0.1, 1, generator=sg, direction='in')
                for v in roots.tolist()]
        if dp:
            _step_dp(mech, subs, C=1.0, sigma=sigma, noise_gen=ng, expected_batch=B)
        elif roots.numel():
            _step_nondp(mech, subs)
    return mech

te = data.test_mask
y = data.y[te].cpu().numpy()
ones = np.ones_like(y, dtype=float)
print(f"{'all-ones (eps=0) predictor':<34} "
      f"micro_f1={_micro_f1(torch.ones_like(data.y[te]).float(), data.y[te].float()):.4f}"
      f"   AUROC={auroc_multilabel(ones, y):.4f}\n", flush=True)
report("non-DP  (p1=0.01, T=2000)", train(False, 2000, 0.01, 0, 0.01))
report("DP eps~2.6 (p1=0.01,s=5,lr=.3)", train(True, 2000, 0.01, 5.0, 0.3))
report("DP eps~4.1 (p1=0.05,s=10,lr=1)", train(True, 800, 0.05, 10.0, 1.0))
EOF
echo "=== AUROC CHECK COMPLETE $(date) ==="
