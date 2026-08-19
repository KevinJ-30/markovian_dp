#!/bin/zsh
# Why did lowering the clipping norm HURT?  The init-time gradient norms
# (median 0.199) predicted that C=0.5 clips almost nothing while halving the
# noise — a free win.  The clip sweep found the opposite, monotonically:
# C=1.0 -> 0.4412, C=0.5 -> 0.4309, C=0.2 -> 0.4145, C=0.1 -> 0.4105.
#
# The untested assumption was that init-time norms represent the whole
# trajectory.  This probe runs the real DP loop (lr=0.3, sigma=5, C=1.0) and
# re-measures the per-root gradient-norm distribution every 250 steps, so we
# can see whether clipping actually becomes binding as training proceeds.
#
#   nohup caffeinate -i ./scripts/_ppi_gradnorm_probe.sh > results/ppi_gradnorm.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python -u - <<'EOF'
import torch
from src.datasets import load_dataset
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
from src.sparse.sparse_expand import (build_adjacency, cap_degrees_undirected,
                                      sample_roots, sparse_expand)
from src.sparse.sparse_gnn import _step_dp

P1, P2, R, SIGMA, CLIP, LR, T, PROBE = 0.01, 0.1, 1, 5.0, 1.0, 0.3, 2000, 250

ds, data = load_dataset('ppi')
ei = torch.unique(data.edge_index.cpu(), dim=1)
ei = cap_degrees_undirected(ei, int(data.num_nodes), 5,
                            generator=torch.Generator().manual_seed(12345))
adj = build_adjacency(ei, int(data.num_nodes), direction='in')
train_nodes = torch.where(data.train_mask)[0]
B = P1 * train_nodes.numel()

mech = MultiLabelGNNMechanism(data, ds.num_features, ds.num_classes,
                              hidden=64, num_layers=2, dropout=0.0)
mech.build_optimizer(lr=LR, weight_decay=0.0, kind='sgd', momentum=0.0)
params = mech.parameters()

sample_gen = torch.Generator().manual_seed(0)
noise_gen = torch.Generator().manual_seed(10_000)

def probe(step, subgraphs):
    norms = []
    for H in subgraphs[:150]:
        loss = mech.subgraph_loss(H)
        g = torch.autograd.grad(loss, params, allow_unused=True)
        g = [x if x is not None else torch.zeros_like(p) for x, p in zip(g, params)]
        norms.append(float(torch.sqrt(sum((x ** 2).sum() for x in g))))
    n = torch.tensor(norms)
    wn = float(torch.sqrt(sum((p ** 2).sum() for p in params)))
    print(f"step {step:>5}  grad-norm median={n.median():.4f} p90={n.quantile(0.9):.4f} "
          f"max={n.max():.4f} | frac>1.0={(n > 1.0).float().mean():.1%} "
          f"frac>0.5={(n > 0.5).float().mean():.1%} frac>0.2={(n > 0.2).float().mean():.1%} "
          f"| ||theta||={wn:.2f}", flush=True)

print(f"PPI DP probe: p2={P2} r={R} sigma={SIGMA} C={CLIP} lr={LR} B~{B:.0f}", flush=True)
for t in range(1, T + 1):
    roots = sample_roots(int(data.num_nodes), P1, generator=sample_gen,
                         candidate_nodes=train_nodes)
    subgraphs = [sparse_expand(adj, int(v), P2, R, generator=sample_gen,
                               direction='in') for v in roots.tolist()]
    if t == 1 or t % PROBE == 0:
        probe(t, subgraphs)
    _step_dp(mech, subgraphs, C=CLIP, sigma=SIGMA, noise_gen=noise_gen,
             expected_batch=B)
m = mech.evaluate(data)
print(f"final test={m['test']:.4f}", flush=True)
EOF
echo "=== GRADNORM PROBE COMPLETE $(date) ==="
