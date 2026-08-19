#!/bin/zsh
# One-off diagnostics that answer "is the mechanism behaving as assumed?".
#
#   ./scripts/diagnose.sh gradnorm [dataset]   per-root gradient norms during DP
#                                              training — tells you whether the
#                                              clipping norm C is binding
#   ./scripts/diagnose.sh metrics  [dataset]   micro-F1 vs AUROC for the trivial,
#                                              non-DP, and DP models
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
WHAT=${1:?usage: diagnose.sh gradnorm|metrics [dataset]}
DS=${2:-ppi}

case $WHAT in
gradnorm)
$PY -u - "$DS" <<'EOF'
import sys, torch
from src.datasets import load_dataset
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
from src.sparse.sparse_expand import (build_adjacency, cap_degrees_undirected,
                                      sample_roots, sparse_expand)
from src.sparse.sparse_gnn import _step_dp

P1, P2, R, SIGMA, CLIP, LR, T, PROBE, K = 0.01, 0.1, 1, 5.0, 1.0, 0.3, 2000, 250, 5
ds, data = load_dataset(sys.argv[1])
ei = torch.unique(data.edge_index.cpu(), dim=1)
ei = cap_degrees_undirected(ei, int(data.num_nodes), K,
                            generator=torch.Generator().manual_seed(12345))
adj = build_adjacency(ei, int(data.num_nodes), direction='in')
train_nodes = torch.where(data.train_mask)[0]
B = P1 * train_nodes.numel()
mech = MultiLabelGNNMechanism(data, ds.num_features, ds.num_classes,
                              hidden=64, num_layers=2, dropout=0.0)
mech.build_optimizer(lr=LR, weight_decay=0.0, kind='sgd', momentum=0.0)
params = mech.parameters()
sg, ng = torch.Generator().manual_seed(0), torch.Generator().manual_seed(10_000)

print(f"p2={P2} r={R} sigma={SIGMA} C={CLIP} lr={LR} B~{B:.0f}", flush=True)
for t in range(1, T + 1):
    roots = sample_roots(int(data.num_nodes), P1, generator=sg,
                         candidate_nodes=train_nodes)
    subs = [sparse_expand(adj, int(v), P2, R, generator=sg, direction='in')
            for v in roots.tolist()]
    if t == 1 or t % PROBE == 0:
        norms = []
        for H in subs[:150]:
            g = torch.autograd.grad(mech.subgraph_loss(H), params,
                                    allow_unused=True)
            g = [x if x is not None else torch.zeros_like(p)
                 for x, p in zip(g, params)]
            norms.append(float(torch.sqrt(sum((x ** 2).sum() for x in g))))
        n = torch.tensor(norms)
        wn = float(torch.sqrt(sum((p ** 2).sum() for p in params)))
        print(f"step {t:>5}  median={n.median():.4f} p90={n.quantile(0.9):.4f} "
              f"max={n.max():.4f} | frac>C={(n > CLIP).float().mean():.1%} "
              f"| ||theta||={wn:.2f}", flush=True)
    _step_dp(mech, subs, C=CLIP, sigma=SIGMA, noise_gen=ng, expected_batch=B)
print(f"final test={mech.evaluate(data)['test']:.4f}")
EOF
;;
metrics)
$PY -u - "$DS" <<'EOF'
import sys, torch
from src.datasets import load_dataset
from src.sparse.multilabel_mechanism import (MultiLabelGNNMechanism, _micro_f1,
                                             _micro_auroc)
from src.sparse.sparse_expand import (build_adjacency, cap_degrees_undirected,
                                      sample_roots, sparse_expand)
from src.sparse.sparse_gnn import _step_dp, _step_nondp

ds, data = load_dataset(sys.argv[1])
ei = torch.unique(data.edge_index.cpu(), dim=1)
ei = cap_degrees_undirected(ei, int(data.num_nodes), 5,
                            generator=torch.Generator().manual_seed(12345))
adj = build_adjacency(ei, int(data.num_nodes), direction='in')
train_nodes = torch.where(data.train_mask)[0]
te = data.test_mask

def train(dp, T, p1, sigma, lr):
    m = MultiLabelGNNMechanism(data, ds.num_features, ds.num_classes,
                               hidden=64, num_layers=2, dropout=0.0)
    m.build_optimizer(lr=lr, weight_decay=0.0,
                      kind='sgd' if dp else 'adam', momentum=0.0)
    sg, ng = torch.Generator().manual_seed(0), torch.Generator().manual_seed(10_000)
    B = p1 * train_nodes.numel()
    for _ in range(T):
        roots = sample_roots(int(data.num_nodes), p1, generator=sg,
                             candidate_nodes=train_nodes)
        subs = [sparse_expand(adj, int(v), 0.1, 1, generator=sg, direction='in')
                for v in roots.tolist()]
        if dp:
            _step_dp(m, subs, C=1.0, sigma=sigma, noise_gen=ng, expected_batch=B)
        elif roots.numel():
            _step_nondp(m, subs)
    return m

ones = torch.ones_like(data.y[te]).float()
print(f"{'all-ones (eps=0)':<32} f1={_micro_f1(ones, data.y[te].float()):.4f}  "
      f"auroc={_micro_auroc(ones, data.y[te]):.4f}")
for tag, kw in (("non-DP", dict(dp=False, T=2000, p1=0.01, sigma=0, lr=0.01)),
                ("DP sigma=5", dict(dp=True, T=2000, p1=0.01, sigma=5.0, lr=0.3))):
    m = train(**kw)
    r = m.evaluate(data)
    print(f"{tag:<32} f1={r['test']:.4f}  auroc={r['test_auroc']:.4f}")
EOF
;;
*) echo "unknown diagnostic: $WHAT" >&2; exit 2 ;;
esac
