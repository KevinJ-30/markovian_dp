#!/bin/zsh
# Compare non-private and Opacus-private multilabel training metrics.
#
#   ./scripts/diagnose.sh metrics [dataset]
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
WHAT=${1:?usage: diagnose.sh metrics [dataset]}
DS=${2:-ppi}

case $WHAT in
metrics)
$PY -u - "$DS" <<'EOF'
import sys, torch
from src.datasets import load_dataset
from src.sparse.multilabel_mechanism import (
    MultiLabelGNNMechanism, _micro_f1, _micro_auroc)
from src.sparse.sparse_expand import (
    build_adjacency, cap_degrees_undirected)
from src.sparse.run import make_training_graph
from src.sparse.sparse_gnn import train_sparse_gnn

ds, data = load_dataset(sys.argv[1])
train_data = make_training_graph(data)
ei = torch.unique(train_data.edge_index.cpu(), dim=1)
ei = cap_degrees_undirected(
    ei, int(train_data.num_nodes), 5,
    generator=torch.Generator().manual_seed(12345))
adj = build_adjacency(ei, int(train_data.num_nodes), direction='in')
te = data.test_mask

def train(dp, T, p1, sigma, lr):
    model = MultiLabelGNNMechanism(
        train_data, ds.num_features, ds.num_classes,
        hidden=64, num_layers=2, dropout=0.0)
    model.build_optimizer(lr=lr, weight_decay=0.0, kind='adam')
    train_sparse_gnn(
        model, train_data, data, adj=adj, direction='in', p1=p1, p2=0.1,
        r=1, T=T, dp=dp, clip=1.0 if dp else None,
        sigma=sigma if dp else None, seed=0)
    return model

ones = torch.ones_like(data.y[te]).float()
print(f"{'all-ones (eps=0)':<32} f1={_micro_f1(ones, data.y[te].float()):.4f}  "
      f"auroc={_micro_auroc(ones, data.y[te]):.4f}")
for tag, kwargs in (
    ('non-DP', dict(dp=False, T=2000, p1=0.01, sigma=0, lr=0.01)),
    ('DP sigma=5', dict(dp=True, T=2000, p1=0.01, sigma=5.0, lr=0.3)),
):
    result = train(**kwargs).evaluate(data)
    print(f"{tag:<32} f1={result['test']:.4f}  "
          f"auroc={result['test_auroc']:.4f}")
EOF
;;
*) echo "unknown diagnostic: $WHAT" >&2; exit 2 ;;
esac
