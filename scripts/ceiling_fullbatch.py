"""
Non-DP utility ceiling, computed full-batch instead of per-root.

Why this is a valid substitute
------------------------------
With mean aggregation the base mechanism g0 evaluated on a rooted subgraph at
p2=1, no degree cap, and r = L is EXACTLY full-graph inference at the root
(verified to 0.00e+00 relative error; see src/sparse/layers.py).  The two
training procedures therefore optimize the same per-node objective and differ
only in how nodes are batched: Poisson root sampling at rate p1 versus all
training nodes every step.  Measured on PPI with the same capped graph, the two
agree to four decimals — per-root 0.5463 vs full-batch 0.5464.

What it buys
------------
The per-root ceiling on PPI expands 787-node subgraphs at ~2.4 s/step, so
T=2000 x 3 seeds costs about four hours.  Full-batch reaches the same number in
about a minute.

This is ONLY valid for the ceiling (p2=1, uncapped, r=L, no DP).  Every capped,
sparsified, or noisy configuration must go through the real per-root engine —
those are genuinely different mechanisms, not just a different batching of the
same one.

Usage:
  python scripts/ceiling_fullbatch.py --dataset ppi --model multilabel_gnn \
      --num_layers 2 --epochs 300 --seeds 3 --out_dir results/inductive_ceiling_ppi
"""

import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import load_dataset                              # noqa: E402
from src.sparse.binary_mechanism import BinaryGNNMechanism         # noqa: E402
from src.sparse.gnn_mechanism import GNNMechanism                  # noqa: E402
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism  # noqa: E402
from src.sparse.run import trivial_baseline                        # noqa: E402

MECHANISMS = {
    'gnn': GNNMechanism,
    'multilabel_gnn': MultiLabelGNNMechanism,
    'binary_gnn': BinaryGNNMechanism,
}


def node_loss(metric, out, y, mask):
    """Full-batch analogue of the mechanism's per-root loss."""
    if metric == 'accuracy':                       # module already log_softmaxes
        return F.nll_loss(out[mask], y[mask].view(-1))
    if metric == 'micro_f1':
        return F.binary_cross_entropy_with_logits(out[mask], y[mask].float())
    return F.binary_cross_entropy_with_logits(out[mask], y[mask].float().view(-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', default='gnn', choices=sorted(MECHANISMS))
    ap.add_argument('--aggr', default='mean', choices=['mean', 'gcn'])
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--inductive', action='store_true')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    ds_slug = args.dataset.replace(':', '_').replace('/', '_')
    csv_path = os.path.join(args.out_dir, f'sparse_gnn_{ds_slug}_results.csv')

    dataset, data = load_dataset(args.dataset, device=str(device))
    data = data.to(device)

    # Training edges: the same graph the per-root ceiling would have expanded on.
    train_ei = data.edge_index
    if args.inductive:
        if hasattr(data, 'train_edge_index'):
            train_ei = data.train_edge_index
        else:
            m = data.train_mask
            train_ei = data.edge_index[:, m[data.edge_index[0]] & m[data.edge_index[1]]]
    print(f"full-batch ceiling  dataset={args.dataset}  device={device}  "
          f"aggr={args.aggr}  L={args.num_layers}  epochs={args.epochs}")
    print(f"  training edges {train_ei.size(1)} of {data.edge_index.size(1)}")

    Mech = MECHANISMS[args.model]
    metric = Mech.metric_name
    trivial = trivial_baseline(data, metric)
    print(f"  trivial baseline ({metric}) on test: {trivial:.4f}")

    rows, tests = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        mech = Mech(data, dataset.num_features, dataset.num_classes,
                    hidden=args.hidden, num_layers=args.num_layers,
                    dropout=args.dropout, aggr=args.aggr, device=device)
        opt = mech.build_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                                   kind='adam')
        for _ in range(args.epochs):
            mech.train_mode()
            opt.zero_grad()
            out = mech.module(data.x, train_ei)
            node_loss(metric, out, data.y, data.train_mask).backward()
            opt.step()
        accs = mech.evaluate(data)
        tests.append(accs['test'])
        extra = (f"  test_auroc={accs['test_auroc']:.4f}"
                 if 'test_auroc' in accs else "")
        print(f"  seed={seed}  train={accs['train']:.4f}  val={accs['val']:.4f}  "
              f"test={accs['test']:.4f}{extra}")
        rows.append([args.dataset, args.model, args.aggr, metric, args.inductive,
                     'in', '', 1.0, args.num_layers, '', '', '', '',
                     '', 'full', args.lr, 0.0,
                     args.epochs, args.num_layers, False, seed, args.epochs,
                     f"{accs['train']:.5f}", f"{accs['val']:.5f}",
                     f"{accs['test']:.5f}", f"{trivial:.5f}",
                     *(f"{accs[k]:.5f}" if k in accs else ''
                       for k in ('train_auroc', 'val_auroc', 'test_auroc'))])

    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        # Same schema as src.sparse.run so downstream analysis is unchanged.
        w.writerow(['dataset', 'model', 'aggr', 'metric', 'inductive',
                    'direction', 'p1', 'p2', 'r', 'sigma', 'clip', 'K_in',
                    'K_out', 'cap_mode', 'eval_graph', 'lr', 'momentum',
                    'T', 'L', 'dp', 'seed', 'step',
                    'train_acc', 'val_acc', 'test_acc', 'trivial_baseline',
                    'train_auroc', 'val_auroc', 'test_auroc'])
        w.writerows(rows)

    mean = sum(tests) / len(tests)
    flag = "" if mean > trivial else "   <-- BELOW TRIVIAL BASELINE"
    print(f"  >> test {mean:.4f}{flag}")
    print(f"results written to {csv_path}")


if __name__ == '__main__':
    main()
