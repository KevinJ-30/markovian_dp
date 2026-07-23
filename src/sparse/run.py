"""
SparseGNN experiment CLI — paper Algorithms 1 & 2 (current default sparsification).

Runs the composite-subsampling mechanism (root sampling p1 + SparseExpand p2/r)
with a GNN base mechanism for node classification.  Defaults to CiteSeer, no DP.

Examples (from repo root):
  # Sanity: p1=p2=1 recovers (near) full-graph GCN
  python -m src.sparse.run --dataset citeseer --p1 1.0 --p2 1.0 --r 2 --T 200 --seeds 3

  # The actual sparsified mechanism
  python -m src.sparse.run --dataset citeseer --p1 0.5 --p2 0.5 --r 2 --T 200 --seeds 3

DP (--dp) is prepared but off by default; see src/sparse/accounting.py for the
Theorem 3 dominating pair used for accounting.
"""

import argparse
import csv
import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets import load_dataset                       # noqa: E402
from src.sparse.gnn_mechanism import GNNMechanism           # noqa: E402
from src.sparse.sparse_expand import (                      # noqa: E402
    build_out_adjacency, cap_degrees, max_degrees,
)
from src.sparse.sparse_gnn import train_sparse_gnn          # noqa: E402


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def plot_sweep(summary, dataset_name, out_dir):
    """Plot test accuracy vs p2, one line per p1 (linestyle per r if r is swept).

    `summary` is a list of (p1, p2, r, test_mean, test_std, val_mean, val_std).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    p1s = sorted({s[0] for s in summary})
    rs = sorted({s[2] for s in summary})
    linestyles = ['-', '--', ':', '-.']

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p1 in p1s:
        for ri, r in enumerate(rs):
            rows = sorted((s for s in summary if s[0] == p1 and s[2] == r),
                          key=lambda s: s[1])
            if not rows:
                continue
            xs = [s[1] for s in rows]
            ys = [s[3] for s in rows]
            es = [s[4] for s in rows]
            label = f'p1={p1}' + (f', r={r}' if len(rs) > 1 else '')
            ax.errorbar(xs, ys, yerr=es, fmt='o' + linestyles[ri % len(linestyles)],
                        capsize=4, label=label)
    ax.set_xlabel('edge-sampling probability p2  (1.0 = all edges)')
    ax.set_ylabel('test accuracy')
    r_txt = f'r={rs[0]}' if len(rs) == 1 else f'r in {rs}'
    ax.set_title(f'{dataset_name}: SparseGNN test accuracy vs sparsification ({r_txt}, no DP)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='root-sampling p1')
    fig.tight_layout()
    path = os.path.join(out_dir, f'sparse_gnn_{dataset_name}_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='citeseer',
                   help='cora | citeseer | pubmed | ...')
    # Paper parameters (each accepts one or more values → swept as a grid)
    p.add_argument('--p1', type=float, nargs='+', default=[0.5],
                   help='root-sampling probability p1 (Bernoulli per node); '
                        'pass several to sweep, e.g. --p1 0.25 0.5 1.0')
    p.add_argument('--p2', type=float, nargs='+', default=[0.5],
                   help='edge-sparsification probability p2 (Bernoulli per arc); '
                        'pass several to sweep')
    p.add_argument('--r', type=int, nargs='+', default=[2],
                   help='maximum expansion distance r (SparseExpand levels); '
                        'pass several to sweep, e.g. --r 1 2 3')
    p.add_argument('--T', type=int, default=200,
                   help='number of training steps T')
    # Model / optimization
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2, help='GCN layers L')
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--roots_from', choices=['train', 'all'], default='train',
                   help="eligible-root pool: 'train' (labeled roots only) or 'all'")
    # DP (off by default)
    p.add_argument('--dp', action='store_true', help='enable DP clip+noise path')
    p.add_argument('--clip', type=float, default=1.0, help='clipping norm C (DP)')
    p.add_argument('--sigma', type=float, nargs='+', default=[1.0],
                   help='noise multiplier(s); pass several to sweep, e.g. '
                        '--sigma 2 5 10 (only swept when --dp)')
    p.add_argument('--K_in', type=int, default=None,
                   help='cap max in-degree before training (required for a '
                        'valid Theorem 4 guarantee; recorded in the CSV for '
                        'post-hoc accounting via src.sparse.compute_epsilon)')
    p.add_argument('--K_out', type=int, default=None,
                   help='cap max out-degree before training (defaults to K_in)')
    # General
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--out_dir', default='results')
    p.add_argument('--plot', action='store_true',
                   help='save a sweep plot (test acc vs p2, line per r, subplot per p1)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--eval_every', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_dp' if args.dp else ''
    csv_path = os.path.join(args.out_dir,
                            f'sparse_gnn_{args.dataset}{tag}_results.csv')

    sigmas = args.sigma if args.dp else [args.sigma[0]]
    grid = list(itertools.product(args.p1, args.p2, args.r, sigmas))

    print(f"\n{'='*66}")
    print(f"SparseGNN (Alg 1&2)  dataset={args.dataset}  device={device}")
    print(f"  p1={args.p1}  p2={args.p2}  r={args.r}  sigma={sigmas}  T={args.T}  "
          f"L={args.num_layers}  dp={args.dp}  seeds={args.seeds}")
    print(f"  sweep: {len(grid)} (p1,p2,r,sigma) combo(s) x {args.seeds} seed(s)")
    print('='*66)

    dataset, data = load_dataset(args.dataset, device=str(device))
    data = data.to(device)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    edge_index = data.edge_index
    K_in, K_out = args.K_in, args.K_out if args.K_out is not None else args.K_in
    if K_in is not None:
        before = max_degrees(edge_index, int(data.num_nodes))
        cap_gen = torch.Generator().manual_seed(12345)
        edge_index = cap_degrees(edge_index, int(data.num_nodes),
                                 K_in=K_in, K_out=K_out, generator=cap_gen)
        after = max_degrees(edge_index, int(data.num_nodes))
        print(f"  degree cap K_in={K_in} K_out={K_out}: max (in,out) "
              f"{before} -> {after}, edges {data.edge_index.size(1)} -> "
              f"{edge_index.size(1)}")
    elif args.dp:
        print("  WARNING: --dp without --K_in — the Theorem 4 accounting "
              "assumption (bounded degrees) is not enforced; post-hoc epsilon "
              "will use the graph's raw max degrees.")
        K_in, K_out = max_degrees(edge_index, int(data.num_nodes))

    # Out-adjacency is deterministic; build once and reuse across all runs.
    adj = build_out_adjacency(edge_index, int(data.num_nodes))

    candidate_nodes = None
    if args.roots_from == 'train':
        candidate_nodes = torch.where(data.train_mask)[0]

    summary = []   # (p1, p2, r, sigma, test_mean, test_std, val_mean, val_std)

    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'p1', 'p2', 'r', 'sigma', 'clip', 'K_in', 'K_out',
                    'T', 'L', 'dp', 'seed', 'train_acc', 'val_acc', 'test_acc'])

        for p1, p2, r, sigma in grid:
            print(f"\n[p1={p1} p2={p2} r={r}" +
                  (f" sigma={sigma}]" if args.dp else "]"))
            tests, vals = [], []
            for seed in range(args.seeds):
                _set_seed(seed)
                mech = GNNMechanism(
                    data, num_features, num_classes,
                    hidden=args.hidden, num_layers=args.num_layers,
                    dropout=args.dropout, device=device,
                )
                mech.build_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                                     kind='sgd' if args.dp else 'adam')

                accs = train_sparse_gnn(
                    mech, data, adj=adj,
                    p1=p1, p2=p2, r=r, T=args.T,
                    candidate_nodes=candidate_nodes,
                    dp=args.dp, clip=args.clip, sigma=sigma,
                    seed=seed, eval_every=args.eval_every, verbose=args.verbose,
                )
                tests.append(accs['test'])
                vals.append(accs['val'])
                print(f"  seed={seed}  train={accs['train']:.4f}  "
                      f"val={accs['val']:.4f}  test={accs['test']:.4f}")
                w.writerow([args.dataset, p1, p2, r, sigma, args.clip,
                            K_in if K_in is not None else '',
                            K_out if K_out is not None else '',
                            args.T, args.num_layers, args.dp, seed,
                            f"{accs['train']:.5f}", f"{accs['val']:.5f}",
                            f"{accs['test']:.5f}"])

            tm, ts = _mean_std(tests)
            vm, vs = _mean_std(vals)
            summary.append((p1, p2, r, sigma, tm, ts, vm, vs))
            print(f"  >> test {tm:.4f} +/- {ts:.4f}   val {vm:.4f} +/- {vs:.4f}")

    # Sweep summary table (sorted by test accuracy, best first)
    print(f"\n{'='*66}")
    print(f"{'p1':>5} {'p2':>5} {'r':>3} {'sigma':>6} {'test':>16} {'val':>16}")
    print('-'*66)
    for p1, p2, r, sigma, tm, ts, vm, vs in sorted(summary, key=lambda s: -s[4]):
        print(f"{p1:>5} {p2:>5} {r:>3} {sigma:>6}   {tm:.4f} +/- {ts:.4f}   "
              f"{vm:.4f} +/- {vs:.4f}")
    print(f"\nresults written to {csv_path}")
    if args.dp:
        print("compute epsilon post-hoc with:  python -m src.sparse.compute_epsilon "
              f"--csv {csv_path}")

    if args.plot:
        plot_path = plot_sweep([(s[0], s[1], s[2], s[4], s[5], s[6], s[7])
                                for s in summary if s[3] == sigmas[0]],
                               args.dataset, args.out_dir)
        if plot_path:
            print(f"plot written to {plot_path}")


if __name__ == '__main__':
    main()
