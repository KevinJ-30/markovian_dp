"""
Privacy-utility frontier plot for SparseGNN DP sweeps.

Reads a *_with_eps.csv produced by `src.sparse.compute_epsilon` and plots test
accuracy vs epsilon (log x), one line per edge-sparsification probability p2,
points ordered by sigma.  Optionally overlays the non-DP ceiling from the
matching non-DP results CSV.

  python scripts/plot_sparse_frontier.py \
      --csv results/sparse_gnn_ogbn-arxiv_dp_results_with_eps.csv \
      --ceiling_csv results/sparse_gnn_ogbn-arxiv_results.csv \
      --out results/arxiv_frontier.png
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True, help='*_with_eps.csv from compute_epsilon')
    ap.add_argument('--ceiling_csv', default=None,
                    help='matching non-DP results CSV (for the ceiling line)')
    ap.add_argument('--out', required=True, help='output PNG path')
    args = ap.parse_args()

    with open(args.csv, newline='') as fh:
        rows = list(csv.DictReader(fh))

    # mean test metric per (p2, sigma), epsilon per config.  `epsilon_thm4` is
    # the pre-orientation-fix column name; current CSVs use `epsilon` and record
    # which theorem produced it in `epsilon_theorem`.
    eps_col = 'epsilon' if 'epsilon' in rows[0] else 'epsilon_thm4'
    acc = defaultdict(list)
    eps = {}
    for row in rows:
        key = (float(row['p2']), float(row['sigma']))
        acc[key].append(float(row['test_acc']))
        eps[key] = float(row[eps_col])
    dataset = rows[0]['dataset']
    metric = rows[0].get('metric', 'accuracy')
    theorem = rows[0].get('epsilon_theorem', 'Theorem 4.5')
    direction = rows[0].get('direction', 'out')
    meta = (f"p1={rows[0]['p1']}, r={rows[0]['r']}, K_in={rows[0]['K_in']}, "
            f"K_out={rows[0].get('K_out', '?')}, dir={direction}, "
            f"T={rows[0]['T']}, delta={rows[0].get('delta', '?')}")

    finite = [k for k in eps if eps[k] != float('inf')]
    if len(finite) < len(eps):
        print(f"note: {len(eps) - len(finite)} config(s) have epsilon=inf "
              f"(beyond the accountant's loss cap) and are omitted from the plot")
        acc = {k: v for k, v in acc.items() if k in finite}
        eps = {k: v for k, v in eps.items() if k in finite}
    if not acc:
        raise SystemExit("every config has epsilon=inf — nothing to plot")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p2 in sorted({k[0] for k in acc}, reverse=True):
        pts = sorted(((eps[k], sum(acc[k]) / len(acc[k]))
                      for k in acc if k[0] == p2))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-',
                label=f'p2={p2}')

    if args.ceiling_csv:
        with open(args.ceiling_csv, newline='') as fh:
            ceil_rows = [float(r['test_acc']) for r in csv.DictReader(fh)]
        if ceil_rows:
            ceiling = sum(ceil_rows) / len(ceil_rows)
            ax.axhline(ceiling, color='gray', linestyle='--',
                       label=f'non-DP ceiling ({ceiling:.3f})')

    ax.set_xscale('log')
    ax.set_xlabel(f'epsilon ({theorem}, post-hoc)')
    ax.set_ylabel(f'test {metric}')
    ax.set_title(f'{dataset}: SparseGNN privacy-utility frontier\n({meta})')
    ax.grid(True, alpha=0.3)
    ax.legend(title='edge sampling')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'frontier plot written to {args.out}')


if __name__ == '__main__':
    main()
