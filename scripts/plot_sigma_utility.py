"""
DP utility vs noise multiplier sigma (no epsilon axis).

Reads a DP results CSV from `src.sparse.run --dp` and plots test accuracy vs
sigma, one line per edge-sampling probability p2, averaged over seeds.

  python scripts/plot_sigma_utility.py \
      --csv results/inductive_stage2/sparse_gnn_ogbn-arxiv_dp_results.csv \
      --ceiling 0.518 --out results/inductive_sigma_utility.png
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ceiling', type=float, default=None,
                    help='non-DP ceiling to draw as a dashed reference line')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.csv, newline='') as fh:
        rows = list(csv.DictReader(fh))

    acc = defaultdict(list)          # (p2, sigma) -> [test_acc, ...]
    for r in rows:
        acc[(float(r['p2']), float(r['sigma']))].append(float(r['test_acc']))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p2 in sorted({k[0] for k in acc}):        # p2=0.1 first
        pts = sorted((s, *_mean_std(acc[(p2, s)])) for s in
                     {k[1] for k in acc if k[0] == p2})
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[p[2] for p in pts], fmt='o-', capsize=3,
                    label=f'p2={p2}')

    if args.ceiling is not None:
        ax.axhline(args.ceiling, color='gray', ls='--',
                   label=f'non-DP ceiling ({args.ceiling:.3f})')

    ax.set_xlabel(r'noise multiplier $\sigma$  (more noise $\rightarrow$)')
    ax.set_ylabel('test accuracy')
    ax.set_title('ogbn-arxiv (inductive): DP utility vs noise\n'
                 r'$p_1{=}0.005$, $K{=}5$, $r{=}1$, $C{=}1$, 2 seeds')
    ax.grid(True, alpha=0.3)
    ax.legend(title='edge sampling', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
