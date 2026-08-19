"""
Stage 0-1 inductive utility ladder for ogbn-arxiv.

Reads the four non-DP CSVs produced by scripts/inductive_stage01.sh and draws
test accuracy vs edge-sampling probability p2, with the MLP baseline and the
full-edge GCN ceiling as horizontal references, one curve per expansion depth r.

  python scripts/plot_inductive_ladder.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RES = 'results'


def _read(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _by_p2(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[float(r['p2'])].append(float(r['test_acc']))
    return {p2: _mean_std(v) for p2, v in acc.items()}


def main():
    mlp = _mean_std([float(r['test_acc'])
                     for r in _read(f'{RES}/inductive_mlp/sparse_gnn_ogbn-arxiv_results.csv')])
    ceil = _mean_std([float(r['test_acc'])
                      for r in _read(f'{RES}/inductive_ceiling/sparse_gnn_ogbn-arxiv_results.csv')])
    r1 = _by_p2(_read(f'{RES}/inductive_stage1/sparse_gnn_ogbn-arxiv_results.csv'))
    r2 = _by_p2(_read(f'{RES}/inductive_stage1_r2/sparse_gnn_ogbn-arxiv_results.csv'))

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for tag, d, style in (('r=1', r1, 'o-'), ('r=2', r2, 's--')):
        xs = sorted(d)
        ax.errorbar(xs, [d[x][0] for x in xs], yerr=[d[x][1] for x in xs],
                    fmt=style, capsize=3, label=f'sparsified GCN ({tag})')

    ax.axhline(ceil[0], color='green', ls=':', lw=1.5,
               label=f'GCN ceiling, full edges ({ceil[0]:.3f})')
    ax.axhline(mlp[0], color='gray', ls=':', lw=1.5,
               label=f'MLP, graph-blind ({mlp[0]:.3f})')

    ax.set_xlabel(r'edge-sampling probability $p_2$  (1.0 = keep all edges)')
    ax.set_ylabel('test accuracy')
    ax.set_title('ogbn-arxiv (inductive): utility vs sparsification\n'
                 r'$p_1{=}0.005$, $K_{in}{=}K_{out}{=}5$, no DP, 3 seeds')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    out = f'{RES}/inductive_ladder.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    os.makedirs('paper/figures/experiments', exist_ok=True)
    fig.savefig('paper/figures/experiments/inductive_ladder.png',
                dpi=150, bbox_inches='tight')
    print(f'wrote {out} (+ copy under paper/figures/experiments/)')


if __name__ == '__main__':
    main()
