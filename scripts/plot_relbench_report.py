"""
Meeting figures for the rel-f1 ladder, plus the epsilon-scaling figure that
answers "which graph attributes make sparsification cheap".

  python scripts/plot_relbench_report.py --out_dir results/figures
"""

import argparse
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.sparse.accounting import sparsegnn_substitution_epsilon as eps_of  # noqa: E402

# Validated categorical palette (light surface #fcfcfb): all six checks pass.
# Contrast WARN on slots 3-4 obliges visible labels, so every series is
# direct-labeled rather than relying on the legend alone.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b8b7b0'

B = 'results/relbench_relbench-f1-top3'


def load(path, val='test_acc'):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def mean_by(rows, key, val='test_acc'):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(float(r[val]))
    return {k: st.mean(v) for k, v in d.items()}


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6, color=MUTED)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_frontier(out):
    """Privacy-utility frontier: AUROC vs epsilon, one line per p2."""
    rows = load(f'{B}/dp_r2/sparse_gnn_relbench-f1-top3_dp_results_with_eps.csv')
    acc = mean_by(rows, lambda r: (float(r['p2']), float(r['sigma'])))
    eps = {(float(r['p2']), float(r['sigma'])): float(r['epsilon']) for r in rows}
    ceiling = st.mean([float(r['test_acc'])
                       for r in load(f'{B}/stage1_r2/sparse_gnn_relbench-f1-top3_results.csv')
                       if float(r['p2']) == 1.0])
    blind = st.mean([float(r['test_acc'])
                     for r in load(f'{B}/blind/sparse_gnn_relbench-f1-top3_results.csv')])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    ax.axhline(ceiling, color=INK2, lw=1.2, ls='--')
    ax.text(1.6, ceiling + .006, f'non-DP ceiling  {ceiling:.3f}',
            color=INK2, fontsize=8.5)
    ax.axhline(blind, color=MUTED, lw=1.2, ls=':')
    ax.text(1.6, blind + .006, f'graph-blind  {blind:.3f}', color=INK2, fontsize=8.5)
    ax.axhline(0.5, color=MUTED, lw=1.2)
    ax.text(1.6, 0.505, 'chance  0.500', color=INK2, fontsize=8.5)

    for i, p2 in enumerate(sorted({k[0] for k in acc}, reverse=True)):
        pts = sorted((eps[k], acc[k]) for k in acc if k[0] == p2)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, 'o-', color=SERIES[i], lw=2, ms=6,
                mec='#fcfcfb', mew=1.5, label=f'$p_2$={p2}', zorder=3)
        ax.annotate(f'$p_2$={p2}', (xs[-1], ys[-1]), textcoords='offset points',
                    xytext=(7, 0), color=SERIES[i], fontsize=9,
                    va='center', fontweight='medium')

    ax.set_xscale('log')
    ax.set_xlabel('privacy budget  $\\varepsilon$   (Theorem 6.4, $\\delta=10^{-6}$)',
                  color=INK2, fontsize=10)
    ax.set_ylabel('test AUROC', color=INK2, fontsize=10)
    ax.set_title('rel-f1 / driver-top3: privacy–utility frontier\n'
                 '$r$=2, $K_{in}$=20, $K_{out}$=3, $T$=900, 3 seeds; '
                 'each line sweeps $\\sigma \\in \\{20,10,5,2\\}$',
                 color=INK, fontsize=11, loc='left')
    ax.set_xlim(1.5, 1600)
    ax.legend(frameon=False, fontsize=9, loc='lower right', title='edge sampling',
              title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    print(f'wrote {out}')


def fig_ladder(out):
    """Non-DP ladder: utility vs p2 at each depth, against the reference lines."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    blind = st.mean([float(r['test_acc'])
                     for r in load(f'{B}/blind/sparse_gnn_relbench-f1-top3_results.csv')])
    ceil = st.mean([float(r['test_acc'])
                    for r in load(f'{B}/ceiling/sparse_gnn_relbench-f1-top3_results.csv')])

    # x-axis is inverted below, so place reference labels at p2=1.0 (left edge)
    # and keep them clear of the series' own direct labels on the right.
    ax.axhline(ceil, color=INK2, lw=1.2, ls='--')
    ax.text(0.99, ceil + .009, f'uncapped ceiling  {ceil:.3f}', color=INK2,
            fontsize=8.5, ha='left')
    ax.axhline(blind, color=MUTED, lw=1.2, ls=':')
    ax.text(0.99, blind + .009, f'graph-blind  {blind:.3f}', color=INK2,
            fontsize=8.5, ha='left')
    ax.axhline(0.5, color=MUTED, lw=1.2)
    ax.text(0.99, 0.505, 'chance  0.500', color=INK2, fontsize=8.5, ha='left')

    for i, R in enumerate((2, 1)):
        rows = load(f'{B}/stage1_r{R}/sparse_gnn_relbench-f1-top3_results.csv')
        d = mean_by(rows, lambda r: float(r['p2']))
        xs = sorted(d, reverse=True)
        ys = [d[x] for x in xs]
        ax.plot(xs, ys, 'o-', color=SERIES[i], lw=2, ms=7, mec='#fcfcfb', mew=1.5,
                label=f'$r$={R} (L={R})', zorder=3)
        ax.annotate(f'$r$={R} (L={R})', (xs[-1], ys[-1]),
                    textcoords='offset points',
                    xytext=(8, -3 if R == 1 else 4), color=SERIES[i],
                    fontsize=9.5, fontweight='medium', va='center')

    ax.set_xlabel('edge-retention probability  $p_2$   (1.0 = keep all edges)',
                  color=INK2, fontsize=10)
    ax.set_ylabel('test AUROC', color=INK2, fontsize=10)
    ax.set_title('rel-f1 / driver-top3: sparsification is nearly free, depth is not\n'
                 'no DP; $K_{in}$=20, $K_{out}$=3, $T$=900, 3 seeds',
                 color=INK, fontsize=11, loc='left')
    ax.invert_xaxis()
    ax.set_ylim(0.47, 0.87)
    ax.set_xlim(1.06, -0.02)   # headroom on the right for the direct labels
    ax.legend(frameon=False, fontsize=9, loc='center left')
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    print(f'wrote {out}')


def fig_eps_scaling(out):
    """What drives epsilon: out-degree and depth, from Theorem 6.4 directly."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    base = dict(p1=0.01, sigma=10.0, steps=2000, delta=1e-6)

    ax = axes[0]
    style(ax)
    kouts = [2, 3, 5, 8, 12, 20]
    for i, R in enumerate((1, 2)):
        ys = [eps_of(p2=1.0, r=R, K_in=5, K_out=k, **base) for k in kouts]
        ys = [y if y < 1e6 else float('nan') for y in ys]
        ax.plot(kouts, ys, 'o-', color=SERIES[i], lw=2, ms=6, mec='#fcfcfb',
                mew=1.5, label=f'$r$={R}')
        ax.annotate(f'$r$={R}', (kouts[-1], ys[-1]), textcoords='offset points',
                    xytext=(-28, 6), color=SERIES[i], fontsize=9.5)
    ax.set_yscale('log')
    ax.set_xlabel('out-degree cap  $K_{out}$', color=INK2, fontsize=10)
    ax.set_ylabel('$\\varepsilon$', color=INK2, fontsize=10)
    ax.set_title('$\\varepsilon$ scales as $K_{out}^{\\,r}$', color=INK,
                 fontsize=10.5, loc='left')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    style(ax)
    kins = [2, 3, 5, 8, 12, 20]
    for i, R in enumerate((1, 2)):
        ys = [eps_of(p2=1.0, r=R, K_in=k, K_out=3, **base) for k in kins]
        ax.plot(kins, ys, 'o-', color=SERIES[i], lw=2, ms=6, mec='#fcfcfb',
                mew=1.5, label=f'$r$={R}')
        ax.annotate(f'$r$={R}', (kins[-1], ys[-1]), textcoords='offset points',
                    xytext=(-28, 6), color=SERIES[i], fontsize=9.5)
    ax.set_yscale('log')
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel('in-degree cap  $K_{in}$   (at $K_{out}$=3)', color=INK2, fontsize=10)
    ax.set_title('$\\varepsilon$ barely moves with $K_{in}$', color=INK,
                 fontsize=10.5, loc='left')
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('In-expansion pays for OUT-degree, but reads signal from IN-degree\n'
                 '$p_1$=0.01, $p_2$=1.0, $\\sigma$=10, $T$=2000, $\\delta=10^{-6}$',
                 color=INK, fontsize=11, x=0.005, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/figures')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fig_frontier(os.path.join(a.out_dir, 'relbench_frontier.png'))
    fig_ladder(os.path.join(a.out_dir, 'relbench_ladder.png'))
    fig_eps_scaling(os.path.join(a.out_dir, 'epsilon_scaling.png'))


if __name__ == '__main__':
    main()
