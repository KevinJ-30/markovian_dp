"""
PPI privacy-utility frontier: utility against epsilon, one panel per metric.

    python scripts/plot_ppi_frontier.py --out results/figures/ppi_frontier.png

Two panels rather than two y-axes on one plot: micro-F1 and AUROC are both
scores in [0, 1] but have different floors, and the contrast between those
floors is the point — the all-positive predictor scores 0.4608 micro-F1 with
chance-level (0.4955) ranking ability, so a model can sit below the F1 floor
while clearly learning.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullLocator,  # noqa: E402
                               NullFormatter)

# Categorical slots 1 and 7 of the validated palette (CVD-checked pair).
BLUE, VIOLET = '#2a78d6', '#4a3aa7'
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e3e2dc'

# Measured references (results/ppi/ppi_ceiling_true, uncapped full-batch).
# Both references are non-private; the floor is the graph-blind model
# (results/ppi/blind_L2, same mechanism at r=0).
REFS = {
    'test_acc':   dict(floor=0.5090, ceiling=0.6998,
                       floor_label='graph-blind, no privacy',
                       ceiling_label='full graph, no privacy',
                       title='micro-F1', ylim=(0.35, 0.75)),
    'test_auroc': dict(floor=0.7550, ceiling=0.8925,
                       floor_label='graph-blind, no privacy',
                       ceiling_label='full graph, no privacy',
                       title='AUROC', ylim=(0.50, 0.95)),
}


def load(pattern):
    """{sigma: [(eps, {metric: mean})]} averaged over seeds, sorted by eps."""
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(path)):
            if not r.get('epsilon'):
                continue
            key = float(r['sigma'])
            eps = float(r['epsilon'])
            for m in REFS:
                if r.get(m):
                    runs[key][eps][m].append(float(r[m]))
    return {s: sorted((e, {m: sum(v) / len(v) for m, v in mv.items()})
                      for e, mv in by_eps.items())
            for s, by_eps in runs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob',
                    default='results/ppi/frontier/*/sparse_gnn_ppi_dp_results_with_eps.csv')
    ap.add_argument('--out', default='results/figures/ppi_frontier.png')
    args = ap.parse_args()

    data = load(args.glob)
    if not data:
        raise SystemExit(f"no epsilon-augmented rows matched {args.glob}")

    all_eps = [e for pts in data.values() for e, _ in pts]
    lo_x, hi_x = min(all_eps), max(all_eps)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (metric, ref) in zip(axes, REFS.items()):
        for color, (sigma, pts) in zip((BLUE, VIOLET), sorted(data.items())):
            xs = [e for e, mv in pts if metric in mv]
            ys = [mv[metric] for _, mv in pts if metric in mv]
            if not xs:
                continue
            ax.plot(xs, ys, '-', color=color, linewidth=2, zorder=3,
                    marker='o', markersize=4, markevery=max(1, len(xs) // 8),
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=f'σ = {sigma:g}')
            ax.annotate(f'σ={sigma:g}', (xs[-1], ys[-1]),
                        textcoords='offset points', xytext=(6, -2),
                        color=color, fontsize=9)

        ax.axhline(ref['ceiling'], color=MUTED, linestyle=(0, (5, 4)),
                   linewidth=1.2, zorder=1)
        ax.axhline(ref['floor'], color=MUTED, linestyle=(0, (2, 3)),
                   linewidth=1.2, zorder=1)
        ax.annotate(ref['ceiling_label'], (0.02, ref['ceiling']),
                    xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(0, 4),
                    color=MUTED, fontsize=8)
        ax.annotate(ref['floor_label'], (0.02, ref['floor']),
                    xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(0, -11),
                    color=MUTED, fontsize=8)

        ax.set_xscale('log', base=2)
        ax.set_xlim(lo_x * 0.8, hi_x * 1.25)
        _t = [t for t in (0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
              if lo_x * 0.7 <= t <= hi_x * 1.4]
        ax.xaxis.set_major_locator(FixedLocator(_t))
        ax.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in _t]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(*ref['ylim'])
        ax.set_xlabel('privacy budget  ε   (δ = 10⁻⁶)', fontsize=9, color=INK)
        ax.set_title(ref['title'], fontsize=11, color=INK, loc='left', pad=8)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    if len(data) > 1:          # one series is named by its direct label
        axes[0].legend(frameon=False, fontsize=9, loc='center right',
                       labelcolor=INK)
    fig.suptitle('PPI: privacy–utility frontier (p₂=0.1, r=1, L=2, K=5)',
                 fontsize=12, color=INK, x=0.02, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
