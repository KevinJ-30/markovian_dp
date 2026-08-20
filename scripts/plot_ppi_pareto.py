"""
PPI privacy-utility curves: one line per sparsification level.

    python scripts/plot_ppi_pareto.py --metric test_auroc \
        --out results/figures/ppi_pareto_auroc.png

Each curve shows, for one edge-sampling probability p2, the best utility reached
at each privacy budget.  Training is checkpointed every 50 steps and epsilon
grows with the number of steps, so one training run traces out a whole curve;
the curve for a given p2 is the best over the noise levels sigma we ran.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,   # noqa: E402
                               NullFormatter)

# Validated categorical slots 1, 2, 3, 7.
P2_COLORS = {1.0: '#2a78d6', 0.5: '#eb6834', 0.25: '#1baf7a', 0.1: '#4a3aa7'}
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e3e2dc'

# Reference lines, both non-private.  The floor is the graph-blind model
# (results/ppi/blind_L2: same mechanism at r=0, so each root sees only its own
# features) rather than a constant predictor — it is the bar the graph has to
# clear to be worth anything.
REFS = {
    'test_acc':   dict(floor=0.5090, ceiling=0.6998, label='test micro-F1',
                       floor_label='graph-blind, no privacy'),
    'test_auroc': dict(floor=0.7550, ceiling=0.8925, label='test AUROC',
                       floor_label='graph-blind, no privacy'),
}


def load_curves(pattern, metric):
    """{p2: [(eps, utility), ...]} — best utility at each budget, per p2.

    Averages over seeds, pools the noise levels, and takes a running best so
    each p2 gives one monotone curve.
    """
    per_p2 = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        by, meta = defaultdict(list), {}
        for r in csv.DictReader(open(path)):
            if not r.get('epsilon') or not r.get(metric):
                continue
            step = int(r['step'])
            by[step].append((float(r['epsilon']), float(r[metric])))
            meta[step] = float(r['p2'])
        for step, vals in by.items():
            eps = sum(e for e, _ in vals) / len(vals)
            util = sum(u for _, u in vals) / len(vals)
            per_p2[meta[step]].append((eps, util))

    curves = {}
    for p2, pts in per_p2.items():
        best, out = float('-inf'), []
        for eps, util in sorted(pts):
            best = max(best, util)
            out.append((eps, best))
        curves[p2] = out
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='results/ppi/pareto/*/*_with_eps.csv')
    ap.add_argument('--metric', default='test_acc')
    ap.add_argument('--out', default='results/figures/ppi_pareto.png')
    args = ap.parse_args()

    curves = load_curves(args.glob, args.metric)
    if not curves:
        raise SystemExit(f"no epsilon-augmented rows matched {args.glob}")
    ref = REFS[args.metric]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for p2 in sorted(curves, reverse=True):
        pts = curves[p2]
        xs = [e for e, _ in pts]
        ys = [u for _, u in pts]
        ax.plot(xs, ys, '-', color=P2_COLORS[p2], linewidth=2, zorder=3,
                label=f'p₂ = {p2:g}')

    ax.axhline(ref['ceiling'], color=MUTED, linestyle=(0, (5, 4)),
               linewidth=1.2, zorder=1)
    ax.axhline(ref['floor'], color=MUTED, linestyle=(0, (2, 3)),
               linewidth=1.2, zorder=1)
    ax.annotate('without privacy', (0.015, ref['ceiling']),
                xycoords=('axes fraction', 'data'), textcoords='offset points',
                xytext=(0, 4), color=MUTED, fontsize=8)
    ax.annotate(ref['floor_label'], (0.015, ref['floor']),
                xycoords=('axes fraction', 'data'), textcoords='offset points',
                xytext=(0, -11), color=MUTED, fontsize=8)

    ax.set_xscale('log')
    allx = [e for pts in curves.values() for e, _ in pts]
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30)
             if min(allx) * 0.9 <= t <= max(allx) * 1.1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('privacy budget  ε   (δ = 10⁻⁶)', fontsize=9, color=INK)
    ax.set_ylabel(ref['label'], fontsize=9, color=INK)
    ax.set_title('PPI: privacy–utility tradeoff by sparsification level',
                 fontsize=12, color=INK, loc='left', pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK)

    fig.text(0.01, 0.005,
             'p₂ is the fraction of edges kept.  2 seeds, r=1, L=2, K=5, '
             'p₁=0.01; ε grows with the number of training steps.',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
