"""
PPI privacy-utility frontier over CONFIGURATIONS, not one configuration.

    python scripts/plot_ppi_pareto.py --out results/figures/ppi_pareto.png

Every tracked checkpoint of every (p2, sigma) cell is one point: its epsilon and
its utility.  The bold line is the Pareto envelope — the best utility reachable
at each budget, whichever configuration achieves it.  Points are coloured by p2
so it is visible WHICH regime owns each part of the frontier, which is the
question the mechanism is actually about (sparsify, or add noise?).
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

# Validated categorical slots 1, 2, 3, 7 (all six checks pass; the aqua's
# contrast WARN is relieved by the legend + direct labels).
P2_COLORS = {1.0: '#2a78d6', 0.5: '#eb6834', 0.25: '#1baf7a', 0.1: '#4a3aa7'}
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e3e2dc'

REFS = {
    'test_acc':   dict(trivial=0.4608, ceiling=0.6998, label='test micro-F1',
                       trivial_label='all-positive predictor (ε=0)'),
    'test_auroc': dict(trivial=0.4955, ceiling=0.8925, label='test AUROC',
                       trivial_label='chance (any constant predictor)'),
}


def load(pattern, metric):
    """[(eps, utility, p2, sigma, step)] over every tracked checkpoint."""
    pts = []
    for path in sorted(glob.glob(pattern)):
        by = defaultdict(list)
        meta = {}
        for r in csv.DictReader(open(path)):
            if not r.get('epsilon') or not r.get(metric):
                continue
            t = int(r['step'])
            by[t].append((float(r['epsilon']), float(r[metric])))
            meta[t] = (float(r['p2']), float(r['sigma']))
        for t, vals in by.items():
            eps = sum(e for e, _ in vals) / len(vals)
            util = sum(u for _, u in vals) / len(vals)
            pts.append((eps, util, meta[t][0], meta[t][1], t))
    return sorted(pts)


def pareto(points):
    """Running best utility as epsilon increases — the achievable envelope."""
    out, best = [], float('-inf')
    for eps, util, *_ in points:
        if util > best:
            best = util
            out.append((eps, util))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='results/ppi/pareto/*/*_with_eps.csv')
    ap.add_argument('--metric', default='test_acc')
    ap.add_argument('--out', default='results/figures/ppi_pareto.png')
    args = ap.parse_args()

    pts = load(args.glob, args.metric)
    if not pts:
        raise SystemExit(f"no epsilon-augmented rows matched {args.glob}")

    ref = REFS[args.metric]
    TRIVIAL, CEILING = ref['trivial'], ref['ceiling']

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for p2 in sorted(P2_COLORS, reverse=True):
        sel = [(e, u) for e, u, q, _, _ in pts if q == p2]
        if not sel:
            continue
        ax.scatter([e for e, _ in sel], [u for _, u in sel], s=14,
                   color=P2_COLORS[p2], alpha=0.55, linewidths=0, zorder=2,
                   label=f'p₂ = {p2:g}')

    front = pareto(pts)
    ax.step([e for e, _ in front], [u for _, u in front], where='post',
            color=INK, linewidth=2, zorder=4, label='Pareto envelope')

    ax.axhline(CEILING, color=MUTED, linestyle=(0, (5, 4)), linewidth=1.2, zorder=1)
    ax.axhline(TRIVIAL, color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
    ax.annotate('non-private ceiling (uncapped)', (0.015, CEILING),
                xycoords=('axes fraction', 'data'), textcoords='offset points',
                xytext=(0, 4), color=MUTED, fontsize=8)
    ax.annotate(ref['trivial_label'], (0.015, TRIVIAL),
                xycoords=('axes fraction', 'data'), textcoords='offset points',
                xytext=(0, -11), color=MUTED, fontsize=8)

    ax.set_xscale('log')
    lo, hi = min(e for e, *_ in pts), max(e for e, *_ in pts)
    ticks = [t for t in (0.3, 1, 3, 10, 30) if lo * 0.9 <= t <= hi * 1.1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('privacy budget  ε   (δ = 10⁻⁶)', fontsize=9, color=INK)
    ax.set_ylabel(ref['label'], fontsize=9, color=INK)
    ax.set_title('PPI: best achievable utility at each privacy budget',
                 fontsize=12, color=INK, loc='left', pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    # Data rises left-to-right, so the upper-left block is the free space;
    # lower-right collides with the chance reference line.
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK,
              ncol=2)

    fig.text(0.01, 0.005,
             f'{len(pts)} tracked checkpoints across 12 (p₂, σ) cells, r=1, L=2, '
             f'K=5, p₁=0.01; each point is a releasable iterate',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}  ({len(pts)} points, {len(front)} on the envelope)")


if __name__ == '__main__':
    main()
