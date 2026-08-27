"""
Privacy-utility curves for any dataset: best utility at each budget, per p2.

    python scripts/plot_frontier.py --glob 'results/reddit/transductive_p*/*_with_eps.csv' \
        --metric test_acc --floor 0.1483 --ceiling 0.9382 \
        --floor-label 'majority class' --ceiling-label 'without privacy' \
        --ylabel 'test accuracy' --title 'Reddit (transductive)' \
        --out results/figures/reddit_frontier.png

One line per edge-keep probability p2; each is the best utility reached at a
given budget (epsilon grows with training steps, so one run traces a curve).
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

P2_COLORS = {1.0: '#2a78d6', 0.5: '#eb6834', 0.25: '#1baf7a', 0.1: '#4a3aa7'}
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e3e2dc'


def load_curves(pattern, metric):
    per_p2 = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        by, meta = defaultdict(list), {}
        for r in csv.DictReader(open(path)):
            if not r.get('epsilon') or not r.get(metric):
                continue
            step = int(r['step'])
            try:
                by[step].append((float(r['epsilon']), float(r[metric])))
            except ValueError:
                continue
            meta[step] = float(r['p2'])
        for step, vals in by.items():
            e = sum(x for x, _ in vals) / len(vals)
            u = sum(x for _, x in vals) / len(vals)
            per_p2[meta[step]].append((e, u))
    curves = {}
    for p2, pts in per_p2.items():
        best, out = float('-inf'), []
        for e, u in sorted(pts):
            best = max(best, u)
            out.append((e, best))
        curves[p2] = out
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--metric', default='test_acc')
    ap.add_argument('--floor', type=float)
    ap.add_argument('--ceiling', type=float)
    ap.add_argument('--floor-label', default='baseline')
    ap.add_argument('--ceiling-label', default='without privacy')
    ap.add_argument('--ylabel', default='test metric')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    curves = load_curves(a.glob, a.metric)
    if not curves:
        raise SystemExit(f"no epsilon-augmented rows matched {a.glob}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for p2 in sorted(curves, reverse=True):
        xs = [e for e, _ in curves[p2]]
        ys = [u for _, u in curves[p2]]
        ax.plot(xs, ys, '-', color=P2_COLORS.get(p2, INK), linewidth=2,
                zorder=3, label=f'p₂ = {p2:g}')

    for val, ls, lbl, dy in ((a.ceiling, (0, (5, 4)), a.ceiling_label, 4),):
        if val is None:
            continue
        ax.axhline(val, color=MUTED, linestyle=ls, linewidth=1.2, zorder=1)
        ax.annotate(lbl, (0.015, val), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(0, dy),
                    color=MUTED, fontsize=8)

    ax.set_xscale('log')
    allx = [e for pts in curves.values() for e, _ in pts]
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100)
             if min(allx) * 0.9 <= t <= max(allx) * 1.1]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('privacy budget  ε', fontsize=9, color=INK)
    ax.set_ylabel(a.ylabel, fontsize=9, color=INK)
    if a.title:
        ax.set_title(a.title, fontsize=12, color=INK, loc='left', pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
