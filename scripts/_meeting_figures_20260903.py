"""
Meeting figures, 2026-09-03: facebook (local, complete), PPI (ICE, partial —
wallclock exceeded), ogbn-arxiv inductive (ICE, complete).

    python scripts/_meeting_figures_20260903.py

Writes PNGs to results/figures/2026-09-03/.  House style copied from
scripts/plot_frontier.py and scripts/plot_ppi_frontier.py: INK/MUTED/GRID,
log2 x-axis with power-of-2 ticks, no top/right spines, dashed muted reference
lines, frameless legends.  No trivial-baseline line on any chart, per Kevin —
the floor of interest is the graph-blind (DP-MLP) reference, not chance.

Data is read directly from the run CSVs, not hardcoded: target p2/epsilon are
parsed from each cell's output directory name (sigma was solved FOR that
target by scripts/calibrate_grid.py, so the directory name is the exact
epsilon by construction — no need to round-trip through compute_epsilon).
"""

import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e3e2dc'
# Categorical slots 1/2/3/7 (existing P2_COLORS convention, plot_frontier.py)
# plus slot 8 (red) for the graph-blind / DP-MLP reference series.
P2_COLORS = {1.0: '#2a78d6', 0.5: '#eb6834', 0.25: '#1baf7a', 0.1: '#4a3aa7'}
BLIND_COLOR = '#e34948'

OUT_DIR = 'results/figures/2026-09-03'
_POW2 = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _log2_axis(ax, lo, hi):
    ax.set_xscale('log', base=2)
    ax.set_xlim(lo * 0.8, hi * 1.3)
    ticks = [t for t in _POW2 if lo * 0.7 <= t <= hi * 1.4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


def _ref_line(ax, value, label, style=(0, (5, 4))):
    # Skip silently-wrong rendering rather than silently-wrong data: a value
    # outside the axes' own ylim still gets its annotation text object placed
    # at that (off-screen) data y, and savefig(bbox_inches='tight') includes
    # every artist's true extent regardless of the visible viewport -- so one
    # out-of-range reference line inflates the whole canvas to reach it. Call
    # this AFTER ax.set_ylim, not before.
    lo, hi = ax.get_ylim()
    if not (lo <= value <= hi):
        print(f'  note: reference "{label}" = {value:.3f} is outside ylim '
              f'({lo:.2f}, {hi:.2f}) — omitted from plot, stated in title instead')
        return False
    ax.axhline(value, color=MUTED, linestyle=style, linewidth=1.2, zorder=1)
    ax.annotate(label, (0.02, value), xycoords=('axes fraction', 'data'),
                textcoords='offset points', xytext=(0, 4),
                color=MUTED, fontsize=8)
    return True


def _save(fig, name, already_tight=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    if not already_tight:
        fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


def final_acc(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None, 0
    step = lambda r: int(float(r.get('step') or r['T']))
    T = max(step(r) for r in rows)
    accs = [float(r['test_acc']) for r in rows if step(r) == T]
    return sum(accs) / len(accs), len(accs)


def load_matched_frontier(root, gnn_glob_csv, blind_glob_csv=None):
    """{p2: [(eps, acc)]} and {eps: acc} for the blind arm, from dir-name targets."""
    gnn = defaultdict(list)
    for f in glob.glob(os.path.join(root, gnn_glob_csv)):
        d = os.path.basename(os.path.dirname(f))
        m = re.match(r'gnn_p2([\d.]+)_eps([\d.]+)', d)
        if not m:
            continue
        acc, n = final_acc(f)
        if acc is not None:
            gnn[float(m.group(1))].append((float(m.group(2)), acc))
    blind = {}
    if blind_glob_csv:
        for f in glob.glob(os.path.join(root, blind_glob_csv)):
            d = os.path.basename(os.path.dirname(f))
            m = re.match(r'blind_eps([\d.]+)', d)
            if not m:
                continue
            acc, n = final_acc(f)
            if acc is not None:
                blind[float(m.group(1))] = acc
    return gnn, blind


def plot_frontier(gnn, blind, ceiling, ceiling_label, title, out_name,
                  ylim=None, blind_ceiling=None, blind_ceiling_label=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    all_eps = [e for pts in gnn.values() for e, _ in pts] + list(blind.keys())

    for p2 in sorted(gnn, reverse=True):
        pts = sorted(gnn[p2])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        color = P2_COLORS.get(p2, INK)
        marker = 'o-' if len(pts) > 1 else 'o'
        ax.plot(xs, ys, marker, color=color, linewidth=2, markersize=5,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3,
                label=f'p₂ = {p2:g}')

    if blind:
        pts = sorted(blind.items())
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        marker = 'D-' if len(pts) > 1 else 'D'
        ax.plot(xs, ys, marker, color=BLIND_COLOR, linewidth=2, markersize=5,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3,
                linestyle='--', label='graph-blind (DP-MLP)')

    _log2_axis(ax, min(all_eps), max(all_eps))
    if ylim:
        ax.set_ylim(*ylim)
    else:
        # Reference lines can be far outside the data's natural range (e.g. a
        # non-DP ceiling well above every DP point); autoscale to the DATA
        # first and freeze it, so a later out-of-range reference is detected
        # by _ref_line rather than silently stretching the axes to fit it.
        ax.relim(); ax.autoscale_view()
        ax.set_ylim(*ax.get_ylim())

    omitted = []
    if ceiling is not None and not _ref_line(ax, ceiling, ceiling_label):
        omitted.append(ceiling_label)
    if blind_ceiling is not None and not _ref_line(ax, blind_ceiling, blind_ceiling_label,
                                                   style=(0, (2, 3))):
        omitted.append(blind_ceiling_label)
    if omitted:
        title = title + '\n(off-scale: ' + '; '.join(omitted) + ')'

    ax.set_xlabel('privacy budget  ε', fontsize=9, color=INK)
    ax.set_ylabel('test accuracy', fontsize=9, color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc='left', pad=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK)
    _save(fig, out_name)


def plot_grouped_bars(groups, series, title, out_name, ylim=None,
                      ref_line=None, ref_label=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    n_groups, n_series = len(groups), len(series)
    bar_w = 0.8 / n_series
    x = range(n_groups)
    for i, (label, values, color) in enumerate(series):
        offs = [xi + (i - (n_series - 1) / 2) * bar_w for xi in x]
        vals = [v if v is not None else 0 for v in values]
        ax.bar(offs, vals, width=bar_w * 0.92, color=color, label=label,
              zorder=3)
        for xi, v, orig in zip(offs, vals, values):
            if orig is None:
                continue
            ax.annotate(f'{orig:.3f}', (xi, v), textcoords='offset points',
                        xytext=(0, 3), ha='center', fontsize=7.5, color=MUTED)
    if ref_line is not None:
        # A legend entry, not on-plot text: a horizontal dashed line crossing
        # bar bodies has nowhere to put a text label that cannot collide with
        # some bar's value label, so identity goes in the legend instead.
        ax.axhline(ref_line, color=MUTED, linestyle=(0, (2, 3)),
                  linewidth=1.2, zorder=1, label=ref_label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=8.5, color=INK)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel('test accuracy', fontsize=9, color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc='left', pad=10)
    _style_axes(ax)
    # Below the axes, not inside it: with a truncated ylim (bars filling most
    # of the visible height) no corner of the plot is ever fully clear.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='upper center',
             bbox_to_anchor=(0.5, -0.13),
             ncol=len(series) + (1 if ref_line is not None else 0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out_name, already_tight=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. FACEBOOK — matched-epsilon sparsification frontier (complete, 20 cells)
# ══════════════════════════════════════════════════════════════════════════
fb_gnn = defaultdict(list)
for f in glob.glob('results/facebook_width/dp_h256_*/sparse_gnn_facebook_dp_results.csv'):
    d = os.path.basename(os.path.dirname(f))
    m = re.match(r'dp_h256_p2([\d.]+)_eps([\d.]+)', d)
    acc, n = final_acc(f)
    if acc is not None:
        fb_gnn[float(m.group(1))].append((float(m.group(2)), acc))

# Non-DP ceiling CSV has both p2=1.0 and p2=0.1 rows at the same step; isolate dense.
_rows = list(csv.DictReader(open('results/facebook_width/nodp_h256_r1/sparse_gnn_facebook_results.csv')))
_T = max(int(float(r['step'])) for r in _rows)
_dense = [float(r['test_acc']) for r in _rows if int(float(r['step'])) == _T and float(r['p2']) == 1.0]
fb_ceiling = sum(_dense) / len(_dense)

plot_frontier(
    fb_gnn, {}, fb_ceiling, f'no privacy, dense ({fb_ceiling:.3f})',
    'Facebook: sparsification frontier at matched ε\n'
    '(hidden=256, K=5, r=1)',
    'facebook_sparsification_frontier.png', ylim=(0.15, 0.68))

# ══════════════════════════════════════════════════════════════════════════
# 2. FACEBOOK — width ablation (ruled out): two panels, r=1 and r=2
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
for ax, r in zip(axes, (1, 2)):
    hiddens = [16, 64, 256]
    dense_vals, sparse_vals = [], []
    for h in hiddens:
        f = f'results/facebook_width/nodp_h{h}_r{r}/sparse_gnn_facebook_results.csv'
        rows = list(csv.DictReader(open(f)))
        T = max(int(float(x['step'])) for x in rows)
        d = [float(x['test_acc']) for x in rows if int(float(x['step'])) == T and float(x['p2']) == 1.0]
        s = [float(x['test_acc']) for x in rows if int(float(x['step'])) == T and float(x['p2']) == 0.1]
        dense_vals.append(sum(d) / len(d))
        sparse_vals.append(sum(s) / len(s))
    x = range(len(hiddens))
    bw = 0.35
    ax.bar([xi - bw / 2 for xi in x], dense_vals, width=bw * 0.92,
           color=P2_COLORS[1.0], label='p₂ = 1.0', zorder=3)
    ax.bar([xi + bw / 2 for xi in x], sparse_vals, width=bw * 0.92,
           color=P2_COLORS[0.1], label='p₂ = 0.1', zorder=3)
    for xi, v in zip(x, dense_vals):
        ax.annotate(f'{v:.3f}', (xi - bw / 2, v), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=7.5, color=MUTED)
    for xi, v in zip(x, sparse_vals):
        ax.annotate(f'{v:.3f}', (xi + bw / 2, v), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=7.5, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'hidden={h}' for h in hiddens], fontsize=8.5, color=INK)
    ax.set_ylim(0, 0.72)
    ax.set_title(f'r = {r}', fontsize=11, color=INK, loc='left', pad=8)
    _style_axes(ax)
axes[0].set_ylabel('non-DP test accuracy', fontsize=9, color=INK)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK,
          loc='upper right', bbox_to_anchor=(0.99, 1.06), ncol=2)
fig.suptitle('Facebook: width ablation (non-DP)', fontsize=12, color=INK,
            x=0.02, ha='left', y=1.1)
_save(fig, 'facebook_width_ablation.png')

# ══════════════════════════════════════════════════════════════════════════
# 3. PPI — capacity ablation (Stage 1, non-DP; the free +10-14pt fix)
# ══════════════════════════════════════════════════════════════════════════
def ppi_stage1_acc(tag, p2):
    f = f'results/ppi_stage1/{tag}/sparse_gnn_ppi_results.csv'
    rows = list(csv.DictReader(open(f)))
    T = max(int(float(x['step'])) for x in rows)
    a = [float(x['test_acc']) for x in rows if int(float(x['step'])) == T and float(x['p2']) == p2]
    return sum(a) / len(a)

cats = ['h=16\nK=5', 'h=64\nK=5', 'h=256\nK=5', 'h=256\nK=10', 'h=256\nK=25']
r1_vals = [ppi_stage1_acc('h16_K5', 1.0), ppi_stage1_acc('h64_K5', 1.0),
           ppi_stage1_acc('h256_K5', 1.0), ppi_stage1_acc('h256_K10', 1.0),
           ppi_stage1_acc('h256_K25', 1.0)]
def r2_from(tag):
    f = f'results/ppi_stage1/{tag}/sparse_gnn_ppi_results.csv'
    rows = list(csv.DictReader(open(f)))
    T = max(int(float(x['step'])) for x in rows)
    a = [float(x['test_acc']) for x in rows if int(float(x['step'])) == T and int(float(x['r'])) == 2]
    return sum(a) / len(a) if a else None
r2_vals = [r2_from('h16_K5'), r2_from('h64_K5'), r2_from('h256_K5'),
           r2_from('h256_K10'), r2_from('h256_K25')]
blind_acc, _ = final_acc('results/ppi_stage1/blind_h256/sparse_gnn_ppi_results.csv')

plot_grouped_bars(
    cats,
    [('r = 1', r1_vals, '#2a78d6'), ('r = 2', r2_vals, '#4a3aa7')],
    'PPI: capacity ablation (non-DP, p2=1.0)',
    'ppi_capacity_ablation.png', ylim=(0.40, 0.82),
    ref_line=blind_acc, ref_label=f'graph-blind (r=0), hidden=256 ({blind_acc:.3f})')

# ══════════════════════════════════════════════════════════════════════════
# 4. PPI — matched-epsilon frontier (PARTIAL: 9/24 cells, blind arm missing)
# ══════════════════════════════════════════════════════════════════════════
ppi_gnn, ppi_blind = load_matched_frontier(
    'results/ppi_matched_eps', 'gnn_p2*/sparse_gnn_ppi_dp_results.csv',
    'blind_eps*/sparse_gnn_ppi_dp_results.csv')
_rows = list(csv.DictReader(open('results/ppi_stage1/h256_K5/sparse_gnn_ppi_results.csv')))
_T = max(int(float(r['step'])) for r in _rows)
_r1 = [float(r['test_acc']) for r in _rows if int(float(r['step'])) == _T and int(float(r['r'])) == 1]
ppi_ceiling = sum(_r1) / len(_r1)

plot_frontier(
    ppi_gnn, ppi_blind, ppi_ceiling, f'no privacy, GNN ({ppi_ceiling:.3f})',
    'PPI: sparsification frontier at matched ε — partial (9/24 cells)\n'
    '(hidden=256, K=5, r=1)',
    'ppi_matched_eps_frontier_partial.png', ylim=(0.35, 0.68))

# ══════════════════════════════════════════════════════════════════════════
# 5. PPI — supplementary: K/p2 trade at matched eps~1.5 (legacy hidden, DP)
# ══════════════════════════════════════════════════════════════════════════
k_cells = [('K5_p1.0_dp1', 5, 1.0), ('K10_p0.5_dp1', 10, 0.5),
          ('K20_p0.25_dp1', 20, 0.25), ('K50_p0.1_dp1', 50, 0.1)]
fig, ax = plt.subplots(figsize=(6.5, 4.3))
xs, ys, colors, labels = [], [], [], []
eps_seen = []
for tag, K, p2 in k_cells:
    f = f'results/ppi/ppi_k/{tag}/sparse_gnn_ppi_dp_results_with_eps.csv'
    rows = list(csv.DictReader(open(f)))
    T = max(int(float(x.get('step') or x['T'])) for x in rows)
    a = [float(x['test_acc']) for x in rows if int(float(x.get('step') or x['T'])) == T]
    xs.append(K); ys.append(sum(a) / len(a)); colors.append(P2_COLORS[p2])
    eps_seen.append(float(rows[0]['epsilon']))
ax.bar([str(k) for k in xs], ys, color=colors, width=0.6, zorder=3)
for i, (K, v, p2) in enumerate(zip(xs, ys, (1.0, 0.5, 0.25, 0.1))):
    ax.annotate(f'p₂={p2:g}\n{v:.3f}', (i, v), textcoords='offset points',
                xytext=(0, 4), ha='center', fontsize=8, color=MUTED)
ax.set_ylim(0.40, 0.50)
ax.set_xlabel('K (K_in = K_out)', fontsize=9, color=INK)
ax.set_ylabel('test accuracy', fontsize=9, color=INK)
ax.set_title(f'PPI: K vs p₂ trade at matched ε≈{sum(eps_seen)/len(eps_seen):.2f}\n'
            '(hidden=64)',
            fontsize=11.5, color=INK, loc='left', pad=10)
_style_axes(ax)
_save(fig, 'ppi_K_p2_tradeoff_supplementary.png')

# ══════════════════════════════════════════════════════════════════════════
# 6. ARXIV — capacity ablation (width transfers here)
# ══════════════════════════════════════════════════════════════════════════
h16_acc, _ = final_acc('results/arxiv_matched_eps/nodp_h16/sparse_gnn_ogbn-arxiv_results.csv')
h256_acc, _ = final_acc('results/arxiv_matched_eps/nodp_h256/sparse_gnn_ogbn-arxiv_results.csv')
blind_acc, _ = final_acc('results/arxiv_matched_eps/nodp_blind_h256/sparse_gnn_ogbn-arxiv_results.csv')

plot_grouped_bars(
    ['hidden=16', 'hidden=256'],
    [('GNN, r=1', [h16_acc, h256_acc], '#2a78d6')],
    'ogbn-arxiv (inductive): capacity ablation (non-DP, p2=1.0)',
    'arxiv_capacity_ablation.png', ylim=(0.40, 0.62),
    ref_line=blind_acc, ref_label=f'graph-blind (r=0), hidden=256 ({blind_acc:.3f})')

# ══════════════════════════════════════════════════════════════════════════
# 7. ARXIV — matched-epsilon frontier (COMPLETE, 24/24 — the concerning one)
# ══════════════════════════════════════════════════════════════════════════
ax_gnn, ax_blind = load_matched_frontier(
    'results/arxiv_matched_eps', 'gnn_p2*/sparse_gnn_ogbn-arxiv_dp_results.csv',
    'blind_eps*/sparse_gnn_ogbn-arxiv_dp_results.csv')

plot_frontier(
    ax_gnn, ax_blind, h256_acc, f'no privacy, GNN ({h256_acc:.3f})',
    'ogbn-arxiv (inductive): sparsification frontier at matched ε\n'
    '(hidden=256, K=5, r=1)',
    'arxiv_matched_eps_frontier.png', ylim=(0.26, 0.32),
    blind_ceiling=blind_acc, blind_ceiling_label=f'no privacy, blind ({blind_acc:.3f})')

print('\nall figures written to', OUT_DIR)
