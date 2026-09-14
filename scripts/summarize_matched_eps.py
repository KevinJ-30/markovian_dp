"""Compare the arms of a matched-epsilon sweep, against the graph-blind baseline.

    python scripts/summarize_matched_eps.py results/yelp_matched_eps

`summarize_sweep.py` reports one metric per cell.  This reports the pair that
matters for a private multilabel model -- the primary metric AND AUROC -- and
the margin over the blind arm at the same epsilon, which is the quantity the
paper actually claims.

Why both metrics.  micro-F1 reads a fixed logit>0 threshold, and DP noise
decalibrates that threshold far more than it damages the ranking, so micro-F1
understates a private model.  On PPI the all-positive predictor scores 0.4608
micro-F1 with AUROC exactly 0.5 -- no ranking ability at all -- so a model
"below trivial" on micro-F1 may still be ranking well.  Measured example: a
cell at 0.4044 micro-F1 (below the 0.4608 floor) had AUROC 0.6139.

Cells are selected on VALIDATION and reported on TEST, and NaN checkpoints are
skipped rather than silently returning the earliest step.
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict


def _cell(directory):
    """Best-on-validation checkpoint of one cell, or None if unreadable."""
    csvs = glob.glob(os.path.join(directory, '*_results.csv'))
    if not csvs:
        return None
    rows = list(csv.DictReader(open(csvs[0])))
    if not rows:
        return None

    by_step = defaultdict(list)
    for r in rows:
        by_step[r.get('step') or r['T']].append(r)

    def avg(rs, key):
        vals = [float(r[key]) for r in rs
                if r.get(key) not in (None, '') and float(r[key]) == float(r[key])]
        return sum(vals) / len(vals) if vals else float('nan')

    curve = {}
    for step, rs in by_step.items():
        v = avg(rs, 'val_acc')
        if v == v:                      # skip NaN validation checkpoints
            curve[step] = (v, avg(rs, 'test_acc'), avg(rs, 'test_auroc'))
    if not curve:
        return None

    head = rows[0]
    lower_better = head.get('metric') in ('mae', 'rmse')
    pick = min if lower_better else max
    best = pick(curve, key=lambda s: curve[s][0])
    return {
        'sigma': float(head['sigma']) if head.get('sigma') else float('nan'),
        'eps_target': head.get('target_epsilon') or '',
        'r': head.get('r', ''), 'p2': head.get('p2', ''),
        'K': head.get('K_out', ''), 'model': head.get('model', ''),
        'metric': head.get('metric', 'accuracy'),
        'trivial': float(head['trivial_baseline']) if head.get('trivial_baseline') else float('nan'),
        'step': best, 'test': curve[best][1], 'auroc': curve[best][2],
        'dp': head.get('dp') == 'True',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='sweep directory, e.g. results/yelp_matched_eps')
    args = ap.parse_args()

    cells = {}
    for d in sorted(glob.glob(os.path.join(args.root, '*/'))):
        got = _cell(d)
        if got:
            cells[os.path.basename(d.rstrip('/'))] = got
    if not cells:
        raise SystemExit(f"no readable result CSVs under {args.root}")

    # eps is encoded in the directory name by the driver (…_eps8.0).
    def eps_of(name):
        m = re.search(r'eps([0-9.]+)', name)
        return m.group(1) if m else None

    blind = {eps_of(n): c for n, c in cells.items() if n.startswith('dpmlp')}
    metric = next(iter(cells.values()))['metric']
    trivial = next((c['trivial'] for c in cells.values() if c['trivial'] == c['trivial']),
                   float('nan'))

    print(f"\n{args.root}   primary metric: {metric}")
    print(f"{'cell':<28}{'sigma':>9}{'step':>7}{metric:>11}{'AUROC':>9}"
          f"{'  vs blind (' + metric + ' / AUROC)':>28}")
    print('-' * 92)

    for name in sorted(cells, key=lambda n: (eps_of(n) or '', n)):
        c = cells[name]
        b = blind.get(eps_of(name))
        if b and not name.startswith('dpmlp'):
            margin = (f"{100*(c['test']-b['test']):+7.2f} / "
                      f"{100*(c['auroc']-b['auroc']):+7.2f}")
        else:
            margin = '' if not name.startswith('dpmlp') else '  (this is the blind arm)'
        sig = f"{c['sigma']:.3f}" if c['dp'] else '--'
        print(f"{name:<28}{sig:>9}{c['step']:>7}{c['test']:>11.4f}"
              f"{c['auroc']:>9.4f}{margin:>28}")

    print('-' * 92)
    print(f"{'trivial baseline':<28}{'--':>9}{'--':>7}{trivial:>11.4f}{0.5:>9.4f}")
    print("\nmargins are in percentage points, GNN minus blind at the same epsilon.")
    print("AUROC is threshold-free; the trivial predictor's AUROC is 0.5 by "
          "construction, so\nAUROC > 0.5 means the model ranks, even when "
          f"{metric} sits below the trivial bar.")


if __name__ == '__main__':
    main()
