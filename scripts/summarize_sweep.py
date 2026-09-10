"""
Summarize a sweep directory: one row per cell, at its best tracked checkpoint.

    python scripts/summarize_sweep.py results/ppi/sweep_lr

Reports the best checkpoint rather than the last, because DP runs frequently
peak mid-training and then decay under the noise.  When a cell has per-checkpoint
epsilon (from compute_epsilon on a --track_every CSV), the epsilon at that same
checkpoint is shown, so cells are read at matched privacy rather than matched
step count.

The checkpoint is selected by VALIDATION performance, not test -- picking the
step that scores best on test and then reporting that same test score is
checkpoint-selection leakage (2026-09-09: caught this after regression runs
with noisy test curves were showing implausibly good "best" numbers). Select
on val, report the test value at that step.
"""

import argparse
import csv
import glob
import os
import sys


def _rows(path):
    return list(csv.DictReader(open(path)))


def _curve(rows, key):
    if not rows or key not in rows[0]:
        return None, None
    by, eps = {}, {}
    for r in rows:
        step = int(float(r.get('step') or r['T']))
        try:
            value = float(r[key])              # blank metric column raises
        except ValueError:
            continue
        by.setdefault(step, []).append(value)
        if r.get('epsilon'):
            eps[step] = float(r['epsilon'])
    if not by:
        return None, None
    return {t: sum(v) / len(v) for t, v in by.items()}, eps


def _lower_is_better(rows, metric_arg):
    # run.py writes every mechanism's primary metric into a column literally
    # named test_acc regardless of what it measures -- accuracy for
    # classification, mae for regression_gnn -- so the --metric column name
    # cannot tell direction. The CSV's own `metric` field (the mechanism's
    # real metric name, e.g. "mae"/"accuracy"/"auroc"/"micro_f1") can.
    declared = (rows[0].get('metric') or '').lower() if rows else ''
    if declared:
        return declared in ('mae', 'rmse')
    return any(tag in metric_arg for tag in ('mae', 'rmse'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_dir')
    ap.add_argument('--metric', default='test_acc')
    args = ap.parse_args()

    cells = sorted(d for d in glob.glob(os.path.join(args.sweep_dir, '*'))
                   if os.path.isdir(d))
    if not cells:
        raise SystemExit(f"no cell directories under {args.sweep_dir}")

    print(f"{'cell':<24} {'best':>8} {'step':>6} {'eps':>8} {'auroc':>8} {'final':>8}")
    print('-' * 66)
    for d in cells:
        # Prefer the epsilon-augmented CSV when compute_epsilon has run.
        csvs = (sorted(glob.glob(f'{d}/*_with_eps.csv'))
                or sorted(glob.glob(f'{d}/*_results.csv')))
        if not csvs:
            continue
        rows = _rows(csvs[0])
        curve, eps = _curve(rows, args.metric)
        if not curve:
            continue
        # Select the checkpoint on VALIDATION, report TEST at that step --
        # selecting on test itself is leakage (see module docstring). Fall
        # back to selecting on test only if there is truly no val column to
        # select on, and say so, rather than silently picking test's optimum.
        val_key = ('val_' + args.metric[len('test_'):]
                  if args.metric.startswith('test_') else None)
        val_curve, _ = _curve(rows, val_key) if val_key else (None, None)
        # mae/rmse are losses (lower = better); every other metric here
        # (accuracy, micro_f1, auroc) is a score (higher = better).  Picking
        # "best" with the wrong direction silently reports the worst
        # checkpoint as the best one.
        lower_is_better = _lower_is_better(rows, args.metric)
        if val_curve:
            if lower_is_better:
                best = min(val_curve, key=val_curve.get)
            else:
                best = max(val_curve, key=val_curve.get)
            if best not in curve:
                continue
        else:
            print(f"  WARNING: no {val_key or 'val'} column for {d}; "
                  f"selecting on test itself (leakage)", file=sys.stderr)
            if lower_is_better:
                best = min(curve, key=curve.get)
            else:
                best = max(curve, key=curve.get)
        au, _ = _curve(rows, 'test_auroc')
        au_s = f"{au[best]:.4f}" if au and best in au else '-'
        eps_s = f"{eps[best]:.3f}" if eps.get(best) else '-'
        print(f"{os.path.basename(d):<24} {curve[best]:>8.4f} {best:>6} "
              f"{eps_s:>8} {au_s:>8} {curve[max(curve)]:>8.4f}")


if __name__ == '__main__':
    main()
