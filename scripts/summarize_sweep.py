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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_dir')
    ap.add_argument('--metric', default='test_acc',
                    help='score column to report (higher is better); test_acc '
                         'holds the primary metric, including R² for regression')
    args = ap.parse_args()

    cells = sorted(d for d in glob.glob(os.path.join(args.sweep_dir, '*'))
                   if os.path.isdir(d))
    if not cells:
        raise SystemExit(f"no cell directories under {args.sweep_dir}")

    print(f"{'cell':<24} {'metric':>10} {'best':>8} {'step':>6} {'eps':>8} {'auroc':>8} {'final':>8}")
    print('-' * 77)
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

        def _pick(c, label):
            """argmax over c, skipping NaN.

            max() seeds with the first element and every comparison against
            NaN is False, so a single leading NaN silently returns the EARLIEST
            step as "best".  NaN is reachable: AUROC on a single-class split,
            R² on fewer than two observations, any metric on an empty mask.
            """
            finite = {s: v for s, v in c.items() if v == v}   # v == v is False for NaN
            dropped = len(c) - len(finite)
            if dropped:
                print(f"  WARNING: {d}: {dropped}/{len(c)} {label} checkpoints "
                      f"are NaN and were skipped", file=sys.stderr)
            if not finite:
                print(f"  WARNING: {d}: all {label} checkpoints are NaN; "
                      f"skipping cell", file=sys.stderr)
                return None
            return max(finite, key=lambda s: finite[s])

        if val_curve:
            best = _pick(val_curve, val_key)
            if best is None or best not in curve:
                continue
        else:
            print(f"  WARNING: no {val_key or 'val'} column for {d}; "
                  f"selecting on test itself (leakage)", file=sys.stderr)
            best = _pick(curve, args.metric)
            if best is None:
                continue
        au, _ = _curve(rows, 'test_auroc')
        au_s = f"{au[best]:.4f}" if au and best in au else '-'
        # `if eps.get(best)` would print '-' for a genuine epsilon of 0.0.
        eps_s = f"{eps[best]:.3f}" if eps.get(best) is not None else '-'
        metric = (rows[0].get('metric', 'accuracy') if args.metric == 'test_acc'
                  else args.metric.removeprefix('test_'))
        print(f"{os.path.basename(d):<24} {metric:>10} {curve[best]:>8.4f} {best:>6} "
              f"{eps_s:>8} {au_s:>8} {curve[max(curve)]:>8.4f}")


if __name__ == '__main__':
    main()
