"""
Summarize a sweep directory: one row per cell, at its best tracked checkpoint.

    python scripts/summarize_sweep.py results/ppi/sweep_lr

Reports the best checkpoint rather than the last, because DP runs frequently
peak mid-training and then decay under the noise.  When a cell has per-checkpoint
epsilon (from compute_epsilon on a --track_every CSV), the epsilon at that same
checkpoint is shown, so cells are read at matched privacy rather than matched
step count.
"""

import argparse
import csv
import glob
import os


def _curve(path, key):
    rows = list(csv.DictReader(open(path)))
    if not rows or key not in rows[0]:
        return None, None
    by, eps = {}, {}
    for r in rows:
        step = int(float(r.get('step') or r['T']))
        try:
            by.setdefault(step, []).append(float(r[key]))
        except ValueError:                       # blank metric column
            continue
        if r.get('epsilon'):
            eps[step] = float(r['epsilon'])
    if not by:
        return None, None
    return {t: sum(v) / len(v) for t, v in by.items()}, eps


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
        curve, eps = _curve(csvs[0], args.metric)
        if not curve:
            continue
        best = max(curve, key=curve.get)
        au, _ = _curve(csvs[0], 'test_auroc')
        au_s = f"{au[best]:.4f}" if au and best in au else '-'
        eps_s = f"{eps[best]:.3f}" if eps.get(best) else '-'
        print(f"{os.path.basename(d):<24} {curve[best]:>8.4f} {best:>6} "
              f"{eps_s:>8} {au_s:>8} {curve[max(curve)]:>8.4f}")


if __name__ == '__main__':
    main()
