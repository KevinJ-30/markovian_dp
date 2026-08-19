"""
Post-hoc privacy accounting for SparseGNN DP sweeps.

Reads the results CSV written by `src.sparse.run --dp` (which records the
mechanism parameters direction, p1, p2, r, sigma, T, K_in, K_out per row),
computes epsilon for every distinct configuration, and writes an augmented CSV.
Utility is measured first; epsilon is attached after the fact — accounting never
touches training.

Which theorem applies is set by the expansion orientation recorded in the CSV:

  direction='in'   Theorem 6.4 (node substitution).  The corrected Algorithm 5
                   orientation; this is the headline number.
  direction='out'  Theorem 4.5 (node insertion/removal).  Only stated for the
                   legacy Algorithm 4, so it is the out-orientation ablation.

Columns written:

  epsilon               the guarantee for this row, per the rule above
  epsilon_theorem       which theorem produced it
  epsilon_substitution  the substitution pair for this row's direction, computed
                        for EVERY row so that in- and out-expansion can be
                        compared under the same adjacency notion
  epsilon_naive_opacus  Opacus PRV epsilon for a Poisson-subsampled Gaussian at
                        sample rate p1.  This is what accounting would claim if
                        a node only influenced its own subgraph — it ignores
                        that a node appears in neighbours' expansions, so it is
                        NOT a valid node-level guarantee; it is a floor showing
                        the price of graph structure.

CSVs written before the orientation fix have no `direction` column; they are
read as direction='out'.

Usage:
  python -m src.sparse.compute_epsilon --csv results/sparse_gnn_citeseer_dp_results.csv
  python -m src.sparse.compute_epsilon --csv ... --delta 1e-6 --grid 1e-3
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sparse.accounting import (                               # noqa: E402
    naive_opacus_epsilon, sparsegnn_substitution_epsilon_schedule,
    sparsegnn_thm4_epsilon_schedule,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True,
                   help='results CSV from `python -m src.sparse.run --dp`')
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--theorem', choices=['auto', 'substitution', 'thm45'],
                   default='auto',
                   help="which dominating pair drives the `epsilon` column; "
                        "'auto' picks Theorem 6.4 for direction='in' and "
                        "Theorem 4.5 for direction='out'")
    p.add_argument('--grid', type=float, default=1e-4,
                   help="dp_accounting value_discretization_interval "
                        "(pessimistic rounding; smaller = tighter but slower)")
    p.add_argument('--out', default=None,
                   help='output CSV (default: <input>_with_eps.csv)')
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or args.csv.replace('.csv', '_with_eps.csv')

    with open(args.csv, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    if rows[0].get('dp', 'False') != 'True':
        print("note: CSV rows have dp=False — epsilon is only meaningful for "
              "DP runs; computing anyway from the recorded parameters.")

    # One accounting pass per distinct mechanism configuration.  Rows written
    # with --track_every carry a `step` column; all checkpoints of a config
    # share one incremental composition schedule, so eps(t) for 40 checkpoints
    # costs about as much as a single full-length composition.
    def _row_key(row):
        # Pre-orientation-fix CSVs have no `direction`; they used Algorithm 4.
        direction = row.get('direction') or 'out'
        return (direction, float(row['p1']), float(row['p2']),
                int(row['r']), float(row['sigma']), int(row['T']),
                int(row['K_in']), int(row['K_out']))

    def _row_step(row):
        # Legacy CSVs have no `step` column: the row is the final iterate.
        s = row.get('step')
        return int(float(s)) if s not in (None, '') else int(row['T'])

    for row in rows:
        if not row.get('K_in'):
            raise SystemExit(
                "CSV has no K_in — rerun with --K_in (degree capping) or use a "
                "CSV produced by the updated src.sparse.run, which records the "
                "graph's max degrees.")

    steps_by_key = {}
    for row in rows:
        steps_by_key.setdefault(_row_key(row), set()).add(_row_step(row))

    eps_cache = {}     # (key, step) -> eps of the row's applicable theorem
    sub_cache = {}     # (key, step) -> substitution eps (always computed)
    naive_cache = {}   # (key, step) -> naive opacus eps
    thm_cache = {}     # key -> theorem label
    for key, steps in steps_by_key.items():
        direction, p1, p2, r, sigma, T, K_in, K_out = key
        # Always computed: comparable across orientations.
        sub_sched = sparsegnn_substitution_epsilon_schedule(
            p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, sigma=sigma,
            steps=steps, delta=args.delta, direction=direction,
            grid=args.grid)

        theorem = args.theorem
        if theorem == 'auto':
            theorem = 'thm45' if direction == 'out' else 'substitution'
        if theorem == 'thm45':
            if direction != 'out':
                raise SystemExit(
                    "Theorem 4.5 is stated for out-expansion (Algorithm 4) "
                    f"only, but this CSV has direction={direction!r}. Use "
                    "--theorem substitution.")
            eps_sched = sparsegnn_thm4_epsilon_schedule(
                p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, sigma=sigma,
                steps=steps, delta=args.delta, grid=args.grid)
            thm_cache[key] = 'thm4.5-insertion-removal'
        else:
            eps_sched = sub_sched
            thm_cache[key] = ('thm6.4-substitution' if direction == 'in'
                              else 'thm1.2-substitution')

        for t in steps:
            sub_cache[(key, t)] = sub_sched[t]
            eps_cache[(key, t)] = eps_sched[t]
            try:
                naive_cache[(key, t)] = naive_opacus_epsilon(
                    sigma, p1, t, args.delta, mechanism='prv')
            except Exception:
                naive_cache[(key, t)] = float('nan')

        t_max = max(steps)
        print(f"dir={direction} p1={p1} p2={p2} r={r} sigma={sigma} T={T} "
              f"K_in={K_in} K_out={K_out}"
              + (f" [{len(steps)} checkpoints]" if len(steps) > 1 else "")
              + f":  eps={eps_cache[(key, t_max)]:.4f} ({thm_cache[key]})  "
                f"eps_sub={sub_cache[(key, t_max)]:.4f}  "
                f"eps_naive={naive_cache[(key, t_max)]:.4f}")

    for row in rows:
        key, t = _row_key(row), _row_step(row)
        row['step'] = t
        row['epsilon'] = f"{eps_cache[(key, t)]:.5f}"
        row['epsilon_theorem'] = thm_cache[key]
        row['epsilon_substitution'] = f"{sub_cache[(key, t)]:.5f}"
        row['epsilon_naive_opacus'] = f"{naive_cache[(key, t)]:.5f}"
        row['delta'] = f"{args.delta:g}"

    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\naugmented results written to {out_path}")

    # Compact frontier table: mean test acc per config vs epsilon, at each
    # config's FINAL checkpoint (tracked intermediate rows live in the CSV).
    from collections import defaultdict
    acc = defaultdict(list)
    for row in rows:
        key = _row_key(row)
        if _row_step(row) == max(steps_by_key[key]):
            acc[key].append(float(row['test_acc']))
    print(f"\n{'dir':>4} {'p1':>6} {'p2':>5} {'r':>3} {'sigma':>6} {'step':>6} "
          f"{'test_acc':>9} {'epsilon':>10} {'eps_subst':>10} {'eps_naive':>10}")
    print('-' * 77)
    for key in sorted(acc, key=lambda k: eps_cache[(k, max(steps_by_key[k]))]):
        direction, p1, p2, r, sigma, T, _, _ = key
        t_max = max(steps_by_key[key])
        m = sum(acc[key]) / len(acc[key])
        print(f"{direction:>4} {p1:>6} {p2:>5} {r:>3} {sigma:>6} {t_max:>6} "
              f"{m:>9.4f} {eps_cache[(key, t_max)]:>10.3f} "
              f"{sub_cache[(key, t_max)]:>10.3f} "
              f"{naive_cache[(key, t_max)]:>10.3f}")


if __name__ == '__main__':
    main()
