"""
Post-hoc privacy accounting for SparseGNN DP sweeps (Theorem 4).

Reads the results CSV written by `src.sparse.run --dp` (which records the
mechanism parameters p1, p2, r, sigma, T, K_in, K_out per row), computes the
Theorem 4 insertion/removal epsilon for every distinct configuration, and
writes an augmented CSV with an `epsilon_thm4` column.  Utility is measured
first; epsilon is attached after the fact — accounting never touches training.

For reference, an `epsilon_naive_opacus` column is also emitted: the standard
Opacus PRV epsilon for a Poisson-subsampled Gaussian with sample rate p1.
This is what accounting would claim if a node only influenced its own
subgraph — it ignores that a node appears in neighbors' expansions, so it is
NOT a valid node-level guarantee; it is a floor showing the price of graph
structure.

Usage:
  python -m src.sparse.compute_epsilon --csv results/sparse_gnn_citeseer_dp_results.csv
  python -m src.sparse.compute_epsilon --csv ... --delta 1e-5 --grid 1e-3
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sparse.accounting import sparsegnn_thm4_epsilon, _load_pld_module  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True,
                   help='results CSV from `python -m src.sparse.run --dp`')
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--grid', type=float, default=1e-3,
                   help='PLD loss discretization (pessimistic rounding; '
                        'slack <= grid * T)')
    p.add_argument('--loss_cap', type=float, default=100.0,
                   help='per-step privacy-loss cap; mass beyond it is treated '
                        'as infinite loss (pessimistic), so eps=inf means '
                        '"beyond the cap", not a failure')
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

    # One accounting call per distinct mechanism configuration.
    eps_cache = {}
    naive_cache = {}
    for row in rows:
        if not row.get('K_in'):
            raise SystemExit(
                "CSV has no K_in — rerun with --K_in (degree capping) or use a "
                "CSV produced by the updated src.sparse.run, which records the "
                "graph's max degrees.")
        key = (float(row['p1']), float(row['p2']), int(row['r']),
               float(row['sigma']), int(row['T']),
               int(row['K_in']), int(row['K_out']))
        if key not in eps_cache:
            p1, p2, r, sigma, T, K_in, K_out = key
            eps_cache[key] = sparsegnn_thm4_epsilon(
                p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, sigma=sigma,
                steps=T, delta=args.delta, grid=args.grid,
                loss_cap=args.loss_cap,
            )
            try:
                naive_cache[key] = _load_pld_module().opacus_epsilon(
                    sigma, p1, T, args.delta, mechanism='prv')
            except Exception:
                naive_cache[key] = float('nan')
            print(f"p1={p1} p2={p2} r={r} sigma={sigma} T={T} "
                  f"K_in={K_in} K_out={K_out}:  eps_thm4={eps_cache[key]:.4f}  "
                  f"eps_naive={naive_cache[key]:.4f}")
        row['epsilon_thm4'] = f"{eps_cache[key]:.5f}"
        row['epsilon_naive_opacus'] = f"{naive_cache[key]:.5f}"
        row['delta'] = f"{args.delta:g}"

    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\naugmented results written to {out_path}")

    # Compact frontier table: mean test acc per config vs epsilon.
    from collections import defaultdict
    acc = defaultdict(list)
    for row in rows:
        key = (float(row['p1']), float(row['p2']), int(row['r']),
               float(row['sigma']), int(row['T']),
               int(row['K_in']), int(row['K_out']))
        acc[key].append(float(row['test_acc']))
    print(f"\n{'p1':>5} {'p2':>5} {'r':>3} {'sigma':>6} "
          f"{'test_acc':>9} {'eps_thm4':>10} {'eps_naive':>10}")
    print('-' * 56)
    for key in sorted(acc, key=lambda k: eps_cache[k]):
        p1, p2, r, sigma, T, _, _ = key
        m = sum(acc[key]) / len(acc[key])
        print(f"{p1:>5} {p2:>5} {r:>3} {sigma:>6} {m:>9.4f} "
              f"{eps_cache[key]:>10.3f} {naive_cache[key]:>10.3f}")


if __name__ == '__main__':
    main()
