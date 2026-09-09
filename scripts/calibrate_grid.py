"""Emit the sigma each (p2, target-epsilon) cell needs, as shell-parseable rows.

A target-epsilon sweep needs every cell to sit at the SAME epsilon, otherwise a
utility difference between two cells cannot be attributed to the parameter under
study.  sigma is therefore not swept -- it is solved for, per cell, by inverting
the accountant (`calibrate_sparsegnn_noise`).

    python scripts/calibrate_grid.py --eps 1 2 4 8 --p2 1.0 0.5 0.25 0.1 \
        --p1 0.0114 --r 1 --K 5 --T 2000 --delta_from_n 56944

Prints one `p2 eps sigma` row per cell (plus `# ...` comments), so a driver can

    while read P2 EPS SIGMA; do ... done < <(python scripts/calibrate_grid.py ...)

Cells whose target is unreachable at any sigma print `SKIP` and are the caller's
problem: with p1, r, K and T fixed there is a floor on epsilon that noise cannot
get under, and that fact is itself a result worth recording.

delta: `--delta_from_n N` sets delta = N^-1.01 with N the FULL node count, the
convention agreed for this suite (negligible vs n, and stricter than the 1e-5 /
1e-6 constants the older scripts hardcoded).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.sparse.accounting import calibrate_sparsegnn_noise   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eps', type=float, nargs='+', required=True)
    p.add_argument('--p2', type=float, nargs='+', default=[1.0])
    p.add_argument('--p1', type=float, required=True)
    p.add_argument('--r', type=int, required=True)
    p.add_argument('--K', type=int, required=True,
                   help='K_in = K_out; keeping them equal holds cap_mode=auto '
                        'on the symmetric/undirected path')
    p.add_argument('--T', type=int, required=True)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--direction', choices=['in', 'out'], default='in')
    p.add_argument('--theorem', choices=['auto', 'substitution', 'thm45'],
                   default='auto')
    p.add_argument('--grid', type=float, default=1e-4,
                   help='dp_accounting discretization.  Pessimistic rounding '
                        'accumulates over composition, so the numerical floor '
                        'is about T*grid -- keep it well under the smallest '
                        'target or the answer is discretization, not privacy.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--delta', type=float)
    g.add_argument('--delta_from_n', type=int,
                   help='delta = n^-1.01 for this (full) node count')
    return p.parse_args()


def main():
    a = parse_args()
    delta = a.delta if a.delta is not None else float(a.delta_from_n) ** -1.01

    floor = a.T * a.grid
    print(f"# p1={a.p1} r={a.r} K={a.K} T={a.T} delta={delta:.4g} "
          f"direction={a.direction} grid={a.grid:g}")
    if floor > min(a.eps) / 10.0:
        print(f"# WARNING discretization floor ~T*grid={floor:.4g} is not "
              f"negligible vs the smallest target {min(a.eps):g}; "
              f"lower --grid", file=sys.stderr)

    for p2 in a.p2:
        for eps in a.eps:
            t0 = time.time()
            try:
                c = calibrate_sparsegnn_noise(
                    target_epsilon=eps, target_delta=delta, p1=a.p1, p2=p2,
                    r=a.r, K_in=a.K, K_out=a.K, steps=a.T, clip=a.clip,
                    direction=a.direction, theorem=a.theorem, grid=a.grid)
            except (RuntimeError, ValueError) as exc:
                print(f"# SKIP p2={p2} eps={eps}: {exc}", file=sys.stderr)
                print(f"{p2} {eps} SKIP", flush=True)
                continue
            print(f"# p2={p2} eps={eps}: sigma={c.noise_multiplier:.4f} "
                  f"var={c.noise_variance:.4f} achieved={c.epsilon:.5f} "
                  f"({c.evaluations} evals, {time.time() - t0:.1f}s) "
                  f"{c.theorem}", file=sys.stderr)
            print(f"{p2} {eps} {c.noise_multiplier:.6f}", flush=True)


if __name__ == '__main__':
    main()
