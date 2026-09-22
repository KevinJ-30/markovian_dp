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
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.privacy.accounting import calibrate_sparsegnn_noise   # noqa: E402


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
    p.add_argument('--legacy_shells', action='store_true',
                   help='drop the union-graph correction (n_d = K^d not 2*K^d)')
    p.add_argument('--grid', type=float, default=1e-3,
                   help='dp_accounting discretization.  Pessimistic rounding '
                        'accumulates over composition, so the numerical floor '
                        'is about T*grid -- keep it well under the smallest '
                        'target or the answer is discretization, not privacy.')
    p.add_argument('--cache_dir', default=os.environ.get(
                       'SIGMA_CACHE', 'results/_sigma_cache'),
                   help='sigma cache; calibration is a deterministic function '
                        'of the cell, so a hit is exact, not an approximation')
    p.add_argument('--no_cache', action='store_true')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--delta', type=float)
    g.add_argument('--delta_from_n', type=int,
                   help='delta = n^-1.01 for this (full) node count')
    return p.parse_args()


def _cache_key(**cell) -> str:
    """Stable digest of every argument sigma depends on.

    Anything omitted here is a silent correctness bug: two different cells
    would collide and the second would read the first's sigma.
    """
    blob = json.dumps(cell, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _cache_write(path, payload):
    """Atomic, so concurrent sbatch jobs cannot read a half-written file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        tmp.replace(path)
    except OSError as exc:                       # a cache miss must never fail a run
        print(f"# cache write failed ({exc})", file=sys.stderr)


def main():
    a = parse_args()
    delta = a.delta if a.delta is not None else float(a.delta_from_n) ** -1.01

    floor = a.T * a.grid
    print(f"# p1={a.p1} r={a.r} K={a.K} T={a.T} delta={delta:.4g} "
          f"grid={a.grid:g}")
    if floor > min(a.eps) / 10.0:
        print(f"# WARNING discretization floor ~T*grid={floor:.4g} is not "
              f"negligible vs the smallest target {min(a.eps):g}; "
              f"lower --grid", file=sys.stderr)

    cache_dir = None if a.no_cache else Path(a.cache_dir)

    for p2 in a.p2:
        for eps in a.eps:
            cell = dict(p1=a.p1, p2=p2, r=a.r, K=a.K, T=a.T, clip=a.clip,
                        eps=eps, delta=delta, grid=a.grid,
                        union_safe=not a.legacy_shells)
            entry = cache_dir / f"sigma_{_cache_key(**cell)}.json" if cache_dir else None

            if entry is not None:
                hit = _cache_read(entry)
                if hit is not None and hit.get('cell') == cell:
                    sigma = hit['sigma']
                    print(f"# p2={p2} eps={eps}: {sigma} (cached {entry.name})",
                          file=sys.stderr)
                    print(f"{p2} {eps} {sigma}", flush=True)
                    continue

            t0 = time.time()
            try:
                c = calibrate_sparsegnn_noise(
                    target_epsilon=eps, target_delta=delta, p1=a.p1, p2=p2,
                    r=a.r, K_in=a.K, K_out=a.K, steps=a.T, clip=a.clip,
                    grid=a.grid, union_safe=not a.legacy_shells)
            except (RuntimeError, ValueError) as exc:
                print(f"# SKIP p2={p2} eps={eps}: {exc}", file=sys.stderr)
                # Cache the SKIP too: an unreachable cell costs the same search
                # to rediscover as a reachable one.
                if entry is not None:
                    _cache_write(entry, dict(cell=cell, sigma='SKIP', reason=str(exc)))
                print(f"{p2} {eps} SKIP", flush=True)
                continue
            if entry is not None:
                _cache_write(entry, dict(cell=cell,
                                         sigma=f"{c.noise_multiplier:.6f}",
                                         achieved_epsilon=c.epsilon,
                                         seconds=round(time.time() - t0, 1)))
            print(f"# p2={p2} eps={eps}: sigma={c.noise_multiplier:.4f} "
                  f"noise_std={c.noise_std:.4f} "
                  f"var={c.noise_variance:.4f} achieved={c.epsilon:.5f} "
                  f"({c.evaluations} evals, {time.time() - t0:.1f}s)",
                  file=sys.stderr)
            print(f"{p2} {eps} {c.noise_multiplier:.6f}", flush=True)


if __name__ == '__main__':
    main()
