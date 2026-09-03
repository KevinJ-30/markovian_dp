"""Invert the accountant: given a target (epsilon, delta), find sigma.

`src.sparse.accounting` runs forward — pick sigma, get epsilon.  Choosing the
noise to hit a privacy budget is the other direction, and it is what a
target-epsilon sweep needs: every cell of a frontier plot must sit at the SAME
epsilon, otherwise a utility difference between two cells cannot be attributed
to the parameter under study.

epsilon is strictly decreasing in sigma, so a geometric bisection converges.
The returned sigma is the SMALLEST noise (i.e. the best utility) whose epsilon
is still <= the target, so the guarantee is never violated by rounding.

    from src.sparse.calibrate import sigma_for_epsilon
    sigma = sigma_for_epsilon(target_epsilon=1.0, delta=1.57e-5, p1=0.0114,
                              p2=1.0, r=1, K_in=5, K_out=5, steps=2000)

    python -m src.sparse.calibrate --eps 1 2 4 8 --delta 1.57e-5 \
        --p1 0.0114 --p2 1.0 0.1 --r 1 --K_in 5 --K_out 5 --T 2000


THE DISCRETIZATION FLOOR
------------------------
dp_accounting composes a DISCRETIZED privacy loss distribution, and with
`pessimistic_estimate=True` every cell's loss is rounded UP to the next multiple
of `grid`.  That rounding accumulates over composition, so a T-step run carries
roughly `T * grid` of purely numerical epsilon on top of the real thing.

Measured on PPI (r=0, p1=0.0114, T=2000), the reported epsilon as sigma grows:

    sigma      grid=1e-3    grid=1e-4    grid=1e-5
       10        1.72966      0.82891      0.73902
      100        1.07098      0.15835      0.06832
     1000        1.06407      0.10534      0.01428

At grid=1e-3 epsilon plateaus near 1.06 no matter how much noise is added --
about T*grid = 2.0 of floor.  Any calibration against a target below that floor
is meaningless: the bisection would run sigma to infinity chasing an epsilon the
discretization cannot represent.

So `grid` is chosen from the target by default (`_auto_grid`), keeping the floor
two orders of magnitude under the target, and `verify=True` re-prices the answer
on a finer grid and warns if the two disagree.  Both can be overridden.
"""

import argparse
import math
import sys
import warnings
from typing import Optional

from src.sparse.accounting import (
    sparsegnn_substitution_epsilon, sparsegnn_thm4_epsilon,
)

# Bracket for the bisection.  sigma below _SIGMA_LO is not a mechanism anyone
# would train with; above _SIGMA_HI the gradient signal is long gone.
_SIGMA_LO = 0.05
_SIGMA_HI = 1e4

# Keep the discretization floor (~steps * grid) this far under the target.
_FLOOR_SAFETY = 100.0
# _GRID_MIN is a MEMORY limit, not an accuracy one.  dp_accounting holds one
# bucket per `grid` of privacy loss and composition widens that range, so cost
# grows as steps/grid: at T=2000, grid=1e-5 composes in ~40s, while grid=1e-6
# exhausted 32GB and was OOM-killed.  Do not lower this without re-measuring.
_GRID_MIN, _GRID_MAX = 1e-5, 1e-4


def _auto_grid(target_epsilon: float, steps: int) -> float:
    """Discretization interval whose accumulated rounding is negligible here.

    The floor is about `steps * grid`, so solving
    `steps * grid = target_epsilon / _FLOOR_SAFETY` keeps it two orders of
    magnitude below the target.  Clamped: below _GRID_MIN the FFT gets
    needlessly large, above _GRID_MAX the floor is never negligible at the step
    counts this project uses.
    """
    g = target_epsilon / (_FLOOR_SAFETY * max(1, steps))
    return min(_GRID_MAX, max(_GRID_MIN, g))


def _epsilon_fn(theorem, p1, p2, r, K_in, K_out, steps, delta, direction):
    """Bind everything except sigma and grid, and pick the dominating pair.

    'auto' mirrors `src.sparse.compute_epsilon`: Theorem 4.5 for out-expansion,
    the substitution pair for in-expansion.
    """
    if theorem == 'auto':
        theorem = 'thm45' if direction == 'out' else 'substitution'
    if theorem == 'thm45':
        if direction != 'out':
            raise ValueError(
                "Theorem 4.5 is stated for out-expansion only, but "
                f"direction={direction!r}; use theorem='substitution'")

        def f(sigma, grid):
            return sparsegnn_thm4_epsilon(
                p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, sigma=sigma,
                steps=steps, delta=delta, grid=grid)
    else:
        def f(sigma, grid):
            return sparsegnn_substitution_epsilon(
                p1=p1, p2=p2, r=r, K_in=K_in, K_out=K_out, sigma=sigma,
                steps=steps, delta=delta, direction=direction, grid=grid)
    return f, theorem


def sigma_for_epsilon(
    target_epsilon: float,
    delta: float,
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    steps: int,
    K_out: Optional[int] = None,
    direction: str = 'in',
    theorem: str = 'auto',
    grid: Optional[float] = None,
    rtol: float = 0.005,
    max_iter: int = 60,
    verify: bool = True,
    verbose: bool = False,
) -> float:
    """Smallest sigma whose epsilon is <= `target_epsilon` at `delta`.

    Args:
        target_epsilon: the budget this cell must not exceed.
        grid: dp_accounting discretization.  None picks one from the target via
            `_auto_grid`; pass a value to pin it (comparisons across cells are
            only fair at a common grid, but see the note below).
        rtol: bisection stops once the bracket is this tight in relative sigma.
        verify: re-price the returned sigma and warn if it exceeds the target.
        verbose: print every epsilon probe, so a long calibration shows progress
            instead of looking hung.

    Returns:
        sigma > 0 with epsilon(sigma) <= target_epsilon.

    Raises:
        ValueError: if the target is unreachable at any sigma in the bracket --
            with p1, r, K and T fixed there is a floor on epsilon that noise
            cannot get under, and the message reports it.

    Note on a common grid: `_auto_grid` scales with the target, so cells at
    different targets get different grids.  That is intentional -- each answer
    is then accurate in its own right.  Pinning one grid across a wide range of
    targets would either be too coarse for the small ones or needlessly slow for
    the large ones.
    """
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must lie in (0, 1), got {delta}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    g = _auto_grid(target_epsilon, steps) if grid is None else grid
    eps_at, _ = _epsilon_fn(theorem, p1, p2, r, K_in, K_out, steps, delta,
                            direction)

    # Phase 1 -- locate sigma on a COARSE grid.  Rounding is pessimistic, so a
    # coarse grid over-states epsilon and the sigma it returns is therefore an
    # over-estimate: the true (fine-grid) answer is never larger.
    coarse = min(_GRID_MAX, max(g * 50.0, 1e-3))
    lo_c, hi_c = _bracket(eps_at, target_epsilon, coarse, verbose)
    sigma_c = _bisect(eps_at, target_epsilon, coarse, lo_c, hi_c, rtol,
                      max_iter, verbose)

    if coarse <= g * 1.01:
        return sigma_c            # already at the target resolution

    # Phase 2 -- walk down from that over-estimate at the target grid, which
    # only ever prices moderate sigma and so stays cheap.
    hi_f, lo_f = sigma_c, None
    for _ in range(max_iter):
        cand = hi_f / 1.5
        if cand < _SIGMA_LO:
            lo_f = _SIGMA_LO
            break
        e = eps_at(cand, g)
        if verbose:
            print(f"    refine:  sigma={cand:<10.4g} eps={e:.4f}", flush=True)
        if e <= target_epsilon:
            hi_f = cand
        else:
            lo_f = cand
            break
    if lo_f is None:
        return hi_f
    sigma = _bisect(eps_at, target_epsilon, g, lo_f, hi_f, rtol, max_iter,
                    verbose)

    if verify:
        e = eps_at(sigma, g)
        if e > target_epsilon * 1.001:
            warnings.warn(
                f"calibration overshot: sigma={sigma:.4g} gives "
                f"epsilon={e:.5f} > target {target_epsilon:g}",
                RuntimeWarning)
    return sigma


def _bisect(eps_at, target, grid, lo, hi, rtol, max_iter, verbose=False):
    """Geometric bisection on sigma; assumes eps(lo) > target >= eps(hi)."""
    for _ in range(max_iter):
        if hi / lo < 1.0 + rtol:
            break
        mid = math.sqrt(lo * hi)       # sigma spans decades, so bisect in log
        e = eps_at(mid, grid)
        if verbose:
            print(f"    bisect:  sigma={mid:<10.4g} eps={e:.4f}", flush=True)
        if e > target:
            lo = mid
        else:
            hi = mid
    return hi


def _bracket(eps_at, target, grid, verbose):
    """Coarse geometric ladder -> (lo, hi) with eps(lo) > target >= eps(hi).

    Small sigma is where a PLD is most expensive: dp_accounting holds one bucket
    per `grid` of privacy loss, and the loss of a sensitivity-2 pair grows like
    1/sigma, so probing sigma=0.05 at grid=1e-5 asks for ~1e8 buckets before
    composition even starts.  Bracketing therefore walks UP from a usable sigma
    on a coarse grid, and only the refinement runs at the target grid.
    """
    prev = None
    s = _SIGMA_LO
    while s <= _SIGMA_HI:
        e = eps_at(s, grid)
        if verbose:
            print(f"    bracket: sigma={s:<10.4g} eps={e:.4f}", flush=True)
        if e <= target:
            return (prev if prev is not None else _SIGMA_LO), s
        prev, s = s, s * 4.0
    raise ValueError(
        f"epsilon={target:g} is unreachable: sigma={_SIGMA_HI:g} still gives "
        f"epsilon={eps_at(_SIGMA_HI, grid):.4f}.  Lower p1/p2/r/K or shorten T.")

    if verify:
        fine = max(_GRID_MIN, g / 2.0)
        if fine < g * 0.9:
            e_coarse, e_fine = eps_at(hi, g), eps_at(hi, fine)
            if e_coarse > 0 and abs(e_coarse - e_fine) / e_coarse > 0.05:
                warnings.warn(
                    f"discretization still binding at sigma={hi:.4g}: "
                    f"epsilon={e_coarse:.4f} at grid={g:g} but {e_fine:.4f} at "
                    f"grid={fine:g}.  The returned sigma is conservative (too "
                    f"much noise); pass a smaller grid= for a tighter answer.",
                    RuntimeWarning)
    return hi


def calibration_table(targets, delta, p1, p2_values, r, K_in, steps,
                      K_out=None, direction='in', theorem='auto', grid=None):
    """{(p2, target_epsilon): sigma} for one (p1, r, K, T) cell."""
    out = {}
    for p2 in p2_values:
        for t in targets:
            try:
                out[(p2, t)] = sigma_for_epsilon(
                    target_epsilon=t, delta=delta, p1=p1, p2=p2, r=r,
                    K_in=K_in, K_out=K_out, steps=steps, direction=direction,
                    theorem=theorem, grid=grid)
            except ValueError:
                out[(p2, t)] = None
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eps', type=float, nargs='+', required=True,
                   help='target epsilon value(s)')
    p.add_argument('--delta', type=float, default=None,
                   help='target delta; or use --delta_from_n')
    p.add_argument('--delta_from_n', type=int, default=None,
                   help='set delta = n^-1.01 for this node count')
    p.add_argument('--p1', type=float, required=True)
    p.add_argument('--p2', type=float, nargs='+', default=[1.0])
    p.add_argument('--r', type=int, required=True)
    p.add_argument('--K_in', type=int, required=True)
    p.add_argument('--K_out', type=int, default=None)
    p.add_argument('--T', type=int, required=True)
    p.add_argument('--direction', choices=['in', 'out'], default='in')
    p.add_argument('--theorem', choices=['auto', 'substitution', 'thm45'],
                   default='auto')
    p.add_argument('--grid', type=float, default=None,
                   help='pin the discretization (default: from the target)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.delta is None and args.delta_from_n is None:
        raise SystemExit("pass --delta or --delta_from_n")
    delta = (args.delta if args.delta is not None
             else float(args.delta_from_n) ** -1.01)
    K_out = args.K_out if args.K_out is not None else args.K_in

    print(f"p1={args.p1}  r={args.r}  K_in={args.K_in}  K_out={K_out}  "
          f"T={args.T}  delta={delta:.4g}  direction={args.direction}")
    print(f"\n{'eps':>8}" + ''.join(f"{'p2=' + str(v):>12}"
                                    for v in args.p2))
    print('-' * (8 + 12 * len(args.p2)))
    for t in args.eps:
        row = f"{t:>8g}"
        for p2 in args.p2:
            try:
                s = sigma_for_epsilon(
                    target_epsilon=t, delta=delta, p1=args.p1, p2=p2, r=args.r,
                    K_in=args.K_in, K_out=K_out, steps=args.T,
                    direction=args.direction, theorem=args.theorem,
                    grid=args.grid)
                row += f"{s:>12.3f}"
            except ValueError:
                row += f"{'unreachable':>12}"
        print(row)
    print(f"\ngrid: {'pinned ' + repr(args.grid) if args.grid else 'auto'} "
          f"(floor ~ T*grid must stay well under each target)")


if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('/src/', 1)[0])
    main()
