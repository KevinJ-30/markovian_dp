"""
Dominating pairs for one SparseGNN step (manuscript v36).

Substitution (Theorem 6.4 for in-expansion, Theorem 1/2 for out-expansion):

    K   = min(K_in, K_out)
    q_0 = 1,   q_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}}      (Eq. 43)
    n_0 = 1,   n_d = K_out^d ('in') or K_in^d ('out')            (Eq. 44)
    sum_k pi_k z^k = prod_{d=0..r} (1 - p1 q_d + p1 q_d z)^{n_d} (Eq. 46)
    P = sum_k pi_k N(-2k, sigma^2),  Q = sum_k pi_k N(+2k, sigma^2)  (Eq. 47)

Insertion/removal (Theorem 4.5, out-expansion only), with a_d = p1 * q_d and
n_d = K_in^d:

    sum_j pi_j z^j = prod_{d=1..r} (1 - a_d + a_d z)^{n_d}       (Eq. 29)
    P_ins = (+)_j pi_j N(-j, sigma^2)
    Q_ins = (+)_j pi_j [ (1-p1) N(j, sigma^2) + p1 N(j+1, sigma^2) ]

Composition and epsilon(delta) are delegated to Google's
`dp_accounting.pld.privacy_loss_distribution`.  The pair handed to it is built
to dominate the analytic pair, so the reported epsilon is an upper bound: cell
masses are exact CDF differences, each cell takes the larger of its two edge
losses (valid because monotonicity is asserted at construction), the
denominator mass is num_mass * exp(-loss) <= the true mass, and all trimmed
mass becomes an infinite-loss outcome.
"""

from dataclasses import asdict, dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Mass below these floors is dropped to the infinite-loss outcome, which is
# pessimistic — the choice of floor cannot make epsilon an underestimate.
_COMPONENT_MASS_FLOOR = 1e-14
_CELL_MASS_FLOOR = 1e-18


def _q_products(p2: float, r: int, K: int) -> List[float]:
    """q_0..q_r from Eq. (3) / (43): the multiplicative path-retention bounds.

    q_0 = 1 and q_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}} for d >= 1.  The
    product is taken in log space so that large K^{l-1} exponents stay stable.
    """
    q = [1.0]
    for d in range(1, r + 1):
        if p2 >= 1.0:
            q.append(1.0)
            continue
        log_keep = sum((K ** (l - 1)) * math.log1p(-(p2 ** l))
                       for l in range(d, r + 1))
        q.append(1.0 - math.exp(log_keep))
    return q


def _binom_pmf(n: int, p: float) -> np.ndarray:
    """pmf of Binomial(n, p) as a length-(n+1) array; exact at p in {0, 1}."""
    if p <= 0.0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1.0:
        out = np.zeros(n + 1)
        out[n] = 1.0
        return out
    from scipy.stats import binom
    pmf = binom.pmf(np.arange(n + 1), n, p)
    pmf = np.clip(pmf, 0.0, None)
    return pmf / pmf.sum()


def shell_sizes(r: int, K_in: int, K_out: Optional[int] = None,
                direction: str = 'in') -> List[int]:
    """n_0..n_r, the per-distance shell sizes of the substitution pairs.

    n_0 = 1 always (the substituted vertex itself as a root).  For d >= 1:

        direction='in'   n_d = K_out^d   (Theorem 6.4, Eq. 44) — a substituted
                         vertex s reaches a root v only if v is in s's FORWARD
                         neighbourhood, whose d-th shell has size <= K_out^d.
        direction='out'  n_d = K_in^d    (Theorem 1/2) — the mirror statement.
    """
    if direction not in ('in', 'out'):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    base = (K_out if K_out is not None else K_in) if direction == 'in' else K_in
    return [1] + [base ** d for d in range(1, r + 1)]


def sparsegnn_mixture_weights(p1: float, p2: float, r: int, K_in: int,
                              K_out: Optional[int] = None,
                              direction: str = 'in') -> np.ndarray:
    """Mixture weights of the substitution pairs (Eq. 46).

    pi is the law of J = sum_d Binomial(n_d, p1 q_d); length N_r + 1.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must lie in [0, 1]")
    if r < 0 or K_in < 1:
        raise ValueError("need r >= 0 and K_in >= 1")
    K = min(K_in, K_out if K_out is not None else K_in)
    q = _q_products(p2, r, K)
    n = shell_sizes(r, K_in, K_out, direction=direction)

    pi = np.array([1.0])
    for d in range(0, r + 1):
        pi = np.convolve(pi, _binom_pmf(n[d], p1 * q[d]))
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    return pi


def thm4_fiber_weights(p1: float, p2: float, r: int, K_in: int,
                       K_out: Optional[int] = None) -> np.ndarray:
    """Fiber weights of the Theorem 4.5 marked mixture: sum_d Binomial(n_d, a_d).

    Applies to direction='out' only — Theorem 4.5 has no in-expansion
    counterpart.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must lie in [0, 1]")
    if r < 0 or K_in < 1:
        raise ValueError("need r >= 0 and K_in >= 1")
    K = min(K_in, K_out if K_out is not None else K_in)
    pi = np.array([1.0])
    for d in range(1, r + 1):
        if p2 >= 1.0:
            qx = 1.0
        else:
            log_keep = sum((K ** (l - 1)) * math.log1p(-(p2 ** l))
                           for l in range(d, r + 1))
            qx = 1.0 - math.exp(log_keep)
        pi = np.convolve(pi, _binom_pmf(K_in ** d, p1 * qx))
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    return pi


# ══ certified-pessimistic PLD construction ════════════════════════════════════

def _mixture_cdf(x: np.ndarray, means: Sequence[float],
                 weights: Sequence[float], sigma: float) -> np.ndarray:
    from scipy.special import ndtr
    out = np.zeros_like(x, dtype=float)
    for mu, w in zip(means, weights):
        out += w * ndtr((x - mu) / sigma)
    return out


def _mixture_logpdf(x: np.ndarray, means: Sequence[float],
                    weights: Sequence[float], sigma: float) -> np.ndarray:
    const = math.log(sigma * math.sqrt(2.0 * math.pi))
    out = np.full_like(x, -np.inf, dtype=float)
    for mu, w in zip(means, weights):
        if w <= 0.0:
            continue
        out = np.logaddexp(
            out, math.log(w) - 0.5 * ((x - mu) / sigma) ** 2 - const)
    return out


def _pld_from_fibers(
    fibers: Sequence[Tuple[float, Sequence[float], Sequence[float],
                           Sequence[float], Sequence[float]]],
    sigma: float,
    discretization: float,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
    cell_mass_floor: float = _CELL_MASS_FLOOR,
):
    """dp_accounting PLD dominating the marked sum of (Num_f, Den_f) fibers.

    Each fiber is (weight, num_means, num_weights, den_means, den_weights).
    Fibers are disjoint, so the fiber weight cancels out of the privacy loss
    and only scales the mass (Theorem 4.4); one fiber of weight 1 is the plain
    unmarked pair.
    """
    from dp_accounting.pld import privacy_loss_distribution as _PLD

    log_upper = {}
    log_lower = {}
    inf_mass = 0.0          # num mass at loss = +infinity
    lower_total = 0.0       # accumulated den mass placed so far

    for f, (w_f, num_means, num_w, den_means, den_w) in enumerate(fibers):
        if w_f <= 0.0:
            continue
        lo = min(min(num_means), min(den_means)) - n_sigma * sigma
        hi = max(max(num_means), max(den_means)) + n_sigma * sigma
        n_atoms = int(min(2_000_000,
                          max(2000, (hi - lo) * atoms_per_sigma / sigma)))
        edges = np.linspace(lo, hi, n_atoms + 1)

        num_mass = w_f * np.diff(_mixture_cdf(edges, num_means, num_w, sigma))
        edge_loss = (_mixture_logpdf(edges, num_means, num_w, sigma)
                     - _mixture_logpdf(edges, den_means, den_w, sigma))
        d = np.diff(edge_loss)
        if not ((d <= 1e-9).all() or (d >= -1e-9).all()):
            raise AssertionError(
                "edge-loss sequence is not monotone; the per-cell endpoint "
                "maximum is not a certified bound for this pair")
        cell_loss = np.maximum(edge_loss[:-1], edge_loss[1:])

        # Grid tails (num mass beyond [lo, hi]) go to the infinity outcome.
        inf_mass += max(0.0, w_f - float(num_mass.sum()))

        keep = num_mass >= cell_mass_floor
        inf_mass += float(num_mass[~keep].sum())
        for i in np.flatnonzero(keep):
            m = float(num_mass[i])
            lm = math.log(m)
            log_upper[(f, int(i))] = lm
            log_lower[(f, int(i))] = lm - float(cell_loss[i])
            lower_total += m * math.exp(-float(cell_loss[i]))

    if inf_mass > 0.0:
        log_upper['inf'] = math.log(inf_mass)      # absent from lower: loss=+inf
    deficit = max(0.0, 1.0 - lower_total)
    if deficit > 0.0:
        log_lower['rest'] = math.log(deficit)      # absent from upper: harmless

    # symmetric=True: this helper builds one direction and callers handle
    # pairing.  With symmetric=False dp_accounting also derives the swapped
    # direction from these dicts, where 'rest' becomes real infinity mass and
    # epsilon collapses to inf.
    return _PLD.from_two_probability_mass_functions(
        log_lower, log_upper, pessimistic_estimate=True,
        value_discretization_interval=discretization, symmetric=True)


# ══ epsilon entry points ══════════════════════════════════════════════════════

def _substitution_pld_from_weights(pi, sigma, grid, n_sigma, atoms_per_sigma):
    """Single-step substitution PLD from already-prepared mixture weights."""
    kept = np.flatnonzero(pi >= _COMPONENT_MASS_FLOOR)
    weights = [float(pi[k]) for k in kept]
    p_means = [-2.0 * float(k) for k in kept]
    q_means = [+2.0 * float(k) for k in kept]
    fibers = [(1.0, p_means, weights, q_means, weights)]
    return _pld_from_fibers(fibers, sigma, discretization=grid,
                            n_sigma=n_sigma, atoms_per_sigma=atoms_per_sigma)


def _substitution_pld(p1, p2, r, K_in, K_out, sigma, direction, grid,
                      n_sigma, atoms_per_sigma):
    """Single-step PLD for the substitution pair (Thm 6.4 / Thm 1-2)."""
    pi = sparsegnn_mixture_weights(p1, p2, r, K_in, K_out, direction=direction)
    return _substitution_pld_from_weights(
        pi, sigma, grid, n_sigma, atoms_per_sigma)


def _compose_schedule(base_plds, steps, eval_fn):
    """{t: eval_fn(compositions)} over sorted checkpoints.

    Single-step PLDs are advanced in lockstep by one incremental compose per
    gap, so the whole schedule costs about one FFT per checkpoint.
    """
    out = {}
    cur = [None] * len(base_plds)
    last = 0
    for t in sorted({int(s) for s in steps}):
        if t < 1:
            raise ValueError(f"checkpoints must be >= 1, got {t}")
        gap = t - last
        if gap > 0:
            for i, base in enumerate(base_plds):
                block = base if gap == 1 else base.self_compose(gap)
                cur[i] = block if cur[i] is None else cur[i].compose(block)
            last = t
        out[t] = eval_fn(cur)
    return out


def sparsegnn_substitution_epsilon_schedule(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps,
    delta: float,
    K_out: Optional[int] = None,
    direction: str = 'in',
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
):
    """{t: epsilon} per checkpoint under node substitution.

    Each entry is a valid (eps(t), delta) guarantee for the iterate released
    at step t.
    """
    base = _substitution_pld(p1, p2, r, K_in, K_out, sigma, direction, grid,
                             n_sigma, atoms_per_sigma)
    return _compose_schedule(
        [base], steps, lambda cur: cur[0].get_epsilon_for_delta(delta))


def sparsegnn_substitution_epsilon(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps: int,
    delta: float,
    K_out: Optional[int] = None,
    direction: str = 'in',
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
) -> float:
    """(eps, delta) for T steps under node substitution.

    Theorem 6.4 for direction='in', Theorem 1/2 for 'out'.  One orientation
    suffices: Q is P reflected through the origin, so the pair is its own
    reverse.  `grid` is dp_accounting's value_discretization_interval.
    """
    pld = _substitution_pld(p1, p2, r, K_in, K_out, sigma, direction, grid,
                            n_sigma, atoms_per_sigma)
    return pld.self_compose(steps).get_epsilon_for_delta(delta)


def _thm4_plds_from_weights(pi, p1, sigma, grid, n_sigma, atoms_per_sigma):
    """Theorem 4.5 PLDs from already-prepared marked-mixture weights."""
    kept = [(j, float(pij)) for j, pij in enumerate(pi)
            if pij >= _COMPONENT_MASS_FLOOR]
    plds = []
    for swap in (False, True):
        fibers = []
        for j, pij in kept:
            p_side = ([-float(j)], [1.0])
            q_side = ([float(j), float(j) + 1.0], [1.0 - p1, p1])
            num, den = (q_side, p_side) if swap else (p_side, q_side)
            fibers.append((pij, num[0], num[1], den[0], den[1]))
        plds.append(_pld_from_fibers(fibers, sigma, discretization=grid,
                                     n_sigma=n_sigma,
                                     atoms_per_sigma=atoms_per_sigma))
    return plds


def _thm4_plds(p1, p2, r, K_in, K_out, sigma, grid, n_sigma, atoms_per_sigma):
    """Single-step PLDs (insertion direction, removal direction) for Thm 4.5."""
    pi = thm4_fiber_weights(p1, p2, r, K_in, K_out)
    return _thm4_plds_from_weights(
        pi, p1, sigma, grid, n_sigma, atoms_per_sigma)


def sparsegnn_thm4_epsilon_schedule(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps,
    delta: float,
    K_out: Optional[int] = None,
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
):
    """{t: epsilon} per checkpoint for the Theorem 4.5 pair (max over both
    orientations)."""
    plds = _thm4_plds(p1, p2, r, K_in, K_out, sigma, grid, n_sigma,
                      atoms_per_sigma)
    return _compose_schedule(
        plds, steps,
        lambda cur: max(c.get_epsilon_for_delta(delta) for c in cur))


def sparsegnn_thm4_epsilon(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps: int,
    delta: float,
    K_out: Optional[int] = None,
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
) -> float:
    """(eps, delta) for T steps via the Theorem 4.5 marked pair, out-expansion
    only.

    The pair is not symmetric, so both orientations (insertion and removal) are
    composed and the max returned.
    """
    plds = _thm4_plds(p1, p2, r, K_in, K_out, sigma, grid, n_sigma,
                      atoms_per_sigma)
    return max(p.self_compose(steps).get_epsilon_for_delta(delta)
               for p in plds)


def resolve_sparsegnn_theorem(direction: str, theorem: str = "auto") -> str:
    """Resolve the applicable SparseGNN dominating-pair theorem selector."""
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    if theorem not in ("auto", "substitution", "thm45"):
        raise ValueError(
            "theorem must be 'auto', 'substitution', or 'thm45', "
            f"got {theorem!r}")
    resolved = "substitution" if theorem == "auto" and direction == "in" else theorem
    if theorem == "auto" and direction == "out":
        resolved = "thm45"
    if resolved == "thm45" and direction != "out":
        raise ValueError(
            "Theorem 4.5 is stated for out-expansion (Algorithm 4) only; "
            "use theorem='substitution' for in-expansion.")
    return resolved


def sparsegnn_theorem_label(direction: str, theorem: str = "auto") -> str:
    """Reporting label for the selected applicable SparseGNN theorem."""
    resolved = resolve_sparsegnn_theorem(direction, theorem)
    if resolved == "thm45":
        return "thm4.5-insertion-removal"
    return ("thm6.4-substitution" if direction == "in"
            else "thm1.2-substitution")


def sparsegnn_epsilon_schedule(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps,
    delta: float,
    K_out: Optional[int] = None,
    direction: str = "in",
    theorem: str = "auto",
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
):
    """Checkpoint epsilon schedule under the selected applicable theorem."""
    if resolve_sparsegnn_theorem(direction, theorem) == "thm45":
        return sparsegnn_thm4_epsilon_schedule(
            p1, p2, r, K_in, sigma, steps, delta, K_out=K_out, grid=grid,
            n_sigma=n_sigma, atoms_per_sigma=atoms_per_sigma)
    return sparsegnn_substitution_epsilon_schedule(
        p1, p2, r, K_in, sigma, steps, delta, K_out=K_out,
        direction=direction, grid=grid, n_sigma=n_sigma,
        atoms_per_sigma=atoms_per_sigma)


def sparsegnn_epsilon(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps: int,
    delta: float,
    K_out: Optional[int] = None,
    direction: str = "in",
    theorem: str = "auto",
    grid: float = 1e-4,
    n_sigma: float = 10.0,
    atoms_per_sigma: float = 400.0,
) -> float:
    """Final-iterate epsilon under the selected applicable theorem."""
    if resolve_sparsegnn_theorem(direction, theorem) == "thm45":
        return sparsegnn_thm4_epsilon(
            p1, p2, r, K_in, sigma, steps, delta, K_out=K_out, grid=grid,
            n_sigma=n_sigma, atoms_per_sigma=atoms_per_sigma)
    return sparsegnn_substitution_epsilon(
        p1, p2, r, K_in, sigma, steps, delta, K_out=K_out,
        direction=direction, grid=grid, n_sigma=n_sigma,
        atoms_per_sigma=atoms_per_sigma)


@dataclass(frozen=True)
class SparseGNNNoiseCalibration:
    """Noise calibration from a certified SparseGNN PLD inversion."""

    noise_multiplier: float
    noise_std: float
    noise_variance: float
    epsilon: float
    target_epsilon: float
    delta: float
    theorem: str
    evaluations: int

    def as_dict(self) -> dict:
        return asdict(self)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def calibrate_sparsegnn_noise(
    *, target_epsilon: float, target_delta: float,
    p1: float, p2: float, r: int, K_in: int, K_out: int,
    steps: int, clip: float = 1.0, direction: str = "in",
    theorem: str = "auto", grid: float = 1e-4,
    sigma_rtol: float = 1e-3, sigma_atol: float = 1e-6,
    max_sigma: float = 1e6,
) -> SparseGNNNoiseCalibration:
    """Find the smallest known-safe multiplier for a SparseGNN privacy budget."""
    target_epsilon = _positive_finite("target_epsilon", target_epsilon)
    target_delta = float(target_delta)
    if not math.isfinite(target_delta) or not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must be finite and lie in (0, 1)")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    clip = _positive_finite("clip", clip)
    grid = _positive_finite("grid", grid)
    sigma_rtol = _positive_finite("sigma_rtol", sigma_rtol)
    sigma_atol = _positive_finite("sigma_atol", sigma_atol)
    max_sigma = _positive_finite("max_sigma", max_sigma)
    if max_sigma < 1.0:
        raise ValueError("max_sigma must be at least 1")

    resolved = resolve_sparsegnn_theorem(direction, theorem)
    if resolved == "substitution":
        weights = sparsegnn_mixture_weights(
            p1, p2, r, K_in, K_out, direction=direction)

        def build(sigma):
            return _substitution_pld_from_weights(
                weights, sigma, grid, n_sigma=10.0, atoms_per_sigma=400.0)

        def epsilon_from_pld(pld):
            return pld.self_compose(steps).get_epsilon_for_delta(target_delta)
    else:
        weights = thm4_fiber_weights(p1, p2, r, K_in, K_out)

        def build(sigma):
            return _thm4_plds_from_weights(
                weights, p1, sigma, grid, n_sigma=10.0, atoms_per_sigma=400.0)

        def epsilon_from_pld(plds):
            return max(p.self_compose(steps).get_epsilon_for_delta(target_delta)
                       for p in plds)

    values = {}

    def epsilon_at(sigma):
        if sigma not in values:
            epsilon = float(epsilon_from_pld(build(sigma)))
            if math.isnan(epsilon):
                raise RuntimeError(
                    f"SparseGNN accountant returned NaN at sigma={sigma}")
            values[sigma] = epsilon
        return values[sigma]

    low, high = 0.0, 1.0
    high_epsilon = epsilon_at(high)
    while math.isinf(high_epsilon) or high_epsilon > target_epsilon:
        low = high
        if high >= max_sigma:
            raise RuntimeError(
                "failed to bracket a SparseGNN noise multiplier at "
                f"max_sigma={max_sigma}")
        high = min(high * 2.0, max_sigma)
        high_epsilon = epsilon_at(high)

    while high - low > max(sigma_atol, sigma_rtol * high):
        mid = (low + high) / 2.0
        mid_epsilon = epsilon_at(mid)
        if math.isinf(mid_epsilon) or mid_epsilon > target_epsilon:
            low = mid
        else:
            high, high_epsilon = mid, mid_epsilon

    noise_std = high * clip
    return SparseGNNNoiseCalibration(
        noise_multiplier=high,
        noise_std=noise_std,
        noise_variance=noise_std ** 2,
        epsilon=high_epsilon,
        target_epsilon=target_epsilon,
        delta=target_delta,
        theorem=sparsegnn_theorem_label(direction, resolved),
        evaluations=len(values),
    )


def naive_opacus_epsilon(sigma: float, sample_rate: float, steps: int,
                         delta: float, mechanism: str = 'prv') -> float:
    """Opacus epsilon for a Poisson-subsampled Gaussian at rate `sample_rate`.

    This is what accounting would claim if a node only influenced its own
    subgraph — it ignores that a node appears in neighbours' expansions, so it
    is NOT a valid node-level guarantee; it is a floor showing the price of
    graph structure.  PRV by default, RDP fallback.
    """
    from opacus.accountants import create_accountant
    try:
        accountant = create_accountant(mechanism=mechanism)
        accountant.history = [(sigma, sample_rate, steps)]
        return accountant.get_epsilon(delta=delta)
    except Exception:
        accountant = create_accountant(mechanism='rdp')
        accountant.history = [(sigma, sample_rate, steps)]
        return accountant.get_epsilon(delta=delta)
