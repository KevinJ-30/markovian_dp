"""
Dominating pairs for one SparseGNN step (manuscript v36).

Two families are implemented, one per neighbouring relation.  Which one applies
is decided by the EXPANSION ORIENTATION used during training — see
`src/sparse/sparse_expand.py`.


1. SUBSTITUTION — Theorem 6.4 (in-expansion) and Theorem 1/2 (out-expansion)
---------------------------------------------------------------------------
Both theorems have the identical form; only the shell size differs, because a
substituted vertex s can affect a root v only if v lies in s's forward directed
neighbourhood (in-expansion) or its backward one (out-expansion):

    K   = min(K_in, K_out)
    q_0 = 1,   q_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}}      (Eq. 3 / 43)
    n_0 = 1,   n_d = K_out^d   for direction='in'   (Thm 6.4, Eq. 44)
               n_d = K_in^d    for direction='out'  (Thm 1/2)
    N_r = sum_{d=0..r} n_d
    sum_k pi_k z^k = prod_{d=0..r} (1 - p1 q_d + p1 q_d z)^{n_d}  (Eq. 4 / 46)
    P = sum_k pi_k N(-2k, sigma^2),  Q = sum_k pi_k N(+2k, sigma^2)  (Eq. 5 / 47)

The means are spaced by 2 because an arbitrary substitution of one rooted
subgraph moves the clipped sum by up to 2C.  Q is the reflection of P through
the origin, so H_a(P||Q) = H_a(Q||P) and the pair is its own reverse.


2. INSERTION/REMOVAL — Theorem 4.5 (out-expansion only)
-------------------------------------------------------
Tighter per unit distance, because a single insertion or removal has
sensitivity C rather than 2C, but the mark j is disclosed, so the pair loses
the mixture-level amplification of the substitution family.  Stated for
Algorithm 4 (out-expansion); it serves as the out-orientation ablation:

    K = min(K_in, K_out),  n_d = K_in^d
    q×_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}},   a_d = p1 * q×_d
    pi_j from the PGF   sum_j pi_j z^j = prod_{d=1..r} (1 - a_d + a_d z)^{n_d}
    P→ = ⊕_j pi_j N(-j, sigma^2)
    Q→ = ⊕_j pi_j [ (1-p1) N(j, sigma^2) + p1 N(j+1, sigma^2) ]

The PGF factors are the SAME Binomial family as the substitution pairs: by
Lemma 17, each of the n_d common-root slots at distance d activates
independently with probability a_d, so pi is the law of sum_d Binomial(n_d,
a_d).  (An earlier revision of this module misread Eq. (29) as a single
Bernoulli with jump n_d, which put mass a_d directly on the mark K_in^d and
inflated epsilon by roughly 4x, or to infinity.)  (P→, Q→) dominates the
insertion direction and (Q→, P→) the removal direction, so the reported
epsilon is the max over both.  Because the fibers are disjoint (the mark), the
marked pair's privacy loss distribution is exactly the pi-weighted mixture of
per-fiber loss distributions, which is how it is discretized below.


NUMERICAL BACKEND — Google dp_accounting
----------------------------------------
Composition and epsilon(delta) are delegated to
`dp_accounting.pld.privacy_loss_distribution`; nothing here hand-rolls PLD
convolution any more.  The discretized input is CERTIFIED pessimistic:

  * cell masses are exact Gaussian-mixture CDF differences (no density * dx
    midpoint approximation);
  * the per-cell privacy loss is the larger of the two cell-edge losses, which
    upper-bounds the loss everywhere in the cell because the edge-loss
    sequence is verified monotone at construction time;
  * the lower (denominator) mass fed to dp_accounting is num_mass *
    exp(-loss), which is <= the true denominator mass, so the resulting pair
    dominates the analytic pair — its hockey-stick divergence, and every
    composition of it, is an upper bound;
  * truncated component/tail/cell mass on the numerator side goes to a
    dedicated infinite-loss outcome, and the denominator deficit to an outcome
    absent from the numerator, both of which are pessimistic.

dp_accounting then applies its own pessimistic rounding of the losses onto the
`value_discretization_interval` grid, composes by FFT, and converts to
epsilon(delta).
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Total numerator mass allowed to be dropped into the infinite-loss outcome by
# component / fiber / cell trimming.  Anything smaller than these floors is
# treated as loss = +infinity, which is pessimistic, so the reported epsilon
# stays a valid upper bound no matter how the floors are chosen.
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
    """Mixture weights pi_0..pi_{N_r} of the substitution pairs (Eq. 4 / 46).

    sum_k pi_k z^k = prod_{d=0..r} (1 - p1 q_d + p1 q_d z)^{n_d}, i.e. pi is
    the law of J = sum_d Binomial(n_d, p1 q_d).  Length N_r + 1 with
    N_r = sum_{d=0..r} n_d.
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
    """Fiber weights pi_0..pi_{N_com_r} of the Theorem 4.5 marked mixture.

    pi_j is the coefficient of z^j in prod_{d=1..r} (1 - a_d + a_d z)^{n_d},
    with a_d = p1 * q×_d, q×_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}},
    K = min(K_in, K_out), n_d = K_in^d — i.e. pi is the law of
    sum_d Binomial(n_d, a_d), per Lemma 17 (each of the n_d slots at distance
    d activates independently with probability a_d).  For r = 0 the product is
    empty and pi = [1.0] (only the j = 0 fiber: the inserted node itself).

    Theorem 4.5 is stated for Algorithm 4 (OUT-expansion); it has no
    in-expansion counterpart, so it applies only to runs recorded with
    direction='out'.
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
    """dp_accounting PLD dominating ⊕_f w_f (Num_f, Den_f) in the Num direction.

    Each fiber is (weight, num_means, num_weights, den_means, den_weights),
    the two sides being Gaussian mixtures with common std `sigma`.  Fibers are
    disjoint marked components: the privacy loss on fiber f is
    log(w_f num_f / (w_f den_f)) = log(num_f / den_f), so the fiber weight
    cancels in the loss and scales only the mass — exactly the marked direct
    sum of Theorem 4.4.  A single fiber of weight 1 is the plain-pair case.

    Pessimism (see module docstring): exact CDF cell masses; per-cell loss =
    max of the two edge losses, valid because the edge-loss sequence is
    verified monotone; lower mass = num_mass * exp(-loss) <= true den mass;
    trimmed num mass -> infinite-loss outcome; den deficit -> a den-only
    outcome.
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

    # symmetric=True means "this PLD covers one direction; treat it as both".
    # That is the correct call here: each invocation of this helper constructs
    # exactly the num-vs-den direction, and the CALLERS handle direction
    # pairing (the substitution pair is analytically its own reverse; the
    # Theorem 4.5 entry point invokes this helper once per orientation and
    # takes the max).  symmetric=False would make dp_accounting additionally
    # build the swapped direction from these same dicts, where the 'rest'
    # outcome becomes genuine infinity mass and epsilon collapses to inf.
    return _PLD.from_two_probability_mass_functions(
        log_lower, log_upper, pessimistic_estimate=True,
        value_discretization_interval=discretization, symmetric=True)


# ══ epsilon entry points ══════════════════════════════════════════════════════

def _substitution_pld(p1, p2, r, K_in, K_out, sigma, direction, grid,
                      n_sigma, atoms_per_sigma):
    """Single-step PLD for the substitution pair (Thm 6.4 / Thm 1-2)."""
    pi = sparsegnn_mixture_weights(p1, p2, r, K_in, K_out, direction=direction)
    kept = np.flatnonzero(pi >= _COMPONENT_MASS_FLOOR)
    weights = [float(pi[k]) for k in kept]
    p_means = [-2.0 * float(k) for k in kept]
    q_means = [+2.0 * float(k) for k in kept]
    # Dropped components lose their mass from the numerator grid cells, and the
    # helper routes exactly that deficit to the infinite-loss outcome.
    fibers = [(1.0, p_means, weights, q_means, weights)]
    return _pld_from_fibers(fibers, sigma, discretization=grid,
                            n_sigma=n_sigma, atoms_per_sigma=atoms_per_sigma)


def _compose_schedule(base_plds, steps, eval_fn):
    """eval_fn over the running composition of `base_plds` at each checkpoint.

    `base_plds` are single-step PLDs advanced in lockstep; eval_fn receives the
    list of current compositions.  Returns {t: eval_fn(...)} for the sorted,
    deduplicated checkpoints.  One incremental compose per gap keeps the cost
    at ~one FFT per checkpoint rather than a full self_compose per t.
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
    """{t: epsilon} for every checkpoint t in `steps` (NODE SUBSTITUTION).

    The single-step pair is built once and composed incrementally across the
    sorted checkpoints, so a 40-checkpoint schedule costs about as much as one
    full-length composition.  Each entry is a valid (eps(t), delta) guarantee
    for the mechanism truncated after t steps — i.e. for the model iterate
    released at step t.
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
    """(eps, delta) for T steps of SparseGNN under NODE SUBSTITUTION.

    Uses Theorem 6.4 when direction='in' (Algorithm 5, the corrected
    orientation) and Theorem 1/2 when direction='out'.  Composes the per-step
    pair over `steps` via dp_accounting PLD self-composition.

    Only one orientation is composed: Q is the reflection of P through the
    origin, so H_a(P||Q) = H_a(Q||P) and the pair already covers both ordered
    neighbouring directions.  (The Theorem 4.5 marked pair is NOT symmetric
    and does need both — see `sparsegnn_thm4_epsilon`.)

    `grid` is dp_accounting's value_discretization_interval; its rounding is
    pessimistic, so together with the certified pair construction the returned
    epsilon is a valid upper bound.
    """
    pld = _substitution_pld(p1, p2, r, K_in, K_out, sigma, direction, grid,
                            n_sigma, atoms_per_sigma)
    return pld.self_compose(steps).get_epsilon_for_delta(delta)


def _thm4_plds(p1, p2, r, K_in, K_out, sigma, grid, n_sigma, atoms_per_sigma):
    """Single-step PLDs (insertion direction, removal direction) for Thm 4.5."""
    pi = thm4_fiber_weights(p1, p2, r, K_in, K_out)
    kept = [(j, float(pij)) for j, pij in enumerate(pi)
            if pij >= _COMPONENT_MASS_FLOOR]
    # Dropped fibers lose their mass from the numerator, and the helper routes
    # the deficit to the infinite-loss outcome — pessimistic in each direction.
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
    """{t: epsilon} for every checkpoint t in `steps` (Theorem 4.5 pair).

    Both orientations are composed incrementally in lockstep and the max is
    reported per checkpoint, matching `sparsegnn_thm4_epsilon`.
    """
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
    """(eps, delta) for T steps of SparseGNN via the Theorem 4.5 marked pair.

    Applies to OUT-expansion runs only (Algorithm 4); use
    `sparsegnn_substitution_epsilon` for the corrected in-expansion.

    Composes the per-step marked pair over `steps` in BOTH orientations
    (insertion: P→ vs Q→; removal: Q→ vs P→) and returns the max — the
    guarantee under symmetric node insertion/removal adjacency.
    """
    plds = _thm4_plds(p1, p2, r, K_in, K_out, sigma, grid, n_sigma,
                      atoms_per_sigma)
    return max(p.self_compose(steps).get_epsilon_for_delta(delta)
               for p in plds)


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
