"""
Theorem 3 and Theorem 4 dominating pairs for SparseGNN.

Theorem 3 gives a graph-independent dominating pair for one SparseGNN step under
the node-level neighboring relation:

    N_r = |U-bar| = 1 + sum_{d=1..r} K_in^d
    U-bar = {(d, j) : 0 <= d <= r, 1 <= j <= n_d},  n_0 = 1,  n_d = K_in^d
    activation probs:  a_{0,1} = p1,  a_{d,j} = p1 * q_d   (d >= 1)
      q_0 = 1,  q_d = min(1, sum_{l=d..r} K^{l-1} p2^l),  K = min(K_in, K_out)
    weights:  w_A = prod_{u in A} a_u * prod_{u not in A} (1 - a_u)
    Gray ordering A_0..A_{2^{N_r}-1} of subsets of U-bar (A_0 = empty)
    means:    mu^(1) = (0, -2, -4, ..., -2(2^{N_r}-1)),  mu^(2) = -mu^(1)
    P = MoG(mu^(1), w, sigma),  Q = MoG(mu^(2), w, sigma)

Then Hα(M_g || M_g') <= Hα(P || Q) for every neighboring pair g ~ g', and
(P^{⊗T}, Q^{⊗T}) dominates the T-step transcript by adaptive composition.

The Theorem 3 mixture has 2^{N_r} components, which explodes with K_in and r;
it is tractable only for small (K_in, r).

Theorem 4 (Section 4) is the pair we actually account with.  It works under the
node INSERTION/REMOVAL relation and gives a marked mixture with only
N_com_r + 1 = 1 + sum_{d=1..r} K_in^d fibers:

    K = min(K_in, K_out),  n_d = K_in^d
    q×_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}},   a_d = p1 * q×_d
    pi_j from the PGF   sum_j pi_j z^j = prod_{d=1..r} (1 - a_d + a_d z^{n_d})
    P→ = ⊕_j pi_j N(-j, sigma^2)
    Q→ = ⊕_j pi_j [ (1-p1) N(j, sigma^2) + p1 N(j+1, sigma^2) ]

(P→, Q→) dominates the insertion direction and (Q→, P→) the removal direction
of ONE SparseGNN step; T steps compose via PLD self-composition, and the
reported epsilon is the max over the two orientations.  Because the fibers are
disjoint (the mark), the marked pair's PLD is exactly the pi-weighted mixture
of per-fiber PLDs, which is how we discretize it.

Both pairs are emitted as discretized (p_atoms, q_atoms) arrays compatible with
`dp-subsample-prelim/accounting.py::PrivacyLossDistribution.from_dominating_pair`,
matching the contract of the `make_novel_mechanism_dominating_pair` hook in
`sparsification_experiments/dp_accounting.py`.
"""

import importlib.util
import math
import os
from typing import List, Optional

import numpy as np


def _load_pld_module():
    """Load dp-subsample-prelim/accounting.py by path (hyphenated dir, not a
    package) without shadowing this module's name."""
    path = os.path.join(os.path.dirname(__file__), '..', '..',
                        'dp-subsample-prelim', 'accounting.py')
    spec = importlib.util.spec_from_file_location('_dp_prelim_accounting', path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PLD module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def activation_probabilities(p1: float, p2: float, r: int, K_in: int,
                             K_out: Optional[int] = None) -> List[float]:
    """Flat list of activation probabilities a_u over U-bar (Theorem 3 / Lemma 10).

    Ordering: index 0 is (d=0, j=1); then (1,1..n_1), (2,1..n_2), ..., (r,1..n_r).
    Length N_r = 1 + sum_{d=1..r} K_in^d.
    """
    K = min(K_in, K_out if K_out is not None else K_in)
    a = [p1]                       # (0, 1): a_{0,1} = p1  (q_0 = 1)
    for d in range(1, r + 1):
        qd = min(1.0, sum((K ** (l - 1)) * (p2 ** l) for l in range(d, r + 1)))
        n_d = K_in ** d
        a.extend([p1 * qd] * n_d)
    return a


def _gray(i: int) -> int:
    return i ^ (i >> 1)


def _subset_weights(a: np.ndarray) -> np.ndarray:
    """w_{A_i} for the Gray ordering of all 2^len(a) subsets of U-bar.

    Returns an array `w` of length 2^len(a) with w[i] = product over the Gray
    subset A_i = gray(i) of a_u for u in A_i and (1 - a_u) otherwise.
    """
    n = len(a)
    M = 1 << n
    log_a = np.log(np.clip(a, 1e-300, 1.0))
    log_1ma = np.log(np.clip(1.0 - a, 1e-300, 1.0))
    w = np.empty(M, dtype=np.float64)
    for i in range(M):
        g = _gray(i)
        s = 0.0
        for k in range(n):
            if (g >> k) & 1:
                s += log_a[k]
            else:
                s += log_1ma[k]
        w[i] = math.exp(s)
    total = w.sum()
    if total > 0:
        w /= total          # guard against tiny numerical drift
    return w


def sparsegnn_dominating_pair(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    K_out: Optional[int] = None,
    sigma: float = 1.0,
    n_atoms: int = 40000,
    n_sigma: float = 10.0,
    max_components: int = 512,
):
    """Discretize the Theorem 3 dominating pair (P, Q) into aligned atom masses.

    Returns (p_atoms, q_atoms): float64 arrays each summing to 1.0, where atom i
    carries P-mass p_atoms[i] and Q-mass q_atoms[i] on a shared x-grid.  Feed
    directly to PrivacyLossDistribution.from_dominating_pair.

    The number of mixture components is 2^{N_r}; `max_components` guards against
    the combinatorial blow-up (raise K_in/r only for small values).
    """
    a = np.array(activation_probabilities(p1, p2, r, K_in, K_out), dtype=np.float64)
    Nr = len(a)
    M = 1 << Nr
    if M > max_components:
        raise ValueError(
            f"Theorem 3 mixture has 2^{Nr} = {M} components > max_components="
            f"{max_components}. This dominating pair is only tractable for small "
            f"(K_in, r); increase max_components deliberately if you understand "
            f"the cost."
        )

    w = _subset_weights(a)                         # [M]
    idx = np.arange(M, dtype=np.float64)
    means_p = -2.0 * idx                           # mu^(1)
    means_q = +2.0 * idx                           # mu^(2)

    # Shared grid covering both mixtures' support.
    span = 2.0 * (M - 1)
    x_lo = -span - n_sigma * sigma
    x_hi = +span + n_sigma * sigma
    edges = np.linspace(x_lo, x_hi, n_atoms + 1)
    dx = edges[1] - edges[0]
    x_mid = 0.5 * (edges[:-1] + edges[1:])

    inv = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    def _mixture_density(means):
        dens = np.zeros_like(x_mid)
        for wi, mu in zip(w, means):
            if wi <= 0.0:
                continue
            dens += wi * inv * np.exp(-0.5 * ((x_mid - mu) / sigma) ** 2)
        return dens

    p_density = _mixture_density(means_p)
    q_density = _mixture_density(means_q)

    p_atoms = p_density * dx
    q_atoms = q_density * dx

    # Absorb residual tail mass so both sum to exactly 1.
    p_atoms = np.append(p_atoms, max(0.0, 1.0 - float(p_atoms.sum())))
    q_atoms = np.append(q_atoms, max(0.0, 1.0 - float(q_atoms.sum())))
    return p_atoms, q_atoms


# ══ Theorem 4: insertion/removal marked pair ═══════════════════════════════════

def thm4_fiber_weights(p1: float, p2: float, r: int, K_in: int,
                       K_out: Optional[int] = None) -> np.ndarray:
    """Fiber weights pi_0..pi_{N_com_r} of the Theorem 4 marked mixture.

    pi_j is the coefficient of z^j in prod_{d=1..r} (1 - a_d + a_d z^{n_d}),
    with a_d = p1 * q×_d, q×_d = 1 - prod_{l=d..r} (1 - p2^l)^{K^{l-1}},
    K = min(K_in, K_out), n_d = K_in^d.  For r = 0 the product is empty and
    pi = [1.0] (only the j = 0 fiber: the inserted node itself).
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must lie in [0, 1]")
    if r < 0 or K_in < 1:
        raise ValueError("need r >= 0 and K_in >= 1")
    K = min(K_in, K_out if K_out is not None else K_in)
    pi = np.array([1.0])
    for d in range(1, r + 1):
        # log-space product for (1 - p2^l)^{K^{l-1}} to stay stable at large K^l
        log_keep = sum((K ** (l - 1)) * math.log1p(-(p2 ** l))
                       for l in range(d, r + 1) if p2 < 1.0)
        qx = 1.0 if p2 >= 1.0 else 1.0 - math.exp(log_keep)
        a_d = p1 * qx
        n_d = K_in ** d
        factor = np.zeros(n_d + 1)
        factor[0] = 1.0 - a_d
        factor[n_d] = a_d
        pi = np.convolve(pi, factor)
    # Guard tiny FFT/rounding drift; pi must be a probability vector.
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    return pi


def sparsegnn_thm4_pair(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    K_out: Optional[int] = None,
    sigma: float = 1.0,
    atoms_per_fiber: int = 4000,
    n_sigma: float = 10.0,
    fiber_mass_floor: float = 1e-14,
):
    """Discretize the Theorem 4 marked pair (P→, Q→) into aligned atom masses.

    Fibers are disjoint, so the atom list is the concatenation of per-fiber
    grids; on fiber j atom masses are pi_j * density * dx for both families
    (the pi_j cancels in the privacy loss, exactly as in the paper's marked
    direct sum).  Fibers with pi_j < fiber_mass_floor are dropped: their P-mass
    goes to a trailing (p > 0, q = 0) infinity atom and their Q-mass to a
    (p = 0, q > 0) atom, which is pessimistic in both orientations.

    Returns (p_atoms, q_atoms): float64 arrays each summing to 1.0, suitable
    for PrivacyLossDistribution.from_dominating_pair (in either order — use
    (p, q) for the insertion direction and (q, p) for removal).
    """
    pi = thm4_fiber_weights(p1, p2, r, K_in, K_out)
    inv = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    p_parts, q_parts = [], []
    for j, pij in enumerate(pi):
        if pij < fiber_mass_floor:
            continue
        # Grid covering N(-j, s^2) and (1-p1)N(j, s^2) + p1 N(j+1, s^2).
        x_lo = -j - n_sigma * sigma
        x_hi = j + 1 + n_sigma * sigma
        edges = np.linspace(x_lo, x_hi, atoms_per_fiber + 1)
        dx = edges[1] - edges[0]
        x = 0.5 * (edges[:-1] + edges[1:])

        p_dens = inv * np.exp(-0.5 * ((x + j) / sigma) ** 2)
        q_dens = ((1.0 - p1) * inv * np.exp(-0.5 * ((x - j) / sigma) ** 2)
                  + p1 * inv * np.exp(-0.5 * ((x - j - 1) / sigma) ** 2))
        p_parts.append(pij * p_dens * dx)
        q_parts.append(pij * q_dens * dx)

    p_atoms = np.concatenate(p_parts)
    q_atoms = np.concatenate(q_parts)

    # Residual mass (dropped fibers + Gaussian tails beyond n_sigma) goes to
    # two one-sided atoms: infinite loss for P-residual, -infinite for
    # Q-residual.  Pessimistic in the P-vs-Q direction; for the swapped
    # (removal) direction the roles flip and it is pessimistic there too.
    p_res = max(0.0, 1.0 - float(p_atoms.sum()))
    q_res = max(0.0, 1.0 - float(q_atoms.sum()))
    p_atoms = np.concatenate([p_atoms, [p_res, 0.0]])
    q_atoms = np.concatenate([q_atoms, [0.0, q_res]])
    return p_atoms, q_atoms


def _pld_from_atoms(p, q, grid: float, loss_cap: float):
    """PrivacyLossDistribution from aligned atoms, with capped loss support.

    Identical to PrivacyLossDistribution.from_dominating_pair except that the
    privacy loss ln(p/q) is clamped into [-loss_cap, +loss_cap]:
      * losses above +loss_cap send their P-mass to +infinity (pessimistic);
      * losses below -loss_cap are rounded UP to -loss_cap (pessimistic).
    Without the cap, high-j fibers put atoms at losses of order (2j+1)^2/sigma^2,
    which makes the discrete pmf support astronomically wide even though those
    atoms carry negligible mass.
    """
    PrivacyLossDistribution = _load_pld_module().PrivacyLossDistribution

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    finite = (p > 0) & (q > 0)
    losses = np.log(p[finite] / q[finite])
    masses = p[finite]

    keep = losses <= loss_cap          # above cap -> +inf (absent from pmf)
    losses = np.maximum(losses[keep], -loss_cap)
    masses = masses[keep]

    ks = np.ceil(losses / grid - 1e-12).astype(np.int64)
    offset = int(ks.min())
    pmf = np.zeros(int(ks.max()) - offset + 1)
    np.add.at(pmf, ks - offset, masses)
    return PrivacyLossDistribution(pmf, offset, grid)


def sparsegnn_thm4_epsilon(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps: int,
    delta: float,
    K_out: Optional[int] = None,
    grid: float = 1e-3,
    loss_cap: float = 50.0,
    atoms_per_fiber: int = 4000,
    n_sigma: float = 10.0,
) -> float:
    """(eps, delta) for T steps of SparseGNN via the Theorem 4 pair.

    Composes the per-step marked pair over `steps` via PLD self-composition in
    BOTH orientations (insertion: P→ vs Q→; removal: Q→ vs P→) and returns the
    max — the guarantee under symmetric node insertion/removal adjacency.

    `grid` is the PLD loss discretization; rounding is pessimistic, so the
    slack is at most grid * steps.  `loss_cap` bounds the per-step loss
    support; atoms beyond it are treated as infinite loss (pessimistic), so a
    finite returned epsilon is always a valid upper bound, and configurations
    whose genuine epsilon exceeds ~loss_cap report inf.
    """
    p_atoms, q_atoms = sparsegnn_thm4_pair(
        p1, p2, r, K_in, K_out=K_out, sigma=sigma,
        atoms_per_fiber=atoms_per_fiber, n_sigma=n_sigma,
    )
    eps = 0.0
    for a, b in ((p_atoms, q_atoms), (q_atoms, p_atoms)):
        pld = _pld_from_atoms(a, b, grid=grid, loss_cap=loss_cap)
        eps = max(eps, pld.self_compose(steps).get_epsilon(delta))
    return eps
