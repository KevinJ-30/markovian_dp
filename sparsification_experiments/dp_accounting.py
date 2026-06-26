"""
Privacy accounting for sparsification_experiments.

Two paths:
  A. Opacus PRV (preferred) / RDP (fallback): standard subsampled-Gaussian.
  B. Dominating-pair PLD: per-step (P, Q) pair constructed analytically for
     the Poisson-subsampled Gaussian, composed via the PLD convolution in
     dp-subsample-prelim/accounting.py.

Both paths consume (sigma, q) where sigma is the SENSITIVITY-NORMALIZED noise
multiplier (noise std = sigma * Delta, Delta = node sensitivity from
sparsify.node_sensitivity). Folding Delta into sigma reduces the problem to
a standard unit-sensitivity Gaussian, which is exactly what both accountants
expect.

The dominating pair for one step of Poisson-subsampled Gaussian (sensitivity 1):
    Q = N(0, sigma^2)
    P = (1 - q) * N(0, sigma^2) + q * N(1, sigma^2)

See Section 3 of Mironov et al. (2017) and Li et al. (2022) for derivation.
"""

import math
import os
import sys

import numpy as np

# Reach dp-subsample-prelim/accounting.py without installing the package
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'dp-subsample-prelim')
)
from accounting import PrivacyLossDistribution, opacus_epsilon   # noqa: E402


# ── A. Opacus path ────────────────────────────────────────────────────────────

def opacus_prv_epsilon(sigma, q, steps, delta):
    """Epsilon via Opacus PRV accountant (RDP fallback)."""
    try:
        return opacus_epsilon(sigma, q, steps, delta, mechanism="prv")
    except Exception:
        return opacus_epsilon(sigma, q, steps, delta, mechanism="rdp")


# ── B. Dominating-pair path ───────────────────────────────────────────────────

def make_subsampled_gaussian_dominating_pair(q, sigma, n_atoms=60000, n_sigma=10.0):
    """
    Discretize the per-step dominating pair for Poisson-subsampled Gaussian DP-SGD
    in sensitivity-normalized units (sensitivity = 1).

    Q = N(0, sigma^2)
    P = (1-q)*N(0, sigma^2) + q*N(1, sigma^2)

    The output arrays are probability mass vectors over n_atoms contiguous
    intervals covering [x_lo, x_hi], where
        x_lo = -n_sigma * sigma,  x_hi = 1 + n_sigma * sigma.
    A final atom absorbs remaining tail mass so both arrays sum exactly to 1.

    The PLD is then computed by PrivacyLossDistribution.from_dominating_pair,
    which computes log(p[i]/q[i]) per atom and bins the result on the loss grid.

    Args:
        q:       float, Poisson sampling rate
        sigma:   float, sensitivity-normalized noise multiplier
        n_atoms: int, number of x-axis discretization bins (more = more accurate)
        n_sigma: float, truncation width (n_sigma * sigma beyond the centres)

    Returns:
        (p_atoms, q_atoms): numpy float64 arrays, each summing to 1.0
    """
    x_lo = -n_sigma * sigma
    x_hi = 1.0 + n_sigma * sigma
    edges = np.linspace(x_lo, x_hi, n_atoms + 1)
    dx = edges[1] - edges[0]
    x_mid = (edges[:-1] + edges[1:]) / 2.0

    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)

    def _pdf(x, mu, s):
        return inv_sqrt2pi / s * np.exp(-0.5 * ((x - mu) / s) ** 2)

    q_density = _pdf(x_mid, 0.0, sigma)
    p_density = (1.0 - q) * _pdf(x_mid, 0.0, sigma) + q * _pdf(x_mid, 1.0, sigma)

    q_atoms = q_density * dx
    p_atoms = p_density * dx

    # Absorb tail mass so both sum to 1.  At n_sigma=10 both tails are <1e-22
    # so this atom carries negligible mass and ~zero privacy loss.
    p_atoms = np.append(p_atoms, max(0.0, 1.0 - float(p_atoms.sum())))
    q_atoms = np.append(q_atoms, max(0.0, 1.0 - float(q_atoms.sum())))

    return p_atoms, q_atoms


# ── NOVEL MECHANISM HOOK ──────────────────────────────────────────────────────
# When the paper's composite-subsampling mechanism is ready, implement this
# function. The rest of the pipeline (PLD composition, eps reporting, plotting)
# requires no changes — the interface is identical to
# make_subsampled_gaussian_dominating_pair.

def make_novel_mechanism_dominating_pair(q, sigma, **kwargs):
    """
    STUB — replace with the paper's per-step dominating pair.

    Must return (p_atoms, q_atoms): numpy arrays of probability masses, each
    summing to 1.  Atom i contributes mass p_atoms[i] under P and q_atoms[i]
    under Q; the privacy loss at atom i is log(p_atoms[i] / q_atoms[i]).

    Signature intentionally matches make_subsampled_gaussian_dominating_pair
    so the caller can swap this in by changing one line.
    """
    raise NotImplementedError(
        "Novel mechanism dominating pair not yet implemented. "
        "Implement this function with the paper's per-step (P, Q) pair. "
        "Return (p_atoms, q_atoms) with the same contract as "
        "make_subsampled_gaussian_dominating_pair."
    )
# ─────────────────────────────────────────────────────────────────────────────


def dompair_epsilon(q, sigma, steps, delta, grid=1e-4, n_atoms=60000):
    """Epsilon via dominating-pair PLD composition."""
    p_atoms, q_atoms = make_subsampled_gaussian_dominating_pair(
        q, sigma, n_atoms=n_atoms
    )
    pld = PrivacyLossDistribution.from_dominating_pair(p_atoms, q_atoms, grid)
    return pld.self_compose(steps).get_epsilon(delta)


# ── Analytic Gaussian composition (q=1 cross-check) ──────────────────────────

def gaussian_composition_epsilon(sigma, steps, delta):
    """
    Analytic epsilon for T i.i.d. Gaussian mechanisms with sensitivity 1 and
    noise std `sigma` (the sensitivity-normalised case, so sensitivity = 1).

    Uses the Gaussian differential privacy (GDP) framework (Dong et al. 2019):
      T Gaussian mechanisms with mu = 1/sigma each compose to a (sqrt(T)/sigma)-GDP
      mechanism.  Converting (mu_T)-GDP to (eps, delta)-DP:

          delta(eps) = Phi(mu_T/2 - eps/mu_T) - exp(eps) * Phi(-mu_T/2 - eps/mu_T)

    where Phi is the standard normal CDF and mu_T = sqrt(T) / sigma.

    This is numerically identical to the closed-form formula from Balle & Wang (2018)
    and gives the ground truth for the q=1 case.  scipy is required.
    """
    try:
        from scipy.special import ndtr as Phi
    except ImportError:
        raise ImportError("scipy is required for gaussian_composition_epsilon")

    mu_T = math.sqrt(steps) / sigma
    lo, hi = 0.0, mu_T * mu_T + 10.0   # delta(0) > delta, delta(hi) < delta
    for _ in range(200):
        mid = (lo + hi) / 2.0
        d = Phi(mu_T / 2.0 - mid / mu_T) - math.exp(mid) * Phi(-mu_T / 2.0 - mid / mu_T)
        if d > delta:
            lo = mid
        else:
            hi = mid
    return hi


# ── Validation ────────────────────────────────────────────────────────────────

def validate_accountants(sigma_grid, q, steps, delta, grid=1e-4, tol=0.1):
    """
    Cross-check Opacus PRV vs dominating-pair PLD across sigma_grid (q < 1).

    Prints a comparison table.  Raises AssertionError if any row exceeds `tol`.
    """
    print(f"\n=== Accountant validation (subsampled Gaussian): "
          f"q={q}, steps={steps}, delta={delta:g} ===")
    print(f"{'sigma':>8} {'eps_opacus':>12} {'eps_dompair':>13} {'diff':>10} {'status':>8}")
    print("-" * 57)
    failures = []
    for s in sigma_grid:
        eps_op = opacus_prv_epsilon(s, q, steps, delta)
        eps_dp = dompair_epsilon(q, s, steps, delta, grid=grid)
        diff = abs(eps_dp - eps_op)
        ok = diff <= tol
        if not ok:
            failures.append((s, eps_op, eps_dp, diff))
        print(f"{s:>8.3f} {eps_op:>12.5f} {eps_dp:>13.5f} {diff:>10.5f} {'OK' if ok else 'FAIL':>8}")
    print()
    if failures:
        raise AssertionError(
            f"Accountants diverge beyond tol={tol} for sigmas: "
            + ", ".join(f"{s}" for s, *_ in failures)
        )


def validate_accountants_q1(sigma_grid, steps, delta, grid=1e-4, tol=0.1):
    """
    Three-way cross-check for q=1 (full batch, no subsampling):
      A. Opacus PRV with sample_rate=1
      B. Dominating-pair PLD with q=1 (reduces to N(1,sigma^2) vs N(0,sigma^2))
      C. Analytic Gaussian-composition formula (ground truth)

    All three must agree within `tol`.  This is the trust anchor for the q=1
    (no-subsampling) mode which is our only configuration with a valid node-DP
    guarantee today.
    """
    print(f"\n=== Accountant validation (q=1, Gaussian composition): "
          f"steps={steps}, delta={delta:g} ===")
    print(f"{'sigma':>8} {'eps_opacus':>12} {'eps_dompair':>13} "
          f"{'eps_analytic':>14} {'max_diff':>10} {'status':>8}")
    print("-" * 71)
    failures = []
    for s in sigma_grid:
        eps_op = opacus_prv_epsilon(s, 1.0, steps, delta)
        eps_dp = dompair_epsilon(1.0, s, steps, delta, grid=grid)
        eps_an = gaussian_composition_epsilon(s, steps, delta)
        max_diff = max(abs(eps_dp - eps_op), abs(eps_dp - eps_an), abs(eps_op - eps_an))
        ok = max_diff <= tol
        if not ok:
            failures.append((s, eps_op, eps_dp, eps_an, max_diff))
        print(f"{s:>8.3f} {eps_op:>12.5f} {eps_dp:>13.5f} "
              f"{eps_an:>14.5f} {max_diff:>10.5f} {'OK' if ok else 'FAIL':>8}")
    print()
    if failures:
        raise AssertionError(
            f"q=1 accountants diverge beyond tol={tol} for sigmas: "
            + ", ".join(f"{s}" for s, *_ in failures)
        )
