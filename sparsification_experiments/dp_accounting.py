"""
Privacy accounting for sparsification_experiments.

Two paths:
  A. Opacus PRV (preferred) / RDP (fallback): standard subsampled-Gaussian.
  B. Google dp_accounting PLD: the same subsampled Gaussian via
     `privacy_loss_distribution.from_gaussian_mechanism` (connect-the-dots,
     pessimistic, ADD_OR_REMOVE_ONE adjacency) — the reference implementation,
     replacing the previous hand-rolled dominating-pair discretization.

Both paths consume (sigma, q) where sigma is the SENSITIVITY-NORMALIZED noise
multiplier (noise std = sigma * Delta, Delta = node sensitivity from
sparsify.node_sensitivity).  Folding Delta into sigma reduces the problem to a
standard unit-sensitivity Gaussian, which is exactly what both accountants
expect.

For the SparseGNN mechanism itself (parameterized by p1, p2, r, K_in, K_out
rather than a single sample rate), use `src.sparse.compute_epsilon` /
`src.sparse.accounting` — those compose the paper's dominating pairs through
dp_accounting directly.
"""

import math


# ── A. Opacus path ────────────────────────────────────────────────────────────

def opacus_epsilon(noise_multiplier, sample_rate, steps, delta, mechanism="rdp"):
    try:
        from opacus.accountants import create_accountant
    except ImportError as e:
        raise ImportError(
            "opacus is required for the opacus accountants: pip install opacus"
        ) from e
    accountant = create_accountant(mechanism=mechanism)
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return accountant.get_epsilon(delta=delta)


def opacus_prv_epsilon(sigma, q, steps, delta):
    """Epsilon via Opacus PRV accountant (RDP fallback)."""
    try:
        return opacus_epsilon(sigma, q, steps, delta, mechanism="prv")
    except Exception:
        return opacus_epsilon(sigma, q, steps, delta, mechanism="rdp")


# ── B. Google dp_accounting PLD path ─────────────────────────────────────────

def dompair_epsilon(q, sigma, steps, delta, grid=1e-4):
    """Epsilon for `steps` compositions of the Poisson-subsampled Gaussian.

    Delegates to dp_accounting's connect-the-dots PLD for the subsampled
    Gaussian (pessimistic estimate, ADD_OR_REMOVE_ONE adjacency, both
    directions handled internally).  `grid` is the loss discretization
    interval.
    """
    from dp_accounting.pld import privacy_loss_distribution as PLD
    pld = PLD.from_gaussian_mechanism(
        standard_deviation=sigma, sensitivity=1.0, pessimistic_estimate=True,
        value_discretization_interval=grid, sampling_prob=q)
    return pld.self_compose(steps).get_epsilon_for_delta(delta)


def make_novel_mechanism_dominating_pair(q, sigma, **kwargs):
    """Removed: the atom-array contract this hook returned no longer exists.

    The SparseGNN dominating pairs are composed directly through
    dp_accounting — call `src.sparse.accounting.sparsegnn_substitution_epsilon`
    (Theorem 6.4 / Theorem 1-2) or `sparsegnn_thm4_epsilon` (Theorem 4.5), or
    run `python -m src.sparse.compute_epsilon --csv <results.csv>`.
    """
    raise NotImplementedError(make_novel_mechanism_dominating_pair.__doc__)


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
    Cross-check Opacus PRV vs dp_accounting PLD across sigma_grid (q < 1).

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
      B. dp_accounting PLD with sampling_prob=1
      C. Analytic Gaussian-composition formula (ground truth)

    All three must agree within `tol`.  This is the trust anchor for the q=1
    (no-subsampling) mode.
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
