"""
Numerical verification of the dominating pair against the ACTUAL mechanism.

Every other test in this repo assumes Theorem 6.4's pair is correct and checks
only that we build and compose it faithfully.  This file checks the theorem
itself: it constructs a real neighbouring graph pair, computes the exact output
distribution of one SparseGNN update on each, evaluates the hockey-stick
divergence H_alpha(M_g || M_g') numerically, and asserts

    H_alpha(M_g || M_g')  <=  H_alpha(P || Q)      for every alpha,

which is precisely the claim of Theorem 6.4 (Eq. 48).

Construction
------------
The paper names its own worst case (Section 3.0.1): a star with the substituted
vertex s at the centre and all paths independent.  Orient it for in-expansion
(Algorithm 5): s -> v_1, ..., s -> v_m, so out-degree K_out = m, and every leaf
has in-degree 1.  A root reaches s in one hop exactly when its single in-edge
survives, which happens with probability p2 — independently across roots, so the
"independent paths" condition holds exactly rather than approximately.

Adversarial base mechanism: g0(H) = +C when s in H and 0 otherwise under g, and
-C when s in H under g' (a feature substitution at s flips the sign).  This is
the largest per-record swing the assumption ||g0||<=C permits, so it maximizes
the divergence over all admissible g0.

The number of contributing roots is then
    J = Bernoulli(p1)                    [the root s itself, always sees s]
        + Binomial(K_out, p1 * p2)       [each leaf: sampled AND edge kept]
which is exactly the law pi of Eq. (46) at r=1, where q_1 = p2 and n_1 = K_out.
The true mechanism is therefore
    M_g  = sum_k pi_k N(+k, sigma^2),   M_g' = sum_k pi_k N(-k, sigma^2)
in units of C, while the theorem's pair sits at -2k and +2k.  The factor of two
in the spacing is the known looseness of Corollary 6.5 (noise variance off by at
most a factor of 2), so these tests also quantify that gap rather than merely
asserting domination.
"""

import math

import numpy as np
import pytest
import torch

from src.sparse.accounting import sparsegnn_mixture_weights
from src.sparse.sparse_expand import build_adjacency, sample_roots, sparse_expand

pytest.importorskip("scipy")


# ── exact distributions ───────────────────────────────────────────────────────

def _star_edges(K_out):
    """s = node 0, leaves 1..K_out, arcs s -> v (in-expansion reaches s in 1 hop)."""
    src = torch.zeros(K_out, dtype=torch.long)
    dst = torch.arange(1, K_out + 1, dtype=torch.long)
    return torch.stack([src, dst])


def _true_contribution_law(p1, p2, K_out):
    """Exact law of J = #roots whose rooted subgraph contains s.

    Root s contributes iff it is sampled (its own subgraph always contains it).
    Leaf v_i contributes iff it is sampled AND its single in-arc survives.
    All draws independent, so J = Bern(p1) + Binom(K_out, p1*p2).
    """
    from scipy.stats import binom
    leaf = binom.pmf(np.arange(K_out + 1), K_out, p1 * p2)
    return np.convolve(np.array([1.0 - p1, p1]), leaf)


def _mixture_pdf(x, means, weights, sigma):
    out = np.zeros_like(x)
    for mu, w in zip(means, weights):
        out += w * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2.0 * math.pi))
    return out


def _hockey_stick(alpha, means_p, w_p, means_q, w_q, sigma, span=60.0, n=800_001):
    """H_alpha(P||Q) = integral of (p - alpha*q)_+ , by fine quadrature."""
    lo = min(min(means_p), min(means_q)) - span
    hi = max(max(means_p), max(means_q)) + span
    x = np.linspace(lo, hi, n)
    p = _mixture_pdf(x, means_p, w_p, sigma)
    q = _mixture_pdf(x, means_q, w_q, sigma)
    integrand = np.clip(p - alpha * q, 0.0, None)
    trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy >=2.0 / <2.0
    return float(trapz(integrand, x))


# ── 1. the sampling model matches the real implementation ────────────────────

@pytest.mark.parametrize("p1,p2,K_out", [(0.5, 0.5, 4), (0.3, 0.8, 3)])
def test_real_expansion_reproduces_the_theorem_sampling_law(p1, p2, K_out):
    """Monte-Carlo the REAL sample_roots/sparse_expand on the star and compare
    the distribution of |{roots whose subgraph contains s}| to the analytic law.

    This is what links the theorem's abstraction to the code that actually runs.
    """
    n_nodes = K_out + 1
    adj = build_adjacency(_star_edges(K_out), n_nodes, direction='in')
    gen = torch.Generator().manual_seed(0)
    trials = 30_000
    counts = np.zeros(K_out + 2)
    for _ in range(trials):
        roots = sample_roots(n_nodes, p1, generator=gen)
        j = 0
        for v in roots.tolist():
            sub = sparse_expand(adj, int(v), p2, 1, generator=gen, direction='in')
            if 0 in sub.nodes.tolist():          # node 0 is s
                j += 1
        counts[j] += 1
    empirical = counts / trials
    exact = _true_contribution_law(p1, p2, K_out)
    assert np.abs(empirical - exact).max() < 0.01

    # ... and the accountant's pi is that same law (r=1, q_1 = p2, n_1 = K_out).
    pi = sparsegnn_mixture_weights(p1, p2, r=1, K_in=1, K_out=K_out,
                                   direction='in')
    assert np.abs(np.asarray(pi) - exact).max() < 1e-9


# ── 2. the theorem's pair dominates the true mechanism ───────────────────────

@pytest.mark.parametrize("p1,p2,K_out,sigma", [
    (0.5, 0.5, 4, 1.0),
    (0.2, 1.0, 3, 1.0),
    (0.8, 0.3, 5, 2.0),
])
@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 5.0, 20.0])
def test_dominating_pair_upper_bounds_the_true_mechanism(p1, p2, K_out, sigma,
                                                         alpha):
    """H_alpha(M_g || M_g') <= H_alpha(P || Q): Theorem 6.4, Eq. (48)."""
    pi = np.asarray(sparsegnn_mixture_weights(p1, p2, r=1, K_in=1, K_out=K_out,
                                              direction='in'))
    ks = np.arange(len(pi))

    # True mechanism, in units of C: adversarial g0 puts the two graphs at
    # +k and -k for k contributing roots.
    true = _hockey_stick(alpha, -ks, pi, +ks, pi, sigma)
    # Theorem's pair: means spaced by 2 (Eq. 47).
    claimed = _hockey_stick(alpha, -2.0 * ks, pi, +2.0 * ks, pi, sigma)

    assert true <= claimed + 1e-9, (
        f"THEOREM VIOLATED: true H_{alpha}={true:.6g} > claimed {claimed:.6g}")


def test_the_slack_is_the_corollary_6_5_factor_of_two():
    """The pair's looseness should be exactly 'sigma off by 2', not more.

    Corollary 6.5 brackets the truth between spacing-1 at sigma and spacing-2 at
    sigma (equivalently spacing-1 at sigma/2).  Check that the claimed bound at
    sigma equals the true divergence at sigma/2, so the slack is understood
    rather than mysterious.
    """
    p1, p2, K_out, sigma, alpha = 0.5, 0.5, 4, 1.0, 2.0
    pi = np.asarray(sparsegnn_mixture_weights(p1, p2, r=1, K_in=1, K_out=K_out,
                                              direction='in'))
    ks = np.arange(len(pi))
    claimed = _hockey_stick(alpha, -2.0 * ks, pi, +2.0 * ks, pi, sigma)
    true_at_half_sigma = _hockey_stick(alpha, -ks, pi, +ks, pi, sigma / 2.0)
    assert claimed == pytest.approx(true_at_half_sigma, rel=1e-6)


# ── 3. the bound is not vacuous ──────────────────────────────────────────────

def test_bound_is_tight_enough_to_be_meaningful():
    """The claimed divergence must stay well below 1 (the trivial bound) and
    above the true one, at a configuration we actually run."""
    pi = np.asarray(sparsegnn_mixture_weights(0.01, 0.1, r=1, K_in=5, K_out=5,
                                              direction='in'))
    ks = np.arange(len(pi))
    for alpha in (1.0, 2.0):
        true = _hockey_stick(alpha, -ks, pi, +ks, pi, 5.0)
        claimed = _hockey_stick(alpha, -2.0 * ks, pi, +2.0 * ks, pi, 5.0)
        assert true <= claimed + 1e-12
        assert claimed < 1.0
