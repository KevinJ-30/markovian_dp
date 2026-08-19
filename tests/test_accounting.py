"""
Tests for the SparseGNN dominating-pair accountants and degree capping.

Two families are covered (see src/sparse/accounting.py):

  * the substitution pairs — Theorem 6.4 (direction='in', Algorithm 5) and
    Theorem 1/2 (direction='out', Algorithm 2/4);
  * the Theorem 4.5 insertion/removal marked pair, which is stated for
    out-expansion only.  Its PGF is the Binomial family of Lemma 17:
    pi = law of sum_d Binomial(n_d, a_d).

Composition and epsilon(delta) are delegated to Google dp_accounting; the
correctness anchors here cross-check the degenerate cases against Opacus,
which computes them through an independent code path:

  * Theorem 4.5 at r=0 (or p2=0) is a plain Poisson-subsampled Gaussian.
  * The substitution pair at p1=1, r=0 collapses to N(-2, s^2) vs N(+2, s^2),
    i.e. an unsubsampled Gaussian mechanism at sensitivity 4C.  (The extremal
    pair places the two families at -2k and +2k; the cross-family constraint
    is deliberately dropped in the paper's final step, so the separation is 4k
    rather than 2k — the factor-2-in-sigma looseness of Corollary 6.5.)

Ours must land at or above the reference in both, since the discretized pair
fed to dp_accounting is constructed to dominate the analytic one.
"""

import math

import pytest
import torch

from src.sparse.accounting import (
    naive_opacus_epsilon, shell_sizes, sparsegnn_mixture_weights,
    sparsegnn_substitution_epsilon, sparsegnn_thm4_epsilon, thm4_fiber_weights,
)
from src.sparse.sparse_expand import (
    cap_degrees, cap_degrees_undirected, edge_set_is_symmetric, max_degrees,
)

pytest.importorskip("dp_accounting")


# ── Theorem 4.5 fiber weights (Binomial, Lemma 17) ────────────────────────────

def test_fiber_weights_shape_and_sum():
    pi = thm4_fiber_weights(0.5, 0.5, 2, 4)
    assert len(pi) == 1 + 4 + 16          # N_com_r + 1
    assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-12)
    assert (pi >= 0).all()


def test_fiber_weights_r0_is_delta_at_zero():
    pi = thm4_fiber_weights(0.3, 0.7, 0, 8)
    assert pi.tolist() == [1.0]


def test_fiber_weights_p2_zero_concentrates_at_zero():
    pi = thm4_fiber_weights(0.5, 0.0, 3, 8)
    assert math.isclose(float(pi[0]), 1.0, abs_tol=1e-12)


def test_fiber_weights_r1_closed_form():
    """r=1: each of the K_in slots activates independently with a_1 = p1*p2,
    so pi is the Binomial(K_in, p1*p2) pmf (Lemma 17), NOT a point mass of
    a_1 at j = K_in."""
    p1, p2, K = 0.4, 0.25, 5
    a = p1 * p2
    pi = thm4_fiber_weights(p1, p2, 1, K)
    assert len(pi) == K + 1
    for j in range(K + 1):
        expected = math.comb(K, j) * a ** j * (1 - a) ** (K - j)
        assert math.isclose(float(pi[j]), expected, rel_tol=1e-9), j


def test_fiber_weights_mean_is_sum_of_shell_means():
    """E[J] = sum_d n_d a_d with n_d = K_in^d and a_d = p1 * qx_d."""
    p1, p2, r, K = 0.2, 0.5, 2, 3
    pi = thm4_fiber_weights(p1, p2, r, K)
    qx = [1.0 - math.prod((1.0 - p2 ** l) ** (K ** (l - 1))
                          for l in range(d, r + 1))
          for d in range(1, r + 1)]
    expected = sum((K ** d) * p1 * qx[d - 1] for d in range(1, r + 1))
    got = float(sum(j * w for j, w in enumerate(pi)))
    assert math.isclose(got, expected, rel_tol=1e-9)


# ── Theorem 4.5 epsilon: degenerate-case cross-check vs Opacus ────────────────

@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_r0_matches_opacus_subsampled_gaussian(sigma):
    """r=0 keeps only the root record -> plain Poisson-subsampled Gaussian."""
    pytest.importorskip("opacus")
    p1, T, delta = 0.5, 200, 1e-5
    eps4 = sparsegnn_thm4_epsilon(p1=p1, p2=0.5, r=0, K_in=4, sigma=sigma,
                                  steps=T, delta=delta)
    eps_op = naive_opacus_epsilon(sigma, p1, T, delta, mechanism="prv")
    # The discretized pair dominates the analytic one, so ours must be >=
    # Opacus (up to its own numerics), and close.
    assert eps4 >= eps_op - 1e-3
    assert eps4 - eps_op < 0.5


def test_p2_zero_matches_r0():
    """p2=0 drops every edge, so any r behaves like r=0."""
    eps_r0 = sparsegnn_thm4_epsilon(p1=0.3, p2=0.9, r=0, K_in=6, sigma=2.0,
                                    steps=100, delta=1e-5)
    eps_p20 = sparsegnn_thm4_epsilon(p1=0.3, p2=0.0, r=3, K_in=6, sigma=2.0,
                                     steps=100, delta=1e-5)
    assert math.isclose(eps_r0, eps_p20, rel_tol=1e-6)


def test_epsilon_monotone_in_p2_and_sigma():
    def eps(p2, sigma):
        return sparsegnn_thm4_epsilon(p1=0.1, p2=p2, r=1, K_in=5, sigma=sigma,
                                      steps=100, delta=1e-5)
    eps_dense = eps(1.0, 5.0)
    eps_sparse = eps(0.1, 5.0)
    eps_noisier = eps(1.0, 10.0)
    assert eps_sparse < eps_dense          # amplification by sparsification
    assert eps_noisier < eps_dense         # more noise, less epsilon


def test_high_fiber_config_is_finite():
    """With the Binomial weights, mass on high marks decays combinatorially,
    so even an aggressive config composes to a finite (if large) epsilon.
    (The pre-fix single-Bernoulli weights put mass 0.25 directly on j = 10
    and drove this configuration to infinity.)"""
    eps = sparsegnn_thm4_epsilon(p1=0.5, p2=0.5, r=1, K_in=10, sigma=1.0,
                                 steps=200, delta=1e-5, grid=1e-3)
    assert math.isfinite(eps)
    eps_r0 = sparsegnn_thm4_epsilon(p1=0.5, p2=0.5, r=0, K_in=10, sigma=1.0,
                                    steps=200, delta=1e-5, grid=1e-3)
    assert eps > eps_r0                    # the graph term costs something


# ══ substitution pairs: Theorem 6.4 (in) and Theorem 1/2 (out) ════════════════

def test_shell_sizes_pick_the_right_degree_bound():
    # in-expansion: a substituted vertex reaches roots in its FORWARD
    # neighbourhood, whose d-th shell is bounded by K_out^d (Eq. 44).
    assert shell_sizes(2, K_in=3, K_out=5, direction='in') == [1, 5, 25]
    assert shell_sizes(2, K_in=3, K_out=5, direction='out') == [1, 3, 9]


def test_mixture_weights_are_a_distribution_of_the_right_length():
    for direction in ('in', 'out'):
        pi = sparsegnn_mixture_weights(0.3, 0.5, 2, 4, 3, direction=direction)
        Nr = sum(shell_sizes(2, 4, 3, direction=direction))
        assert len(pi) == Nr + 1
        assert math.isclose(float(pi.sum()), 1.0, abs_tol=1e-12)
        assert (pi >= 0).all()


def test_mixture_weights_mean_matches_paper_expectation():
    """E[J] = p1 (1 + sum_d n_d q_d), Eq. (49)."""
    p1, p2, r, K = 0.2, 0.5, 2, 4
    pi = sparsegnn_mixture_weights(p1, p2, r, K, K, direction='in')
    n = shell_sizes(r, K, K, direction='in')
    q = [1.0] + [1.0 - math.prod((1.0 - p2 ** l) ** (K ** (l - 1))
                                 for l in range(d, r + 1))
                 for d in range(1, r + 1)]
    expected = p1 * sum(n_d * q_d for n_d, q_d in zip(n, q))
    got = float(sum(k * w for k, w in enumerate(pi)))
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_mixture_weights_p2_zero_is_root_only():
    """p2=0 kills every path, so only the d=0 slot can activate."""
    p1 = 0.3
    pi = sparsegnn_mixture_weights(p1, 0.0, 3, 5, direction='in')
    assert math.isclose(float(pi[0]), 1 - p1, rel_tol=1e-12)
    assert math.isclose(float(pi[1]), p1, rel_tol=1e-12)
    assert float(pi[2:].sum()) == 0.0


@pytest.mark.parametrize("sigma", [5.0, 10.0])
def test_p1_one_r0_matches_gaussian_at_sensitivity_four(sigma):
    """p1=1, r=0 => P=N(-2,s^2), Q=N(+2,s^2): a sensitivity-4C Gaussian."""
    pytest.importorskip("opacus")
    T, delta = 100, 1e-6
    ours = sparsegnn_substitution_epsilon(p1=1.0, p2=0.0, r=0, K_in=5,
                                          sigma=sigma, steps=T, delta=delta)
    ref = naive_opacus_epsilon(sigma / 4.0, 1.0, T, delta, mechanism="prv")
    assert ours >= ref - 1e-3          # our pair dominates the analytic one
    assert ours - ref < 0.05 * ref     # and tight


def test_substitution_epsilon_p2_zero_matches_r0():
    kw = dict(p1=0.3, K_in=6, sigma=5.0, steps=100, delta=1e-6)
    assert math.isclose(
        sparsegnn_substitution_epsilon(p2=0.9, r=0, **kw),
        sparsegnn_substitution_epsilon(p2=0.0, r=3, **kw),
        rel_tol=1e-6)


def test_substitution_epsilon_monotone_in_p2_and_sigma():
    def eps(p2, sigma):
        return sparsegnn_substitution_epsilon(
            p1=0.1, p2=p2, r=1, K_in=5, sigma=sigma, steps=100, delta=1e-6)
    dense = eps(1.0, 5.0)
    assert eps(0.1, 5.0) < dense       # amplification by sparsification
    assert eps(1.0, 10.0) < dense      # more noise, less epsilon


def test_orientation_follows_the_binding_degree_bound():
    """in-expansion pays K_out, out-expansion pays K_in (Eq. 44).

    With K_out >> K_in the corrected orientation is the more expensive one, and
    the ordering flips when the degree asymmetry does.
    """
    kw = dict(p1=0.005, p2=0.5, r=1, sigma=5.0, steps=500, delta=1e-6)
    wide_out = dict(K_in=2, K_out=10)
    assert (sparsegnn_substitution_epsilon(direction='in', **kw, **wide_out) >
            sparsegnn_substitution_epsilon(direction='out', **kw, **wide_out))
    wide_in = dict(K_in=10, K_out=2)
    assert (sparsegnn_substitution_epsilon(direction='in', **kw, **wide_in) <
            sparsegnn_substitution_epsilon(direction='out', **kw, **wide_in))


def test_symmetric_degrees_make_the_two_orientations_agree():
    kw = dict(p1=0.005, p2=0.5, r=2, K_in=5, K_out=5, sigma=5.0,
              steps=200, delta=1e-6)
    assert math.isclose(sparsegnn_substitution_epsilon(direction='in', **kw),
                        sparsegnn_substitution_epsilon(direction='out', **kw),
                        rel_tol=1e-12)


# ── degree capping ─────────────────────────────────────────────────────────────

def _random_graph(n=200, m=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    return torch.stack([src, dst]), n


def _random_undirected_graph(n=150, m=1200, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n, (m,), generator=g)
    b = torch.randint(0, n, (m,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    ei = torch.stack([torch.cat([a, b]), torch.cat([b, a])])
    return torch.unique(ei, dim=1), n


def test_cap_degrees_respects_bounds_and_subset():
    ei, n = _random_graph()
    gen = torch.Generator().manual_seed(1)
    capped = cap_degrees(ei, n, K_in=7, K_out=9, generator=gen)
    mi, mo = max_degrees(capped, n)
    assert mi <= 7 and mo <= 9
    before = set(map(tuple, ei.t().tolist()))
    assert all(tuple(e) in before for e in capped.t().tolist())


def test_cap_degrees_noop_when_bounds_loose():
    ei, n = _random_graph()
    mi, mo = max_degrees(ei, n)
    capped = cap_degrees(ei, n, K_in=mi, K_out=mo,
                         generator=torch.Generator().manual_seed(0))
    assert capped.size(1) == ei.size(1)


def test_cap_degrees_deterministic_under_seed():
    ei, n = _random_graph()
    a = cap_degrees(ei, n, K_in=5, K_out=5,
                    generator=torch.Generator().manual_seed(3))
    b = cap_degrees(ei, n, K_in=5, K_out=5,
                    generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_cap_degrees_breaks_symmetry_but_undirected_variant_keeps_it():
    """The motivation for cap_degrees_undirected: independent per-direction
    capping of an undirected graph leaves many arcs without their reverse."""
    ei, n = _random_undirected_graph()
    assert edge_set_is_symmetric(ei, n)

    directed = cap_degrees(ei, n, K_in=5, K_out=5,
                           generator=torch.Generator().manual_seed(4))
    assert not edge_set_is_symmetric(directed, n)

    undirected = cap_degrees_undirected(ei, n, 5,
                                        generator=torch.Generator().manual_seed(4))
    assert edge_set_is_symmetric(undirected, n)


def test_cap_degrees_undirected_bounds_subset_and_determinism():
    ei, n = _random_undirected_graph(seed=7)
    gen = lambda: torch.Generator().manual_seed(11)  # noqa: E731
    capped = cap_degrees_undirected(ei, n, 4, generator=gen())
    mi, mo = max_degrees(capped, n)
    assert mi <= 4 and mo <= 4
    before = set(map(tuple, ei.t().tolist()))
    assert all(tuple(e) in before for e in capped.t().tolist())
    assert torch.equal(capped, cap_degrees_undirected(ei, n, 4, generator=gen()))


def test_edge_set_is_symmetric_detects_asymmetry():
    ei = torch.tensor([[0, 1, 2], [1, 0, 0]])
    assert not edge_set_is_symmetric(ei, 3)
    ei = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
    assert edge_set_is_symmetric(ei, 3)
