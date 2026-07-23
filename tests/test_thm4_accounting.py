"""
Tests for the Theorem 4 (insertion/removal) accountant and degree capping.

Key correctness anchor: at r=0 (or p2=0) SparseGNN degenerates to plain
Poisson-subsampled Gaussian DP-SGD, whose epsilon Opacus PRV computes
independently — the Theorem 4 accountant must agree (from above, since all
of our discretization choices are pessimistic).
"""

import math

import pytest
import torch

from src.sparse.accounting import (
    sparsegnn_thm4_epsilon, sparsegnn_thm4_pair, thm4_fiber_weights,
)
from src.sparse.sparse_expand import cap_degrees, max_degrees


# ── fiber weights ──────────────────────────────────────────────────────────────

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
    # r=1: q×_1 = p2, a_1 = p1*p2, mass a_1 at j = K_in, rest at j = 0.
    p1, p2, K = 0.4, 0.25, 5
    pi = thm4_fiber_weights(p1, p2, 1, K)
    assert math.isclose(float(pi[0]), 1 - p1 * p2, rel_tol=1e-12)
    assert math.isclose(float(pi[K]), p1 * p2, rel_tol=1e-12)
    assert float(pi[1:K].sum()) == 0.0


# ── discretized pair ───────────────────────────────────────────────────────────

def test_pair_masses_sum_to_one():
    p_atoms, q_atoms = sparsegnn_thm4_pair(0.2, 0.3, 1, 4, sigma=2.0)
    assert math.isclose(float(p_atoms.sum()), 1.0, abs_tol=1e-9)
    assert math.isclose(float(q_atoms.sum()), 1.0, abs_tol=1e-9)
    assert (p_atoms >= 0).all() and (q_atoms >= 0).all()


# ── epsilon: degenerate-case cross-check vs Opacus ─────────────────────────────

@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_r0_matches_opacus_subsampled_gaussian(sigma):
    """r=0 keeps only the root record -> plain Poisson-subsampled Gaussian."""
    opacus = pytest.importorskip("opacus")  # noqa: F841
    from src.sparse.accounting import _load_pld_module
    p1, T, delta = 0.5, 200, 1e-5
    eps4 = sparsegnn_thm4_epsilon(p1=p1, p2=0.5, r=0, K_in=4, sigma=sigma,
                                  steps=T, delta=delta)
    eps_op = _load_pld_module().opacus_epsilon(sigma, p1, T, delta,
                                               mechanism="prv")
    # Pessimistic discretization: ours must be >= Opacus, and close
    # (slack <= grid * T = 0.2 plus discretization crumbs).
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
                                      steps=100, delta=1e-5, loss_cap=100.0)
    eps_dense = eps(1.0, 5.0)
    eps_sparse = eps(0.1, 5.0)
    eps_noisier = eps(1.0, 10.0)
    assert eps_sparse < eps_dense          # amplification by sparsification
    assert eps_noisier < eps_dense         # more noise, less epsilon


def test_epsilon_inf_when_loss_exceeds_cap():
    """High-j fibers beyond the cap have real mass -> honest inf, not crash."""
    eps = sparsegnn_thm4_epsilon(p1=0.5, p2=0.5, r=1, K_in=10, sigma=1.0,
                                 steps=200, delta=1e-5, loss_cap=50.0)
    assert math.isinf(eps)


# ── degree capping ─────────────────────────────────────────────────────────────

def _random_graph(n=200, m=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    return torch.stack([src, dst]), n


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
