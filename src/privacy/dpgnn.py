"""DP-GNN bounded-sensitivity and multi-term RDP accounting."""

from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats

def max_terms_per_node(max_degree: int) -> int:
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    return max_degree + 1


def base_sensitivity(max_degree: int) -> float:
    return float(2 * max_terms_per_node(max_degree))


def multiterm_dpsgd_epsilon(*, steps: int, noise_multiplier: float,
                             delta: float, num_samples: int,
                             batch_size: int, max_terms: int) -> float:
    """Port of DP-GNN's hypergeometric multi-term RDP accountant."""
    if steps < 1 or num_samples < 1 or batch_size < 1:
        raise ValueError("steps, num_samples, and batch_size must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if noise_multiplier < 1e-20:
        return float("inf")
    from dp_accounting import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant, compute_epsilon

    batch_size = min(batch_size, num_samples)
    max_terms = min(max_terms, num_samples)
    terms = np.arange(max_terms + 1)
    terms_logprobs = scipy.stats.hypergeom(
        num_samples, max_terms, batch_size).logpmf(terms)
    orders = np.arange(1, 10, 0.1)[1:]
    accountant = RdpAccountant(orders)
    accountant.compose(GaussianDpEvent(noise_multiplier))
    unamplified = np.asarray(accountant._rdp)  # DP-Accounting has no public RDP accessor.
    amplified = []
    for order, rdp in zip(orders, unamplified):
        beta = rdp * (order - 1)
        log_factors = beta * np.square(terms / max_terms)
        amplified.append(scipy.special.logsumexp(terms_logprobs + log_factors) /
                         (order - 1))
    amplified = np.asarray(amplified)
    if not np.all(unamplified * (batch_size / num_samples) ** 2 <= amplified + 1e-6):
        raise ValueError("DP-GNN multi-term RDP lower bound was violated")
    return float(compute_epsilon(orders, amplified * steps, delta)[0])
