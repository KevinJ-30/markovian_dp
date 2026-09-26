"""DP-GNN bounded-sensitivity and multi-term RDP accounting."""

from __future__ import annotations


def max_terms_per_node(max_degree: int, radius: int = 1) -> int:
    """Bound root losses influenced by one node on the sampled topology.

    Arcs point from a root to its dependencies. The sampler caps incoming
    degree, so at most K**hop roots can reach a node in ``hop`` steps. The
    union over distances 0..radius is bounded by the geometric sum (cycles
    only reduce it). This is not an outgoing fanout or a neighborhood-size
    bound, nor a privacy guarantee for the data-dependent preprocessing of
    an arbitrary raw topology.
    """
    if type(max_degree) is not int or max_degree < 1:
        raise ValueError("max_degree must be a positive integer")
    if type(radius) is not int or radius < 1:
        raise ValueError("radius must be a positive integer")
    return sum(max_degree ** hop for hop in range(radius + 1))


def multiterm_dpsgd_epsilon(*, steps: int, noise_multiplier: float,
                             delta: float, num_samples: int,
                             batch_size: int, max_terms: int) -> float:
    """Hypergeometric multi-term RDP for uniform batches without replacement.

    ``noise_multiplier`` scales the sum sensitivity ``2 * max_terms * clip``,
    not just the per-root clipping bound used by Opacus.
    """
    import numpy as np
    import scipy.special
    import scipy.stats

    if steps < 1 or num_samples < 1 or batch_size < 1:
        raise ValueError("steps, num_samples, and batch_size must be positive")
    if batch_size > num_samples:
        raise ValueError("batch_size must not exceed num_samples")
    if max_terms < 1 or max_terms > num_samples:
        raise ValueError("max_terms must be between 1 and num_samples")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if noise_multiplier < 1e-20:
        return float("inf")
    from dp_accounting import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant, compute_epsilon

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
