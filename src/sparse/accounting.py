"""Theorem 5.4 substitution accounting for SparseGNN.

For in-expansion, let ``pi`` be the law of the number of affected sampled
rooted subgraphs.  The one-step mechanism is dominated by

    P = sum_k pi[k] N(-2k, sigma^2)
    Q = sum_k pi[k] N(+2k, sigma^2).

Here ``sigma`` is the Opacus noise multiplier; training adds Gaussian noise
with standard deviation ``sigma*C`` after clipping each contribution at ``C``.
The analytic pair is handed to Google ``dp_accounting`` for pessimistic
connect-the-dots discretization, composition, and epsilon(delta).  Training
orientation is intentionally not validated here; this module always constructs
the in-expansion shell law.
"""

from dataclasses import asdict, dataclass
import math
from typing import List, Optional, Sequence

import numpy as np

from .privacy_loss import DoubleMixtureGaussianPrivacyLoss


def _q_products(p2: float, r: int, K: int) -> List[float]:
    """Path-retention bounds q_0..q_r from Theorem 5.4."""
    q = [1.0]
    for d in range(1, r + 1):
        if p2 >= 1.0:
            q.append(1.0)
            continue
        log_keep = sum((K ** (level - 1)) * math.log1p(-(p2 ** level))
                       for level in range(d, r + 1))
        q.append(1.0 - math.exp(log_keep))
    return q


def _binom_pmf(n: int, p: float) -> np.ndarray:
    """Binomial(n, p) PMF, including exact endpoint behavior."""
    if p <= 0.0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1.0:
        out = np.zeros(n + 1)
        out[n] = 1.0
        return out
    from scipy.stats import binom
    pmf = np.clip(binom.pmf(np.arange(n + 1), n, p), 0.0, None)
    return pmf / pmf.sum()


def shell_sizes(r: int, K_out: int, union_safe: bool = True) -> List[int]:
    """In-expansion shell bounds n_0..n_r.

    The default factor two is the union-graph correction: only the substituted
    node's first step can double, hence n_d = 2*K_out**d rather than (2K)^d.
    """
    if r < 0 or K_out < 1:
        raise ValueError("need r >= 0 and K_out >= 1")
    factor = 2 if union_safe else 1
    return [1] + [factor * K_out ** d for d in range(1, r + 1)]


def sparsegnn_mixture_weights(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    K_out: Optional[int] = None,
    union_safe: bool = True,
) -> np.ndarray:
    """Theorem 5.4 mixture weights for in-expansion."""
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("p1 and p2 must lie in [0, 1]")
    if r < 0 or K_in < 1:
        raise ValueError("need r >= 0 and K_in >= 1")
    K_out = K_in if K_out is None else K_out
    if K_out < 1:
        raise ValueError("K_out must be at least one")
    q = _q_products(p2, r, min(K_in, K_out))
    sizes = shell_sizes(r, K_out, union_safe=union_safe)
    weights = np.array([1.0])
    for n_d, q_d in zip(sizes, q):
        weights = np.convolve(weights, _binom_pmf(n_d, p1 * q_d))
    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum()


def mixture_gaussian_pld(
    weights: Sequence[float], sigma: float, grid: float = 1e-4,
):
    """Build a pessimistic dp_accounting PLD from integer-mark weights.

    Clipping and noise both scale by ``C`` during training, so normalization by
    ``C`` leaves mixture centers ``+-2k`` and Gaussian standard deviation
    ``sigma``.
    """
    from dp_accounting.pld.privacy_loss_distribution import (
        PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not len(weights):
        raise ValueError("weights must be a non-empty one-dimensional distribution")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("weights must sum to one")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError("grid must be finite and positive")

    support = np.flatnonzero(weights > 0)
    if support.size == 1 and int(support[0]) == 0:
        return PrivacyLossDistribution.identity(grid)
    probabilities = weights[support]
    sensitivities = 2.0 * support.astype(float)
    privacy_loss = DoubleMixtureGaussianPrivacyLoss(
        standard_deviation=sigma,
        sensitivities_upper=sensitivities,
        sensitivities_lower=sensitivities,
        sampling_probs_upper=probabilities,
        sampling_probs_lower=probabilities,
        pessimistic_estimate=True,
    )
    pmf = _create_pld_pmf_from_additive_noise(
        privacy_loss,
        pessimistic_estimate=True,
        value_discretization_interval=grid,
        use_connect_dots=True,
    )
    return PrivacyLossDistribution(pmf)


def _compose_schedule(base_pld, steps, delta: float):
    out = {}
    current = None
    last = 0
    for step in sorted({int(value) for value in steps}):
        if step < 1:
            raise ValueError(f"checkpoints must be >= 1, got {step}")
        gap = step - last
        if gap:
            block = base_pld if gap == 1 else base_pld.self_compose(gap)
            current = block if current is None else current.compose(block)
            last = step
        out[step] = current.get_epsilon_for_delta(delta)
    return out


def sparsegnn_epsilon_schedule(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps,
    delta: float,
    K_out: Optional[int] = None,
    grid: float = 1e-4,
    union_safe: bool = True,
):
    weights = sparsegnn_mixture_weights(
        p1, p2, r, K_in, K_out, union_safe=union_safe)
    return _compose_schedule(
        mixture_gaussian_pld(weights, sigma, grid), steps, delta)


def sparsegnn_epsilon(
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    sigma: float,
    steps: int,
    delta: float,
    K_out: Optional[int] = None,
    grid: float = 1e-4,
    union_safe: bool = True,
) -> float:
    return sparsegnn_epsilon_schedule(
        p1, p2, r, K_in, sigma, [steps], delta, K_out=K_out,
        grid=grid, union_safe=union_safe)[int(steps)]


@dataclass(frozen=True)
class SparseGNNNoiseCalibration:
    noise_multiplier: float
    noise_std: float
    noise_variance: float
    epsilon: float
    target_epsilon: float
    delta: float
    evaluations: int

    def as_dict(self) -> dict:
        return asdict(self)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def calibrate_sparsegnn_noise(
    *,
    target_epsilon: float,
    target_delta: float,
    p1: float,
    p2: float,
    r: int,
    K_in: int,
    K_out: int,
    steps: int,
    clip: float = 1.0,
    grid: float = 1e-4,
    sigma_rtol: float = 1e-3,
    sigma_atol: float = 1e-6,
    max_sigma: float = 1e6,
    union_safe: bool = True,
) -> SparseGNNNoiseCalibration:
    """Find the smallest known-safe Opacus noise multiplier."""
    target_epsilon = _positive_finite("target_epsilon", target_epsilon)
    if not math.isfinite(target_delta) or not 0 < target_delta < 1:
        raise ValueError("target_delta must be finite and lie in (0, 1)")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    clip = _positive_finite("clip", clip)
    grid = _positive_finite("grid", grid)
    sigma_rtol = _positive_finite("sigma_rtol", sigma_rtol)
    sigma_atol = _positive_finite("sigma_atol", sigma_atol)
    max_sigma = _positive_finite("max_sigma", max_sigma)
    if max_sigma < 1:
        raise ValueError("max_sigma must be at least 1")

    weights = sparsegnn_mixture_weights(
        p1, p2, r, K_in, K_out, union_safe=union_safe)
    values = {}

    def epsilon_at(sigma):
        if sigma not in values:
            epsilon = float(
                mixture_gaussian_pld(weights, sigma, grid)
                .self_compose(steps).get_epsilon_for_delta(target_delta))
            if math.isnan(epsilon):
                raise RuntimeError(f"SparseGNN accountant returned NaN at sigma={sigma}")
            values[sigma] = epsilon
        return values[sigma]

    low, high = 0.0, 1.0
    high_epsilon = epsilon_at(high)
    while math.isinf(high_epsilon) or high_epsilon > target_epsilon:
        low = high
        if high >= max_sigma:
            raise RuntimeError(
                f"failed to bracket a SparseGNN noise multiplier at "
                f"max_sigma={max_sigma}")
        high = min(high * 2, max_sigma)
        high_epsilon = epsilon_at(high)
    while high - low > max(sigma_atol, sigma_rtol * high):
        midpoint = (low + high) / 2
        midpoint_epsilon = epsilon_at(midpoint)
        if math.isinf(midpoint_epsilon) or midpoint_epsilon > target_epsilon:
            low = midpoint
        else:
            high, high_epsilon = midpoint, midpoint_epsilon

    noise_std = high * clip
    return SparseGNNNoiseCalibration(
        noise_multiplier=high,
        noise_std=noise_std,
        noise_variance=noise_std ** 2,
        epsilon=high_epsilon,
        target_epsilon=target_epsilon,
        delta=target_delta,
        evaluations=len(values),
    )


def naive_opacus_epsilon(
    sigma: float, sample_rate: float, steps: int, delta: float,
    mechanism: str = "prv",
) -> float:
    """Conventional subsampled-Gaussian comparator, not a graph guarantee."""
    from opacus.accountants import create_accountant
    try:
        accountant = create_accountant(mechanism=mechanism)
        accountant.history = [(sigma, sample_rate, steps)]
        return accountant.get_epsilon(delta=delta)
    except Exception:
        accountant = create_accountant(mechanism="rdp")
        accountant.history = [(sigma, sample_rate, steps)]
        return accountant.get_epsilon(delta=delta)
