"""One-step Rényi bounds for the reflected Gaussian mixtures used by SparseGNN."""

import numpy as np
from scipy.special import log_ndtr, logsumexp

# SparseGNN training adds integer orders 10 through 16 to the baseline grid.
ORDERS = np.r_[np.arange(11, 100) / 10, np.arange(10, 17)]


def _log_expm1(t):
    """log(exp(t)-1) for nonnegative t, including zero and large t."""
    return t + np.log(-np.expm1(-t))


def mixture_gaussian_rdp(weights, sigma):
    """Upper-bound the continuous pair's log moments by Gaussian-bin quadrature.

    Work in standard-deviation units. On each bin, factoring out the bin
    masses leaves the Rényi divergence of two conditional distributions.
    Their log likelihood ratio has range d, so its conditional RDP is at
    most alpha*d**2/8 (the tilted log moment has curvature <= d**2/4).
    """
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or np.any(weights < 0) or
            not np.isclose(weights.sum(), 1, atol=0, rtol=1e-12)):
        raise ValueError("expected normalized, nonnegative mixture weights")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    support = np.flatnonzero(weights)
    means = support / sigma
    log_w = np.log(weights[support])
    if len(means) == 1:
        return 2 * ORDERS * means[0]**2

    exponent = 2 * (ORDERS * (ORDERS - 1))[:, None] * means**2
    # Integrate both tails analytically using the component-revealing bound.
    # Its tilted Gaussian for component k is centered at -(2*alpha-1)*k/sigma.
    # Offset the joint moment's exponential prefactor as well: after
    # Gaussian tail decay the omitted moment is bounded by 2*exp(-72).
    span = (2 * ORDERS[-1] - 1) * means[-1] + np.sqrt(144 + 2 * exponent[-1, -1])
    edges = np.linspace(-span, 0, int(np.ceil(span / 0.04)) + 1)
    shift = (2 * ORDERS - 1)[:, None] * means
    tails = np.logaddexp(log_ndtr(-span + shift), log_ndtr(-span - shift))
    log_excess = logsumexp(log_w + exponent + tails, axis=1)
    intercept = log_w - means**2 / 2

    def log_bin_mass(left, right):
        # Reflect positive intervals to keep tail CDF differences accurate.
        lo = np.where(left > 0, -right, left)
        hi = np.where(left > 0, -left, right)
        log_hi = log_ndtr(hi)
        return logsumexp(log_w + log_hi + np.log(-np.expm1(log_ndtr(lo) - log_hi)), axis=1)

    for start in range(0, len(edges) - 1, 512):
        x = edges[start:start + 513, None]
        lp = log_bin_mass(x[:-1] + means, x[1:] + means)
        lq = log_bin_mass(x[:-1] - means, x[1:] - means)
        # The reflected pair's log likelihood ratio is decreasing, so its
        # range on a bin is exactly the difference at the endpoints.
        loss = logsumexp(intercept - x * means, axis=1) - logsumexp(intercept + x * means, axis=1)
        alpha = ORDERS[:, None]
        # Pair each negative bin with its reflection. Accumulate M_alpha - 1
        # directly to retain precision when the noise is large. For P_bin >=
        # Q_bin and l=log(P_bin/Q_bin), their combined excess above their
        # probability mass is Q_bin*(exp(alpha*l)-1)*(1-exp(-(alpha-1)*l)).
        ratio = np.maximum(lp - lq, 0)
        with np.errstate(divide="ignore"):
            excess = lq + _log_expm1(alpha * ratio) + np.log(-np.expm1(-(alpha - 1) * ratio))
            base = np.logaddexp(lp + (alpha - 1) * ratio, lq - (alpha - 1) * ratio)
            correction = _log_expm1(alpha * (alpha - 1) * np.diff(loss)**2 / 8)
        bins = np.logaddexp(excess, base + correction)
        log_excess = np.logaddexp(log_excess, logsumexp(bins, axis=1))

    rdp = np.logaddexp(0, log_excess) / (ORDERS - 1)
    if not np.all(np.isfinite(rdp)):
        raise ArithmeticError("nonfinite Gaussian-mixture Rényi integral")
    return rdp
