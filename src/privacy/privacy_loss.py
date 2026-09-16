"""Two-mixture Gaussian privacy loss used by the SparseGNN accountant.

Adapted from ``DoubleMixtureGaussianPrivacyLoss`` in
``other_papers/dp_forecasting/src/dp_timeseries/privacy/pld.py``.  It extends
Google ``dp_accounting``'s additive-noise interface to a Gaussian mixture on
both sides of a dominating pair.
"""

import math
import numbers
from typing import Iterable, Sequence, Union

import numpy as np
import scipy
from dp_accounting.pld import common
from dp_accounting.pld.privacy_loss_mechanism import (
    AdditiveNoisePrivacyLoss, AdjacencyType, ConnectDotsBounds,
    TailPrivacyLossDistribution)
from scipy import stats
from scipy.special import logsumexp


class DoubleMixtureGaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss for two Gaussian mixtures with nonnegative shifts.

    ``mu_upper = sum_i p_i N(-s_i, sigma^2)`` and
    ``mu_lower = sum_i q_i N(+t_i, sigma^2)``.
    """

    def __init__(
        self,
        standard_deviation: float,
        sensitivities_upper: Sequence[float],
        sensitivities_lower: Sequence[float],
        sampling_probs_upper: Sequence[float],
        sampling_probs_lower: Sequence[float],
        pessimistic_estimate: bool = True,
        log_mass_truncation_bound: float = -50,
    ) -> None:
        if standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        if log_mass_truncation_bound > 0:
            raise ValueError("log_mass_truncation_bound must be non-positive")

        upper_s = np.asarray(sensitivities_upper, dtype=float)
        lower_s = np.asarray(sensitivities_lower, dtype=float)
        upper_p = np.asarray(sampling_probs_upper, dtype=float)
        lower_p = np.asarray(sampling_probs_lower, dtype=float)
        if upper_s.ndim != 1 or lower_s.ndim != 1:
            raise ValueError("sensitivities must be one-dimensional")
        if upper_s.shape != upper_p.shape or lower_s.shape != lower_p.shape:
            raise ValueError("sensitivities and probabilities must have equal lengths")
        upper_keep, lower_keep = upper_p > 0, lower_p > 0
        upper_s, upper_p = upper_s[upper_keep], upper_p[upper_keep]
        lower_s, lower_p = lower_s[lower_keep], lower_p[lower_keep]
        if not len(upper_s) or not len(lower_s):
            raise ValueError("each mixture must contain positive mass")
        if np.any(upper_s < 0) or np.any(lower_s < 0):
            raise ValueError("sensitivities must be nonnegative")
        if not math.isclose(float(upper_p.sum()), 1.0) or not math.isclose(
                float(lower_p.sum()), 1.0):
            raise ValueError("probabilities on each side must sum to one")
        if upper_s.max() == 0 and lower_s.max() == 0:
            raise ValueError("at least one sensitivity must be positive")

        self.discrete_noise = False
        self.sensitivities_upper = upper_s
        self.sensitivities_lower = lower_s
        self.sampling_probs_upper = upper_p
        self.sampling_probs_lower = lower_p
        self._log_probs_upper = np.log(upper_p)
        self._log_probs_lower = np.log(lower_p)
        self._standard_deviation = float(standard_deviation)
        self._pessimistic_estimate = bool(pessimistic_estimate)
        self._log_mass_truncation_bound = float(log_mass_truncation_bound)
        self._max_sens_upper = float(upper_s.max())
        self._gaussian_random_variable = stats.norm(scale=standard_deviation)

    def mu_upper_cdf(self, x):
        points = np.add.outer(np.atleast_1d(x), self.sensitivities_upper)
        result = (self.noise_cdf(points) * self.sampling_probs_upper).sum(axis=1)
        return float(result[0]) if isinstance(x, numbers.Number) else result

    def mu_lower_log_cdf(self, x):
        points = np.add.outer(np.atleast_1d(x), -self.sensitivities_lower)
        result = scipy.special.logsumexp(
            self.noise_log_cdf(points), axis=1, b=self.sampling_probs_lower)
        return float(result[0]) if isinstance(x, numbers.Number) else result

    def get_delta_for_epsilon(self, epsilon):
        scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.atleast_1d(epsilon).astype(float)
        if not np.all(epsilons[1:] >= epsilons[:-1]):
            raise ValueError("epsilon values must be non-decreasing")
        cutoffs = self.inverse_privacy_losses(epsilons)
        deltas = self.mu_upper_cdf(cutoffs) - np.exp(
            epsilons + self.mu_lower_log_cdf(cutoffs))
        deltas = np.clip(deltas, 0, 1)
        for index in reversed(range(len(deltas) - 1)):
            deltas[index] = max(deltas[index], deltas[index + 1])
        return float(deltas[0]) if scalar else deltas

    def privacy_loss_tail(self, precision: float = 1e-4):
        tail_mass = 0.5 * np.exp(self._log_mass_truncation_bound)
        z_value = float(self._gaussian_random_variable.ppf(tail_mass))
        upper_x = -z_value
        lower_x = common.inverse_monotone_function(
            self.mu_upper_cdf,
            tail_mass,
            common.BinarySearchParameters(
                z_value - self._max_sens_upper, z_value,
                tolerance=precision),
            increasing=True,
        )
        if self._pessimistic_estimate:
            tail_pmf = {
                math.inf: self.mu_upper_cdf(lower_x),
                self.privacy_loss(upper_x): 1 - self.mu_upper_cdf(upper_x),
            }
        else:
            tail_pmf = {self.privacy_loss(lower_x): self.mu_upper_cdf(lower_x)}
        return TailPrivacyLossDistribution(lower_x, upper_x, tail_pmf)

    def connect_dots_bounds(self):
        tail = self.privacy_loss_tail()
        return ConnectDotsBounds(
            epsilon_upper=self.privacy_loss(tail.lower_x_truncation),
            epsilon_lower=self.privacy_loss(tail.upper_x_truncation),
        )

    def privacy_loss(self, x):
        scalar = isinstance(x, numbers.Number)
        values = np.atleast_1d(x).astype(float)
        upper = logsumexp(
            stats.norm.logpdf(
                values[:, None], loc=-self.sensitivities_upper[None, :],
                scale=self._standard_deviation),
            axis=1, b=self.sampling_probs_upper[None, :],
        )
        lower = logsumexp(
            stats.norm.logpdf(
                values[:, None], loc=self.sensitivities_lower[None, :],
                scale=self._standard_deviation),
            axis=1, b=self.sampling_probs_lower[None, :],
        )
        result = upper - lower
        return float(result[0]) if scalar else result

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        return float(self.inverse_privacy_losses(np.atleast_1d(privacy_loss))[0])

    def inverse_privacy_losses(
        self, privacy_losses: np.ndarray, precision: float = 1e-6,
    ) -> np.ndarray:
        privacy_losses = np.asarray(privacy_losses, dtype=float)
        if not np.all(np.diff(privacy_losses) >= 0):
            raise ValueError("privacy_losses must be non-decreasing")
        if not len(privacy_losses):
            return np.array([], dtype=float)

        left = -1.0
        while self.privacy_loss(left) < privacy_losses[-1]:
            left *= 2
        right = 1.0
        while self.privacy_loss(right) > privacy_losses[0]:
            right *= 2

        lower = np.full_like(privacy_losses, left)
        upper = np.full_like(privacy_losses, right)
        while float(np.max(upper - lower)) > precision:
            midpoint = (lower + upper) / 2
            losses = self.privacy_loss(midpoint)
            move_right = losses > privacy_losses
            lower[move_right] = midpoint[move_right]
            upper[~move_right] = midpoint[~move_right]
        # Match the pessimistic endpoint convention used by dp_forecasting.
        return np.floor(upper / precision) * precision

    def noise_cdf(self, x: Union[float, Iterable[float]]):
        return self._gaussian_random_variable.cdf(x)

    def noise_log_cdf(self, x: Union[float, Iterable[float]]):
        return self._gaussian_random_variable.logcdf(x)

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError("two-mixture privacy loss has no unsampled form")

    def inverse_privacy_loss_without_subsampling(self, privacy_loss: float) -> float:
        raise NotImplementedError("two-mixture privacy loss has no unsampled form")

    @classmethod
    def from_privacy_guarantee(
        cls,
        privacy_parameters: common.DifferentialPrivacyParameters,
        sensitivity: float = 1,
        pessimistic_estimate: bool = True,
        sampling_prob: float = 1.0,
        adjacency_type: AdjacencyType = AdjacencyType.REMOVE,
    ):
        raise NotImplementedError(
            "two-mixture privacy loss is not determined by an (epsilon, delta) pair")
