import math

import numpy as np
import pytest
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant, compute_epsilon

from src.privacy.dpgnn import multiterm_dpsgd_epsilon


def test_multiterm_epsilon_matches_finite_hypergeometric_sum():
    population, batch_size, max_terms, steps = 100, 10, 6, 10
    noise_multiplier, delta = 2.0, 1e-5
    orders = np.arange(1, 10, 0.1)[1:]
    probabilities = [
        math.comb(max_terms, affected)
        * math.comb(population - max_terms, batch_size - affected)
        / math.comb(population, batch_size)
        for affected in range(max_terms + 1)
    ]
    rdp = np.array([
        steps * math.log(math.fsum(
            probability * math.exp(
                order * (order - 1) * affected ** 2
                / (2 * max_terms ** 2 * noise_multiplier ** 2)
            )
            for affected, probability in enumerate(probabilities)
        )) / (order - 1)
        for order in orders
    ])
    expected = compute_epsilon(orders, rdp, delta)[0]
    actual = multiterm_dpsgd_epsilon(
        steps=steps, noise_multiplier=noise_multiplier, delta=delta,
        num_samples=population, batch_size=batch_size, max_terms=max_terms,
    )
    assert expected == pytest.approx(1.48272331904, rel=0, abs=1e-10)
    assert actual == pytest.approx(expected, rel=0, abs=1e-10)


def test_full_census_matches_composed_gaussian():
    orders = np.arange(1, 10, 0.1)[1:]
    accountant = RdpAccountant(orders)
    accountant.compose(GaussianDpEvent(2.0), count=10)
    actual = multiterm_dpsgd_epsilon(
        steps=10, noise_multiplier=2.0, delta=1e-5,
        num_samples=100, batch_size=100, max_terms=6,
    )
    assert actual == pytest.approx(accountant.get_epsilon(1e-5), rel=0, abs=1e-10)


@pytest.mark.parametrize("batch_size,max_terms", [(11, 6), (5, 11), (5, 0)])
def test_invalid_population_counts_are_rejected(batch_size, max_terms):
    with pytest.raises(ValueError):
        multiterm_dpsgd_epsilon(
            steps=1, noise_multiplier=2.0, delta=1e-5,
            num_samples=10, batch_size=batch_size, max_terms=max_terms,
        )
