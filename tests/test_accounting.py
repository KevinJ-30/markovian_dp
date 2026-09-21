"""Tests for Theorem 5.4 PLD accounting."""

import math

import numpy as np
import pytest
import torch

from src.privacy.accountants import (
    DPARAccountant, SparseGNNAccountant, calibrate_dpar_noise)
from src.privacy import accounting as sparse_accounting
from src.privacy.accounting import (
    calibrate_sparsegnn_noise, mixture_gaussian_pld, naive_opacus_epsilon,
    shell_sizes, sparsegnn_epsilon, sparsegnn_epsilon_schedule,
    sparsegnn_mixture_weights,
)
from src.privacy.privacy_loss import DoubleMixtureGaussianPrivacyLoss

pytest.importorskip("dp_accounting")


def test_shell_sizes_use_union_safe_out_degree_bound():
    assert shell_sizes(2, K_out=5) == [1, 10, 50]
    assert shell_sizes(2, K_out=5, union_safe=False) == [1, 5, 25]
    assert shell_sizes(3, K_out=4)[0] == 1


def test_mixture_weights_are_a_distribution_of_the_right_length():
    weights = sparsegnn_mixture_weights(0.3, 0.5, 2, 4, 3)
    assert len(weights) == sum(shell_sizes(2, 3)) + 1
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)
    assert (weights >= 0).all()


def test_mixture_weights_mean_matches_paper_expectation():
    p1, p2, radius, k_in, k_out = 0.2, 0.5, 2, 4, 5
    weights = sparsegnn_mixture_weights(p1, p2, radius, k_in, k_out)
    sizes = shell_sizes(radius, k_out)
    K = min(k_in, k_out)
    retention = [1.0] + [
        1.0 - math.prod((1.0 - p2 ** level) ** (K ** (level - 1))
                        for level in range(distance, radius + 1))
        for distance in range(1, radius + 1)
    ]
    expected = p1 * sum(n * q for n, q in zip(sizes, retention))
    observed = float(sum(index * mass for index, mass in enumerate(weights)))
    assert math.isclose(observed, expected, rel_tol=1e-9)


def test_mixture_weights_p2_zero_is_root_only():
    weights = sparsegnn_mixture_weights(0.3, 0.0, 3, 5, 5)
    assert math.isclose(float(weights[0]), 0.7, rel_tol=1e-12)
    assert math.isclose(float(weights[1]), 0.3, rel_tol=1e-12)
    assert float(weights[2:].sum()) == 0.0


def test_double_mixture_maps_to_theorem_pair():
    weights = np.array([0.6, 0.3, 0.1])
    support = 2.0 * np.arange(len(weights))
    sigma = 2.5
    privacy_loss = DoubleMixtureGaussianPrivacyLoss(
        sigma, support, support, weights, weights)
    from scipy.special import logsumexp
    from scipy.stats import norm
    for value in (-3.0, 0.0, 2.0):
        upper = logsumexp(
            norm.logpdf(value, loc=-support, scale=sigma), b=weights)
        lower = logsumexp(
            norm.logpdf(value, loc=support, scale=sigma), b=weights)
        assert privacy_loss.privacy_loss(value) == pytest.approx(upper - lower)


def test_identity_pair_has_zero_epsilon():
    pld = mixture_gaussian_pld([1.0], sigma=1.0, grid=1e-3)
    assert pld.self_compose(10).get_epsilon_for_delta(1e-6) == 0.0


def test_connect_dots_is_pessimistic_for_analytic_profile():
    weights = np.array([0.7, 0.2, 0.1])
    support = 2.0 * np.arange(len(weights))
    privacy_loss = DoubleMixtureGaussianPrivacyLoss(
        3.0, support, support, weights, weights)
    pld = mixture_gaussian_pld(weights, sigma=3.0, grid=1e-3)
    epsilons = np.array([0.0, 0.5, 1.0])
    assert np.all(
        pld.get_delta_for_epsilon(epsilons)
        >= privacy_loss.get_delta_for_epsilon(epsilons) - 1e-12)


@pytest.mark.parametrize("sigma", [5.0, 10.0])
def test_p1_one_r0_matches_gaussian_at_sensitivity_four(sigma):
    T, delta = 100, 1e-6
    ours = sparsegnn_epsilon(
        p1=1.0, p2=0.0, r=0, K_in=5, K_out=5, sigma=sigma,
        steps=T, delta=delta)
    ref = naive_opacus_epsilon(
        sigma / 4.0, 1.0, T, delta, mechanism="prv")
    assert abs(ours - ref) < 0.05 * ref


def test_p2_zero_matches_r0():
    kwargs = dict(
        p1=0.3, K_in=6, K_out=6, sigma=5.0, steps=20, delta=1e-6,
        grid=1e-3)
    assert math.isclose(
        sparsegnn_epsilon(p2=0.9, r=0, **kwargs),
        sparsegnn_epsilon(p2=0.0, r=3, **kwargs),
        rel_tol=1e-6)


def test_epsilon_monotone_in_p2_and_sigma():
    def epsilon(p2, sigma):
        return sparsegnn_epsilon(
            p1=0.1, p2=p2, r=1, K_in=3, K_out=3, sigma=sigma,
            steps=20, delta=1e-6, grid=1e-3)
    dense = epsilon(1.0, 5.0)
    assert epsilon(0.1, 5.0) < dense
    assert epsilon(1.0, 10.0) < dense




def test_checkpoint_schedule_matches_direct_composition():
    kwargs = dict(
        p1=0.1, p2=0.2, r=1, K_in=2, K_out=3, sigma=4.0,
        delta=1e-6, grid=1e-3)
    schedule = sparsegnn_epsilon_schedule(steps=[1, 3], **kwargs)
    assert schedule[1] == pytest.approx(
        sparsegnn_epsilon(steps=1, **kwargs))
    assert schedule[3] == pytest.approx(
        sparsegnn_epsilon(steps=3, **kwargs))


def test_calibration_returns_a_safe_noise_multiplier():
    params = dict(
        target_epsilon=1.0, target_delta=1e-5,
        p1=0.05, p2=0.1, r=1, K_in=2, K_out=2, steps=2,
        grid=1e-3, sigma_rtol=1e-2)
    calibration = calibrate_sparsegnn_noise(**params)
    epsilon = sparsegnn_epsilon(
        p1=params["p1"], p2=params["p2"], r=params["r"],
        K_in=params["K_in"], K_out=params["K_out"],
        sigma=calibration.noise_multiplier, steps=params["steps"],
        delta=params["target_delta"], grid=params["grid"])
    assert calibration.epsilon <= params["target_epsilon"]
    assert epsilon <= params["target_epsilon"]
    assert sparsegnn_epsilon(
        p1=params["p1"], p2=params["p2"], r=params["r"],
        K_in=params["K_in"], K_out=params["K_out"],
        sigma=calibration.noise_multiplier * 0.98, steps=params["steps"],
        delta=params["target_delta"], grid=params["grid"]) > params["target_epsilon"]


def test_calibration_reports_pre_normalization_noise_scale():
    calibration = calibrate_sparsegnn_noise(
        target_epsilon=1.0, target_delta=1e-5,
        p1=0.05, p2=0.1, r=1, K_in=2, K_out=2, steps=2,
        clip=3.0, grid=1e-3, sigma_rtol=1e-2)
    assert calibration.noise_std == calibration.noise_multiplier * 3.0
    assert calibration.noise_variance == calibration.noise_std ** 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_epsilon": 0.0}, {"target_delta": 0.0},
        {"sigma_rtol": 0.0}, {"sigma_atol": 0.0},
        {"grid": 0.0}, {"max_sigma": 0.5},
    ],
)
def test_calibration_rejects_invalid_inputs(kwargs):
    params = dict(
        target_epsilon=1.0, target_delta=1e-5,
        p1=0.05, p2=0.1, r=1, K_in=2, K_out=2, steps=2)
    params.update(kwargs)
    with pytest.raises(ValueError):
        calibrate_sparsegnn_noise(**params)


def test_calibration_rejects_exhausted_bracket():
    with pytest.raises(RuntimeError, match="failed to bracket"):
        calibrate_sparsegnn_noise(
            target_epsilon=1e-12, target_delta=1e-5,
            p1=0.05, p2=0.1, r=1, K_in=2, K_out=2, steps=2,
            max_sigma=1.0, grid=1e-3)


def test_calibration_prepares_sigma_independent_weights_once(monkeypatch):
    original = sparse_accounting.sparsegnn_mixture_weights
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse_accounting, "sparsegnn_mixture_weights", counted)
    calibrate_sparsegnn_noise(
        target_epsilon=1.0, target_delta=1e-5,
        p1=0.05, p2=0.1, r=1, K_in=2, K_out=2, steps=2,
        grid=1e-3, sigma_rtol=1e-2)
    assert calls == 1


def test_sparsegnn_accountant_calibrates_like_direct_solver():
    kwargs = dict(
        p1=0.05, p2=0.1, radius=1, k_in=2, k_out=2, steps=2,
        clip=1.5, grid=1e-3, sigma_rtol=1e-2)
    direct = calibrate_sparsegnn_noise(
        target_epsilon=1.0, target_delta=1e-5,
        p1=kwargs["p1"], p2=kwargs["p2"], r=kwargs["radius"],
        K_in=kwargs["k_in"], K_out=kwargs["k_out"], steps=kwargs["steps"],
        clip=kwargs["clip"], grid=kwargs["grid"],
        sigma_rtol=kwargs["sigma_rtol"])
    assert SparseGNNAccountant().calibrate(
        1.0, 1e-5, **kwargs) == direct.as_dict()
    result = SparseGNNAccountant().account(
        p1=kwargs["p1"], p2=kwargs["p2"], radius=kwargs["radius"],
        k_in=kwargs["k_in"], k_out=kwargs["k_out"],
        sigma=direct.noise_multiplier, steps=kwargs["steps"], delta=1e-5,
        grid=kwargs["grid"])
    assert result.epsilon <= 1.0
    assert result.parameters["grid"] == kwargs["grid"]


# ── DPAR target-privacy calibration ─────────────────────────────────────────

@pytest.mark.parametrize("ppr_releases", [1, 2])
def test_dpar_ppr_calibration_reconstructs_equal_component_budget(ppr_releases):
    params = dict(
        target_epsilon=8.0, target_delta=5e-4, train_nodes=12,
        sampled_train_nodes=6, ppr_releases=ppr_releases,
        ppr_clip=2.0, sgd_clip=3.0, batch_size=3, steps=2,
    )
    calibration = calibrate_dpar_noise(**params)
    amplification = params["sampled_train_nodes"] / params["train_nodes"]
    primitive = math.sqrt(2.0 * math.log(1.25 / calibration.ppr_delta_per_release))
    primitive *= params["ppr_clip"] / calibration.ppr_noise_std
    base_delta = 2.0 * calibration.ppr_delta_per_release * ppr_releases
    reconstructed = DPARAccountant._inverse_composition(primitive, ppr_releases, base_delta)
    assert math.isclose(reconstructed * amplification, params["target_epsilon"] / 2.0, rel_tol=1e-12)
    assert calibration.ppr_delta == params["target_delta"] / 2.0
    assert calibration.sampled_train_nodes == 6
    first = DPARAccountant().account(
        ppr_releases=ppr_releases, amplification_rate=amplification,
        delta=calibration.ppr_delta_per_release, ppr_clip=params["ppr_clip"],
        ppr_noise=calibration.ppr_noise_std, topk=1,
    )
    second = DPARAccountant().account(
        ppr_releases=ppr_releases, amplification_rate=amplification,
        delta=calibration.ppr_delta_per_release, ppr_clip=params["ppr_clip"],
        ppr_noise=calibration.ppr_noise_std, topk=99,
    )
    assert first.epsilon == second.epsilon
    assert math.isclose(first.epsilon, calibration.ppr_epsilon, rel_tol=1e-12)
    assert first.delta == second.delta == calibration.ppr_delta
    assert first.composition_count == ppr_releases
    assert first.sampling_probability == 0.5


def test_dpar_sgd_calibration_uses_outer_population_and_final_delta():
    params = dict(
        target_epsilon=8.0, target_delta=5e-4, train_nodes=12,
        sampled_train_nodes=6, ppr_releases=2, ppr_clip=2.0, sgd_clip=3.0,
        batch_size=3, steps=4,
    )
    calibration = calibrate_dpar_noise(**params)
    accounted = DPARAccountant().account_training(
        noise_multiplier=calibration.sgd_noise_multiplier,
        sample_rate=params["batch_size"] / params["sampled_train_nodes"],
        steps=params["steps"], delta=calibration.sgd_delta,
        amplification_rate=calibration.amplification_rate,
    )
    lower = DPARAccountant().account_training(
        noise_multiplier=calibration.sgd_noise_multiplier / 2.0,
        sample_rate=params["batch_size"] / params["sampled_train_nodes"],
        steps=params["steps"], delta=calibration.sgd_delta,
        amplification_rate=calibration.amplification_rate,
    )
    assert calibration.amplification_rate == 0.5
    assert calibration.ppr_releases == 2
    assert calibration.sampled_train_nodes == 6
    assert accounted.sampling_probability == 0.5
    assert accounted.epsilon <= calibration.sgd_epsilon <= params["target_epsilon"] / 2.0
    assert lower.epsilon > params["target_epsilon"] / 2.0
    assert calibration.sgd_noise_std == calibration.sgd_noise_multiplier * params["sgd_clip"]
    assert calibration.sgd_noise_variance == calibration.sgd_noise_std ** 2
    assert accounted.delta == calibration.sgd_delta == params["target_delta"] / 2.0


def test_dpar_calibration_caps_batches_at_outer_population():
    calibration = calibrate_dpar_noise(
        target_epsilon=8.0, target_delta=5e-4, train_nodes=12,
        sampled_train_nodes=6, ppr_releases=2, ppr_clip=2.0, sgd_clip=3.0,
        batch_size=7, steps=4,
    )
    assert calibration.sampled_train_nodes == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_epsilon": 0.0}, {"target_delta": 1.0}, {"train_nodes": 0},
        {"sampled_train_nodes": 0}, {"sampled_train_nodes": 13},
        {"ppr_releases": 7}, {"steps": 0},
        {"ppr_clip": 0.0}, {"sgd_clip": 0.0}, {"sigma_rtol": 0.0},
        {"sigma_atol": 0.0}, {"max_noise_multiplier": 0.0},
    ],
)
def test_dpar_calibration_rejects_invalid_inputs(kwargs):
    params = dict(
        target_epsilon=8.0, target_delta=5e-4, train_nodes=12,
        sampled_train_nodes=6, ppr_releases=2,
        ppr_clip=2.0, sgd_clip=3.0, batch_size=3, steps=4,
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        calibrate_dpar_noise(**params)


def test_dpar_calibration_rejects_an_unbracketed_sgd_budget():
    with pytest.raises(RuntimeError, match="failed to bracket a DPAR DP-SGD noise multiplier"):
        calibrate_dpar_noise(
            target_epsilon=1e-12, target_delta=5e-4, train_nodes=12,
            sampled_train_nodes=6, ppr_releases=2,
            ppr_clip=2.0, sgd_clip=3.0, batch_size=3, steps=4,
            max_noise_multiplier=1.0,
        )


