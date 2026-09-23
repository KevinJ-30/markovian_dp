"""Numerical contracts for exact chunked DP-MLP updates, without CUDA."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.baselines import MLP
from results.eight_gpu_domain_graphsaint.dpmlp.engine import (
    clipped_sums,
    poisson_epoch,
    private_step,
)


def _fixture(task):
    torch.manual_seed(812)
    outputs = 1 if task == "binary" else 3
    model = MLP(inputs=4, classes=outputs, hidden=5, layers=2, dropout=0.4).double()
    x = torch.randn(13, 4, dtype=torch.float64)
    if task == "categorical":
        labels = torch.arange(13) % outputs
    else:
        labels = torch.randint(0, 2, (13, outputs)).double()
    return model, x, labels


def _autograd_sums(model, x, labels, mask, clip, task, valid=None):
    """Independent materialized per-example reference using PyTorch autograd."""
    parameters = tuple(model.parameters())
    sums = [torch.zeros_like(parameter) for parameter in parameters]
    loss_total = torch.zeros((), dtype=x.dtype)
    for row in range(len(x)):
        if valid is not None and not bool(valid[row]):
            continue
        hidden = F.relu(F.linear(x[row:row + 1], parameters[0], parameters[1])) * mask[row:row + 1]
        logits = F.linear(hidden, parameters[2], parameters[3])
        if task == "categorical":
            loss = F.cross_entropy(logits, labels[row:row + 1])
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels[row:row + 1])
        gradients = torch.autograd.grad(loss, parameters)
        norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
        scale = (clip / norm.clamp_min(1e-12)).clamp(max=1)
        for total, gradient in zip(sums, gradients):
            total.add_(gradient * scale)
        loss_total.add_(loss.detach())
    return sums, loss_total


def _optimizer(model, *, weight_decay=0.0005):
    return torch.optim.Adam(
        model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=weight_decay, amsgrad=False,
    )


@pytest.mark.parametrize("task", ["categorical", "multilabel", "binary"])
def test_global_clipped_sums_equal_per_example_autograd(task):
    model, x, labels = _fixture(task)
    mask = torch.empty((len(x), 5), dtype=x.dtype).bernoulli_(0.6) / 0.6
    valid = torch.ones(len(x), dtype=x.dtype)
    valid[-2:] = 0
    expected, expected_loss = _autograd_sums(model, x, labels, mask, 0.19, task, valid)
    actual, actual_loss = clipped_sums(
        x, labels, *tuple(model.parameters()), mask, valid,
        task == "multilabel", 0.19, binary=task == "binary", return_loss=True,
    )
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("task", ["categorical", "multilabel", "binary"])
def test_chunked_adam_matches_unchunked_and_autograd_with_fixed_mask_noise(task):
    model, x, labels = _fixture(task)
    models = [model, copy.deepcopy(model), copy.deepcopy(model)]
    optimizers = [_optimizer(candidate) for candidate in models]
    roots = torch.tensor([12, 2, 9, 3, 5, 11, 0, 7, 4], dtype=torch.long)
    clip, sigma, denominator = 0.3, 1.7, 1024
    for _ in range(2):
        mask = torch.empty((len(roots), 5), dtype=x.dtype).bernoulli_(0.6) / 0.6
        noise = tuple(torch.randn_like(parameter) for parameter in model.parameters())
        expected_sums, expected_loss = _autograd_sums(
            models[2], x[roots], labels[roots], mask, clip, task
        )
        optimizers[2].zero_grad(set_to_none=True)
        for parameter, summed, standard_normal in zip(models[2].parameters(), expected_sums, noise):
            parameter.grad = (summed + standard_normal * (sigma * clip)) / denominator
        optimizers[2].step()
        for index, physical_chunk_size in enumerate((3, len(roots))):
            observation = private_step(
                models[index], optimizers[index], x, labels, roots,
                multilabel=task == "multilabel", binary=task == "binary",
                clip=clip, sigma=sigma, expected_batch=denominator,
                physical_chunk_size=physical_chunk_size,
                dropout_mask=mask, standard_noise=noise,
            )
            torch.testing.assert_close(observation.loss_sum, expected_loss, rtol=1e-11, atol=1e-12)
            assert observation.roots == len(roots)
            assert observation.physical_chunks == (3 if index == 0 else 1)
            assert observation.noise_draws == 4
            assert observation.optimizer_updates == 1
        for candidate, optimizer in zip(models[:2], optimizers[:2]):
            for actual, expected in zip(candidate.parameters(), models[2].parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                for field in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(
                        optimizer.state[actual][field], optimizers[2].state[expected][field],
                        rtol=1e-11, atol=1e-12,
                    )


def test_empty_poisson_draw_still_receives_exactly_one_noise_only_adam_update(monkeypatch):
    model, x, labels = _fixture("binary")
    reference = copy.deepcopy(model)
    optimizer, reference_optimizer = _optimizer(model, weight_decay=0), _optimizer(reference, weight_decay=0)
    noises = [torch.randn_like(parameter) for parameter in model.parameters()]
    calls = []

    def draw(parameter):
        index = len(calls)
        calls.append(parameter.shape)
        return noises[index].clone()

    monkeypatch.setattr(torch, "randn_like", draw)
    observed = private_step(
        model, optimizer, x, labels, torch.empty(0, dtype=torch.long),
        multilabel=False, binary=True, clip=0.5, sigma=2.0,
        expected_batch=1024, physical_chunk_size=3,
    )
    for parameter, noise in zip(reference.parameters(), noises):
        parameter.grad = noise / 1024
    reference_optimizer.step()
    assert len(calls) == 4
    assert observed.roots == observed.physical_chunks == 0
    assert observed.loss_sum.item() == 0
    assert observed.noise_draws == 4
    assert observed.optimizer_updates == 1
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert optimizer.state[actual]["step"].item() == 1
        torch.testing.assert_close(
            optimizer.state[actual]["exp_avg"], reference_optimizer.state[expected]["exp_avg"],
            rtol=0, atol=0,
        )


def test_poisson_sampler_never_truncates_a_full_population_draw():
    draws = list(poisson_epoch(np.random.default_rng(41), n=13, q=1.0, steps=2))
    assert len(draws) == 2
    for roots in draws:
        np.testing.assert_array_equal(np.sort(roots), np.arange(13))
