"""Behavioral tests for the Opacus rooted-subgraph DP mechanism."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.sparse.base_mechanism import BaseMechanism
from src.sparse.sparse_expand import (
    RootedSubgraph, build_adjacency, sample_roots, sparse_expand)
from src.sparse.sparse_gnn import OpacusPrivateUpdate


def _subgraph(root):
    return RootedSubgraph(
        root, torch.tensor([root]), torch.zeros((2, 0), dtype=torch.long))


def _flat(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


class _LinearMechanism(BaseMechanism):
    """One prescribed feature vector—and therefore gradient—per root."""

    def __init__(self, targets):
        dimension = next(iter(targets.values())).numel() if targets else 8
        count = max(targets, default=-1) + 1
        count = max(count, 1)
        x = torch.zeros((count, dimension))
        for root, target in targets.items():
            x[root] = target
        data = Data(
            x=x, y=torch.zeros(count),
            train_mask=torch.ones(count, dtype=torch.bool),
            edge_index=torch.zeros((2, 0), dtype=torch.long))
        data.val_mask = data.test_mask = data.train_mask
        module = nn.Linear(dimension, 1, bias=False)
        nn.init.zeros_(module.weight)
        super().__init__(module)
        self.data = data

    def subgraph_loss(self, subgraph):
        return self.module(self.data.x[subgraph.root:subgraph.root + 1]).sum()

    def build_private_module(self):
        return self.module

    def private_losses(self, private_module, batch):
        rows = torch.arange(batch.batch_size)
        values = private_module(batch.features[rows, batch.root_index]).view(-1)
        return values * batch.loss_mask.to(values.dtype)

    def evaluate(self, data=None):
        return {"train": 0.0, "val": 0.0, "test": 0.0}


def _private_update(targets, roots, *, clip=1.0, sigma=0.0,
                    expected_batch=1.0, seed=0):
    mechanism = _LinearMechanism(targets)
    mechanism.build_optimizer(lr=0.0, kind="sgd")
    update = OpacusPrivateUpdate(
        mechanism, C=clip, sigma=sigma, expected_batch=expected_batch,
        noise_gen=torch.Generator().manual_seed(seed))
    update.step([_subgraph(root) for root in roots])
    return mechanism, update


def test_per_subgraph_contribution_is_capped_at_C():
    vector = torch.ones(8) * 100
    mechanism, _ = _private_update({0: vector}, [0], clip=2.0)
    assert _flat([p.grad for p in mechanism.parameters()]).norm() == pytest.approx(2.0)


def test_small_gradients_pass_through_unclipped():
    vector = torch.zeros(8)
    vector[0] = 0.25
    mechanism, _ = _private_update({0: vector}, [0])
    assert _flat([p.grad for p in mechanism.parameters()]).norm() == pytest.approx(0.25)


def test_clipping_is_per_subgraph_not_per_batch():
    vector = torch.zeros(8)
    vector[0] = 2.0
    targets = {index: vector.clone() for index in range(4)}
    mechanism, _ = _private_update(targets, list(targets), clip=1.0)
    assert _flat([p.grad for p in mechanism.parameters()]).norm() == pytest.approx(4.0)


@pytest.mark.parametrize("seed", range(5))
def test_substitution_sensitivity_at_most_two_C(seed):
    generator = torch.Generator().manual_seed(seed)
    targets = {index: torch.randn(8, generator=generator) for index in range(5)}
    base, _ = _private_update(targets, list(targets), clip=1.5)
    changed = dict(targets)
    changed[2] = torch.randn(8, generator=generator)
    other, _ = _private_update(changed, list(changed), clip=1.5)
    difference = _flat([p.grad for p in base.parameters()]) - _flat(
        [p.grad for p in other.parameters()])
    assert difference.norm() <= 3.0 + 1e-6


@pytest.mark.parametrize("sigma,clip", [(1.0, 1.0), (5.0, 1.0), (2.0, 0.5)])
def test_noise_std_is_sigma_times_clip(sigma, clip):
    mechanism = _LinearMechanism({})
    mechanism.build_optimizer(lr=0.0, kind="sgd")
    update = OpacusPrivateUpdate(
        mechanism, C=clip, sigma=sigma, expected_batch=1.0,
        noise_gen=torch.Generator().manual_seed(9))
    draws = []
    for _ in range(400):
        update.step([])
        draws.append(_flat([p.grad.clone() for p in mechanism.parameters()]))
    assert torch.stack(draws).std() == pytest.approx(sigma * clip, rel=0.08)


def test_empty_batch_draws_noise_and_steps_once():
    mechanism = _LinearMechanism({})
    mechanism.build_optimizer(lr=0.0, kind="sgd")
    seed, sigma, clip, expected = 17, 2.0, 0.5, 3.0
    update = OpacusPrivateUpdate(
        mechanism, C=clip, sigma=sigma, expected_batch=expected,
        noise_gen=torch.Generator().manual_seed(seed))
    update.step([])
    reference_gen = torch.Generator().manual_seed(seed)
    expected_noise = torch.normal(
        mean=0.0, std=sigma * clip,
        size=mechanism.parameters()[0].shape, generator=reference_gen) / expected
    assert torch.allclose(mechanism.parameters()[0].grad, expected_noise)


def test_noise_added_once_per_step_not_per_subgraph():
    def variance(count):
        mechanism = _LinearMechanism({i: torch.zeros(8) for i in range(count)})
        mechanism.build_optimizer(lr=0.0, kind="sgd")
        update = OpacusPrivateUpdate(
            mechanism, C=1.0, sigma=1.0, expected_batch=1.0,
            noise_gen=torch.Generator().manual_seed(3))
        draws = []
        for _ in range(300):
            update.step([_subgraph(i) for i in range(count)])
            draws.append(float(mechanism.parameters()[0].grad[0, 0]))
        return torch.tensor(draws).var().item()
    assert variance(16) == pytest.approx(variance(1), rel=0.2)


def test_noise_is_fresh_every_step():
    mechanism = _LinearMechanism({})
    mechanism.build_optimizer(lr=0.0, kind="sgd")
    update = OpacusPrivateUpdate(
        mechanism, C=1.0, sigma=1.0, expected_batch=1.0,
        noise_gen=torch.Generator().manual_seed(10))
    update.step([])
    first = mechanism.parameters()[0].grad.clone()
    update.step([])
    second = mechanism.parameters()[0].grad.clone()
    assert not torch.allclose(first, second)


def test_gradient_is_signal_plus_noise_over_expected_batch():
    vector = torch.zeros(8)
    vector[0] = 0.5
    mechanism, _ = _private_update(
        {0: vector, 1: vector}, [0, 1], clip=1.0, sigma=0.0,
        expected_batch=4.0)
    expected = torch.zeros_like(mechanism.parameters()[0].grad)
    expected[0, 0] = 0.25
    assert torch.allclose(mechanism.parameters()[0].grad, expected)


def test_expected_batch_denominator_uses_public_quantities_only():
    vector = torch.zeros(8)
    vector[0] = 1.0
    for count in (1, 5):
        targets = {index: vector.clone() for index in range(count)}
        mechanism, _ = _private_update(
            targets, list(targets), expected_batch=10.0)
        norm = _flat([p.grad for p in mechanism.parameters()]).norm()
        assert norm == pytest.approx(count / 10.0)


def test_root_sampling_is_poisson_at_rate_p1():
    n, p1, trials = 500, 0.1, 400
    generator = torch.Generator().manual_seed(0)
    counts = [sample_roots(n, p1, generator=generator).numel()
              for _ in range(trials)]
    mean = sum(counts) / trials
    variance = sum((count - mean) ** 2 for count in counts) / (trials - 1)
    assert mean == pytest.approx(n * p1, rel=0.05)
    assert variance == pytest.approx(n * p1 * (1 - p1), rel=0.25)


def test_root_sampling_is_independent_across_steps():
    generator = torch.Generator().manual_seed(0)
    first = set(sample_roots(2000, 0.05, generator=generator).tolist())
    second = set(sample_roots(2000, 0.05, generator=generator).tolist())
    assert len(first & second) / max(len(first), 1) < 0.35


def test_edge_retention_matches_p2():
    degree, p2 = 4000, 0.3
    edges = torch.stack([
        torch.arange(1, degree + 1), torch.zeros(degree, dtype=torch.long)])
    adjacency = build_adjacency(edges, degree + 1, direction="in")
    generator = torch.Generator().manual_seed(0)
    retained = sum(
        sparse_expand(adjacency, 0, p2, 1, generator=generator,
                      direction="in").num_edges
        for _ in range(20))
    assert retained / (20 * degree) == pytest.approx(p2, rel=0.05)


@pytest.mark.parametrize("radius,reads_two_hop", [(1, False), (2, True)])
def test_model_depth_does_not_widen_privacy_radius(radius, reads_two_hop):
    from src.sparse.gnn_mechanism import GNNMechanism

    edges = torch.tensor([[2, 1], [1, 0]])
    data = Data(
        x=torch.randn(3, 4), y=torch.tensor([0, 1, 0]), edge_index=edges,
        train_mask=torch.ones(3, dtype=torch.bool))
    data.val_mask = data.test_mask = data.train_mask
    torch.manual_seed(0)
    mechanism = GNNMechanism(
        data, 4, 2, hidden=8, num_layers=2, dropout=0.0)
    subgraph = sparse_expand(
        build_adjacency(edges, 3, direction="in"), 0, p2=1.0,
        r=radius, direction="in")
    before = float(mechanism.subgraph_loss(subgraph).detach())
    data.x[2] += 100.0
    after = float(mechanism.subgraph_loss(subgraph).detach())
    assert (before != after) == reads_two_hop
    assert (2 in subgraph.nodes.tolist()) == reads_two_hop


class _TwoOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3, 4, bias=False), nn.Linear(3, 4, bias=False)])
        for layer in self.layers:
            nn.init.zeros_(layer.weight)

    def forward(self, features):
        return sum(layer(features).sum(dim=-1) for layer in self.layers)


class _TwoLinearMechanism(_LinearMechanism):
    def __init__(self):
        data = Data(
            x=torch.zeros((1, 3)), y=torch.zeros(1),
            train_mask=torch.ones(1, dtype=torch.bool),
            edge_index=torch.zeros((2, 0), dtype=torch.long))
        data.val_mask = data.test_mask = data.train_mask
        BaseMechanism.__init__(self, _TwoOutput())
        self.data = data

    def subgraph_loss(self, subgraph):
        return self.module(self.data.x).sum()

    def private_losses(self, private_module, batch):
        rows = torch.arange(batch.batch_size)
        features = batch.features[rows, batch.root_index]
        value = private_module(features)
        return value * batch.loss_mask.to(value.dtype)


def test_noise_is_independent_across_same_shaped_parameters():
    mechanism = _TwoLinearMechanism()
    mechanism.build_optimizer(lr=0.0, kind="sgd")
    update = OpacusPrivateUpdate(
        mechanism, C=1.0, sigma=1.0, expected_batch=1.0,
        noise_gen=torch.Generator().manual_seed(5))
    update.step([])
    first, second = [parameter.grad for parameter in mechanism.parameters()]
    assert not torch.allclose(first, second)
