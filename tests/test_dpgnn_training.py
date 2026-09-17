from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

from src.models.baselines import _OneHopGCN, _PaddedOneHopGCN
from src.processing.dpgnn import iter_dpgnn_batches
from src.processing.sparse_expand import build_adjacency
from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


@pytest.fixture
def one_hop_stars():
    torch.manual_seed(31)
    model = _OneHopGCN(inputs=3, hidden=5, classes=2)
    with torch.no_grad():
        model.encoder.bias.copy_(torch.tensor([0.3, -0.4, 0.2, 0.5, -0.1]))
        model.core.bias.fill_(0.2)
        model.decoder.bias.copy_(torch.tensor([-0.3, 0.4]))

    x = torch.tensor([
        [0.2, -0.4, 1.0],
        [1.1, 0.3, -0.7],
        [-0.6, 0.8, 0.5],
        [0.9, -1.2, 0.1],
        [-0.2, 0.7, -0.9],
    ])
    # Root 0 is truncated to three nodes; root 4 is isolated. Repeating root 0
    # exercises root order and multiplicity independently of production sampling.
    full_stars = ([0, 1, 2, 3, 4], [4], [1, 3], [0, 1, 2, 3, 4])
    max_subgraph_nodes = 3
    stars = [x[ids[:max_subgraph_nodes]] for ids in full_stars]
    features = torch.zeros(len(stars), max_subgraph_nodes, x.size(1))
    node_mask = torch.zeros(len(stars), max_subgraph_nodes, dtype=torch.bool)
    for index, star in enumerate(stars):
        features[index, :len(star)] = star
        node_mask[index, :len(star)] = True
    labels = torch.tensor([1, 0, 0, 1])
    return model, features, node_mask, labels, stars


def _explicit_star_logits(model, features):
    size = features.size(0)
    edge_index = torch.stack((
        torch.zeros(size, dtype=torch.long, device=features.device),
        torch.arange(size, device=features.device),
    ))
    edge_weight = features.new_full((size,), 1.0 / size)
    return model(features, edge_index, edge_weight)[0]


def test_padded_root_logits_match_explicit_one_hop_stars(one_hop_stars):
    model, features, node_mask, _, stars = one_hop_stars
    with torch.no_grad():
        expected = torch.stack([_explicit_star_logits(model, star) for star in stars])
        actual = _PaddedOneHopGCN(model)(features, node_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_opacus_grad_samples_match_explicit_per_root_autograd(one_hop_stars):
    model, features, node_mask, labels, stars = one_hop_stars
    parameters = dict(model.named_parameters())
    gradients = {name: [] for name in parameters}
    for star, label in zip(stars, labels):
        logits = _explicit_star_logits(model, star)
        loss = F.cross_entropy(logits.unsqueeze(0), label.view(1))
        per_root = torch.autograd.grad(loss, tuple(parameters.values()))
        for name, gradient in zip(parameters, per_root):
            gradients[name].append(gradient)

    wrapped = GradSampleModule(
        _PaddedOneHopGCN(model), batch_first=True, loss_reduction="mean", strict=True)
    try:
        F.cross_entropy(wrapped(features, node_mask), labels).backward()
        # Inspect the original model's parameters: the private view must train
        # the same layers that full-graph evaluation will later consume.
        for name, parameter in parameters.items():
            torch.testing.assert_close(
                parameter.grad_sample, torch.stack(gradients[name]),
                rtol=1e-5, atol=1e-6)
    finally:
        wrapped.to_standard_module()


@pytest.fixture
def private_star_graph(one_hop_stars):
    model = one_hop_stars[0]
    x = torch.tensor([
        [0.2, -0.4, 1.0],
        [1.1, 0.3, -0.7],
        [-0.6, 0.8, 0.5],
        [0.9, -1.2, 0.1],
        [-0.2, 0.7, -0.9],
        [0.4, -0.3, 0.8],
        [-0.5, 0.2, -1.1],
    ])
    y = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    edge_index = torch.tensor([[0, 0, 0, 0, 0, 2], [1, 2, 3, 4, 5, 6]])
    adjacency = build_adjacency(edge_index, x.size(0), direction="out")
    # These explicit stars are independent of the production CSR gather.
    star_nodes = ([0, 1, 2, 3, 4, 5], [1], [2, 6], [3], [4], [5], [6])
    return model, x, y, adjacency, star_nodes


def _reference_adam_step(model, optimizer, stars, labels, *, clip, noise_std,
                         generator):
    optimizer.zero_grad()
    parameters = tuple(model.parameters())
    flat_gradients = []
    for star, label in zip(stars, labels):
        loss = F.cross_entropy(
            _explicit_star_logits(model, star).unsqueeze(0), label.view(1))
        gradients = torch.autograd.grad(loss, parameters)
        flat_gradients.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
    flat_gradients = torch.stack(flat_gradients)
    # Clip the complete parameter vector, not its individual tensor blocks.
    # Opacus includes this numerical stabilizer in the norm denominator.
    factors = (clip / (flat_gradients.norm(dim=1) + 1e-6)).clamp(max=1.0)
    clipped_sum = (flat_gradients * factors[:, None]).sum(dim=0)
    totals = clipped_sum.split([parameter.numel() for parameter in parameters])
    for parameter, total in zip(parameters, totals):
        # Match the installed Opacus non-secure draw convention: one normal
        # tensor per parameter, with no RNG consumption when noise is zero.
        noise = torch.zeros_like(parameter) if noise_std == 0 else torch.normal(
            mean=0, std=noise_std, size=parameter.shape, device=parameter.device,
            dtype=parameter.dtype, generator=generator)
        parameter.grad = (total.view_as(parameter) + noise) / len(stars)
    optimizer.step()


def _assert_adam_matches(model, optimizer, reference, reference_optimizer, *, step):
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
        state, expected_state = optimizer.state[parameter], reference_optimizer.state[expected]
        assert state["step"].item() == expected_state["step"].item() == step
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[name], expected_state[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("optimizer_lambda", [0.0, 0.4])
def test_private_step_matches_global_clipping_and_real_noise(
        private_star_graph, optimizer_lambda):
    model, x, y, adjacency, star_nodes = private_star_graph
    reference = deepcopy(model)
    roots = torch.tensor([0, 1, 2, 3])
    clip, max_terms, noise_seed = 0.2, 2, 1234
    config = DPGNNConfig(
        num_classes=2, steps=1, batch_size=4, noise_multiplier=0.4,
        max_degree=1, clip=clip, latent_size=5)
    trainer = PartitionedDPGNN(config)
    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    reference_adam = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
    # Zero noise is only a direct optimizer fixture, not a public trainer mode.
    optimizer = DPOptimizer(
        optimizer=adam, noise_multiplier=2 * max_terms * optimizer_lambda,
        max_grad_norm=clip, expected_batch_size=4, loss_reduction="mean",
        generator=torch.Generator().manual_seed(noise_seed), secure_mode=False)
    wrapped = GradSampleModule(
        _PaddedOneHopGCN(model), batch_first=True, loss_reduction="mean", strict=True)
    try:
        _reference_adam_step(
            reference, reference_adam, [x[star_nodes[root]] for root in roots.tolist()],
            y[roots], clip=clip, noise_std=2 * max_terms * clip * optimizer_lambda,
            generator=torch.Generator().manual_seed(noise_seed))
        trainer._private_step(wrapped, optimizer, iter_dpgnn_batches(
            roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=6,
            max_padded_nodes=100, device=torch.device("cpu")))
        _assert_adam_matches(model, adam, reference, reference_adam, step=1)
    finally:
        wrapped.to_standard_module()


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_noisy_adam_updates_are_independent_of_unequal_physical_chunks(
        private_star_graph, device):
    initial, x, y, adjacency, star_nodes = private_star_graph
    device = torch.device(device)
    x, y = x.to(device), y.to(device)
    models = [deepcopy(initial).to(device) for _ in range(2)]
    reference = deepcopy(initial).to(device)
    clip, max_terms, noise_lambda, noise_seed = 0.2, 2, 0.4, 2718
    config = DPGNNConfig(
        num_classes=2, steps=2, batch_size=4, noise_multiplier=noise_lambda,
        max_degree=1, clip=clip, latent_size=5)
    trainer = PartitionedDPGNN(config, device=device)
    adams = [
        torch.optim.Adam(model.parameters(), lr=config.learning_rate) for model in models]
    reference_adam = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
    optimizers = [
        DPOptimizer(
            optimizer=adam, noise_multiplier=2 * max_terms * noise_lambda,
            max_grad_norm=clip, expected_batch_size=4, loss_reduction="mean",
            generator=torch.Generator(device=device).manual_seed(noise_seed),
            secure_mode=False)
        for adam in adams
    ]
    wrappers = [
        GradSampleModule(
            _PaddedOneHopGCN(model), batch_first=True, loss_reduction="mean", strict=True)
        for model in models
    ]
    reference_generator = torch.Generator(device=device).manual_seed(noise_seed)
    root_sequences = ([0, 1, 2, 3], [2, 3, 0, 1])
    expected_small_chunks = ([1, 2, 1], [2, 1, 1])
    try:
        for step, (root_ids, chunk_sizes) in enumerate(
                zip(root_sequences, expected_small_chunks), start=1):
            roots = torch.tensor(root_ids)
            _reference_adam_step(
                reference, reference_adam, [x[star_nodes[root]] for root in root_ids],
                y[roots.to(device)], clip=clip, noise_std=2 * max_terms * clip * noise_lambda,
                generator=reference_generator)
            for model, adam, wrapped, optimizer, budget in zip(
                    models, adams, wrappers, optimizers, (100, 4)):
                batches = list(iter_dpgnn_batches(
                    roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=6,
                    max_padded_nodes=budget, device=device))
                assert [batch.batch_size for batch in batches] == (
                    [4] if budget == 100 else chunk_sizes)
                if budget == 4:
                    # Root 0's six-node star must remain an oversized singleton.
                    assert any(batch.batch_size == 1 and batch.node_mask.numel() == 6
                               for batch in batches)
                trainer._private_step(wrapped, optimizer, iter(batches))
                _assert_adam_matches(
                    model, adam, reference, reference_adam, step=step)
    finally:
        for wrapped in wrappers:
            wrapped.to_standard_module()


def test_small_population_fit_uses_effective_terms_for_release_and_accounting():
    class ObservedDPGNN(PartitionedDPGNN):
        def _private_step(self, model, optimizer, batches):
            super()._private_step(model, optimizer, batches)
            # Observe real Adam state without replacing the optimizer or release.
            self.adam = optimizer.original_optimizer

    x = torch.tensor([[0.2, -0.4, 1.0], [1.1, 0.3, -0.7], [-0.6, 0.8, 0.5]])
    y = torch.tensor([0, 1, 0])
    edges = torch.empty((2, 0), dtype=torch.long)
    train = SimpleNamespace(num_nodes=3, x=x, y=y, edge_index=edges)
    held_out = SimpleNamespace(num_nodes=1, x=x[:1], y=y[:1], edge_index=edges)
    config = DPGNNConfig(
        num_classes=2, steps=2, batch_size=3, noise_multiplier=0.7, seed=0,
        max_degree=5, latent_size=5, clip=0.2, max_private_batch_nodes=2)
    torch.manual_seed(config.seed)
    reference = _OneHopGCN(inputs=3, hidden=5, classes=2)
    reference_adam = torch.optim.Adam(reference.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(10_000)
    for _ in range(config.steps):
        _reference_adam_step(
            reference, reference_adam, [x[root:root + 1] for root in range(3)], y,
            clip=config.clip, noise_std=2 * 3 * config.clip * config.noise_multiplier,
            generator=generator)

    trainer = ObservedDPGNN(config)
    result = trainer.fit(train, held_out, held_out)
    _assert_adam_matches(
        result["model"], trainer.adam, reference, reference_adam, step=config.steps)
    delta = 1.0 / 30
    accountant = RdpAccountant(np.arange(1, 10, 0.1)[1:])
    accountant.compose(GaussianDpEvent(config.noise_multiplier), count=config.steps)
    assert result["delta"] == pytest.approx(delta)
    assert result["epsilon"] == pytest.approx(
        accountant.get_epsilon(delta), rel=0, abs=1e-10)


@pytest.mark.parametrize("num_nodes,batch_size", [(0, 1), (3, 4)])
def test_invalid_fit_population_is_rejected_before_graph_preparation(num_nodes, batch_size):
    trainer = PartitionedDPGNN(DPGNNConfig(
        num_classes=2, steps=1, batch_size=batch_size, noise_multiplier=1.0))
    # Graph payload is deliberately unusable. Population validation must win
    # over any attempted feature transfer, edge preprocessing, or evaluation.
    invalid_graph = SimpleNamespace(num_nodes=num_nodes, x=None, y=None, edge_index=None)
    with pytest.raises(ValueError):
        trainer.fit(invalid_graph, object(), object())
