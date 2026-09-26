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

from src.models.baselines import (
    _OneHopGCN, _OneHopGIN, _OneHopGraphSAGE,
    _PaddedOneHopGCN, _PaddedOneHopGIN, _PaddedOneHopGraphSAGE,
)
from src.processing.dpgnn import iter_dpgnn_batches
from src.processing.sparse_expand import build_adjacency
from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


@pytest.fixture
def one_hop_stars():
    torch.manual_seed(31)
    model = _OneHopGCN(inputs=3, hidden=5, classes=2, dropout=0.0)
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
    # An isolated root and a repeated root exercise independent loss terms.
    full_stars = ([0, 1, 2, 3, 4], [4], [1, 3], [0, 1, 2, 3, 4])
    stars = [x[ids] for ids in full_stars]
    width = max(map(len, stars))
    features = torch.zeros(len(stars), width, x.size(1))
    node_mask = torch.zeros(len(stars), width, dtype=torch.bool)
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


@pytest.mark.parametrize("model_type,padded_type", [
    (_OneHopGraphSAGE, _PaddedOneHopGraphSAGE),
    (_OneHopGIN, _PaddedOneHopGIN),
])
def test_padded_neighbour_logits_match_explicit_one_hop_stars(
        one_hop_stars, model_type, padded_type):
    _, features, node_mask, _, stars = one_hop_stars
    model = model_type(inputs=3, hidden=5, classes=2, dropout=0.0)
    with torch.no_grad():
        expected = torch.stack([_explicit_star_logits(model, star) for star in stars])
        actual = padded_type(model)(features, node_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gin_sums_neighbours_and_root_once_and_ignores_padding():
    model = _OneHopGIN(inputs=2, hidden=2, classes=2, dropout=0.0)
    with torch.no_grad():
        for layer in (model.mlp[0], model.mlp[2], model.decoder):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    # Duplicate self-loops must not change epsilon=0; root 2 is isolated.
    edges = torch.tensor([[0, 0, 0, 0, 1, 1, 2], [0, 0, 1, 2, 1, 0, 2]])
    expected = torch.tensor([[11.0, 18.0], [4.0, 7.0], [7.0, 11.0]])
    torch.testing.assert_close(model(x, edges, torch.ones(edges.size(1))), expected)
    mask = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    features = torch.stack((x, x[[1, 0, 2]], x[[2, 0, 1]]))
    features = features.masked_fill(~mask.unsqueeze(-1), float("nan")).requires_grad_()
    logits = _PaddedOneHopGIN(model)(features, mask)
    torch.testing.assert_close(logits, expected)
    logits.sum().backward()
    torch.testing.assert_close(features.grad, mask.unsqueeze(-1).expand_as(features).float())


@pytest.mark.parametrize("model_type,padded_type", [
    (_OneHopGCN, _PaddedOneHopGCN),
    (_OneHopGraphSAGE, _PaddedOneHopGraphSAGE),
    (_OneHopGIN, _PaddedOneHopGIN),
])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_opacus_grad_samples_match_explicit_per_root_autograd(
        one_hop_stars, model_type, padded_type, device):
    _, features, node_mask, labels, stars = one_hop_stars
    model = model_type(inputs=3, hidden=5, classes=2, dropout=0.0).to(device)
    # Nonzero padding must not alter shared parameter gradients.
    features = features.masked_fill(~node_mask.unsqueeze(-1), 1234.0).to(device)
    node_mask, labels = node_mask.to(device), labels.to(device)
    stars = [star.to(device) for star in stars]
    parameters = dict(model.named_parameters())
    gradients = {name: [] for name in parameters}
    for star, label in zip(stars, labels):
        logits = _explicit_star_logits(model, star)
        loss = F.cross_entropy(logits.unsqueeze(0), label.view(1))
        per_root = torch.autograd.grad(loss, tuple(parameters.values()))
        for name, gradient in zip(parameters, per_root):
            gradients[name].append(gradient)

    wrapped = GradSampleModule(
        padded_type(model), batch_first=True, loss_reduction="mean", strict=True)
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


@pytest.mark.parametrize("model_type,padded_type", [
    (_OneHopGCN, _PaddedOneHopGCN),
    (_OneHopGraphSAGE, _PaddedOneHopGraphSAGE),
    (_OneHopGIN, _PaddedOneHopGIN),
])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_hidden_dropout_matches_full_and_padded_training_and_evaluation(
        model_type, padded_type, dropout):
    # Identity decoding exposes hidden activations in the observable logits.
    options = {} if dropout == 0.5 else {"dropout": dropout}
    model = model_type(inputs=3, hidden=32, classes=32, **options)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
        model.decoder.weight.copy_(torch.eye(32))
        model.decoder.bias.zero_()
    padded = padded_type(model)
    x = torch.ones(4, 3)
    nodes = torch.arange(4)
    edges = torch.stack((nodes, nodes))
    weights = torch.ones(4)
    features, mask = x[:, None, :], torch.ones(4, 1, dtype=torch.bool)
    model.eval()
    padded.eval()
    expected = model(x, edges, weights)
    torch.testing.assert_close(padded(features, mask), expected)
    torch.testing.assert_close(model(x, edges, weights), expected, rtol=0, atol=0)
    torch.testing.assert_close(padded(features, mask), expected, rtol=0, atol=0)

    model.train()
    padded.train()
    torch.manual_seed(17)
    full_train = model(x, edges, weights)
    torch.manual_seed(17)
    padded_train = padded(features, mask)
    torch.testing.assert_close(full_train, padded_train)
    if dropout == 0:
        torch.testing.assert_close(full_train, expected, rtol=0, atol=0)
    else:
        assert (full_train == 0).any()
        assert (full_train != 0).any()
        torch.testing.assert_close(
            full_train, torch.where(full_train == 0, 0, expected / (1 - dropout)))
        assert not torch.equal(model(x, edges, weights), full_train)
    model.eval()
    padded.eval()
    torch.testing.assert_close(model(x, edges, weights), expected, rtol=0, atol=0)
    torch.testing.assert_close(padded(features, mask), expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_type,padded_type", [
    (_OneHopGCN, _PaddedOneHopGCN),
    (_OneHopGraphSAGE, _PaddedOneHopGraphSAGE),
    (_OneHopGIN, _PaddedOneHopGIN),
])
def test_dropout_opacus_grad_samples_match_per_root_autograd(
        one_hop_stars, model_type, padded_type):
    _, features, node_mask, labels, _ = one_hop_stars
    model = model_type(inputs=3, hidden=16, classes=2)
    reference = padded_type(deepcopy(model))
    # Replaying the same batched dropout draw keeps the stochastic masks fixed
    # while comparing each root's true derivative to Opacus's grad_sample.
    torch.manual_seed(43)
    losses = F.cross_entropy(reference(features, node_mask), labels, reduction="none")
    gradients = [
        torch.autograd.grad(loss, tuple(reference.parameters()), retain_graph=True)
        for loss in losses
    ]
    wrapped = GradSampleModule(
        padded_type(model), batch_first=True, loss_reduction="mean", strict=True)
    try:
        torch.manual_seed(43)
        F.cross_entropy(wrapped(features, node_mask), labels).backward()
        for index, parameter in enumerate(model.parameters()):
            torch.testing.assert_close(
                parameter.grad_sample,
                torch.stack([per_root[index] for per_root in gradients]),
                rtol=1e-5, atol=1e-6)
    finally:
        wrapped.to_standard_module()


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_dropout_is_rejected_before_training(dropout):
    with pytest.raises(ValueError, match="dropout"):
        PartitionedDPGNN(DPGNNConfig(
            num_classes=2, steps=1, batch_size=1, noise_multiplier=1.0,
            dropout=dropout))


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
                         generator, multilabel=False):
    optimizer.zero_grad()
    parameters = tuple(model.parameters())
    flat_gradients = []
    for star, label in zip(stars, labels):
        logits = _explicit_star_logits(model, star).unsqueeze(0)
        loss = (F.binary_cross_entropy_with_logits(logits, label.view(1, -1).float())
                if multilabel else F.cross_entropy(logits, label.view(1)))
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
        max_degree=1, clip=clip, latent_size=5, dropout=0.0)
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
            roots, adjacency=adjacency, x=x, y=y,
            max_padded_nodes=100, device=torch.device("cpu")))
        _assert_adam_matches(model, adam, reference, reference_adam, step=1)
    finally:
        wrapped.to_standard_module()


@pytest.mark.parametrize("multilabel", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_noisy_adam_updates_are_independent_of_unequal_physical_chunks(
        private_star_graph, device, multilabel):
    initial, x, y, adjacency, star_nodes = private_star_graph
    if multilabel:
        y = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [0, 1], [1, 0]])
    device = torch.device(device)
    x, y = x.to(device), y.to(device)
    models = [deepcopy(initial).to(device) for _ in range(2)]
    reference = deepcopy(initial).to(device)
    clip, max_terms, noise_lambda, noise_seed = 0.2, 2, 0.4, 2718
    config = DPGNNConfig(
        num_classes=2, steps=2, batch_size=4, noise_multiplier=noise_lambda,
        max_degree=1, clip=clip, latent_size=5, multilabel=multilabel, dropout=0.0)
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
                generator=reference_generator, multilabel=multilabel)
            for model, adam, wrapped, optimizer, budget in zip(
                    models, adams, wrappers, optimizers, (100, 4)):
                batches = list(iter_dpgnn_batches(
                    roots, adjacency=adjacency, x=x, y=y,
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


@pytest.mark.parametrize("requested_delta,weight_decay", [(None, 0.0), (1.0 / 3, 5e-4)])
def test_small_population_fit_uses_effective_terms_for_release_and_accounting(
        requested_delta, weight_decay):
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
        delta=requested_delta, weight_decay=weight_decay,
        evaluate_every=2,  # Compare the final update, not an earlier selected model.
        max_degree=5, latent_size=5, clip=0.2, max_private_batch_nodes=2, dropout=0.0)
    torch.manual_seed(config.seed)
    reference = _OneHopGraphSAGE(inputs=3, hidden=5, classes=2, dropout=0.0)
    reference_adam = torch.optim.Adam(reference.parameters(), lr=config.learning_rate,
                                      weight_decay=weight_decay)
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
    delta = requested_delta if requested_delta is not None else 1.0 / 30
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


def test_evaluate_forwards_full_context_but_scores_eval_mask_and_ignore_label():
    observed = {}

    class FixedModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_weight):
            observed["nodes"] = x.size(0)
            return x

    logits = torch.zeros((4, 20))
    logits[1, 1] = 5.0
    logits[2, 0] = 5.0
    data = SimpleNamespace(
        num_nodes=4,
        x=logits,
        y=torch.tensor([0, 1, 19, 0]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        eval_mask=torch.tensor([False, True, True, False]),
    )
    trainer = PartitionedDPGNN(DPGNNConfig(
        num_classes=20, steps=1, batch_size=1, noise_multiplier=1.0,
        metric_ignore_label=19))
    assert trainer.evaluate(FixedModel(), data, seed=0) == 1.0
    assert observed["nodes"] == 4


def test_binary_fit_uses_one_logit_and_auroc_result_keys():
    train = SimpleNamespace(
        num_nodes=6,
        x=torch.randn(6, 3),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    held_out = SimpleNamespace(
        num_nodes=4,
        x=torch.randn(4, 3),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        eval_mask=torch.tensor([True, True, False, False]),
    )
    result = PartitionedDPGNN(DPGNNConfig(
        num_classes=2, steps=1, batch_size=3, noise_multiplier=1.0,
        latent_size=5, binary=True, max_private_batch_nodes=16,
    )).fit(train, held_out, held_out)
    assert result["model"].decoder.out_features == 1
    assert result["metric"] == "auroc"
    assert set(result) >= {"validation_auroc", "test_auroc"}


def test_regression_evaluate_scores_only_selected_targets_with_negative_r2():
    class FixedModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_weight):
            return x

    data = SimpleNamespace(
        num_nodes=4,
        x=torch.tensor([[1000.0], [2.0], [2.0], [-1000.0]]),
        y=torch.tensor([1000.0, 0.0, 1.0, -1000.0]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        eval_mask=torch.tensor([False, True, True, False]),
    )
    trainer = PartitionedDPGNN(DPGNNConfig(
        num_classes=1, regression=True, steps=1, batch_size=1, noise_multiplier=1.0))
    assert trainer.evaluate(FixedModel(), data, seed=0) == pytest.approx(-9.0)


@pytest.mark.parametrize("architecture", ["graphsage", "gin"])
@pytest.mark.parametrize("candidates,undefined,selected_step,validation_score,test_score", [
    ((3.0, 3.0, 4.0), False, 2, -4.0, 1.0),
    ((4.0, 3.0, 2.0), False, 5, -1.0, 0.0),
    ((3.0, 2.0, 4.0), True, 2, None, 1.0),
])
def test_fit_restores_validation_selected_model_for_test_and_bootstrap(
        candidates, undefined, selected_step, validation_score, test_score, architecture):
    edges = torch.empty((2, 0), dtype=torch.long)
    train = SimpleNamespace(
        num_nodes=2, x=torch.ones(2, 1), y=torch.tensor([0.0, 2.0]),
        edge_index=edges)
    val = SimpleNamespace(
        **vars(train), eval_mask=torch.full((2,), not undefined, dtype=torch.bool))
    test = SimpleNamespace(
        num_nodes=2, x=train.x, y=torch.full((2,), 3.0), edge_index=edges)

    class ControlledDPGNN(PartitionedDPGNN):
        updates = 0

        def _private_step(self, model, optimizer, batches):
            # Real private updates exercise Opacus after each validation pass.
            super()._private_step(model, optimizer, batches)
            self.updates += 1
            # Give candidates distinct predictions, including a negative-R² tie.
            prediction = candidates[{2: 0, 4: 1, 5: 2}.get(self.updates, 0)]
            with torch.no_grad():
                model._module.decoder.weight.zero_()
                model._module.decoder.bias.fill_(prediction)

        def evaluate(self, model, data, *, seed, bootstrap=None):
            if data is val:
                self.validation_steps.append(self.updates)
                assert seed == self.config.seed + 3
                assert bootstrap is None
            else:
                assert data is test
                assert self.updates == self.config.steps
            return super().evaluate(model, data, seed=seed, bootstrap=bootstrap)

    config = DPGNNConfig(
        num_classes=1, regression=True, steps=5, batch_size=2,
        noise_multiplier=1.0, evaluate_every=2, seed=17,
        latent_size=128, learning_rate=0.001, architecture=architecture,
        bootstrap_resamples=20)
    trainer = ControlledDPGNN(config)
    trainer.validation_steps = []
    result = trainer.fit(train, val, test)
    assert trainer.updates == 5
    assert trainer.validation_steps == [2, 4, 5]
    assert result["selection"] == {
        "metric": "r2", "step": selected_step,
        "validation_score": validation_score, "evaluate_every": 2,
    }
    if validation_score is None:
        assert np.isnan(result["validation_r2"])
    else:
        assert result["validation_r2"] == pytest.approx(validation_score)
    assert result["test_r2"] == pytest.approx(test_score)
    selected_prediction = candidates[{2: 0, 4: 1, 5: 2}[selected_step]]
    torch.testing.assert_close(
        result["model"](test.x, torch.tensor([[0, 1], [0, 1]]), torch.ones(2)),
        torch.full((2, 1), selected_prediction))
    intervals = result["test_confidence_intervals"]
    assert intervals["n_observations"] == 2
    assert intervals["metrics"]["r2"] == {
        "lower": test_score, "upper": test_score, "valid_resamples": 20,
    }
    accountant = RdpAccountant(np.arange(1, 10, 0.1)[1:])
    accountant.compose(GaussianDpEvent(config.noise_multiplier), count=config.steps)
    assert result["epsilon"] == pytest.approx(
        accountant.get_epsilon(result["delta"]), rel=0, abs=1e-10)


def test_validation_cadence_preserves_private_updates_and_dropout_rng():
    class ObservedDPGNN(PartitionedDPGNN):
        def _private_step(self, model, optimizer, batches):
            super()._private_step(model, optimizer, batches)
            self.states.append({
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            })

    graph = SimpleNamespace(
        num_nodes=6, x=torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10,
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
    trajectories = []
    for evaluate_every in (1, 0):
        trainer = ObservedDPGNN(DPGNNConfig(
            num_classes=2, steps=5, batch_size=4, noise_multiplier=1.0,
            evaluate_every=evaluate_every, seed=11, latent_size=4,
            dropout=0.5, bootstrap_resamples=0))
        trainer.states = []
        result = trainer.fit(graph, graph, graph)
        assert result["selection"]["evaluate_every"] == (evaluate_every or 2)
        trajectories.append(trainer.states)
    for frequent, epoch in zip(*trajectories):
        for name in frequent:
            torch.testing.assert_close(frequent[name], epoch[name], rtol=0, atol=0)


def test_negative_evaluation_interval_is_rejected():
    with pytest.raises(ValueError, match="evaluate_every"):
        PartitionedDPGNN(DPGNNConfig(
            num_classes=2, steps=1, batch_size=1, noise_multiplier=1.0,
            evaluate_every=-1))
