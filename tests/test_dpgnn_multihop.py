"""Complete receptive fields and per-root DP clipping for deeper DP-GNNs."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

from src.models.dpgnn import _MultiHopDPGNN, _PaddedMultiHopDPGNN
from src.privacy.dpgnn import max_terms_per_node
from src.processing.dpgnn import iter_dpgnn_batches
from src.processing.sparse_expand import build_adjacency
from src.training.dpgnn import DPGNNConfig, PartitionedDPGNN


def _fixture():
    # A cycle, shared dependencies, distinct neighborhood sizes, and an isolate.
    edges = torch.tensor([[0, 0, 1, 2, 2, 3, 4, 5], [1, 2, 3, 3, 4, 0, 5, 6]])
    generator = torch.Generator().manual_seed(13)
    x = torch.rand((8, 3), generator=generator)
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    roots = torch.tensor([0, 2, 7, 0])
    return edges, x, labels, roots


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("architecture", ["graphsage", "gin"])
def test_rooted_logits_and_opacus_gradients_match_full_graph(radius, architecture):
    edges, x, labels, roots = _fixture()
    torch.manual_seed(29)
    model = _MultiHopDPGNN(3, 5, 2, radius=radius, architecture=architecture, dropout=0)
    parameters = tuple(model.parameters())
    full_logits = model(x, edges, torch.ones(edges.size(1)))[roots]
    losses = F.cross_entropy(full_logits, labels[roots], reduction="none")
    expected_gradients = [
        torch.autograd.grad(loss, parameters, retain_graph=True) for loss in losses
    ]
    batch, = iter_dpgnn_batches(
        roots, adjacency=build_adjacency(edges, len(x), direction="out"),
        x=x, y=labels, radius=radius, max_padded_nodes=1000, device=torch.device("cpu"))
    # Invalid padding must never enter root logits or gradients.
    batch.features.masked_fill_(~batch.node_mask.unsqueeze(-1), float("nan"))
    wrapped = GradSampleModule(_PaddedMultiHopDPGNN(model), loss_reduction="mean", strict=True)
    try:
        actual = wrapped(batch.features, batch.node_mask, batch.edge_index, batch.edge_mask)
        torch.testing.assert_close(actual, full_logits)
        F.cross_entropy(actual, batch.labels).backward()
        for index, parameter in enumerate(parameters):
            torch.testing.assert_close(
                parameter.grad_sample,
                torch.stack([gradient[index] for gradient in expected_gradients]),
                rtol=1e-5, atol=1e-6)
    finally:
        wrapped.to_standard_module()


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("architecture", ["graphsage", "gin"])
def test_physical_chunks_clip_complete_root_gradients_once(radius, architecture):
    edges, x, labels, roots = _fixture()
    torch.manual_seed(31)
    model = _MultiHopDPGNN(3, 5, 2, radius=radius, architecture=architecture, dropout=0)
    reference = deepcopy(model)
    reference_parameters = tuple(reference.parameters())
    logits = reference(x, edges, torch.ones(edges.size(1)))[roots]
    losses = F.cross_entropy(logits, labels[roots], reduction="none")
    gradients = torch.stack([
        torch.cat([gradient.flatten() for gradient in torch.autograd.grad(
            loss, reference_parameters, retain_graph=True)]) for loss in losses
    ])
    clip = 0.07
    factors = (clip / (gradients.norm(dim=1) + 1e-6)).clamp(max=1)
    expected = (gradients * factors[:, None]).mean(dim=0)
    expected_optimizer = torch.optim.Adam(reference_parameters, lr=0.01)
    for parameter, gradient in zip(reference_parameters, expected.split([
            parameter.numel() for parameter in reference_parameters])):
        parameter.grad = gradient.reshape_as(parameter)
    expected_optimizer.step()

    wrapped = GradSampleModule(_PaddedMultiHopDPGNN(model), loss_reduction="mean", strict=True)
    optimizer = DPOptimizer(
        torch.optim.Adam(model.parameters(), lr=0.01), noise_multiplier=0,
        max_grad_norm=clip, expected_batch_size=len(roots), loss_reduction="mean")
    trainer = PartitionedDPGNN(DPGNNConfig(
        num_classes=2, steps=1, batch_size=len(roots), noise_multiplier=1,
        radius=radius, architecture=architecture))
    try:
        trainer._private_step(wrapped, optimizer, iter_dpgnn_batches(
            roots, adjacency=build_adjacency(edges, len(x), direction="out"),
            x=x, y=labels, radius=radius, max_padded_nodes=4, device=torch.device("cpu")))
        for actual, wanted in zip(model.parameters(), reference_parameters):
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
    finally:
        wrapped.to_standard_module()


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_reverse_tree_attains_depth_sensitive_participation_bound(radius):
    count = 2 ** (radius + 1) - 1
    children = torch.arange(1, count)
    edges = torch.stack((children, (children - 1) // 2))
    occurrences = torch.zeros(count, dtype=torch.long)
    for batch in iter_dpgnn_batches(
            torch.arange(count), adjacency=build_adjacency(edges, count, direction="out"),
            x=torch.ones(count, 1), y=torch.zeros(count, dtype=torch.long), radius=radius,
            max_padded_nodes=7, device=torch.device("cpu")):
        occurrences += torch.bincount(batch.node_ids[batch.node_mask], minlength=count)
    bound = max_terms_per_node(2, radius)
    assert int(occurrences[0]) == bound
    assert bool((occurrences <= bound).all())


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_high_outdegree_neighborhood_is_never_truncated(radius):
    count = 131
    edges = torch.stack((torch.zeros(count - 1, dtype=torch.long), torch.arange(1, count)))
    batch, = iter_dpgnn_batches(
        torch.tensor([0]), adjacency=build_adjacency(edges, count, direction="out"),
        x=torch.arange(count).float()[:, None], y=torch.zeros(count, dtype=torch.long),
        radius=radius, max_padded_nodes=10, device=torch.device("cpu"))
    assert batch.node_ids[batch.node_mask].tolist() == list(range(count))
    assert batch.features[0, -1, 0] == count - 1


@pytest.mark.parametrize("radius", [2, 3])
@pytest.mark.parametrize("architecture", ["graphsage", "gin"])
def test_message_passing_reaches_exact_radius(radius, architecture):
    count = radius + 2
    edges = torch.stack((torch.arange(count - 1), torch.arange(1, count)))
    model = _MultiHopDPGNN(1, 2, 1, radius=radius, architecture=architecture, dropout=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.2)
        features = torch.ones(count, 1)
        baseline = model(features, edges, torch.ones(count - 1))[0]
        features[radius] += 1
        inside = model(features, edges, torch.ones(count - 1))[0]
        features[radius] -= 1
        features[radius + 1] += 1
        outside = model(features, edges, torch.ones(count - 1))[0]
    assert not torch.equal(inside, baseline)
    torch.testing.assert_close(outside, baseline, rtol=0, atol=0)
