"""Regression coverage for disconnected SparseGNN forward batching."""

import pytest
import torch
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from torch_geometric.data import Data

from src.models.gnn_mechanism import GNNMechanism
from src.processing.padded import pad_rooted_subgraphs
from src.training.sparse_gnn import OpacusPrivateUpdate
from src.processing.sparse_expand import RootedSubgraph


def _data():
    return Data(
        x=torch.tensor([
            [1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]),
        y=torch.tensor([0, 1, 0, 1, 0]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        train_mask=torch.tensor([True, True, True, True, False]),
    )


def _subgraphs():
    return [
        RootedSubgraph(0, torch.tensor([0, 1]), torch.tensor([[1], [0]])),
        RootedSubgraph(2, torch.tensor([2, 3]), torch.tensor([[1], [0]])),
        RootedSubgraph(4, torch.tensor([4]), torch.zeros((2, 0), dtype=torch.long)),
    ]


def _mechanism(data, *, max_nodes, aggr="mean"):
    return GNNMechanism(
        data, 3, 2, hidden=4, num_layers=2, dropout=0.0, aggr=aggr,
        device=torch.device('cpu'), max_batched_subgraph_nodes=max_nodes,
    )


def _clipped_sum(mechanism, losses, clip=1.0):
    params = mechanism.parameters()
    accum = [torch.zeros_like(parameter) for parameter in params]
    active = [loss for loss in losses if float(loss.detach()) != 0.0]
    for index, loss in enumerate(active):
        grads = torch.autograd.grad(
            loss, params, retain_graph=index < len(active) - 1,
            allow_unused=True,
        )
        grads = [grad if grad is not None else torch.zeros_like(parameter)
                 for grad, parameter in zip(grads, params)]
        norm = torch.sqrt(sum(grad.square().sum() for grad in grads))
        scale = min(1.0, clip / (float(norm) + 1e-12))
        for total, grad in zip(accum, grads):
            total.add_(grad, alpha=scale)
    return accum


def test_batched_losses_and_clipped_gradients_match_sequential():
    torch.manual_seed(4)
    data = _data()
    sequential = _mechanism(data, max_nodes=1)
    batched = _mechanism(data, max_nodes=32)
    batched.module.load_state_dict(sequential.module.state_dict())
    subgraphs = _subgraphs()

    reference_losses = [sequential.subgraph_loss(subgraph) for subgraph in subgraphs]
    batched_losses = batched.subgraph_losses(subgraphs)

    assert len(batched_losses) == len(reference_losses)
    for got, expected in zip(batched_losses, reference_losses):
        assert torch.allclose(got, expected, atol=1e-6)
    assert batched_losses[-1].requires_grad

    reference_grads = _clipped_sum(sequential, reference_losses)
    batched_grads = _clipped_sum(batched, batched_losses)
    for got, expected in zip(batched_grads, reference_grads):
        assert torch.allclose(got, expected, atol=1e-6)


def test_unsupervised_root_is_differentiable_zero_without_gradient_signal():
    mechanism = _mechanism(_data(), max_nodes=32)
    loss = mechanism.subgraph_losses([_subgraphs()[-1]])[0]

    assert loss.requires_grad
    assert float(loss.detach()) == 0.0
    grads = torch.autograd.grad(loss, mechanism.parameters(), allow_unused=True)
    for grad, parameter in zip(grads, mechanism.parameters()):
        assert grad is None or torch.equal(grad, torch.zeros_like(parameter))


def test_chunked_and_oversized_fallback_match_unbounded_batch():
    torch.manual_seed(9)
    data = _data()
    chunked = _mechanism(data, max_nodes=2)
    unbounded = _mechanism(data, max_nodes=32)
    unbounded.module.load_state_dict(chunked.module.state_dict())
    subgraphs = _subgraphs()

    chunked_losses = chunked.subgraph_losses(subgraphs)
    unbounded_losses = unbounded.subgraph_losses(subgraphs)
    for got, expected in zip(chunked_losses, unbounded_losses):
        assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("aggr", ["mean", "gcn", "gin"])
def test_padded_losses_match_sparse_pyg(aggr):
    torch.manual_seed(12)
    data = _data()
    mechanism = _mechanism(data, max_nodes=32, aggr=aggr)
    subgraphs = _subgraphs()
    reference = torch.stack([
        mechanism.subgraph_loss(subgraph) for subgraph in subgraphs])
    batch = pad_rooted_subgraphs(
        subgraphs, x=data.x, y=data.y, train_mask=data.train_mask,
        device=torch.device("cpu"))
    private_module = mechanism.build_private_module()
    private_module.eval()
    padded = mechanism.private_losses(private_module, batch)

    assert torch.allclose(padded, reference, atol=1e-6)


@pytest.mark.parametrize("aggr", ["mean", "gcn", "gin"])
@pytest.mark.parametrize("empty", [False, True])
def test_private_update_matches_manual_clipping_noise_and_sgd(aggr, empty):
    torch.manual_seed(21)
    mechanism = _mechanism(_data(), max_nodes=3, aggr=aggr)
    mechanism.build_optimizer(lr=.05, weight_decay=0., kind="sgd")
    subgraphs = [] if empty else [
        RootedSubgraph(0, torch.tensor([0, 1, 2]),
                       torch.tensor([[1, 2, 0], [0, 0, 0]])),
        *_subgraphs()[1:],
    ]
    parameters = list(mechanism.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    totals = [torch.zeros_like(parameter) for parameter in parameters]
    for subgraph in subgraphs:
        gradients = torch.autograd.grad(
            mechanism.subgraph_loss(subgraph), parameters, allow_unused=True)
        gradients = [torch.zeros_like(parameter) if gradient is None else gradient
                     for parameter, gradient in zip(parameters, gradients)]
        norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
        factor = (.7 / (norm + 1e-6)).clamp(max=1.)
        for total, gradient in zip(totals, gradients):
            total.add_(gradient * factor)
    generator = torch.Generator().manual_seed(7)
    expected = [
        (total + torch.normal(0., .4 * .7, size=total.shape, generator=generator)) / 4
        for total in totals
    ]
    update = OpacusPrivateUpdate(
        mechanism, C=.7, sigma=.4, expected_batch=4.,
        noise_gen=torch.Generator().manual_seed(7))
    update.step(subgraphs)

    for parameter, initial, gradient in zip(parameters, before, expected):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(parameter, initial - .05 * gradient, atol=1e-6, rtol=1e-5)


def test_private_physical_chunks_match_one_padded_batch():
    torch.manual_seed(24)
    data = _data()
    chunked = _mechanism(data, max_nodes=2)
    unbounded = _mechanism(data, max_nodes=32)
    unbounded.module.load_state_dict(chunked.module.state_dict())
    for mechanism in (chunked, unbounded):
        mechanism.build_optimizer(lr=0.0, kind="sgd")
    chunked_update = OpacusPrivateUpdate(
        chunked, C=1.0, sigma=0.0, expected_batch=2.0,
        noise_gen=torch.Generator().manual_seed(1))
    unbounded_update = OpacusPrivateUpdate(
        unbounded, C=1.0, sigma=0.0, expected_batch=2.0,
        noise_gen=torch.Generator().manual_seed(1))

    chunked_update.step(_subgraphs()[:2])
    unbounded_update.step(_subgraphs()[:2])

    for left, right in zip(chunked.parameters(), unbounded.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-6)


@pytest.mark.parametrize("aggr", ["mean", "gcn", "gin"])
def test_opacus_grad_samples_match_vmap_per_root_gradients(aggr):
    torch.manual_seed(31)
    data = _data()
    mechanism = _mechanism(data, max_nodes=32, aggr=aggr)
    batch = pad_rooted_subgraphs(
        _subgraphs()[:2], x=data.x, y=data.y, train_mask=data.train_mask,
        device=torch.device("cpu"))
    private_module = mechanism.build_private_module()
    private_module.eval()
    parameters = dict(private_module.named_parameters())
    buffers = dict(private_module.named_buffers())

    def single_loss(params, state, features, node_mask, edge_index,
                    edge_mask, root_index, label, loss_mask):
        output = torch.func.functional_call(
            private_module, (params, state),
            (features.unsqueeze(0), edge_index.unsqueeze(0),
             edge_mask.unsqueeze(0), node_mask.unsqueeze(0)))
        root_logits = F.log_softmax(output[0, 0], dim=-1)
        loss = F.nll_loss(root_logits.unsqueeze(0), label.long().view(1))
        return loss * loss_mask.to(loss.dtype)

    reference = torch.func.vmap(
        torch.func.grad(single_loss),
        in_dims=(None, None, 0, 0, 0, 0, 0, 0, 0),
    )(
        parameters, buffers, batch.features, batch.node_mask,
        batch.edge_index, batch.edge_mask, batch.root_index, batch.labels,
        batch.loss_mask)

    wrapped = GradSampleModule(
        private_module, batch_first=True, loss_reduction="mean", strict=True)
    wrapped.train()
    mechanism.private_losses(wrapped, batch).mean().backward()
    for name, parameter in private_module.named_parameters():
        assert parameter.grad_sample.shape[0] == batch.batch_size
        assert torch.allclose(
            parameter.grad_sample, reference[name], atol=1e-6, rtol=1e-5)
