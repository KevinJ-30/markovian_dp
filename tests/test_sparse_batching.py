"""Regression coverage for disconnected SparseGNN forward batching."""

import torch
from torch_geometric.data import Data

from src.sparse.gnn_mechanism import GNNMechanism
from src.sparse.sparse_expand import RootedSubgraph


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


def _mechanism(data, *, max_nodes):
    return GNNMechanism(
        data, 3, 2, hidden=4, num_layers=2, dropout=0.0,
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
        for total, grad in zip(accum, mechanism.clip_flat_grad(grads, clip)):
            total.add_(grad)
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
