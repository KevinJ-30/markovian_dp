"""DP-GNN Bernoulli edge sampling, root law, and padded star contracts."""

import pytest
import torch
from torch_geometric.data import Data

from src.processing.dpgnn import (
    iter_dpgnn_batches,
    sample_dpgnn_roots,
    sample_training_edges,
)
from src.processing.sparse_expand import build_adjacency


def _scalar_sample(data, max_degree, uniforms):
    incoming = [[] for _ in range(data.num_nodes)]
    original = list(zip(*data.edge_index.tolist()))
    for sender, receiver in original + [(v, u) for u, v in original]:
        incoming[receiver].append(sender)
    draws = iter(uniforms)
    arcs = [(node, node) for node in range(data.num_nodes)]
    for receiver, candidates in enumerate(incoming):
        if not candidates:
            continue
        probability = max_degree / (2.0 * len(candidates))
        selected = []
        for sender in candidates:
            if next(draws) <= probability:
                selected.append(sender)
        unique = sorted(set(selected))
        if len(unique) <= max_degree:
            arcs.extend((sender, receiver) for sender in unique)
    return torch.tensor(arcs, dtype=torch.long).reshape(-1, 2).t().contiguous()


def _multiplicity_graph():
    return Data(
        num_nodes=8,
        edge_index=torch.tensor([
            [0, 0, 1, 2, 3, 4, 5],
            [1, 1, 0, 1, 1, 4, 6],
        ]),
    )


def _controlled_draws(monkeypatch, values):
    uniforms = torch.tensor(values)

    def draw(*shape, **kwargs):
        return uniforms.reshape(*shape)

    monkeypatch.setattr("src.processing.dpgnn.torch.rand", draw)


def _outgoing_adjacency(edges, num_nodes):
    return build_adjacency(
        edges[:, edges[0] != edges[1]], num_nodes, direction="out")


def _gather_fixture():
    # Receiver-major arcs produce ascending neighbors after stable CSR grouping.
    edges = torch.tensor([[0, 0, 1, 0, 0, 2, 2], [1, 2, 2, 3, 4, 4, 6]])
    adjacency = _outgoing_adjacency(edges, 7)
    x = torch.arange(1, 22, dtype=torch.float32).reshape(7, 3)
    y = torch.tensor([1, 0, 2, 1, 2, 0, 1])
    return adjacency, x, y


def test_sampling_draws_multiplicities_before_dedup_and_drops_overflow(monkeypatch):
    data = _multiplicity_graph()
    # Receiver 0 retains a later duplicate; receiver 1 selects three distinct
    # predecessors and is discarded entirely. Receiver 7 has no candidates.
    uniforms = [0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.7, 0.8, 0.5, 0.9, 0.9, 0.9]
    _controlled_draws(monkeypatch, uniforms)
    sampled = sample_training_edges(data, max_degree=2, seed=41)
    expected = torch.tensor([
        list(range(8)) + [1, 1, 1, 4, 6, 5],
        list(range(8)) + [0, 2, 3, 4, 5, 6],
    ])
    assert torch.equal(sampled, expected)
    assert torch.equal(sampled, _scalar_sample(data, 2, uniforms))
    # The <= boundary retains (4, 4); its explicit self-loop is not deduplicated.
    assert int(((sampled[0] == 4) & (sampled[1] == 4)).sum()) == 2


def test_sampling_matches_seeded_scalar_reference_without_global_rng_use():
    data = _multiplicity_graph()
    seed = 27
    generator = torch.Generator().manual_seed(seed)
    uniforms = torch.rand(2 * data.edge_index.shape[1], generator=generator).tolist()
    state = torch.random.get_rng_state()
    sampled = sample_training_edges(data, max_degree=2, seed=seed)
    assert torch.equal(sampled, _scalar_sample(data, 2, uniforms))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(sample_training_edges(data, max_degree=2, seed=seed), sampled)


def test_sampling_probability_above_one_keeps_every_candidate():
    data = Data(num_nodes=3, edge_index=torch.tensor([[0], [1]]))
    sampled = sample_training_edges(data, max_degree=4, seed=12)
    assert sampled.tolist() == [[0, 1, 2, 1, 0], [0, 1, 2, 0, 1]]


def test_sampling_all_rejected_candidates_returns_only_explicit_loops(monkeypatch):
    data = Data(num_nodes=2, edge_index=torch.tensor([[0], [1]]))
    _controlled_draws(monkeypatch, [0.9, 0.9])
    sampled = sample_training_edges(data, max_degree=1, seed=0)
    assert sampled.tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize("num_nodes", [0, 3])
def test_sampling_empty_edges_returns_exact_explicit_loops(num_nodes):
    data = Data(num_nodes=num_nodes, edge_index=torch.empty((2, 0), dtype=torch.long))
    sampled = sample_training_edges(data, max_degree=1, seed=3)
    expected = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    assert torch.equal(sampled, expected)
    assert sampled.shape == (2, num_nodes)
    assert sampled.dtype == torch.long
    assert sampled.device.type == "cpu"


def test_sampling_rejects_nonpositive_degree():
    data = Data(num_nodes=0, edge_index=torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(ValueError):
        sample_training_edges(data, max_degree=0, seed=0)


def test_root_batches_are_distinct_valid_and_locally_reproducible():
    left = torch.Generator().manual_seed(19)
    right = torch.Generator().manual_seed(19)
    state = torch.random.get_rng_state()
    draws = []
    for _ in range(32):
        roots = sample_dpgnn_roots(6, 2, generator=left)
        assert roots.shape == (2,)
        assert roots.dtype == torch.long
        assert roots.device.type == "cpu"
        assert roots.unique().numel() == 2
        assert bool(((roots >= 0) & (roots < 6)).all())
        assert torch.equal(roots, sample_dpgnn_roots(6, 2, generator=right))
        draws.append(tuple(sorted(roots.tolist())))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert len(set(draws)) > 1


def test_full_population_root_batch_contains_each_node_once():
    roots = sample_dpgnn_roots(6, 6, generator=torch.Generator().manual_seed(4))
    assert torch.equal(roots.sort().values, torch.arange(6))


@pytest.mark.parametrize("num_nodes,batch_size", [(0, 1), (6, 0), (6, 7)])
def test_root_sampling_rejects_invalid_population_or_batch(num_nodes, batch_size):
    with pytest.raises(ValueError):
        sample_dpgnn_roots(num_nodes, batch_size, generator=torch.Generator())


def test_gather_preserves_order_repeated_roots_truncation_and_star_orientation():
    adjacency, x, y = _gather_fixture()
    roots = torch.tensor([5, 0, 6, 2, 0, 1])
    batches = list(iter_dpgnn_batches(
        roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=3,
        max_padded_nodes=100, device=torch.device("cpu")))
    assert len(batches) == 1
    batch = batches[0]
    expected_nodes = [[5], [0, 1, 2], [6], [2, 4, 6], [0, 1, 2], [1, 2]]
    assert torch.equal(batch.roots, roots)
    assert torch.equal(batch.labels, y[roots])
    assert batch.root_index.tolist() == [0] * len(expected_nodes)
    assert batch.loss_mask.tolist() == [True] * len(expected_nodes)
    assert torch.equal(batch.node_mask, batch.edge_mask)
    for row, nodes in enumerate(expected_nodes):
        mask = batch.node_mask[row]
        assert batch.node_ids[row, mask].tolist() == nodes
        assert torch.equal(batch.features[row, mask], x[nodes])
        assert torch.equal(batch.features[row, ~mask], torch.zeros_like(batch.features[row, ~mask]))
        assert bool((batch.node_ids[row, ~mask] == 0).all())
        assert batch.edge_index[row, :, mask].tolist() == [[0] * len(nodes), list(range(len(nodes)))]
        assert bool((batch.edge_index[row, :, ~mask] == 0).all())


@pytest.mark.parametrize("budget", [4, 6])
def test_gather_uses_ordered_padded_slot_budget_and_preserves_oversized_star(budget):
    adjacency, x, y = _gather_fixture()
    roots = torch.tensor([5, 1, 2, 0, 6, 1])
    batches = list(iter_dpgnn_batches(
        roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=100,
        max_padded_nodes=budget, device=torch.device("cpu")))
    # At budget 6, real sizes 1+2+3 would fit, but three padded stars cost 9.
    assert [batch.roots.tolist() for batch in batches] == [[5, 1], [2], [0], [6, 1]]
    assert batches[2].node_ids.tolist() == [[0, 1, 2, 3, 4]]
    for batch in batches:
        slots = batch.node_mask.numel()
        assert slots <= budget or (batch.batch_size == 1 and slots == 5)
    assert torch.equal(torch.cat([batch.roots for batch in batches]), roots)


def test_empty_csr_gathers_isolated_repeated_roots_without_sentinel():
    adjacency = _outgoing_adjacency(torch.empty((2, 0), dtype=torch.long), 3)
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    y = torch.tensor([1, 0, 1])
    roots = torch.tensor([2, 0, 2])
    batches = list(iter_dpgnn_batches(
        roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=100,
        max_padded_nodes=3, device=torch.device("cpu")))
    assert len(batches) == 1
    batch = batches[0]
    assert torch.equal(batch.node_ids, roots.unsqueeze(1))
    assert torch.equal(batch.features, x[roots].unsqueeze(1))
    assert torch.equal(batch.labels, y[roots])
    assert batch.node_mask.tolist() == [[True], [True], [True]]
    assert batch.edge_mask.tolist() == [[True], [True], [True]]
    assert torch.equal(batch.edge_index, torch.zeros((3, 2, 1), dtype=torch.long))


def test_root_only_cap_ignores_existing_neighbors():
    adjacency, x, y = _gather_fixture()
    roots = torch.tensor([0, 2])
    batch, = iter_dpgnn_batches(
        roots, adjacency=adjacency, x=x, y=y, max_subgraph_nodes=1,
        max_padded_nodes=2, device=torch.device("cpu"))
    assert torch.equal(batch.node_ids, roots.unsqueeze(1))
    assert torch.equal(batch.features, x[roots].unsqueeze(1))
    assert batch.edge_index.tolist() == [[[0], [0]], [[0], [0]]]


def test_empty_roots_yield_no_batches_even_for_empty_graph():
    adjacency = _outgoing_adjacency(torch.empty((2, 0), dtype=torch.long), 0)
    batches = list(iter_dpgnn_batches(
        torch.empty(0, dtype=torch.long), adjacency=adjacency,
        x=torch.empty((0, 2)), y=torch.empty(0, dtype=torch.long),
        max_subgraph_nodes=3, max_padded_nodes=3, device=torch.device("cpu")))
    assert batches == []


@pytest.mark.parametrize("subgraph_limit,slot_limit", [(0, 4), (4, 0)])
def test_gather_rejects_nonpositive_limits(subgraph_limit, slot_limit):
    adjacency, x, y = _gather_fixture()
    with pytest.raises(ValueError):
        list(iter_dpgnn_batches(
            torch.tensor([0]), adjacency=adjacency, x=x, y=y,
            max_subgraph_nodes=subgraph_limit, max_padded_nodes=slot_limit,
            device=torch.device("cpu")))


def test_incoming_degree_bound_limits_participation_not_root_star_size(monkeypatch):
    data = Data(num_nodes=5, edge_index=torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]))
    _controlled_draws(monkeypatch, [0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0])
    sampled = sample_training_edges(data, max_degree=1, seed=0)
    assert sampled.tolist() == [list(range(5)) + [0, 0, 0, 0], list(range(5)) + [1, 2, 3, 4]]
    adjacency = _outgoing_adjacency(sampled, 5)
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    y = torch.zeros(5, dtype=torch.long)
    batch, = iter_dpgnn_batches(
        torch.arange(5), adjacency=adjacency, x=x, y=y,
        max_subgraph_nodes=100, max_padded_nodes=25, device=torch.device("cpu"))
    assert batch.node_ids[0, batch.node_mask[0]].tolist() == [0, 1, 2, 3, 4]
    occurrences = torch.bincount(batch.node_ids[batch.node_mask], minlength=5)
    assert occurrences.tolist() == [1, 2, 2, 2, 2]
    assert bool((occurrences <= 2).all())
    truncated, = iter_dpgnn_batches(
        torch.tensor([0]), adjacency=adjacency, x=x, y=y,
        max_subgraph_nodes=3, max_padded_nodes=3, device=torch.device("cpu"))
    assert truncated.node_ids.tolist() == [[0, 1, 2]]
