"""
Tests for incoming SparseExpand (Algorithm 5), root sampling, and SparseGNN.
"""

import math

import pytest
import torch

from src.processing.sparse_expand import (
    SparseAdjacency, batch_sparse_expand, build_adjacency,
    sample_roots, sparse_expand,
)


def _toy_graph():
    # Directed chain 0->1->2->3 plus a branch 1->4.  num_nodes=5.
    edge_index = torch.tensor([[0, 1, 2, 1],
                               [1, 2, 3, 4]], dtype=torch.long)
    return edge_index, 5


def _reachable(adj, root, r):
    """Deterministic BFS reachable set within r hops (ground truth for p2=1)."""
    seen = {root}
    frontier = [root]
    for _ in range(r):
        nxt = []
        for u in frontier:
            for w in adj.neighbors(u).tolist():
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return seen


def _assert_same_subgraph(actual, expected):
    assert actual.root == expected.root
    assert torch.equal(actual.nodes, expected.nodes)
    assert torch.equal(actual.edge_index, expected.edge_index)


def test_batch_sparse_expand_empty_and_preserves_root_order():
    edge_index, n = _toy_graph()
    adjacency = build_adjacency(edge_index, n, direction='in')
    assert batch_sparse_expand(
        adjacency, torch.empty(0, dtype=torch.long), p2=1.0, r=3) == []

    roots = torch.tensor([3, 1, 3, 0], dtype=torch.long)
    subgraphs = batch_sparse_expand(
        adjacency, roots, p2=1.0, r=3)
    assert [subgraph.root for subgraph in subgraphs] == roots.tolist()
    for subgraph, root in zip(subgraphs, roots.tolist()):
        _assert_same_subgraph(
            subgraph,
            sparse_expand(adjacency, root, p2=1.0, r=3),
        )


@pytest.mark.parametrize('p2', [0.0, 1.0])
def test_batch_sparse_expand_matches_scalar_at_probability_boundaries(p2):
    edge_index, n = _toy_graph()
    adjacency = build_adjacency(edge_index, n)
    roots = torch.tensor([3, 1, 4, 0, 1], dtype=torch.long)
    batch_generator = torch.Generator().manual_seed(19)
    untouched_generator = torch.Generator().manual_seed(19)
    actual = batch_sparse_expand(
        adjacency, roots, p2=p2, r=4, generator=batch_generator)
    expected = [
        sparse_expand(adjacency, root, p2=p2, r=4)
        for root in roots.tolist()
    ]
    for batch_subgraph, scalar_subgraph in zip(actual, expected):
        _assert_same_subgraph(batch_subgraph, scalar_subgraph)
    assert torch.equal(
        torch.rand(8, generator=batch_generator),
        torch.rand(8, generator=untouched_generator),
    )


def test_batch_sparse_expand_keeps_converging_and_parallel_arcs():
    edge_index = torch.tensor(
        [[1, 2, 3, 3, 3], [0, 0, 1, 1, 2]], dtype=torch.long)
    adjacency = build_adjacency(edge_index, 4)
    subgraph = batch_sparse_expand(
        adjacency, torch.tensor([0]), p2=1.0, r=2)[0]

    assert subgraph.nodes.tolist() == [0, 1, 2, 3]
    assert subgraph.edge_index.tolist() == [
        [1, 2, 3, 3, 3],
        [0, 0, 1, 1, 2],
    ]


def test_batch_sparse_expand_is_deterministic_under_fixed_seed():
    edge_index = torch.tensor(
        [[0, 1, 2, 1, 1, 3, 3], [1, 2, 3, 3, 3, 1, 3]],
        dtype=torch.long)
    adjacency = build_adjacency(edge_index, 4)
    roots = torch.tensor([0, 3, 1, 1, 2], dtype=torch.long)
    first = batch_sparse_expand(
        adjacency, roots, p2=0.37, r=3,
        generator=torch.Generator().manual_seed(91))
    second = batch_sparse_expand(
        adjacency, roots, p2=0.37, r=3,
        generator=torch.Generator().manual_seed(91))
    for expected, actual in zip(first, second):
        _assert_same_subgraph(actual, expected)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('radius', [1, 2, 3, 4])
def test_incoming_cap_applies_at_every_expansion_hop(batched, radius):
    frontier = torch.tensor([0])
    size = 1
    levels = []
    for degree in (21, 11, 6, 6)[:radius]:
        children = torch.arange(size, size + frontier.numel() * degree)
        levels.append(torch.stack((children, frontier.repeat_interleave(degree))))
        size += children.numel()
        frontier = children
    edges = torch.cat(levels, dim=1)
    adj = build_adjacency(edges, size, direction='in')
    generator = torch.Generator().manual_seed(7)
    if batched:
        sg = batch_sparse_expand(adj, torch.tensor([0]), 1.0, radius, generator=generator)[0]
    else:
        sg = sparse_expand(adj, 0, 1.0, radius, generator=generator)
    incoming = torch.bincount(sg.edge_index[1], minlength=sg.num_nodes)
    offset, width = 0, 1
    for cap in (20, 10, 5, 5)[:radius]:
        assert torch.all(incoming[offset:offset + width] == cap)
        offset += width
        width *= cap
    assert torch.all(incoming[offset:] == 0)
    assert sg.num_nodes == offset + width
    assert sg.num_edges == sg.num_nodes - 1
    assert sg.nodes.unique().numel() == sg.num_nodes
    real_edges = set(map(tuple, edges.t().tolist()))
    assert all(tuple(edge) in real_edges for edge in sg.nodes[sg.edge_index].t().tolist())


def test_incoming_cap_handles_mixed_degrees_and_zero_probability():
    degrees = torch.tensor([0, 5, 20, 21, 1000])
    sources = torch.arange(5, 5 + int(degrees.sum()))
    edges = torch.stack((sources, torch.repeat_interleave(torch.arange(5), degrees)))
    adj = build_adjacency(edges, 5 + int(degrees.sum()), direction='in')
    roots = torch.arange(5)
    full = batch_sparse_expand(
        adj, roots, 1.0, 1, generator=torch.Generator().manual_seed(3))
    assert [sg.num_edges for sg in full] == [0, 5, 20, 20, 20]
    empty = batch_sparse_expand(adj, roots, 0.0, 1)
    assert [sg.nodes.tolist() for sg in empty] == [[root] for root in roots.tolist()]


def test_capped_incoming_counts_and_neighbor_selection_are_unbiased():
    degree, p2, trials = 40, 0.5, 2000
    edges = torch.stack((torch.arange(1, degree + 1), torch.zeros(degree, dtype=torch.long)))
    adj = build_adjacency(edges, degree + 1, direction='in')
    subgraphs = batch_sparse_expand(
        adj, torch.zeros(trials, dtype=torch.long), p2, 1,
        generator=torch.Generator().manual_seed(17))
    counts = torch.tensor([sg.num_edges for sg in subgraphs], dtype=torch.float64)
    probabilities = [math.comb(degree, k) * p2**k * (1-p2)**(degree-k)
                     for k in range(degree + 1)]
    expected = sum(min(k, 20) * prob for k, prob in enumerate(probabilities))
    variance = sum((min(k, 20) - expected)**2 * prob
                   for k, prob in enumerate(probabilities))
    assert float(counts.mean()) == pytest.approx(expected, abs=0.15)
    assert float(counts.var()) == pytest.approx(variance, rel=0.15)
    frequencies = torch.bincount(torch.cat([sg.nodes[1:] for sg in subgraphs]))[1:]
    assert torch.all((frequencies - trials * expected / degree).abs()
                     < 0.1 * trials * expected / degree)
    assert all(sg.nodes.unique().numel() == sg.num_nodes for sg in subgraphs)


def test_p2_one_matches_reachable_set():
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=1.0, r=10)
        assert sg.root == root
        assert int(sg.nodes[0]) == root           # root is local index 0
        assert set(sg.nodes.tolist()) == _reachable(adj, root, 10)


def test_in_expansion_reaches_backward_neighbours():
    """In-expansion from node 3 must collect the chain 0->1->2->3 backwards."""
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction='in')
    sg = sparse_expand(adj, 3, p2=1.0, r=10)
    assert set(sg.nodes.tolist()) == {3, 2, 1, 0}


def test_in_expansion_orients_edges_toward_root():
    """In-expansion must deliver neighbour features to the root.

    Under Algorithm 5 every level-1 arc must have the root (local index 0) as
    its TARGET, so a message-passing layer actually delivers the neighbour's
    features to the root.
    """
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction='in')
    sg = sparse_expand(adj, 2, p2=1.0, r=1)
    assert sg.num_edges > 0
    # every retained arc points INTO the root
    assert sg.edge_index[1].tolist() == [0] * sg.num_edges
    assert 0 not in sg.edge_index[0].tolist()

def test_p2_zero_is_isolated_root():
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n)
    sg = sparse_expand(adj, 0, p2=0.0, r=5)
    assert sg.nodes.tolist() == [0]
    assert sg.num_edges == 0


def test_edges_are_real_and_local():
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n)
    real = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    gen = torch.Generator().manual_seed(7)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=0.7, r=3, generator=gen)
        # local indices are within range
        if sg.num_edges:
            assert int(sg.edge_index.max()) < sg.num_nodes
            # remap to original ids: every retained arc must exist in G with the
            # SAME orientation it had there (Algorithm 5 line 8).
            src = sg.nodes[sg.edge_index[0]]
            dst = sg.nodes[sg.edge_index[1]]
            for u, v in zip(src.tolist(), dst.tolist()):
                assert (u, v) in real


@pytest.mark.parametrize('p2', [0.0, 0.37, 1.0])
def test_csr_adjacency_has_seeded_expected_expansions(p2):
    edge_index = torch.tensor(
        [[0, 1, 2, 1, 1, 3, 3], [1, 2, 3, 3, 3, 1, 3]],
        dtype=torch.long)
    adjacency = build_adjacency(edge_index, 4)
    assert isinstance(adjacency, SparseAdjacency)
    first = [sparse_expand(adjacency, root, p2, 3,
                           generator=torch.Generator().manual_seed(91 + root))
             for root in range(4)]
    second = [sparse_expand(adjacency, root, p2, 3,
                            generator=torch.Generator().manual_seed(91 + root))
              for root in range(4)]
    for expected, actual in zip(first, second):
        assert torch.equal(expected.nodes, actual.nodes)
        assert torch.equal(expected.edge_index, actual.edge_index)


@pytest.mark.parametrize("expand,root", [
    (sparse_expand, 0),
    (batch_sparse_expand, torch.tensor([0])),
])
def test_expansion_rejects_outgoing_adjacency(expand, root):
    edge_index, n = _toy_graph()
    adjacency = build_adjacency(edge_index, n, direction='out')
    with pytest.raises(ValueError):
        expand(adjacency, root, .5, 1)




def test_determinism_under_fixed_seed():
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sparse_expand(adj, 0, p2=0.5, r=3, generator=g1)
    b = sparse_expand(adj, 0, p2=0.5, r=3, generator=g2)
    assert a.nodes.tolist() == b.nodes.tolist()
    assert a.edge_index.tolist() == b.edge_index.tolist()


def test_root_sampling_expected_count():
    n, p1 = 2000, 0.3
    gen = torch.Generator().manual_seed(0)
    counts = [sample_roots(n, p1, generator=gen).numel() for _ in range(20)]
    mean = sum(counts) / len(counts)
    assert math.isclose(mean, p1 * n, rel_tol=0.1)


def test_root_sampling_p1_one_returns_all():
    roots = sample_roots(50, 1.0)
    assert roots.tolist() == list(range(50))


def test_sparse_gnn_smoke_reduces_loss():
    from torch_geometric.datasets import Planetoid
    from src.models.gnn_mechanism import GNNMechanism
    from src.training.sparse_gnn import train_sparse_gnn

    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    data = dataset[0]
    device = torch.device('cpu')

    torch.manual_seed(0)
    adj = build_adjacency(data.edge_index, int(data.num_nodes), direction='in')
    mech = GNNMechanism(data, dataset.num_features, dataset.num_classes,
                        hidden=16, num_layers=2, device=device)
    mech.build_optimizer(lr=0.01, weight_decay=5e-4, kind='adam')

    cand = torch.where(data.train_mask)[0]
    # subgraph_loss on a labeled root is a finite scalar
    root = int(cand[0])
    sg = sparse_expand(adj, root, p2=1.0, r=2)
    loss0 = mech.subgraph_loss(sg)
    assert torch.isfinite(loss0)

    accs = train_sparse_gnn(
        mech, data, data, adj=adj, p1=1.0, p2=1.0,
        r=2, T=30, seed=0)
    # After 30 full-batch steps on CiteSeer, train accuracy should clear chance.
    assert accs['train'] > 1.0 / dataset.num_classes


def test_in_expansion_actually_reaches_the_root_representation():
    """The root's GNN output must depend on the retained incoming neighbours."""
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    conv = GCNConv(4, 3, add_self_loops=True, normalize=True)
    isolated = conv(x, torch.zeros((2, 0), dtype=torch.long))[0]
    edges = torch.tensor([[1], [0]], dtype=torch.long)
    subgraph = sparse_expand(build_adjacency(edges, 2), 0, p2=1.0, r=1)
    toward_root = conv(x[subgraph.nodes], subgraph.edge_index)[0]

    assert not torch.allclose(toward_root, isolated)
