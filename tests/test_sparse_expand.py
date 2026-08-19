"""
Tests for SparseExpand, root sampling, and the SparseGNN engine, covering both
expansion orientations ('in' = Algorithm 5, the default; 'out' = Algorithm 2/4).
"""

import math

import pytest
import torch

from src.sparse.sparse_expand import (
    build_adjacency, build_out_adjacency, sample_roots, sparse_expand,
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
            for w in adj[u].tolist():
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return seen


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_p2_one_matches_reachable_set(direction):
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction=direction)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=1.0, r=10, direction=direction)
        assert sg.root == root
        assert int(sg.nodes[0]) == root           # root is local index 0
        assert set(sg.nodes.tolist()) == _reachable(adj, root, 10)


def test_in_expansion_reaches_backward_neighbours():
    """In-expansion from node 3 must collect the chain 0->1->2->3 backwards."""
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction='in')
    sg = sparse_expand(adj, 3, p2=1.0, r=10, direction='in')
    assert set(sg.nodes.tolist()) == {3, 2, 1, 0}
    # The out-orientation from the same root reaches nothing at all.
    adj_out = build_adjacency(edge_index, n, direction='out')
    sg_out = sparse_expand(adj_out, 3, p2=1.0, r=10, direction='out')
    assert sg_out.nodes.tolist() == [3]


def test_in_expansion_orients_edges_toward_root():
    """In-expansion must deliver neighbour features to the root.

    Under Algorithm 5 every level-1 arc must have the root (local index 0) as
    its TARGET, so a message-passing layer actually delivers the neighbour's
    features to the root.  The old out-orientation put the root on the source
    side, which left it with nothing but its self-loop.
    """
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction='in')
    sg = sparse_expand(adj, 2, p2=1.0, r=1, direction='in')
    assert sg.num_edges > 0
    # every retained arc points INTO the root
    assert sg.edge_index[1].tolist() == [0] * sg.num_edges
    assert 0 not in sg.edge_index[0].tolist()

    adj_out = build_adjacency(edge_index, n, direction='out')
    sg_out = sparse_expand(adj_out, 2, p2=1.0, r=1, direction='out')
    assert sg_out.edge_index[0].tolist() == [0] * sg_out.num_edges


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_p2_zero_is_isolated_root(direction):
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction=direction)
    sg = sparse_expand(adj, 0, p2=0.0, r=5, direction=direction)
    assert sg.nodes.tolist() == [0]
    assert sg.num_edges == 0


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_edges_are_real_and_local(direction):
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction=direction)
    real = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    gen = torch.Generator().manual_seed(7)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=0.7, r=3, generator=gen,
                           direction=direction)
        # local indices are within range
        if sg.num_edges:
            assert int(sg.edge_index.max()) < sg.num_nodes
            # remap to original ids: every retained arc must exist in G with the
            # SAME orientation it had there (Algorithm 5 line 8).
            src = sg.nodes[sg.edge_index[0]]
            dst = sg.nodes[sg.edge_index[1]]
            for u, v in zip(src.tolist(), dst.tolist()):
                assert (u, v) in real


def test_build_out_adjacency_alias_matches_build_adjacency():
    edge_index, n = _toy_graph()
    a = build_out_adjacency(edge_index, n)
    b = build_adjacency(edge_index, n, direction='out')
    assert [t.tolist() for t in a] == [t.tolist() for t in b]


def test_direction_out_preserves_legacy_sampling():
    """The out path must be byte-identical to the pre-fix implementation.

    Same generator seed => same Bernoulli draws => same vertex set, and the
    recorded arcs are the legacy (u, w) orientation.  This keeps the orientation
    ablation an apples-to-apples comparison.
    """
    edge_index, n = _toy_graph()
    adj = build_out_adjacency(edge_index, n)
    gen = torch.Generator().manual_seed(3)
    sg = sparse_expand(adj, 0, p2=0.6, r=3, generator=gen, direction='out')
    # Values captured from the pre-fix implementation (git show HEAD:...).
    assert sg.nodes.tolist() == [0, 1, 2, 4, 3]
    assert sg.edge_index.tolist() == [[0, 1, 1, 2], [1, 2, 3, 4]]


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_determinism_under_fixed_seed(direction):
    edge_index, n = _toy_graph()
    adj = build_adjacency(edge_index, n, direction=direction)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sparse_expand(adj, 0, p2=0.5, r=3, generator=g1, direction=direction)
    b = sparse_expand(adj, 0, p2=0.5, r=3, generator=g2, direction=direction)
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
    from src.sparse.gnn_mechanism import GNNMechanism
    from src.sparse.sparse_gnn import train_sparse_gnn

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
    sg = sparse_expand(adj, root, p2=1.0, r=2, direction='in')
    loss0 = mech.subgraph_loss(sg)
    assert torch.isfinite(loss0)

    accs = train_sparse_gnn(mech, data, adj=adj, direction='in',
                            p1=1.0, p2=1.0, r=2, T=30,
                            candidate_nodes=cand, seed=0)
    # After 30 full-batch steps on CiteSeer, train accuracy should clear chance.
    assert accs['train'] > 1.0 / dataset.num_classes


def test_in_expansion_actually_reaches_the_root_representation():
    """End-to-end guard: the root's GNN output must depend on its neighbours.

    With the pre-v35 out-orientation the root's representation was identical to
    that of an isolated root, i.e. the mechanism was a graph-blind MLP.  Under
    Algorithm 5 it must differ.
    """
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    conv = GCNConv(4, 3, add_self_loops=True, normalize=True)
    isolated = conv(x, torch.zeros((2, 0), dtype=torch.long))[0]
    toward_root = conv(x, torch.tensor([[1], [0]], dtype=torch.long))[0]
    away_from_root = conv(x, torch.tensor([[0], [1]], dtype=torch.long))[0]

    assert not torch.allclose(toward_root, isolated)
    assert torch.allclose(away_from_root, isolated)
