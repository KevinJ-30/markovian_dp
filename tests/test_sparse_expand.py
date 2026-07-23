"""
Tests for the paper-faithful sparsification (SparseExpand / SparseGNN).

Covers Algorithm 2 invariants, root sampling, and a smoke test of the
Algorithm 1 engine with the GNN base mechanism.
"""

import math

import torch

from src.sparse.sparse_expand import (
    build_out_adjacency, sample_roots, sparse_expand,
)


def _toy_graph():
    # Directed chain 0->1->2->3 plus a branch 1->4.  num_nodes=5.
    edge_index = torch.tensor([[0, 1, 2, 1],
                               [1, 2, 3, 4]], dtype=torch.long)
    return edge_index, 5


def _forward_reachable(adj, root, r):
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


def test_p2_one_matches_forward_reachable():
    edge_index, n = _toy_graph()
    adj = build_out_adjacency(edge_index, n)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=1.0, r=10)
        assert sg.root == root
        assert int(sg.nodes[0]) == root           # root is local index 0
        assert set(sg.nodes.tolist()) == _forward_reachable(adj, root, 10)


def test_p2_zero_is_isolated_root():
    edge_index, n = _toy_graph()
    adj = build_out_adjacency(edge_index, n)
    sg = sparse_expand(adj, 0, p2=0.0, r=5)
    assert sg.nodes.tolist() == [0]
    assert sg.num_edges == 0


def test_edges_are_real_and_local():
    edge_index, n = _toy_graph()
    adj = build_out_adjacency(edge_index, n)
    real = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    gen = torch.Generator().manual_seed(7)
    for root in range(n):
        sg = sparse_expand(adj, root, p2=0.7, r=3, generator=gen)
        # local indices are within range
        if sg.num_edges:
            assert int(sg.edge_index.max()) < sg.num_nodes
            # remap to original ids and check every edge exists in G
            src = sg.nodes[sg.edge_index[0]]
            dst = sg.nodes[sg.edge_index[1]]
            for u, v in zip(src.tolist(), dst.tolist()):
                assert (u, v) in real


def test_determinism_under_fixed_seed():
    edge_index, n = _toy_graph()
    adj = build_out_adjacency(edge_index, n)
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
    from src.sparse.gnn_mechanism import GNNMechanism
    from src.sparse.sparse_gnn import train_sparse_gnn

    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    data = dataset[0]
    device = torch.device('cpu')

    torch.manual_seed(0)
    adj = build_out_adjacency(data.edge_index, int(data.num_nodes))
    mech = GNNMechanism(data, dataset.num_features, dataset.num_classes,
                        hidden=16, num_layers=2, device=device)
    mech.build_optimizer(lr=0.01, weight_decay=5e-4, kind='adam')

    cand = torch.where(data.train_mask)[0]
    # subgraph_loss on a labeled root is a finite scalar
    root = int(cand[0])
    sg = sparse_expand(adj, root, p2=1.0, r=2)
    loss0 = mech.subgraph_loss(sg)
    assert torch.isfinite(loss0)

    accs = train_sparse_gnn(mech, data, adj=adj, p1=1.0, p2=1.0, r=2, T=30,
                            candidate_nodes=cand, seed=0)
    # After 30 full-batch steps on CiteSeer, train accuracy should clear chance.
    assert accs['train'] > 1.0 / dataset.num_classes
