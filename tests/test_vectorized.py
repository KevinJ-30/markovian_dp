"""The vectorized DP path must equal the per-root loop exactly.

The loop in `sparse_gnn._step_dp` is the reference implementation of the
mechanism the accounting prices: one gradient per root, each clipped to
||g||_2 <= C, summed, then one Gaussian draw.  `_step_dp_vectorized` computes
the same quantity via torch.func.vmap over padded subgraphs.  If the two ever
disagree, the reported epsilon describes the loop while the released weights
came from the vectorized path -- so these tests pin the agreement rather than
merely checking that the fast path "looks reasonable".

Everything here runs on a small synthetic graph so the suite stays fast.
"""

import pytest
import torch

from src.sparse.sparse_expand import build_adjacency, sparse_expand
from src.sparse.vectorized import (
    build_mirror, build_padded_batch, clipped_grad_sum, per_sample_grads,
    multilabel_tail, single_label_tail, binary_tail, regression_tail,
)


def _vec(mech, subgraphs, data, C, kind='sage'):
    """Run the vectorized path for a mechanism, returning name -> summed grad."""
    cfg = mech.vectorized_config()
    assert cfg is not None, "mechanism declined the fast path"
    mirror = build_mirror(mech.module, cfg['kind'], cfg['dropout'])
    names = [n for n, _ in mech.module.named_parameters()]
    batch = build_padded_batch(subgraphs, data.x, data.y,
                               supervised=data.train_mask)
    per_root, _ = per_sample_grads(dict(mech.module.named_parameters()), batch,
                                   mirror, loss_tail=cfg['loss_tail'],
                                   kind=cfg['kind'])
    return dict(zip(names, clipped_grad_sum(per_root, names, C)))


class _Data:
    """Minimal stand-in for a PyG Data object."""

    def __init__(self, x, y, edge_index):
        self.x, self.y, self.edge_index = x, y, edge_index
        n = x.shape[0]
        self.num_nodes = n
        self.train_mask = torch.ones(n, dtype=torch.bool)
        self.val_mask = torch.zeros(n, dtype=torch.bool)
        self.test_mask = torch.zeros(n, dtype=torch.bool)


def _graph(n=40, f=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (120,), generator=g)
    dst = torch.randint(0, n, (120,), generator=g)
    keep = src != dst
    return torch.randn(n, f, generator=g), torch.stack([src[keep], dst[keep]])


def _subgraphs(edge_index, n, *, p2=0.6, r=2, count=24, seed=1):
    adj = build_adjacency(edge_index, n)
    g = torch.Generator().manual_seed(seed)
    return [sparse_expand(adj, v, p2, r, generator=g, direction='in')
            for v in range(count)]


def _loop_reference(mech, subgraphs, C):
    """The per-root loop from `sparse_gnn._step_dp`, verbatim in spirit."""
    params = mech.parameters()
    acc = [torch.zeros_like(p) for p in params]
    for sg in subgraphs:
        grads = torch.autograd.grad(mech.subgraph_loss(sg), params,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        for a, c in zip(acc, mech.clip_flat_grad(grads, C)):
            a.add_(c)
    return acc


# ── the dense mirror IS the PyG layer, not an approximation of it ──────────

def test_dense_mirror_matches_sageconv():
    """PyG's DenseSAGEConv, fed the mechanism's weights, IS SAGEConv(mean).

    This is the load-bearing assumption of the whole fast path: if the rename
    in `dense_param_map` were wrong (SAGEConv puts the bias on lin_l, the dense
    layer on lin_root) the gradients would be subtly wrong everywhere.
    """
    from torch.func import functional_call
    from src.sparse.multilabel_mechanism import _MultiLabelGNN
    from src.sparse.vectorized import dense_param_map
    torch.manual_seed(0)
    x, ei = _graph(n=12, f=5)
    mod = _MultiLabelGNN(5, 7, 3, dropout=0.0, num_layers=2, aggr='mean')
    ref = mod(x, ei)

    # ACCUMULATE, do not assign: SAGEConv's mean averages over the edge
    # MULTISET, so a repeated arc counts twice.  `build_padded_batch` uses
    # index_add_ for the same reason; assigning 1.0 here silently dedupes and
    # the two disagree by ~5e-2 on a graph with repeated arcs.
    adj = torch.zeros(12, 12)
    adj.index_put_((ei[1], ei[0]), torch.ones(ei.shape[1]), accumulate=True)

    mirror = build_mirror(mod, 'sage', 0.0)
    mirror.eval()
    name_map = dense_param_map(2)
    dense_params = {name_map[k]: v for k, v in mod.named_parameters()}
    got = functional_call(mirror, dense_params,
                          (x.unsqueeze(0), adj.unsqueeze(0))).squeeze(0)
    assert torch.allclose(ref, got, atol=1e-6)


# ── vectorized == loop, for every mechanism that declares the fast path ────

@pytest.mark.parametrize("C", [1.0, 0.01, 0.0005])
def test_vectorized_matches_loop_multilabel(C):
    from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
    torch.manual_seed(0)
    x, ei = _graph()
    data = _Data(x, (torch.rand(40, 4) > 0.5).float(), ei)
    mech = MultiLabelGNNMechanism(data, 6, 4, hidden=8, num_layers=2, dropout=0.0)
    sgs = _subgraphs(ei, 40)

    ref = dict(zip([n for n, _ in mech.module.named_parameters()],
                   _loop_reference(mech, sgs, C)))
    got = _vec(mech, sgs, data, C)
    for name in ref:
        assert torch.allclose(ref[name], got[name], atol=1e-6, rtol=1e-4), name


def test_vectorized_matches_loop_single_label():
    from src.sparse.gnn_mechanism import GNNMechanism
    torch.manual_seed(0)
    x, ei = _graph()
    data = _Data(x, torch.randint(0, 3, (40,)), ei)
    mech = GNNMechanism(data, 6, 3, hidden=8, num_layers=2, dropout=0.0)
    sgs = _subgraphs(ei, 40)
    ref = dict(zip([n for n, _ in mech.module.named_parameters()],
                   _loop_reference(mech, sgs, 0.05)))
    got = _vec(mech, sgs, data, 0.05)
    for name in ref:
        assert torch.allclose(ref[name], got[name], atol=1e-6, rtol=1e-4), name


def test_vectorized_matches_loop_binary_and_regression():
    from src.sparse.binary_mechanism import BinaryGNNMechanism
    from src.sparse.regression_mechanism import RegressionGNNMechanism
    x, ei = _graph()
    for cls, y, tail in [
            (BinaryGNNMechanism, (torch.rand(40) > 0.5).long(), binary_tail),
            (RegressionGNNMechanism, torch.randn(40), regression_tail)]:
        torch.manual_seed(0)
        data = _Data(x, y, ei)
        mech = cls(data, 6, 1, hidden=8, num_layers=2, dropout=0.0)
        sgs = _subgraphs(ei, 40)
        ref = dict(zip([n for n, _ in mech.module.named_parameters()],
                       _loop_reference(mech, sgs, 0.05)))
        got = _vec(mech, sgs, data, 0.05)
        for name in ref:
            assert torch.allclose(ref[name], got[name], atol=1e-6, rtol=1e-4), \
                f"{cls.__name__}:{name}"


def test_vectorized_matches_loop_blind_mlp():
    """The graph-blind arm uses a Linear stack, not SAGEConv (kind='mlp')."""
    from src.sparse.mlp_mechanism import MLPMechanism
    torch.manual_seed(0)
    x, ei = _graph()
    data = _Data(x, (torch.rand(40, 4) > 0.5).float(), ei)
    mech = MLPMechanism(data, 6, 4, hidden=8, num_layers=2, dropout=0.0)
    sgs = _subgraphs(ei, 40, p2=1.0, r=0)
    ref = dict(zip([n for n, _ in mech.module.named_parameters()],
                   _loop_reference(mech, sgs, 0.05)))
    got = _vec(mech, sgs, data, 0.05)
    for name in ref:
        assert torch.allclose(ref[name], got[name], atol=1e-6, rtol=1e-4), name


# ── padding must not change the answer ──────────────────────────────────────

def test_padding_is_inert():
    """Growing the pad width leaves every gradient unchanged.

    Padded rows pick up the layer bias, so they are not numerically zero; what
    makes them harmless is that no real node has an arc from them and the loss
    reads only the root.  This is the test that would catch a leak.
    """
    from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
    torch.manual_seed(0)
    x, ei = _graph()
    data = _Data(x, (torch.rand(40, 4) > 0.5).float(), ei)
    mech = MultiLabelGNNMechanism(data, 6, 4, hidden=8, num_layers=2, dropout=0.0)
    sgs = _subgraphs(ei, 40)
    pdict = dict(mech.module.named_parameters())
    names = [n for n, _ in mech.module.named_parameters()]
    mirror = build_mirror(mech.module, 'sage', 0.0)

    def run(batch):
        per_root, _ = per_sample_grads(pdict, batch, mirror,
                                       loss_tail=multilabel_tail, kind='sage')
        return clipped_grad_sum(per_root, names, 0.05)

    base = build_padded_batch(sgs, data.x, data.y, supervised=data.train_mask)
    ref = run(base)

    extra, n = 7, base.pad_to
    wide = type(base)(
        x=torch.cat([base.x, torch.zeros(len(base), extra, base.x.shape[-1])], 1),
        adj=torch.nn.functional.pad(base.adj, (0, extra, 0, extra)),
        y=base.y, sup=base.sup)
    assert wide.pad_to == n + extra
    for a, b in zip(ref, run(wide)):
        assert torch.allclose(a, b, atol=1e-6)


# ── capability probe ────────────────────────────────────────────────────────

def test_gcn_aggregator_declines_the_fast_path():
    """'gcn' needs the source degree, which the dense form does not model."""
    from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
    x, ei = _graph()
    data = _Data(x, (torch.rand(40, 4) > 0.5).float(), ei)
    assert MultiLabelGNNMechanism(data, 6, 4, hidden=8, num_layers=2,
                                  dropout=0.0, aggr='gcn').vectorized_config() is None
    assert MultiLabelGNNMechanism(data, 6, 4, hidden=8, num_layers=2,
                                  dropout=0.0).vectorized_config() is not None


# ── end to end: same trajectory through the real training engine ────────────

def test_training_trajectory_is_identical():
    from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism
    from src.sparse.sparse_gnn import train_sparse_gnn
    x, ei = _graph()
    data = _Data(x, (torch.rand(40, 4) > 0.5).float(), ei)
    adj = build_adjacency(ei, 40)

    def run(vectorized):
        torch.manual_seed(0)
        mech = MultiLabelGNNMechanism(data, 6, 4, hidden=8, num_layers=2,
                                      dropout=0.0)
        mech.build_optimizer(lr=0.05)
        train_sparse_gnn(mech, data, p1=0.5, p2=0.6, r=2, T=12, adj=adj,
                         candidate_nodes=torch.arange(40), dp=True, clip=0.01,
                         sigma=1.5, direction='in', seed=0,
                         vectorized=vectorized)
        return [p.detach().clone() for p in mech.parameters()]

    for a, b in zip(run(False), run(True)):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
