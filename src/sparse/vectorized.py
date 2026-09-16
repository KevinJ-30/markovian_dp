"""Vectorized per-root DP gradients: padded subgraphs + torch.func.vmap.

The DP path needs ONE gradient per root, clipped to C before summing, so it
cannot use a single batched backward the way the non-DP path can.  The obvious
implementation -- and the one `sparse_gnn._step_dp` uses -- is a Python loop
calling `torch.autograd.grad` once per root.  That is 512 backward passes per
step at B=512, each over a graph averaging 1.4 nodes: all launch overhead, no
arithmetic.  Measured on PPI-large (K=5, p2=0.1, r=2, hidden=512) the loop is
387 ms of a 396 ms step, against 9 ms for the expansion that produced it.

This module removes the loop using STOCK LIBRARY PIECES ONLY.  There is no
hand-derived gradient math here, deliberately: this code sits on the privacy
path, where a silent error is most expensive and hardest to detect.

  * `torch.func.vmap(grad(...))` with `functional_call` is PyTorch's own
    documented recipe for per-sample gradients.  It is also what the reference
    implementation does -- google-research/differentially_private_gnns/train.py
    uses `jax.vmap(jax.grad(subgraph_loss))` over padded subgraphs.
  * `torch_geometric.nn.DenseSAGEConv` is PyG's dense counterpart of
    `SAGEConv(aggr='mean')`, built for exactly this [B, n, n] batched-graph
    layout.  Verified numerically identical to the sparse layer the mechanisms
    actually train with (test_dense_mirror_matches_sageconv).

    Only the bias PLACEMENT differs: SAGEConv carries it on lin_l, PyG's dense
    layer on lin_root.  The two terms are summed, so that is a rename rather
    than a change in the arithmetic; `dense_param_map` performs it.

WHY PADDING IS SAFE HERE

SparseExpand at depth r on a graph capped to K_out yields at most
1 + K + ... + K^r nodes, so a padded representation is finite.  We pad to the
largest subgraph IN THE BATCH, which is far smaller than that bound -- at
K=5, p2=0.1, r=2 the mean is 1.38 nodes, the batch max 6, the static bound 31.

Padded rows carry zero features and an all-zero adjacency row, and no real node
has an arc FROM a padded node, so they cannot reach the root.  Since the loss
reads only local index 0 (the root, by RootedSubgraph convention), padded nodes
contribute exactly zero gradient.  Their own activations are nonzero -- a padded
node still picks up the layer bias -- but nothing consumes them.
`test_padding_is_inert` pins this.

SCOPE.  aggr='mean' only.  The 'gcn' aggregator's symmetric normalization needs
the SOURCE degree, which a rooted subgraph does not know for boundary nodes.
Mechanisms decline the fast path in that case and fall back to the loop.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, vmap
from torch_geometric.nn import DenseSAGEConv


# Dense [B, n, n] adjacency is the memory ceiling.  At 32 M elements (128 MB in
# float32) a batch is split into chunks that each fit, so a large K x p2 cell
# degrades in throughput rather than failing to allocate.  Measured dense cost
# per 512-root batch: 0.1 MB at K=5/p2=0.1, 1.4 MB at K=5/p2=1.0, 37 MB at
# K=25/p2=0.5, 235 MB at K=25/p2=1.0 (the only cell that chunks).
_MAX_DENSE_ELEMS = 32_000_000


@dataclass
class PaddedBatch:
    """A batch of rooted subgraphs padded to a common node count.

    x:    [B, n, F]  node features, zero on padded rows
    adj:  [B, n, n]  UNNORMALIZED adjacency; adj[b, i, j] = number of arcs
                     j -> i.  DenseSAGEConv applies the mean normalization.
    y:    [B, ...]   the ROOT's label, one per subgraph
    sup:  [B]        1.0 where the root carries training supervision, else 0.0
    """

    x: torch.Tensor
    adj: torch.Tensor
    y: torch.Tensor
    sup: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def pad_to(self) -> int:
        return int(self.x.shape[1])


def build_padded_batch(subgraphs: Sequence, x_all: torch.Tensor,
                       y_all: torch.Tensor, *,
                       supervised: Optional[torch.Tensor] = None,
                       device=None) -> PaddedBatch:
    """Stack rooted subgraphs into one padded dense batch.

    `subgraphs` are RootedSubgraphs whose `nodes[0]` is the root and whose
    `edge_index` is already in LOCAL indices.  Built with tensor ops over the
    concatenated edge lists rather than a per-subgraph Python loop, so this
    does not reintroduce the cost the module exists to remove.
    """
    if not subgraphs:
        raise ValueError("cannot build a padded batch from zero subgraphs")

    device = device or x_all.device
    B = len(subgraphs)

    # Assemble every index on the CPU and cross to the device ONCE.  Doing the
    # gathers per subgraph (`[sg.nodes.to(device) for sg in subgraphs]`) issues
    # B separate host-to-device copies, which on MPS cost more than the whole
    # backward: measured 215 ms/step that way against 24 ms for the same work
    # on CPU.  Transfer count, not transfer volume, is what hurts.
    sizes = torch.tensor([sg.num_nodes for sg in subgraphs])
    n = int(sizes.max())
    roots_cpu = torch.tensor([int(sg.nodes[0]) for sg in subgraphs])

    # Node ids, padded with the root's id (any valid id works -- padded rows are
    # zeroed immediately after the gather, and this keeps the index in range).
    node_idx = roots_cpu[:, None].repeat(1, n)
    valid = torch.arange(n)[None, :] < sizes[:, None]             # [B, n]
    node_idx[valid] = torch.cat([sg.nodes.cpu() for sg in subgraphs])

    counts = torch.tensor([sg.num_edges for sg in subgraphs])
    total_edges = int(counts.sum())
    if total_edges > 0:
        flat_ei = torch.cat([sg.edge_index for sg in subgraphs
                             if sg.num_edges > 0], dim=1).cpu()
        batch_of_edge = torch.repeat_interleave(torch.arange(B), counts)
        # Flat offset into a [B, n, n] block-diagonal layout.
        adj_flat_idx = batch_of_edge * n * n + flat_ei[1] * n + flat_ei[0]
    else:
        adj_flat_idx = torch.empty(0, dtype=torch.long)

    node_idx = node_idx.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    adj_flat_idx = adj_flat_idx.to(device, non_blocking=True)
    roots = roots_cpu.to(device, non_blocking=True)

    x = x_all[node_idx] * valid[..., None]                        # [B, n, F]

    # ACCUMULATE, do not assign: SAGEConv's mean averages over the edge
    # MULTISET, so a repeated arc counts twice.  Left unnormalized here --
    # DenseSAGEConv divides by the row sum itself.
    adj = torch.zeros(B * n * n, device=device, dtype=x.dtype)
    if total_edges > 0:
        adj.index_add_(0, adj_flat_idx,
                       torch.ones(total_edges, device=device, dtype=x.dtype))
    adj = adj.view(B, n, n)

    y = y_all[roots]
    sup = (torch.ones(B, device=device, dtype=x.dtype) if supervised is None
           else supervised[roots].to(x.dtype))
    return PaddedBatch(x=x, adj=adj, y=y, sup=sup)


# ── dense mirrors of the mechanisms' own stacks ─────────────────────────────

class DenseStack(nn.Module):
    """PyG DenseSAGEConv layers mirroring a mechanism's SAGEConv stack.

    Structurally identical to the `_GNN.forward` bodies in the mechanism
    modules: conv, then ReLU and dropout on every layer but the last.  Holds no
    trained state of its own -- `per_sample_grads` always supplies the
    mechanism's live parameters through `functional_call`.
    """

    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList([DenseSAGEConv(dims[i], dims[i + 1])
                                    for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DenseMLPStack(nn.Module):
    """Linear stack mirroring MLPMechanism's `_MLP` (the graph-blind arm)."""

    def __init__(self, dims: Sequence[int], dropout: float = 0.0):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                   for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, adj=None):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def dense_param_map(num_layers: int) -> dict:
    """SAGEConv parameter name -> DenseSAGEConv parameter name.

    SAGEConv computes lin_l(mean_agg(x)) + lin_r(x) with the bias on lin_l;
    PyG's dense layer computes lin_rel(mean_agg(x)) + lin_root(x) with the bias
    on lin_root.  The two terms are summed, so relocating the bias between them
    is a rename and not a change in the arithmetic.
    """
    out = {}
    for i in range(num_layers):
        out[f'convs.{i}.lin_l.weight'] = f'convs.{i}.lin_rel.weight'
        out[f'convs.{i}.lin_l.bias'] = f'convs.{i}.lin_root.bias'
        out[f'convs.{i}.lin_r.weight'] = f'convs.{i}.lin_root.weight'
    return out


def build_mirror(module: nn.Module, kind: str, dropout: float) -> nn.Module:
    """A DenseStack/DenseMLPStack shaped like `module`, for `functional_call`."""
    if kind == 'mlp':
        dims = ([module.lins[0].in_features]
                + [l.out_features for l in module.lins])
        return DenseMLPStack(dims, dropout)
    dims = [module.convs[0].in_channels] + [c.out_channels for c in module.convs]
    return DenseStack(dims, dropout)


# ── per-sample gradients, the stock torch.func way ──────────────────────────

def per_sample_grads(params: dict, batch: PaddedBatch, mirror: nn.Module, *,
                     loss_tail: Callable, kind: str = 'sage',
                     training: bool = True,
                     max_dense_elems: int = _MAX_DENSE_ELEMS):
    """One gradient per root, plus the summed loss.

    Returns `(grads, total_loss)` where `grads` is a dict of
    [B, *param_shape] tensors.

    `loss_tail(root_out, y)` maps the root's raw output row and its label to a
    scalar -- the only part that differs between mechanisms (BCE-with-logits for
    multilabel, cross-entropy for single-label, MSE for regression).

    Returned keys are the MECHANISM's parameter names, not the mirror's, so
    callers never see the rename.  Chunked so a large subgraph x batch never
    allocates more than `max_dense_elems` of dense adjacency at once.

    `grad_and_value` rather than `grad` so the engine's progress line can report
    a real loss; recomputing it would cost a second forward pass.
    """
    name_map = {} if kind == 'mlp' else dense_param_map(len(mirror.convs))
    inv = {v: k for k, v in name_map.items()}
    dense_params = {name_map.get(k, k): v for k, v in params.items()}
    mirror.train(training)

    def loss_one(p, x_i, adj_i, y_i, sup_i):
        args = ((x_i.unsqueeze(0),) if kind == 'mlp'
                else (x_i.unsqueeze(0), adj_i.unsqueeze(0)))
        out = functional_call(mirror, p, args).squeeze(0)
        # Local index 0 is the root (RootedSubgraph convention).  Multiplying by
        # `sup` rather than branching keeps this traceable under vmap and gives
        # an unsupervised root exactly the zero gradient `zero_loss` gives it.
        return loss_tail(out[0], y_i) * sup_i

    grad_fn = vmap(grad_and_value(loss_one), in_dims=(None, 0, 0, 0, 0),
                   randomness='different' if training else 'error')

    B, n = len(batch), batch.pad_to
    per_chunk = max(1, min(B, max_dense_elems // max(n * n, 1)))
    out, total = None, 0.0
    for s in range(0, B, per_chunk):
        e = min(s + per_chunk, B)
        g, losses = grad_fn(dense_params, batch.x[s:e], batch.adj[s:e],
                            batch.y[s:e], batch.sup[s:e])
        total += float(losses.detach().sum())
        out = g if out is None else {k: torch.cat([out[k], g[k]]) for k in g}
    return {inv.get(k, k): v for k, v in out.items()}, total


def clipped_grad_sum(per_sample: dict, names: Sequence[str], C: float
                     ) -> List[torch.Tensor]:
    """Clip each root's gradient to L2 norm C, then sum over the batch.

    The vectorized form of `BaseMechanism.clip_flat_grad` followed by the
    accumulate loop: norms are taken over the FLATTENED concatenation of all
    parameters (one norm per root, not one per tensor), which is the
    ||g0(H)||_2 <= C the accounting assumes.
    """
    if not names:
        return []
    B = per_sample[names[0]].shape[0]
    flat = torch.cat([per_sample[k].reshape(B, -1) for k in names], dim=1)
    coef = (C / (flat.norm(dim=1) + 1e-12)).clamp(max=1.0)          # [B]
    return [(per_sample[k] * coef.view(B, *([1] * (per_sample[k].dim() - 1)))
             ).sum(0) for k in names]


# ── per-mechanism loss tails ────────────────────────────────────────────────
# Each mirrors the corresponding `subgraph_loss`, applied to the root's output
# row.  Unbatched: vmap supplies the batch axis.  log_softmax is row-wise, so
# cross_entropy on the root's logits equals the mechanisms' log_softmax over all
# nodes followed by nll_loss on the root.

def multilabel_tail(root_out, y):
    return F.binary_cross_entropy_with_logits(root_out, y.to(root_out.dtype))


def single_label_tail(root_out, y):
    return F.cross_entropy(root_out.unsqueeze(0), y.view(1))


def binary_tail(root_out, y):
    return F.binary_cross_entropy_with_logits(
        root_out.reshape(()), y.reshape(()).to(root_out.dtype))


def regression_tail(root_out, y):
    return F.mse_loss(root_out.reshape(()), y.reshape(()).to(root_out.dtype))
