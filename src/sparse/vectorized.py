"""Vectorized per-root gradients: padded dense subgraphs + vmap(grad(.)).

The DP path needs ONE gradient per root, clipped to C before summing, so it
cannot use a single batched backward the way the non-DP path can.  The obvious
implementation -- and the one `sparse_gnn._step_dp` uses -- is a Python loop
calling `torch.autograd.grad` once per root.  That is 512 backward passes per
step at B=512, each over a graph averaging 1.4 nodes: all launch overhead, no
arithmetic, and a GPU cannot help.  Measured on PPI-large (K=5, p2=0.1, r=2,
hidden=512) the loop is 387 ms of a 396 ms step -- 98%, against 9 ms for the
expansion that produced the subgraphs.

This module removes the loop the way Google's own DP-GNN implementation does
(google-research/differentially_private_gnns/train.py):

    per_example_gradient_fn = jax.vmap(jax.grad(subgraph_loss),
                                       in_axes=(None, None, 0, 0))

i.e. pad every rooted subgraph to a common size, stack them into one leading
batch axis, and let vmap turn the per-sample gradient into batched arithmetic.
torch.func.vmap/grad is the direct PyTorch equivalent.

TWO THINGS MAKE THIS WORK HERE

1. Subgraphs are bounded.  SparseExpand at depth r on a graph capped to K_out
   yields at most 1 + K + ... + K^r nodes, so a padded representation is
   finite.  We pad to the largest subgraph IN THE BATCH rather than to that
   bound, which is much smaller in practice -- at K=5, p2=0.1, r=2 the mean is
   1.38 nodes, the batch max 6, and the static bound 31.

2. Mean aggregation is a matmul.  SAGEConv(aggr='mean') is exactly

       out = A_norm @ X @ W_l^T + b_l + X @ W_r^T

   with A_norm[i, j] = 1/indeg(i) for each arc j -> i.  Verified bit-identical
   (max|diff| = 0.0) against the PyG layer.  So the whole forward is dense
   batched matmuls, which is what a GPU is for.

PADDING IS INERT, NOT APPROXIMATE.  Padded rows carry zero features and an
all-zero adjacency row, and no real node has an arc FROM a padded node, so
A_norm[:, pad] = 0 everywhere.  Since the loss reads only local index 0 (the
root, by RootedSubgraph convention), padded nodes cannot reach the output and
contribute exactly zero gradient.  Their own activations are nonzero -- a
padded node still picks up the layer bias -- but nothing consumes them.

SCOPE.  aggr='mean' only.  The 'gcn' aggregator's symmetric normalization needs
the SOURCE degree, which a rooted subgraph does not know for boundary nodes;
that is a modelling difference the loop path also has, and it is not worth
reproducing here.  Callers must fall back for 'gcn'.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.func import grad, vmap


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
    adj:  [B, n, n]  row-normalized mean-aggregation matrix (A_norm above)
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

    adj = torch.zeros(B * n * n, device=device, dtype=x.dtype)
    if total_edges > 0:
        adj.index_add_(0, adj_flat_idx,
                       torch.ones(total_edges, device=device, dtype=x.dtype))
    adj = adj.view(B, n, n)
    # Mean aggregation: divide each target row by its in-degree.  A row with no
    # in-edges stays all-zero, which is what SAGEConv gives an isolated node.
    adj = adj / adj.sum(-1, keepdim=True).clamp(min=1.0)

    y = y_all[roots]
    sup = (torch.ones(B, device=device, dtype=x.dtype) if supervised is None
           else supervised[roots].to(x.dtype))
    return PaddedBatch(x=x, adj=adj, y=y, sup=sup)


def dense_forward(params: dict, x: torch.Tensor, adj: torch.Tensor, *,
                  num_layers: int, dropout: float = 0.0,
                  training: bool = True) -> torch.Tensor:
    """The SAGEConv(aggr='mean') stack as dense matmuls, on one subgraph.

    Shapes here are UNBATCHED ([n, F], [n, n]) -- vmap supplies the batch axis.
    Mirrors the `_GNN.forward` bodies in the mechanism modules exactly: conv,
    then ReLU and dropout on every layer but the last.
    """
    h = x
    for i in range(num_layers):
        w_l = params[f'convs.{i}.lin_l.weight']
        b_l = params[f'convs.{i}.lin_l.bias']
        w_r = params[f'convs.{i}.lin_r.weight']
        h = (adj @ h) @ w_l.T + b_l + h @ w_r.T
        if i < num_layers - 1:
            h = F.relu(h)
            if dropout > 0.0:
                h = F.dropout(h, p=dropout, training=training)
    return h


def per_sample_grads(params: dict, batch: PaddedBatch, *,
                     loss_tail: Callable, num_layers: int,
                     dropout: float = 0.0, training: bool = True,
                     max_dense_elems: int = _MAX_DENSE_ELEMS) -> dict:
    """One gradient per root, as a dict of [B, *param_shape] tensors.

    `loss_tail(root_out, y)` maps the root's raw output row and its label to a
    scalar -- this is the only part that differs between the mechanisms
    (BCE-with-logits for multilabel, NLL on log-softmax for single-label, MSE
    for regression).  Unbatched: vmap supplies the batch axis.  The ghost path
    below wants the BATCHED form instead; `batched_tail` adapts between them.

    Chunked so a large subgraph x batch never allocates more than
    `max_dense_elems` of dense adjacency at once.
    """
    def loss_one(p, x_i, adj_i, y_i, sup_i):
        out = dense_forward(p, x_i, adj_i, num_layers=num_layers,
                            dropout=dropout, training=training)
        # Local index 0 is the root (RootedSubgraph convention).  Multiplying by
        # `sup` rather than branching keeps this traceable under vmap and gives
        # an unsupervised root exactly the zero gradient `zero_loss` gives it.
        return loss_tail(out[0], y_i) * sup_i

    # randomness='different' so each root draws its own dropout mask, matching
    # the loop path where every subgraph gets an independent forward.
    grad_fn = vmap(grad(loss_one), in_dims=(None, 0, 0, 0, 0),
                   randomness='different' if dropout > 0.0 else 'error')

    B, n = len(batch), batch.pad_to
    per_chunk = max(1, min(B, max_dense_elems // max(n * n, 1)))
    if per_chunk >= B:
        return grad_fn(params, batch.x, batch.adj, batch.y, batch.sup)

    out: Optional[dict] = None
    for s in range(0, B, per_chunk):
        e = min(s + per_chunk, B)
        g = grad_fn(params, batch.x[s:e], batch.adj[s:e],
                    batch.y[s:e], batch.sup[s:e])
        out = g if out is None else {k: torch.cat([out[k], g[k]]) for k in g}
    assert out is not None
    return out


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


@torch.no_grad()
def clipped_grad_sum_ghost(params: dict, batch: PaddedBatch, *,
                           loss_tail: Callable, num_layers: int, C: float,
                           kind: str = 'sage',
                           dropout: float = 0.0, training: bool = True,
                           generator: Optional[torch.Generator] = None
                           ) -> tuple:
    """Clipped per-root gradient sum WITHOUT materializing per-root gradients.

    `per_sample_grads` is correct but allocates one full copy of every parameter
    per root: at B=512, hidden=512 and 121 labels the second layer's weight
    alone is a [512, 121, 512] tensor, 127 MB, and the step becomes
    memory-bandwidth bound.  That is why the vmap path only buys ~2x at
    hidden=512 while buying 6.5x at hidden=64.

    This is the standard ghost-clipping route around that.  For a linear map
    with per-root input U_i and output-gradient S_i, the gradient is
    G_i = S_i^T U_i, and its squared Frobenius norm is

        ||S_i^T U_i||_F^2 = <S_i S_i^T, U_i U_i^T>

    -- two n x n Gram matrices, where n is the padded subgraph size.  Here n is
    the batch's largest rooted subgraph (1.4 nodes on average at K=5, p2=0.1),
    so the norms cost essentially nothing and never touch d_in x d_out.  With
    the per-root clip coefficients c_i in hand, the clipped sum is

        sum_i c_i G_i = sum_i (c_i S_i)^T U_i

    i.e. rescale the output gradients and contract -- ONE matmul producing a
    single [d_out, d_in], never a batch of them.

    The backward is written out by hand because the forward is a handful of
    explicit matmuls; the only piece taken from autograd is the loss tail's
    dL/d(root output), which is cheap and keeps each mechanism's loss authoritative
    rather than re-derived here.

    Returns (grads_in_param_order, summed_loss, mean_clip_coefficient).
    """
    names = ghost_param_names(num_layers, kind)
    x, adj, y, sup = batch.x, batch.adj, batch.y, batch.sup
    B, n, _ = x.shape

    # Everything below runs under no_grad: the backward is written out by hand,
    # so an autograd graph over the forward would be built and never used --
    # pure memory and time.  The ONE exception is the loss tail, which is
    # re-enabled explicitly so each mechanism's own loss stays authoritative.
    # ---- forward, keeping what the backward needs -------------------------
    h = x
    hs, us, pres, masks = [], [], [], []
    for i in range(num_layers):
        if kind == 'mlp':
            # Graph-blind: no aggregation term at all.
            w_r, b_l = params[f'lins.{i}.weight'], params[f'lins.{i}.bias']
            u = None
            pre = h @ w_r.T + b_l
        else:
            w_l = params[f'convs.{i}.lin_l.weight']
            b_l = params[f'convs.{i}.lin_l.bias']
            w_r = params[f'convs.{i}.lin_r.weight']
            u = adj @ h
            pre = u @ w_l.T + b_l + h @ w_r.T
        hs.append(h); us.append(u); pres.append(pre)
        if i < num_layers - 1:
            a = F.relu(pre)
            if dropout > 0.0 and training:
                keep = torch.rand(a.shape, device=a.device, generator=generator,
                                  dtype=a.dtype) >= dropout
                m = keep.to(a.dtype) / (1.0 - dropout)
                a = a * m
                masks.append(m)
            else:
                masks.append(None)
            h = a
        else:
            masks.append(None)

    # ---- dL/d(root output), per root, from the mechanism's own tail --------
    # `loss_tail` here is the BATCHED form: [B, d_out] x [B, ...] -> [B].  Each
    # root's loss depends only on its own row, so one grad of the sum gives every
    # root's dL/dz at once.  (A Python loop over B here costs more than the whole
    # rest of the step -- measured 87 ms against 29 ms for the vmap path.)
    root_out = pres[-1][:, 0, :].detach().requires_grad_(True)
    with torch.enable_grad():
        per_loss = loss_tail(root_out, y) * sup
        g_root, = torch.autograd.grad(per_loss.sum(), root_out)

    # Only the root row carries loss gradient; every other row is zero.
    g_pre = torch.zeros_like(pres[-1])
    g_pre[:, 0, :] = g_root

    # ---- backward, collecting (S_i, U_i) pairs per parameter --------------
    pairs = {}                       # name -> (S, U) with G_i = S_i^T U_i
    for i in range(num_layers - 1, -1, -1):
        if kind == 'mlp':
            w_r = params[f'lins.{i}.weight']
            pairs[f'lins.{i}.weight'] = (g_pre, hs[i])
            pairs[f'lins.{i}.bias'] = (g_pre, None)        # sum over nodes
            d_h = g_pre @ w_r if i > 0 else None
        else:
            w_l = params[f'convs.{i}.lin_l.weight']
            w_r = params[f'convs.{i}.lin_r.weight']
            pairs[f'convs.{i}.lin_l.weight'] = (g_pre, us[i])
            pairs[f'convs.{i}.lin_r.weight'] = (g_pre, hs[i])
            pairs[f'convs.{i}.lin_l.bias'] = (g_pre, None)  # sum over nodes
            d_h = (adj.transpose(1, 2) @ (g_pre @ w_l) + g_pre @ w_r
                   if i > 0 else None)
        if i > 0:
            if masks[i - 1] is not None:
                d_h = d_h * masks[i - 1]
            g_pre = d_h * (pres[i - 1] > 0).to(d_h.dtype)

    # ---- per-root squared norms via the Gram identity ---------------------
    sq = torch.zeros(B, device=x.device, dtype=x.dtype)
    for name in names:
        S, U = pairs[name]
        if U is None:                                  # bias: G_i = S_i.sum(0)
            sq = sq + S.sum(1).pow(2).sum(-1)
        else:
            sq = sq + ((S @ S.transpose(1, 2)) * (U @ U.transpose(1, 2))
                       ).sum((1, 2))
    coef = (C / (sq.clamp(min=0).sqrt() + 1e-12)).clamp(max=1.0)       # [B]

    # ---- clipped sum: rescale S, then contract once -----------------------
    out = []
    for name in names:
        S, U = pairs[name]
        Sc = S * coef.view(B, 1, 1)
        if U is None:
            out.append(Sc.sum((0, 1)))
        else:
            d_out, d_in = S.shape[-1], U.shape[-1]
            out.append(Sc.reshape(-1, d_out).T @ U.reshape(-1, d_in))
    return out, float(per_loss.sum().detach()), float(coef.mean())


def ghost_param_names(num_layers: int, kind: str = 'sage') -> List[str]:
    """Parameter order `clipped_grad_sum_ghost` returns, for zipping.

    'sage' is the SAGEConv(aggr='mean') stack the GNN mechanisms build; 'mlp' is
    the plain Linear stack the graph-blind arm builds.  Supporting both matters
    because the blind arm runs at every epsilon: left on the per-root loop it
    costs 221 ms/step against the vectorized GNN's 46 ms, which would make the
    simplest model in the suite the most expensive thing in the grid.
    """
    if kind == 'mlp':
        return [f'lins.{i}.{p}' for i in range(num_layers)
                for p in ('weight', 'bias')]
    return [f'convs.{i}.lin_{sfx}.{p}'
            for i in range(num_layers)
            for sfx, p in (('l', 'weight'), ('l', 'bias'), ('r', 'weight'))]


# ── per-mechanism loss tails ────────────────────────────────────────────────
# Each mirrors the corresponding `subgraph_loss`, applied to the root's output
# row.  log_softmax is row-wise, so applying it to the root alone is identical
# to applying it to all nodes and then slicing, as `_GNN.forward` does.
#
# These are the BATCHED forms: [B, d_out] x [B, ...] -> [B], one loss per root.
# The ghost path uses them directly; `unbatched_tail` adapts one for vmap.

def multilabel_tail(root_out, y):
    """BCE-with-logits averaged over label columns, per root (PPI, Yelp, Amazon)."""
    return F.binary_cross_entropy_with_logits(
        root_out, y.to(root_out.dtype), reduction='none').mean(-1)


def single_label_tail(root_out, y):
    """NLL on log-softmax, per root -- identical to cross-entropy on the logits."""
    return F.cross_entropy(root_out, y.view(-1), reduction='none')


def binary_tail(root_out, y):
    return F.binary_cross_entropy_with_logits(
        root_out.reshape(-1), y.reshape(-1).to(root_out.dtype), reduction='none')


def regression_tail(root_out, y):
    return F.mse_loss(root_out.reshape(-1), y.reshape(-1).to(root_out.dtype),
                      reduction='none')


def unbatched_tail(tail: Callable) -> Callable:
    """Adapt a batched tail for `per_sample_grads`, where vmap removes the axis."""
    def one(root_out, y):
        return tail(root_out.unsqueeze(0), y.unsqueeze(0) if y.dim() else y.view(1))[0]
    return one
