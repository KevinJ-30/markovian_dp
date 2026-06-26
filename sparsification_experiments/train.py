"""
Training loops for utility and DP modes.

Uses a DirectedGCN model (SAGEConv with mean aggregation over in-neighbors only)
to match the out-degree-sparsified directed graph.  No symmetric normalization,
no added reverse edges, no added self-loops inside the model.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sparsify import node_sensitivity    # noqa: E402


class DirectedGCN(nn.Module):
    """
    L-layer GCN with directed in-neighbor mean aggregation.

    Each layer computes:
        h_v^{l+1} = relu(W_self * h_v^l + W_neigh * mean_{u: u->v} h_u^l)

    implemented via PyG SAGEConv(aggr='mean').  No symmetric normalization,
    no added reverse edges, no added self-loops in the convolution.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1], aggr='mean', normalize=False)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class SymmetricGCN(nn.Module):
    """
    L-layer GCN with standard symmetric normalization (Kipf & Welling 2017).

    Each layer computes:
        h^{l+1} = ReLU( D^{-1/2} A_hat D^{-1/2} H^l W^l )
    where A_hat = A + I (self-loops added inside GCNConv).

    Self-loops do NOT expand cross-node receptive field: they only let a node
    re-weight its own previous representation.  So |R_out(u)| and Delta are
    unchanged relative to DirectedGCN at the same D.

    VALID ONLY with a symmetrically-capped graph (sparsify_symmetric, D both
    ways).  On an out-degree-only-capped graph the symmetric normalization
    re-introduces incoming edges from high-degree nodes, leaving fan-out
    unbounded and invalidating the sensitivity bound.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


def make_model(num_features, num_classes, hidden=64, dropout=0.5,
               num_layers=2, model_type='symmetric'):
    if model_type == 'symmetric':
        model = SymmetricGCN(num_features, hidden, num_classes,
                             dropout=dropout, num_layers=num_layers)
        print(f"[model] SymmetricGCN  layers={num_layers}  hidden={hidden}  "
              f"aggregation=symmetric-gcn  normalization=sym  self_loops=yes")
    else:
        model = DirectedGCN(num_features, hidden, num_classes,
                            dropout=dropout, num_layers=num_layers)
        print(f"[model] DirectedGCN  layers={num_layers}  hidden={hidden}  "
              f"aggregation=in-neighbor-mean  normalization=none  "
              f"reverse_edges=none  self_loops=none")
    return model


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, data, edge_index):
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index)
        pred = out.argmax(dim=1)
        accs = {}
        for split in ('train', 'val', 'test'):
            mask = getattr(data, f'{split}_mask')
            n = int(mask.sum().item())
            accs[split] = float((pred[mask] == data.y[mask]).sum().item()) / n if n else float('nan')
    return accs


# ── utility training ──────────────────────────────────────────────────────────

def train_utility(model, data, edge_index, lr=0.01, weight_decay=5e-4,
                  epochs=200, verbose=False):
    """Standard GCN training with no DP on the (possibly sparsified) graph."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            accs = evaluate(model, data, edge_index)
            print(f"  epoch {epoch:3d}  loss={float(loss):.4f}  "
                  f"val={accs['val']:.4f}  test={accs['test']:.4f}")

    return evaluate(model, data, edge_index)


# ── DP-SGD training ───────────────────────────────────────────────────────────

def train_dp(model, data, sparse_edge_index, *,
             steps, C, sigma, D, L, q, adjacency='add_remove',
             no_subsampling=False,
             lr=0.01, weight_decay=5e-4, verbose=False):
    """
    DP-SGD on the out-degree-sparsified directed graph.

    Per step:
      1. Sample batch:
         - no_subsampling=False (q<1): Poisson-sample each training node
           independently with probability q.  PLACEHOLDER ACCOUNTING — see run.py.
         - no_subsampling=True  (q=1): use ALL training nodes every step.
           This is the only configuration with a valid node-DP guarantee today.
      2. ONE forward pass on the full sparsified graph (shared computation).
      3. Microbatch: per-seed backward, clip each per-node gradient to L2 norm C,
         accumulate.
      4. Add Gaussian noise N(0, (sigma * Delta)^2 * I) where
         Delta = node_sensitivity(C, D, L, adjacency).
         sigma is the SENSITIVITY-NORMALISED noise multiplier that accountants
         consume.  noise_std = sigma * Delta  (not sigma * C alone).
      5. Divide by expected (or actual) batch size; step.

    NOTE on scale: full-graph forward + per-seed microbatching is tractable for
    Planetoid.  For ogbn-arxiv, replace with per-seed NeighborLoader subgraphs.

    Returns:
        accs dict (train/val/test accuracy)
        actual_steps int (optimizer steps actually taken)
    """
    Delta = node_sensitivity(C, D, L, adjacency)
    noise_std = sigma * Delta

    train_nodes = torch.where(data.train_mask)[0]
    n_train = int(train_nodes.size(0))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    actual_steps = 0
    for step in range(steps):
        if no_subsampling:
            batch = train_nodes          # full batch every step
            expected_bs = float(n_train)
        else:
            included = torch.bernoulli(torch.full((n_train,), q)).bool()
            batch = train_nodes[included]
            if batch.numel() == 0:
                continue
            expected_bs = max(q * n_train, 1.0)

        model.train()
        out = model(data.x, sparse_edge_index)

        grad_accum = [torch.zeros_like(p) for p in params]
        batch_list = batch.tolist()

        for i, v in enumerate(batch_list):
            loss_v = F.nll_loss(out[v:v + 1], data.y[v:v + 1])
            retain = (i < len(batch_list) - 1)
            grads = torch.autograd.grad(loss_v, params, retain_graph=retain)
            norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
            coef = float(min(1.0, C / (float(norm) + 1e-8)))
            for acc, g in zip(grad_accum, grads):
                acc.add_(g, alpha=coef)

        optimizer.zero_grad()
        for p, acc in zip(params, grad_accum):
            noise = torch.randn_like(acc) * noise_std
            p.grad = (acc + noise) / expected_bs
        optimizer.step()
        actual_steps += 1

        if verbose and (step + 1) % max(steps // 5, 1) == 0:
            accs = evaluate(model, data, sparse_edge_index)
            print(f"  step {step+1:4d}/{steps}  batch={batch.numel()}  "
                  f"val={accs['val']:.4f}  test={accs['test']:.4f}  "
                  f"noise_std={noise_std:.3f}")

    return evaluate(model, data, sparse_edge_index), actual_steps
