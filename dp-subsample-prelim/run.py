"""
Preliminary subsampling experiment: per-batch random node dropout in a
neighbor-sampled pipeline for transductive node classification on Cora.

Four modes:
  - baseline:  vanilla 2-layer GCN trained on k-hop neighbor-sampled minibatches.
  - subsample: identical, but on each sampled batch we randomly drop a
               fraction of the non-seed nodes (and all edges incident to
               them) before the forward pass. Seed nodes are never dropped,
               so loss/accuracy on the seed set are always well-defined.
  - dp:        standard graph DP-SGD in the style of DP-GNN (Daigavane et
               al., 2021): per-seed-node gradient clipping to --clip, sum,
               plus Gaussian noise with std --sigma * --clip, averaged over
               the batch. Opacus's GradSampleModule assumes batch rows are
               independent examples, which message passing violates, so the
               per-seed gradients are computed by explicit microbatching;
               opacus is still used for privacy accounting (see
               accounting.py). After training, epsilon at --delta is
               reported via the chosen --accountant. If you have a custom
               per-step dominating pair (e.g. from a Markovian subsampling
               analysis), pass --accountant dominating-pair
               --dominating_pair pair.json and it is composed over all
               training steps instead of the subsampled-Gaussian analysis.
  - dp_subsample: dp combined with the per-batch non-seed node dropout of
               subsample mode. Note the opacus accountants do NOT credit
               this extra subsampling (they only see batch-level Poisson
               sampling, which is conservative); a dominating pair that
               captures it is exactly what the dominating-pair accountant
               is for.

The dropped set is re-randomized for every batch in every epoch (stochastic
per-batch, not a fixed deletion). Final test accuracy is always evaluated on
the held-out Planetoid test mask using the full graph, not a subsampled one.

We roll a small in-file neighbor sampler instead of using PyG's NeighborLoader
because that loader requires pyg-lib / torch-sparse compiled extensions; the
sampler here keeps the same contract (seeds occupy the first `batch_size`
rows of each batch) and only needs torch + torch_geometric.nn.GCNConv.

Example commands (CPU only):
  python run.py --mode baseline  --epochs 100
  python run.py --mode subsample --epochs 100 --drop_rate 0.10
  python run.py --mode dp        --epochs 100 --clip 1.0 --sigma 1.0
  python run.py --mode dp --accountant dominating-pair \
      --dominating_pair example_dominating_pair.json
"""

import argparse
import os
import sys
import random


def _check_imports():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[error] PyTorch is not installed.")
        print("        Install with: pip install torch")
        sys.exit(1)
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.nn import GCNConv  # noqa: F401
        from torch_geometric.datasets import Planetoid  # noqa: F401
    except ImportError:
        print("[error] PyTorch Geometric is not installed.")
        print("        Install with: pip install torch_geometric")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "subsample", "dp", "dp_subsample"],
                   default="baseline")
    p.add_argument("--dataset", choices=["cora", "citeseer", "pubmed"], default="cora")
    p.add_argument("--drop_rate", type=float, default=0.10,
                   help="fraction of non-seed nodes to drop per batch (subsample mode)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 10])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    # DP-SGD options (dp / dp_subsample modes)
    p.add_argument("--clip", type=float, default=1.0,
                   help="per-seed gradient l2 clipping norm C")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="noise multiplier; Gaussian noise std is sigma * clip")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--accountant",
                   choices=["opacus-rdp", "opacus-prv", "dominating-pair", "none"],
                   default="opacus-rdp")
    p.add_argument("--dominating_pair", type=str, default=None,
                   help="JSON file with the per-step dominating pair "
                        "(see accounting.py for the format)")
    p.add_argument("--pld_grid", type=float, default=1e-4,
                   help="loss discretization for dominating-pair accounting")
    p.add_argument("--occurrence_bound", type=float, default=1.0,
                   help="bound K on how many per-seed gradient terms one node "
                        "can contribute to in a single step; the accountant is "
                        "given an effective noise multiplier sigma / K. K=1 "
                        "recovers example-level (per-seed-label) DP-SGD; for "
                        "node-level DP set K from your sampler's fan-out bounds "
                        "as in DP-GNN.")
    return p.parse_args()


def set_seed(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)


_PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def load_planetoid(name):
    from torch_geometric.datasets import Planetoid
    canonical = _PLANETOID_NAMES[name.lower()]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dataset = Planetoid(root=root, name=canonical)
    return dataset, dataset[0]


class SimpleBatch:
    """Lightweight stand-in for a PyG NeighborLoader output. Holds the
    sampled subgraph with seeds in rows 0..batch_size-1, plus full-graph
    indices in `n_id` for any downstream lookup."""

    def __init__(self, x, edge_index, y, batch_size, n_id):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.batch_size = batch_size
        self.n_id = n_id

    @property
    def num_nodes(self):
        return self.x.size(0)

    def to(self, device):
        return self  # all tensors already on CPU; no-op


class NeighborSampler:
    """Minimal k-hop neighbor sampler that mirrors NeighborLoader's contract.

    Given a single Data object, build a per-node neighbor list once, then on
    every iteration:
      1. Shuffle the seed pool (nodes where input_mask is True).
      2. For each batch of seed nodes, sample up to num_neighbors[k] of each
         frontier node's neighbors at hop k+1 (without replacement when the
         degree is small enough, else with replacement).
      3. Concatenate sampled node IDs in order [seeds, hop1, hop2, ...],
         deduplicate keeping first occurrence so seeds stay in rows 0..bs-1.
      4. Build the induced edge_index using a vectorized membership remap.
    """

    def __init__(self, data, input_nodes, num_neighbors, batch_size, shuffle=True):
        import torch
        self.data = data
        self.num_neighbors = list(num_neighbors)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if input_nodes.dtype == torch.bool:
            self.input_nodes = torch.nonzero(input_nodes, as_tuple=False).view(-1)
        else:
            self.input_nodes = input_nodes.clone()

        src = data.edge_index[0]
        dst = data.edge_index[1]
        num_nodes = data.num_nodes

        # Build CSR-style neighbor lookup keyed on dst (incoming edges), since
        # GCN aggregates each node from its neighbors via incoming messages.
        # Cora's edge_index is symmetric so direction doesn't matter in
        # practice, but we keep dst as the key for clarity.
        order = torch.argsort(dst)
        sorted_dst = dst[order]
        sorted_src = src[order]
        counts = torch.bincount(sorted_dst, minlength=num_nodes)
        ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        self._ptr = ptr
        self._nbrs = sorted_src

    def _neighbors(self, node):
        s = int(self._ptr[node].item())
        e = int(self._ptr[node + 1].item())
        return self._nbrs[s:e]

    def __iter__(self):
        import torch
        n = self.input_nodes.numel()
        if self.shuffle:
            perm = torch.randperm(n)
        else:
            perm = torch.arange(n)
        seeds_all = self.input_nodes[perm]

        for start in range(0, n, self.batch_size):
            seeds = seeds_all[start:start + self.batch_size]
            yield self._build_batch(seeds)

    def _build_batch(self, seeds):
        import torch

        # Sample frontiers hop by hop. node_ids collects every encountered
        # node id in visit order; we'll dedupe at the end so seeds stay first.
        node_ids = [seeds]
        frontier = seeds
        for k_hop in range(len(self.num_neighbors)):
            fanout = self.num_neighbors[k_hop]
            sampled_neighbors = []
            for node in frontier.tolist():
                nbrs = self._neighbors(node)
                if nbrs.numel() == 0:
                    continue
                if nbrs.numel() <= fanout:
                    sampled_neighbors.append(nbrs)
                else:
                    idx = torch.randperm(nbrs.numel())[:fanout]
                    sampled_neighbors.append(nbrs[idx])
            if sampled_neighbors:
                frontier = torch.cat(sampled_neighbors)
            else:
                frontier = torch.empty(0, dtype=torch.long)
            node_ids.append(frontier)

        # Deduplicate preserving first-occurrence order so seeds occupy rows
        # 0..bs-1. torch.unique with return_inverse gives us sorted unique
        # ids, not first-occurrence — so we do it ourselves.
        all_ids = torch.cat(node_ids)
        seen = {}
        keep_order = []
        for v in all_ids.tolist():
            if v not in seen:
                seen[v] = len(keep_order)
                keep_order.append(v)
        n_id = torch.tensor(keep_order, dtype=torch.long)

        # Build induced edge_index: take all edges of the full graph whose
        # endpoints are both in n_id, then relabel to local 0..K-1 indices.
        num_nodes_global = self.data.num_nodes
        in_batch = torch.zeros(num_nodes_global, dtype=torch.bool)
        in_batch[n_id] = True
        src, dst = self.data.edge_index[0], self.data.edge_index[1]
        edge_mask = in_batch[src] & in_batch[dst]
        sub_src = src[edge_mask]
        sub_dst = dst[edge_mask]

        global_to_local = torch.full((num_nodes_global,), -1, dtype=torch.long)
        global_to_local[n_id] = torch.arange(n_id.numel())
        local_edge_index = torch.stack([global_to_local[sub_src],
                                        global_to_local[sub_dst]], dim=0)

        x = self.data.x[n_id]
        y = self.data.y[n_id]
        return SimpleBatch(x=x, edge_index=local_edge_index, y=y,
                           batch_size=int(seeds.numel()), n_id=n_id)


def build_model(in_channels, hidden, out_channels, dropout):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)
            self.dropout = dropout

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x  # logits; cross-entropy applied externally

    return GCN()


def drop_non_seed_nodes(batch, drop_rate):
    """Randomly drop a fraction of non-seed nodes in a NeighborLoader batch.

    NeighborLoader places the seed nodes as the first `batch.batch_size`
    rows of the sampled subgraph (this is a stable contract across PyG
    versions). We mask those out, randomly drop a fraction of the rest,
    filter `edge_index` to remove any edge touching a dropped node, and
    relabel node indices so the kept nodes are contiguous 0..K-1.

    Returns (x, edge_index, y_seed, seed_logit_idx) where seed_logit_idx
    indexes into the kept-node row order to locate the seed nodes' logits.
    """
    import torch

    num_nodes = batch.num_nodes
    bs = int(batch.batch_size)

    # keep_mask[i] = True if node i is kept (seeds always kept)
    keep_mask = torch.ones(num_nodes, dtype=torch.bool)
    if drop_rate > 0.0 and num_nodes > bs:
        non_seed_idx = torch.arange(bs, num_nodes)
        n_non_seed = non_seed_idx.numel()
        n_drop = int(round(drop_rate * n_non_seed))
        if n_drop > 0:
            perm = torch.randperm(n_non_seed)
            drop_local = non_seed_idx[perm[:n_drop]]
            keep_mask[drop_local] = False

    # Filter edges: keep only edges where both endpoints are kept.
    src, dst = batch.edge_index[0], batch.edge_index[1]
    edge_keep = keep_mask[src] & keep_mask[dst]
    new_edge_index = batch.edge_index[:, edge_keep]

    # Relabel node indices to be contiguous over kept nodes.
    new_idx = torch.full((num_nodes,), -1, dtype=torch.long)
    kept_positions = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    new_idx[kept_positions] = torch.arange(kept_positions.numel())
    new_edge_index = new_idx[new_edge_index]

    x = batch.x[kept_positions]
    # Seeds are the first bs nodes in original order; since seeds are never
    # dropped, after relabeling their new indices are 0..bs-1 (kept_positions
    # is sorted, so the first bs entries are exactly the seed positions).
    seed_logit_idx = torch.arange(bs)
    y_seed = batch.y[:bs]

    return x, new_edge_index, y_seed, seed_logit_idx


def batch_forward_loss(model, batch, mode, drop_rate, criterion):
    """Run a forward pass on one NeighborLoader batch and return (loss, n_seed)."""
    bs = int(batch.batch_size)
    if mode == "subsample":
        x, edge_index, y_seed, seed_idx = drop_non_seed_nodes(batch, drop_rate)
        out = model(x, edge_index)
        seed_logits = out[seed_idx]
    else:
        out = model(batch.x, batch.edge_index)
        seed_logits = out[:bs]
        y_seed = batch.y[:bs]

    loss = criterion(seed_logits, y_seed)
    return loss, bs


def dp_train_step(model, batch, mode, drop_rate, criterion, optimizer,
                  clip, sigma):
    """One DP-SGD step on a neighbor-sampled batch.

    Per-seed microbatching: the batch loss decomposes across seed nodes, so
    we backprop each seed's loss separately, clip each per-seed gradient to
    l2 norm `clip`, sum, add Gaussian noise with std sigma * clip, and
    average over the batch before the optimizer step. (Opacus's hook-based
    per-sample gradients require independent batch rows, which message
    passing violates — hence the explicit loop. The forward pass is shared;
    only backward runs per seed.)

    Note on sensitivity: clipping bounds each seed TERM's contribution. A
    single graph node can appear in several seeds' computation subgraphs
    within a batch, so for node-level DP the per-step sensitivity is K * clip
    where K bounds the number of terms a node touches (see
    --occurrence_bound); K=1 corresponds to example-level DP on seed labels.

    Returns the mean (unclipped) loss over seeds for logging.
    """
    import torch

    bs = int(batch.batch_size)
    if mode == "dp_subsample":
        x, edge_index, y_seed, seed_idx = drop_non_seed_nodes(batch, drop_rate)
    else:
        x, edge_index = batch.x, batch.edge_index
        y_seed = batch.y[:bs]
        seed_idx = torch.arange(bs)

    params = [p for p in model.parameters() if p.requires_grad]
    grad_sum = [torch.zeros_like(p) for p in params]

    out = model(x, edge_index)
    seed_logits = out[seed_idx]

    total_loss = 0.0
    for i in range(bs):
        loss_i = criterion(seed_logits[i:i + 1], y_seed[i:i + 1])
        grads = torch.autograd.grad(loss_i, params, retain_graph=(i < bs - 1))
        norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        coef = min(1.0, clip / (float(norm) + 1e-6))
        for acc, g in zip(grad_sum, grads):
            acc.add_(g, alpha=coef)
        total_loss += float(loss_i.detach())

    optimizer.zero_grad()
    for p, acc in zip(params, grad_sum):
        noise = torch.randn_like(acc) * (sigma * clip)
        p.grad = (acc + noise) / bs
    optimizer.step()
    return total_loss / max(bs, 1)


def evaluate_full_graph(model, data):
    """Evaluate on the full graph using the public Planetoid masks."""
    import torch
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = {}
        for split in ["train", "val", "test"]:
            mask = getattr(data, f"{split}_mask")
            correct = (pred[mask] == data.y[mask]).sum().item()
            total = int(mask.sum().item())
            accs[split] = correct / total if total > 0 else float("nan")
    return accs


def run_experiment(args, verbose=True):
    """Train + eval one (mode, drop_rate, seed) configuration.

    Returns the final {train,val,test} accuracy dict so sweep drivers can
    aggregate over seeds without re-parsing stdout.
    """
    set_seed(args.seed)

    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    dataset_name = getattr(args, "dataset", "cora")
    dataset, data = load_planetoid(dataset_name)
    data = data.to(device)

    dp_mode = args.mode in ("dp", "dp_subsample")

    if verbose:
        print(f"{dataset_name} loaded: nodes={data.num_nodes}, edges={data.num_edges}, "
              f"features={dataset.num_features}, classes={dataset.num_classes}")
        print(f"Mode={args.mode}  drop_rate={args.drop_rate}  epochs={args.epochs}  "
              f"num_neighbors={args.num_neighbors}  batch_size={args.batch_size}  "
              f"seed={args.seed}")
        if dp_mode:
            print(f"DP-SGD: clip={args.clip}  sigma={args.sigma}  "
                  f"delta={args.delta:g}  accountant={args.accountant}")

    train_loader = NeighborSampler(
        data,
        input_nodes=data.train_mask,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = build_model(
        in_channels=dataset.num_features,
        hidden=args.hidden,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            if dp_mode:
                loss_val = dp_train_step(model, batch, args.mode,
                                         args.drop_rate, criterion, optimizer,
                                         args.clip, args.sigma)
                epoch_loss += loss_val
            else:
                optimizer.zero_grad()
                loss, _ = batch_forward_loss(model, batch, args.mode,
                                             args.drop_rate, criterion)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            n_batches += 1
            total_steps += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == args.epochs):
            accs = evaluate_full_graph(model, data)
            print(f"epoch {epoch:3d}  loss={epoch_loss / max(n_batches, 1):.4f}  "
                  f"train={accs['train']:.4f}  val={accs['val']:.4f}  "
                  f"test={accs['test']:.4f}")

    final_accs = evaluate_full_graph(model, data)

    if dp_mode and getattr(args, "accountant", "none") != "none":
        import accounting
        num_train = int(data.train_mask.sum().item())
        sample_rate = min(1.0, args.batch_size / num_train)
        occurrence_bound = max(getattr(args, "occurrence_bound", 1.0), 1.0)
        epsilon = accounting.compute_epsilon(
            args.accountant,
            noise_multiplier=args.sigma / occurrence_bound,
            sample_rate=sample_rate,
            steps=total_steps,
            delta=args.delta,
            dominating_pair=getattr(args, "dominating_pair", None),
            grid=getattr(args, "pld_grid", 1e-4),
        )
        final_accs["epsilon"] = epsilon
        if verbose:
            print(f"\nprivacy: epsilon={epsilon:.4f} at delta={args.delta:g}  "
                  f"({args.accountant}, steps={total_steps}, "
                  f"sample_rate={sample_rate:.4f})")

    if verbose:
        eps_str = ""
        if "epsilon" in final_accs:
            eps_str = f"eps={final_accs['epsilon']:.4f}  "
        print(f"\nfinal  mode={args.mode}  drop_rate={args.drop_rate}  {eps_str}"
              f"train={final_accs['train']:.4f}  val={final_accs['val']:.4f}  "
              f"test={final_accs['test']:.4f}")
    return final_accs


def main():
    _check_imports()
    args = parse_args()
    run_experiment(args, verbose=True)


if __name__ == "__main__":
    main()
