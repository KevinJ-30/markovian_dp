"""
SparseGNN experiment CLI — paper Algorithms 1 & 2 (current default sparsification).

Runs the composite-subsampling mechanism (root sampling p1 + SparseExpand p2/r)
with a GNN base mechanism for node classification.  Defaults to CiteSeer, no DP.

Examples (from repo root):
  # Sanity: p1=p2=1 recovers (near) full-graph GCN
  python -m src.sparse.run --dataset citeseer --p1 1.0 --p2 1.0 --r 2 --T 200 --seeds 3

  # The actual sparsified mechanism
  python -m src.sparse.run --dataset citeseer --p1 0.5 --p2 0.5 --r 2 --T 200 --seeds 3

DP (--dp) is prepared but off by default; see src/sparse/accounting.py for the
Theorem 3 dominating pair used for accounting.
"""

import argparse
import csv
import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets import load_dataset                       # noqa: E402
from src.sparse.gnn_mechanism import GNNMechanism           # noqa: E402
from src.sparse.mlp_mechanism import MLPMechanism           # noqa: E402
from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism  # noqa: E402
from src.sparse.binary_mechanism import BinaryGNNMechanism  # noqa: E402
from src.sparse.sparse_expand import (                      # noqa: E402
    build_adjacency, cap_degrees, cap_degrees_undirected, edge_set_is_symmetric,
    max_degrees, sparse_expand,
)
from src.sparse.sparse_gnn import train_sparse_gnn          # noqa: E402


_MECHANISMS = {
    'gnn': GNNMechanism,
    'mlp': MLPMechanism,
    'multilabel_gnn': MultiLabelGNNMechanism,
    'binary_gnn': BinaryGNNMechanism,
}


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def _report_subgraph_size(adj, candidate_nodes, num_nodes, *, p2, r, direction,
                          n_probe=512):
    """Log the mean rooted-subgraph size at the widest sweep setting.

    This is the diagnostic that catches an expansion which reaches nothing: if
    the mean is ~1.0 the roots are isolated and the GNN degenerates to an MLP
    regardless of p2 and r, which is exactly what the pre-v35 out-orientation
    did on graphs whose degree mass sits on incoming edges (ogbn-arxiv: max
    in-degree 3015 vs max out-degree 221).
    """
    pool = (torch.arange(num_nodes) if candidate_nodes is None
            else candidate_nodes.cpu())
    if pool.numel() == 0 or r == 0:
        return
    gen = torch.Generator().manual_seed(999)
    probe = pool[torch.randperm(int(pool.numel()), generator=gen)[:n_probe]]
    sizes = [sparse_expand(adj, int(v), p2, r, generator=gen,
                           direction=direction).num_nodes
             for v in probe.tolist()]
    mean = sum(sizes) / len(sizes)
    print(f"  direction={direction}: mean rooted-subgraph size at p2={p2}, "
          f"r={r} is {mean:.2f} nodes (over {len(sizes)} probe roots)")
    if mean < 1.05:
        print("  WARNING: roots are effectively isolated — the graph "
              "contributes nothing beyond the root's own features.")


def trivial_baseline(data, metric):
    """Score of the best label-only predictor, for the dataset's own metric.

    This is the floor every result must clear, and it is recorded in the CSV so
    a sweep can never again look like a result while sitting under chance — the
    PPI runs of 2026-08-11 spent a night doing exactly that (best 0.4756 against
    a trivial 0.4608).

      accuracy  -> most frequent training class, evaluated on test
      micro_f1  -> predict every label positive: 2p/(1+p) at positive rate p
      auroc     -> 0.5 by definition
    """
    import torch as _t
    if metric == "auroc":
        return 0.5
    y, te = data.y, data.test_mask
    if metric == "micro_f1":
        p = float(y[te].float().mean())
        return 2 * p / (1 + p) if p > 0 else float("nan")
    tr_counts = _t.bincount(y[data.train_mask].view(-1))
    majority = int(tr_counts.argmax())
    return float((y[te].view(-1) == majority).float().mean())


def plot_sweep(summary, dataset_name, out_dir):
    """Plot test accuracy vs p2, one line per p1 (linestyle per r if r is swept).

    `summary` is a list of (p1, p2, r, test_mean, test_std, val_mean, val_std).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    p1s = sorted({s[0] for s in summary})
    rs = sorted({s[2] for s in summary})
    linestyles = ['-', '--', ':', '-.']

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p1 in p1s:
        for ri, r in enumerate(rs):
            rows = sorted((s for s in summary if s[0] == p1 and s[2] == r),
                          key=lambda s: s[1])
            if not rows:
                continue
            xs = [s[1] for s in rows]
            ys = [s[3] for s in rows]
            es = [s[4] for s in rows]
            label = f'p1={p1}' + (f', r={r}' if len(rs) > 1 else '')
            ax.errorbar(xs, ys, yerr=es, fmt='o' + linestyles[ri % len(linestyles)],
                        capsize=4, label=label)
    ax.set_xlabel('edge-sampling probability p2  (1.0 = all edges)')
    ax.set_ylabel('test accuracy')
    r_txt = f'r={rs[0]}' if len(rs) == 1 else f'r in {rs}'
    ax.set_title(f'{dataset_name}: SparseGNN test accuracy vs sparsification ({r_txt}, no DP)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='root-sampling p1')
    fig.tight_layout()
    path = os.path.join(out_dir, f'sparse_gnn_{dataset_name}_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='citeseer',
                   help='cora | citeseer | pubmed | ...')
    p.add_argument('--model',
                   choices=['gnn', 'mlp', 'multilabel_gnn', 'binary_gnn'],
                   default='gnn',
                   help="base mechanism g0: 'gnn' (GCN, single-label), 'mlp' "
                        "(graph-blind Stage-0 baseline; use with --r 0), "
                        "'multilabel_gnn' (BCE + micro-F1, for PPI), or "
                        "'binary_gnn' (BCE + AUROC, for RelBench entity tasks)")
    p.add_argument('--aggr', choices=['mean', 'gcn'], default='mean',
                   help="message-passing aggregator: 'mean' (GraphSAGE) makes "
                        "the rooted-subgraph computation agree EXACTLY with "
                        "full-graph inference, so the eval protocol is exact; "
                        "'gcn' (symmetric normalization) only approximates it, "
                        "with error growing in graph density (~0.3%% on capped "
                        "ogbn-arxiv, 150-400%% on PPI). Use 'gcn' to reproduce "
                        "pre-2026-08-12 results.")
    p.add_argument('--relbench_root', choices=['row', 'entity'], default='row',
                   help='RelBench only: root one prediction per task ROW (all '
                        'supervision) or per ENTITY (labels aggregated)')
    p.add_argument('--relbench_reverse_edges', action='store_true',
                   help='RelBench only: also add parent->child arcs; enriches '
                        'neighbourhoods but raises K_out and hence epsilon')
    p.add_argument('--inductive', action='store_true',
                   help='train on the train-induced subgraph only (expansion '
                        'never touches val/test nodes — the privacy-honest '
                        'setting); evaluate with full-graph inductive inference')
    p.add_argument('--direction', choices=['in', 'out'], default='in',
                   help="SparseExpand orientation: 'in' = Algorithm 5, expand "
                        "along incoming edges so messages flow toward the root "
                        "(correct for message passing; accounted by Theorem "
                        "6.4); 'out' = legacy Algorithm 2/4, kept for the "
                        "orientation ablation (accounted by Theorem 4.5)")
    # Paper parameters (each accepts one or more values → swept as a grid)
    p.add_argument('--p1', type=float, nargs='+', default=[0.5],
                   help='root-sampling probability p1 (Bernoulli per node); '
                        'pass several to sweep, e.g. --p1 0.25 0.5 1.0')
    p.add_argument('--p2', type=float, nargs='+', default=[0.5],
                   help='edge-sparsification probability p2 (Bernoulli per arc); '
                        'pass several to sweep')
    p.add_argument('--r', type=int, nargs='+', default=[2],
                   help='maximum expansion distance r (SparseExpand levels); '
                        'pass several to sweep, e.g. --r 1 2 3')
    p.add_argument('--T', type=int, default=200,
                   help='number of training steps T')
    # Model / optimization
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2, help='GCN layers L')
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--optimizer', choices=['auto', 'adam', 'sgd'],
                   default='auto',
                   help="'auto' = Adam for non-DP, SGD for DP (the historical "
                        "default).  Pin it to 'sgd' to make a non-DP reference "
                        "differ from its DP runs ONLY by the noise, so the gap "
                        "measures the cost of privacy and not the cost of "
                        "changing optimizer; 'adam' gives the best achievable "
                        "non-private number.  The choice is post-processing "
                        "either way and costs no privacy.")
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.0,
                   help='SGD momentum for the DP path (post-processing, no '
                        'privacy cost; ignored by the non-DP Adam path)')
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--roots_from', choices=['train', 'all'], default='train',
                   help="eligible-root pool: 'train' (labeled roots only) or 'all'")
    # DP (off by default)
    p.add_argument('--dp', action='store_true', help='enable DP clip+noise path')
    p.add_argument('--clip', type=float, default=1.0, help='clipping norm C (DP)')
    p.add_argument('--sigma', type=float, nargs='+', default=[1.0],
                   help='noise multiplier(s); pass several to sweep, e.g. '
                        '--sigma 2 5 10 (only swept when --dp)')
    p.add_argument('--K_in', type=int, default=None,
                   help='cap max in-degree before training (required for a '
                        'valid Theorem 6.4 guarantee; recorded in the CSV for '
                        'post-hoc accounting via src.sparse.compute_epsilon)')
    p.add_argument('--K_out', type=int, default=None,
                   help='cap max out-degree before training (defaults to K_in)')
    p.add_argument('--cap_mode', choices=['auto', 'directed', 'undirected'],
                   default='auto',
                   help="degree capping: 'directed' caps in- and out-arcs "
                        "independently (destroys edge symmetry on undirected "
                        "graphs); 'undirected' caps the undirected degree at "
                        "K_in (=K_out) and keeps both arcs of every surviving "
                        "edge; 'auto' picks undirected iff the graph is "
                        "symmetric and K_in == K_out")
    p.add_argument('--eval_graph', choices=['full', 'train'], default='full',
                   help="graph for `evaluate`: 'full' = data.edge_index "
                        "(uncapped, unfiltered — the deployment view; for "
                        "RelBench the test-cutoff graph); 'train' = the exact "
                        "training graph (inductive-filtered, deduplicated, "
                        "capped), so utility is measured on what the model "
                        "was trained on")
    # General
    p.add_argument('--track_every', type=int, default=0,
                   help='if >0, evaluate every this many steps and write one '
                        'CSV row per checkpoint (step column).  Evaluation '
                        'draws no sampling randomness, so the trajectory is '
                        'identical to an untracked run.  Post-hoc accounting '
                        'then attaches eps(t) to every checkpoint, giving the '
                        'whole privacy-utility curve from a single run.')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--out_dir', default='results')
    p.add_argument('--plot', action='store_true',
                   help='save a sweep plot (test acc vs p2, line per r, subplot per p1)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--eval_every', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_dp' if args.dp else ''
    # relbench:<db>/<task> names contain separators that are not filename-safe.
    ds_slug = args.dataset.replace(':', '_').replace('/', '_')
    csv_path = os.path.join(args.out_dir,
                            f'sparse_gnn_{ds_slug}{tag}_results.csv')

    sigmas = args.sigma if args.dp else [args.sigma[0]]
    grid = list(itertools.product(args.p1, args.p2, args.r, sigmas))

    print(f"\n{'='*66}")
    print(f"SparseGNN  dataset={args.dataset}  device={device}  "
          f"direction={args.direction}  aggr={args.aggr}")
    print(f"  p1={args.p1}  p2={args.p2}  r={args.r}  sigma={sigmas}  T={args.T}  "
          f"L={args.num_layers}  dp={args.dp}  seeds={args.seeds}")
    print(f"  sweep: {len(grid)} (p1,p2,r,sigma) combo(s) x {args.seeds} seed(s)")
    print('='*66)

    dataset, data = load_dataset(
        args.dataset, device=str(device),
        root=args.relbench_root, reverse_edges=args.relbench_reverse_edges,
    ) if str(args.dataset).startswith('relbench') else load_dataset(
        args.dataset, device=str(device))
    data = data.to(device)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Model/task guard: fail fast on pairings that would crash deep in a shape
    # error (single-label GNN on multilabel PPI) or silently report a
    # misleading metric (accuracy on an imbalanced binary RelBench task).
    if getattr(dataset, 'multilabel', False) and args.model != 'multilabel_gnn':
        raise SystemExit(
            f"{args.dataset} is multilabel — use --model multilabel_gnn "
            f"(got --model {args.model})")
    task_type = str(getattr(dataset, 'task_type', ''))
    if 'BINARY' in task_type.upper() and args.model != 'binary_gnn':
        print(f"  WARNING: {args.dataset} is a binary task "
              f"({task_type}) — --model binary_gnn (AUROC) is recommended, "
              f"got --model {args.model}")

    edge_index = data.edge_index
    if args.inductive:
        if hasattr(data, 'train_edge_index'):
            # The loader already built a training graph (RelBench: everything at
            # or before the train cutoff).  Masking on train_mask would be wrong
            # here — a labelled root's neighbours are unlabelled DB rows.
            edge_index = data.train_edge_index
            print(f"  inductive: using the loader's training graph, edges "
                  f"{data.edge_index.size(1)} -> {edge_index.size(1)}")
        else:
            # Training graph = subgraph induced on train nodes: keep only arcs
            # whose BOTH endpoints are training nodes, so SparseExpand can never
            # reach a val/test node during training (no privacy leak).
            # Evaluation still uses the full data.edge_index for inductive
            # inference on held-out nodes.
            is_train = data.train_mask
            both_train = is_train[edge_index[0]] & is_train[edge_index[1]]
            edge_index = edge_index[:, both_train]
            print(f"  inductive: restrict to train-induced subgraph, edges "
                  f"{data.edge_index.size(1)} -> {edge_index.size(1)} "
                  f"(train nodes {int(is_train.sum())}/{int(data.num_nodes)})")

    # The accounting (path counts, Lemma 20) assumes graphs WITHOUT parallel
    # edges; duplicates also get outsized survival odds under capping.  All
    # shipped loaders are simple graphs, but enforce it here so e.g. a RelBench
    # table with two foreign keys to the same parent row cannot break the
    # assumption silently.
    n_arcs_raw = edge_index.size(1)
    edge_index = torch.unique(edge_index.cpu(), dim=1)
    if edge_index.size(1) < n_arcs_raw:
        print(f"  removed {n_arcs_raw - edge_index.size(1)} parallel arc(s): "
              f"{n_arcs_raw} -> {edge_index.size(1)} (simple-graph assumption)")

    K_in, K_out = args.K_in, args.K_out if args.K_out is not None else args.K_in
    if K_in is not None:
        before = max_degrees(edge_index, int(data.num_nodes))
        cap_gen = torch.Generator().manual_seed(12345)
        cap_mode = args.cap_mode
        if cap_mode == 'auto':
            cap_mode = ('undirected' if K_in == K_out and
                        edge_set_is_symmetric(edge_index, int(data.num_nodes))
                        else 'directed')
        if cap_mode == 'undirected':
            if K_in != K_out:
                raise SystemExit("--cap_mode undirected needs K_in == K_out")
            edge_index = cap_degrees_undirected(
                edge_index, int(data.num_nodes), K_in, generator=cap_gen)
        else:
            # NOTE: on a symmetric graph this caps the two arc directions
            # independently and so destroys edge symmetry (~2/3 of surviving
            # arcs lose their reverse at K=5); that is why 'auto' prefers
            # 'undirected' there.
            edge_index = cap_degrees(edge_index, int(data.num_nodes),
                                     K_in=K_in, K_out=K_out, generator=cap_gen)
        after = max_degrees(edge_index, int(data.num_nodes))
        print(f"  degree cap K_in={K_in} K_out={K_out} (mode={cap_mode}): "
              f"max (in,out) {before} -> {after}, edges "
              f"{data.edge_index.size(1)} -> {edge_index.size(1)}")
    else:
        cap_mode = ''
        if args.dp:
            print("  WARNING: --dp without --K_in — the degree-bound "
                  "accounting assumption (Assumption 3.1 / 6.2) is not "
                  "enforced; post-hoc epsilon will use the graph's raw max "
                  "degrees.")
            K_in, K_out = max_degrees(edge_index, int(data.num_nodes))

    # The adjacency is deterministic; build once and reuse across all runs.
    adj = build_adjacency(edge_index, int(data.num_nodes),
                          direction=args.direction)

    candidate_nodes = None
    if args.roots_from == 'train':
        candidate_nodes = torch.where(data.train_mask)[0]

    _report_subgraph_size(adj, candidate_nodes, int(data.num_nodes),
                          p2=max(args.p2), r=max(args.r),
                          direction=args.direction)

    _probe = _MECHANISMS[args.model]
    _metric = getattr(_probe, 'metric_name', 'accuracy')
    trivial = trivial_baseline(data, _metric)
    print(f"  trivial baseline ({_metric}) on test: {trivial:.4f} "
          f"— every result below must clear this")

    summary = []   # (p1, p2, r, sigma, test_mean, test_std, val_mean, val_std)

    # Write to <name>.partial and rename only on success.  Rows are still
    # flushed as they complete, so a killed run leaves an inspectable partial
    # file — but the final path never exists unless the sweep finished, which is
    # what the ladder scripts' resume guard keys on.
    partial_path = csv_path + '.partial'
    with open(partial_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        # train_acc/val_acc/test_acc hold whatever `metric` names — accuracy for
        # single-label GNN/MLP, micro-F1 for multilabel, AUROC for binary.
        w.writerow(['dataset', 'model', 'aggr', 'metric', 'inductive',
                    'direction', 'p1', 'p2', 'r', 'sigma', 'clip', 'K_in',
                    'K_out', 'cap_mode', 'eval_graph', 'optimizer', 'lr',
                    'momentum', 'T', 'L', 'dp', 'seed', 'step',
                    'train_acc', 'val_acc', 'test_acc', 'trivial_baseline',
                    # Secondary, threshold-free metric where the mechanism
                    # reports one (multilabel).  Blank otherwise.
                    'train_auroc', 'val_auroc', 'test_auroc'])

        for p1, p2, r, sigma in grid:
            print(f"\n[p1={p1} p2={p2} r={r}" +
                  (f" sigma={sigma}]" if args.dp else "]"))
            tests, vals = [], []
            for seed in range(args.seeds):
                _set_seed(seed)
                Mechanism = _MECHANISMS[args.model]
                extra = {} if args.model == 'mlp' else {'aggr': args.aggr}
                mech = Mechanism(
                    data, num_features, num_classes,
                    hidden=args.hidden, num_layers=args.num_layers,
                    dropout=args.dropout, device=device, **extra,
                )
                if args.eval_graph == 'train':
                    mech.eval_edge_index = edge_index.to(device)
                opt_kind = (args.optimizer if args.optimizer != 'auto'
                            else ('sgd' if args.dp else 'adam'))
                mech.build_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                                     kind=opt_kind, momentum=args.momentum)

                accs = train_sparse_gnn(
                    mech, data, adj=adj, direction=args.direction,
                    p1=p1, p2=p2, r=r, T=args.T,
                    candidate_nodes=candidate_nodes,
                    dp=args.dp, clip=args.clip, sigma=sigma,
                    seed=seed, eval_every=args.eval_every,
                    track_every=args.track_every, verbose=args.verbose,
                )
                history = accs.pop('history', [])
                tests.append(accs['test'])
                vals.append(accs['val'])
                print(f"  seed={seed}  train={accs['train']:.4f}  "
                      f"val={accs['val']:.4f}  test={accs['test']:.4f}")

                def _write_row(step, m):
                    w.writerow([args.dataset, args.model,
                                '' if args.model == 'mlp' else args.aggr,
                                mech.metric_name, args.inductive,
                                args.direction, p1, p2, r, sigma, args.clip,
                                K_in if K_in is not None else '',
                                K_out if K_out is not None else '',
                                cap_mode, args.eval_graph,
                                opt_kind, args.lr, args.momentum,
                                args.T, args.num_layers, args.dp, seed, step,
                                f"{m['train']:.5f}", f"{m['val']:.5f}",
                                f"{m['test']:.5f}", f"{trivial:.5f}",
                                *(f"{m[k]:.5f}" if k in m else ''
                                  for k in ('train_auroc', 'val_auroc',
                                            'test_auroc'))])

                for h in history:
                    if h['step'] < args.T:   # final checkpoint == the T row
                        _write_row(h['step'], h)
                _write_row(args.T, accs)
                fh.flush()   # persist each row so a killed run keeps its rows

            tm, ts = _mean_std(tests)
            vm, vs = _mean_std(vals)
            summary.append((p1, p2, r, sigma, tm, ts, vm, vs))
            mark = "" if tm > trivial else "   <-- BELOW TRIVIAL BASELINE"
            print(f"  >> test {tm:.4f} +/- {ts:.4f}   "
                  f"val {vm:.4f} +/- {vs:.4f}{mark}")

    os.replace(partial_path, csv_path)

    # Sweep summary table (sorted by test accuracy, best first)
    print(f"\n{'='*66}")
    print(f"{'p1':>5} {'p2':>5} {'r':>3} {'sigma':>6} {'test':>16} {'val':>16}")
    print('-'*66)
    for p1, p2, r, sigma, tm, ts, vm, vs in sorted(summary, key=lambda s: -s[4]):
        print(f"{p1:>5} {p2:>5} {r:>3} {sigma:>6}   {tm:.4f} +/- {ts:.4f}   "
              f"{vm:.4f} +/- {vs:.4f}")
    print(f"\nresults written to {csv_path}")
    if args.dp:
        print("compute epsilon post-hoc with:  python -m src.sparse.compute_epsilon "
              f"--csv {csv_path}")

    if args.plot:
        plot_path = plot_sweep([(s[0], s[1], s[2], s[4], s[5], s[6], s[7])
                                for s in summary if s[3] == sigmas[0]],
                               args.dataset, args.out_dir)
        if plot_path:
            print(f"plot written to {plot_path}")


if __name__ == '__main__':
    main()
