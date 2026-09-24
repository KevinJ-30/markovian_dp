"""
SparseGNN experiment CLI: root sampling (p1) + SparseExpand (p2, r) with a GNN
base mechanism, swept over (p1, p2, r, sigma) and written to a results CSV.

  python -m src.experiments.sparse --dataset ppi-large --model multilabel_gnn --direction in \
      --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --K_in 5 --K_out 5

Add --dp for the clip+noise path; epsilon is attached afterwards by
`python -m src.experiments.compute_epsilon --csv <results.csv>`.
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.datasets import load_dataset                       # noqa: E402
from src.models.gnn_mechanism import GNNMechanism           # noqa: E402
from src.models.multilabel_mechanism import MultiLabelGNNMechanism  # noqa: E402
from src.models.binary_mechanism import BinaryGNNMechanism  # noqa: E402
from src.models.regression_mechanism import RegressionGNNMechanism  # noqa: E402
from src.models.bootstrap import BootstrapConfig  # noqa: E402
from src.models.layers import VALID_AGGR  # noqa: E402
from src.processing.sparse_expand import build_adjacency, sparse_expand  # noqa: E402
from src.privacy.accounting import calibrate_sparsegnn_noise  # noqa: E402
from src.training.sparse_gnn import train_sparse_gnn          # noqa: E402

from src.processing.graphs import (
    make_training_graph,
    max_degrees,
    preprocess_edges,
    preprocess_graph,
)

from src.models.objectives import trivial_baseline


_MECHANISMS = {
    'gnn': GNNMechanism,
    'multilabel_gnn': MultiLabelGNNMechanism,
    'binary_gnn': BinaryGNNMechanism,
    'regression_gnn': RegressionGNNMechanism,
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

    A mean near 1.0 means roots are isolated and the GNN has degenerated to an
    MLP regardless of p2 and r.
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


def plot_sweep(summary, dataset_name, out_dir, metric):
    """Plot the test metric vs p2, one line per p1 (linestyle per swept r).

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
    ax.set_ylabel(f'test {metric}')
    r_txt = f'r={rs[0]}' if len(rs) == 1 else f'r in {rs}'
    ax.set_title(f'{dataset_name}: SparseGNN test {metric} vs sparsification ({r_txt}, no DP)')
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
                   choices=['gnn', 'multilabel_gnn', 'binary_gnn',
                            'regression_gnn'],
                   default='gnn',
                   help="base mechanism g0: 'gnn' (single-label GNN), "
                        "'multilabel_gnn' (BCE + micro-F1, for multilabel tasks), "
                        "'binary_gnn' (BCE + AUROC, for binary tasks), or "
                        "'regression_gnn' (MSE training + R² evaluation, for regression tasks)")
    p.add_argument('--aggr', choices=VALID_AGGR, default='mean',
                   help="message-passing aggregator: 'mean' (GraphSAGE), "
                        "'gcn' (normalized GCN), or 'gin' (sum aggregation, "
                        "two-layer ReLU MLP, fixed epsilon=0)")
    p.add_argument('--train_domains', nargs='+',
                   help='domain datasets: canonical training domain slugs')
    p.add_argument('--val_domains', nargs='+',
                   help='domain datasets: canonical validation domain slugs')
    p.add_argument('--test_domains', nargs='+',
                   help='domain datasets: canonical test domain slugs')
    p.add_argument('--domain_split_seed', type=int, default=0,
                   help='seed for a shared validation/test domain split')
    p.add_argument('--domain_val_ratio', type=float, default=0.2,
                   help='validation fraction within a shared target domain')
    # Training is always inductive: run.py constructs separate training and
    # evaluation graphs before dispatching to SparseGNN.
    p.add_argument('--common_inductive_split', action='store_true',
                   help='use the saved deterministic 60/20/20 split (or native '
                        'masks for fixed-split, regression, and multilabel datasets) '
                        'and delete all inter-partition edges before private training')
    p.add_argument('--split_root', default='data/inductive_splits',
                   help='directory holding common saved inductive split indices')
    p.add_argument('--split_seed', type=int, default=0,
                   help='seed identifying the common saved inductive split')
    p.add_argument('--direction', choices=['in', 'out'], default='in',
                   help="SparseExpand orientation: 'in' expands along incoming "
                        "edges so messages flow toward the root; 'out' is the "
                        "legacy orientation ablation. Privacy accounting uses "
                        "the in-expansion Theorem 5.4 pair independently.")
    # Paper parameters (each accepts one or more values → swept as a grid)
    p.add_argument('--p1', type=float, nargs='+', default=[0.5],
                   help='root-sampling probability p1 (Bernoulli per node); '
                        'pass several to sweep, e.g. --p1 0.25 0.5 1.0')
    p.add_argument('--p2', type=float, nargs='+', default=[0.5],
                   help='edge-sparsification probability p2 before the '
                        '20-edge incoming sampling cap; pass several to sweep')
    p.add_argument('--r', type=int, nargs='+', default=[2],
                   help='maximum expansion distance r (SparseExpand levels); '
                        'pass several to sweep, e.g. --r 1 2 3')
    p.add_argument('--T', type=int, default=200,
                   help='number of training steps T')
    # Model / optimization
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2, help='message-passing layers L')
    p.add_argument('--dropout', type=float, default=0.5)
    # 'auto' = adam, DP or not (see the opt_kind comment in main()).
    p.add_argument('--optimizer', choices=['auto', 'adam', 'sgd'],
                   default='auto',
                   help="'auto' = Adam for non-DP, SGD for DP.  Pin to 'sgd' "
                        "so a non-DP reference differs from its DP runs only "
                        "by the noise; 'adam' gives the best achievable "
                        "non-private number.  Costs no privacy either way.")
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.0,
                   help='SGD momentum for the DP path (post-processing, no '
                        'privacy cost; ignored by the non-DP Adam path)')
    p.add_argument('--weight_decay', type=float, default=5e-4)
    # DP (off by default)
    p.add_argument('--dp', action='store_true', help='enable DP clip+noise path')
    p.add_argument('--clip', type=float, default=1.0, help='clipping norm C (DP)')
    noise_selection = p.add_mutually_exclusive_group()
    noise_selection.add_argument(
        '--sigma', type=float, nargs='+',
        help='Opacus noise multiplier(s); pass several to sweep, e.g. --sigma 2 5 10')
    noise_selection.add_argument(
        '--target_epsilon', type=float,
        help='calibrate one noise multiplier per (p1, p2, r) configuration')
    p.add_argument('--target_delta', type=float,
                   help='target delta required with --target_epsilon')
    p.add_argument('--legacy_shells', action='store_true',
                   help='drop the union-graph correction in the in-process '
                        'calibration (n_d = K^d instead of 2*K^d)')
    p.add_argument('--accounting_grid', type=float, default=1e-3,
                   help='dp_accounting value discretization interval')
    p.add_argument('--calibration_rtol', type=float, default=1e-3,
                   help='relative tolerance for calibrated noise multiplier')
    p.add_argument('--calibration_atol', type=float, default=1e-6,
                   help='absolute tolerance for calibrated noise multiplier')
    p.add_argument('--K_in', type=int, default=None,
                   help='accounting parameter and default for K_out; does not '
                        'cap incoming arcs in directed mode. Undirected mode '
                        'requires K_in == K_out')
    p.add_argument('--K_out', type=int, default=None,
                   help='cap max out-degree before training (defaults to K_in)')
    p.add_argument('--cap_mode', choices=['auto', 'directed', 'undirected'],
                   default='auto',
                   help="degree capping: 'auto' (default) = 'directed' for "
                        "every graph -- an undirected graph is treated as a "
                        "directed arc set, so dropping an arc does not drop "
                        "its reverse; only outgoing degree is capped. "
                        "'undirected' caps the undirected degree at "
                        "K_in (=K_out) and keeps both arcs of every surviving "
                        "edge, which preserves symmetry but is not required")
    p.add_argument('--cap_seed', type=int, default=None,
                   help='RNG seed for the degree cap.  Default (unset) uses the '
                        'run seed, so each --seed trains on its own capped '
                        'graph and the reported spread includes the cap as a '
                        'source of variance.  Pin it to an int to hold the '
                        'graph fixed across seeds (isolating model/sampling/'
                        'noise variance instead).')
    # General
    p.add_argument('--bootstrap-confidence', type=float, default=0.95,
                   help='confidence level for final test node-bootstrap intervals')
    p.add_argument('--bootstrap-resamples', type=int, default=1000,
                   help='node bootstrap replicates; zero disables intervals')
    p.add_argument('--bootstrap-seed', type=int, default=0,
                   help='independent random seed for test bootstrap resampling')
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
                   help='save a sweep plot (test metric vs p2, line per p1, linestyle per r)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--progress_every', type=int,
                   help='verbose progress interval; defaults to the validation '
                        'interval and does not change checkpoint selection')
    p.add_argument('--eval_every', type=int, default=0,
                   help='validate/select every N updates and at the final update; '
                        '0 (default) uses one expected epoch (ceil(1/p1))')
    args = p.parse_args()
    if args.eval_every < 0:
        p.error("--eval_every must be nonnegative")
    supplied_domain_roles = [
        args.train_domains is not None,
        args.val_domains is not None,
        args.test_domains is not None,
    ]
    if any(supplied_domain_roles) and not all(supplied_domain_roles):
        p.error("--train_domains, --val_domains, and --test_domains must be "
                "provided together")
    if args.target_epsilon is None and args.target_delta is not None:
        p.error("--target_delta requires --target_epsilon")
    if args.target_epsilon is not None and args.target_delta is None:
        p.error("--target_epsilon requires --target_delta")
    if args.target_epsilon is not None and not args.dp:
        p.error("--target_epsilon requires --dp")
    if args.sigma is None:
        args.sigma = [1.0]
    return args


def main():
    args = parse_args()
    bootstrap = BootstrapConfig(
        confidence_level=args.bootstrap_confidence,
        n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    target_mode = args.target_epsilon is not None
    sigmas = args.sigma if args.dp else [args.sigma[0]]
    grid = list(itertools.product(args.p1, args.p2, args.r))
    if not target_mode:
        grid = [(*cell, sigma) for cell in grid for sigma in sigmas]
    noise_description = (
        f"target_epsilon={args.target_epsilon} target_delta={args.target_delta}"
        if target_mode else f"sigma={sigmas}")

    print(f"\n{'='*66}")
    print(f"SparseGNN  dataset={args.dataset}  device={device}  "
          f"direction={args.direction}  aggr={args.aggr}")
    print(f"  p1={args.p1}  p2={args.p2}  r={args.r}  {noise_description}  "
          f"T={args.T}  L={args.num_layers}  dp={args.dp}  seeds={args.seeds}")
    print("  training=inductive  evaluation=test graph")
    print(f"  sweep: {len(grid)} configuration(s) x {args.seeds} seed(s)")
    print('='*66)

    domain_split = None
    if args.train_domains is not None:
        domain_split = {
            'train': args.train_domains,
            'val': args.val_domains,
            'test': args.test_domains,
            'seed': args.domain_split_seed,
            'val_ratio': args.domain_val_ratio,
        }

    dataset, data = load_dataset(
        args.dataset, device=str(device), domain_split=domain_split)
    is_domain_dataset = bool(getattr(dataset, 'domain_dataset', False))
    normalized_domain_split = (
        getattr(dataset, 'domain_split', None) if is_domain_dataset else None)
    domain_split_id = (
        str(getattr(dataset, 'domain_split_id', ''))
        if is_domain_dataset else '')
    normalized_domain_split_json = (
        json.dumps(normalized_domain_split, sort_keys=True, separators=(',', ':'))
        if normalized_domain_split is not None else '')

    if args.dataset.lower() == 'twitch-explicit' and args.model != 'binary_gnn':
        raise SystemExit(
            "twitch-explicit is a binary AUROC task — use "
            f"--model binary_gnn (got --model {args.model})")
    if args.common_inductive_split and is_domain_dataset:
        raise SystemExit(
            "--common_inductive_split cannot be used with a domain dataset; "
            "its domain_split masks already define the inductive protocol")

    tag = '_dp' if args.dp else ''
    # Dataset names may contain separators; domain runs additionally carry their
    # normalized split fingerprint so distinct protocols cannot overwrite.
    ds_slug = args.dataset.replace(':', '_').replace('/', '_')
    split_tag = f'_{domain_split_id}' if domain_split_id else ''
    csv_path = os.path.join(
        args.out_dir, f'sparse_gnn_{ds_slug}{split_tag}{tag}_results.csv')
    data = data.to(device)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    if args.common_inductive_split:
        from src.processing.splits import load_or_create_inductive_split
        from src.experiments.run import _resolve_task_metadata
        task = _resolve_task_metadata(dataset, {})
        split_strategy = getattr(dataset, 'split_strategy', None) or (
            'native' if task['regression'] or task['multilabel'] else 'stratified')
        split = load_or_create_inductive_split(
            data.clone().cpu(), args.dataset, root=args.split_root,
            seed=args.split_seed, split_strategy=split_strategy, **task)
        masks = split.masks
        data.train_mask = masks['train'].to(device)
        data.val_mask = masks['val'].to(device)
        data.test_mask = masks['test'].to(device)
        full_edge_index = data.edge_index
        within_partition = (
            (data.train_mask[full_edge_index[0]] & data.train_mask[full_edge_index[1]])
            | (data.val_mask[full_edge_index[0]] & data.val_mask[full_edge_index[1]])
            | (data.test_mask[full_edge_index[0]] & data.test_mask[full_edge_index[1]])
        )
        data.edge_index = full_edge_index[:, within_partition]
        print(f"  common inductive split: {split.path}; removed "
              f"{int((~within_partition).sum())} crossing edges")

    evaluation_edges_before = int(data.edge_index.size(1))
    data = preprocess_graph(data)
    print("  preprocessing=remove self-loops, make bidirectional, "
          "deduplicate arcs, optional non-self degree cap")
    print(f"  evaluation graph: edges {evaluation_edges_before} -> "
          f"{int(data.edge_index.size(1))}; max (in,out) "
          f"{max_degrees(data.edge_index, int(data.num_nodes))}")
    # Model/task guard: fail fast on pairings that would crash deep in a shape
    # error (single-label GNN on multilabel targets) or silently report a
    # misleading metric (accuracy on an imbalanced binary task).
    if (getattr(dataset, 'multilabel', False)
            and args.model != 'multilabel_gnn'):
        raise SystemExit(
            f"{args.dataset} is multilabel — use --model multilabel_gnn "
            f"(got --model {args.model})")
    task_type = str(getattr(dataset, 'task_type', ''))
    if 'REGRESSION' in task_type.upper() and args.model != 'regression_gnn':
        raise SystemExit(
            f"{args.dataset} is a regression task ({task_type}) — use "
            f"--model regression_gnn (got --model {args.model}); every "
            f"other mechanism expects integer class labels and will crash "
            f"on this task's float targets")
    # At r=0 a GNN's neighbour weights receive no training signal, but
    # full-graph evaluation still uses them. Keep the warning for archived
    # ablations, but do not describe that configuration as graph-blind.
    # r and L are INDEPENDENT knobs and both are legitimate to vary.  r is the
    # expansion depth and is priced as K_out^r; L is the model depth and does
    # not enter the accounting at all, so depth is free in epsilon.  Only the
    # L > r direction has a (small) measured cost -- the subgraph's boundary
    # nodes have no in-edges during training but do at evaluation, measured at
    # 0.6% mean / 1.9% max on capped arxiv -- so note it and move on.
    for _r in args.r:
        if _r > 0 and args.num_layers > _r:
            print(f"  note: L={args.num_layers} > r={_r}; the model reads "
                  f"{args.num_layers} hops but expansion materializes {_r}, so "
                  f"boundary nodes are aggregated differently at train and "
                  f"eval time. Not a privacy issue -- L is free in epsilon.")
    if 0 in args.r:
        print(f"  WARNING: --r 0 with --model {args.model} is not graph-blind: "
              "its neighbour weights never train but are still used at "
              "evaluation. Use the portable MLP/DP-MLP baseline for a "
              "feature-only comparison.")
    if 'BINARY' in task_type.upper() and args.model != 'binary_gnn':
        print(f"  WARNING: {args.dataset} is a binary task "
              f"({task_type}) — --model binary_gnn (AUROC) is recommended, "
              f"got --model {args.model}")

    test_data = data
    train_data = make_training_graph(test_data)
    edge_index = train_data.edge_index
    source = ('loader training graph' if hasattr(test_data, 'train_edge_index')
              else 'train-induced graph')
    print(f"  {source}: edges {test_data.edge_index.size(1)} -> "
          f"{edge_index.size(1)}; training roots "
          f"{int(train_data.train_mask.sum())}/{int(train_data.num_nodes)}")

    n_nodes = int(train_data.num_nodes)
    K_in_req = args.K_in
    K_out_req = args.K_out if args.K_out is not None else args.K_in
    cap_mode = "directed" if args.cap_mode == "auto" else args.cap_mode
    if K_out_req is not None and cap_mode == "undirected" and K_in_req != K_out_req:
        raise SystemExit("--cap_mode undirected needs K_in == K_out")
    raw_train_ei = edge_index

    def _build_graphs(cap_seed, label):
        """Preprocess one graph draw and derive every cap-dependent artifact."""
        before = max_degrees(raw_train_ei, n_nodes)
        # Offset from the root-sampling stream, which is seeded with the same
        # integer by default. Independent generators from the same seed would
        # otherwise couple cap selection to the first root-sampling draw.
        cap_gen = torch.Generator().manual_seed(int(cap_seed) + 20_000)
        train_ei = preprocess_edges(
            raw_train_ei,
            n_nodes,
            max_in_degree=K_in_req,
            max_out_degree=K_out_req,
            degree_cap_mode=cap_mode if K_out_req is not None else "directed",
            add_self_loops=False,
            generator=cap_gen,
        )
        applied_mode = cap_mode if K_out_req is not None else ""
        achieved = max_degrees(train_ei, n_nodes)
        if label:
            cap_description = (
                f"K_out={K_out_req} mode={applied_mode} "
                if K_out_req is not None else "no degree cap "
            )
            print(f"  structural preprocessing [{label}]: {cap_description}"
                  f"cap_seed={cap_seed}; max (in,out) {before} -> {achieved}; "
                  f"edges {raw_train_ei.size(1)} -> {train_ei.size(1)}")
        k_in = K_in_req if K_in_req is not None else achieved[0]
        k_out = K_out_req
        if k_out is None and args.dp:
            if label:
                print("  WARNING: --dp without --K_out (or --K_in as its "
                      "default) — no outgoing degree cap is enforced; post-hoc "
                      "epsilon will use the graph's preprocessed max degrees.")
            k_out = achieved[1]
        return {'train_ei': train_ei, 'cap_mode': applied_mode,
                'K_in': k_in, 'K_out': k_out, 'cap_seed': cap_seed,
                'achieved': achieved,
                'adj': build_adjacency(train_ei, n_nodes,
                                       direction=args.direction)}

    # One capped graph per seed, unless --cap_seed pins it (which reproduces the
    # old behaviour: identical graph for every seed, so the reported spread
    # isolates model/sampling/noise variance from the cap's).
    if args.cap_seed is None:
        graphs = {s: _build_graphs(s, 'training graph' if s == 0 else '')
                  for s in range(args.seeds)}
    else:
        shared = _build_graphs(args.cap_seed, 'training graph')
        graphs = {s: shared for s in range(args.seeds)}

    adj = graphs[0]['adj']          # for the subgraph-size report only

    candidate_nodes = torch.where(train_data.train_mask)[0]

    _report_subgraph_size(adj, candidate_nodes, int(train_data.num_nodes),
                          p2=max(args.p2), r=max(args.r),
                          direction=args.direction)

    _probe = _MECHANISMS[args.model]
    _metric = getattr(_probe, 'metric_name', 'accuracy')
    trivial = trivial_baseline(test_data, _metric)
    print(f"  trivial baseline ({_metric}) on test: {trivial:.4f} "
          f"— every result below must clear this")

    summary = []   # (p1, p2, r, sigma, test_mean, test_std, val_mean, val_std)

    # Write to <name>.partial and rename only on success.  Rows are still
    # flushed as they complete, so a killed run leaves an inspectable partial
    # file — the final path is published only after the entire sweep succeeds.
    partial_path = csv_path + '.partial'
    with open(partial_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        # train_acc/val_acc/test_acc hold whatever `metric` names — accuracy for
        # single-label GNN, micro-F1 for multilabel, AUROC for binary, R² for regression.
        w.writerow(['dataset', 'domain_split', 'domain_split_id',
                    'model', 'aggr', 'metric', 'direction', 'p1', 'p2', 'r',
                    'sigma', 'clip', 'K_in',
                    'K_out', 'cap_mode', 'optimizer', 'lr', 'momentum', 'T',
                    'L', 'dp', 'target_epsilon', 'target_delta',
                    'calibrated_epsilon', 'accounting_grid',
                    'calibration_rtol', 'calibration_evaluations',
                    'noise_std', 'noise_variance', 'seed', 'step', 'hidden',
                    'dropout', 'weight_decay', 'seeds', 'cap_seed',
                    'K_in_achieved', 'K_out_achieved',
                    'train_acc', 'val_acc', 'test_acc', 'trivial_baseline',
                    'train_auroc', 'val_auroc', 'test_auroc',
                    # Secondary metric for binary_gnn (metric_name="auroc" is
                    # primary, above): plain accuracy, meaningful only next to
                    # AUROC on an imbalanced split -- see binary_mechanism.py.
                    'train_bin_acc', 'val_bin_acc', 'test_bin_acc',
                    'test_confidence_intervals', 'selection'])

        for cell in grid:
            calibration = None
            if target_mode:
                p1, p2, r = cell
                calibration = calibrate_sparsegnn_noise(
                    target_epsilon=args.target_epsilon,
                    target_delta=args.target_delta, p1=p1, p2=p2, r=r,
                    K_in=(K_in_req if K_in_req is not None else
                          max(graph['K_in'] for graph in graphs.values())),
                    K_out=K_out_req, steps=args.T, clip=args.clip,
                    grid=args.accounting_grid,
                    sigma_rtol=args.calibration_rtol,
                    sigma_atol=args.calibration_atol,
                    union_safe=not args.legacy_shells,
                )
                sigma = calibration.noise_multiplier
                print(f"\n[p1={p1} p2={p2} r={r}]")
                print("  calibrated "
                      f"target=(epsilon={calibration.target_epsilon:g}, "
                      f"delta={calibration.delta:g}) "
                      f"epsilon={calibration.epsilon:.6g} "
                      f"sigma={calibration.noise_multiplier:.6g} "
                      f"noise_std={calibration.noise_std:.6g} "
                      f"evaluations={calibration.evaluations}")
            else:
                p1, p2, r, sigma = cell
                print(f"\n[p1={p1} p2={p2} r={r}" +
                      (f" sigma={sigma}]" if args.dp else "]"))
            calibration_fields = (
                [calibration.target_epsilon, calibration.delta,
                 calibration.epsilon, args.accounting_grid,
                 args.calibration_rtol, calibration.evaluations]
                if calibration is not None else [""] * 6)
            noise_fields = (
                [sigma * args.clip, (sigma * args.clip) ** 2]
                if args.dp else ["", ""])
            tests, vals = [], []
            for seed in range(args.seeds):
                _set_seed(seed)
                gph = graphs[seed]
                Mechanism = _MECHANISMS[args.model]
                extra = {'aggr': args.aggr}
                if args.model == 'gnn':
                    extra['metric_ignore_label'] = getattr(
                        dataset, 'metric_ignore_label', None)
                mech = Mechanism(
                    train_data, num_features, num_classes,
                    hidden=args.hidden, num_layers=args.num_layers,
                    dropout=args.dropout, device=device, **extra,
                )
                # Adam everywhere, DP or not.  Three reasons:
                #   1. Every baseline we compare against is Adam -- DPAR is
                #      literally DPAdamGaussianOptimizer upstream, ProGAP
                #      defaults to it -- so a same-optimizer comparison is
                #      the defensible one.
                #   2. It lets us adopt GraphSAINT's published per-dataset
                #      config (lr=0.01 and their dropout values) as a package;
                #      those were grid-searched under Adam.
                #   3. Under DP it is still free: the optimizer is
                #      post-processing of the already-noised gradient, so the
                #      accountant never sees it.
                # Measured on PPI-large (p2=0.1, r=2, K=5, T=300, sigma=5.25,
                # eps~2): Adam@0.01 -> 0.4129 vs SGD@1.0 -> 0.4141, a tie
                # inside seed noise, so this costs nothing in utility.
                # --optimizer sgd remains available for the ablation.
                opt_kind = (args.optimizer if args.optimizer != 'auto'
                            else 'adam')
                mech.build_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                                     kind=opt_kind, momentum=args.momentum)

                accs = train_sparse_gnn(
                    mech, train_data, test_data, adj=gph['adj'],
                    direction=args.direction, p1=p1, p2=p2, r=r, T=args.T,
                    dp=args.dp, clip=args.clip, sigma=sigma, seed=seed,
                    eval_every=args.eval_every, track_every=args.track_every,
                    progress_every=args.progress_every,
                    verbose=args.verbose, bootstrap=bootstrap,
                )
                history = accs.pop('history', [])
                tests.append(accs['test'])
                vals.append(accs['val'])
                print(f"  seed={seed}  metric={_metric}  train={accs['train']:.4f}  "
                      f"val={accs['val']:.4f}  test={accs['test']:.4f}")
                if 'selection' in accs:
                    print("    selection=" + json.dumps(accs['selection'], allow_nan=False))
                if 'test_confidence_intervals' in accs:
                    print("    test_confidence_intervals=" + json.dumps(
                        accs['test_confidence_intervals'], allow_nan=False))

                def _write_row(step, m):
                    w.writerow([args.dataset, normalized_domain_split_json,
                                domain_split_id, args.model, args.aggr,
                                mech.metric_name, args.direction, p1, p2, r,
                                sigma, args.clip,
                                gph['K_in'] if gph['K_in'] is not None else '',
                                gph['K_out'] if gph['K_out'] is not None else '',
                                gph['cap_mode'], opt_kind, args.lr, args.momentum,
                                args.T, args.num_layers, args.dp,
                                *calibration_fields, *noise_fields, seed, step,
                                args.hidden, args.dropout, args.weight_decay,
                                args.seeds, gph['cap_seed'],
                                gph['achieved'][0], gph['achieved'][1],
                                f"{m['train']:.5f}", f"{m['val']:.5f}",
                                f"{m['test']:.5f}", f"{trivial:.5f}",
                                *(f"{m[k]:.5f}" if k in m else ''
                                  for k in ('train_auroc', 'val_auroc',
                                            'test_auroc',
                                            'train_bin_acc', 'val_bin_acc',
                                            'test_bin_acc')),
                                (json.dumps(m['test_confidence_intervals'],
                                            allow_nan=False)
                                 if 'test_confidence_intervals' in m else ''),
                                (json.dumps(m['selection'], allow_nan=False)
                                 if 'selection' in m else '')])

                for h in history:
                    # The T row is the selected model, charged for all T updates.
                    if h['step'] < args.T:
                        _write_row(h['step'], h)
                _write_row(args.T, accs)
                fh.flush()   # persist each row so a killed run keeps its rows

            tm, ts = _mean_std(tests)
            vm, vs = _mean_std(vals)
            summary.append((p1, p2, r, sigma, tm, ts, vm, vs))
            beats_trivial = tm > trivial
            mark = "" if beats_trivial else "   <-- BELOW TRIVIAL BASELINE"
            print(f"  >> test {_metric} {tm:.4f} +/- {ts:.4f}   "
                  f"val {vm:.4f} +/- {vs:.4f}{mark}")

    os.replace(partial_path, csv_path)

    # Sweep summary table (sorted by validation metric, best first).
    print(f"\n{'='*66}")
    print(f"metric={_metric} (higher is better; ranked by validation)")
    print(f"{'p1':>5} {'p2':>5} {'r':>3} {'sigma':>6} {'test':>16} {'val':>16}")
    print('-'*66)
    for p1, p2, r, sigma, tm, ts, vm, vs in sorted(
            summary, key=lambda s: s[6] if s[6] == s[6] else float('-inf'),
            reverse=True):
        print(f"{p1:>5} {p2:>5} {r:>3} {sigma:>6}   {tm:.4f} +/- {ts:.4f}   "
              f"{vm:.4f} +/- {vs:.4f}")
    print(f"\nresults written to {csv_path}")
    if args.dp:
        print("compute epsilon post-hoc with:  python -m src.experiments.compute_epsilon "
              f"--csv {csv_path} --delta <delta> --grid {args.accounting_grid:g}")

    if args.plot:
        plot_dataset_name = (
            f"{args.dataset}_{domain_split_id}"
            if domain_split_id else args.dataset)
        plot_path = plot_sweep(
            [(s[0], s[1], s[2], s[4], s[5], s[6], s[7])
             for s in summary if s[3] == sigmas[0]],
            plot_dataset_name, args.out_dir, _metric)
        if plot_path:
            print(f"plot written to {plot_path}")


if __name__ == '__main__':
    main()
