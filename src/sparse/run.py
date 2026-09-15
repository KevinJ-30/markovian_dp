"""
SparseGNN experiment CLI: root sampling (p1) + SparseExpand (p2, r) with a GNN
base mechanism, swept over (p1, p2, r, sigma) and written to a results CSV.

  python -m src.sparse.run --dataset ppi --model multilabel_gnn --direction in \
      --batch_size 512 --epochs 10 --p2 0.1 --r 2 --num_layers 2 \
      --K_in 5 --K_out 5

Prefer --batch_size/--epochs over --p1/--T.  The accountant prices p1 and T, but
they are the wrong units to think in: p1 is a rate, so the same p1 is a very
different batch on two graphs, and a fixed T is a different number of passes
over the data.  run.py converts once the root pool is known --
p1 = B/pool_size and T = epochs/p1 -- and records pool_size, batch_size and
epochs in the CSV so a run is reproducible from its own output.

Add --dp for the clip+noise path; epsilon is attached afterwards by
`python -m src.sparse.compute_epsilon --csv <results.csv>`.
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
from src.sparse.regression_mechanism import RegressionGNNMechanism  # noqa: E402
from src.sparse.sparse_expand import (                      # noqa: E402
    build_adjacency, cap_degrees, cap_degrees_undirected, dedup_arcs,
    max_degrees, sparse_expand,
)
from src.sparse.accounting import calibrate_sparsegnn_noise  # noqa: E402
from src.sparse.sparse_gnn import train_sparse_gnn          # noqa: E402


_MECHANISMS = {
    'gnn': GNNMechanism,
    'mlp': MLPMechanism,
    'multilabel_gnn': MultiLabelGNNMechanism,
    'binary_gnn': BinaryGNNMechanism,
    'regression_gnn': RegressionGNNMechanism,
}

# metric_name -> whether a larger value is better.  accuracy/micro_f1/auroc are
# scores (higher = better); mae is a loss (lower = better).  Console sorting and
# the "below trivial baseline" comparison both need to know which.
_HIGHER_IS_BETTER = {'accuracy': True, 'micro_f1': True, 'auroc': True,
                    'mae': False}


def p1_for_batch(batch_size, pool_size):
    """Root-sampling rate that yields `batch_size` roots per step in expectation.

    Poisson sampling draws each eligible root independently with probability p1,
    so the expected batch is p1 * pool_size (`sparse_gnn.train_sparse_gnn`
    computes exactly this) and p1 = B / pool_size.

    Fixing B rather than p1 is what makes a cross-dataset comparison mean
    anything: p1 is a rate, so the same p1 is a 500-root batch on one graph and
    a 500,000-root batch on another.  It is also the direction that HELPS under
    DP -- p1 falls as the graph grows, and epsilon falls with it through
    amplification by subsampling.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > pool_size:
        raise ValueError(
            f"batch_size {batch_size} exceeds the eligible root pool "
            f"({pool_size}); p1 would exceed 1")
    return batch_size / pool_size


def steps_for_epochs(epochs, p1):
    """Steps T that put `epochs` passes over the root pool at sampling rate p1.

    One epoch is pool_size/B steps, and p1 = B/pool_size, so

        T = epochs / p1 = epochs * pool_size / batch_size

    and the pool size cancels -- T depends only on the rate.

    Worth knowing before raising this: epsilon grows with T, but at a FIXED
    epoch count the two effects partly cancel, because p1 = B/pool_size shrinks
    on a larger graph exactly as T = epochs/p1 grows.  Measured sigma for eps=8,
    delta=1e-6, p2=0.1, r=2, K=5, B=512 -- equal epochs, so these ARE comparable:

        epochs   ppi-large   reddit   yelp   amazon
             1        1.87     1.43   1.17     1.05
             5        3.10     1.87   1.37     1.21
            10        4.36     2.47   1.61     1.37
            20        6.20     3.48   2.17     1.84

    i.e. at equal epochs the BIGGER graph is cheaper, the usual DP-SGD result
    that more data buys accuracy.  Do not extrapolate the ordering: it holds
    over this range and breaks once composition dominates (at 100 epochs Yelp
    needs 10.89 against Reddit's 9.13, and Amazon fails to bracket at all,
    T=245,098).  Pick an epoch budget from a table like this one rather than
    from the asymptotics.
    """
    if not 0 < p1 <= 1:
        raise ValueError(f"p1 must be in (0, 1], got {p1}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    return max(1, round(epochs / p1))


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


def trivial_baseline(data, metric):
    """Score of the best label-only predictor under the dataset's own metric.

      accuracy  -> most frequent training class, evaluated on test
      micro_f1  -> predict every label positive: 2p/(1+p) at positive rate p
      auroc     -> 0.5 by definition
      mae       -> MAE of "always predict the train mean" on test, i.e.
                   mean(|y_test - mean(y_train)|) * target_std.

                   NOTE: targets are scaled by target_std but NOT centred
                   (relbench_data.load_relbench divides by the train std and
                   leaves the mean alone), so the train mean is NOT 0 in the
                   scaled space.  An earlier version computed
                   mean(|y_test|) * target_std, which is the MAE of the
                   ALL-ZERO predictor -- a much weaker bar on the non-negative
                   heavy-tailed targets RelBench regression uses (LTV, sales),
                   so "beats trivial" was too easy to clear.

    Recorded in the CSV as the floor every result must clear (for mae, the
    ceiling every result must undercut -- see _HIGHER_IS_BETTER).  Note
    micro_f1's floor is high but has no ranking ability (its AUROC is 0.5), so
    a model below it may still be learning — compare AUROC too.
    """
    import torch as _t
    if metric == "auroc":
        return 0.5
    y, te = data.y, data.test_mask
    if metric == "micro_f1":
        p = float(y[te].float().mean())
        return 2 * p / (1 + p) if p > 0 else float("nan")
    if metric == "mae":
        target_std = float(getattr(data, 'target_std', 1.0))
        train_mean = float(y[data.train_mask].view(-1).float().mean())
        return float((y[te].view(-1).float() - train_mean).abs().mean()) * target_std
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
                   choices=['gnn', 'mlp', 'multilabel_gnn', 'binary_gnn',
                           'regression_gnn'],
                   default='gnn',
                   help="base mechanism g0: 'gnn' (GCN, single-label), 'mlp' "
                        "(graph-blind Stage-0 baseline; use with --r 0), "
                        "'multilabel_gnn' (BCE + micro-F1, for PPI), "
                        "'binary_gnn' (BCE + AUROC, for RelBench binary entity "
                        "tasks), or 'regression_gnn' (MSE + MAE/RMSE, for "
                        "RelBench REGRESSION entity tasks e.g. rel-f1/"
                        "driver-position, rel-amazon/user-ltv)")
    p.add_argument('--aggr', choices=['mean', 'gcn'], default='mean',
                   help="message-passing aggregator: 'mean' (GraphSAGE) makes "
                        "the rooted-subgraph computation agree EXACTLY with "
                        "full-graph inference; 'gcn' only approximates it, with "
                        "error growing in graph density")
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
    p.add_argument('--common_inductive_split', action='store_true',
                   help='use the saved deterministic 60/20/20 split and delete '
                        'all inter-partition edges before private training')
    p.add_argument('--split_root', default='data/inductive_splits',
                   help='directory holding common saved inductive split indices')
    p.add_argument('--split_seed', type=int, default=0,
                   help='seed identifying the common saved inductive split')
    p.add_argument('--direction', choices=['in', 'out'], default='in',
                   help="SparseExpand orientation: 'in' = Algorithm 5, expand "
                        "along incoming edges so messages flow toward the root "
                        "(correct for message passing; accounted by Theorem "
                        "6.4); 'out' = legacy Algorithm 2/4, kept for the "
                        "orientation ablation (accounted by Theorem 4.5)")
    # Paper parameters (each accepts one or more values → swept as a grid)
    p.add_argument('--p1', type=float, nargs='+', default=None,
                   help='root-sampling probability p1 (Bernoulli per node); '
                        'pass several to sweep, e.g. --p1 0.25 0.5 1.0. '
                        'Prefer --batch_size, which derives p1 = B/pool_size '
                        'and so keeps the expected batch fixed across '
                        'datasets. Default 0.5 if neither is given.')
    p.add_argument('--p2', type=float, nargs='+', default=[0.5],
                   help='edge-sparsification probability p2 (Bernoulli per arc); '
                        'pass several to sweep')
    p.add_argument('--r', type=int, nargs='+', default=[2],
                   help='maximum expansion distance r (SparseExpand levels); '
                        'pass several to sweep, e.g. --r 1 2 3')
    p.add_argument('--T', type=int, default=None,
                   help='number of training steps T. Prefer --epochs, which '
                        'derives T = epochs/p1 so every arm of a p1 sweep sees '
                        'the same amount of data. Default 200 if neither is '
                        'given.')
    # --- schedule in data units, not step units -------------------------------
    # T and p1 are what the accountant prices, but they are the wrong units to
    # THINK in: a fixed T means a different number of passes over the data on
    # every dataset, and a fixed p1 means a different expected batch.  These two
    # express the schedule in units that transfer across datasets, and run.py
    # converts them to (p1, T) once the root pool is known.
    p.add_argument('--batch_size', type=int, default=None,
                   help='expected roots per step B; sets p1 = B/pool_size. '
                        'Mutually exclusive with --p1.')
    p.add_argument('--epochs', type=float, default=None,
                   help='passes over the root pool; sets T = epochs/p1, per '
                        'cell, so a p1 sweep compares equal-data arms rather '
                        'than equal-step ones. Mutually exclusive with --T.')
    p.add_argument('--expect_pool_size', type=int, default=None,
                   help='assert the eligible root pool has exactly this many '
                        'nodes. For drivers that compute p1 = B/N_train in '
                        'shell and pass the result to BOTH this script and the '
                        'noise calibration: if their N drifts from the real '
                        'pool, sigma is solved for the wrong p1 and the '
                        'reported epsilon is wrong. This turns that into a '
                        'crash.')
    # Model / optimization
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2, help='GCN layers L')
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
    p.add_argument('--roots_from', choices=['train', 'all'], default='train',
                   help="eligible-root pool: 'train' (labeled roots only) or 'all'")
    # DP (off by default)
    p.add_argument('--dp', action='store_true', help='enable DP clip+noise path')
    p.add_argument('--clip', type=float, default=1.0, help='clipping norm C (DP)')
    noise_selection = p.add_mutually_exclusive_group()
    noise_selection.add_argument(
        '--sigma', type=float, nargs='+',
        help='noise multiplier(s); pass several to sweep, e.g. --sigma 2 5 10')
    noise_selection.add_argument(
        '--target_epsilon', type=float,
        help='calibrate one noise multiplier per (p1, p2, r) configuration')
    p.add_argument('--target_delta', type=float,
                   help='target delta required with --target_epsilon')
    p.add_argument('--accounting_theorem',
                   choices=['auto', 'substitution', 'thm45'], default='auto',
                   help='SparseGNN dominating-pair theorem used for calibration')
    p.add_argument('--no_vectorized', action='store_true',
                   help='force the per-root Python loop for the DP gradient '
                        'instead of the ghost-clipped batched path. Same '
                        'mechanism and same epsilon -- only the arithmetic '
                        'differs, and the two agree to ~1e-7 -- but ~8x '
                        'slower. For A/B checks and for mechanisms the fast '
                        'path declines (aggr=gcn).')
    p.add_argument('--legacy_shells', action='store_true',
                   help='drop the union-graph correction in the in-process '
                        'calibration (n_d = K^d instead of 2*K^d)')
    p.add_argument('--accounting_grid', type=float, default=1e-4,
                   help='dp_accounting value discretization interval')
    p.add_argument('--calibration_rtol', type=float, default=1e-3,
                   help='relative tolerance for calibrated noise multiplier')
    p.add_argument('--calibration_atol', type=float, default=1e-6,
                   help='absolute tolerance for calibrated noise multiplier')
    p.add_argument('--K_in', type=int, default=None,
                   help='cap max in-degree before training (required for a '
                        'valid Theorem 6.4 guarantee; recorded in the CSV for '
                        'post-hoc accounting via src.sparse.compute_epsilon)')
    p.add_argument('--K_out', type=int, default=None,
                   help='cap max out-degree before training (defaults to K_in)')
    p.add_argument('--cap_mode', choices=['auto', 'directed', 'undirected'],
                   default='auto',
                   help="degree capping: 'auto' (default) = 'directed' for "
                        "every graph -- an undirected graph is treated as a "
                        "directed arc set, so dropping an arc does not drop "
                        "its reverse, and in/out degree are capped "
                        "independently (exactly the two bounds the accounting "
                        "assumes); 'undirected' caps the undirected degree at "
                        "K_in (=K_out) and keeps both arcs of every surviving "
                        "edge, which preserves symmetry but is not required")
    p.add_argument('--cap_seed', type=int, default=None,
                   help='RNG seed for the degree cap.  Default (unset) uses the '
                        'run seed, so each --seed trains on its own capped '
                        'graph and the reported spread includes the cap as a '
                        'source of variance.  Pin it to an int to hold the '
                        'graph fixed across seeds (isolating model/sampling/'
                        'noise variance instead).')
    p.add_argument('--eval_graph', choices=['auto', 'full', 'train'],
                   default='auto',
                   help="graph for `evaluate`.  'auto' (default) = 'full' "
                        "always: no training-side preprocessing (degree cap, "
                        "dedup, split filter) is applied to the graph a result "
                        "is measured on.  'train' reproduces the older "
                        "transductive policy of scoring on the capped graph.  "
                        "'full' = data.edge_index (uncapped, unfiltered; for "
                        "RelBench the test-cutoff graph); 'train' = the exact "
                        "training graph (inductive-filtered, deduplicated, "
                        "capped).  The other graph's metrics are always "
                        "recorded alongside under the *_alt columns.")
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
    p.add_argument('--progress_every', type=int,
                   help='verbose progress/evaluation interval; defaults to '
                        '--eval_every and has no effect without --verbose')
    p.add_argument('--eval_every', type=int, default=50)
    args = p.parse_args()
    if args.p1 is not None and args.batch_size is not None:
        p.error("--p1 and --batch_size both set p1; pass one")
    if args.T is not None and args.epochs is not None:
        p.error("--T and --epochs both set the step count; pass one")
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
    if args.eval_graph == 'auto':
        # ALWAYS the unprocessed full graph, transductive or inductive.  No
        # preprocessing we apply for training's benefit — degree capping,
        # deduplication, the inductive split filter — may touch the graph a
        # result is measured on.
        #
        # This reverses an earlier policy under which a transductive run scored
        # on the CAPPED training graph, on the argument that the model never saw
        # a node of degree > K so the uncapped graph is off-distribution.  That
        # argument is real, but it buys in-distribution evaluation by
        # preprocessing the test set, which is not a protocol we can defend --
        # and it contradicted what both README.md and the paper already claimed.
        # The capped-graph number is still recorded, as the `*_alt` columns.
        args.eval_graph = 'full'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_dp' if args.dp else ''
    # relbench:<db>/<task> names contain separators that are not filename-safe.
    ds_slug = args.dataset.replace(':', '_').replace('/', '_')
    csv_path = os.path.join(args.out_dir,
                            f'sparse_gnn_{ds_slug}{tag}_results.csv')


    dataset, data = load_dataset(
        args.dataset, device=str(device),
        root=args.relbench_root, reverse_edges=args.relbench_reverse_edges,
    ) if str(args.dataset).startswith('relbench') else load_dataset(
        args.dataset, device=str(device))
    data = data.to(device)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    if args.common_inductive_split:
        from src.experiments.inductive import load_or_create_inductive_split
        split = load_or_create_inductive_split(
            data.clone().cpu(), args.dataset, root=args.split_root,
            seed=args.split_seed)
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

    # ---- schedule: resolve (batch_size, epochs) into the (p1, T) the
    # accountant actually prices.  This has to happen HERE, after the mask
    # handling above may have replaced data.train_mask, because the pool size
    # is what p1 is measured against.
    pool_size = (int(data.train_mask.sum()) if args.roots_from == 'train'
                 else int(data.num_nodes))
    if pool_size == 0:
        raise SystemExit(
            f"{args.dataset} has an empty root pool under "
            f"--roots_from {args.roots_from}; nothing to train on")

    if args.expect_pool_size is not None and pool_size != args.expect_pool_size:
        raise SystemExit(
            f"root pool is {pool_size} but --expect_pool_size says "
            f"{args.expect_pool_size}. Whatever derived p1 from the expected "
            f"value computed the wrong sampling rate; if a noise multiplier "
            f"was calibrated against it, its epsilon does not hold. Fix the "
            f"caller's N_train rather than this flag.")

    if args.batch_size is not None:
        try:
            args.p1 = [p1_for_batch(args.batch_size, pool_size)]
        except ValueError as e:
            raise SystemExit(str(e))
    elif args.p1 is None:
        args.p1 = [0.5]

    # T is per-cell, not global: with --epochs and a swept p1 each arm gets the
    # step count that puts it at the SAME number of passes over the data.  A
    # fixed T across a p1 sweep silently compares arms at different points on
    # their learning curves, which is how a convergence-rate difference gets
    # read as a utility difference.
    _T_fixed = None if args.epochs is not None else (
        200 if args.T is None else args.T)

    def steps_of(p1):
        """Step count for one cell: fixed T, or whatever reaches --epochs."""
        return (steps_for_epochs(args.epochs, p1) if _T_fixed is None
                else _T_fixed)

    target_mode = args.target_epsilon is not None
    sigmas = args.sigma if args.dp else [args.sigma[0]]
    grid = list(itertools.product(args.p1, args.p2, args.r))
    if not target_mode:
        grid = [(*cell, sigma) for cell in grid for sigma in sigmas]
    noise_description = (
        f"target_epsilon={args.target_epsilon} target_delta={args.target_delta}"
        if target_mode else f"sigma={sigmas}")

    _T_all = sorted({steps_of(x) for x in args.p1})
    _T_desc = (str(_T_all[0]) if len(_T_all) == 1
               else f"{_T_all[0]}-{_T_all[-1]} (per p1)")
    print(f"\n{'='*66}")
    print(f"SparseGNN  dataset={args.dataset}  device={device}  "
          f"direction={args.direction}  aggr={args.aggr}")
    print(f"  p1={args.p1}  p2={args.p2}  r={args.r}  {noise_description}  "
          f"T={_T_desc}  L={args.num_layers}  dp={args.dp}  seeds={args.seeds}")
    print(f"  root pool={pool_size:,} ({args.roots_from})  "
          f"expected batch={args.p1[0]*pool_size:.1f}"
          + (f"  epochs={args.epochs:g}" if args.epochs is not None else "")
          + (f"  [p1 from --batch_size {args.batch_size}]"
             if args.batch_size is not None else ""))
    print(f"  inductive={args.inductive}  eval_graph={args.eval_graph}  "
          f"roots_from={args.roots_from}")
    if args.dp:
        print(f"  dp gradient path: "
              f"{'per-root loop (--no_vectorized)' if args.no_vectorized else 'vectorized'}")
    print(f"  sweep: {len(grid)} configuration(s) x {args.seeds} seed(s)")
    print('='*66)

    # Model/task guard: fail fast on pairings that would crash deep in a shape
    # error (single-label GNN on multilabel PPI) or silently report a
    # misleading metric (accuracy on an imbalanced binary RelBench task).
    # `mlp` is allowed: it handles multilabel too, and it is the ONLY genuinely
    # graph-blind arm.  Excluding it here is what forced the drivers to use
    # `--model multilabel_gnn --r 0` for the blind arm on PPI/Yelp/Amazon, which
    # is not blind (its untrained neighbour weight is still used at evaluation).
    if (getattr(dataset, 'multilabel', False)
            and args.model not in ('multilabel_gnn', 'mlp')):
        raise SystemExit(
            f"{args.dataset} is multilabel — use --model multilabel_gnn, or "
            f"--model mlp for the graph-blind arm (got --model {args.model})")
    task_type = str(getattr(dataset, 'task_type', ''))
    if 'REGRESSION' in task_type.upper() and args.model != 'regression_gnn':
        raise SystemExit(
            f"{args.dataset} is a regression task ({task_type}) — use "
            f"--model regression_gnn (got --model {args.model}); every "
            f"other mechanism expects integer class labels and will crash "
            f"on this task's float targets")
    # The graph-blind arm must be `--model mlp`, not a GNN mechanism at --r 0.
    # At r=0 a rooted subgraph has no edges, so SAGEConv.lin_l receives exactly
    # zero gradient and never trains -- but evaluate() still runs a full forward
    # over a REAL graph, multiplying real neighbour means by those weights.
    # Under --dp lin_l becomes a pure Gaussian random walk.  Measured on
    # rel-hm/user-churn the arm scores 0.509 AUROC on one eval graph and 0.602
    # on the other; a genuinely blind model would be identical on both.
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
    if 0 in args.r and args.model != 'mlp':
        print(f"  WARNING: --r 0 with --model {args.model} is NOT graph-blind: "
              f"its neighbour weights never train but are still used at "
              f"evaluation. Use --model mlp for the blind arm.")
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
    n_nodes = int(data.num_nodes)
    K_in_req = args.K_in
    K_out_req = args.K_out if args.K_out is not None else args.K_in
    raw_train_ei = edge_index

    def _simplify_and_cap(ei, label, cap_seed):
        """Deduplicate arcs, then enforce the degree bound."""
        n_raw = ei.size(1)
        ei = dedup_arcs(ei, n_nodes)
        if ei.size(1) < n_raw and label:
            print(f"  removed {n_raw - ei.size(1)} parallel arc(s): "
                  f"{n_raw} -> {ei.size(1)} (simple-graph assumption)")
        if K_in_req is None:
            # `--K_out N` alone used to land here and silently apply NO cap at
            # all, because K_out_req is only consulted below.  Fail instead.
            if args.K_out is not None:
                raise SystemExit(
                    "--K_out requires --K_in: capping is driven by K_in here, "
                    "so --K_out on its own silently applies no cap at all. "
                    f"Pass --K_in (e.g. --K_in {args.K_out} --K_out {args.K_out}).")
            return ei, ''
        before = max_degrees(ei, n_nodes)
        # Offset from the root-sampling stream, which is seeded with the same
        # integer (sparse_gnn._make_generator(seed), seed == cap_seed by
        # default).  Two fresh Generators from one seed draw the SAME uniforms,
        # so which arcs survived the cap and which nodes are roots at step 1
        # were deterministically coupled.  The amplification argument wants the
        # Bernoulli root draws independent of graph construction.
        cap_gen = torch.Generator().manual_seed(int(cap_seed) + 20_000)
        mode = args.cap_mode
        if mode == 'auto':
            # ALWAYS directed, symmetric input or not.  Every graph is treated
            # as a directed arc set: dropping an arc does not oblige us to drop
            # its reverse.  This is what the accounting actually assumes --
            # Assumption 5.2 bounds in- and out-degree of a directed graph, and
            # cap_degrees enforces exactly those two bounds.
            #
            # This reverses an earlier policy that preferred the undirected
            # variant on a symmetric graph to preserve edge symmetry (capping
            # the two arcs of an edge independently loses the reverse of ~2/3 of
            # survivors at K=5).  Symmetry was never required by the mechanism;
            # preserving it just made in- and out-expansion coincide.  One
            # capping rule for all graphs is simpler to state and to account.
            mode = 'directed'
        if mode == 'undirected':
            if K_in_req != K_out_req:
                raise SystemExit("--cap_mode undirected needs K_in == K_out")
            ei = cap_degrees_undirected(ei, n_nodes, K_in_req, generator=cap_gen)
        else:
            ei = cap_degrees(ei, n_nodes, K_in=K_in_req, K_out=K_out_req,
                             generator=cap_gen)
        if label:
            print(f"  degree cap K_in={K_in_req} K_out={K_out_req} "
                  f"(mode={mode}, cap_seed={cap_seed}) [{label}]: "
                  f"max (in,out) {before} -> {max_degrees(ei, n_nodes)}, "
                  f"edges {n_raw} -> {ei.size(1)}")
        return ei, mode

    def _build_graphs(cap_seed, label):
        """Cap the graph at `cap_seed` and derive everything that depends on it.

        The capped graph is a random draw, so it belongs inside the seed loop:
        holding it fixed makes every seed train on the same graph and hides the
        cap's contribution to run-to-run variance.
        """
        train_ei, mode = _simplify_and_cap(raw_train_ei, label, cap_seed)
        # Reference graph for the second set of metrics: capped but NOT
        # split-filtered.  Filtering would leave held-out nodes with no edges at
        # all (on PPI their mean in-degree drops 29.3 -> 0), so the comparison
        # has to isolate the cap from the inductive filter.
        if args.inductive and K_in_req is not None:
            eval_capped, _ = _simplify_and_cap(data.edge_index, '', cap_seed)
        else:
            eval_capped = train_ei
        k_in, k_out = K_in_req, K_out_req
        if k_in is None and args.dp:
            if label:
                print("  WARNING: --dp without --K_in — the degree-bound "
                      "assumption (Assumption 5.2) is not enforced; post-hoc "
                      "epsilon will use the graph's raw max degrees.")
            k_in, k_out = max_degrees(train_ei, n_nodes)
        achieved = max_degrees(train_ei, n_nodes)
        return {'train_ei': train_ei, 'eval_capped': eval_capped,
                'cap_mode': mode, 'K_in': k_in, 'K_out': k_out,
                'cap_seed': cap_seed, 'achieved': achieved,
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

    candidate_nodes = None
    if args.roots_from == 'train':
        candidate_nodes = torch.where(data.train_mask)[0]

    _report_subgraph_size(adj, candidate_nodes, int(data.num_nodes),
                          p2=max(args.p2), r=max(args.r),
                          direction=args.direction)

    _probe = _MECHANISMS[args.model]
    _metric = getattr(_probe, 'metric_name', 'accuracy')
    # MLPMechanism picks its metric from the target shape at CONSTRUCTION
    # (micro_f1 on a multilabel dataset), which this class-level probe cannot
    # see — without this, trivial_baseline would take the accuracy branch and
    # call bincount on a float multi-hot target.
    if args.model == 'mlp' and getattr(dataset, 'multilabel', False):
        _metric = 'micro_f1'
    trivial = trivial_baseline(data, _metric)
    _better_high = _HIGHER_IS_BETTER.get(_metric, True)
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
                    'momentum', 'T', 'L', 'dp',
                    # Schedule in data units. Always derived, whether the run
                    # was specified as (p1, T) or (batch_size, epochs), so
                    # every CSV is comparable across datasets and p1 values.
                    'pool_size', 'batch_size', 'epochs',
                    'target_epsilon', 'target_delta', 'calibrated_epsilon',
                    'accounting_theorem', 'accounting_grid',
                    'calibration_rtol', 'calibration_evaluations',
                    'noise_std', 'noise_variance', 'seed', 'step',
                    'roots_from', 'hidden', 'dropout', 'weight_decay', 'seeds',
                    'cap_seed', 'K_in_achieved', 'K_out_achieved',
                    'train_acc', 'val_acc', 'test_acc', 'trivial_baseline',
                    'train_auroc', 'val_auroc', 'test_auroc',
                    # Secondary metrics for regression (RegressionGNNMechanism
                    # reports MAE as the primary train/val/test columns above,
                    # RMSE and R^2 here).  Blank otherwise.  R^2 uses the
                    # evaluated split's own mean as the baseline (RelBench's
                    # and sklearn's convention), not the trivial_baseline
                    # column above, which predicts the TRAIN mean -- R^2=0
                    # means "no better than predicting this split's own mean".
                    'train_rmse', 'val_rmse', 'test_rmse',
                    'train_r2', 'val_r2', 'test_r2',
                    # Secondary metric for binary_gnn (metric_name="auroc" is
                    # primary, above): plain accuracy, meaningful only next to
                    # AUROC on an imbalanced split -- see binary_mechanism.py.
                    'train_bin_acc', 'val_bin_acc', 'test_bin_acc',
                    # Same metrics on the OTHER graph: the training graph when
                    # eval_graph=full, the full graph when eval_graph=train.
                    # They differ by the degree cap (and, for inductive runs,
                    # the split filter), so both are recorded.
                    'train_acc_alt', 'val_acc_alt', 'test_acc_alt',
                    'train_auroc_alt', 'val_auroc_alt', 'test_auroc_alt',
                    # _evaluate() suffixes EVERY key of the alt-graph pass, so
                    # these were already computed and then dropped on the floor.
                    # Without them the "measure the cap gap rather than assume
                    # it" claim held only for the primary metric and AUROC, not
                    # for regression or binary accuracy.
                    'train_rmse_alt', 'val_rmse_alt', 'test_rmse_alt',
                    'train_r2_alt', 'val_r2_alt', 'test_r2_alt',
                    'train_bin_acc_alt', 'val_bin_acc_alt', 'test_bin_acc_alt'])

        for cell in grid:
            calibration = None
            if target_mode:
                p1, p2, r = cell
                T = steps_of(p1)
                calibration = calibrate_sparsegnn_noise(
                    target_epsilon=args.target_epsilon,
                    target_delta=args.target_delta, p1=p1, p2=p2, r=r,
                    K_in=K_in_req, K_out=K_out_req, steps=T, clip=args.clip,
                    direction=args.direction, theorem=args.accounting_theorem,
                    grid=args.accounting_grid,
                    sigma_rtol=args.calibration_rtol,
                    sigma_atol=args.calibration_atol,
                    union_safe=not args.legacy_shells,
                )
                sigma = calibration.noise_multiplier
                print(f"\n[p1={p1} p2={p2} r={r} T={T}]")
                print("  calibrated "
                      f"target=(epsilon={calibration.target_epsilon:g}, "
                      f"delta={calibration.delta:g}) "
                      f"epsilon={calibration.epsilon:.6g} "
                      f"sigma={calibration.noise_multiplier:.6g} "
                      f"theorem={calibration.theorem} "
                      f"evaluations={calibration.evaluations}")
            else:
                p1, p2, r, sigma = cell
                T = steps_of(p1)
                print(f"\n[p1={p1} p2={p2} r={r} T={T}" +
                      (f" sigma={sigma}]" if args.dp else "]"))
            calibration_fields = (
                [calibration.target_epsilon, calibration.delta,
                 calibration.epsilon, calibration.theorem,
                 args.accounting_grid, args.calibration_rtol,
                 calibration.evaluations]
                if calibration is not None else [""] * 7)
            noise_fields = (
                [sigma * args.clip, (sigma * args.clip) ** 2]
                if args.dp else ["", ""])
            tests, vals = [], []
            for seed in range(args.seeds):
                _set_seed(seed)
                gph = graphs[seed]
                Mechanism = _MECHANISMS[args.model]
                extra = {} if args.model == 'mlp' else {'aggr': args.aggr}
                mech = Mechanism(
                    data, num_features, num_classes,
                    hidden=args.hidden, num_layers=args.num_layers,
                    dropout=args.dropout, device=device, **extra,
                )
                if args.eval_graph == 'train':
                    mech.eval_edge_index = gph['train_ei'].to(device)
                    alt_ei = data.edge_index                  # uncapped
                else:
                    alt_ei = gph['eval_capped'].to(device)    # capped
                # Adam everywhere, DP or not.  Three reasons:
                #   1. Every baseline we compare against is Adam -- DPAR is
                #      literally DPAdamGaussianOptimizer upstream, ProGAP
                #      defaults to it, HeterPoisson uses it -- so a same-
                #      optimizer comparison is the defensible one.
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

                progress_every = (args.progress_every
                                  if args.verbose and args.progress_every is not None
                                  else args.eval_every)
                accs = train_sparse_gnn(
                    mech, data, adj=gph['adj'], direction=args.direction,
                    p1=p1, p2=p2, r=r, T=T,
                    candidate_nodes=candidate_nodes,
                    dp=args.dp, clip=args.clip, sigma=sigma,
                    seed=seed, eval_every=progress_every,
                    vectorized=not args.no_vectorized,
                    track_every=args.track_every, eval_alt_edge_index=alt_ei,
                    verbose=args.verbose,
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
                                gph['K_in'] if gph['K_in'] is not None else '',
                                gph['K_out'] if gph['K_out'] is not None else '',
                                gph['cap_mode'], args.eval_graph,
                                opt_kind, args.lr, args.momentum,
                                T, args.num_layers, args.dp,
                                pool_size, f"{p1 * pool_size:.1f}",
                                f"{T * p1:.4f}",
                                *calibration_fields, *noise_fields, seed, step,
                                args.roots_from, args.hidden, args.dropout,
                                args.weight_decay, args.seeds,
                                gph['cap_seed'], gph['achieved'][0],
                                gph['achieved'][1],
                                f"{m['train']:.5f}", f"{m['val']:.5f}",
                                f"{m['test']:.5f}", f"{trivial:.5f}",
                                *(f"{m[k]:.5f}" if k in m else ''
                                  for k in ('train_auroc', 'val_auroc',
                                            'test_auroc',
                                            'train_rmse', 'val_rmse',
                                            'test_rmse',
                                            'train_r2', 'val_r2', 'test_r2',
                                            'train_bin_acc', 'val_bin_acc',
                                            'test_bin_acc',
                                            'train_alt', 'val_alt', 'test_alt',
                                            'train_auroc_alt', 'val_auroc_alt',
                                            'test_auroc_alt',
                                            'train_rmse_alt', 'val_rmse_alt',
                                            'test_rmse_alt',
                                            'train_r2_alt', 'val_r2_alt',
                                            'test_r2_alt',
                                            'train_bin_acc_alt',
                                            'val_bin_acc_alt',
                                            'test_bin_acc_alt'))])

                for h in history:
                    if h['step'] < T:        # final checkpoint == the T row
                        _write_row(h['step'], h)
                _write_row(T, accs)
                fh.flush()   # persist each row so a killed run keeps its rows

            tm, ts = _mean_std(tests)
            vm, vs = _mean_std(vals)
            summary.append((p1, p2, r, sigma, tm, ts, vm, vs))
            beats_trivial = (tm > trivial) if _better_high else (tm < trivial)
            mark = "" if beats_trivial else "   <-- BELOW TRIVIAL BASELINE"
            print(f"  >> test {tm:.4f} +/- {ts:.4f}   "
                  f"val {vm:.4f} +/- {vs:.4f}{mark}")

    os.replace(partial_path, csv_path)

    # Sweep summary table (sorted by test metric, best first)
    print(f"\n{'='*66}")
    print(f"{'p1':>5} {'p2':>5} {'r':>3} {'sigma':>6} {'test':>16} {'val':>16}")
    print('-'*66)
    _sort_sign = -1 if _better_high else 1
    for p1, p2, r, sigma, tm, ts, vm, vs in sorted(
            summary, key=lambda s: _sort_sign * s[4]):
        print(f"{p1:>5} {p2:>5} {r:>3} {sigma:>6}   {tm:.4f} +/- {ts:.4f}   "
              f"{vm:.4f} +/- {vs:.4f}")
    print(f"\nresults written to {csv_path}")
    if args.dp:
        print("compute epsilon post-hoc with:  python -m src.sparse.compute_epsilon "
              f"--csv {csv_path} --delta <delta> --grid {args.accounting_grid:g}")

    if args.plot:
        plot_path = plot_sweep([(s[0], s[1], s[2], s[4], s[5], s[6], s[7])
                                for s in summary if s[3] == sigmas[0]],
                               args.dataset, args.out_dir)
        if plot_path:
            print(f"plot written to {plot_path}")


if __name__ == '__main__':
    main()
