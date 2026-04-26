"""
Unified experiment runner.

All experiment types (benchmark, noise sweep, single run, ogbn) are controlled
via a single modular CLI. Every knob is a flag.

Usage examples:
    # Single run: Algo 3 on Cora with Poisson subsampling
    python run.py --dataset cora --algo 3 --num-bins 4 --poisson --q-epoch 0.1 --q-step 0.3

    # Benchmark: sweep algos and bins on cora + pubmed (5 seeds)
    python run.py --dataset cora pubmed --algo 1 2 3 --num-bins 2 4 8 16 --seeds 5

    # Noise sweep: accuracy vs epsilon
    python run.py --dataset cora --algo 2 3 --num-bins 4 8 --dp --noise-multiplier 0.5 1.0 2.0 4.0

    # DP with fixed epsilon
    python run.py --dataset pubmed --algo 3 --num-bins 8 --dp --epsilon 1.0

    # ogbn-products with Poisson subsampling + convergence
    python run.py --dataset ogbn-products --algo 2 3 --num-bins 4 8 16 32 \
        --poisson --q-epoch 0.1 --q-step 0.3 --converge --seeds 3

    # Baseline only
    python run.py --dataset cora --baseline gcn mlp

    # Everything: baselines + algos + Poisson variant
    python run.py --dataset cora pubmed --baseline gcn mlp --algo 1 2 3 \
        --num-bins 2 4 8 --poisson --q-epoch 0.1 --q-step 0.3 --seeds 5
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.datasets import load_dataset
from src.models import make_model
from src.algorithms import get_algorithm
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.subgraph_trainer import SubgraphTrainer
from src.trainers.link_pred_trainer import LinkPredTrainer


# ── Default hyperparameters ─────────────────────────────────────────────────

ALL_SEEDS = [42, 123, 456, 789, 1024]

DEFAULTS = dict(
    lr=0.01,
    weight_decay=5e-4,
    hidden_channels=64,
    dropout=0.5,
    steps_per_epoch=10,
    subsample_prob=0.3,
    clip_norm=1.0,
    dp_delta=1e-5,
    batch_size=1024,
    num_neighbors=[15, 10],
)


# ── Argument parser ─────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Unified experiment runner for subgraph DP-GCN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── What to run ──────────────────────────────────────────────────────
    g = p.add_argument_group("experiment setup")
    g.add_argument('--dataset', nargs='+', default=['cora'],
                   choices=['cora', 'citeseer', 'pubmed',
                            'ogbn-products', 'ogbn-arxiv', 'reddit',
                            'ogbl-collab'],
                   help='Dataset(s) to run on (default: cora)')
    g.add_argument('--algo', type=int, nargs='+', default=None,
                   choices=[1, 2, 3],
                   help='Algorithm(s) to run (default: none, use --baseline)')
    g.add_argument('--model', default='gcn', choices=['gcn', 'mlp'],
                   help='Model architecture for node-classification tasks (default: gcn). '
                        'Link-prediction tasks always use the link-pred GCN encoder.')
    g.add_argument('--task', default='node', choices=['node', 'link'],
                   help='Supervision task: node classification (default) or link prediction. '
                        '--task link requires a linkprop dataset (e.g. ogbl-collab).')
    g.add_argument('--num-bins', type=int, nargs='+', default=[4, 8],
                   help='Bin count(s) to sweep (default: 4 8)')
    g.add_argument('--baseline', nargs='*', default=None,
                   choices=['gcn', 'mlp'],
                   help='Run baseline(s). No args = gcn. e.g. --baseline gcn mlp')
    g.add_argument('--seeds', type=int, default=1,
                   help='Number of seeds to use (1-5, default: 1)')

    # ── Training ─────────────────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument('--epochs', type=int, default=200,
                   help='Number of training epochs (default: 200)')
    g.add_argument('--lr', type=float, default=DEFAULTS['lr'],
                   help=f'Learning rate (default: {DEFAULTS["lr"]})')
    g.add_argument('--weight-decay', type=float, default=DEFAULTS['weight_decay'],
                   help=f'Weight decay (default: {DEFAULTS["weight_decay"]})')
    g.add_argument('--hidden-channels', type=int, default=DEFAULTS['hidden_channels'],
                   help=f'Hidden layer size (default: {DEFAULTS["hidden_channels"]})')
    g.add_argument('--dropout', type=float, default=DEFAULTS['dropout'],
                   help=f'Dropout rate (default: {DEFAULTS["dropout"]})')

    # ── Convergence / early stopping ─────────────────────────────────────
    g = p.add_argument_group("convergence")
    g.add_argument('--converge', action='store_true',
                   help='Enable early stopping on train loss')
    g.add_argument('--max-epochs', type=int, default=500,
                   help='Max epochs in convergence mode (default: 500)')
    g.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (default: 20)')
    g.add_argument('--es-delta', type=float, default=1e-4,
                   help='Min loss improvement for early stopping (default: 1e-4)')

    # ── Subsampling ──────────────────────────────────────────────────────
    g = p.add_argument_group("subsampling")
    g.add_argument('--subsample-prob', type=float, default=DEFAULTS['subsample_prob'],
                   help=f'Algo 3 dummy-bin drop probability (default: {DEFAULTS["subsample_prob"]})')
    g.add_argument('--epoch-assignment', action='store_true',
                   help='Deterministic chunking of train nodes into steps')
    g.add_argument('--poisson', action='store_true',
                   help='Two-phase Poisson subsampling of train nodes')
    g.add_argument('--single-phase', action='store_true',
                   help='Single-phase Poisson: fresh independent Bernoulli(q) per step '
                        '(matches standard DP-SGD accounting)')
    g.add_argument('--q', type=float, default=1.0,
                   help='Single-phase Poisson per-step inclusion prob (default: 1.0)')
    g.add_argument('--q-epoch', type=float, default=1.0,
                   help='Poisson epoch-level inclusion prob (default: 1.0)')
    g.add_argument('--q-step', type=float, default=1.0,
                   help='Poisson step-level inclusion prob (default: 1.0)')
    g.add_argument('--steps-per-epoch', type=int, default=DEFAULTS['steps_per_epoch'],
                   help=f'Steps per epoch for epoch-assignment/poisson (default: {DEFAULTS["steps_per_epoch"]})')
    g.add_argument('--coverage', action='store_true',
                   help='Enable coverage correction')

    # ── Differential privacy ─────────────────────────────────────────────
    g = p.add_argument_group("differential privacy")
    g.add_argument('--dp', action='store_true',
                   help='Enable DP (gradient clipping + noise)')
    g.add_argument('--epsilon', type=float, nargs='+', default=None,
                   help='Privacy budget(s). Multiple values = sweep.')
    g.add_argument('--noise-multiplier', type=float, nargs='+', default=None,
                   help='Noise multiplier(s) sigma/C. Multiple values = sweep.')
    g.add_argument('--clip-norm', type=float, default=DEFAULTS['clip_norm'],
                   help=f'Max gradient norm for clipping (default: {DEFAULTS["clip_norm"]})')
    g.add_argument('--dp-delta', type=float, default=DEFAULTS['dp_delta'],
                   help=f'Privacy parameter delta (default: {DEFAULTS["dp_delta"]})')
    g.add_argument('--accountant', choices=['off', 'rdp', 'prv'], default='off',
                   help='Privacy accountant for computing epsilon: off (skip), '
                        'rdp (default Opacus), or prv (PLD-based, tighter)')
    g.add_argument('--max-in-degree', type=int, default=None,
                   help='Cap per-node in-degree by random subsampling (preprocessing). '
                        'Bounds node-DP sensitivity on high-degree graphs.')

    # ── ogbn-products specific ───────────────────────────────────────────
    g = p.add_argument_group("ogbn-products (large graph)")
    g.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'],
                   help=f'NeighborLoader batch size for baselines (default: {DEFAULTS["batch_size"]})')
    g.add_argument('--num-neighbors', type=int, nargs='+', default=DEFAULTS['num_neighbors'],
                   help=f'Per-layer neighbor sampling (default: {DEFAULTS["num_neighbors"]})')

    # ── Output ───────────────────────────────────────────────────────────
    g = p.add_argument_group("output")
    g.add_argument('--no-plot', action='store_true',
                   help='Skip plotting')
    g.add_argument('--output-dir', default='results',
                   help='Output directory (default: results)')
    g.add_argument('--tag', default=None,
                   help='Optional tag for output filenames')

    return p


def validate_args(args):
    """Validate argument combinations and set defaults."""
    if args.algo is None and args.baseline is None:
        raise SystemExit("Error: specify --algo and/or --baseline")

    if args.dp:
        if args.epsilon is not None and args.noise_multiplier is not None:
            raise SystemExit("Error: specify --epsilon or --noise-multiplier, not both")
        if args.epsilon is None and args.noise_multiplier is None:
            raise SystemExit("Error: --dp requires --epsilon or --noise-multiplier")

    if sum([args.poisson, args.epoch_assignment, args.single_phase]) > 1:
        raise SystemExit(
            "Error: --poisson, --epoch-assignment, --single-phase are mutually exclusive"
        )

    link_datasets = {'ogbl-collab'}
    is_link_dataset = lambda d: d in link_datasets
    if args.task == 'link':
        if any(not is_link_dataset(d) for d in args.dataset):
            raise SystemExit(
                f"Error: --task link requires a linkprop dataset; got {args.dataset}. "
                f"Supported: {sorted(link_datasets)}"
            )
        if args.baseline:
            raise SystemExit("Error: --baseline is not supported for --task link yet")
    else:
        if any(is_link_dataset(d) for d in args.dataset):
            raise SystemExit(
                f"Error: link-prediction dataset(s) {[d for d in args.dataset if is_link_dataset(d)]} "
                f"require --task link"
            )

    # --baseline with no args defaults to ['gcn']
    if args.baseline is not None and len(args.baseline) == 0:
        args.baseline = ['gcn']

    args.seed_list = ALL_SEEDS[:args.seeds]

    return args


# ── Training loop ───────────────────────────────────────────────────────────

def train_loop(trainer, data, args):
    """Train with fixed epochs or early stopping. Returns (actual_epochs, losses)."""
    if not args.converge:
        for _ in range(args.epochs):
            trainer.train_epoch(data)
        return args.epochs

    best_loss = float('inf')
    wait = 0
    epoch = 0
    while epoch < args.max_epochs:
        losses = trainer.train_epoch(data)
        epoch += 1
        if isinstance(losses, list):
            epoch_loss = sum(losses) / len(losses) if losses else float('inf')
        else:
            epoch_loss = losses
        if best_loss - epoch_loss > args.es_delta:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    return epoch


# ── Baseline runner ─────────────────────────────────────────────────────────

def run_baseline(dataset, data, device, args, *, model_type, seed):
    """Run baseline training (full-graph or NeighborLoader for large graphs)."""
    torch.manual_seed(seed)
    model = make_model(dataset, model_type=model_type,
                       hidden_channels=args.hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    is_large = data.num_nodes > 100_000

    if is_large and model_type == 'gcn':
        from torch_geometric.loader import NeighborLoader
        train_loader = NeighborLoader(
            data,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
        )
        actual_epochs = _train_baseline_minibatch(
            model, optimizer, train_loader, device, args)
    else:
        trainer = BaselineTrainer(model, optimizer, device=device)
        actual_epochs = train_loop(trainer, data, args)

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return dict(
        method=model_type.upper(),
        model=model_type,
        algorithm=None,
        num_bins=None,
        seed=seed,
        train_acc=round(train_acc, 6),
        test_acc=round(test_acc, 6),
        epochs=actual_epochs,
    )


def _train_baseline_minibatch(model, optimizer, train_loader, device, args):
    """Mini-batch training loop for large graphs using NeighborLoader."""
    def _one_epoch():
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            train_mask = batch.train_mask[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            out = out[:batch.batch_size]
            if not train_mask.any():
                continue
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            n = train_mask.sum().item()
            total_loss += loss.item() * n
            total_nodes += n
        return total_loss / max(total_nodes, 1)

    if not args.converge:
        for _ in range(args.epochs):
            _one_epoch()
        return args.epochs

    best_loss = float('inf')
    wait = 0
    epoch = 0
    while epoch < args.max_epochs:
        epoch_loss = _one_epoch()
        epoch += 1
        if best_loss - epoch_loss > args.es_delta:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    return epoch


# ── Subgraph runner ─────────────────────────────────────────────────────────

def run_subgraph(dataset, data, device, args, *, algo_id, num_bins, seed,
                 noise_multiplier=None, epsilon=None):
    """Run one subgraph training configuration."""
    torch.manual_seed(seed)

    algo_kwargs = {}
    if algo_id == 3:
        algo_kwargs['subsample_prob'] = args.subsample_prob
    algorithm = get_algorithm(algo_id, **algo_kwargs)

    model_type = 'link_pred_gcn' if args.task == 'link' else args.model
    model = make_model(dataset, model_type=model_type,
                       hidden_channels=args.hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    trainer_cls = LinkPredTrainer if args.task == 'link' else SubgraphTrainer
    trainer = trainer_cls(
        model, optimizer,
        num_bins=num_bins,
        algorithm=algorithm,
        use_coverage_correction=args.coverage,
        use_epoch_assignment=args.epoch_assignment,
        poisson_subsampling=args.poisson,
        single_phase_poisson=args.single_phase,
        q=args.q,
        q_epoch=args.q_epoch,
        q_step=args.q_step,
        steps_per_epoch=args.steps_per_epoch,
        device=device,
        dp=args.dp,
        max_grad_norm=args.clip_norm,
        epsilon=epsilon,
        noise_multiplier=noise_multiplier,
        delta=args.dp_delta,
    )

    actual_epochs = train_loop(trainer, data, args)
    train_acc, test_acc = trainer.evaluate(data)

    # Build result row — always include all fields for clarity
    result = dict(
        # ── identity ──
        method=f'Algo {algo_id}',
        task=args.task,
        model=model_type,
        algorithm=algo_id,
        num_bins=num_bins,
        seed=seed,
        epochs=actual_epochs,

        # ── accuracy ──
        train_acc=round(train_acc, 6),
        test_acc=round(test_acc, 6),

        # ── subsampling config ──
        subsample_prob=args.subsample_prob if algo_id == 3 else 0.0,
        poisson=args.poisson,
        q_epoch=args.q_epoch if args.poisson else None,
        q_step=args.q_step if args.poisson else None,
        epoch_assignment=args.epoch_assignment,
        steps_per_epoch=args.steps_per_epoch if (args.poisson or args.epoch_assignment) else None,
        coverage=args.coverage,

        # ── DP config ──
        dp=args.dp,
        noise_multiplier=noise_multiplier,
        epsilon=epsilon,
        clip_norm=args.clip_norm if args.dp else None,
        dp_delta=args.dp_delta if args.dp else None,
        training_steps=trainer.training_steps if args.dp else None,
    )

    # Optionally compute epsilon via Opacus accountant (RDP or PRV)
    if args.dp and args.accountant != 'off' and noise_multiplier is not None:
        # Node-level sampling factor: q for single-phase, 1 otherwise.
        # (Two-phase poisson uses correlated draws within an epoch, so
        # plugging q_epoch*q_step here would not be tight — left as 1 and
        # flagged as known-loose.)
        q_factor = args.q if args.single_phase else 1.0
        if algo_id == 3:
            sample_rate = q_factor * (1 - args.subsample_prob) / num_bins
        else:
            sample_rate = q_factor / num_bins
        result['sample_rate'] = sample_rate
        result['accountant'] = args.accountant
        try:
            from src.privacy_accountant import compute_epsilon
            result['computed_epsilon'] = round(compute_epsilon(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                num_steps=trainer.training_steps,
                delta=args.dp_delta,
                accountant=args.accountant,
            ), 4)
        except Exception as e:
            result['computed_epsilon'] = None
            result['accountant_error'] = f"{type(e).__name__}: {e}"
            print(f"  [accountant] {args.accountant} failed: "
                  f"{type(e).__name__}: {e}")

    return result


# ── Main experiment loop ────────────────────────────────────────────────────

def run_all(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {args.seed_list}")

    # Summarize config
    if args.baseline:
        print(f"Baselines: {args.baseline}")
    if args.algo:
        print(f"Algorithms: {args.algo} | Bins: {args.num_bins}")
    if args.dp:
        dp_vals = args.epsilon or args.noise_multiplier
        dp_type = "epsilon" if args.epsilon else "noise_multiplier"
        print(f"DP: {dp_type}={dp_vals} | clip={args.clip_norm} | delta={args.dp_delta}")
    if args.poisson:
        print(f"Poisson (two-phase): q_epoch={args.q_epoch}, q_step={args.q_step}, "
              f"steps_per_epoch={args.steps_per_epoch}")
    if args.single_phase:
        print(f"Poisson (single-phase): q={args.q}, "
              f"steps_per_epoch={args.steps_per_epoch}")
    if args.epoch_assignment:
        print(f"Epoch assignment: steps_per_epoch={args.steps_per_epoch}")
    if args.coverage:
        print("Coverage correction: on")
    if args.converge:
        print(f"Convergence: max_epochs={args.max_epochs}, patience={args.patience}")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    ds_str = "_".join(args.dataset)
    jsonl_path = os.path.join(args.output_dir, f"run_{ds_str}{tag}_{timestamp}.jsonl")

    all_results = []

    for ds_name in args.dataset:
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        dataset, data = load_dataset(ds_name, device)
        info = (f"  {data.num_nodes} nodes, {data.num_edges} edges, "
                f"{dataset.num_features} features")
        if args.task == 'node':
            info += f", {dataset.num_classes} classes"
        else:
            info += f", {data.train_pos_edge.size(1)} train edges"
        print(info)

        if args.max_in_degree is not None:
            from src.utils import sparsify_by_degree
            # Fixed generator seed: sparsification is a property of the public
            # input graph, so it should be identical across training seeds.
            gen = torch.Generator(device=data.edge_index.device)
            gen.manual_seed(0)
            orig_edges = data.edge_index.size(1)
            data.edge_index = sparsify_by_degree(
                data.edge_index, data.num_nodes, args.max_in_degree,
                generator=gen,
            )
            print(f"  sparsified: {orig_edges} -> {data.edge_index.size(1)} "
                  f"edges (max_in_degree={args.max_in_degree})")

        for seed in args.seed_list:
            # ── Baselines ──
            if args.baseline:
                for model_type in args.baseline:
                    label = f"{model_type.upper()} baseline"
                    print(f"  [seed={seed}] {label} ...", end=' ', flush=True)
                    result = run_baseline(dataset, data, device, args,
                                          model_type=model_type, seed=seed)
                    result['dataset'] = ds_name
                    all_results.append(result)
                    print(f"train={result['train_acc']:.4f}  "
                          f"test={result['test_acc']:.4f}  "
                          f"epochs={result['epochs']}")

            # ── Subgraph algorithms ──
            if args.algo:
                # Build list of (noise_multiplier, epsilon) configs
                if args.dp:
                    if args.noise_multiplier:
                        dp_configs = [(nm, None) for nm in args.noise_multiplier]
                    else:
                        dp_configs = [(None, eps) for eps in args.epsilon]
                else:
                    dp_configs = [(None, None)]

                for algo_id in args.algo:
                    for num_bins in args.num_bins:
                        for noise_mult, eps in dp_configs:
                            label = _make_label(algo_id, num_bins, args,
                                                noise_mult=noise_mult, eps=eps)
                            print(f"  [seed={seed}] {label} ...",
                                  end=' ', flush=True)

                            result = run_subgraph(
                                dataset, data, device, args,
                                algo_id=algo_id, num_bins=num_bins, seed=seed,
                                noise_multiplier=noise_mult, epsilon=eps,
                            )
                            result['dataset'] = ds_name
                            all_results.append(result)

                            out_str = (f"train={result['train_acc']:.4f}  "
                                       f"test={result['test_acc']:.4f}  "
                                       f"epochs={result['epochs']}")
                            if result.get('computed_epsilon') is not None:
                                out_str += f"  eps={result['computed_epsilon']:.2f}"
                            print(out_str)

        # Flush after each dataset
        _save_results(all_results, jsonl_path)

    print(f"\nResults saved to {jsonl_path}")

    if not args.no_plot:
        try:
            plot_results(all_results, timestamp, args)
        except ImportError as e:
            print(f"\nPlotting skipped: {e}")

    return all_results, timestamp


def _make_label(algo_id, num_bins, args, noise_mult=None, eps=None):
    """Build a human-readable label for a run."""
    parts = [f"Algo {algo_id}", f"bins={num_bins}"]
    if algo_id == 3:
        parts.append(f"p={args.subsample_prob}")
    if args.poisson:
        parts.append(f"poisson(qe={args.q_epoch},qs={args.q_step})")
    if args.epoch_assignment:
        parts.append(f"ea(T={args.steps_per_epoch})")
    if args.coverage:
        parts.append("cov")
    if noise_mult is not None:
        parts.append(f"sigma={noise_mult}")
    if eps is not None:
        parts.append(f"eps={eps}")
    return ", ".join(parts)


def _save_results(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(all_results, timestamp, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    has_noise_sweep = any(r.get('noise_multiplier') is not None for r in all_results)

    for ds_name in args.dataset:
        ds_rows = [r for r in all_results if r['dataset'] == ds_name]
        if not ds_rows:
            continue

        if has_noise_sweep:
            _plot_noise_sweep(ds_rows, ds_name, timestamp, args, plt, np)
        else:
            _plot_bins(ds_rows, ds_name, timestamp, args, plt, np)


def _plot_bins(ds_rows, ds_name, timestamp, args, plt, np):
    """Plot: accuracy vs number of bins."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Baselines
    for method in ['MLP', 'GCN']:
        accs = [r['test_acc'] for r in ds_rows if r['method'] == method]
        if not accs:
            continue
        mean, std = np.mean(accs), np.std(accs)
        ax.axhline(mean, linestyle='--', linewidth=1.5,
                    label=f'{method} ({mean:.3f}+/-{std:.3f})')
        ax.axhspan(mean - std, mean + std, alpha=0.08)

    # Algo lines
    markers = {1: 'o', 2: 's', 3: '^'}
    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    algos = sorted(set(r['algorithm'] for r in ds_rows if r['algorithm'] is not None))
    for algo_id in algos:
        algo_rows = [r for r in ds_rows if r['algorithm'] == algo_id]
        bins_list = sorted(set(r['num_bins'] for r in algo_rows))
        means, stds, xs = [], [], []
        for nb in bins_list:
            accs = [r['test_acc'] for r in algo_rows if r['num_bins'] == nb]
            if accs:
                xs.append(nb)
                means.append(np.mean(accs))
                stds.append(np.std(accs))
        label = f'Algo {algo_id}'
        if algo_id == 3:
            label += f' (p={args.subsample_prob})'
        ax.errorbar(xs, np.array(means), yerr=np.array(stds),
                    marker=markers.get(algo_id, 'o'),
                    color=colors.get(algo_id, 'tab:purple'),
                    capsize=4, linewidth=1.5, label=label)

    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'{ds_name} -- Accuracy vs Bins')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    tag = f"_{args.tag}" if args.tag else ""
    save_path = os.path.join(args.output_dir,
                             f"plot_{ds_name}_bins{tag}_{timestamp}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def _plot_noise_sweep(ds_rows, ds_name, timestamp, args, plt, np):
    """Plot: accuracy vs noise multiplier (sigma)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Baselines
    for method in ['MLP', 'GCN']:
        accs = [r['test_acc'] for r in ds_rows if r['method'] == method]
        if not accs:
            continue
        mean, std = np.mean(accs), np.std(accs)
        ax.axhline(mean, linestyle='--', linewidth=1.5,
                    label=f'{method} ({mean:.3f}+/-{std:.3f})')
        ax.axhspan(mean - std, mean + std, alpha=0.08)

    # Group by (algorithm, num_bins)
    algo_colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    algo_markers = {1: 'o', 2: 's', 3: '^'}
    bin_styles = {2: '-', 4: '--', 8: '-.', 16: ':', 32: '-'}

    algos = sorted(set(r['algorithm'] for r in ds_rows
                       if r['algorithm'] is not None and r.get('noise_multiplier') is not None))
    for algo_id in algos:
        algo_rows = [r for r in ds_rows
                     if r['algorithm'] == algo_id and r.get('noise_multiplier') is not None]
        bins_list = sorted(set(r['num_bins'] for r in algo_rows))
        for nb in bins_list:
            bin_rows = [r for r in algo_rows if r['num_bins'] == nb]
            noise_vals = sorted(set(r['noise_multiplier'] for r in bin_rows))
            sigmas, means, stds = [], [], []
            for nv in noise_vals:
                nv_rows = [r for r in bin_rows if r['noise_multiplier'] == nv]
                accs = [r['test_acc'] for r in nv_rows]
                sigmas.append(nv)
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            label = f'Algo {algo_id}, bins={nb}'
            ax.errorbar(sigmas, np.array(means), yerr=np.array(stds),
                        marker=algo_markers.get(algo_id, 'o'),
                        color=algo_colors.get(algo_id, 'tab:purple'),
                        linestyle=bin_styles.get(nb, '-'),
                        capsize=4, linewidth=1.5, label=label, alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel('Noise Multiplier (sigma, log scale)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'{ds_name} -- Accuracy vs Noise Multiplier')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    tag = f"_{args.tag}" if args.tag else ""
    save_path = os.path.join(args.output_dir,
                             f"plot_{ds_name}_noise{tag}_{timestamp}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# ── Print summary table ────────────────────────────────────────────────────

def print_summary(all_results):
    """Print a detailed summary table with all relevant config."""
    import numpy as np

    # Group results by all config that matters (everything except seed and accuracy)
    def group_key(r):
        return (
            r['dataset'],
            r['method'],
            r.get('algorithm'),
            r.get('num_bins'),
            r.get('noise_multiplier'),
            r.get('epsilon'),
            r.get('poisson', False),
            r.get('q_epoch'),
            r.get('q_step'),
            r.get('epoch_assignment', False),
            r.get('coverage', False),
            r.get('subsample_prob', 0.0),
        )

    groups = {}
    for r in all_results:
        groups.setdefault(group_key(r), []).append(r)

    # Check if any results have computed_epsilon (i.e. --accountant was used)
    has_accountant = any(r.get('computed_epsilon') is not None for r in all_results)

    # ── Header ──
    header = (f"{'Dataset':<12} {'Method':<22} {'Bins':>4} {'Subsamp':>8} "
              f"{'Poisson':>14} {'Cov':>3} {'Sigma':>6} "
              f"{'Seeds':>5} {'Train Acc':>12} {'Test Acc':>12}")
    if has_accountant:
        header += f" {'Eps':>8}"
    width = len(header)
    print(f"\n{'='*width}")
    print(header)
    print(f"{'-'*width}")

    for key, rows in groups.items():
        (ds, method, algo, nb, nm, eps, poisson,
         q_e, q_s, ea, cov, sub_p) = key

        n = len(rows)
        train_accs = [r['train_acc'] for r in rows]
        test_accs = [r['test_acc'] for r in rows]

        if n > 1:
            train_str = f"{np.mean(train_accs):.4f}+/-{np.std(train_accs):.4f}"
            test_str = f"{np.mean(test_accs):.4f}+/-{np.std(test_accs):.4f}"
        else:
            train_str = f"{train_accs[0]:.4f}"
            test_str = f"{test_accs[0]:.4f}"

        # Bins column
        bins_str = str(nb) if nb is not None else "-"

        # Subsampling column: algo3 subsample_prob or epoch_assignment
        if sub_p and sub_p > 0:
            sub_str = f"p={sub_p}"
        elif ea:
            sub_str = "epoch"
        else:
            sub_str = "-"

        # Poisson column
        if poisson:
            poi_str = f"qe={q_e},qs={q_s}"
        else:
            poi_str = "-"

        # Coverage
        cov_str = "Y" if cov else "-"

        # Sigma column (noise multiplier)
        if nm is not None:
            sig_str = f"{nm}"
        else:
            sig_str = "-"

        line = (f"{ds:<12} {method:<22} {bins_str:>4} {sub_str:>8} "
                f"{poi_str:>14} {cov_str:>3} {sig_str:>6} "
                f"{n:>5} {train_str:>12} {test_str:>12}")

        if has_accountant:
            computed = [r.get('computed_epsilon') for r in rows
                        if r.get('computed_epsilon') is not None]
            eps_str = f"{np.mean(computed):.2f}" if computed else ""
            line += f" {eps_str:>8}"

        print(line)

    print(f"{'='*width}")

    # ── Legend ──
    print("\nColumn legend:")
    print("  Bins       = number of subgraph partitions")
    print("  Subsamp    = algo3 dummy-bin drop prob (p=X) or epoch-assignment mode")
    print("  Poisson    = two-phase Poisson subsampling (qe=epoch prob, qs=step prob)")
    print("  Cov        = coverage correction (Y/N)")
    print("  Sigma      = noise multiplier (noise_std / clip_norm)")
    if has_accountant:
        print("  Eps        = computed epsilon via RDP accountant (--accountant)")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = validate_args(parser.parse_args())
    all_results, timestamp = run_all(args)
    print_summary(all_results)


if __name__ == '__main__':
    main()
