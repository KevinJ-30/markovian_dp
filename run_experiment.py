"""
CLI entry point for utility experiments.

Usage examples:
    python run_experiment.py --dataset cora --model gcn --num-bins 4 8
    python run_experiment.py --dataset pubmed --model mlp
    python run_experiment.py --dataset cora --algorithm 1 --num-bins 2 4 8
"""

import argparse
import json
import os
from datetime import datetime
import torch
import torch.optim as optim

from src.datasets import load_dataset
from src.models import make_model
from src.algorithms import get_algorithm
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.subgraph_trainer import SubgraphTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Subgraph GCN utility experiments")
    p.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora')
    p.add_argument('--algorithm', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--model', choices=['gcn', 'mlp'], default='gcn')
    p.add_argument('--num-bins', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--hidden-channels', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps-per-epoch', type=int, default=10)
    p.add_argument('--subsample-prob', type=float, default=0.0,
                   help='p_perp for Algorithm 3 (probability of dummy bin assignment)')
    p.add_argument('--coverage', action='store_true')
    p.add_argument('--epoch-assignment', action='store_true')
    p.add_argument('--baseline', action='store_true', help='Run baseline (no subgraph)')
    p.add_argument('--no-plot', action='store_true')
    return p.parse_args()


def run_baseline(dataset, data, device, args):
    torch.manual_seed(args.seed)
    model = make_model(dataset, model_type=args.model, hidden_channels=args.hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = BaselineTrainer(model, optimizer, device=device)

    for epoch in range(args.epochs):
        trainer.train_epoch(data)

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc


def run_subgraph(dataset, data, device, num_bins, use_coverage, use_epoch_assignment, args):
    torch.manual_seed(args.seed)
    algo_kwargs = {}
    if args.algorithm == 3:
        algo_kwargs['subsample_prob'] = args.subsample_prob
    algorithm = get_algorithm(args.algorithm, **algo_kwargs)
    model = make_model(dataset, model_type=args.model, hidden_channels=args.hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = SubgraphTrainer(
        model, optimizer,
        num_bins=num_bins,
        algorithm=algorithm,
        use_coverage_correction=use_coverage,
        use_epoch_assignment=use_epoch_assignment,
        steps_per_epoch=args.steps_per_epoch,
        device=device,
    )

    if use_epoch_assignment:
        num_epochs = max(1, args.epochs // args.steps_per_epoch)
    else:
        num_epochs = args.epochs

    for epoch in range(num_epochs):
        trainer.train_epoch(data)

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading {args.dataset}...")
    dataset, data = load_dataset(args.dataset, device)
    print(f"{args.dataset}: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{dataset.num_features} features, {dataset.num_classes} classes")
    print(f"Train: {data.train_mask.sum().item()}, "
          f"Val: {data.val_mask.sum().item()}, "
          f"Test: {data.test_mask.sum().item()}")
    print()

    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def make_row(train_acc, test_acc, *, is_baseline, num_bins=None,
                 algorithm=None, coverage=None, epoch_assignment=None):
        row = {
            'timestamp': timestamp,
            'dataset': args.dataset,
            'model': args.model,
            'algorithm': algorithm,
            'num_bins': num_bins,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'hidden_channels': args.hidden_channels,
            'seed': args.seed,
            'steps_per_epoch': args.steps_per_epoch,
            'coverage': coverage,
            'epoch_assignment': epoch_assignment,
            'is_baseline': is_baseline,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }
        if args.algorithm == 3:
            row['subsample_prob'] = args.subsample_prob
        return row

    if args.baseline:
        print(f"Running: Baseline {args.model.upper()} ({args.epochs} epochs)...")
        train_acc, test_acc = run_baseline(dataset, data, device, args)
        results.append(make_row(train_acc, test_acc, is_baseline=True))
        print(f"  train={train_acc:.4f}, test={test_acc:.4f}")
    else:
        for num_bins in args.num_bins:
            if args.epoch_assignment:
                num_epochs = max(1, args.epochs // args.steps_per_epoch)
            else:
                num_epochs = args.epochs

            cov_str = 'cov=on' if args.coverage else 'cov=off'
            ea_str = 'ea=on' if args.epoch_assignment else 'ea=off'
            name = f'Algo{args.algorithm} N={num_bins} {cov_str} {ea_str}'
            if args.algorithm == 3 and args.subsample_prob > 0:
                name += f' p={args.subsample_prob}'
            print(f"Running: {name} ({num_epochs} epochs)...")

            train_acc, test_acc = run_subgraph(
                dataset, data, device, num_bins, args.coverage, args.epoch_assignment, args
            )
            results.append(make_row(
                train_acc, test_acc,
                is_baseline=False,
                num_bins=num_bins,
                algorithm=args.algorithm,
                coverage=args.coverage,
                epoch_assignment=args.epoch_assignment,
            ))
            print(f"  train={train_acc:.4f}, test={test_acc:.4f}")

    # Print results
    print()
    print("=" * 80)
    print(f"{'Run':<40} {'Bins':>4} {'Cov':>5} {'EA':>4} {'Train':>7} {'Test':>7}")
    print("-" * 80)
    for r in results:
        if r['is_baseline']:
            name = f"Baseline {r['model'].upper()}"
            bins_s, cov_s, ea_s = '-', '-', '-'
        else:
            cov_s = 'on' if r['coverage'] else 'off'
            ea_s = 'on' if r['epoch_assignment'] else 'off'
            name = f"Algo{r['algorithm']} N={r['num_bins']} cov={cov_s} ea={ea_s}"
            bins_s = str(r['num_bins'])
        print(f"{name:<40} {bins_s:>4} {cov_s:>5} {ea_s:>4} "
              f"{r['train_acc']:>7.4f} {r['test_acc']:>7.4f}")
    print("=" * 80)

    # Save results to JSON
    save_results(results, args)

    if not args.no_plot:
        try:
            plot_results(results, args)
        except ImportError:
            print("\nmatplotlib not available, skipping plot.")

    return results


def save_results(results, args):
    """Save experiment results as JSON Lines — one self-contained row per run."""
    timestamp = results[0]['timestamp']
    filename = f"results/{timestamp}_{args.dataset}_{args.model}.jsonl"

    os.makedirs('results', exist_ok=True)
    with open(filename, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved to: {filename}")


def plot_results(results, args):
    import matplotlib.pyplot as plt

    baseline_test = results[0]['test_acc']
    subgraph = [r for r in results if not r['is_baseline']]

    styles = {
        (False, False): dict(linestyle='-',  marker='o', label='cov=off, ea=off'),
        (False, True):  dict(linestyle='--', marker='s', label='cov=off, ea=on'),
        (True,  False): dict(linestyle='-',  marker='^', label='cov=on,  ea=off'),
        (True,  True):  dict(linestyle='--', marker='D', label='cov=on,  ea=on'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for (cov, ea), style in styles.items():
        points = [(r['num_bins'], r['test_acc'])
                  for r in subgraph if r['coverage'] == cov and r['epoch_assignment'] == ea]
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, **style)
    ax.axhline(baseline_test, color='red', linestyle=':', linewidth=1.5, label=f'Baseline ({baseline_test:.3f})')
    ax.set_xlabel('Number of Bins (N)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Test Accuracy vs Number of Bins ({args.dataset}, {args.model})')
    ax.set_xticks(args.num_bins)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (cov, ea), style in styles.items():
        points = [(r['num_bins'], r['test_acc'] - baseline_test)
                  for r in subgraph if r['coverage'] == cov and r['epoch_assignment'] == ea]
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, **style)
    ax.axhline(0, color='red', linestyle=':', linewidth=1.5, label='Baseline (0)')
    ax.set_xlabel('Number of Bins (N)')
    ax.set_ylabel('Accuracy Delta vs Baseline')
    ax.set_title(f'Accuracy Drop vs Number of Bins ({args.dataset}, {args.model})')
    ax.set_xticks(args.num_bins)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'experiments/{args.dataset}_{args.model}_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
