"""
Utility check: Balls-and-bins subgraph GCN vs baseline GCN on Cora (no DP).

Run from repo root: python experiments/subgraph_cora.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid

from src.models import make_model
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.subgraph_trainer import SubgraphTrainer
from src.algorithms.balls_and_bins import BallsAndBins

CONFIG = {
    'num_epochs': 200,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'hidden_channels': 64,
    'seed': 42,
    'num_bins_list': [2, 4, 8],
    'coverage_flags': [False, True],
    'steps_per_epoch': 10,  # T
}


def load_cora(device):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    return dataset, data


def run_baseline(dataset, data, device, num_epochs):
    torch.manual_seed(CONFIG['seed'])
    model = make_model(dataset, hidden_channels=CONFIG['hidden_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    trainer = BaselineTrainer(model, optimizer, device=device)

    for epoch in range(num_epochs):
        trainer.train_epoch(data)

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc


def run_subgraph(dataset, data, device, num_bins, use_coverage, use_epoch_assignment, num_epochs):
    torch.manual_seed(CONFIG['seed'])
    model = make_model(dataset, hidden_channels=CONFIG['hidden_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    trainer = SubgraphTrainer(
        model, optimizer,
        num_bins=num_bins,
        algorithm=BallsAndBins(),
        use_coverage_correction=use_coverage,
        use_epoch_assignment=use_epoch_assignment,
        steps_per_epoch=CONFIG['steps_per_epoch'],
        device=device,
    )

    for epoch in range(num_epochs):
        trainer.train_epoch(data)

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading Cora...")
    dataset, data = load_cora(device)
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{dataset.num_features} features, {dataset.num_classes} classes")
    print(f"Train: {data.train_mask.sum().item()}, "
          f"Val: {data.val_mask.sum().item()}, "
          f"Test: {data.test_mask.sum().item()}")
    print()

    results = []
    T = CONFIG['steps_per_epoch']
    base_epochs = CONFIG['num_epochs']

    # Baseline
    print(f"Running: Baseline GCN ({base_epochs} epochs)...")
    train_acc, test_acc = run_baseline(dataset, data, device, base_epochs)
    results.append({
        'name': 'Baseline GCN',
        'bins': '-',
        'coverage': '-',
        'epoch_assign': '-',
        'epochs': base_epochs,
        'train_acc': train_acc,
        'test_acc': test_acc,
    })
    print(f"  train={train_acc:.4f}, test={test_acc:.4f}")

    # Subgraph variants
    for num_bins in CONFIG['num_bins_list']:
        for use_coverage in CONFIG['coverage_flags']:
            for use_epoch_assignment in [False, True]:
                # Match total gradient steps to baseline
                if use_epoch_assignment:
                    num_epochs = max(1, base_epochs // T)
                else:
                    num_epochs = base_epochs

                cov_str = 'cov=on' if use_coverage else 'cov=off'
                ea_str = 'ea=on' if use_epoch_assignment else 'ea=off'
                name = f'Subgraph N={num_bins} {cov_str} {ea_str}'
                print(f"Running: {name} ({num_epochs} epochs)...")

                train_acc, test_acc = run_subgraph(
                    dataset, data, device, num_bins, use_coverage, use_epoch_assignment, num_epochs
                )
                results.append({
                    'name': name,
                    'bins': num_bins,
                    'coverage': 'on' if use_coverage else 'off',
                    'epoch_assign': 'on' if use_epoch_assignment else 'off',
                    'epochs': num_epochs,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                })
                print(f"  train={train_acc:.4f}, test={test_acc:.4f}")

    # Print comparison table
    print()
    print("=" * 80)
    print(f"{'Model':<40} {'Bins':>4} {'Cov':>5} {'EA':>4} {'Epochs':>6} {'Train':>7} {'Test':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<40} {str(r['bins']):>4} {str(r['coverage']):>5} "
              f"{str(r['epoch_assign']):>4} {r['epochs']:>6} "
              f"{r['train_acc']:>7.4f} {r['test_acc']:>7.4f}")
    print("=" * 80)

    # Summary: test acc delta vs baseline
    baseline_test = results[0]['test_acc']
    print(f"\nBaseline test acc: {baseline_test:.4f}")
    print("Delta (subgraph - baseline):")
    for r in results[1:]:
        delta = r['test_acc'] - baseline_test
        sign = '+' if delta >= 0 else ''
        print(f"  {r['name']:<40} {sign}{delta:.4f}")

    return results


def plot_results(results, save_path='experiments/subgraph_cora_results.png'):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    baseline_test = results[0]['test_acc']
    subgraph = [r for r in results if r['bins'] != '-']

    # Group by (coverage, epoch_assignment) for line style
    styles = {
        ('off', 'off'): dict(linestyle='-',  marker='o', label='cov=off, ea=off'),
        ('off', 'on'):  dict(linestyle='--', marker='s', label='cov=off, ea=on'),
        ('on',  'off'): dict(linestyle='-',  marker='^', label='cov=on,  ea=off'),
        ('on',  'on'):  dict(linestyle='--', marker='D', label='cov=on,  ea=on'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: test accuracy vs N
    ax = axes[0]
    for (cov, ea), style in styles.items():
        points = [(r['bins'], r['test_acc'])
                  for r in subgraph if r['coverage'] == cov and r['epoch_assign'] == ea]
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, **style)
    ax.axhline(baseline_test, color='red', linestyle=':', linewidth=1.5, label=f'Baseline ({baseline_test:.3f})')
    ax.set_xlabel('Number of Bins (N)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs Number of Bins')
    ax.set_xticks(CONFIG['num_bins_list'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: accuracy drop vs N (delta from baseline)
    ax = axes[1]
    for (cov, ea), style in styles.items():
        points = [(r['bins'], r['test_acc'] - baseline_test)
                  for r in subgraph if r['coverage'] == cov and r['epoch_assign'] == ea]
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, **style)
    ax.axhline(0, color='red', linestyle=':', linewidth=1.5, label='Baseline (0)')
    ax.set_xlabel('Number of Bins (N)')
    ax.set_ylabel('Accuracy Delta vs Baseline')
    ax.set_title('Accuracy Drop vs Number of Bins')
    ax.set_xticks(CONFIG['num_bins_list'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    results = main()
    plot_results(results)
