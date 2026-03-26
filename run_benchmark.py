"""
Benchmark: Accuracy vs Bins across Algorithms.

Produces one plot per dataset (cora, pubmed) showing test accuracy (y) vs
number of bins (x) for 5 methods: MLP, GCN, Algo 1, Algo 2, Algo 3.
Each configuration is averaged over 5 seeds.

Usage:
    python run_benchmark.py              # full run (200 epochs)
    python run_benchmark.py --epochs 3   # quick smoke test
    python run_benchmark.py --converge --max-epochs 500 --patience 20  # convergence mode
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

# ── Fixed hyperparameters ────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]
NUM_BINS = [2, 4, 8, 16]
DATASETS = ['cora', 'pubmed']
HIDDEN_CHANNELS = 64
LR = 0.01
WEIGHT_DECAY = 5e-4
STEPS_PER_EPOCH = 10
SUBSAMPLE_PROB = 0.3  # Algo 3 only


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_loop(trainer, data, *, epochs, converge, max_epochs, patience, delta):
    """Run training with either fixed epochs or convergence-based early stopping."""
    if not converge:
        for _ in range(epochs):
            trainer.train_epoch(data)
        return epochs

    best_loss = float('inf')
    wait = 0
    epoch = 0
    while epoch < max_epochs:
        losses = trainer.train_epoch(data)
        epoch += 1
        # BaselineTrainer returns a single loss; SubgraphTrainer returns a list
        if isinstance(losses, list):
            epoch_loss = sum(losses) / len(losses)
        else:
            epoch_loss = losses
        if best_loss - epoch_loss > delta:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return epoch


def run_baseline(dataset, data, device, *, model_type, seed, epochs,
                 converge=False, max_epochs=500, patience=20, delta=1e-4):
    torch.manual_seed(seed)
    model = make_model(dataset, model_type=model_type, hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trainer = BaselineTrainer(model, optimizer, device=device)

    actual_epochs = _train_loop(
        trainer, data, epochs=epochs, converge=converge,
        max_epochs=max_epochs, patience=patience, delta=delta,
    )

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc, actual_epochs


def run_subgraph(dataset, data, device, *, algorithm_id, num_bins, seed, epochs,
                 converge=False, max_epochs=500, patience=20, delta=1e-4):
    torch.manual_seed(seed)
    algo_kwargs = {}
    if algorithm_id == 3:
        algo_kwargs['subsample_prob'] = SUBSAMPLE_PROB
    algorithm = get_algorithm(algorithm_id, **algo_kwargs)

    model = make_model(dataset, model_type='gcn', hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trainer = SubgraphTrainer(
        model, optimizer,
        num_bins=num_bins,
        algorithm=algorithm,
        use_coverage_correction=False,
        use_epoch_assignment=False,
        steps_per_epoch=STEPS_PER_EPOCH,
        device=device,
    )

    actual_epochs = _train_loop(
        trainer, data, epochs=epochs, converge=converge,
        max_epochs=max_epochs, patience=patience, delta=delta,
    )

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc, actual_epochs


# ── Main benchmark loop ─────────────────────────────────────────────────────

def run_benchmark(epochs, *, converge=False, max_epochs=500, patience=20, delta=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if converge:
        print(f"Convergence mode: max_epochs={max_epochs}, patience={patience}, delta={delta}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    jsonl_path = f'results/benchmark_{timestamp}.jsonl'

    conv_kwargs = dict(converge=converge, max_epochs=max_epochs, patience=patience, delta=delta)
    all_results = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        dataset, data = load_dataset(ds_name, device)
        print(f"  {data.num_nodes} nodes, {data.num_edges} edges, "
              f"{dataset.num_features} features, {dataset.num_classes} classes")

        for seed in SEEDS:
            # ── MLP baseline ──
            print(f"\n  [seed={seed}] MLP baseline ...", end=' ', flush=True)
            train_acc, test_acc, actual_epochs = run_baseline(
                dataset, data, device, model_type='mlp', seed=seed, epochs=epochs,
                **conv_kwargs,
            )
            print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
            all_results.append(dict(
                dataset=ds_name, method='MLP', algorithm=None, num_bins=None,
                seed=seed, train_acc=train_acc, test_acc=test_acc,
                epochs=actual_epochs,
            ))

            # ── GCN baseline ──
            print(f"  [seed={seed}] GCN baseline ...", end=' ', flush=True)
            train_acc, test_acc, actual_epochs = run_baseline(
                dataset, data, device, model_type='gcn', seed=seed, epochs=epochs,
                **conv_kwargs,
            )
            print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
            all_results.append(dict(
                dataset=ds_name, method='GCN', algorithm=None, num_bins=None,
                seed=seed, train_acc=train_acc, test_acc=test_acc,
                epochs=actual_epochs,
            ))

            # ── Subgraph algorithms ──
            for algo_id in [1, 2, 3]:
                for nb in NUM_BINS:
                    label = f"Algo {algo_id}, bins={nb}"
                    if algo_id == 3:
                        label += f", p={SUBSAMPLE_PROB}"
                    print(f"  [seed={seed}] {label} ...", end=' ', flush=True)
                    train_acc, test_acc, actual_epochs = run_subgraph(
                        dataset, data, device,
                        algorithm_id=algo_id, num_bins=nb, seed=seed, epochs=epochs,
                        **conv_kwargs,
                    )
                    print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
                    all_results.append(dict(
                        dataset=ds_name, method=f'Algo {algo_id}', algorithm=algo_id,
                        num_bins=nb, seed=seed, train_acc=train_acc, test_acc=test_acc,
                        epochs=actual_epochs,
                        **(dict(subsample_prob=SUBSAMPLE_PROB) if algo_id == 3 else {}),
                    ))

        # Flush results after each dataset
        with open(jsonl_path, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')

    print(f"\nResults saved to {jsonl_path}")
    return all_results, timestamp


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_benchmark(all_results, timestamp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for ds_name in DATASETS:
        ds_rows = [r for r in all_results if r['dataset'] == ds_name]
        if not ds_rows:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        # ── Baselines (horizontal lines) ──
        for method, color, ls in [('MLP', 'tab:gray', '--'), ('GCN', 'tab:red', '--')]:
            accs = [r['test_acc'] for r in ds_rows if r['method'] == method]
            if not accs:
                continue
            mean = np.mean(accs)
            std = np.std(accs)
            ax.axhline(mean, color=color, linestyle=ls, linewidth=1.5,
                        label=f'{method} ({mean:.3f}±{std:.3f})')
            ax.axhspan(mean - std, mean + std, color=color, alpha=0.08)

        # ── Algo lines ──
        markers = {1: 'o', 2: 's', 3: '^'}
        colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
        for algo_id in [1, 2, 3]:
            algo_rows = [r for r in ds_rows if r['algorithm'] == algo_id]
            if not algo_rows:
                continue
            means, stds, xs = [], [], []
            for nb in NUM_BINS:
                accs = [r['test_acc'] for r in algo_rows if r['num_bins'] == nb]
                if accs:
                    xs.append(nb)
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))
            means, stds = np.array(means), np.array(stds)
            label = f'Algo {algo_id}'
            if algo_id == 3:
                label += f' (p={SUBSAMPLE_PROB})'
            ax.errorbar(xs, means, yerr=stds, marker=markers[algo_id],
                        color=colors[algo_id], capsize=4, linewidth=1.5, label=label)

        ax.set_xlabel('Number of Bins', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{ds_name.capitalize()} — Accuracy vs Bins', fontsize=13)
        ax.set_xticks(NUM_BINS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        save_path = f'results/benchmark_{ds_name}_{timestamp}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark: accuracy vs bins across algorithms")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Fixed number of epochs (ignored in convergence mode)')
    parser.add_argument('--converge', action='store_true',
                        help='Enable convergence mode (early stopping on train loss)')
    parser.add_argument('--max-epochs', type=int, default=500,
                        help='Upper bound on epochs in convergence mode (default: 500)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Epochs with no improvement before stopping (default: 20)')
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='Minimum loss decrease to count as improvement (default: 1e-4)')
    args = parser.parse_args()

    all_results, timestamp = run_benchmark(
        args.epochs,
        converge=args.converge,
        max_epochs=args.max_epochs,
        patience=args.patience,
        delta=args.delta,
    )
    plot_benchmark(all_results, timestamp)


if __name__ == '__main__':
    main()
