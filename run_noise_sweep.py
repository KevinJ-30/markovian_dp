"""
Noise-multiplier sweep: Accuracy vs Privacy Budget (epsilon).

Sweeps over noise multiplier values, trains with DP, then uses Opacus's
RDP accountant to compute the resulting epsilon. Plots accuracy vs epsilon.

Usage:
    python run_noise_sweep.py                          # full run (200 epochs)
    python run_noise_sweep.py --epochs 3               # quick smoke test
    python run_noise_sweep.py --noise-mults 0.5 1.0 2.0 4.0 8.0
    python run_noise_sweep.py --num-bins 4 8 16
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
from src.privacy_accountant import compute_epsilon

# ── Fixed hyperparameters ────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]
NUM_BINS = [2, 4, 8, 16]
NOISE_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0, 8.0]
DATASETS = ['cora', 'pubmed']
HIDDEN_CHANNELS = 64
LR = 0.01
WEIGHT_DECAY = 5e-4
STEPS_PER_EPOCH = 10
SUBSAMPLE_PROB = 0.3  # Algo 3 only


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_loop(trainer, data, *, epochs, converge, max_epochs, patience, delta):
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


def run_subgraph_noise(dataset, data, device, *, algorithm_id, num_bins, seed,
                       epochs, noise_multiplier, clip_norm=1.0, dp_delta=1e-5,
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
        dp=True,
        max_grad_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        delta=dp_delta,
    )

    actual_epochs = _train_loop(
        trainer, data, epochs=epochs, converge=converge,
        max_epochs=max_epochs, patience=patience, delta=delta,
    )

    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc, actual_epochs, trainer.training_steps


# ── Main sweep loop ──────────────────────────────────────────────────────────

def run_noise_sweep(epochs, *, noise_multipliers, num_bins_list, converge=False,
                    max_epochs=500, patience=20, delta=1e-4,
                    clip_norm=1.0, dp_delta=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Noise multipliers: {noise_multipliers}")
    print(f"Num bins: {num_bins_list}")
    if converge:
        print(f"Convergence mode: max_epochs={max_epochs}, patience={patience}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    jsonl_path = f'results/noise_sweep_{timestamp}.jsonl'

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
                noise_multiplier=None, computed_epsilon=None,
                seed=seed, train_acc=train_acc, test_acc=test_acc, epochs=actual_epochs,
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
                noise_multiplier=None, computed_epsilon=None,
                seed=seed, train_acc=train_acc, test_acc=test_acc, epochs=actual_epochs,
            ))

            # ── Subgraph algorithms with noise sweep ──
            for algo_id in [1, 2, 3]:
                for nb in num_bins_list:
                    for noise_mult in noise_multipliers:
                        label = f"Algo {algo_id}, bins={nb}, σ={noise_mult}"
                        if algo_id == 3:
                            label += f", p={SUBSAMPLE_PROB}"
                        print(f"  [seed={seed}] {label} ...", end=' ', flush=True)

                        train_acc, test_acc, actual_epochs, num_steps = run_subgraph_noise(
                            dataset, data, device,
                            algorithm_id=algo_id, num_bins=nb, seed=seed,
                            epochs=epochs, noise_multiplier=noise_mult,
                            clip_norm=clip_norm, dp_delta=dp_delta,
                            **conv_kwargs,
                        )

                        # Compute sample rate for privacy accounting
                        if algo_id == 3:
                            sample_rate = (1 - SUBSAMPLE_PROB) / nb
                        else:
                            sample_rate = 1.0 / nb

                        eps = compute_epsilon(
                            noise_multiplier=noise_mult,
                            sample_rate=sample_rate,
                            num_steps=num_steps,
                            delta=dp_delta,
                        )

                        print(f"train={train_acc:.4f}  test={test_acc:.4f}  "
                              f"ε={eps:.2f}  steps={num_steps}")

                        row = dict(
                            dataset=ds_name,
                            method=f'Algo {algo_id}',
                            algorithm=algo_id,
                            num_bins=nb,
                            noise_multiplier=noise_mult,
                            computed_epsilon=eps,
                            delta=dp_delta,
                            clip_norm=clip_norm,
                            sample_rate=sample_rate,
                            seed=seed,
                            train_acc=train_acc,
                            test_acc=test_acc,
                            epochs=actual_epochs,
                            training_steps=num_steps,
                        )
                        if algo_id == 3:
                            row['subsample_prob'] = SUBSAMPLE_PROB
                        all_results.append(row)

        # Flush results after each dataset
        with open(jsonl_path, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')

    print(f"\nResults saved to {jsonl_path}")
    return all_results, timestamp


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_noise_sweep(all_results, timestamp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for ds_name in DATASETS:
        ds_rows = [r for r in all_results if r['dataset'] == ds_name]
        if not ds_rows:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

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

        # ── Algo lines: one per (algo, num_bins) ──
        algo_colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
        markers = {2: 'o', 4: 's', 8: '^', 16: 'D'}
        linestyles = {2: '-', 4: '--', 8: '-.', 16: ':'}

        for algo_id in [1, 2, 3]:
            algo_rows = [r for r in ds_rows if r['algorithm'] == algo_id]
            if not algo_rows:
                continue

            bins_in_data = sorted(set(r['num_bins'] for r in algo_rows))
            for nb in bins_in_data:
                bin_rows = [r for r in algo_rows if r['num_bins'] == nb]
                # Group by noise_multiplier (each gives a different epsilon)
                noise_vals = sorted(set(r['noise_multiplier'] for r in bin_rows))

                epsilons, means, stds = [], [], []
                for nv in noise_vals:
                    nv_rows = [r for r in bin_rows if r['noise_multiplier'] == nv]
                    accs = [r['test_acc'] for r in nv_rows]
                    eps_vals = [r['computed_epsilon'] for r in nv_rows]
                    epsilons.append(np.mean(eps_vals))
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))

                epsilons, means, stds = np.array(epsilons), np.array(means), np.array(stds)
                # Sort by epsilon for clean line
                order = np.argsort(epsilons)
                epsilons, means, stds = epsilons[order], means[order], stds[order]

                label = f'Algo {algo_id}, bins={nb}'
                if algo_id == 3:
                    label += f' (p={SUBSAMPLE_PROB})'
                ax.errorbar(epsilons, means, yerr=stds,
                            marker=markers.get(nb, 'o'),
                            color=algo_colors[algo_id],
                            linestyle=linestyles.get(nb, '-'),
                            capsize=4, linewidth=1.5, label=label, alpha=0.85)

        ax.set_xscale('log')
        ax.set_xlabel('Privacy Budget ε (log scale)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{ds_name.capitalize()} — Accuracy vs Privacy Budget', fontsize=13)
        ax.legend(fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)

        save_path = f'results/noise_sweep_{ds_name}_{timestamp}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Noise-multiplier sweep: accuracy vs computed epsilon")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--converge', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='Early stopping delta')
    parser.add_argument('--noise-mults', type=float, nargs='+',
                        default=NOISE_MULTIPLIERS,
                        help='Noise multiplier values to sweep')
    parser.add_argument('--num-bins', type=int, nargs='+',
                        default=NUM_BINS,
                        help='Number of bins to sweep')
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--dp-delta', type=float, default=1e-5,
                        help='Privacy parameter delta')
    args = parser.parse_args()

    all_results, timestamp = run_noise_sweep(
        args.epochs,
        noise_multipliers=sorted(args.noise_mults),
        num_bins_list=sorted(args.num_bins),
        converge=args.converge,
        max_epochs=args.max_epochs,
        patience=args.patience,
        delta=args.delta,
        clip_norm=args.clip_norm,
        dp_delta=args.dp_delta,
    )
    plot_noise_sweep(all_results, timestamp)


if __name__ == '__main__':
    main()
