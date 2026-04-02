"""
Utility benchmark for ogbn-products only.

Runs MLP baseline, GCN baseline, and Algos 2/3 across bin counts.
No DP noise — pure utility evaluation.
Baselines use NeighborLoader mini-batching for tractable runtimes.

Usage:
    python run_ogbn_utility.py                        # 1 seed, convergence mode
    python run_ogbn_utility.py --seeds 3              # 3 seeds
    python run_ogbn_utility.py --seeds 5 --max-epochs 200
    python run_ogbn_utility.py --no-converge --epochs 50
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader

from src.datasets import load_dataset
from src.models import make_model
from src.algorithms import get_algorithm
from src.trainers.subgraph_trainer import SubgraphTrainer

# ── Hyperparameters (tuned for ogbn-products) ──────────────────────────────

ALL_SEEDS = [42, 123, 456, 789, 1024]
NUM_BINS = [4, 8, 16, 32]
HIDDEN_CHANNELS = 64
LR = 0.01
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]       # per-layer neighbor sampling (2-layer GCN)
STEPS_PER_EPOCH = 20
SUBSAMPLE_PROB = 0.3           # Algo 3 only


# ── Mini-batch baseline training ───────────────────────────────────────────

def train_baseline_epoch(model, optimizer, train_loader, device):
    """One epoch of mini-batch training using NeighborLoader."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # Only compute loss on seed nodes (the first batch_size nodes)
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


@torch.no_grad()
def eval_baseline(model, data, device):
    """Full-graph evaluation (inference is cheaper than training)."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return train_acc, test_acc


# ── Convergence / fixed-epoch loop ─────────────────────────────────────────

def _baseline_loop(model, optimizer, train_loader, device, *,
                   epochs, converge, max_epochs, patience, delta):
    if not converge:
        for _ in range(epochs):
            train_baseline_epoch(model, optimizer, train_loader, device)
        return epochs

    best_loss = float('inf')
    wait = 0
    epoch = 0
    while epoch < max_epochs:
        loss = train_baseline_epoch(model, optimizer, train_loader, device)
        epoch += 1
        if best_loss - loss > delta:
            best_loss = loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return epoch


def _subgraph_loop(trainer, data, *, epochs, converge, max_epochs, patience, delta):
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
        epoch_loss = sum(losses) / len(losses) if losses else float('inf')
        if best_loss - epoch_loss > delta:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return epoch


# ── Run helpers ─────────────────────────────────────────────────────────────

def run_baseline(dataset, data, device, *, model_type, seed, **loop_kwargs):
    torch.manual_seed(seed)
    model = make_model(dataset, model_type=model_type, hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    actual_epochs = _baseline_loop(model, optimizer, train_loader, device, **loop_kwargs)
    train_acc, test_acc = eval_baseline(model, data, device)
    return train_acc, test_acc, actual_epochs


def run_subgraph(dataset, data, device, *, algorithm_id, num_bins, seed, **loop_kwargs):
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

    actual_epochs = _subgraph_loop(trainer, data, **loop_kwargs)
    train_acc, test_acc = trainer.evaluate(data)
    return train_acc, test_acc, actual_epochs


# ── Main ────────────────────────────────────────────────────────────────────

def run_ogbn_utility(epochs, *, seeds, converge, max_epochs, patience, delta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mode_str = (f"Convergence mode: max_epochs={max_epochs}, patience={patience}, delta={delta}"
                if converge else f"Fixed epochs: {epochs}")
    print(f"{mode_str}  |  seeds={len(seeds)}  |  batch_size={BATCH_SIZE}")

    print("\nLoading ogbn-products ...")
    dataset, data = load_dataset('ogbn-products', device)
    print(f"  {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{dataset.num_features} features, {dataset.num_classes} classes")

    loop_kwargs = dict(epochs=epochs, converge=converge,
                       max_epochs=max_epochs, patience=patience, delta=delta)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    jsonl_path = f'results/ogbn_utility_{timestamp}.jsonl'
    all_results = []

    for seed in seeds:
        # ── MLP baseline ──
        print(f"\n  [seed={seed}] MLP baseline ...", end=' ', flush=True)
        train_acc, test_acc, actual_epochs = run_baseline(
            dataset, data, device, model_type='mlp', seed=seed, **loop_kwargs,
        )
        print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
        all_results.append(dict(
            method='MLP', algorithm=None, num_bins=None,
            seed=seed, train_acc=train_acc, test_acc=test_acc, epochs=actual_epochs,
        ))

        # ── GCN baseline ──
        print(f"  [seed={seed}] GCN baseline ...", end=' ', flush=True)
        train_acc, test_acc, actual_epochs = run_baseline(
            dataset, data, device, model_type='gcn', seed=seed, **loop_kwargs,
        )
        print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
        all_results.append(dict(
            method='GCN', algorithm=None, num_bins=None,
            seed=seed, train_acc=train_acc, test_acc=test_acc, epochs=actual_epochs,
        ))

        # ── Subgraph algorithms (Algo 2 & 3 only) ──
        for algo_id in [2, 3]:
            for nb in NUM_BINS:
                label = f"Algo {algo_id}, bins={nb}"
                if algo_id == 3:
                    label += f", p={SUBSAMPLE_PROB}"
                print(f"  [seed={seed}] {label} ...", end=' ', flush=True)
                train_acc, test_acc, actual_epochs = run_subgraph(
                    dataset, data, device,
                    algorithm_id=algo_id, num_bins=nb, seed=seed, **loop_kwargs,
                )
                print(f"train={train_acc:.4f}  test={test_acc:.4f}  epochs={actual_epochs}")
                all_results.append(dict(
                    method=f'Algo {algo_id}', algorithm=algo_id, num_bins=nb,
                    seed=seed, train_acc=train_acc, test_acc=test_acc,
                    epochs=actual_epochs,
                    **(dict(subsample_prob=SUBSAMPLE_PROB) if algo_id == 3 else {}),
                ))

        # Flush after each seed
        with open(jsonl_path, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')

    print(f"\nResults saved to {jsonl_path}")
    return all_results, timestamp


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(all_results, timestamp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Baselines (horizontal lines) ──
    for method, color, ls in [('MLP', 'tab:gray', '--'), ('GCN', 'tab:red', '--')]:
        accs = [r['test_acc'] for r in all_results if r['method'] == method]
        if not accs:
            continue
        mean, std = np.mean(accs), np.std(accs)
        ax.axhline(mean, color=color, linestyle=ls, linewidth=1.5,
                    label=f'{method} ({mean:.3f}\u00b1{std:.3f})')
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.08)

    # ── Algo lines ──
    markers = {2: 's', 3: '^'}
    colors = {2: 'tab:orange', 3: 'tab:green'}
    for algo_id in [2, 3]:
        algo_rows = [r for r in all_results if r.get('algorithm') == algo_id]
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
    ax.set_title('ogbn-products \u2014 Utility (Accuracy vs Bins)', fontsize=13)
    ax.set_xticks(NUM_BINS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = f'results/ogbn_utility_{timestamp}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Utility benchmark for ogbn-products")
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to run (1-5, default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Fixed number of epochs (ignored in convergence mode)')
    parser.add_argument('--converge', action='store_true', default=True,
                        help='Enable convergence mode (default: on)')
    parser.add_argument('--no-converge', dest='converge', action='store_false',
                        help='Disable convergence mode; use fixed epochs')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Upper bound on epochs in convergence mode (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Epochs with no improvement before stopping (default: 10)')
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='Minimum loss decrease to count as improvement (default: 1e-4)')
    args = parser.parse_args()

    seeds = ALL_SEEDS[:args.seeds]

    all_results, timestamp = run_ogbn_utility(
        args.epochs,
        seeds=seeds,
        converge=args.converge,
        max_epochs=args.max_epochs,
        patience=args.patience,
        delta=args.delta,
    )
    plot_results(all_results, timestamp)


if __name__ == '__main__':
    main()
