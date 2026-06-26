"""
Run the seed sweep across drop rates for Cora and PubMed, plot the results,
and save two figures into figures/. Also records per-epoch test accuracy
curves so we can visualize how subsampling affects training dynamics.

Usage:
  python plot.py --seeds 1 2 3 4 5 --epochs 50
"""

import argparse
import os
import statistics

import matplotlib.pyplot as plt

import run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--drop_rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 10])
    p.add_argument("--out_dir", default="figures")
    return p.parse_args()


def make_run_args(args, dataset, mode, drop_rate, seed):
    return argparse.Namespace(
        mode=mode,
        dataset=dataset,
        drop_rate=drop_rate,
        epochs=args.epochs,
        seed=seed,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        hidden=16,
        dropout=0.5,
        lr=0.01,
        weight_decay=5e-4,
    )


def run_with_history(run_args):
    """Train one config and return (final_accs, per_epoch_test_accs).

    Re-implements the run loop here so we can record per-epoch test accuracy
    without modifying run.py's CLI surface.
    """
    import torch
    import torch.nn as nn

    run.set_seed(run_args.seed)
    dataset, data = run.load_planetoid(run_args.dataset)

    loader = run.NeighborSampler(
        data,
        input_nodes=data.train_mask,
        num_neighbors=run_args.num_neighbors,
        batch_size=run_args.batch_size,
        shuffle=True,
    )
    model = run.build_model(
        in_channels=dataset.num_features,
        hidden=run_args.hidden,
        out_channels=dataset.num_classes,
        dropout=run_args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=run_args.lr, weight_decay=run_args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    for _ in range(run_args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = run.batch_forward_loss(model, batch, run_args.mode,
                                             run_args.drop_rate, criterion)
            loss.backward()
            optimizer.step()
        accs = run.evaluate_full_graph(model, data)
        history.append(accs["test"])
    return history[-1], history


def sweep(args, dataset):
    configs = [("baseline", 0.0)] + [("subsample", dr) for dr in args.drop_rates if dr > 0]
    # results[(mode, drop_rate)] = list of (final_test, per_epoch_history)
    results = {}
    for mode, dr in configs:
        runs = []
        for seed in args.seeds:
            ra = make_run_args(args, dataset, mode, dr, seed)
            final, hist = run_with_history(ra)
            runs.append((final, hist))
            print(f"  {dataset:7s} {mode:9s} drop={dr:.2f} seed={seed} -> {final:.4f}")
        results[(mode, dr)] = runs
    return results


def plot_dataset(results, dataset, out_path):
    """Two-panel figure: (a) per-config final test acc with error bars,
    (b) mean per-epoch test-acc curves with shaded std band."""
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = []
    means = []
    stds = []
    for (mode, dr), runs in results.items():
        finals = [r[0] for r in runs]
        label = "baseline" if mode == "baseline" else f"drop={dr:.2f}"
        labels.append(label)
        means.append(statistics.mean(finals))
        stds.append(statistics.stdev(finals) if len(finals) > 1 else 0.0)

    xs = list(range(len(labels)))
    bars = ax_bar.bar(xs, means, yerr=stds, capsize=4,
                      color=["#888888"] + ["#1f77b4"] * (len(labels) - 1))
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("final test accuracy")
    ax_bar.set_title(f"{dataset}: final test acc (mean +/- std over seeds)")
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    pad = 0.01
    ax_bar.set_ylim(lo - pad, hi + pad)
    for bar, m in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, m + pad * 0.2,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    for (mode, dr), runs in results.items():
        histories = [r[1] for r in runs]
        n_epochs = len(histories[0])
        # mean and std across seeds at each epoch
        per_epoch_mean = []
        per_epoch_std = []
        for e in range(n_epochs):
            vals = [h[e] for h in histories]
            per_epoch_mean.append(statistics.mean(vals))
            per_epoch_std.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        xs = list(range(1, n_epochs + 1))
        label = "baseline" if mode == "baseline" else f"drop={dr:.2f}"
        line, = ax_curve.plot(xs, per_epoch_mean, label=label)
        lo_band = [m - s for m, s in zip(per_epoch_mean, per_epoch_std)]
        hi_band = [m + s for m, s in zip(per_epoch_mean, per_epoch_std)]
        ax_curve.fill_between(xs, lo_band, hi_band, alpha=0.15, color=line.get_color())

    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("test accuracy")
    ax_curve.set_title(f"{dataset}: test accuracy over training")
    ax_curve.legend(loc="lower right", fontsize=9)
    ax_curve.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def main():
    run._check_imports()
    args = parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n=== sweep on {dataset} ===")
        results = sweep(args, dataset)
        out_path = os.path.join(out_dir, f"{dataset}.png")
        plot_dataset(results, dataset, out_path)


if __name__ == "__main__":
    main()
