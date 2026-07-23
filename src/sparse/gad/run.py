"""
XGB-Graph graph anomaly detection on sparsified graphs (GADBench) — CLI.

Sweeps the edge-sampling probability p2 and reports the utility drop (AUROC / AUPRC / Rec@K)
as the graph is sparsified. p2=1.0 is the full-graph reference.

Run in the PytorchEnv conda env (has torch, PyG, xgboost, scikit-learn):

  conda run -n PytorchEnv python -m src.sparse.gad.run --dataset tolokers \\
      --p2 1.0 0.75 0.5 0.25 0.1 --num_layers 2 --seeds 5 --plot
"""

import argparse
import csv
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.datasets import load_dataset                        # noqa: E402
from src.sparse.gad.xgb_graph import XGBGraphDetector        # noqa: E402
from src.sparse.gad.metrics import auroc, auprc, rec_at_k    # noqa: E402


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def plot_sweep(summary, dataset_name, out_dir):
    """metric vs p2, one line per metric (mean over seeds, with error bars)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    rows = sorted(summary, key=lambda s: s[0])
    p2s = [s[0] for s in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, name in enumerate(('AUROC', 'AUPRC', 'Rec@K')):
        means = [s[1 + 2 * j] for s in rows]
        stds = [s[2 + 2 * j] for s in rows]
        ax.errorbar(p2s, means, yerr=stds, fmt='o-', capsize=4, label=name)
    ax.set_xlabel('edge-sampling probability p2  (1.0 = full graph)')
    ax.set_ylabel('score')
    ax.set_title(f'{dataset_name}: XGB-Graph utility vs sparsification')
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, f'gad_{dataset_name}_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='tolokers', help='tolokers | questions')
    p.add_argument('--p2', type=float, nargs='+', default=[1.0, 0.75, 0.5, 0.25, 0.1],
                   help='edge-sampling probabilities to sweep')
    p.add_argument('--num_layers', '-L', type=int, default=2,
                   help='neighbor-aggregation hops L (GADBench default 2)')
    p.add_argument('--aggr', choices=['mean', 'sum', 'max', 'min'], default='mean')
    p.add_argument('--sparsifier', choices=['global', 'expand'], default='global',
                   help="'global' (fast, default) or 'expand' (per-root SparseExpand, slow)")
    p.add_argument('--split_idx', type=int, default=0,
                   help='which predefined split to use (Tolokers/Questions have 10)')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--out_dir', default='results')
    p.add_argument('--plot', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'gad_{args.dataset}_results.csv')

    print(f"\n{'='*66}")
    print(f"XGB-Graph GAD  dataset={args.dataset}  L={args.num_layers}  aggr={args.aggr}  "
          f"sparsifier={args.sparsifier}  split={args.split_idx}  seeds={args.seeds}")
    print(f"  p2 sweep: {args.p2}")
    print('='*66)

    dataset, data = load_dataset(args.dataset, split_idx=args.split_idx)
    y = data.y
    test_mask = data.test_mask
    n_anom = int(y[test_mask].sum().item())
    print(f"  nodes={data.num_nodes}  edges={data.edge_index.size(1)}  feat={dataset.num_features}"
          f"  test_anomalies={n_anom}/{int(test_mask.sum())}")

    summary = []   # (p2, auroc_m, auroc_s, auprc_m, auprc_s, rec_m, rec_s)

    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'p2', 'L', 'aggr', 'sparsifier', 'split_idx',
                    'seed', 'auroc', 'auprc', 'rec_at_k'])

        for p2 in args.p2:
            # At p2>=1 no edge is dropped, so features (and XGBoost) are deterministic:
            # extra seeds just recompute identical results. One seed suffices.
            n_seeds = 1 if p2 >= 1.0 else args.seeds
            print(f"\n[p2={p2}]" + ("  (deterministic: 1 seed)" if n_seeds == 1 else ""))
            aurocs, auprcs, recs = [], [], []
            for seed in range(n_seeds):
                _set_seed(seed)
                gen = torch.Generator().manual_seed(seed)
                det = XGBGraphDetector(num_layers=args.num_layers, aggr=args.aggr)
                X = det.build_features(data, p2=p2, sparsifier=args.sparsifier, generator=gen)
                det.fit(X, y, data.train_mask)
                scores = det.predict_scores(X)

                yt = y[test_mask]
                st = scores[test_mask.cpu().numpy()]
                a, p_, rk = auroc(yt, st), auprc(yt, st), rec_at_k(yt, st)
                aurocs.append(a); auprcs.append(p_); recs.append(rk)
                print(f"  seed={seed}  AUROC={a:.4f}  AUPRC={p_:.4f}  Rec@K={rk:.4f}")
                w.writerow([args.dataset, p2, args.num_layers, args.aggr, args.sparsifier,
                            args.split_idx, seed, f"{a:.5f}", f"{p_:.5f}", f"{rk:.5f}"])

            am, asd = _mean_std(aurocs)
            pm, psd = _mean_std(auprcs)
            rm, rsd = _mean_std(recs)
            summary.append((p2, am, asd, pm, psd, rm, rsd))
            print(f"  >> AUROC {am:.4f}+/-{asd:.4f}  AUPRC {pm:.4f}+/-{psd:.4f}  "
                  f"Rec@K {rm:.4f}+/-{rsd:.4f}")

    print(f"\n{'='*66}")
    print(f"{'p2':>6} {'AUROC':>16} {'AUPRC':>16} {'Rec@K':>16}")
    print('-'*66)
    for p2, am, asd, pm, psd, rm, rsd in sorted(summary, key=lambda s: -s[0]):
        print(f"{p2:>6} {am:.4f}+/-{asd:.4f}   {pm:.4f}+/-{psd:.4f}   {rm:.4f}+/-{rsd:.4f}")
    print(f"\nresults written to {csv_path}")

    if args.plot:
        path = plot_sweep(summary, args.dataset, args.out_dir)
        if path:
            print(f"plot written to {path}")


if __name__ == '__main__':
    main()
