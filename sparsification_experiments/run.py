"""
Sparsification DP-GNN experiments — unified CLI.

Modes:
  utility  Sweep degree bound D; no noise. Measures accuracy cost of sparsification.
  dp       Sweep sigma; DP-SGD on sparsified graph. Reports accuracy + epsilon from
           two accountants.

Sparsifier / model flags:
  --sparsifier {symmetric, out_degree}   default: symmetric
  --model      {symmetric, directed}     default: symmetric

  Valid pairings:
    symmetric model  REQUIRES  symmetric sparsifier (GCNConv re-symmetrizes edges;
                               on an out-degree-only cap it expands fan-out and
                               invalidates the sensitivity bound).
    directed  model  works with either sparsifier.

Soundness flag:
  --no_subsampling  Sets q=1 (all training nodes every step).  This is the ONLY
                    configuration with a valid node-DP guarantee today.

Example commands (from inside sparsification_experiments/):
  python run.py --dataset cora --mode utility --degree_bounds 2 5 10 --seeds 3
  python run.py --dataset cora --mode utility --sparsifier out_degree --model directed \\
      --degree_bounds 2 5 10 --seeds 3
  python run.py --dataset cora --mode dp --no_subsampling \\
      --degree_bounds 5 --sigmas 1.0 2.0 4.0 --steps 200 --seeds 2 --plot
"""

import argparse
import csv
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import load_dataset                                         # noqa: E402
from sparsify import sparsify_by_outdegree, sparsify_symmetric, node_sensitivity  # noqa: E402
from dp_accounting import (                                      # noqa: E402
    opacus_prv_epsilon, dompair_epsilon,
    validate_accountants, validate_accountants_q1,
)
from train import make_model, train_utility, train_dp            # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _dataset_meta(dataset, data):
    return {
        'num_features': dataset.num_features,
        'num_classes': dataset.num_classes,
        'n_train': int(data.train_mask.sum().item()),
    }


def _set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)


def _sparsify(edge_index, num_nodes, D, seed, sparsifier):
    if sparsifier == 'symmetric':
        return sparsify_symmetric(edge_index, num_nodes, D, seed=seed)
    return sparsify_by_outdegree(edge_index, num_nodes, D, seed=seed)


def _check_pairing(args):
    """Raise if --model symmetric is combined with --sparsifier out_degree."""
    if args.model == 'symmetric' and args.sparsifier == 'out_degree':
        raise ValueError(
            "Invalid pairing: --model symmetric requires --sparsifier symmetric.\n"
            "SymmetricGCN's normalization re-introduces edges from ALL in-neighbors,\n"
            "including those pruned only from one direction by the out-degree cap.\n"
            "This leaves fan-out unbounded and invalidates the Delta sensitivity bound.\n"
            "Use --sparsifier symmetric  OR  --model directed."
        )


def _csv_writer(path):
    fh = open(path, 'w', newline='')
    fieldnames = [
        'dataset', 'mode', 'sparsifier', 'model_type', 'D', 'L', 'q', 'C',
        'sigma', 'T', 'delta', 'seed', 'test_acc', 'val_acc',
        'eps_opacus', 'eps_dompair', 'eps_sound',
    ]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    return fh, w


def _row(dataset_name, args, D, sigma, seed, accs, eps_op, eps_dp, q):
    sound = (q == 1.0) if q is not None else ''
    return {
        'dataset': dataset_name,
        'mode': args.mode,
        'sparsifier': args.sparsifier,
        'model_type': args.model,
        'D': D,
        'L': args.depth,
        'q': q if q is not None else '',
        'C': args.clip if args.mode == 'dp' else '',
        'sigma': sigma if sigma is not None else '',
        'T': args.steps if args.mode == 'dp' else '',
        'delta': args.delta if args.mode == 'dp' else '',
        'seed': seed,
        'test_acc': f"{accs['test']:.5f}",
        'val_acc': f"{accs['val']:.5f}",
        'eps_opacus': f"{eps_op:.5f}" if eps_op is not None else '',
        'eps_dompair': f"{eps_dp:.5f}" if eps_dp is not None else '',
        'eps_sound': sound,
    }


# ── utility mode ──────────────────────────────────────────────────────────────

# Directed-model / out-degree-sparsifier reference from the previous run (Cora,
# 3 seeds, 200 epochs).  Printed alongside new symmetric results for comparison.
_DIRECTED_REFERENCE = {
    'cora': {'full': 0.800, '10': 0.783, '5': 0.773, '2': 0.696},
}


def run_utility(args, dataset_name, dataset, data, writer, device):
    meta = _dataset_meta(dataset, data)
    degree_bounds = [None] + list(args.degree_bounds)   # None = full un-sparsified graph

    ref = _DIRECTED_REFERENCE.get(dataset_name.lower(), {}) if args.model == 'symmetric' else {}
    ref_col = '  ref(directed)' if ref else ''

    print(f"\n[utility] dataset={dataset_name}  sparsifier={args.sparsifier}  "
          f"model={args.model}  L={args.depth}  epochs={args.epochs}  seeds={args.seeds}")
    print(f"{'D':>6} {'seed':>5} {'val':>8} {'test':>8}{ref_col}")
    print("-" * (35 + (16 if ref else 0)))

    results = {}
    for D in degree_bounds:
        label = 'full' if D is None else str(D)
        accs_all = []
        for seed in range(args.seeds):
            _set_seed(seed)
            if D is None:
                edge_index = data.edge_index.to(device)
            else:
                edge_index = _sparsify(
                    data.edge_index, data.num_nodes, D, seed, args.sparsifier
                ).to(device)

            model = make_model(
                meta['num_features'], meta['num_classes'],
                hidden=args.hidden, dropout=args.dropout,
                num_layers=args.depth, model_type=args.model,
            ).to(device)

            accs = train_utility(
                model, data, edge_index,
                lr=args.lr, weight_decay=args.weight_decay,
                epochs=args.epochs, verbose=args.verbose,
            )
            accs_all.append(accs)
            ref_val = ref.get(label) if ref and seed == 0 else None
            ref_str = (f"  {ref_val:>13.3f}" if ref_val is not None else
                       ('               ' if ref else ''))
            print(f"{label:>6} {seed:>5}   {accs['val']:.4f}   {accs['test']:.4f}{ref_str}")
            writer.writerow(_row(dataset_name, args, label, None, seed, accs, None, None, None))

        results[label] = [a['test'] for a in accs_all]

    print()
    return results


# ── dp mode ───────────────────────────────────────────────────────────────────

_PLACEHOLDER_WARNING = """\
*** ACCOUNTING WARNING ***
q < 1 (Poisson subsampling).  The subsampled-Gaussian epsilon reported here
treats each training step as an independent Poisson draw on example-level
sensitivity = Delta.  This IGNORES that a single node can appear in multiple
seeds' L-hop subgraphs across many batches, potentially contributing to many
gradient terms over training.  The reported epsilon is a PLACEHOLDER — it is
NOT a valid node-DP bound and should not be used in papers or deployments.

A valid bound requires the novel per-step dominating pair from
make_novel_mechanism_dominating_pair (currently stubbed in dp_accounting.py).
Run with --no_subsampling (q=1) for a genuine, though conservative, guarantee.
*** END WARNING ***
"""


def run_dp(args, dataset_name, dataset, data, writer, device):
    meta = _dataset_meta(dataset, data)
    no_sub = args.no_subsampling
    q = 1.0 if no_sub else args.sample_rate

    print(f"\n[dp] dataset={dataset_name}  D={args.degree_bounds}  L={args.depth}  "
          f"q={q}  C={args.clip}  steps={args.steps}  delta={args.delta:g}  "
          f"adjacency={args.adjacency}  seeds={args.seeds}")

    if not no_sub:
        warnings.warn(_PLACEHOLDER_WARNING, stacklevel=2)
        print("\n" + _PLACEHOLDER_WARNING)

    for D in args.degree_bounds:
        Delta = node_sensitivity(args.clip, D, args.depth, args.adjacency)
        print(f"  D={D}  Delta={Delta:.4f}  (adjacency={args.adjacency}  "
              f"noise_std = sigma * {Delta:.4f})")

    # Accountant validation (once, before training)
    if no_sub:
        validate_accountants_q1(
            args.sigmas, args.steps, args.delta,
            grid=args.pld_grid, tol=args.validation_tol,
        )
    else:
        validate_accountants(
            args.sigmas, q, args.steps, args.delta,
            grid=args.pld_grid, tol=args.validation_tol,
        )

    all_rows = []

    for D in args.degree_bounds:
        for sigma in args.sigmas:
            accs_all, eps_ops, eps_dps = [], [], []
            for seed in range(args.seeds):
                _set_seed(seed)
                edge_index = _sparsify(
                    data.edge_index, data.num_nodes, D, seed, args.sparsifier
                ).to(device)

                model = make_model(
                    meta['num_features'], meta['num_classes'],
                    hidden=args.hidden, dropout=args.dropout,
                    num_layers=args.depth, model_type=args.model,
                ).to(device)

                accs, actual_steps = train_dp(
                    model, data, edge_index,
                    steps=args.steps, C=args.clip, sigma=sigma,
                    D=D, L=args.depth, q=q,
                    adjacency=args.adjacency,
                    no_subsampling=no_sub,
                    lr=args.lr, weight_decay=args.weight_decay,
                    verbose=args.verbose,
                )

                eps_op = opacus_prv_epsilon(sigma, q, actual_steps, args.delta)
                eps_dp = dompair_epsilon(q, sigma, actual_steps, args.delta, grid=args.pld_grid)
                accs_all.append(accs)
                eps_ops.append(eps_op)
                eps_dps.append(eps_dp)

                sound_str = 'SOUND' if no_sub else 'placeholder'
                print(f"  D={D}  sigma={sigma:.2f}  seed={seed}  "
                      f"test={accs['test']:.4f}  "
                      f"eps_op={eps_op:.4f}  eps_dp={eps_dp:.4f}  [{sound_str}]")
                writer.writerow(
                    _row(dataset_name, args, D, sigma, seed, accs, eps_op, eps_dp, q)
                )

            mean_test = sum(a['test'] for a in accs_all) / len(accs_all)
            mean_eps_op = sum(eps_ops) / len(eps_ops)
            mean_eps_dp = sum(eps_dps) / len(eps_dps)
            all_rows.append((D, sigma, mean_eps_op, mean_eps_dp, mean_test))
            print(f"  >> D={D} sigma={sigma:.2f} "
                  f"mean_test={mean_test:.4f} "
                  f"eps_op={mean_eps_op:.4f} eps_dp={mean_eps_dp:.4f}")

    return all_rows


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_utility_results(results_by_dataset, out_dir):
    """Bar/line plot of test accuracy vs degree cap D for utility mode."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)

    for dataset_name, results in results_by_dataset.items():
        # results: dict label -> list of test_acc per seed
        # Put numeric D values in ascending order, then "full" at the end
        numeric = sorted((k for k in results if k != 'full'), key=int)
        labels = numeric + (['full'] if 'full' in results else [])
        means = [sum(results[k]) / len(results[k]) for k in labels]
        stds  = [(sum((x - m) ** 2 for x in results[k]) / max(len(results[k]) - 1, 1)) ** 0.5
                 for k, m in zip(labels, means)]

        fig, ax = plt.subplots(figsize=(6, 4))
        x = list(range(len(labels)))
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=4, linewidth=1.5,
                    markersize=5, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('degree cap D  (full = no sparsification)')
        ax.set_ylabel('test accuracy')
        ax.set_title(f'{dataset_name}: accuracy vs degree cap (utility, no DP)')
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f'{dataset_name}_utility.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {path}")


def plot_results(all_rows, dataset_name, out_dir, sound):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    sound_tag = ' (q=1, sound)' if sound else ' (q<1, placeholder)'

    by_D = {}
    for D, sigma, eps_op, eps_dp, test_acc in all_rows:
        by_D.setdefault(D, []).append((sigma, eps_op, eps_dp, test_acc))

    # Plot 1: privacy-utility frontier (test acc vs eps_dompair)
    fig, ax = plt.subplots(figsize=(6, 4))
    for D, rows in sorted(by_D.items()):
        rows_s = sorted(rows, key=lambda r: r[2])
        ax.plot([r[2] for r in rows_s], [r[3] for r in rows_s], 'o-', label=f'D={D}')
    ax.set_xlabel('epsilon (dominating pair)')
    ax.set_ylabel('test accuracy')
    ax.set_title(f'{dataset_name}: privacy-utility frontier{sound_tag}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path1 = os.path.join(out_dir, f'{dataset_name}_frontier.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path1}")

    # Plot 2: epsilon vs sigma (both accountants overlaid)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10.colors
    for i, (D, rows) in enumerate(sorted(by_D.items())):
        rows_s = sorted(rows, key=lambda r: r[0])
        c = colors[i % len(colors)]
        ax.plot([r[0] for r in rows_s], [r[1] for r in rows_s],
                'o-', color=c, label=f'D={D} opacus')
        ax.plot([r[0] for r in rows_s], [r[2] for r in rows_s],
                's--', color=c, label=f'D={D} dompair', alpha=0.7)
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_title(f'{dataset_name}: epsilon vs sigma{sound_tag}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path2 = os.path.join(out_dir, f'{dataset_name}_eps_vs_sigma.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path2}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dataset', nargs='+', default=['cora'],
                   help='dataset(s): cora, citeseer, pubmed, ogbn-arxiv')
    p.add_argument('--mode', choices=['utility', 'dp'], required=True)

    # Sparsifier / model
    p.add_argument('--sparsifier', choices=['symmetric', 'out_degree'], default='symmetric',
                   help='graph sparsifier: symmetric (default) caps undirected degree; '
                        'out_degree caps only outgoing arcs')
    p.add_argument('--model', choices=['symmetric', 'directed'], default='symmetric',
                   help='GNN variant: symmetric (default, GCNConv + sym-norm) or '
                        'directed (SAGEConv, in-neighbor mean only). '
                        'symmetric model requires symmetric sparsifier.')

    # Graph / model
    p.add_argument('--degree_bounds', type=int, nargs='+', default=[5],
                   help='degree cap D values to sweep')
    p.add_argument('--depth', type=int, default=2,
                   help='number of GNN layers L')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=5e-4)

    # Utility mode
    p.add_argument('--epochs', type=int, default=200)

    # DP mode
    p.add_argument('--clip', type=float, default=1.0,
                   help='per-node gradient clipping norm C')
    p.add_argument('--sigmas', type=float, nargs='+', default=[1.0, 2.0, 4.0],
                   help='sensitivity-normalised noise multipliers to sweep')
    p.add_argument('--sample_rate', type=float, default=0.5,
                   help='Poisson sampling rate q (ignored when --no_subsampling)')
    p.add_argument('--no_subsampling', action='store_true',
                   help='q=1: use all training nodes every step. '
                        'The ONLY configuration with a valid node-DP guarantee today.')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--delta', type=float, default=1e-3)
    p.add_argument('--adjacency', choices=['remove', 'add_remove'], default='add_remove',
                   help='sensitivity bound: conservative add/remove (default) or remove-only')
    p.add_argument('--pld_grid', type=float, default=1e-4)
    p.add_argument('--validation_tol', type=float, default=0.1)

    # General
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--out_dir', type=str, default='results')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--verbose', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    _check_pairing(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, 'results.csv')
    csv_fh, csv_writer_obj = _csv_writer(csv_path)

    utility_results = {}   # dataset_name -> {label -> [test_acc per seed]}

    try:
        for dataset_name in args.dataset:
            print(f"\n{'='*60}")
            print(f"dataset: {dataset_name}  device: {device}")
            dataset, data = load_dataset(dataset_name, device=str(device))
            data = data.to(device)

            if args.mode == 'utility':
                utility_results[dataset_name] = run_utility(
                    args, dataset_name, dataset, data, csv_writer_obj, device
                )

            elif args.mode == 'dp':
                all_rows = run_dp(args, dataset_name, dataset, data, csv_writer_obj, device)
                if args.plot:
                    plot_results(all_rows, dataset_name, args.out_dir,
                                 sound=args.no_subsampling)

    finally:
        csv_fh.close()

    if args.mode == 'utility' and args.plot and utility_results:
        plot_utility_results(utility_results, args.out_dir)

    print(f"\nresults written to {csv_path}")


if __name__ == '__main__':
    main()
