"""
Multi-seed sweep over (mode, drop_rate) configurations to sanity-check the
subsampling result. Prints per-seed test accuracy and the mean +/- std across
seeds for each configuration.

Usage:
  python sweep.py --seeds 1 2 3 4 5 --epochs 50 --drop_rates 0.0 0.1 0.25 0.5

Pass --dp_sigmas to also sweep DP-SGD runs (run.py mode `dp`), e.g.:
  python sweep.py --seeds 1 2 3 --epochs 50 --dp_sigmas 0.5 1.0 2.0
The reported eps column is epsilon at --delta from the chosen --accountant
(use --accountant dominating-pair --dominating_pair pair.json to account
with a custom per-step dominating pair instead of subsampled Gaussian).
"""

import argparse
import statistics

import run  # local module (run.py in same folder)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cora", "citeseer", "pubmed"], default="cora")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--drop_rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 10])
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    # DP-SGD sweep options
    p.add_argument("--dp_sigmas", type=float, nargs="+", default=[],
                   help="noise multipliers to sweep in dp mode (empty = no dp runs)")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--accountant",
                   choices=["opacus-rdp", "opacus-prv", "dominating-pair", "none"],
                   default="opacus-rdp")
    p.add_argument("--dominating_pair", type=str, default=None)
    p.add_argument("--pld_grid", type=float, default=1e-4)
    p.add_argument("--occurrence_bound", type=float, default=1.0)
    return p.parse_args()


def make_run_args(sweep_args, mode, drop_rate, seed, sigma=None):
    return argparse.Namespace(
        mode=mode,
        dataset=sweep_args.dataset,
        drop_rate=drop_rate,
        epochs=sweep_args.epochs,
        seed=seed,
        num_neighbors=sweep_args.num_neighbors,
        batch_size=sweep_args.batch_size,
        hidden=sweep_args.hidden,
        dropout=sweep_args.dropout,
        lr=sweep_args.lr,
        weight_decay=sweep_args.weight_decay,
        clip=sweep_args.clip,
        sigma=sigma if sigma is not None else 1.0,
        delta=sweep_args.delta,
        accountant=sweep_args.accountant if sigma is not None else "none",
        dominating_pair=sweep_args.dominating_pair,
        pld_grid=sweep_args.pld_grid,
        occurrence_bound=sweep_args.occurrence_bound,
    )


def main():
    run._check_imports()
    args = parse_args()

    # (mode, drop_rate, sigma); sigma is None for the non-DP arms
    configs = [("baseline", 0.0, None)]
    configs += [("subsample", dr, None) for dr in args.drop_rates if dr > 0]
    configs += [("dp", 0.0, s) for s in args.dp_sigmas]

    print(f"sweep: dataset={args.dataset}  seeds={args.seeds}  "
          f"epochs={args.epochs}  configs={configs}\n")
    print(f"{'config':<25} {'per-seed test':<45} {'mean':<8} {'std':<8} {'eps':<8}")
    print("-" * 100)

    for mode, drop_rate, sigma in configs:
        per_seed = []
        epsilon = None
        for seed in args.seeds:
            run_args = make_run_args(args, mode, drop_rate, seed, sigma=sigma)
            accs = run.run_experiment(run_args, verbose=False)
            per_seed.append(accs["test"])
            epsilon = accs.get("epsilon", epsilon)
        mean = statistics.mean(per_seed)
        std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        if sigma is not None:
            cfg_label = f"{mode}(sigma={sigma:.2f})"
        else:
            cfg_label = f"{mode}(drop={drop_rate:.2f})"
        per_seed_str = ", ".join(f"{v:.4f}" for v in per_seed)
        eps_str = f"{epsilon:.3f}" if epsilon is not None else "-"
        print(f"{cfg_label:<25} {per_seed_str:<45} {mean:.4f}   {std:.4f}   {eps_str}")


if __name__ == "__main__":
    main()
