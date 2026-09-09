"""Run the configured small-root SparseGNN ablation with an epoch-equivalent budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sparse_inductive_ablation.json"))
    parser.add_argument("--dataset", help="run one configured dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="show SparseGNN progress every 500 steps")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    datasets = [args.dataset] if args.dataset else config["datasets"]
    root_rates = config["root_sampling_probabilities"]
    edge_rates = config["edge_retention_probabilities"]
    out_root = Path("results/inductive/sparse_ablation")
    for dataset in datasets:
        if not args.dry_run:
            from src.datasets import load_dataset
            from src.experiments.inductive import load_or_create_inductive_split
        if not args.dry_run:
            _, data = load_dataset(dataset, device="cpu")
            load_or_create_inductive_split(data, dataset, seed=int(config["seed"]))
        for p1 in root_rates:
            # Each node is selected an expected 100 times: the sparse engine's
            # native T is steps, not epochs.
            steps = math.ceil(config["training_epochs"] / p1)
            output = out_root / dataset / f"p1-{p1:g}"
            command = [
                sys.executable, "-m", "src.sparse.run", "--dataset", dataset, "--inductive",
                "--common_inductive_split", "--split_seed", str(config["seed"]),
                "--direction", "in", "--model", "gnn", "--aggr", "mean", "--p1", str(p1),
                "--p2", *(str(rate) for rate in edge_rates), "--r", str(config["radius"]),
                "--K_in", str(config["degree_caps"]["k_in"]), "--K_out", str(config["degree_caps"]["k_out"]),
                "--T", str(steps), "--seeds", "1", "--roots_from", "train",
                "--out_dir", str(output),
            ]
            if config["privacy"]["enabled"]:
                command.extend(["--dp", "--sigma", str(config["privacy"]["sigma"]),
                                "--clip", str(config["privacy"]["clip"])])
            if args.verbose:
                command.extend(["--verbose", "--progress_every", "500"])
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)
                if config["privacy"]["enabled"]:
                    result_csv = output / f"sparse_gnn_{dataset}_dp_results.csv"
                    subprocess.run(
                        [sys.executable, "-m", "src.sparse.compute_epsilon",
                         "--csv", str(result_csv), "--delta",
                         str(config["privacy"]["delta"])],
                        check=True)


if __name__ == "__main__":
    main()
