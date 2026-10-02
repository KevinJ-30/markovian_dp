#!/usr/bin/env python
"""Approved 280-run, seed-0 FB gender / Coauthor Physics initial screen."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOLS = ("fb100-gender-1", "fb100-gender-3", "fb100-gender-6",
             "fb100-gender-16", "coauthor-physics")
METHODS = ("sparse_sage", "sparse_gin", "progap")
TRAIN_SCHOOLS = ("johns-hopkins55", "caltech36", "amherst41", "reed98",
                 "brandeis99", "princeton12")
SPEC = {
    "phase": "initial", "adaptive_rounds": False, "training_seeds": [0],
    "fb_target": "recorded gender categories: raw 1/2 -> 0/1; raw 0 nodes excluded",
    "fb_features": "fixed categorical vocabulary over all 18 raw schools",
    "fb_validation": ["cornell5"], "fb_test": ["penn94"],
    "physics_split": "seed-0 stratified 60/20/20, separate induced graphs",
    "gin_pooling": "mean neighbors plus separate root, fixed epsilon_GIN=0",
    "physical_gpu_indices": [4, 5, 6, 7], "max_jobs_per_gpu": 2,
}
INTERPRETATION = {
    "configuration_selection": "highest validation score within protocol/method/epsilon; ties use immutable request ordinal; test scores never select configurations",
    "checkpoint_selection": "first strict validation-primary-metric maximum; no early stopping",
    "uncertainty": "one training seed (0), no bootstrap; no across-seed uncertainty or confidence intervals",
    "privacy": "per-run epsilon is not a composed privacy guarantee for the sweep or retries; SparseGNN uses the repository's fixed chi=2 mixture formula, not an independently established guarantee",
    "gpu_ownership": "GPUs 4-7 only; at most two jobs/GPU, sharing only after a successful full-lifecycle memory profile; cooperative leases cannot reserve against unrelated users",
    "scope": "280 initial configurations only; adaptive rounds disabled; historical missingness-target FB results are not comparable",
}


def protocol(name: str) -> tuple[str, dict | None]:
    if name not in PROTOCOLS:
        raise ValueError(f"unknown ideation protocol: {name}")
    if name == "coauthor-physics":
        return name, None
    from src.data.domain_datasets import FB100_DOMAINS

    count = int(name.rsplit("-", 1)[1])
    schools = (list(TRAIN_SCHOOLS[:count]) if count < 16 else
               [school for school in FB100_DOMAINS if school not in {"cornell5", "penn94"}])
    return "facebook100-gender", {"train": schools, "val": ["cornell5"],
                                  "test": ["penn94"], "seed": 0, "val_ratio": 0.2}


def grid_rows() -> list[dict]:
    from scripts import full_matrix_records as records

    grouped = defaultdict(list)
    for name in PROTOCOLS:
        for method in METHODS:
            for batch in (256, 1024):
                for lr in (0.01, 0.001):
                    for epsilon in (2, 8):
                        for p2 in ((0.1, 0.5, 1.0) if method.startswith("sparse_") else (None,)):
                            row = {
                                "protocol": name, "method": method, "batch_size": batch,
                                "lr": lr, "epsilon": epsilon, "p2": p2, "seed": 0,
                                "epochs": 20, "dropout": 0.5, "mlp_hidden": 64,
                                "gnn_hidden": 128, "degree_bound": 10,
                                "bootstrap_resamples": 0, "r": 2 if method == "progap" else 1,
                                "gin_pooling": "mean" if method == "sparse_gin" else "sum",
                            }
                            command = [records.MAIN_PYTHON, str(REPO_ROOT / "scripts/full_matrix_run.py")]
                            for key in ("protocol", "method", "batch_size", "lr", "epsilon", "p2",
                                        "seed", "epochs", "dropout", "mlp_hidden", "gnn_hidden",
                                        "degree_bound", "bootstrap_resamples", "gin_pooling"):
                                if row[key] is not None:
                                    command += ["--dataset" if key == "protocol" else "--" + key.replace("_", "-"), str(row[key])]
                            command += ["--device", "cuda", "--out-dir", str(REPO_ROOT / "__ideation_only__")]
                            if method == "progap":
                                command += ["--progap-depth", str(row["r"]),
                                            "--progap-python", records.PROGAP_PYTHON]
                            row["argv"] = command
                            grouped[name].append(row)
    return [grouped[name][index] for index in range(56) for name in PROTOCOLS]


def build_requests(prepared: dict, fingerprint: str) -> list[dict]:
    from scripts import full_matrix_records as records

    records._same(set(prepared), set(PROTOCOLS), "ideation prepared protocols")
    return [records._make_request(row, prepared[row["protocol"]], fingerprint, ordinal)
            for ordinal, row in enumerate(grid_rows())]


def select_by_validation(rows: list[dict], results: list[dict]) -> list[dict]:
    """Select one single-seed configuration per cell without consulting test scores."""
    by_source = {result["result_csv"]: result for result in results if result["accepted"]}
    chosen = {}
    for row in rows:
        result = by_source[row["sources"]]
        key = (result["protocol"], result["method"], result["epsilon"])
        rank = (-float(result["validation_metric"]), result["ordinal"])
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, {**row, "selection": "best_validation"})
    return [value[1] for value in chosen.values()]


def main() -> int:
    from scripts import full_matrix_records as records

    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    for name in ("prepare", "run", "report", "verify", "_prepare-protocol"):
        command = commands.add_parser(name)
        command.add_argument("--out-root", required=True, type=Path)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--gpus", default="4,5,6,7")
            command.add_argument("--max-jobs-per-gpu", type=int, choices=(1, 2), default=2)
        elif name == "_prepare-protocol":
            command.add_argument("--dataset", required=True, choices=PROTOCOLS)
    args = cli.parse_args()
    if args.command == "dry-run":
        from collections import Counter
        rows = grid_rows()
        print(json.dumps({"spec": SPEC, "count": len(rows),
                          "protocol_counts": dict(Counter(row["protocol"] for row in rows)),
                          "method_counts": dict(Counter(row["method"] for row in rows))}, indent=2))
    elif args.command == "prepare":
        records.prepare_campaign(args.out_root, purpose="ideation")
        print(f"Prepared 280 immutable initial-screen requests: {args.out_root}", flush=True)
    elif args.command == "_prepare-protocol":
        records.prepare_protocol(args.dataset, args.out_root)
    elif args.command == "run":
        from scripts.full_matrix_campaign import CampaignQueue
        from scripts.full_matrix_runtime import resolve_gpus
        gpus = resolve_gpus(args.gpus)
        if not gpus or any(int(gpu["index"]) not in SPEC["physical_gpu_indices"] for gpu in gpus):
            raise ValueError("this approved study may use physical GPUs 4-7 only")
        return CampaignQueue(args.out_root, purpose="ideation", gpus=args.gpus,
                             resume=args.resume, max_jobs_per_gpu=args.max_jobs_per_gpu).run()
    elif args.command == "report":
        print(json.dumps(records.write_reports(args.out_root), indent=2))
    else:
        result = records.verify_campaign(args.out_root)
        print(json.dumps(result, indent=2))
        return 2 if result.get("corrupt") else 0 if result.get("fully_executed") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
