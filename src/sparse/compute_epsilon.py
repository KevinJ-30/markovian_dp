"""Attach Theorem 5.4 substitution epsilon to SparseGNN result rows.

The accountant always uses the in-expansion shell law.  A row's training
``direction`` remains provenance and is intentionally not used to select or
validate a different guarantee.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.sparse.accounting import (  # noqa: E402
    naive_opacus_epsilon, sparsegnn_epsilon_schedule)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--delta", required=True, type=float)
    parser.add_argument(
        "--grid", default=1e-4, type=float,
        help="dp_accounting privacy-loss discretization interval")
    parser.add_argument(
        "--legacy_shells", action="store_true",
        help="drop the union-graph correction for reproducing older numbers")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.delta < 1:
        raise SystemExit(f"--delta must lie in (0, 1), got {args.delta}")
    if args.grid <= 0:
        raise SystemExit(f"--grid must be positive, got {args.grid}")
    out_path = args.out or args.csv.replace(".csv", "_with_eps.csv")

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    def row_key(row):
        for field in ("K_in", "K_out"):
            if not row.get(field):
                raise SystemExit(
                    f"CSV has no {field}; a finite degree bound is required")
        return (
            float(row["p1"]), float(row["p2"]), int(row["r"]),
            float(row["sigma"]), int(row["T"]), int(row["K_in"]),
            int(row["K_out"]),
        )

    def row_step(row):
        value = row.get("step")
        return int(float(value)) if value not in (None, "") else int(row["T"])

    steps_by_key = {}
    for row in rows:
        steps_by_key.setdefault(row_key(row), set()).add(row_step(row))

    epsilon_cache = {}
    naive_cache = {}
    for key, checkpoints in steps_by_key.items():
        p1, p2, radius, sigma, total_steps, k_in, k_out = key
        schedule = sparsegnn_epsilon_schedule(
            p1=p1, p2=p2, r=radius, K_in=k_in, K_out=k_out,
            sigma=sigma, steps=checkpoints, delta=args.delta, grid=args.grid,
            union_safe=not args.legacy_shells)
        for step in checkpoints:
            epsilon_cache[(key, step)] = schedule[step]
            try:
                naive_cache[(key, step)] = naive_opacus_epsilon(
                    sigma, p1, step, args.delta, mechanism="prv")
            except Exception:
                naive_cache[(key, step)] = float("nan")
        final = max(checkpoints)
        print(
            f"p1={p1} p2={p2} r={radius} sigma={sigma} T={total_steps} "
            f"K_in={k_in} K_out={k_out}: "
            f"epsilon={schedule[final]:.4f} "
            f"naive={naive_cache[(key, final)]:.4f}")

    obsolete = {"epsilon_theorem", "epsilon_substitution", "epsilon_thm4"}
    for row in rows:
        key, step = row_key(row), row_step(row)
        for field in obsolete:
            row.pop(field, None)
        row["step"] = step
        row["epsilon"] = f"{epsilon_cache[(key, step)]:.5f}"
        row["epsilon_naive_opacus"] = f"{naive_cache[(key, step)]:.5f}"
        row["delta"] = f"{args.delta:g}"
        row["epsilon_grid"] = f"{args.grid:g}"
        row["union_safe_shells"] = str(not args.legacy_shells)

    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
