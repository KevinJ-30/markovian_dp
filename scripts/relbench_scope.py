"""Scope a RelBench task: is its epsilon affordable before we spend a night on it?

rel-trial showed the failure mode.  Its FK graph has K_out=3, which under
in-expansion should be cheap (shells are n_d = K_out^d), but it has only ~1.4k
training rows, so p1=0.05 was needed for a usable batch -- and epsilon is
superlinear in p1.  The result was a floor of eps~16 on the dense arm, with the
only sub-eps-2 point scoring 0.5531 AUROC against a 0.5 chance baseline.

The lever is the training-row count: p1 = batch / n_train, so a task with many
labelled rows gets a small p1 for free, and small p1 is what makes epsilon
affordable.  This script reports the counts and then prices the grid, so the
go/no-go is made before any training runs.

    python scripts/relbench_scope.py --dataset rel-amazon --task user-churn

Prints a human summary to stderr and KEY=VALUE lines to stdout, so a driver can

    eval "$(python scripts/relbench_scope.py --dataset rel-amazon --task user-churn)"

exporting SCOPE_P1, SCOPE_N_NODES, SCOPE_N_TRAIN, SCOPE_K_IN, SCOPE_K_OUT.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np                                            # noqa: E402

from src.datasets import load_dataset                         # noqa: E402
from src.sparse.accounting import calibrate_sparsegnn_noise   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='rel-amazon')
    p.add_argument('--task', default='user-churn')
    p.add_argument('--batch', type=int, default=512,
                   help='target roots per step; p1 = batch / n_train')
    p.add_argument('--T', type=int, default=500)
    p.add_argument('--eps', type=float, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--r', type=int, default=2,
                   help='a root is a prediction row: r=1 reaches its entity, '
                        'r=2 its history')
    p.add_argument('--grid', type=float, default=1e-5)
    return p.parse_args()


def main():
    a = parse_args()
    name = f'relbench:{a.dataset}/{a.task}'
    print(f"loading {name} (RelBench databases are multi-GB; first run "
          f"downloads)", file=sys.stderr)
    _, data = load_dataset(name)

    n = int(data.num_nodes)
    ei = data.edge_index
    n_edges = int(ei.size(1))
    train_mask = getattr(data, 'train_mask', None)
    n_train = int(train_mask.sum()) if train_mask is not None else n

    out_deg = np.bincount(ei[0].cpu().numpy(), minlength=n)
    in_deg = np.bincount(ei[1].cpu().numpy(), minlength=n)
    q = lambda d, p: int(np.percentile(d, p))

    p1 = min(1.0, a.batch / max(1, n_train))
    # K_out prices epsilon under in-expansion (shells n_d = K_out^d), so read it
    # off the graph rather than guessing: the 99th percentile keeps the cap from
    # being set by a single hub.
    k_out = max(1, q(out_deg, 99))
    k_in = max(1, q(in_deg, 99))

    w = sys.stderr
    print(f"\n=== {name} ===", file=w)
    print(f"  nodes {n:,}   edges {n_edges:,}   train rows {n_train:,}", file=w)
    print(f"  out-degree  mean {out_deg.mean():6.2f}  med {q(out_deg,50):4d}  "
          f"p99 {q(out_deg,99):5d}  max {out_deg.max():,}", file=w)
    print(f"  in-degree   mean {in_deg.mean():6.2f}  med {q(in_deg,50):4d}  "
          f"p99 {q(in_deg,99):5d}  max {in_deg.max():,}", file=w)
    print(f"  delta = n^-1.01 = {float(n) ** -1.01:.3g}", file=w)
    print(f"  p1 = {a.batch}/{n_train:,} = {p1:.6f}", file=w)
    print(f"  suggested cap K_in={k_in} K_out={k_out} (p99)", file=w)

    delta = float(n) ** -1.01
    print(f"\n  sigma needed at r={a.r}, T={a.T}, K_out={k_out}:", file=w)
    for p2 in (1.0, 0.1):
        for eps in a.eps:
            try:
                c = calibrate_sparsegnn_noise(
                    target_epsilon=eps, target_delta=delta, p1=p1, p2=p2,
                    r=a.r, K_in=k_in, K_out=k_out, steps=a.T, clip=1.0,
                    direction='in', grid=a.grid)
                print(f"    p2={p2:<5} eps={eps:<4} -> sigma {c.noise_multiplier:9.3f}",
                      file=w)
            except (RuntimeError, ValueError) as exc:
                print(f"    p2={p2:<5} eps={eps:<4} -> UNREACHABLE ({exc})",
                      file=w)

    print(f"SCOPE_N_NODES={n}")
    print(f"SCOPE_N_TRAIN={n_train}")
    print(f"SCOPE_P1={p1:.6f}")
    print(f"SCOPE_K_IN={k_in}")
    print(f"SCOPE_K_OUT={k_out}")


if __name__ == '__main__':
    main()
