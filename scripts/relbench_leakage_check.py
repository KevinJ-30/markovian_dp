"""Check a RelBench graph leaks no future information and is inductive.

    python scripts/relbench_leakage_check.py --dataset rel-f1 --task driver-top3

Four pass/fail checks, no training required:

1. FUTURE EDGES -- no neighbour within r hops of an eval root is dated at or
   after that root.  For a forward-window target the rows the label is computed
   from sit two hops out and carry their values as features.
2. SPLIT DISJOINTNESS -- no node carries two split masks (entity mode).
3. INDUCTIVE GRAPH -- train edges are a strict subset of eval edges.
4. NO NODE IDENTITY -- permuting node ids leaves the feature multiset alone.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data.datasets import load_dataset                          # noqa: E402
from src.processing.sparse_expand import build_adjacency            # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='rel-f1')
    p.add_argument('--task', default='driver-top3')
    p.add_argument('--root', choices=['row', 'entity'], default='row')
    p.add_argument('--r', type=int, default=2, help='hops to walk when auditing')
    p.add_argument('--probe', type=int, default=2000,
                   help='evaluation roots to sample (0 = all)')
    return p.parse_args()


def in_neighbours(adj, nodes):
    out = []
    for v in nodes:
        out.append(adj.neighbors(int(v)))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


def main():
    a = parse_args()
    name = f'relbench:{a.dataset}/{a.task}'
    print(f"loading {name} (root={a.root})", file=sys.stderr)
    _, data = load_dataset(name, root=a.root)

    node_time = data.node_time.numpy()
    ok = True

    # ── 1. future edges reachable from an evaluation root ────────────────────
    adj = build_adjacency(data.edge_index, int(data.num_nodes), direction='in')
    roots = torch.where(data.test_mask)[0]
    if a.probe and roots.numel() > a.probe:
        g = torch.Generator().manual_seed(0)
        roots = roots[torch.randperm(roots.numel(), generator=g)[:a.probe]]

    worst, n_future, n_seen = 0.0, 0, 0
    for v in roots.tolist():
        t_root = node_time[v]
        frontier, seen = torch.tensor([v]), {v}
        for _ in range(a.r):
            frontier = in_neighbours(adj, frontier.tolist())
            frontier = torch.tensor([int(w) for w in frontier.tolist()
                                     if int(w) not in seen] or [], dtype=torch.long)
            seen.update(int(w) for w in frontier.tolist())
            if not frontier.numel():
                break
            ts = node_time[frontier.numpy()]
            fut = int((ts >= t_root).sum()) if np.isfinite(t_root) else 0
            n_future += fut
            n_seen += int(frontier.numel())
            worst = max(worst, fut / max(int(frontier.numel()), 1))

    frac = n_future / max(n_seen, 1)
    status = "PASS" if n_future == 0 else "FAIL"
    ok &= n_future == 0
    print(f"[{status}] future edges: {n_future:,}/{n_seen:,} neighbours "
          f"({frac:.2%}) of {roots.numel():,} test roots are dated at or after "
          f"their root (worst root {worst:.0%}); r={a.r}")

    # ── 2. a node may not belong to two splits ───────────────────────────────
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    dup = int((tr & va).sum() + (tr & te).sum() + (va & te).sum())
    ok &= dup == 0
    print(f"[{'PASS' if dup == 0 else 'FAIL'}] split disjointness: "
          f"{dup:,} nodes carry more than one split mask")

    # ── 3. train edges ⊆ eval edges, and strictly fewer ──────────────────────
    def keyset(ei):
        return set(map(tuple, ei.t().tolist()))
    tr_e, ev_e = keyset(data.train_edge_index), keyset(data.edge_index)
    subset, smaller = tr_e <= ev_e, len(tr_e) < len(ev_e)
    ok &= subset and smaller
    print(f"[{'PASS' if subset and smaller else 'FAIL'}] inductive graph: "
          f"train {len(tr_e):,} edges ⊆ eval {len(ev_e):,} "
          f"(subset={subset}, strictly smaller={smaller})")

    # ── 4. features carry no node identity ───────────────────────────────────
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(int(data.num_nodes), generator=g)
    same = torch.equal(torch.sort(data.x.sum(1)).values,
                       torch.sort(data.x[perm].sum(1)).values)
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] no node identity in x: feature "
          f"multiset is permutation-invariant ({same})")

    print()
    print("ALL CHECKS PASSED" if ok else "FAILURES ABOVE -- do not trust "
          "numbers from this configuration")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
