"""Does the DP training engine compute the same thing on a GPU as on a CPU?

`tests/test_vectorized.py` pins `_step_dp_vectorized` against the per-root loop
in `_step_dp`, which is the reference implementation of the mechanism the
accountant prices.  Every one of those tests runs on the CPU.  This script runs
the same comparison ACROSS DEVICES, which is the question that matters before
trusting a GPU sweep: if the released weights came from a CUDA run and the
epsilon describes the CPU loop, the reported guarantee is about a different
mechanism than the one that produced the numbers.

Four arms, all seeded identically:

    cpu  / loop        the reference
    cpu  / vectorized  pinned by the test suite already
    cuda / loop        isolates "does CUDA change the reference?"
    cuda / vectorized  what a GPU sweep would actually run

Everything stochastic is device-independent BY CONSTRUCTION, so the arms are
genuinely comparable rather than merely similar:

  * root sampling and expansion use CPU generators over a CPU adjacency
    (`sparse_expand.py` forces `.cpu()`), so every arm sees the same subgraphs;
  * `gaussian_noise_like` draws from a CPU generator and only then moves the
    draw to the parameter's device, so every arm sees the same noise;
  * the module is built on the CPU under a fixed seed before `.to(device)`.

What remains is floating-point arithmetic, so the arms will NOT agree bitwise.
cuBLAS picks different reduction orders and tensor-core kernels than CPU BLAS.
The script therefore reports the worst observed deviation and checks it against
a tolerance rather than demanding equality -- and prints the number either way,
because "how far apart" is the thing you actually want to know.

    python scripts/check_device_equivalence.py
    python scripts/check_device_equivalence.py --steps 60 --hidden 32
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.sparse.multilabel_mechanism import MultiLabelGNNMechanism  # noqa: E402
from src.sparse.sparse_expand import build_adjacency                # noqa: E402
from src.sparse.sparse_gnn import train_sparse_gnn                  # noqa: E402


class _Data:
    """Minimal PyG-shaped Data the mechanisms accept, with a .to(device)."""

    def __init__(self, x, y, edge_index):
        self.x, self.y, self.edge_index = x, y, edge_index
        n = x.shape[0]
        self.num_nodes = n
        self.train_mask = torch.ones(n, dtype=torch.bool)
        self.val_mask = torch.zeros(n, dtype=torch.bool)
        self.test_mask = torch.zeros(n, dtype=torch.bool)
        # Give val/test something to score so `evaluate` is exercised too.
        self.val_mask[: n // 4] = True
        self.test_mask[n // 4: n // 2] = True

    def to(self, device):
        out = object.__new__(_Data)
        out.__dict__ = {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in self.__dict__.items()}
        return out


def _graph(n, f, c, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (n * 6,), generator=g)
    dst = torch.randint(0, n, (n * 6,), generator=g)
    keep = src != dst
    x = torch.randn(n, f, generator=g)
    y = (torch.rand(n, c, generator=g) > 0.5).float()
    return _Data(x, y, torch.stack([src[keep], dst[keep]]))


def run_arm(cpu_data, adj, device, vectorized, args):
    """Train one arm and return its final parameters, on the CPU for comparison."""
    torch.manual_seed(0)
    data = cpu_data.to(device)
    mech = MultiLabelGNNMechanism(
        data, args.features, args.classes, hidden=args.hidden,
        num_layers=2, dropout=0.0, device=torch.device(device))
    mech.build_optimizer(lr=0.05)
    train_sparse_gnn(
        mech, data, adj=adj, direction='in',
        p1=args.p1, p2=0.5, r=2, T=args.steps,
        candidate_nodes=torch.arange(args.nodes),
        dp=True, clip=0.1, sigma=1.5, seed=0,
        vectorized=vectorized)
    return [p.detach().cpu().clone() for p in mech.parameters()]


def compare(label, a, b, tol):
    worst = max(float((x - y).abs().max()) for x, y in zip(a, b))
    scale = max(float(x.abs().max()) for x in a) or 1.0
    status = 'OK  ' if worst <= tol else 'FAIL'
    print(f"  [{status}] {label:<34} max|diff| = {worst:.3e}  "
          f"(rel {worst / scale:.3e})")
    return worst <= tol


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nodes', type=int, default=400)
    p.add_argument('--features', type=int, default=16)
    p.add_argument('--classes', type=int, default=5)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--p1', type=float, default=0.25)
    p.add_argument('--tol', type=float, default=1e-4,
                   help='max absolute weight deviation treated as agreement')
    args = p.parse_args()

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"graph: {args.nodes} nodes, {args.features} feat, {args.classes} labels; "
          f"T={args.steps} hidden={args.hidden} p1={args.p1}\n")

    cpu_data = _graph(args.nodes, args.features, args.classes)
    adj = build_adjacency(cpu_data.edge_index, args.nodes)

    print("running cpu/loop (reference) ...")
    cpu_loop = run_arm(cpu_data, adj, 'cpu', False, args)
    print("running cpu/vectorized ...")
    cpu_vec = run_arm(cpu_data, adj, 'cpu', True, args)

    ok = [compare('cpu/loop  vs cpu/vectorized', cpu_loop, cpu_vec, args.tol)]

    if not torch.cuda.is_available():
        print("\nNo CUDA device visible -- the cross-device arms were SKIPPED.\n"
              "This script only proves something when run on a GPU node.")
        return 0 if all(ok) else 1

    print("running cuda/loop ...")
    gpu_loop = run_arm(cpu_data, adj, 'cuda', False, args)
    print("running cuda/vectorized ...")
    gpu_vec = run_arm(cpu_data, adj, 'cuda', True, args)

    ok.append(compare('cuda/loop vs cuda/vectorized', gpu_loop, gpu_vec, args.tol))
    ok.append(compare('cpu/loop  vs cuda/loop', cpu_loop, gpu_loop, args.tol))
    # The one that decides whether a GPU sweep is trustworthy: what a GPU run
    # actually executes, against the reference the accountant is written for.
    ok.append(compare('cpu/loop  vs cuda/vectorized', cpu_loop, gpu_vec, args.tol))

    print()
    if all(ok):
        print("PASS -- the GPU path reproduces the reference loop within tolerance.")
        return 0
    print("FAIL -- a GPU sweep would NOT be computing the priced mechanism.\n"
          "Do not trust GPU results until this is explained.  If the deviation is\n"
          "only slightly over tolerance, re-run with --tol to see whether it is\n"
          "drift or a genuine disagreement; if it grows with --steps, it is drift.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
