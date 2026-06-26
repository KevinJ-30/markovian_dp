"""
Privacy accounting for the DP-SGD GNN runs in this folder (run.py modes
`dp` and `dp_subsample`).

Two accountants are available:

1. Opacus (default, `opacus-rdp` / `opacus-prv`): standard subsampled-Gaussian
   accounting. Each step is treated as a Gaussian mechanism with noise
   multiplier sigma applied to a Poisson-subsampled batch with sampling rate
   q = batch_size / num_train. (The sampler in run.py actually shuffles and
   partitions, but reporting shuffled batches under Poisson amplification is
   the standard convention used by opacus and most DP-SGD papers.)

2. Dominating pair (`dominating-pair`): if you have a custom per-step
   dominating pair (P, Q) for your mechanism — e.g. from a Markovian
   subsampling analysis — supply it as a JSON file. The pair is interpreted
   as dominating ONE training step; this module composes it over all steps
   by convolving the discretized privacy loss distribution (PLD).
   Discretization uses pessimistic (round-up) rounding and truncated tail
   mass is pushed to +infinity, so the reported epsilon is an upper bound.

Dominating pair JSON format — discrete distributions over shared atoms:

    {
      "p": [0.75, 0.25],
      "q": [0.25, 0.75]
    }

or equivalently {"atoms": [{"p": 0.75, "q": 0.25}, ...]}. Atom i has mass
p[i] under P and q[i] under Q; both must sum to 1. Atoms with q[i] == 0 and
p[i] > 0 carry infinite privacy loss and feed straight into delta.
delta(eps) is computed in the P-vs-Q direction (hockey-stick divergence
H_{e^eps}(P || Q)), the usual convention when (P, Q) dominates the mechanism
for the worst-case adjacency direction.

CLI examples:

  # epsilon for 300 steps of subsampled Gaussian via opacus
  python accounting.py --accountant opacus-rdp --sigma 1.0 \
      --sample_rate 0.45 --steps 300 --delta 1e-5

  # epsilon for 300 compositions of a custom dominating pair
  python accounting.py --accountant dominating-pair \
      --dominating_pair example_dominating_pair.json --steps 300 --delta 1e-5
"""

import argparse
import json
import math

import numpy as np

try:
    from scipy.signal import fftconvolve as _fftconvolve

    def _convolve(a, b):
        # fft round-off can produce tiny negatives; clamp to keep pmfs valid
        return np.clip(_fftconvolve(a, b), 0.0, None)
except ImportError:
    def _convolve(a, b):
        return np.convolve(a, b)


class PrivacyLossDistribution:
    """Discrete privacy loss distribution on a uniform grid.

    pmf[i] is the probability under P that the privacy loss ln(dP/dQ)
    equals (offset + i) * grid. pmf may sum to less than 1: the deficit
    1 - pmf.sum() is treated as mass at loss = +infinity (genuine infinity
    atoms where q == 0, plus any tail mass dropped during truncation), which
    transfers directly into delta — i.e. it is always pessimistic.
    """

    def __init__(self, pmf, offset, grid):
        self.pmf = np.asarray(pmf, dtype=float)
        self.offset = int(offset)
        self.grid = float(grid)

    @property
    def inf_mass(self):
        return max(0.0, 1.0 - float(self.pmf.sum()))

    @classmethod
    def from_dominating_pair(cls, p, q, grid):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if grid <= 0:
            raise ValueError("grid must be positive")
        if p.shape != q.shape:
            raise ValueError("p and q must have the same length")
        if (p < 0).any() or (q < 0).any():
            raise ValueError("p and q must be non-negative")
        for name, arr in (("p", p), ("q", q)):
            if abs(arr.sum() - 1.0) > 1e-6:
                raise ValueError(f"{name} must sum to 1 (got {arr.sum():.8f})")

        finite = (p > 0) & (q > 0)
        losses = np.log(p[finite] / q[finite])
        # Pessimistic discretization: rounding losses up can only increase
        # delta(eps), so the final epsilon is an upper bound (the slack is at
        # most grid per composed step).
        ks = np.ceil(losses / grid - 1e-12).astype(np.int64)
        offset = int(ks.min())
        pmf = np.zeros(int(ks.max()) - offset + 1)
        np.add.at(pmf, ks - offset, p[finite])
        # mass with q == 0 and p > 0 is loss = +inf; it is simply absent from
        # pmf and accounted for through inf_mass.
        return cls(pmf, offset, grid)

    def _truncate(self, tail_mass):
        """Drop up to `tail_mass` total probability from the two tails.
        Dropped mass implicitly moves to +infinity (pessimistic)."""
        if tail_mass <= 0 or self.pmf.size <= 2:
            return self
        csum = np.cumsum(self.pmf)
        total = csum[-1]
        lo = int(np.searchsorted(csum, tail_mass / 2))
        hi = int(np.searchsorted(csum, total - tail_mass / 2, side="right")) + 1
        hi = min(max(hi, lo + 1), self.pmf.size)
        return PrivacyLossDistribution(self.pmf[lo:hi], self.offset + lo, self.grid)

    def compose(self, other, tail_mass=0.0):
        if other.grid != self.grid:
            raise ValueError("grids must match")
        pmf = _convolve(self.pmf, other.pmf)
        out = PrivacyLossDistribution(pmf, self.offset + other.offset, self.grid)
        return out._truncate(tail_mass)

    def self_compose(self, n, tail_mass_total=1e-10):
        """n-fold composition via exponentiation by squaring."""
        if n < 1:
            raise ValueError("n must be >= 1")
        if n == 1:
            return self
        budget = tail_mass_total / (2 * math.ceil(math.log2(n)))
        result = None
        base = self
        m = n
        while m > 0:
            if m & 1:
                result = base if result is None else result.compose(base, budget)
            m >>= 1
            if m > 0:
                base = base.compose(base, budget)
        assert result is not None
        return result

    def get_delta(self, eps):
        losses = (self.offset + np.arange(self.pmf.size)) * self.grid
        finite = float(np.sum(self.pmf * np.clip(-np.expm1(eps - losses), 0.0, None)))
        return self.inf_mass + finite

    def get_epsilon(self, delta):
        if self.inf_mass >= delta:
            return math.inf
        if self.get_delta(0.0) <= delta:
            return 0.0
        lo = 0.0
        # delta(eps) hits inf_mass (< delta) once eps exceeds the max loss
        hi = max((self.offset + self.pmf.size - 1) * self.grid, 1.0)
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if self.get_delta(mid) > delta:
                lo = mid
            else:
                hi = mid
        return hi


def load_dominating_pair(path):
    with open(path) as f:
        spec = json.load(f)
    if "atoms" in spec:
        p = [a["p"] for a in spec["atoms"]]
        q = [a["q"] for a in spec["atoms"]]
    else:
        p, q = spec["p"], spec["q"]
    return p, q


def dominating_pair_epsilon(path, steps, delta, grid=1e-4):
    p, q = load_dominating_pair(path)
    pld = PrivacyLossDistribution.from_dominating_pair(p, q, grid)
    return pld.self_compose(steps).get_epsilon(delta)


def opacus_epsilon(noise_multiplier, sample_rate, steps, delta, mechanism="rdp"):
    try:
        from opacus.accountants import create_accountant
    except ImportError as e:
        raise ImportError(
            "opacus is required for the opacus-* accountants: pip install opacus"
        ) from e
    accountant = create_accountant(mechanism=mechanism)
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return accountant.get_epsilon(delta=delta)


def compute_epsilon(accountant, *, noise_multiplier=None, sample_rate=None,
                    steps=None, delta=None, dominating_pair=None, grid=1e-4):
    """Single entry point used by run.py."""
    if accountant == "dominating-pair":
        if not dominating_pair:
            raise ValueError("a --dominating_pair JSON file is required for "
                             "the dominating-pair accountant")
        return dominating_pair_epsilon(dominating_pair, steps, delta, grid)
    if accountant in ("opacus-rdp", "opacus-prv"):
        return opacus_epsilon(noise_multiplier, sample_rate, steps, delta,
                              mechanism=accountant.split("-", 1)[1])
    raise ValueError(f"unknown accountant: {accountant}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accountant",
                   choices=["opacus-rdp", "opacus-prv", "dominating-pair"],
                   default="opacus-rdp")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=None,
                   help="noise multiplier (opacus accountants)")
    p.add_argument("--sample_rate", type=float, default=None,
                   help="per-step sampling rate q (opacus accountants)")
    p.add_argument("--dominating_pair", type=str, default=None,
                   help="JSON file with the per-step dominating pair")
    p.add_argument("--grid", type=float, default=1e-4,
                   help="PLD loss discretization (dominating-pair accountant)")
    args = p.parse_args()

    eps = compute_epsilon(
        args.accountant,
        noise_multiplier=args.sigma,
        sample_rate=args.sample_rate,
        steps=args.steps,
        delta=args.delta,
        dominating_pair=args.dominating_pair,
        grid=args.grid,
    )
    print(f"accountant={args.accountant}  steps={args.steps}  "
          f"delta={args.delta:g}  epsilon={eps:.4f}")


if __name__ == "__main__":
    main()
