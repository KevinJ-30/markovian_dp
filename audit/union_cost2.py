"""Union-graph accounting: four variants side by side.

v42 Assumption 5.2 bounds H = g u g', not g. run.py caps g. This prices the gap.

KEY STRUCTURAL FACT (v42 Def 5.1): E(triangle)E' is contained in ({s}xV) u (Vx{s}).
So for u != s and w != s, the arc (u,w) is in E iff it is in E'. Every arc of H
between two non-s vertices is a common arc. The union can ONLY inflate degree
at s (factor 2) or via a single extra arc incident to s.

Shortest paths from s never revisit s, so intermediate steps see the
un-inflated degree:

  n_d  = |{v : d_H(s,v)=d}| : first step out of s <= 2K_out, each later step
         to a non-s target <= K_out            =>  n_d <= 2*K_out^d
  |P^(l)_{s,v}| : forward trace 2*K_out^(l-1); backward trace from v visits only
         non-s vertices, so K_in^(l-1) with NO factor of 2
                                               =>  P_l <= min(2*K_out^(l-1), K_in^(l-1))

When K_in <= 2*K_out (every shipped config except relbench) the path bound and
hence q_d are UNCHANGED; only n_d doubles.
"""
import sys, math
import numpy as np
sys.path.insert(0, '/Users/kevinjacob/markovian_dp copy')
from src.sparse.accounting import (
    sparsegnn_substitution_epsilon, _binom_pmf,
    _substitution_pld_from_weights)

GRID, NS, APS = 1e-5, 10.0, 400.0

def _pi(n, q, p1):
    pi = np.array([1.0])
    for d in range(len(n)):
        pi = np.convolve(pi, _binom_pmf(n[d], p1 * q[d]))
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()

def _q(p2, r, paths):
    q = [1.0]
    for d in range(1, r + 1):
        if p2 >= 1.0:
            q.append(1.0); continue
        q.append(1.0 - math.exp(sum(paths(l) * math.log1p(-(p2 ** l))
                                    for l in range(d, r + 1))))
    return q

def variant(kind, p1, p2, r, K_in, K_out):
    """Return (n, q) for one accounting variant."""
    if kind == 'current':                      # what the code does today
        n = [1] + [K_out ** d for d in range(1, r + 1)]
        K = min(K_in, K_out)
        paths = lambda l: K ** (l - 1)
    elif kind == 'tight':                      # union-safe, exploiting Def 5.1
        n = [1] + [2 * K_out ** d for d in range(1, r + 1)]
        paths = lambda l: min(2 * K_out ** (l - 1), K_in ** (l - 1))
    elif kind == 'loose':                      # union-safe, degree+1 at every step
        n = [1] + [2 * K_out * (K_out + 1) ** (d - 1) for d in range(1, r + 1)]
        paths = lambda l: (1 if l <= 1 else
                           min(2 * K_out * (K_out + 1) ** (l - 2), (K_in + 1) ** (l - 1)))
    elif kind == 'naive':                      # K -> 2K everywhere
        n = [1] + [(2 * K_out) ** d for d in range(1, r + 1)]
        K = min(2 * K_in, 2 * K_out)
        paths = lambda l: K ** (l - 1)
    return n, _q(p2, r, paths)

def eps(kind, p1, p2, r, K_in, K_out, sigma, T, delta):
    n, q = variant(kind, p1, p2, r, K_in, K_out)
    pld = _substitution_pld_from_weights(_pi(n, q, p1), sigma, GRID, NS, APS)
    return pld.self_compose(T).get_epsilon_for_delta(delta), n

CELLS = [
 ("facebook headline   p1=.013  p2=1.0 r=1 K=5/5    s=5.00   T=500  d=1e-6",
  dict(p1=0.013, p2=1.0, r=1, K_in=5, K_out=5, sigma=5.0, T=500, delta=1e-6)),
 ("facebook            p1=.013  p2=1.0 r=2 K=5/5    s=5.00   T=500  d=1e-6",
  dict(p1=0.013, p2=1.0, r=2, K_in=5, K_out=5, sigma=5.0, T=500, delta=1e-6)),
 ("PPI matched-eps=1   p1=.0114 p2=1.0 r=1 K=5/5    s=45.44  T=2000 d=1.574e-5",
  dict(p1=0.0114, p2=1.0, r=1, K_in=5, K_out=5, sigma=45.4375, T=2000, delta=1.574e-5)),
 ("PPI matched-eps=1   p1=.0114 p2=0.1 r=1 K=5/5    s=11.36  T=2000 d=1.574e-5",
  dict(p1=0.0114, p2=0.1, r=1, K_in=5, K_out=5, sigma=11.359375, T=2000, delta=1.574e-5)),
 ("relbench            p1=.05   p2=1.0 r=2 K=20/3   s=5.00   T=900  d=1e-6",
  dict(p1=0.05, p2=1.0, r=2, K_in=20, K_out=3, sigma=5.0, T=900, delta=1e-6)),
]

print(f"{'cell':<70}{'current':>10}{'TIGHT':>10}{'ratio':>8}{'loose':>10}{'naive':>10}")
print("-" * 118)
for name, c in CELLS:
    a = {k: c[k] for k in ('p1','p2','r','K_in','K_out','sigma','T','delta')}
    e0, n0 = eps('current', **a)
    e1, n1 = eps('tight',   **a)
    e2, _  = eps('loose',   **a)
    e3, _  = eps('naive',   **a)
    print(f"{name:<70}{e0:>10.4f}{e1:>10.4f}{e1/e0:>7.2f}x{e2:>10.4f}{e3:>10.4f}")
    print(f"{'   n_d:':<70}{str(n0[1:]):>10}{str(n1[1:]):>28}")
