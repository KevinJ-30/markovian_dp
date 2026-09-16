import sys, math
import numpy as np
sys.path.insert(0, '/Users/kevinjacob/markovian_dp copy')
from src.sparse.accounting import (
    sparsegnn_epsilon, _binom_pmf, mixture_gaussian_pld)

GRID = 1e-5

def refined_pi(p1, p2, r, K_in, K_out):
    """Mixture weights under the UNION degree bounds implied by capping g at K.

    H = g u g' with E(+)E' inside ({s}xV) u (Vx{s}):
      deg_out^H(s)   <= 2*K_out      deg_out^H(u!=s) <= K_out+1
      deg_in^H(s)    <= 2*K_in       deg_in^H(v!=s)  <= K_in+1
    Lemma 5 recounted with those:
      n_d = 2*K_out*(K_out+1)^(d-1),  d>=1
      |P^(l)_{s,v}| <= min( 2*K_out*(K_out+1)^(l-2), (K_in+1)^(l-1) ), l>=2; =1 at l=1
    """
    n = [1] + [2 * K_out * (K_out + 1) ** (d - 1) for d in range(1, r + 1)]
    def paths(l):
        if l <= 1: return 1
        return min(2 * K_out * (K_out + 1) ** (l - 2), (K_in + 1) ** (l - 1))
    q = [1.0]
    for d in range(1, r + 1):
        if p2 >= 1.0:
            q.append(1.0); continue
        log_keep = sum(paths(l) * math.log1p(-(p2 ** l)) for l in range(d, r + 1))
        q.append(1.0 - math.exp(log_keep))
    pi = np.array([1.0])
    for d in range(0, r + 1):
        pi = np.convolve(pi, _binom_pmf(n[d], p1 * q[d]))
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum(), n

def eps_from_pi(pi, sigma, steps, delta):
    pld = mixture_gaussian_pld(pi, sigma, GRID)
    return pld.self_compose(steps).get_epsilon_for_delta(delta)

def row(label, eps, base):
    print(f"  {label:<44} eps = {eps:9.4f}   {eps/base:6.2f}x")

CELLS = [
    ("facebook headline  p1=.013 p2=1.0 r=1 K=5  s=5   T=500  d=1e-6",
     dict(p1=0.013, p2=1.0, r=1, K=5, sigma=5.0, T=500, delta=1e-6)),
    ("facebook           p1=.013 p2=1.0 r=2 K=5  s=5   T=500  d=1e-6",
     dict(p1=0.013, p2=1.0, r=2, K=5, sigma=5.0, T=500, delta=1e-6)),
    ("PPI matched-eps    p1=.0114 p2=1.0 r=1 K=5 s=45.44 T=2000 d=1.574e-5",
     dict(p1=0.0114, p2=1.0, r=1, K=5, sigma=45.4375, T=2000, delta=1.574e-5)),
    ("PPI matched-eps    p1=.0114 p2=0.1 r=1 K=5 s=11.36 T=2000 d=1.574e-5",
     dict(p1=0.0114, p2=0.1, r=1, K=5, sigma=11.359375, T=2000, delta=1.574e-5)),
]

for name, c in CELLS:
    p1, p2, r, K, sg, T, dl = (c['p1'], c['p2'], c['r'], c['K'],
                               c['sigma'], c['T'], c['delta'])
    print(f"\n{name}")
    cur = sparsegnn_epsilon(
        p1, p2, r, K, sg, T, dl, K_out=K, grid=GRID)
    row("as accounted now (cap g at K, use K)", cur, cur)
    pi, n = refined_pi(p1, p2, r, K, K)
    row(f"union-safe, refined  (n_d={n[1:]})", eps_from_pi(pi, sg, T, dl), cur)
    naive = sparsegnn_epsilon(
        p1, p2, r, 2*K, sg, T, dl, K_out=2*K, grid=GRID)
    row("union-safe, naive (K -> 2K everywhere)", naive, cur)
    half = max(1, K // 2)
    hp = sparsegnn_epsilon(
        p1, p2, r, K, sg, T, dl, K_out=K, grid=GRID)
    print(f"  {'FIX (b): cap g at ' + str(half) + ', union <= ' + str(2*half) + ' <= K, account at K':<44}"
          f" eps = {hp:9.4f}   {1.00:6.2f}x   (accounting unchanged; costs utility)")
