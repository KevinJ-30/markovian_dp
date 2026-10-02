# Numerical figures

`compare.py` computes privacy-accounting comparison curves. 

Run
`python numerics/compare.py --help` for its numerical parameters. It writes
numerical CSVs, provenance, and PNG/PDF/SVG figures. The four-panel comparison
uses horizontal spacing `wspace=0.32` to separate neighboring axes and labels.

The main comparison, expanded comparison, and all three sweeps include lower
pairs alongside the existing upper curves. Their independent count law is

```text
J = Bernoulli(p1) + sum_{ell=1}^r Binomial(K_out**ell, p1*p2**ell)
P = sum_j Pr(J=j) Normal(-j, sigma**2)
Q = sum_j Pr(J=j) Normal(+j, sigma**2)
```

`compare.lower_mixture_weights` implements this law locally; it has neither
the upper pair's factor of two in shell sizes nor its conditional-probability
denominator. The production accountant in `src/privacy/` is unchanged.

Lower curves are dashed in the corresponding upper curve's color; the
`p2=1` lower curve is black dash-dot. CSVs identify these curves as
`method=lower`, including the new `method` column in `pair_weights.csv`.
Both pairs use the existing pessimistic PLD discretization at `--grid`;
lower-pair curves are numerical approximations, not certified downward-rounded
bounds.

Upper curves use non-root shells `2*K_out**ell` for SparseGNN and its `p2=1`
group baseline. Regenerate figures and provenance with `python numerics/compare.py`
and `python numerics/run_all.py`; do not relabel historical results.

Empirical model ablations live in `scripts/`; see the
[ablation pathway](../scripts/README.md#sparseexpand-paper-ablations).

