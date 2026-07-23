# Sparsification DP-GNN Experiments

Self-contained module in `sparsification_experiments/`.  Does **not** import
from `src/algorithms/`, `subgraph_trainer.py`, or the old `run.py` sweeps.

---

## Sparsification spec

### 1. Directed message passing
The GNN is treated as directed: node v aggregates only from its **incoming**
neighbours {u : u → v}.  If the input graph is undirected, both arcs u→v and
v→u are materialised before sparsification.

### 2. In-degree cap
Fix bound D.  For every node v with in-degree > D, keep a uniformly-random
subset of exactly D incoming arcs; drop the rest.  This is a **one-time
structural operation** done before training; the sparsified graph is fixed
for all training steps.

### 3. Sensitivity formula

With per-node gradient clipping at L2 norm C and an L-layer GNN, removing
one node u from the dataset can change at most

    1 + D + D² + … + D^L

per-node gradient terms (u's own loss term, plus the loss terms of every node
within L directed hops from u in the sparsified graph).  The L2 sensitivity
of the **summed** gradient is therefore:

    Δ = C · (1 + D + D² + … + D^L)

**One-sided vs add/remove sensitivity:**  The formula above is the
*removal* sensitivity.  For full add/remove adjacency (inserting **or**
deleting a node), both directions must be bounded, giving

    Δ_add_remove = 2 · C · (1 + D + D² + … + D^L)

Toggle with `--adjacency {remove,add_remove}`.

*Reference:* Daigavane et al. 2021, "Node-Level Differentially Private
Graph Neural Networks", Theorem 1.

---

## Noise-scaling convention

The noise added to the gradient sum each step is

    N(0, (σ · Δ)² · I)

where **σ is the sensitivity-normalised noise multiplier** — the quantity both
accountants consume.  This is distinct from the raw noise std σ·Δ; folding Δ
into the std reduces the mechanism to a unit-sensitivity Gaussian, which is
the standard Opacus and PLD accountant interface.

---

## Mode 1 — utility (no DP)

Sweeps the in-degree bound D (plus a full-graph reference row at D=∞).  No
clipping, no noise.  The accuracy gap between full-graph and sparsified GCN
isolates the cost of sparsification alone.

---

## Mode 2 — dp

DP-SGD on the sparsified graph:
- Poisson-subsample training nodes at rate q each step.
- Shared forward pass on the sparsified graph.
- Per-node backward (microbatching); clip each to C; sum.
- Add N(0, (σ·Δ)² · I) noise; average by expected batch size q·n_train; step.
- Sweep σ; report test accuracy + two epsilon estimates.

**Scale note:** the current DP training loop runs one backward pass per
sampled seed node (microbatching with retain_graph).  This is tractable for
Planetoid datasets.  For ogbn-arxiv, replace the full-graph forward with
per-seed subgraph extraction via `NeighborLoader` (or use `functorch.vmap`
for parallel per-sample gradients).

---

## Accounting (two paths)

### A — Opacus PRV (path A, reference)
`opacus.accountants.create_accountant('prv')` with:
- `noise_multiplier = sigma`
- `sample_rate = q`
- `steps = T`

### B — Dominating-pair PLD (path B, the one we trust/report)
Per-step dominating pair for the Poisson-subsampled Gaussian:

    Q = N(0, σ²)
    P = (1 − q)·N(0, σ²) + q·N(1, σ²)

Implemented in `dp_accounting.make_subsampled_gaussian_dominating_pair`.
The pair is discretised onto a fine x-axis grid (60 000 atoms covering
±10σ), then the privacy loss per atom is computed, rounded **up** to the
nearest PLD grid point (pessimistic — makes eps an upper bound), and the
resulting PLD is composed T times via exponentiation-by-squaring using the
FFT-based convolution in `dp-subsample-prelim/accounting.py`.

### C — Novel mechanism hook (SparseGNN, Theorem 4)
`dp_accounting.make_novel_mechanism_dominating_pair` now delegates to
`src/sparse/accounting.py`: the Theorem 4 insertion/removal marked pair by
default (`theorem=3` selects the substitution pair, tractable only for tiny
K_in and r).  Note the Theorem 4 pair is oriented — compose both (P,Q) and
(Q,P) and report the max epsilon; `sparsegnn_thm4_epsilon` does this.

---

## Validation

Before reporting results, the two accountants are cross-checked across the
full sigma grid.  The comparison table is printed and an `AssertionError` is
raised if any row exceeds `--validation_tol` (default 0.1 ε units).

---

## Quick-start

```bash
cd sparsification_experiments

# Utility sweep on Cora (3 seeds, D ∈ {2, 5, 10} + full graph)
python run.py --dataset cora --mode utility \
    --degree_bounds 2 5 10 --seeds 3

# DP sweep on Cora (σ ∈ {0.5, 1.0, 2.0, 4.0}, D=5, 3 seeds)
python run.py --dataset cora --mode dp \
    --degree_bounds 5 --sigmas 0.5 1.0 2.0 4.0 \
    --steps 200 --sample_rate 0.5 --seeds 3 --plot
```

Results land in `results/results.csv` and (with `--plot`) PNG files.
