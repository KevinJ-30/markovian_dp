# Open questions

## 1. Aggregator: why SAGE-mean and not GCN

For node `u` with in-neighbours `N(u)`:

**SAGE-mean** (`SAGEConv`, what we run)

    h'_u = W_r h_u  +  W_l · (1/|N(u)|) Σ_{w∈N(u)} h_w

**GCN** (`GCNConv`)

    h'_u = W Σ_{w∈N(u)∪{u}} (1 / sqrt(d̂_w · d̂_u)) h_w        d̂ = deg + 1

The coefficient is the whole difference. SAGE's is `1/|N(u)|` — a function of
the **target's** in-degree, which SparseExpand always materializes in full.
GCN's is `1/sqrt(d̂_w d̂_u)` — it needs the **source's** degree, which a rooted
subgraph truncates: boundary nodes' in-edges were never expanded, so `d̂_w`
reads as 1 instead of its true value.

Consequence: with SAGE-mean the rooted computation *equals* full-graph
inference at the root, so training and evaluation compute the same function.
Measured relative error at the root (PPI, K=5, p2=1, r=1, L=1):

| aggr | median | max |
|---|---|---|
| mean | 0.00e+00 | 1.7e-07 |
| gcn | 1.12 | 1.50 |

And on PPI non-DP, p2=1: mean 0.609 vs gcn 0.337 micro-F1.

**Open:** worth stating in the paper as a methodological requirement — *any*
aggregator whose normalisation reads source-side degrees is incompatible with
rooted-subgraph training unless evaluation is also rooted. Also note
`project=True` in SAGEConv would reintroduce the problem by a different route
(it transforms boundary representations before averaging); unmeasured.

## 2. Evaluation procedure

**How full-graph inference predicts a single node.** It doesn't, separately —
one forward pass produces an `[N, C]` matrix and node `v`'s prediction is row
`v`. An L-layer model's output at `v` depends only on `v`'s L-hop
neighbourhood, so the full-batch pass is just the batched form of doing that
per node, reusing shared intermediates. This is the standard protocol
(GraphSAGE trains on sampled neighbourhoods, evaluates full-batch layer by
layer).

**Three candidate evaluation graphs.**

| | edges | what it answers |
|---|---|---|
| full, uncapped | all | deployment: model applied to the real graph |
| capped (K), unsparsified | ~K per node | isolates the cost of the degree cap |
| rooted + p2-sparsified | as in training | matches the training input distribution |

We currently report the first two (the second under `<metric>_alt`).

**Is full-graph evaluation fair?** Argument that it is: sparsification (`p2`)
and capping (`K`) exist to amplify *privacy during training*; they are not
deployment constraints. Once the model is released, applying it to full data is
post-processing and costs no budget. Evaluating under sparsification would
handicap the model for no reason.

Argument that it is a distribution shift: the model was trained on sparse
inputs and is scored on dense ones. But with **mean** aggregation both
perturbations are *unbiased* — a uniformly random subset of neighbours (which
is what both `p2` sampling and our random degree cap produce) has the same
expected aggregate as the full neighbourhood. So dense evaluation supplies the
exact quantity the sparse training inputs were noisy estimates of. Empirically
dense evaluation helps rather than hurts (PPI: 0.5953 uncapped vs 0.5645
capped).

**Open:**
- Unbiasedness holds per layer but not through the ReLU (`E[f(X)] ≠ f(E[X])`),
  so the argument is first-order only at L=2. Unquantified.
- Reddit is a 111x gap between training (1.03M arcs) and evaluation (114.6M);
  PPI is only 8x. Does the gap stay benign at that ratio?
- Should the paper report the rooted+sparsified number as a third column, or is
  it measuring a handicap nobody would deploy?
