# Privacy Amplification by Composite Subsampling for GNNs

Node-level differentially private GNN training, where the privacy amplification
comes from *two* stages of subsampling rather than one: Poisson sampling of root
nodes, followed by randomized sparsification of each root's neighbourhood.

One training step:

1. **Root sampling.** Each node is selected independently with probability `p1`.
2. **SparseExpand.** Each selected root grows a rooted subgraph by walking
   *incoming* edges for `r` levels, keeping each examined arc independently with
   probability `p2` (Algorithm 5 of the manuscript).
3. **Noisy update.** Each rooted subgraph contributes one gradient `g0`, clipped
   to L2 norm `C`; the clipped gradients are summed and one draw of
   `N(0, (sigma*C)^2 I)` is added (Assumption 6.3).

The composition of both sampling stages amplifies privacy beyond what
Poisson subsampling alone gives, which is what the dominating pairs in
`src/sparse/accounting.py` quantify.

## Layout

```
src/
  datasets.py              loader: Planetoid, OGB, Reddit, Flickr, PPI, RelBench
  sparse/
    sparse_expand.py       SparseExpand, root sampling, degree capping
    sparse_gnn.py          training engine (non-DP and DP paths)
    base_mechanism.py      g0 interface + clipping / noise / evaluation helpers
    layers.py              message-passing stack (SAGE-mean or GCN)
    gnn_mechanism.py       single-label node classification g0
    multilabel_mechanism.py  multilabel g0 (PPI): micro-F1 + AUROC
    binary_mechanism.py    binary g0 (RelBench entity tasks): AUROC
    mlp_mechanism.py       graph-blind baseline g0
    relbench_data.py       RelBench database -> homogeneous directed graph
    accounting.py          dominating pairs -> Google dp_accounting
    compute_epsilon.py     post-hoc epsilon for a results CSV
    run.py                 experiment CLI
    gad/                   graph anomaly detection (XGBoost, GADBench) — side pipeline

scripts/                   drivers and figures (see scripts/README.md)
sbatch/                    SLURM jobs for the cluster runs
tests/                     106 tests; see "Tests" below
results/                   experiment output, grouped by dataset (results/README.md)
paper/                     manuscript and figures
```

## Install

```bash
pip install torch torch_geometric ogb opacus dp_accounting scipy pandas matplotlib pytest
```

`relbench` is needed only for RelBench datasets, and `xgboost` + `scikit-learn`
only for `src/sparse/gad/` (its test skips when absent).

## Usage

Utility is measured first; epsilon is attached afterwards from the mechanism
parameters recorded in the CSV. Accounting never touches training.

```bash
# 1. train (--dp adds clip+noise; omit it for the non-private reference)
python -m src.sparse.run --dataset ppi --model multilabel_gnn --direction in \
    --dp --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma 5 \
    --K_in 5 --K_out 5 --lr 0.3 --seeds 3 --track_every 50 \
    --out_dir results/ppi/myrun

# 2. attach epsilon
python -m src.sparse.compute_epsilon \
    --csv results/ppi/myrun/sparse_gnn_ppi_dp_results.csv --delta 1e-6
```

`--track_every N` evaluates every N steps and writes one CSV row per
checkpoint. Since epsilon grows with the step count, a single run then yields a
whole privacy–utility curve, and each checkpoint carries the guarantee for the
model as released at that step. Evaluation consumes no sampling randomness, so a
tracked run follows exactly the same trajectory as an untracked one.

Higher-level drivers live in `scripts/`: `ladder_stage01.sh` (baselines and the
sparsification sweep, no DP), `ladder_stage2.sh` (clip+noise, then epsilon),
`sweep.sh <axis>` for one-axis tuning, and `diagnose.sh` for gradient-norm and
metric probes.

### Parameters that price epsilon, and parameters that do not

Only `p1`, `p2`, `r`, `K_in`/`K_out`, `sigma`, and `T` enter the accounting.
The clipping norm `C` does **not**: sensitivity and noise both scale with it, so
it cancels. Neither do the learning rate, momentum, optimizer, or model depth
`L` — those are free to tune.

Note `L` and `r` are independent. `r` is the expansion depth and sets the
privacy radius; `L` is the number of GNN layers. An `L`-layer model on an
`r`-hop subgraph still reads only `r` hops, because the subgraph simply does not
contain anything further out — the extra layers add depth, not reach.

## Accounting

`src/sparse/accounting.py` builds the manuscript's dominating pairs and hands
them to Google's `dp_accounting` for composition and epsilon(delta):

- **Theorem 6.4** (node substitution) for `--direction in`, the corrected
  expansion orientation. This is the headline guarantee.
- **Theorem 4.5** (node insertion/removal) for `--direction out`, which is the
  orientation ablation; the pair is not symmetric, so both directions are
  composed and the max reported.

The pair handed to `dp_accounting` is built to dominate the analytic one —
exact CDF cell masses, per-cell loss taken at the worse cell edge, trimmed mass
routed to an infinite-loss outcome — so the reported epsilon is an upper bound.

Under in-expansion the accounting shells are `K_out^d`, so it is the **out**-degree
cap that prices the guarantee. Epsilon is charged for the worst-case bound
`K^d` while utility only ever sees `E[min(deg, K)]`, which saturates: on a
heavy-tailed degree distribution a generous cap costs a great deal of epsilon
for very little signal.

## Tests

```bash
pytest tests/
```

- `test_accounting.py` — dominating-pair weights and epsilon, with degenerate
  cases cross-checked against Opacus.
- `test_dp_mechanics.py` — *measures* the DP path rather than reading it:
  per-subgraph (not per-batch) clipping, the 2C substitution / C insertion
  sensitivity bounds, noise calibrated to `sigma*C` and drawn once per step,
  Poisson root sampling with the right variance, and that model depth cannot
  widen the privacy radius.
- `test_theorem_numerical.py` — verifies Theorem 6.4 itself, by computing the
  hockey-stick divergence of the actual mechanism on a star graph and checking
  the dominating pair upper-bounds it.
- `test_sparse_expand.py`, `test_mechanisms.py`, `test_gad.py` — expansion,
  orientation, degree capping, and the base mechanisms.

## Things worth knowing before reading results

- **Aggregator.** The default `--aggr mean` (GraphSAGE) makes the rooted-subgraph
  computation *exactly* equal full-graph inference, because its normalizer reads
  only the target's in-neighbourhood, which SparseExpand always materializes in
  full. `--aggr gcn` normalizes by the *source* degree, which a subgraph
  boundary truncates; measured rooted-vs-full relative error is ~0 for mean and
  ~1.1 for gcn, and GCN loses ~27 accuracy points on PPI as a result.
- **Evaluation graph.** Training uses the deduplicated, degree-capped graph;
  evaluation defaults to the full one. Every run therefore records both, the
  second set under `<metric>_alt`, so the gap is measured rather than assumed.
- **Metrics.** On PPI the all-positive predictor scores 0.4608 micro-F1 while
  having no ranking ability at all (AUROC 0.4955), so a model below that floor
  may still be learning. AUROC is recorded alongside micro-F1 for this reason.
- **Inductive settings differ.** PPI and RelBench are natively inductive (disjoint
  graphs; temporal splits). ogbn-arxiv, Flickr, and Reddit are single graphs made
  inductive by `--inductive`, which drops every arc crossing a split — 68–76% of
  edges on those datasets.
