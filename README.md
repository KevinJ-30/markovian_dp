# Markovian DP Subsampling for Graph Neural Networks

Differentially private node classification and link prediction on graphs via random subgraph partitioning. The core idea: at each training step, assign nodes uniformly at random to K bins and train independently on each bin's induced subgraph. Per-bin gradient clipping and Gaussian noise then give node-level DP guarantees with a privacy amplification factor of 1/K from the Markovian (bin-assignment) subsampling.

## Algorithms

| ID | Name | Description |
|----|------|-------------|
| 1 | `BallsAndBins` | Bin k keeps all edges where the source is in bin k (neighbors outside become sinks). |
| 2 | `RemoveSinks` | Bin k keeps only edges where **both** endpoints are in bin k. Produces the same gradients as Algo 1 (sinks don't affect the forward pass or loss) but with fewer edges. |
| 3 | `RemoveSinksSubsampled` | Algo 2 with an additional per-node Poisson drop (probability `p_perp`). Excluded nodes are assigned to a dummy bin, giving privacy amplification by subsampling. |

## Project Structure

```
run.py                   # Unified experiment CLI (recommended entry point)
run_benchmark.py         # Sweep across algos and bins on Cora/PubMed (5 seeds)
run_noise_sweep.py       # Sweep noise multipliers; plot accuracy vs epsilon
run_ogbn_utility.py      # ogbn-products utility benchmark with NeighborLoader

src/
  sparse/                # SparseGNN / SparseExpand (paper Algorithms 1 & 2 — current default)
    sparse_expand.py     #   Algorithm 2: randomized rooted expansion
    sparse_gnn.py        #   Algorithm 1: model-agnostic training engine
    base_mechanism.py    #   BaseMechanism g0 interface + shared clip/noise helpers
    gnn_mechanism.py     #   GNN node-classification g0 (CiteSeer)
    anomaly_mechanism.py #   gradient-based non-GNN g0 (stub; future DP-boosting variant)
    accounting.py        #   Theorem 3 + Theorem 4 dominating pairs (Thm 4 = default accountant)
    compute_epsilon.py   #   post-hoc epsilon for DP sweep CSVs (Theorem 4 + naive Opacus ref)
    run.py               #   CLI entry point (--dp --sigma sweeps, --K_in degree capping)
    gad/                 #   Graph anomaly detection — XGB-Graph on sparsified graphs (GADBench)
      neighbor_aggregation.py  #   parameter-free multi-hop features (global + per-root expand)
      xgb_graph.py             #   XGBGraphDetector (XGBoost on aggregated features)
      metrics.py               #   AUROC / AUPRC / Rec@K
      run.py                   #   CLI: sweep p2, measure utility drop
  algorithms/            # BallsAndBins, RemoveSinks, RemoveSinksSubsampled
  models/                # SubgraphGCN, NodeMLP, LinkPredGCN
  trainers/
    subgraph_trainer.py  # Main trainer: standard and per-bin DP paths
    sparsified_dp_trainer.py  # Per-node clip + degree-aware noise (Daigavane 2021)
    link_pred_trainer.py # BCE link-prediction head (ogbl-collab)
    baseline_trainer.py  # Full-graph GCN/MLP baseline
  datasets.py            # Unified loader: Planetoid, OGB, Reddit, Bluesky
  privacy_accountant.py  # Opacus RDP/PRV epsilon computation
  utils.py               # Degree computation, sparsify_by_degree

sbatch/                  # SLURM job scripts (one per experiment condition)
scripts/
  plot_headline_results.py  # Paper figures and LaTeX tables from JSONL results
  run_baselines_node.sh     # Baseline sweep across all datasets

tests/
  test_algo_equivalence.py  # Gradient-equivalence test for Algos 1 & 2

dp-subsample-prelim/     # Preliminary accounting experiments
sparsification_experiments/  # Degree-sparsification baseline comparisons
paper/                   # Paper draft and experiment figures
```

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install ogb opacus
```

Bluesky dataset support additionally requires the GraphBench package (see `src/datasets.py`).

The graph anomaly detection pipeline (`src/sparse/gad/`) additionally requires `xgboost` and
`scikit-learn`. In this project they live in the `PytorchEnv` conda env — run that pipeline
with `conda run -n PytorchEnv python -m src.sparse.gad.run ...`.

## Usage

### SparseGNN / SparseExpand (paper Algorithms 1 & 2 — current default)

The default sparsification is the composite-subsampling mechanism from *Privacy
Amplification by Composite Subsampling*: each step, roots are Poisson-sampled
(prob `p1`); each root is expanded by `SparseExpand` (keep each outgoing edge
with prob `p2`, up to distance `r`); the base mechanism `g0` contributes one
clipped per-subgraph gradient, summed over roots (Assumption 3.2). `g0` is
pluggable — a GNN node classifier now, a non-GNN anomaly detector later — via
the `BaseMechanism` interface in `src/sparse/`.

```bash
# Sanity: p1=p2=1 recovers a (near) full-graph GCN on CiteSeer (~0.68 test)
python -m src.sparse.run --dataset citeseer --p1 1.0 --p2 1.0 --r 2 --T 200 --seeds 3

# The actual sparsified mechanism
python -m src.sparse.run --dataset citeseer --p1 0.5 --p2 0.5 --r 2 --T 200 --seeds 3
```

Runs are non-DP by default.

#### DP sweeps + post-hoc accounting (Theorem 4)

The workflow is: measure utility first (sweep noise multipliers under `--dp`),
then attach epsilon after the fact. Training never touches the accountant.

```bash
# 1. Utility: sweep sigma x p2 with DP clip+noise on the degree-capped graph.
#    --K_in caps max in/out-degree (required for the Theorem 4 assumption);
#    all mechanism parameters are recorded in the CSV.
python -m src.sparse.run --dataset citeseer --dp --K_in 5 \
    --p1 0.1 --p2 1.0 0.5 0.25 0.1 --r 1 --sigma 2 5 10 --T 200 --seeds 2

# 2. Post-hoc epsilon (Theorem 4 insertion/removal pair, PLD-composed over T,
#    max over both orientations) + a naive Opacus reference column:
python -m src.sparse.compute_epsilon --csv results/sparse_gnn_citeseer_dp_results.csv
```

The accountant lives in `src/sparse/accounting.py`: `sparsegnn_thm4_epsilon`
implements the paper's Theorem 4 marked mixture (fiber weights from the PGF
prod_d (1 - a_d + a_d z^{n_d})), discretized per fiber and composed via the
PLD machinery in `dp-subsample-prelim/accounting.py`. Its r=0 / p2=0
degenerate case is cross-checked against Opacus PRV in
`tests/test_thm4_accounting.py`. Opacus itself cannot account for this
mechanism — it only knows the plain Poisson-subsampled Gaussian — so it serves
as the degenerate-case validator and the (invalid-for-node-DP) reference
curve. The Theorem 3 substitution pair is also implemented (`theorem=3` in the
`make_novel_mechanism_dominating_pair` hook) but has 2^{N_r} components and is
only tractable for tiny (K_in, r).

### Graph anomaly detection (XGB-Graph, GADBench)

Tree-based graph anomaly detection (no GNN): parameter-free multi-hop neighbor aggregation
feeds an XGBoost classifier (GADBench §3.1). The graph enters only through the aggregation
step, so sparsifying edges degrades the aggregated features — the pipeline measures the
resulting utility drop (AUROC / AUPRC / Rec@K) as the edge-sampling probability `p2` falls.

```bash
# Sweep p2 on Tolokers (p2=1.0 is the full-graph reference); plot the utility drop
conda run -n PytorchEnv python -m src.sparse.gad.run --dataset tolokers \
    --p2 1.0 0.75 0.5 0.25 0.1 --num_layers 2 --seeds 5 --plot
```

`--sparsifier global` (default) drops each edge iid with prob `p2` then aggregates; `--sparsifier
expand` uses the paper's per-root `SparseExpand` mechanism (slower, for composite-mechanism
fidelity). Results/plot are written to `results/gad_<dataset>_*.{csv,png}`. No DP for now.

### Unified CLI (`run.py`)

```bash
# Non-DP: Algo 3 on Cora, sweep bins 4 and 8
python run.py --dataset cora --algo 3 --num-bins 4 8 --subsample-prob 0.3

# DP with noise multiplier sweep (single-phase Poisson, q=0.1)
python run.py --dataset ogbn-arxiv --algo 2 3 --num-bins 8 16 \
    --single-phase --q 0.1 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant prv --tag e2_arxiv

# GCN + MLP baselines only
python run.py --dataset cora pubmed --baseline gcn mlp --seeds 5

# DP with epsilon budget (sparsified paradigm, capped in-degree)
python run.py --dataset reddit --algo 2 --num-bins 8 \
    --dp --epsilon 1.0 --max-in-degree 32 --paradigm sparsified-dp

# Link prediction on ogbl-collab
python run.py --dataset ogbl-collab --task link --algo 2 3 \
    --num-bins 8 --dp --noise-multiplier 1.0 --accountant prv
```

Key flags:

| Group | Flag | Description |
|-------|------|-------------|
| Setup | `--dataset` | One or more of: `cora citeseer pubmed ogbn-products ogbn-arxiv reddit ogbl-collab bluesky` |
| Setup | `--algo` | Algorithm(s): `1 2 3` |
| Setup | `--num-bins` | Bin count(s) K to sweep |
| Setup | `--baseline` | Run `gcn` and/or `mlp` full-graph baselines |
| DP | `--dp` | Enable gradient clipping + Gaussian noise |
| DP | `--noise-multiplier` | σ/C value(s) (sweep with multiple) |
| DP | `--epsilon` | Privacy budget ε (alternative to `--noise-multiplier`) |
| DP | `--clip-norm` | Gradient clipping threshold C (default 1.0) |
| DP | `--dp-delta` | δ (default 1e-5) |
| DP | `--accountant` | Compute ε via `rdp` or `prv` (PLD-based, tighter) |
| DP | `--paradigm` | `standard` (per-bin clip) or `sparsified-dp` (per-node clip + degree-aware noise) |
| DP | `--max-in-degree` | Cap in-degree via random sampling (needed for `sparsified-dp`) |
| Subsampling | `--single-phase` | Independent Bernoulli(q) per step — matches standard DP-SGD accounting |
| Subsampling | `--q` | Per-step inclusion probability for `--single-phase` |
| Subsampling | `--poisson` | Two-phase Poisson: epoch pool (q-epoch) then per-step (q-step) |
| Subsampling | `--subsample-prob` | Algo 3 dummy-bin drop probability p_perp (default 0.3) |
| Training | `--epochs` / `--converge` | Fixed epochs or early stopping on train loss |
| Output | `--output-dir` | Results directory (default `results/`) |
| Output | `--tag` | Optional tag appended to output filenames |

### SLURM

The `sbatch/` directory has pre-configured job scripts for each experimental condition. For example:

```bash
# E1: non-DP utility on ogbn-arxiv
sbatch sbatch/e1_arxiv.sbatch

# E2: DP with Poisson subsampling sweep
sbatch sbatch/e2_arxiv.sbatch
```

Set `OGB_DATA_ROOT` to the scratch path for OGB datasets before submitting.

### Paper figures

```bash
python scripts/plot_headline_results.py \
    --data-dir /path/to/headline/jsonl \
    --out-dir paper/figures/experiments \
    --fixed-k 8
```

Outputs: `utility_nondp.png`, `dp_K8_<dataset>.png`, and LaTeX tables under `paper/figures/experiments/tables/`.

## Privacy Accounting

The per-step sample rate fed to the Opacus accountant depends on the algorithm and subsampling mode:

- **Algo 2, single-phase:** `sample_rate = q / K`
- **Algo 3, single-phase:** `sample_rate = q * (1 - p_perp) / K`

Two-phase Poisson (`--poisson`) uses correlated draws within an epoch; the resulting sample rate plugged into the accountant is a loose upper bound.

To compute ε post-hoc from a results JSONL:

```python
from src.privacy_accountant import compute_epsilon
eps = compute_epsilon(noise_multiplier=1.0, sample_rate=0.1/8,
                      num_steps=2000, delta=1e-5, accountant='prv')
```

## Tests

```bash
pytest tests/
```

`test_algo_equivalence.py` verifies that Algorithms 1 and 2 produce identical per-parameter gradients (sinks do not affect the loss) and that Algorithm 3 with `p_perp=0` exactly matches Algorithm 2.
