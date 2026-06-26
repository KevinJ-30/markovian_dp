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

## Usage

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
