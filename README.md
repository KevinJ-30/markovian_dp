# Privacy Amplification by Composite Subsampling for GNNs

This repository contains the code for the experiments in the paper [blank]. 

## Layout

```
src/
  data/
    datasets.py           dataset dispatch and graph loaders
    domain_datasets.py    domain-disjoint Twitch, Facebook100, and MAG loaders
  processing/
    splits.py             saved graph-disjoint inductive partitions
    graphs.py             separate training-graph selection
    sparse_expand.py      SparseExpand, root sampling, degree capping
    padded.py             lossless root-first private batch representation
    dpgnn.py              DP-GNN degree sampling and padded one-hop batches
  models/
    base_mechanism.py      g0 interface, optimizer, and evaluation helpers
    *_mechanism.py        task-specific networks and mechanisms
    layers.py             sparse PyG and padded batch-first message passing
    baselines.py          MLP, GraphSAGE, DPAR, and DP-GNN networks
    objectives.py         shared baseline losses, metrics, trivial predictors
  training/
    sparse_gnn.py         model-agnostic non-DP and DP training engine
    baselines.py          portable MLP/GraphSAGE/DP-MLP training
    dpar.py               DPAR training and private PPR
    dpgnn.py              partitioned DP-GNN training
  privacy/
    accounting.py         dominating pairs -> Google dp_accounting
    privacy_loss.py       two-mixture Gaussian dp_accounting primitive
    accountants.py        baseline accounting and calibration adapters
    dpgnn.py              DP-GNN multi-term RDP accounting
  experiments/
    sparse.py             SparseGNN experiment CLI
    compute_epsilon.py    post-hoc epsilon for a results CSV
    run.py                graph-disjoint comparison CLI
    upstream.py           external baseline manifest/result integration
    dpgnn_adapter.py      first-party DP-GNN manifest/result adapter

scripts/                   drivers and figures (see scripts/README.md)
  setup_graphsaint.sh      unpack the manually-downloaded GraphSAINT graphs
sbatch/                    SLURM jobs for the cluster runs
tests/                     mechanism, accounting, and integration tests
results/                   experiment output, grouped by dataset (results/README.md)
paper/                     manuscript and figures
```

The source packages are organized by responsibility; method-specific training
loops and graph protocols remain separate. 

Entry points:
- `python -m src.experiments.sparse` — SparseGNN sweeps.
- `python -m src.experiments.compute_epsilon` — post-hoc privacy accounting.
- `python -m src.experiments.run` — graph-disjoint baseline comparisons.

The comparison runner supports `mlp`, `dp_mlp`, `graphsage`, `dpar`, `dp_gnn`,
and `progap`. SparseGNN uses its separate CLI above. HeterPoisson support,
presets, and vendored PNPiGNNs source have been removed; historical result
artifacts are retained but are not supported launch configurations.

Baseline dropout defaults to `0.5` for MLP, GraphSAGE, DP-MLP, DPAR, DP-GNN,
and ProGAP. Shipped baseline presets and the non-private full-batch ceiling use
the same default. Explicit overrides remain supported: set `parameters.dropout`
in a baseline JSON config, or `--dropout` for the ceiling CLI; `0.0` disables it.
Historical experiment recipes and recorded results retain their original rates.

The old SparseGNN import and CLI paths have been removed. Existing command-line
flags, dataset/split caches, and result filenames and schemas are unchanged.

## Install

```bash
pip install torch torch_geometric ogb opacus dp_accounting scipy pandas matplotlib pytest "scikit-learn>=1.5" pyyaml
```

## Datasets

Most datasets download themselves on first use, into `data/` (gitignored).
Planetoid, OGB node datasets, and PyG's Reddit/Flickr need no setup.

Supported dataset keys:

| Family | Dataset keys |
|---|---|
| Citation networks | `cora`, `cora-ml`, `citeseer`, `pubmed` |
| OGB node classification | `ogbn-arxiv`, `ogbn-products` |
| PyG node classification | `reddit`, `flickr` |
| Single-university Facebook | `facebook` |
| GraphSAINT | `ppi-large`, `saint-flickr`, `saint-reddit`, `saint-yelp`, `saint-amazon` |
| Domain-disjoint classification | `twitch-explicit`, `facebook100`, `mag-countries` |
| GraphLand regression | `hm-prices`, `avazu-ctr` |

GraphSAINT also accepts `graphsaint:<name>` for `ppi-large`, `flickr`,
`reddit`, `yelp`, and `amazon`. Bare `reddit` and `flickr` retain their
distinct PyG releases; `ppi-large` uses the GraphSAINT release.

**The four large GraphSAINT graphs are the exception and need a manual
download.** Zeng et al. distribute them as a Google Drive folder with no
programmatic endpoint, so nothing in this repo can fetch them for you, and a
fresh clone will fail on `--dataset ppi-large` until you do this:

```bash
# 1. Download from the Google Drive link in github.com/GraphSAINT/GraphSAINT
#    (README, "Dataset"). Drive splits a folder into -001, -002, ... parts;
#    take all of them for each dataset you want. They land in ~/Downloads.

# 2. Unpack into the layout the loader expects, and verify.
./scripts/setup_graphsaint.sh ~/Downloads

# 3. Point the loader at the result (add to your shell profile to make it stick).
export GRAPHSAINT_DATA_ROOT=$PWD/data/graphsaint
```

`GRAPHSAINT_DATA_ROOT` defaults to `data/graphsaint`, so step 3 is only needed
if you extracted somewhere else — `setup_graphsaint.sh <zips> <dest>` takes a
destination, which is what you want on a cluster where the data belongs on
scratch rather than in the repo. The script is idempotent; re-run it freely.

| `--dataset` | nodes | edges (Table 1) | labels | extracted |
|---|---:|---:|---|---:|
| `ppi-large`    |    56,944 |     818,716 | 121 multilabel | 36 MB |
| `saint-flickr` |    89,250 |     899,756 | 7 classes      | — |
| `saint-reddit` |   232,965 |  11,606,919 | 41 classes     | 1.2 GB |
| `saint-yelp`    |   716,847 |   6,977,410 | 100 multilabel | 2.2 GB |
| `saint-amazon`  | 1,598,960 | 132,169,734 | 107 multilabel | 3.7 GB |

The edge column is GraphSAINT's Table 1 verbatim, and the loader reproduces it
from the raw files — but **that figure counts self-loops and the loaded graph
does not**, because the accounting counts paths in a simple graph. PPI-large
carries 25,084 of them, so `data.edge_index` holds 793,632 undirected edges,
not 818,716. Reddit has none and is unaffected. Extracted sizes are measured
except `saint-flickr`, which the loader supports but we have never downloaded
or run — treat that row as untested. First load writes a `_labels_cache.pt`
next to the raw files (27 MB on PPI-large), so budget roughly double the
extracted size.

All four public aliases use the `saint-` prefix: `saint-flickr`,
`saint-reddit`, `saint-yelp`, and `saint-amazon`. For Flickr and Reddit the
prefix also distinguishes GraphSAINT's releases from the bare PyG datasets;
PyG's Reddit has 57.3M undirected edges against GraphSAINT's 11.6M, and the
splits differ too. The raw GraphSAINT directory names remain `flickr`, `reddit`,
`yelp`, and `amazon`. The `_load_graphsaint` docstring documents the
preprocessing needed to reconcile the released files with the paper's Table 1.

### GraphLand regression

`hm-prices` (product price) and `avazu-ctr` (device click-through rate) use
PyG's `GraphLandDataset` to download and preprocess the released graphs.
This requires a PyG version providing that dataset class, scikit-learn >= 1.5,
and PyYAML. The cache defaults to `data/graphland`; set
`GRAPHLAND_DATA_ROOT` or pass `root=` to `load_dataset` to relocate it.

Both use a **custom random 80/10/10 train/validation/test node split**, fixed
at split seed 0 across training seeds. Training and validation counts are
rounded down; test receives the remainder. This replaces the published RH
masks and is **not** GraphLand's published RL, RH, TH, or THI protocol.
The generic comparison runner defaults to these loader-provided (`native`)
masks and rejects a conflicting split strategy. `--common_inductive_split`
also preserves them rather than stratifying continuous targets.

```bash
python -m src.experiments.sparse \
  --dataset hm-prices --model regression_gnn --aggr mean \
  --common_inductive_split --T 200 --seeds 1
```

Use `--dataset avazu-ctr` for CTR, or `--aggr gin` for SparseGIN. Generic
`src.experiments.run` configs can select either dataset with the existing
regression-capable methods, without explicit regression or split flags:

```json
{
  "dataset": "avazu-ctr",
  "method": "graphsage",
  "device": "auto",
  "parameters": {"epochs": 20, "hidden_size": 64}
}
```

The task uses one scalar output and MSE loss. Targets are standardized using
only the **new 80% training labels**, with mean and scale exposed as
`dataset.target_mean` and `dataset.target_std`. Constant training targets use
scale 1. **R² is the sole regression evaluation metric** (`r2`), and higher
is better, including when all candidate scores are negative. Checkpoint
selection maximizes validation R². The score is unitless and unchanged by
target standardization; it is computed over the entire scored split, not
averaged over minibatches. Existing result fields named `*_accuracy` or
`*_acc` retain their legacy names but contain R². No auxiliary regression
metric columns are emitted.

R² uses the evaluated split's own target mean in its denominator. The trivial
reference predictor instead predicts the training mean and can score below
zero. As in scikit-learn's default `r2_score`, constant targets score 1 for
perfect predictions and 0 otherwise; fewer than two scored nodes yield NaN.

The loader rejects nonfinite targets rather than admitting unlabeled roots
into the training/accounting population.

Feature encoding, quantile transforms, and missing-feature imputation retain
PyG's **full-graph** preprocessing. Message-passing contexts still follow the
selected runner: generic comparisons and SparseGNN with
`--common_inductive_split` use graph-disjoint partitions; ordinary SparseGNN
uses train-induced edges for fitting and the full graph for evaluation.
Thus this is not a strictly train-only feature-preprocessing benchmark.
DP training accounting does **not** account for releasing or fitting these
data-dependent feature/target transforms; treating their statistics as public
or otherwise accounting for them is a separate privacy assumption.

**ProGAP adapted for regression** is supported through the retained inductive
adapter. The runner derives the task from dataset metadata (`primary_metric:
"r2"`); no separate regression flag is needed. Each progressive stage uses one
unbounded scalar output and per-root MSE. Prediction bypasses softmax, and
validation/test R² is reduced over the entire scored split using centered
float64 statistics, independently of evaluation chunk size. Selection maximizes
validation R² even when all checkpoints have negative scores. Results declare
`metric: "r2"`; legacy accuracy/macro-F1 result fields carry that same R².
The adapter requires at least two scored nodes for a defined evaluation.

For example, reuse the existing ProGAP smoke configuration and its configured
ProGAP Python environment:

```bash
python -m src.experiments.run \
  --config configs/cora_ml_progap_smoke.json --dataset hm-prices \
  --out results/inductive/hm-prices/progap.json
```

Use `--dataset avazu-ctr` for CTR. ProGAP's `epochs` applies **per progressive
stage**: depth `d` trains `d + 1` stages. NAP normalization/noise, degree bounding,
per-example gradient clipping/noise, sampling, and composed calibration remain
unchanged by the regression adaptation. This is task-adapted ProGAP, not an
unmodified upstream classification baseline. Raw private-training losses are
not published in adapter histories; validation metrics assume public/fixed
held-out data. Data-dependent preprocessing and private validation selection
still require separate privacy treatment.

### Domain-disjoint datasets

Three dataset names expose provenance-defined domains rather than a random node
split. `facebook100` is a new 18-school benchmark; the existing `facebook`
dataset remains the single UIllinois20 graph and is unchanged.

| dataset | canonical domains | default train | default validation | default test | reported metric |
|---|---|---|---|---|---|
| `twitch-explicit` | `de`, `engb`, `es`, `fr`, `ptbr`, `ru`, `tw` | `de` | `engb` | `es`, `fr`, `ptbr`, `ru`, `tw` | AUROC |
| `facebook100` | `penn94`, `amherst41`, `cornell5`, `johns-hopkins55`, `reed98`, `caltech36`, `berkeley13`, `brown11`, `columbia2`, `yale4`, `virginia63`, `texas80`, `bingham82`, `duke14`, `princeton12`, `washu32`, `brandeis99`, `carnegie49` | `johns-hopkins55`, `caltech36`, `amherst41` | `cornell5`, `yale4` | `penn94`, `brown11`, `texas80` | accuracy |
| `mag-countries` | `us`, `cn`, `de`, `fr`, `ru`, `jp` | `us` | `cn` | `cn` | accuracy |

The defaults apply when no domain role is supplied. A custom split must provide
all of `train`, `val`, and `test`, with nonempty lists of canonical lower-case
names. Names cannot repeat within a role, and no training domain may occur in
either held-out role. Validation and test may overlap. Their overlapping domain
is present in full in both evaluation graphs, so message passing has the same
complete target-domain context; deterministic, class-stratified, complementary
node masks decide which nodes each role scores. `seed` controls that assignment
and `val_ratio` is its validation fraction (defaults: `0` and `0.2`). Classes
with at least two nodes contribute to both masks. Thus the default shared `cn`
MAG target implements a 20/80 validation/test split without cutting its
topology. Different normalized domain selections receive different split,
cache, and result fingerprints.

Configuration-driven experiments put the mapping at the top level. For example,
save the following as `/tmp/mag-domain.json` and run
`python -m src.experiments.run --config /tmp/mag-domain.json`:

```json
{
  "dataset": "mag-countries",
  "method": "graphsage",
  "seed": 0,
  "device": "auto",
  "split_root": "data/inductive_splits",
  "domain_split": {
    "train": ["us"],
    "val": ["cn"],
    "test": ["cn"],
    "seed": 0,
    "val_ratio": 0.2
  },
  "parameters": {
    "epochs": 100,
    "layers": 2,
    "batch_size": 1024,
    "max_fanout": 10,
    "graphsage_sampling": "hierarchical"
  }
}
```

The same dataset metadata is consumed by the first-party `mlp`, `dp_mlp`,
`graphsage`, `dpar`, and `dp_gnn` methods and by the retained ProGAP adapter.

The first-party GraphSAGE trainer uses fresh fixed-fanout neighborhoods for
each root minibatch. `graphsage_sampling: "hierarchical"` (the default) uses
the same sampled seed-node computation as `"neighbor"`, but progressively
trims the deepest unused hop before each layer. Set `max_fanout` to the per-layer
bound; the trainer repeats it for `layers` hops. Each training root appears
once per shuffled epoch, with a final partial batch. Validation and test use
deterministic full-neighbor propagation over each complete held-out context
graph and score only its `eval_mask`.

The first-party DPAR trainer retains the sampled subgraph as feature context
but supervises only its `M = min(ppr_num, sampled_nodes)` selected APPR roots.
The APPR matrix is `M × sampled_nodes`; it has no identity rows for other
nodes. Each epoch visits those `M` roots once, including a final partial batch,
so it makes `ceil(M / batch_size)` updates. SGD calibration uses
`min(batch_size, M) / M`, separately from outer graph sampling amplification.
The ISTA recurrence uses float64 to prevent residual roundoff from blocking
convergence at the requested tolerance. Converged weights are converted to
float32 before clipping, noise addition, and release.
The released PPR/SGD accounting arithmetic remains a qualified repository
convention, not an independently certified node-level DP guarantee.

SparseGNN uses matching flags instead:

```bash
python -m src.experiments.sparse \
    --dataset twitch-explicit --model binary_gnn \
    --train_domains de --val_domains engb --test_domains es fr ptbr ru tw \
    --domain_split_seed 0 --domain_val_ratio 0.2 \
    --T 500 --seeds 3 --out_dir results/twitch-explicit/default
```

Twitch is binary and must use `binary_gnn`; it trains a single logit and reports
tie-correct AUROC (`validation_auroc` and `test_auroc`), not accuracy.
`facebook100` is a two-class accuracy task; matching GraphOOD, raw missing
gender `0` is collapsed into class `0`. MAG is a 20-class task. Label 19 still
participates in training loss but is excluded from validation/test accuracy.

All three families download automatically on first use over HTTPS. The cache
roots can be overridden, and otherwise are:

| variable | default | acquisition |
|---|---|---|
| `GRAPHOOD_TWITCH_DATA_ROOT` | `data/graphood/twitch` | selected domains only, from `CUAI/Non-Homophily-Benchmarks` commit `af14a88470d30b1dadd3803d911dfc1064bcf172` |
| `GRAPHOOD_FB100_DATA_ROOT` | `data/graphood/facebook100` | all 18 schools, from `sisaman/pyg-datasets` commit `9a92bf1e84f73b7b24dd745eb14f13e4d1979769` |
| `PAIR_ALIGN_MAG_DATA_ROOT` | `data/pair_align_mag` | selected countries only, from Zenodo record `10681285` |

FB-100 downloads all schools even when only one is selected because GraphOOD's
categorical feature vocabulary is shared across the 18 matrices. Twitch and MAG
download only selected domains. First use therefore needs network access and
enough space in the chosen roots; subsequent loads reuse valid cached files and
can run offline. Downloads go to temporary files and are renamed atomically, so
an interrupted transfer is not accepted as cache and a retry is safe. A failure
reports both the source URL and cache root.

MAG files are untrusted until both their published byte size and MD5 digest have
been checked; only then are the PyTorch products deserialized. The record's
digests are `us` `677b46f78e5fb946b2d9d2e4f76418fb`, `cn`
`3e09b899d12d5801f39bf9cd187edcad`, `de`
`3e3830bd6102db954f1b0163761aebc3`, `fr`
`a2387bdff7841edb395f23d224b0b1c5`, `ru`
`3c86cf9b3b2052d31a433d5422a7ec5f`, and `jp`
`7910d054a972897fc2466f177cb9fed4`. GitHub sources are pinned to immutable
commits and their parsed filenames and schemas are validated, but those
upstreams do not publish conventional artifact checksums.

## Usage

Utility is measured first; epsilon is attached afterwards from the mechanism
parameters recorded in the CSV. Accounting never touches training.

```bash
# 1. train (--dp adds clip+noise; omit it for the non-private reference)
python -m src.experiments.sparse --dataset ppi-large --model multilabel_gnn --direction in \
    --dp --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma 5 \
    --K_in 5 --K_out 5 --lr 0.3 --seeds 3 --track_every 50 \
    --out_dir results/ppi-large/myrun

# 2. attach epsilon
python -m src.experiments.compute_epsilon \
    --csv results/ppi-large/myrun/sparse_gnn_ppi-large_dp_results.csv --delta 1e-6
```

Select the SparseGNN architecture with `--aggr mean` (the default GraphSAGE),
`--aggr gcn`, or `--aggr gin`. GIN uses sum aggregation and
`MLP((1 + epsilon) * x + sum(neighbors))`, with fixed `epsilon=0`.
Each layer's MLP is `Linear(in, out) -> ReLU -> Linear(out, out)`, without
batch normalization; `--hidden` and `--num_layers` set the stack dimensions.
It supports non-private training, padded per-root Opacus clipping/noise, and
full-graph CSR inference. For the study runner, set `"aggregation": "gin"`
in the JSON cell; the existing `"mean"` setting remains the default.
Changing the architecture does not change calibration for fixed sampling,
degree bounds, clipping, and update count.

`--track_every N` evaluates every N steps and writes one CSV row per
checkpoint. Since epsilon grows with the step count, a single run then yields a
whole privacy–utility curve, and each checkpoint carries the guarantee for the
model as released at that step. Evaluation consumes no sampling randomness, so a
tracked run follows exactly the same trajectory as an untracked one.

Higher-level drivers live in `scripts/`: `ladder_stage01.sh` (baselines and the
sparsification sweep, no DP), `ladder_stage2.sh` (clip+noise, then epsilon), and
`sweep.sh <axis>` for one-axis tuning.

The SparseGNN study runner (`results/eight_gpu_domain_graphsaint/sparse/run.py`)
accepts a positive integer `batch_size <= n_train` in its JSON cell. This is
the expected Poisson root count: set `p1 = batch_size / n_train`,
`steps_per_epoch = ceil(n_train / batch_size)`, and
`steps = epochs * steps_per_epoch` for a full run. Changing batch size at a
fixed epoch budget changes both sampling probability and update count;
recalibrate noise for the new schedule rather than reusing the old multiplier.

### Degree capping

SparseGNN's default `--cap_mode auto` resolves to `directed`: after removing
self-loops, symmetrizing, and deduplicating, it retains a random subset of up to
`--K_out` outgoing arcs per node. Incoming degree is unrestricted, and reverse
arcs are selected independently. The cap is applied once per seed, not per
training step; `--cap_seed` shares a capped graph across seeds.

`--K_out` can be supplied alone. If omitted, it defaults to `--K_in`.
`--K_in` remains an accounting parameter but does not cap incoming arcs in
directed mode; when omitted, its recorded value comes from the capped graph's
observed maximum incoming degree. `--cap_mode undirected` is unchanged:
it requires equal `K_in` and `K_out`, bounds both endpoints' degrees, and keeps
both arcs of each retained edge. Evaluation graphs remain uncapped.

Separately, incoming SparseExpand sampling now caps retained arcs at **20 per
expanded node per hop**, in both private and non-private SparseGNN runs.
`MAX_INCOMING_EDGES` in `src/processing/sparse_expand.py` sets this cap.
The sampler draws the capped Binomial count and samples CSR positions directly;
it does not construct a candidate tensor spanning a high-degree node's entire
neighborhood. This bounds local expansion work, not the total number of roots
or nodes in a batch. Outgoing expansion, preprocessing degree limits, other
models, and full-graph evaluation are unchanged. Accounting formulas are
unchanged; this implementation change does not revalidate them for the cap.


## Accounting

`src/privacy/accounting.py` constructs the manuscript's **Theorem 5.4 node-
substitution pair** for incoming-edge expansion. The pair is represented as two
Gaussian mixtures through `DoubleMixtureGaussianPrivacyLoss`; Google's
`dp_accounting` performs pessimistic connect-the-dots discretization,
composition, and epsilon(delta).

The training `direction` is not inspected by the accountant: it always applies
the in-expansion shell law. The private optimizer is likewise separate from
accounting—Opacus computes per-root gradients, clips at `C`, and injects noise
with standard deviation `sigma*C`; no generic Opacus accountant is attached.

Under in-expansion the accounting shells are `K_out^d`, so it is the **out**-degree
cap that prices the guarantee. Epsilon is charged for the worst-case bound
`K^d` while utility only ever sees `E[min(deg, K)]`, which saturates: on a
heavy-tailed degree distribution a generous cap costs a great deal of epsilon
for very little signal.

## DP-GNN baseline

DP-GNN is a separate learner, not SparseGNN with different sampling flags:

```bash
python -m src.experiments.run --config configs/cora_ml_dp_gnn_smoke.json \
    --out /tmp/dpgnn-smoke.json
```

Its training partition is sampled once: incoming arcs are retained independently
with probability `min(1, K/(2*d))`, selected neighbors are deduplicated, and an
entire incoming list is discarded if it exceeds `K`. Each update draws a fresh
fixed-size root batch **without replacement**, then gathers cached one-hop stars.
This follows the subset sampling in the
[DP-GNN paper's Algorithm 4](https://arxiv.org/html/2111.15521), rather than the
replacement draws in Google's executable implementation.

Method `parameters` accept:
- `clip` (default `1.0`): global L2 bound `C` on each root's complete gradient.
- `dropout` (default `0.5`): hidden-activation dropout immediately before the
  decoder in both GCN and GraphSAGE, including private padded batches. Disabled
  during evaluation; set `0.0` to disable it during training as well.
- `max_private_batch_nodes` (default `8192`): physical padded-slot budget.
  Chunking preserves one noise addition and one Adam update per logical batch;
  a single oversized star is processed alone.
- `batch_size`: positive logical batch size `B`, no larger than training size `N`.
- `noise_multiplier`: sensitivity-normalized multiplier `lambda`.
- `architecture` (default `graphsage`): `graphsage` uses separate root and
  mean-neighbor transforms; `gcn` selects the original one-hop model. This
  changes only the clipped per-root model, so sampling and privacy accounting
  are unchanged.

For `M = min(K+1, N)`, Opacus adds isotropic Gaussian noise with standard deviation
`2*M*C*lambda` to the clipped sum, then divides by `B`. Its internal multiplier
is therefore `2*M*lambda`, **not** the value passed to the hypergeometric multi-term
RDP accountant in `src/privacy/dpgnn.py`. No SparseGNN PLD or generic Opacus
accountant is used. The former per-parameter percentile clipping and manual
noise path have been removed.

Graph-disjoint partitions, one-hop architecture, training-star truncation, and
the existing sampled full-partition evaluation are unchanged. `evaluate_every`
remains accepted but only final validation/test metrics are returned. Historical
DP-GNN results predate this clipping/root-sampling change and are not rewritten.

The direct trainer also accepts `DPGNNConfig(multilabel=True)` for multi-hot
targets. It averages binary cross-entropy over labels within each root, retains
the same global per-root clipping and accounting, and returns
`validation_micro_f1` / `test_micro_f1`. The default remains categorical
cross-entropy with accuracy.

## Tests

```bash
pytest tests/
```

- `test_accounting.py` — dominating-pair weights and epsilon, with degenerate
  cases cross-checked against Opacus.
- `test_dp_mechanics.py` — measures the Opacus DP path: per-subgraph (not
  per-batch) clipping, the 2C substitution sensitivity bound, noise calibrated
  to `sigma*C` and drawn once per step, Poisson root sampling with the right
  variance, and that model depth cannot widen the privacy radius.
- `test_theorem_numerical.py` — verifies Theorem 5.4 itself (Theorem 6.4 in
  manuscript v36), by computing the
  hockey-stick divergence of the actual mechanism on a star graph and checking
  the dominating pair upper-bounds it.
- `test_sparse_expand.py`, `test_mechanisms.py` — expansion, orientation,
  degree capping, and the base mechanisms.
- `test_dpgnn_sampling.py`, `test_dpgnn_training.py`, `test_dpgnn_accounting.py` —
  DP-GNN sampler semantics, padded per-root gradients, global clipping/noise,
  physical-chunk equivalence, and hypergeometric accounting.

## Things worth knowing before reading SparseGNN results

- **Aggregator.** `--aggr mean` uses GraphSAGE neighbor means; `--aggr gin`
  uses neighbor sums and a two-layer MLP; `--aggr gcn` uses symmetric degree
  normalization. Sparse and padded implementations compute the same function
  on a given rooted subgraph. This does not imply equality with full-graph
  inference: edge sampling, expansion depth, and degree caps can remove needed
  context. GCN additionally depends on source degrees at the subgraph boundary.
- **Separate graphs.** Training always uses the loader's training graph or the
  graph induced by `train_mask`; evaluation always receives the separate,
  uncapped test graph.
- **Metrics.** For multilabel tasks, micro-F1 depends on a fixed decision
  threshold and can be misleading for poorly calibrated predictions. AUROC
  is recorded alongside micro-F1 to measure ranking quality.
- **Inductive settings differ.** GraphSAINT releases supply training-only
  adjacency. For ogbn-arxiv, Flickr, Reddit, and other single graphs,
  `src.experiments.sparse` drops arcs whose endpoints are not both training nodes.
