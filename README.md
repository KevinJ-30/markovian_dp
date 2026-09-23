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
   `N(0, (sigma*C)^2 I)` is added. `sigma` is the Opacus noise multiplier.

The composition of both sampling stages amplifies privacy beyond what
Poisson subsampling alone gives, which is what the dominating pairs in
`src/privacy/accounting.py` quantify. I'm making an edit here for no specific reason.

## Layout

```
src/
  data/
    datasets.py           dataset dispatch and graph loaders
    relbench.py           RelBench database -> homogeneous directed graph
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
loops and graph protocols remain separate. Import definitions from their owning
modules rather than package-level facades.

Entry points:
- `python -m src.experiments.sparse` — SparseGNN sweeps.
- `python -m src.experiments.compute_epsilon` — post-hoc privacy accounting.
- `python -m src.experiments.run` — graph-disjoint baseline comparisons.

The old SparseGNN import and CLI paths have been removed. Existing command-line
flags, dataset/split caches, and result filenames and schemas are unchanged.

## Install

```bash
pip install torch torch_geometric ogb opacus dp_accounting scipy pandas matplotlib pytest
```

`relbench` is needed only for RelBench datasets.

## Datasets

Most datasets download themselves on first use, into `data/` (gitignored).
Planetoid, OGB, and PyG's Reddit/Flickr/PPI need no setup, while RelBench pulls
its databases through the `relbench` package.

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
    "epochs": 100
  }
}
```

The same dataset metadata is consumed by the first-party `mlp`, `dp_mlp`,
`graphsage`, `dpar`, and `dp_gnn` methods and by the retained ProGAP adapter.
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
python -m src.experiments.sparse --dataset ppi --model multilabel_gnn --direction in \
    --dp --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma 5 \
    --K_in 5 --K_out 5 --lr 0.3 --seeds 3 --track_every 50 \
    --out_dir results/ppi/myrun

# 2. attach epsilon
python -m src.experiments.compute_epsilon \
    --csv results/ppi/myrun/sparse_gnn_ppi_dp_results.csv --delta 1e-6
```

`--track_every N` evaluates every N steps and writes one CSV row per
checkpoint. Since epsilon grows with the step count, a single run then yields a
whole privacy–utility curve, and each checkpoint carries the guarantee for the
model as released at that step. Evaluation consumes no sampling randomness, so a
tracked run follows exactly the same trajectory as an untracked one.

Higher-level drivers live in `scripts/`: `ladder_stage01.sh` (baselines and the
sparsification sweep, no DP), `ladder_stage2.sh` (clip+noise, then epsilon), and
`sweep.sh <axis>` for one-axis tuning.

### Parameters that price epsilon, and parameters that do not

Only `p1`, `p2`, `r`, `K_in`/`K_out`, `sigma`, and `T` enter the accounting.
`sigma` is a noise multiplier, not an absolute noise scale: the clipping norm
`C` bounds each rooted subgraph's gradient (`||g0|| <= C`), and the noise
actually injected is `N(0, (sigma*C)^2 I)`. The dominating-pair reduction
divides sensitivity and noise through by the same `C`, so only `sigma`
survives in the privacy formula. Concretely, changing `C` while holding
`sigma` fixed changes both the sensitivity bound and absolute noise by the same
factor, leaving epsilon unchanged. Neither the learning rate, momentum,
optimizer, nor model depth `L` enters the accountant.

Note `L` and `r` are independent. `r` is the expansion depth and sets the
privacy radius; `L` is the number of GNN layers. An `L`-layer model on an
`r`-hop subgraph still reads only `r` hops, because the subgraph simply does not
contain anything further out — the extra layers add depth, not reach.

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

- **Aggregator.** The default `--aggr mean` (GraphSAGE) makes the rooted-subgraph
  computation *exactly* equal full-graph inference, because its normalizer reads
  only the target's in-neighbourhood, which SparseExpand always materializes in
  full. `--aggr gcn` normalizes by the *source* degree, which a subgraph
  boundary truncates; measured rooted-vs-full relative error is ~0 for mean and
  ~1.1 for gcn, and GCN loses ~27 accuracy points on PPI as a result.
- **Separate graphs.** Training always uses the loader's training graph or the
  graph induced by `train_mask`; evaluation always receives the separate,
  uncapped test graph.
- **Metrics.** On PPI the all-positive predictor scores 0.4608 micro-F1 while
  having no ranking ability at all (AUROC 0.4955), so a model below that floor
  may still be learning. AUROC is recorded alongside micro-F1 for this reason.
- **Inductive settings differ.** PPI and RelBench supply disjoint or temporal
  training graphs. For ogbn-arxiv, Flickr, Reddit, and other single graphs,
  `src.experiments.sparse` always drops arcs whose endpoints are not both training nodes.
