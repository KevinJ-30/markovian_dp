# Supporting scripts

Training entry points are `python -m src.experiments.sparse` and
`python -m src.experiments.run`. The legacy local experiment drivers and
study-specific figure recipes have been removed.

## Reusable utilities

| Script | Purpose |
|---|---|
| `setup_graphsaint.sh` | Extract and check manually downloaded GraphSAINT datasets. |
| `calibrate_grid.py` | Emit noise multipliers for a grid of privacy targets. |
| `ceiling_fullbatch.py` | Run a non-private full-batch classification comparison. |
| `full_matrix.sh` | Run the complete sequential final-experiment grid, then export best-test bootstrap summaries. |
| `full_matrix_run.py` | Execute one matrix cell with method-specific calibration and normalized CSV/JSON results. |
| `ideation_study.py` | Prepare/run/report the approved 280-cell, seed-0 corrected FB gender / Physics screen on GPUs 4–7. |
| `summarize_sweep.py` | Select a validation-best step and report seed-averaged test results; one configuration per child directory. |
| `summarize_matched_eps.py` | Summarize matched-budget studies using their expected directory naming conventions. |
| `summarize_results.py` | Combine arbitrary result CSVs into CSV/Markdown tables using stored bootstrap CIs or seed mean ± sample SD; optional best-test selection and named regimes. |
| `plot_frontier.py` | Plot privacy–utility curves from an explicit CSV glob. |
| `plot_sparse_frontier.py` | Plot a compatible SparseGNN CSV containing epsilon values. |

The Python utilities expose `--help`. GraphSAINT setup accepts an input ZIP
directory and an optional destination; see the root README.

`full_matrix_run.py` defaults to ProGAP propagation depth **1**: two native
training stages, each using `--epochs` (40 stage-epochs at `--epochs 20`).
Use `--progap-depth 2` for three stages. Privacy calibration accounts for the
requested depth; equal per-stage epochs do not make different depths compute-matched.

### FB gender / Physics initial screen

`ideation_study.py` seals **280 configurations**, with adaptive rounds disabled.
Each of five protocols has 24 SGNN-SAGE, 24 SGNN-mean-GIN, and eight ProGAP
configurations: batch `{256,1024}`, LR `{0.01,0.001}`, epsilon `{2,8}`, and
SGNN-only p2 `{0.1,0.5,1}`. Training seed is **0 only**; bootstrap is disabled.
SGNN uses 20 epochs, hidden 128, dropout 0.5, radius 1, outgoing cap 10,
incoming sampling cap 20, clip 1, weight decay 0.0005, p1=batch/N_train, and
the repository's chi=2 accountant. This fixed study explicitly pins ProGAP to
depth 2, independently of the runner default: 20 epochs per each of three native
stages, degree bound 10, and its existing weight decay **0**.
Every method uses delta=1/N_train; epsilon is per run, not sweep-composed.

| Protocol | Training schools / graph | N_train |
|---|---|---:|
| `fb100-gender-1` | Johns Hopkins | 4,762 |
| `fb100-gender-3` | previous + Caltech, Amherst | 7,497 |
| `fb100-gender-6` | previous + Reed, Brandeis, Princeton | 17,782 |
| `fb100-gender-16` | all schools except Cornell and Penn | 145,535 |
| `coauthor-physics` | seed-0 stratified 60% induced training graph | 20,696 |

All FB protocols validate on Cornell (16,822 nodes) and test on Penn
(38,815 nodes). `facebook100-gender` maps raw gender 1/2 to 0/1 and excludes
unknown raw-0 nodes before inducing graphs. Its 13,778-column categorical
vocabulary is fitted on all 18 raw schools, including unknown-label nodes.
Old `facebook100` missingness-target results remain separate and incomparable.
Physics uses raw PyG features and one shared seed-0 stratified 60/20/20
graph-disjoint split. Mean-GIN averages **neighbors only**, then adds the root
before the unchanged GIN MLP; fixed epsilon_GIN=0.

```bash
PYTHON=/usr/scratch/asaha92/envs/graph_subsampling/bin/python
OUT_ROOT=results/ideation_gender_physics_initial
$PYTHON scripts/ideation_study.py dry-run
$PYTHON scripts/ideation_study.py prepare --out-root \"$OUT_ROOT\"
$PYTHON scripts/ideation_study.py run --out-root \"$OUT_ROOT\" \\
  --gpus 4,5,6,7 --max-jobs-per-gpu 2
$PYTHON scripts/ideation_study.py report --out-root \"$OUT_ROOT\"
$PYTHON scripts/ideation_study.py verify --out-root \"$OUT_ROOT\"
```

Use a fresh root for preparation; `run --resume` revalidates frozen source,
packages, prepared partitions, and attempt evidence. Only physical GPUs 4–7
are permitted. Each GPU has one controller-owned cooperative lease and at
most two concurrent workers. Unseen shapes run alone; sharing requires a
successful full-run peak profile for both shapes, verified owned processes,
utilization <=70%, host capacity, 25% memory margin plus 512 MiB per job, and
at least 10% device headroom (minimum 2 GiB). ProGAP profiles include its child
process tree. Busy/full-memory jobs remain exclusive; unrelated jobs are never
preempted. Profiles are relearned after OOM and reconstructed on resume.

`requests.json` and `campaign_manifest.json` define the immutable initial
screen; `queue_state.json` is progress, not scientific evidence. Reports retain
all attempts in `attempts.csv` and per-request results/coverage in
`results.csv`/`coverage.csv`, select configurations by
**validation** score within protocol/method/epsilon (ordinal breaks ties),
and report the selected final test score in `summary.csv` and `comparison.csv`.
One seed and no bootstrap means uncertainty is unavailable, not zero.

### SparseExpand paper ablations

All **60 one-factor configurations** belong to one `OUT_ROOT`, following the
`full_matrix.sh` layout. The study uses `ogbn-arxiv`, `saint-yelp`, and
`twitch-allbut2`, both SAGE and GIN backends, epsilon 8, and training seed 0.
The anchor is radius 1, edge-retention probability 0.5, and outgoing-degree
cap 10. Vary radius `{1,2,3}`, probability `{0.05,0.1,0.25,0.5,1}`, or outgoing
cap `{5,10,20,40}` while holding the other two at the anchor. The shared anchor
runs once; there are no probability-by-cap interactions.

Batch 256, LR 0.01, 20 epochs, hidden 128, two GNN layers, dropout 0.5, and
seed-0 splits stay fixed. Expansion depth does not change architecture depth.
The outgoing preprocessing cap is distinct from the fixed 20-edge incoming
sampling cap; `p2=1` is not an uncapped full-graph baseline. Noise is recalibrated
per configuration at the dataset-specific delta using the repository's fixed
chi=2 mixture formula. This setting alone does not establish a privacy guarantee.

Sequential execution, followed by rendering:

```bash
PYTHON=/path/to/environment/bin/python DEVICE=cuda \
  OUT_ROOT=results/sparse_ablation_ofat \
  bash scripts/sparse_ablation_ofat.sh
```

Alternatively, use the existing opportunistic multi-GPU queue, validation, and
rendering pipeline:

```bash
PYTHON=/path/to/environment/bin/python \
  OUT_ROOT=results/sparse_ablation_ofat \
  bash scripts/sparse_ablation_paper.sh
```

Choose one mode, not both. Add `--dry-run` to either script to preview the
60 worker commands and rendering commands without creating files or training.
Paths are relative to the repository; use a fresh `OUT_ROOT` for new training.
Queued resumes/retries use `full_matrix_queue.py --ablation-ofat --batch-size 256`
with the same root; diagnose cleanly exited failures before `--retry-failed`.
`--report-only` validates completed outputs without launching training.
Changed source fingerprints require a fresh training study, not mixed versions.
Current campaign validation and ablation rendering require chi=2 metadata and
reject historical chi=1 SparseGNN artifacts rather than relabeling them. Keep
historical outputs and their source snapshots unchanged; use a fresh chi=2
study for current rendering or depth-baseline comparisons.

To extend a completed SGNN study with **27 depth-baseline runs**, use the same
supervised pathway with `OFAT_ROOT` pointing to a completed chi=2 60-run study:

```bash
PYTHON=/usr/scratch/asaha92/envs/graph_subsampling/bin/python GPUS=auto \
  OFAT_ROOT=results/sparse_ablation_ofat_chi2 \
  OUT_ROOT=results/sparse_ablation_depth_chi2 \
  bash scripts/sparse_ablation_paper.sh
```

This runs DP-GNN-SAGE and DP-GNN-GIN at radius `{1,2,3}` and ProGAP at depth
`{1,2,3}` on the same three datasets. Epsilon 8, seed 0, batch 256, LR 0.01,
hidden 128, dropout 0.5, and 20 epochs remain fixed. ProGAP uses the existing
separate interpreter configured by `full_matrix_records.PROGAP_PYTHON`; its
20 epochs apply **per stage**, and depth `d` trains `d+1` stages. DP-GNN uses
20 training-population epochs at each radius. Both baselines retain degree
bound 5 and recalibrate noise for the complete requested schedule.

The original SGNN study is read-only. The new root gets its own manifest,
source snapshot, attempts, accepted runs, tables, and comparative figures.
`--dry-run` prints all 27 baseline worker commands without launching them.
Resume/report/retry this study with
`full_matrix_queue.py --ablation-depth-baselines --batch-size 256 --out-root OUT_ROOT`;
add `--report-only` or, after diagnosing failed attempts, `--retry-failed`.

Re-render a completed comparison without training:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925 \
  --depth-root results/sparse_ablation_depth_supervised_20260925 \
  --out-dir results/sparse_ablation_depth_supervised_20260925/figures_rebuilt
```

`--depth-root` requires all 27 baseline runs and verifies matching dataset/task/
partition evidence across both studies. Its default output is `DEPTH_ROOT/figures`.

The renderer consumes that **same run folder**, without retraining:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925 \
  --out-dir results/sparse_ablation_ofat_supervised_20260925/figures_compact
```

Layout:

```text
OUT_ROOT/
  runs/<dataset>/<backend>/<configuration>/seed0/
  manifest.json
  source_snapshot/
  figures/
    ablation_sage.png
    ablation_sage.pdf
    ablation_gin.png
    ablation_gin.pdf
    per_run.csv
    curves.csv
    provenance.json
```

Sequential workers write directly under `runs/`, with logs under `logs/`.
The supervised queue retains immutable `attempts/` and publishes relative
`runs/` links to accepted outputs; it also writes `requests.json`,
`queue_state.json`, `results.csv`, and `summary.csv`/`summary.md` at the run root.
Keep the entire folder when archiving. Historical source snapshots and invocation
paths remain unchanged evidence, even if the current renderer has moved.

Each backend gets three horizontal panels: **(a) depth lines**, **(b) probability
bars**, and **(c) outgoing-cap bars**, with labels inside the upper-left corners.
Dataset colors are shared across all panels. In panel (a), SGNN is solid,
ProGAP dashed, and DP-GNN dotted; the latter matches the SAGE/GIN backend, while
the same ProGAP runs appear in both figures. Without `--depth-root`, only SGNN
depth curves appear. Depth lines are fully opaque and 3 points wide, with no
uncertainty whiskers. Panels (b)/(c) remain SGNN-only grouped bars with intervals.
The shared legend sits to the left and contains only dataset colors and method
line styles; backend and privacy headings are omitted. Backend identity remains
in the filenames and privacy settings in the CSV/provenance. The compact canvas
is 18.7 × 3.6 inches (before tight cropping), with a shared 0–1 metric scale.
The y-axis reads "Test metric": accuracy for ogbn-arxiv, micro-F1 for Yelp, and
AUROC for Twitch. These different metrics are not averaged. Bar panels show the
exact stored 95% node-bootstrap endpoints (1,000 resamples, bootstrap seed 0)
from each validation-selected checkpoint. Depth-panel intervals are omitted
visually but retained in the CSV exports. These intervals quantify test-node
uncertainty, not training-seed variability or dependence between graph nodes.

Depth has method-specific meaning: SGNN changes expansion radius while retaining
its two-layer network; DP-GNN changes actual message-passing depth; ProGAP changes
progressive aggregation depth and stage count. These are not equal architectures
or equal training schedules. DP-GNN's influence bound is
`min(N_train, 1 + K + ... + K^r)` (6/31/156 for K=5 before population clipping),
conditional on a fixed sampled topology. It does not establish a raw-topology
node-deletion guarantee or account for data-dependent preprocessing.

`per_run.csv` retains all 60 SGNN runs (87 with depth baselines); `curves.csv`
contains 72 SGNN points (99 with baselines), because the SGNN anchor is referenced
in all three parameter panels without extra training. ProGAP curve records are
stored once and reused visually in both backend figures.
Both preserve peak process RSS, peak CUDA allocation, calibration/training time,
sampled-node/edge statistics, raw result paths, and exact intervals. Diagnostics
are retained as data, not separate plots or additional DP releases. CUDA memory
is runner-process allocator peak, RSS is runner-process lifetime high-water mark,
and CPU CUDA values are unavailable rather than zero. ProGAP trains in a child
process, so these runner metrics do not measure its child-process memory usage.
Timing includes calibration, training, and final evaluation but excludes loading.

The renderer verifies manifests, artifact hashes, actual settings, and checkpoint
selection. It never overwrites a figure directory. For another reconstruction,
pass a fresh `--out-dir`, as in the `figures_compact` example above; choose a new
directory name for subsequent renders. Provenance records input/output hashes,
analysis sources, and versions. Existing full-matrix studies and original raw
ablation outputs are never rewritten.

### Full final-experiment matrix

For the fixed **1,584-run** campaign (528 configurations, three seeds each),
start the direct opportunistic queue in a fresh output root:

```bash
python scripts/full_matrix_queue.py --out-root results/full_matrix_k10_s012_YYYYMMDD --gpus auto
```

The main matrix fixes requested batch size at 1024. Historical single-seed
campaigns have different registries and must not be resumed with this grid.
The separate ablation modes retain their batch-256 and bootstrap contracts.


It admits authorized GPUs at utilization strictly below 30% with positive free
memory, including GPUs with other workloads. Two distinct eligible samples and
a fresh probe under the UUID lock are required. Running workers are not cancelled
when utilization rises. Each attempt has a **3,600-second wall deadline**,
including loading, calibration, and training; owned processes receive termination
and then forced cleanup after the standard grace period. Timed-out jobs retain
their logs and artifacts, and the next pending job fills the GPU slot. OOMs
retain their attempt evidence and defer for five minutes after three attempts.
Reuse the same output root to resume. After diagnosing a cleanly exited failure,
use `--retry-failed` to retry it in a new attempt directory; unresolved ownership
remains blocked. `--report-only` validates saved commitments and regenerates tables
without GPU discovery or process recovery, returning zero only for 1,584 valid results.
These two flags are mutually exclusive.

`summary.md`, `summary.csv`, and `results.csv` retain every run in registry order,
including pending, failed, and timed-out rows, actual parameters, and original
per-attempt CSV paths. Raw attempt outputs are never rewritten by reporting.
The queue also publishes CSV/Markdown pairs:

- `seed_summary`: mean and sample SD across available seeds for every configuration;
  the `n` and `seeds` columns expose incomplete cohorts.
- `best_test`: highest mean test score across configurations within each comparable
  dataset/privacy/method cell, using only complete seed sets `0;1;2`.
- `best_test_by_p2`: the same selection while retaining a separate SparseGNN group
  for each `p2`; baseline methods are grouped separately.

Bootstrap is disabled. Best-test selection is explicitly labeled as test-selected
and does not provide unbiased model-selection estimates. Checkpoints within each
training run are still selected by validation. No separate dataset-preparation
campaign is required.

The general sequential shell utility remains available:

Preview without loading data, creating outputs, or allocating a GPU:

```bash
bash scripts/full_matrix.sh --dry-run
```

Run from the repository root with the appropriate Python environments:

```bash
PYTHON=python PROGAP_PYTHON=/path/to/progap/bin/python \
  bash scripts/full_matrix.sh
```

`PROGAP_PYTHON` defaults to `PYTHON`; use a separate environment if the retained
ProGAP implementation's dependencies differ. Dataset assets and training
dependencies must already be available. The launcher is sequential, defaults to
`DEVICE=cuda`, performs no GPU scheduling, stops on failure, and refuses existing
run directories. Use a fresh `OUT_ROOT` for a new campaign.

The default grid contains **1,584 runs total**: non-private MLP/GraphSAGE/GIN,
private DP-MLP/ProGAP/DPAR/DP-GNN-SAGE/DP-GNN-GIN/SparseGNN-SAGE/SparseGNN-GIN,
learning rates `0.01 0.001`, requested batch `1024`, epochs `20`, private epsilon
targets `2 8`, seeds `0 1 2`, and SparseGNN-only `p2=0.1 0.25 0.5 0.75 1.0`.
Hidden sizes are 64 for MLP/DP-MLP and 128 for graph methods; dropout is 0.5.
GraphSAGE/GIN use fanout 10, SparseGNN outgoing cap 10, DP-GNN/ProGAP degree
bound 10, and DPAR PPR `topk=10`. These controls have method-specific meanings;
PPR top-k and neighborhood fanout are not graph-degree guarantees. MLPs have no
graph-degree parameter. No bootstrap resampling is performed.

The eight protocols are `ogbn-arxiv`, `ogbn-products`, `saint-reddit`,
`saint-yelp`, `saint-amazon`, `twitch-allbut2`, `facebook100-allbut2`, and
`mag-allbut2`. The last three train on all registered domains except the
validation/test pair: respectively `engb/es`, `cornell5/penn94`, and `cn/de`.
Splits remain fixed at seed 0 across training seeds.

Private noise is calibrated per configuration at `delta=1/N_train`.
SparseGNN uses `p1=min(batch_size,N_train)/N_train`, `r=1`, directed outgoing
degree cap 10, clip 1, and the fixed chi=2 mixture formula. This setting alone
does not establish a privacy guarantee. SparseGNN and DP-GNN
use `E*ceil(N_train/effective_batch)` updates and validate each such epoch.
ProGAP retains its native **E epochs per stage**, three stages, and drop-last
batch convention. DPAR retains `ppr_num=70` and `sampled_train_rate=0.09`,
with unchanged native privacy accounting. Its requested batch 1024 therefore
has an effective batch of at most 70 released roots and one update per epoch;
smaller sampled graphs use fewer roots.
Actual root counts, effective batches, updates, and calibration are recorded.

Outputs default to `results/full_matrix/`: isolated
`runs/<dataset>/<method>/<privacy>/<regime>/seed<seed>/` directories contain
`config.json`, `result.json`, and `result.csv`; `logs/` contains per-run logs.
After every run succeeds, the launcher writes `summary.csv` and `summary.md`.
Regenerate those tables independently, including from a partially completed
campaign's successful runs:

```bash
python scripts/summarize_results.py 'results/full_matrix/runs/**/result.csv' \
  --seed --best --out results/full_matrix/summary
```

The summary selects the configuration with the highest mean test result within
each comparable dataset/privacy/method-backbone cell, retaining its sample SD;
it labels test-selection bias. To inspect every configuration, omit `--best`.
The generic reporter exposes partial cohorts through `n`; the supervised queue's
best-test tables additionally require all three seeds before admitting a winner.
Space-separated environment overrides `DATASETS`, `METHODS`, `EPSILONS`,
`SEEDS`, `LEARNING_RATES`, `BATCH_SIZES`, `EPOCHS`, and `P2_VALUES` can restrict or
extend the grid. `BOOTSTRAP_RESAMPLES` controls the final-test bootstrap count.
See `bash scripts/full_matrix.sh --help` for the complete launch interface.

### Result tables

`summarize_results.py` is standalone and uses only the Python standard library.
Pass CSV files and/or quoted globs; datasets, methods, metrics, and privacy
budgets come from the rows, not directory names. For example:

```bash
python scripts/summarize_results.py 'results/campaign/**/*.csv' \
  --bootstrap --best --out reports/bootstrap

python scripts/summarize_results.py results/run1.csv results/run2.csv \
  --seed --out reports/seeds
```

Each invocation writes both `PREFIX.csv` and `PREFIX.md`. Choose one mode:

- `--bootstrap`: original final-run score with its stored bootstrap CI. Markdown
  uses `$score \pm radius$`, where
  `radius = max(abs(score - lower), abs(upper - score))`: a conservative symmetric
  envelope, not a change to the point estimate. Exact original CI endpoints and
  confidence levels remain in the CSV. Missing CIs show `N/A`; they are not
  recomputed or averaged across seeds.
- `--seed`: mean and sample SD (`ddof=1`) across unique seeds **within each
  configuration**. One seed gives `N/A` SD. Conflicting results for the same
  seed/configuration are errors; use separate groups for distinct runs.

`--best` deliberately selects on **test**, not validation: bootstrap mode picks
the highest-scoring final run and its CI; seed mode picks the configuration with
the highest mean test score and retains its mean/SD. This introduces test-selection
bias, which the output labels. It never chooses an intermediate checkpoint by
test score. Without `--best`, all final runs/configurations remain in the table.
`--metric auto` respects declared primary metrics; `--metric accuracy`, `auroc`,
`micro_f1`, `r2`, or `macro_f1` selects a particular available test metric.
Scores stay on their input scale.

For named regimes, create a JSON configuration such as:

```json
{
  "groups": [
    {
      "name": "small learning rate",
      "files": ["results/campaign/**/*.csv"],
      "where": {"lr": [0.001], "p1": 0.01}
    },
    {
      "name": "large learning rate",
      "files": ["results/campaign/**/*.csv"],
      "where": {"lr": [0.01], "p1": 0.01}
    }
  ]
}
```

Run `python scripts/summarize_results.py --groups groups.json --bootstrap --best
--out reports/regimes`. Group paths are relative to the JSON file. Optional
`where` filters use raw CSV column names and scalar or list values; numeric
strings compare numerically. Overlapping file globs are deduplicated per group.
Positional inputs, if supplied as well, form a separate `default` group.

Dataset/split, method, metric, privacy budget, and named groups stay separate.
Hyperparameter aliases are normalized; calibrated noise does not split seed
cohorts at a fixed target epsilon. Non-private rows are labeled `non-private`,
never inferred from `epsilon_context`; private rows without epsilon are labeled
`unknown`. For unknown private epsilon, `--best` keeps different configurations
separate rather than comparing unrecorded privacy budgets. The CSV includes
configuration IDs/settings, selected seeds, source paths, and selection labels.

## Study-specific campaign helpers

`sparsegnn_final_sweep.py`, `sparsegnn_final_reports.py`,
`sparsegnn_final_verification.py`, and `sparsegnn_partial_nonprivate.py` are
retained for the initial-tuning study. They depend on that study's Python
modules and original layout; they are not general-purpose experiment commands.
Moving the study under `results/old_stuff/` does not automatically migrate
their imports or paths.
