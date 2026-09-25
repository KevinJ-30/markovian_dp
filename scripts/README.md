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
| `summarize_sweep.py` | Select a validation-best step and report seed-averaged test results; one configuration per child directory. |
| `summarize_matched_eps.py` | Summarize matched-budget studies using their expected directory naming conventions. |
| `summarize_results.py` | Combine arbitrary result CSVs into CSV/Markdown tables using stored bootstrap CIs or seed mean ± sample SD; optional best-test selection and named regimes. |
| `plot_frontier.py` | Plot privacy–utility curves from an explicit CSV glob. |
| `plot_sparse_frontier.py` | Plot a compatible SparseGNN CSV containing epsilon values. |

The Python utilities expose `--help`. GraphSAINT setup accepts an input ZIP
directory and an optional destination; see the root README.

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
per configuration at the dataset-specific delta. The current chi=1,
`union_safe=False` accounting policy is retained, not silently corrected.

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

The renderer consumes that **same run folder**, without retraining:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925
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

Each backend gets one chart with **three horizontally arranged grouped bar
panels**: radius, probability, and outgoing cap. Each parameter label has three
dataset-colored bars, identified by a shared legend; there is no chart title.
The y-axis reads "Test metric": accuracy for ogbn-arxiv, micro-F1 for Yelp, and
AUROC for Twitch. These different metrics are not averaged. Bars retain the exact stored
95% node-bootstrap endpoints (1,000 resamples, bootstrap seed 0) from each
validation-selected checkpoint. They quantify test-node uncertainty, not
training-seed variability or dependence between graph nodes.

`per_run.csv` retains all 60 runs; `curves.csv` contains 72 plotted points because
the anchor is referenced in all three parameter panels, without extra training.
Both preserve peak process RSS, peak CUDA allocation, calibration/training time,
sampled-node/edge statistics, raw result paths, and exact intervals. Diagnostics
are retained as data, not separate plots or additional DP releases. CUDA memory
is allocator peak, RSS is process-lifetime high-water mark, and CPU CUDA values
are unavailable rather than zero. Timing includes calibration, training, and
final evaluation but excludes loading.

The renderer verifies manifests, artifact hashes, actual settings, and checkpoint
selection. It never overwrites a figure directory. For another reconstruction,
pass `--out-dir OUT_ROOT/figures_rebuilt`. Provenance records input/output hashes,
analysis sources, and versions. Existing full-matrix studies and original raw
ablation outputs are never rewritten.

### Full final-experiment matrix

For the fixed **336-configuration** campaign, start the direct opportunistic queue:

```bash
python scripts/full_matrix_queue.py --out-root results/full_matrix_queued_YYYYMMDD
```
For a separate batch-256 study, add `--batch-size 256` and use a fresh output
root. Supply the same flag when resuming or using `--report-only`; a mismatched
saved grid is rejected. The default batch-1024 grid and existing results remain
unchanged.


It admits authorized GPUs at utilization strictly below 30% with positive free
memory, including GPUs with other workloads. Two distinct eligible samples and
a fresh probe under the UUID lock are required. Running workers are not cancelled
when utilization rises. Healthy workers have no wall-time limit; OOMs retain their
attempt evidence and defer for five minutes after three attempts.
Reuse the same output root to resume. After diagnosing a cleanly exited failure,
use `--retry-failed` to retry it in a new attempt directory; unresolved ownership
remains blocked. `--report-only` validates saved commitments and regenerates tables
without GPU discovery or process recovery, returning zero only for 336 valid results.
These two flags are mutually exclusive.

`summary.md`, `summary.csv`, and `results.csv` retain every configuration in registry
order, including pending and failed rows. The summaries include validation-selected
scores, task-specific metrics, exact stored bootstrap endpoints, actual JSON
parameters, and original per-attempt CSV paths. No winners or rankings are selected.
Raw attempt outputs are never rewritten by reporting. No separate
dataset-preparation or smoke campaign is required.

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

The default grid contains **1,848 runs per seed**: non-private MLP/GraphSAGE/GIN,
private DP-MLP/ProGAP/DPAR/DP-GNN-SAGE/DP-GNN-GIN/SparseGNN-SAGE/SparseGNN-GIN,
learning rates `0.01 0.001`, batches `256 1024`, epochs `10 20`, private epsilon
targets `2 8`, and SparseGNN-only `p2=0.5 0.1`. Hidden sizes are 64 for MLP/DP-MLP
and 128 for all graph methods; dropout is 0.5. The default training seed is 0.
All methods use validation-selected final test results and 95% node-bootstrap CIs
with 1,000 resamples.

The 11 protocols are `ogbn-arxiv`, `ogbn-products`, `reddit`, `facebook`,
`saint-reddit`, `saint-yelp`, `saint-flickr`, `saint-amazon`, `twitch-allbut2`,
`facebook100-allbut2`, and `mag-allbut2`. The last three train on all registered
domains except the validation/test pair: respectively `engb/es`,
`cornell5/penn94`, and `cn/de`. Splits remain fixed at seed 0 across training seeds.

Private noise is calibrated per configuration at `delta=1/N_train`.
SparseGNN uses `p1=min(batch_size,N_train)/N_train`, `r=1`, directed outgoing
degree cap 10, clip 1, and the current chi=1 accountant. SparseGNN and DP-GNN
use `E*ceil(N_train/effective_batch)` updates and validate each such epoch.
ProGAP retains its native **E epochs per stage**, three stages, and drop-last
batch convention. DPAR retains its native defaults: `ppr_num=70` and
`sampled_train_rate=0.09`, with unchanged native privacy accounting. With 70
released roots, requested batch sizes 256 and 1024 both give effective batches
of 70 and one update per epoch; smaller sampled graphs use fewer roots.
Actual root counts, effective batches, updates, and calibration are recorded.

Outputs default to `results/full_matrix/`: isolated
`runs/<dataset>/<method>/<privacy>/<regime>/seed<seed>/` directories contain
`config.json`, `result.json`, and `result.csv`; `logs/` contains per-run logs.
After every run succeeds, the launcher writes `summary.csv` and `summary.md`.
Regenerate those tables independently, including from a partially completed
campaign's successful runs:

```bash
python scripts/summarize_results.py 'results/full_matrix/runs/**/result.csv' \
  --bootstrap --best --out results/full_matrix/summary
```

The summary selects best test results across the grid within each comparable
dataset/privacy/method-backbone cell; it labels test-selection bias. To inspect
every configuration instead, omit `--best`.
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
