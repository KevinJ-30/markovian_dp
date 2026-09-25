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

### Full final-experiment matrix

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

Private noise is calibrated per configuration at `delta=1/(10*N_train)`.
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

## Cluster helpers

- `_ice_env.sh`: sourced by the ICE Slurm launchers to configure their environment.
- `_matched_eps_grid.sh`: sourced by `sbatch/graphsaint_meps.sbatch` to run its
  matched-budget grid.

These are source-only helpers, not standalone training commands.

## Study-specific campaign helpers

`sparsegnn_final_sweep.py`, `sparsegnn_final_reports.py`,
`sparsegnn_final_verification.py`, and `sparsegnn_partial_nonprivate.py` are
retained for the initial-tuning study. They depend on that study's Python
modules and original layout; they are not general-purpose experiment commands.
Moving the study under `results/old_stuff/` does not automatically migrate
their imports or paths.
