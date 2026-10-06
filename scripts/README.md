# Supporting scripts

Use `python scripts/run_experiments.py CONFIG.json --gpus auto` for new
experiments. One JSON-driven runner schedules all supported methods; dataset
presets and training implementations remain in its single-run worker.

## Reusable utilities

| Script | Purpose |
|---|---|
| `run_experiments.py` | Expand JSON configurations, pack owned GPU jobs, retain logs/results, and resume. |
| `run_experiment.py` | Execute one experiment with method-specific calibration and normalized CSV/JSON results. |
| `make_repeat_config.py` | Generate additional-seed jobs from validation winners of a complete single-seed study. |
| `runner_runtime.py` | Shared GPU authorization, process supervision, and ownership-safe cleanup. |
| `summarize_sweep.py` | Select a validation-best step and report seed-averaged test results; one configuration per child directory. |
| `summarize_matched_eps.py` | Summarize matched-budget studies using their expected directory naming conventions. |
| `summarize_results.py` | Combine arbitrary result CSVs into CSV/Markdown tables using stored bootstrap CIs or seed mean ± sample SD; optional best-test selection and named regimes. |
| `sparse_ablation.py` | Plot five-seed ablation means ± standard errors as three line panels, with dataset colors and SAGE/GIN line styles. |
| `plot_frontier.py` | Plot privacy–utility curves from an explicit CSV glob. |

The Python utilities expose `--help`. For manual GraphSAINT download and
extraction instructions, see the [root README](../README.md#datasets).

`run_experiment.py` defaults to ProGAP propagation depth **3**: four native
training stages, each using `--epochs` (80 stage-epochs at `--epochs 20`).
Explicit positive depths override it. Privacy calibration accounts for the
requested depth; equal per-stage epochs do not make different depths compute-matched.

All methods default to **`weight_decay=0.0`**, including ProGAP. Set JSON
`weight_decay` or single-run `--weight-decay` to a finite nonnegative value
to override it; explicit values are preserved. The local baseline/DPAR training
configs and standalone SparseGNN CLI also default to zero.
The tuning and repeat JSON configs explicitly set **`weight_decay=0.0005`**
for `mlp` and `dp_mlp`; other methods retain their existing settings.
Historical results retain their recorded weight decay. Use a fresh output root
when changing weight decay; do not select repeats using a modified source config
against historical results.

`main_r1_eps1258_tune.json` covers epsilon **1, 2, 5, 8**, with ProGAP depths
**1 and 5 only**. Its 1,632 tuning jobs form 280 validation-selection groups.
`main_r1_eps1258_repeats.json` already freezes the primary tables' selected
parameters for 1,400 runs across seeds 1–5. Fresh tuning can generate a separate
`main_r1_eps1258_retuned_repeats.json`; historical depth-3 results are retained.

### Unified experiment runner

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 4,5 --dry-run
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 4,5
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 4,5 --resume
```

The output defaults to `results/<name>/` relative to the repository, not the
current directory. `name` defaults to the config filename stem; `--out-dir`
overrides the root. Existing roots require `--resume`; failed jobs are retried
only with `--resume --retry-failed`. Changed scientific parameters require a new
root. GPU choices, concurrency cap, timeout, and ProGAP interpreter may change
on resume. A moved root remains readable for analysis but cannot be resumed.

Example config with eight jobs:

```json
{
  "name": "amazon-sgnn",
  "defaults": {
    "dataset": "saint-amazon", "batch_size": 1024,
    "epochs": 20, "bootstrap_resamples": 0
  },
  "grid": {"seed": [0, 1], "lr": [0.001]},
  "runs": [
    {"parameters": {"method": "sparse_sage"},
     "grid": {"epsilon": [2, 8], "p2": [0.1, 0.5]}}
  ]
}
```

`defaults` and block `parameters` are scalar worker settings. `grid` explicitly
declares Cartesian axes; lists/objects inside scalar settings remain literal.
Block parameters replace a common scalar or axis, and block axes replace a
common axis or scalar. A key cannot be both scalar and axis in the same scope.
Use separate blocks for method-specific epsilon, p2, pooling, or depth settings.
Unknown/inapplicable settings, duplicate scientific jobs, and invalid numbers
are errors. `domain_split` accepts canonical-domain train/val/test lists, seed,
and val_ratio; it cannot override a named preset's domains.

Optional JSON `gpus` accepts an index/UUID list or string; CLI `--gpus` takes
precedence. Inherited GPU visibility is always respected. GPUs must initially
have no compute processes, utilization <=5%, and <=1024 MiB used memory across
two recent observations and a fresh check under the UUID lock. The runner never
preempts other users. Later foreign activity stops additional admissions.

There is **no default jobs-per-GPU cap or wall timeout**. An unprofiled candidate
can share an occupied GPU using **30% of device memory as an estimated peak**,
with the normal sharing margins below; this is not a memory cap. All existing
jobs on that GPU still need measured profiles or explicit estimates before
another admission, so a new unmeasured shape must warm up before further packing.
After GPU activity and 30 seconds of stable observed GPU and host memory peaks,
provisional profiles permit further sharing before the first execution finishes.
Growth above 10% restarts this warm-up; stale or
unverified observations revoke provisional profiles. These profiles are not
persisted; successful whole-run peaks remain the durable measurements. Admission
prefers GPUs with fewer active jobs,
filling eligible idle GPUs before packing busy ones, including waiting for the
second idle observation. The controller refreshes observations between launches
and waits for host-memory evidence before acquiring an idle GPU lease. Admission
revalidates recorded process identities rather than rescanning every host process;
unobserved children block sharing until the observer records them. Discovery and
termination retain exhaustive ownership checks. Shared GPU reservations add 25% plus 512 MiB
per job, leaving at least 2 GiB or 10% device headroom; host reservations also
have a 25% margin. On an idle GPU, a successful measured whole-run peak needs
only that much free memory: sharing margins must not reject a shape that fits
alone. ProGAP profiles include its child process. Missing observations block
further sharing, not healthy training. Optional block `resources` estimates
(`gpu_memory_mib`, `host_memory_mib`) authorize initial packing with the same
margins. Explicit GPU estimates above the measured peak retain the margins even
for solo admission. Estimates and measured peaks are not OOM
guarantees. Shared CUDA OOM triggers one fresh exclusive retry without changing
scientific settings.

Use `--max-jobs-per-gpu N` for an explicit upper bound and
`--timeout-seconds SECONDS` for a wall deadline. `--device cpu` runs serially
and rejects GPU options; CUDA never silently falls back to CPU. `--dry-run`
creates no files, probes no GPUs, and does not import training dependencies.
`--progap-python` selects the separate native environment if needed.
`split_root` paths in configs are repository-relative (for example,
`data/inductive_splits`); `progap_python` paths, including relative CLI overrides,
are relative to the config's directory. Bare executable names such as `python3`
are looked up on `PATH`; use `./python` for an executable beside the config.
Omitting `progap_python` uses the launching interpreter.

Workers load and reconstruct cached datasets independently; there is no
dataset-wide exclusive loading lock. Split and GraphSAINT label caches lock
only first creation and publish atomically. OGB/PyG constructors use shared
locks for concurrent warm reads and exclusive locks for cold download/processing.
Raw domain downloads similarly lock only acquisition or required checksum repair.

All private jobs use the hardcoded target `delta = N_train ** -1.01`, where
`N_train` is the training partition's node count. Historical `1/N_train` results
retain their original budget; use a fresh output root rather than resuming an
older study and mixing delta conventions.

DP-GNN's multi-term accountant evaluates Rényi orders 1.1–199.9 in increments
of 0.1 (`np.arange(1, 200, 0.1)[1:]`). This expands the upstream 1.1–9.9 grid
to support tighter epsilon targets; the accounting formula is unchanged.
Existing results retain their previously calibrated noise and privacy bounds.

DP-MLP calibrates and composes the symmetric pair
`(1-q) N(0, sigma²) + q N(+1, sigma²)` versus
`(1-q) N(0, sigma²) + q N(-1, sigma²)`, with `q = batch_size / N_train`,
using pessimistic PLD discretization at grid `1e-3`. Results identify this as
`dp_accounting.symmetric_gaussian_mixture`; historical Opacus results are unchanged.

DPAR's training RDP accountant lives in `src/privacy/dpar_rdp.py`, alongside
the method-specific privacy wrappers. It preserves the released accountant's
arithmetic and Apache-2.0 header. DPAR training uses `src/training/dpar.py`;
the accountant uses NumPy, SciPy, and `six`, not TensorFlow. Its result
identifier is `dpar.upstream_rdp_accountant`.

Each root contains `experiment.json`, atomic `state.json`, aggregate JSON-lines
`runner.log`, and `results.csv` with one row per planned job. Each attempt is
retained under `runs/<run-id>/attempts/<number>/`: `process.log`, operational
launch/exit/ownership records, and `output/` with config, results, and native
artifacts. Summarize the root CSV rather than globbing all attempts, so retries
do not become extra seeds. No sealed manifests, source snapshots, or hashes are
required. See the [script retirement inventory](RETIRED_SCRIPTS.md) for replacements and removal prerequisites.

### Unified primary-table comparison

`configs/main_r1_eps1258_repeats.json` freezes 280 validation-selected
configurations across all eight datasets and private epsilon targets `{1,2,5,8}`.
Each runs with seeds 1–5, giving 1,400 jobs. MLP and DP-MLP use
`weight_decay=0.0005`; every other method uses zero. Nonprivate configurations
are not duplicated across epsilon values.

The parameters come directly from completed repeat states:

- Non-MLP private methods at epsilon 1/5: `results/main_r1_eps15_zero_decay_repeats/`.
- Non-MLP private methods at epsilon 2/8: `results/main_r1_zero_decay_repeats/`.
- MLP and DP-MLP at all budgets: `results/mlp_wd5e4_eps1258_repeats/`.
- Nonprivate GraphSAGE/GIN: the latest epsilon-1/5 table's configurations.

All 280 choices match their source tuning study's seed-0 validation winners.
The older epsilon-2/8 nonprivate GIN selections differ in batch size for Yelp,
Amazon, and MAG; the unified config uses the latest table's choices of 256,
1024, and 256 respectively. Historical alternatives remain archived.

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus auto
python scripts/summarize_results.py results/main_r1_eps1258_repeats/results.csv \
  --seed --out results/main_r1_eps1258_repeats/performance
```

The summary reports mean ± sample SD over seeds 1–5 in CSV and Markdown.
Do not add `--best` or `--best-validation`: selection was already frozen on
seed 0. The existing historical result roots are not rewritten.

### Fresh unified tuning

`configs/main_r1_eps1258_tune.json` defines 1,632 seed-0 tuning jobs on the same
eight datasets. Batch sizes are `{256,1024}`, learning rates `{0.01,0.001}`, and
private epsilon targets `{1,2,5,8}`. SparseSAGE and SparseGIN (sum) use radius 1
and p2 `{0.1,0.5,1}`. ProGAP depths `{1,5}` remain separate comparisons.
All methods use hidden width 128, dropout 0.5, degree setting 5, bootstrap
resamples 0, and 20 epochs; ProGAP uses those epochs per stage. Weight decay is
0.0005 for MLP/DP-MLP and zero elsewhere.

The main tuning and repeat configs use the launching Python interpreter for
ProGAP by default. ProGAP uses Opacus 1.6.0's `forbid_grad_accumulation()` API
and out-of-place dropout; no separate interpreter is required. The configs
retain automatic GPU selection and the existing split cache.
No per-GPU concurrency limit is imposed.

`configs/main_r1_eps1258_repeat_selection.json` is generator settings, not a
runner config. It selects over `lr`, `batch_size`, and `p2` using validation
only, preserving dataset, method, epsilon, and ProGAP depth. It writes the
separate `configs/main_r1_eps1258_retuned_repeats.json` after tuning completes,
leaving the frozen primary-table config unchanged.

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_tune.json --gpus auto
python scripts/make_repeat_config.py configs/main_r1_eps1258_repeat_selection.json
python scripts/run_experiments.py configs/main_r1_eps1258_retuned_repeats.json --gpus auto
python scripts/summarize_results.py results/main_r1_eps1258_retuned_repeats/results.csv \
  --seed --out results/main_r1_eps1258_retuned_repeats/performance
```

This produces 280 winners × 5 additional seeds = 1,400 repeat jobs, for 3,032
executions including tuning. Retain the selected seed-0 results separately if
a six-seed summary is needed. Exact validation ties keep the first candidate.
The generator refuses incomplete or mismatched studies, nonfinite metrics,
reused tuning seeds, and existing output files. `--results-dir`, `--out`, and
`--seeds` can override generator settings.

Generated repeat configs store split-cache paths relative to the repository root
and recorded interpreter paths relative to the output config's directory, even
when `--out` writes elsewhere. Interpreter symlinks and bare `PATH` names are
preserved. This retains the selected paths and scientific run identities without
copying absolute machine paths from results back into configs. External paths
use `..`; their relative directory layout must be preserved or overridden on a
different machine. Existing result files are not rewritten.

Superseded epsilon-pair, MLP-only, and baseline-depth configs are kept locally in
the Git-ignored `old_configs/` directory. The active ablation tuning configs are
`configs/sparse_ablation.json` (epsilon 8) and `configs/sparse_ablation_eps125.json`
(epsilon 1, 2, and 5).

### Facebook dataset presets

These dataset presets remain available in the scientific worker for custom
JSON configurations. The exploratory sweep config has been removed.

| Protocol | Training schools / graph | N_train |
|---|---|---:|
| `fb100-gender-1` | Johns Hopkins | 4,762 |
| `fb100-gender-3` | previous + Caltech, Amherst | 7,497 |
| `fb100-gender-6` | previous + Reed, Brandeis, Princeton | 17,782 |
| `fb100-gender-16` | all schools except Cornell and Penn | 145,535 |

All FB protocols validate on Cornell (16,822 nodes) and test on Penn
(38,815 nodes). `facebook100-gender` maps raw gender 1/2 to 0/1 and excludes
unknown raw-0 nodes before inducing graphs. Its 13,778-column categorical
vocabulary is fitted on all 18 raw schools, including unknown-label nodes.
Old `facebook100` missingness-target results remain separate and incomparable.
Mean-GIN averages **neighbors only**, then adds the root before the unchanged
GIN MLP; fixed epsilon_GIN=0.

The separate worker protocol `--dataset fb100-year-6` reuses the six training
schools above, Cornell validation, and Penn test, but loads `facebook100-year`.
It keeps only years 2004–2009 (classes 0–5), excludes year from input features,
and induces the retained-node graphs: **16,557 / 15,374 / 33,748** nodes in
train/validation/test, with **13,697** features. The default ProGAP depth 3
has four native stages (80 stage-epochs at `--epochs 20`); SGNN still uses
20 epochs. Historical
hyperparameters can be reused, but noise and delta must be recalibrated for
the year-task population using the current accountants.

### SparseExpand paper ablations

All **240 tuning runs** in `configs/sparse_ablation.json` share one output root.
The study uses `saint-yelp`, `twitch-allbut2`, and `mag-allbut2`, both SparseSAGE
and SparseGIN, epsilon 8, and training seed 0. DP-GNN and ProGAP are excluded.
The anchor is radius 1, edge-retention probability 0.1, and outgoing-degree
cap 5, matching the main experiments. Vary radius `{1,2,3}`, probability `{0.05,0.1,0.25,0.5,1}`, or outgoing
cap `{5,10,20,40}` while holding the other two at the anchor. The shared anchor
runs once for each dataset, method, batch size, and learning rate; there are no
probability-by-cap interactions. The 10 unique one-factor settings cross batch
sizes `{256,1024}` and learning rates `{0.01,0.001}`, matching the main grids.
This gives 10 × 3 datasets × 2 methods × 2 batch sizes × 2 learning rates = 240.

The secondary config `configs/sparse_ablation_eps125.json` uses the same datasets,
methods, ablation points, and batch-size/LR grid at epsilon `{1,2,5}`: **720 tuning
runs**, or **960** across both configs. Run it with
`python scripts/run_experiments.py configs/sparse_ablation_eps125.json --gpus auto`;
its default output root is `results/sparse_ablation_eps125`. Both configs use
seed 0. `configs/sparse_ablation_eps125_repeat_selection.json` selects batch
size/LR independently within each epsilon/dataset/method/ablation point and
generates `configs/sparse_ablation_eps125_repeats.json` after tuning completes.
The 180 winners × seeds 1–5 give **900 final runs**, with weight decay 0.0,
under `results/sparse_ablation_eps125_repeats`. The chained tuning, selection,
and repeat commands in [`reproduce.md`](../reproduce.md#ablation-studies)
start each stage only if its predecessor succeeds.

Twenty epochs, hidden width 128, dropout 0.5, and seed-0 splits stay fixed.
SparseGNN uses two message-passing layers at radius 1/2 and three at radius 3.
Incoming expansion caps each expanded node at 20 neighbors on hop 1, 10 on hop 2,
and 5 on hop 3 and beyond, after Bernoulli edge thinning. At radius 3 this bounds
each rooted subgraph by 1 + 20 + 200 + 1,000 = 1,221 nodes. The outgoing
preprocessing cap is separate; `p2=1` is not an uncapped full-graph baseline.
Both samplers use this schedule globally; the main radius-1 settings are unchanged.
Results record the per-hop list as `parameters.incoming_sampling_caps` and the
actual architecture depth as `parameters.layers`. The standalone SparseGNN CLI
uses the same depth defaults, permits explicit `--num_layers` overrides, and
records the schedule in its `incoming_sampling_caps` CSV column.
Radius now changes both the sampled receptive field and, at radius 3, network
depth; this is not a fixed-architecture radius comparison.
Noise is recalibrated
per configuration at the dataset-specific delta using the repository's mixture
formula with non-root shells `2*K_out**ell`. This setting alone does not establish
a privacy guarantee.

Run the unified study and summarize all candidates:

```bash
python scripts/run_experiments.py configs/sparse_ablation.json --gpus auto \
  --out-dir results/sparse_ablation_yelp_twitch_mag_k5_tune
python scripts/summarize_results.py results/sparse_ablation_yelp_twitch_mag_k5_tune/results.csv \
  --bootstrap --out results/sparse_ablation_yelp_twitch_mag_k5_tune/summary
```

`--dry-run` previews the config; `--resume` and `--resume --retry-failed` use the
same generic execution path. Use a fresh root for the expanded grid and changed
sampling/depth rules; do not resume or pool historical radius-2/3 runs with new
ones. Preserve historical roots unchanged. Select learning rate and batch size by validation
within each dataset/method/radius/probability/cap group, not across ablation points.

The generated `configs/sparse_ablation_repeats.json` freezes the completed
epsilon-8 study's 60 validation-best batch-size/LR choices using
`make_repeat_config.py configs/sparse_ablation_repeat_selection.json`.
It runs seeds 1–5 (300 jobs), with weight decay 0.0, under
`results/sparse_ablation_yelp_twitch_mag_k5_repeats`.
Launch it with `python scripts/run_experiments.py configs/sparse_ablation_repeats.json --gpus 0,1,2,3,4,5,6,7`;
add `--resume` for an existing root. Do not regenerate over the frozen config.
After completion, use `summarize_results.py --seed` on the repeat root only,
without another best-configuration selection. Full commands are in
[`reproduce.md`](../reproduce.md#ablation-studies).

#### Five-seed ablation figures

Plot completed final repeats without retraining or selecting configurations again:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_yelp_twitch_mag_k5_repeats --epsilon 8
```

For the epsilon-1/2/5 repeat study, use the same command with
`--ofat-root results/sparse_ablation_eps125_repeats` and the desired
`--epsilon 1`, `--epsilon 2`, or `--epsilon 5`. Each selected epsilon must have
all 300 runs complete: 10 ablation points × 3 datasets × 2 models × 5 seeds.
Other epsilon cohorts are filtered out, never pooled.

The three horizontal panels are line graphs of **radius**, **edge-retention
probability**, and **outgoing-degree cap**. Probability and cap settings are
evenly spaced categorical positions, labeled with their actual values;
radius uses its numeric values.
Colors identify Yelp (micro-F1), Twitch (AUROC), and MAG (accuracy).
**SAGE is solid; GIN is dotted** in every panel. The shared y-axis displays each
dataset's test metric on its original 0–1 scale; different metrics are never
averaged across datasets.

Styling follows `numerics/compare.py` (serif fonts, 22-point axis labels,
17-point ticks, light grids) and the historical compact ablation layout:
18.7 × 3.6 inches, a left-side legend, bold panel labels inside the axes,
and 3.5-point lines with 7-point white-filled circular markers. The shared y-axis
spans 0–0.75. There is no figure title, panel subtitle, or footer.
Metric definitions and uncertainty semantics remain in the data exports and
documentation rather than expanding the figure.

Each point is the mean over training seeds **1–5**, with error bars of
**±1 standard error = sample SD (ddof=1) / sqrt(5)**. These are neither
95% confidence intervals nor the per-run test-node bootstrap intervals.
Seed-0 tuning runs are rejected. Batch size/LR may vary between ablation
points, but must remain fixed across the five seeds at each point.
No test-based selection or selection across repeat seeds occurs.

The shared anchor is `r=1`, `p2=0.1`, `K_out=5`. Radius 1/2 uses two layers;
radius 3 uses three. Incoming sampling caps are 20, 10, and 5 on successive
hops. The radius panel therefore changes both expansion and architecture depth.
`p2=1` removes Bernoulli thinning, not preprocessing or incoming sampling caps.

Outputs default to `OFAT_ROOT/figures_eps<EPSILON>/`:

- `ablation_eps<EPSILON>.png` and `.pdf`: one combined three-panel figure.
- `per_run.csv`: 300 validated individual results and their source paths.
- `points.csv`: 60 distinct means, sample SDs, standard errors, seed cohorts,
  and frozen batch-size/LR choices.
- `curves.csv`: 72 plotted points; the shared anchor appears in all three panels.
- `analysis.json`: arguments, uncertainty policy, split evidence, and versions.

The renderer reads `results.csv` and verifies the selected per-run
`config.json`, `result.json`, and `result.csv`. It rejects missing or duplicate
point/seed pairs, mixed repeat hyperparameters, incompatible splits/metrics,
and obsolete layer/sampling settings. It never overwrites a figure directory;
pass a fresh `--out-dir` to render again. Moved roots use their recorded
`state.json` root to resolve output paths.

Historical seed-0 figure directories remain untouched. Their fixed-setting
renderer and optional DP-GNN/ProGAP depth input are no longer supported by
this script.

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
`--best-validation` instead ranks by validation score, averaged over the exact
same unique-seed cohort as the reported test result under `--seed`. It retains
test statistics and emits `validation_value` and `selection=best_validation`.
Missing/nonfinite validation is an error; ties use the lowest `run_index` when
available, otherwise deterministic historical ordering. The two selection
flags are mutually exclusive.
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

