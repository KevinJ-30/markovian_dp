# Supporting scripts

Use `python scripts/run_experiments.py CONFIG.json --gpus auto` for new
experiments. One JSON-driven runner schedules all supported methods; dataset
presets and training implementations remain in its single-run worker.

## Reusable utilities

| Script | Purpose |
|---|---|
| `run_experiments.py` | Expand JSON configurations, pack owned GPU jobs, retain logs/results, and resume. |
| `run_experiment.py` | Execute one experiment with method-specific calibration and normalized CSV/JSON results. |
| `runner_runtime.py` | Shared GPU authorization, process supervision, and ownership-safe cleanup. |
| `summarize_sweep.py` | Select a validation-best step and report seed-averaged test results; one configuration per child directory. |
| `summarize_matched_eps.py` | Summarize matched-budget studies using their expected directory naming conventions. |
| `summarize_results.py` | Combine arbitrary result CSVs into CSV/Markdown tables using stored bootstrap CIs or seed mean ± sample SD; optional best-test selection and named regimes. |
| `sparse_ablation.py` | Render current ablation figures from ordinary result indexes or supported historical summaries. |
| `plot_frontier.py` | Plot privacy–utility curves from an explicit CSV glob. |

The Python utilities expose `--help`. For manual GraphSAINT download and
extraction instructions, see the [root README](../README.md#datasets).

`run_experiment.py` defaults to ProGAP propagation depth **3**: four native
training stages, each using `--epochs` (80 stage-epochs at `--epochs 20`).
Explicit positive depths override it. Privacy calibration accounts for the
requested depth; equal per-stage epochs do not make different depths compute-matched.

### Unified experiment runner

```bash
python scripts/run_experiments.py configs/main.json --gpus 4,5 --dry-run
python scripts/run_experiments.py configs/main.json --gpus 4,5 \
  --progap-python /path/to/progap/bin/python
python scripts/run_experiments.py configs/main.json --gpus 4,5 --resume
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

There is **no default jobs-per-GPU cap or wall timeout**. Unknown memory shapes
run alone for their full first execution; successful whole-run peaks allow
later matching jobs to overlap. Admission prefers GPUs with fewer active jobs,
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
margins. Unmeasured GPU estimates, or estimates above the measured peak, retain
the margins even for solo admission. Estimates and measured peaks are not OOM
guarantees. Shared CUDA OOM triggers one fresh exclusive retry without changing
scientific settings.

Use `--max-jobs-per-gpu N` for an explicit upper bound and
`--timeout-seconds SECONDS` for a wall deadline. `--device cpu` runs serially
and rejects GPU options; CUDA never silently falls back to CPU. `--dry-run`
creates no files, probes no GPUs, and does not import training dependencies.
`--progap-python` selects the separate native environment if needed; paths in
the config are config-relative, while `split_root` is repository-relative.

Each root contains `experiment.json`, atomic `state.json`, aggregate JSON-lines
`runner.log`, and `results.csv` with one row per planned job. Each attempt is
retained under `runs/<run-id>/attempts/<number>/`: `process.log`, operational
launch/exit/ownership records, and `output/` with config, results, and native
artifacts. Summarize the root CSV rather than globbing all attempts, so retries
do not become extra seeds. No sealed manifests, source snapshots, or hashes are
required. See the [script retirement inventory](RETIRED_SCRIPTS.md) for replacements and removal prerequisites.

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

All **60 one-factor configurations** in `configs/sparse_ablation.json` share one
output root. The study uses `ogbn-arxiv`, `saint-yelp`, and `twitch-allbut2`,
both SAGE and GIN backends, epsilon 8, and training seed 0.
The anchor is radius 1, edge-retention probability 0.5, and outgoing-degree
cap 10. Vary radius `{1,2,3}`, probability `{0.05,0.1,0.25,0.5,1}`, or outgoing
cap `{5,10,20,40}` while holding the other two at the anchor. The shared anchor
runs once; there are no probability-by-cap interactions.

Batch 256, LR 0.01, 20 epochs, hidden 128, two GNN layers, dropout 0.5, and
seed-0 splits stay fixed. Expansion depth does not change architecture depth.
The outgoing preprocessing cap is distinct from the fixed 20-edge incoming
sampling cap; `p2=1` is not an uncapped full-graph baseline. Noise is recalibrated
per configuration at the dataset-specific delta using the repository's mixture
formula with non-root shells `2*K_out**ell`. This setting alone does not establish
a privacy guarantee.

Run both studies through the same scheduler, then render without training:

```bash
python scripts/run_experiments.py configs/sparse_ablation.json --gpus auto
python scripts/run_experiments.py configs/depth_ablation.json --gpus auto \
  --progap-python /path/to/progap/bin/python
python scripts/sparse_ablation.py --ofat-root results/sparse_ablation \
  --depth-root results/depth_ablation --out-dir results/depth_ablation/figures
```

`--dry-run` previews either config; `--resume` and `--resume --retry-failed`
use the same generic execution path. Preserve historical roots unchanged.

This runs DP-GNN-SAGE and DP-GNN-GIN at radius `{1,2,3}` and ProGAP at depth
`{1,2,3}` on the same three datasets. Epsilon 8, seed 0, batch 256, LR 0.01,
hidden 128, dropout 0.5, and 20 epochs remain fixed. ProGAP can use the separate
interpreter selected by `--progap-python`; its 20 epochs apply **per stage**, and
depth `d` trains `d+1` stages. DP-GNN uses 20 training-population epochs at each
radius. Both baselines retain degree bound 5 and recalibrate noise for the
complete requested schedule. Explicit depth values 1/2/3 are unchanged by the
new default.

Re-render a completed comparison without training:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925 \
  --depth-root results/sparse_ablation_depth_supervised_20260925 \
  --out-dir results/sparse_ablation_depth_supervised_20260925/figures_rebuilt
```

`--depth-root` requires all 27 baseline runs and matching dataset/task/split
identities across both studies. Its default output is `DEPTH_ROOT/figures`.

The renderer consumes that **same run folder**, without retraining:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925 \
  --out-dir results/sparse_ablation_ofat_supervised_20260925/figures_compact
```

The renderer reads the root `results.csv`, or historical `summary.csv` with
`result_csv` pointers, without manifests or hashes. It writes `ablation_sage`
and `ablation_gin` PNG/PDF pairs, `per_run.csv`, `curves.csv`, and ordinary
`analysis.json` metadata. The training root and its per-attempt outputs remain
unchanged.

Each backend gets three horizontal panels: **(a) depth lines**, **(b) probability
bars**, and **(c) outgoing-cap bars**, with labels inside the upper-left corners.
Dataset colors are shared across all panels. In panel (a), SGNN is solid,
ProGAP dashed, and DP-GNN dotted; the latter matches the SAGE/GIN backend, while
the same ProGAP runs appear in both figures. Without `--depth-root`, only SGNN
depth curves appear. Depth lines are fully opaque and 3 points wide, with no
uncertainty whiskers. Panels (b)/(c) remain SGNN-only grouped bars with intervals.
The shared legend sits to the left and contains only dataset colors and method
line styles; backend and privacy headings are omitted. Backend identity remains
in the filenames and privacy settings in the CSV exports. The compact canvas
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

The renderer checks required curve membership, duplicate settings, split/metric
consistency, and stored intervals required for bar panels. It never overwrites
a figure directory; pass a fresh `--out-dir` for another reconstruction.

### Main experiment matrix

`configs/main.json` contains **1,584 runs** across the eight dataset protocols:
`ogbn-arxiv`, `ogbn-products`, `saint-reddit`, `saint-yelp`, `saint-amazon`,
`twitch-allbut2`, `facebook100-allbut2`, and `mag-allbut2`. The domain protocols
hold out `engb/es`, `cornell5/penn94`, and `cn/de` respectively; split seed 0 is
independent of training seeds 0/1/2.

Methods are non-private MLP/GraphSAGE/GIN and private DP-MLP, ProGAP, DPAR,
DP-GNN-SAGE/GIN, and SparseGNN-SAGE/GIN. LR is .01/.001, requested batch 1024,
epochs 20, private epsilon 2/8, SparseGNN-only p2 .1/.25/.5/.75/1, dropout .5,
and hidden width 64 for MLPs or 128 for graph methods. Degree controls are 10;
bootstrap is disabled. These controls have method-specific meanings: fanout,
PPR top-k, preprocessing cap, and graph-degree bounds are not interchangeable.

Private noise is calibrated per configuration at delta=1/N_train. SparseGNN
retains its mixture formula, which alone does not establish a privacy guarantee.
ProGAP is explicitly depth 3, with four native drop-last stages at 20 epochs
each; DPAR retains ppr_num=70 and sampled_train_rate=.09, so its effective batch
is at most 70 released roots. Actual schedules and privacy values are recorded.
Historical ProGAP depths are not rewritten or silently reproduced by this config.

```bash
python scripts/run_experiments.py configs/main.json --gpus auto \
  --progap-python /path/to/progap/bin/python
python scripts/summarize_results.py results/main/results.csv \
  --seed --best-validation --out results/main/summary
```

Edit JSON blocks/axes for future experiments rather than adding a launcher.
Omit the selection flag to report every configuration; `--best` still supports
explicitly labeled test-based selection, with its test-selection bias warning.

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

