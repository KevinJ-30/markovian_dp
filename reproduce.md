# Reproducing the experiments

Run these commands from the repository directory after following the [installation and dataset instructions](README.md). The examples use GPU 0; replace `--gpus 0` with the devices available to you. The numerical experiments run on the CPU.

## Main experiments

The unified configurations evaluate SparseGNN and the baselines on Arxiv, Products, Reddit, Yelp, Amazon, Twitch, Facebook, and MAG at private epsilon targets 1, 2, 5, and 8. MLP and DP-MLP use weight decay 0.0005; every other method uses zero. SparseGNN uses expansion radius 1, and ProGAP depths 1 and 5 remain separate.

Twitch predicts the binary mature-content label and reports AUROC. Facebook predicts the six year cohorts from 2004 through 2009 using the `fb100-year-6` school split. Yelp and Amazon report micro-F1, and the other datasets report accuracy. The runners select checkpoints using validation performance.

The frozen repeat configuration contains the validation-selected parameters used in the primary tables: 280 configurations, each run with seeds 1 through 5. No new tuning is needed to run it:

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0 --dry-run
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0
python scripts/summarize_results.py results/main_r1_eps1258_repeats/results.csv \
  --seed --out results/main_r1_eps1258_repeats/summary
```

This writes `summary.csv` and `summary.md` with mean test scores and sample standard deviations across seeds 1 through 5. The tuning seed is excluded, and test scores never select configurations. Add `--resume` to continue an interrupted run, or `--resume --retry-failed` after addressing failed-job errors.

The frozen configuration combines non-MLP private winners from `main_r1_eps15_zero_decay_tune` and `main_r1_zero_decay_tune`, MLP/DP-MLP winners from `mlp_wd5e4_eps1258_tune`, and nonprivate GraphSAGE/GIN winners from the latest epsilon-1/5 tables. Older nonprivate GIN batch-size choices for Yelp, Amazon, and MAG remain in the archived configs rather than being duplicated in the unified table.

To perform fresh seed-0 tuning instead, then generate a separate repeat config:

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_tune.json --gpus 0
python scripts/make_repeat_config.py configs/main_r1_eps1258_repeat_selection.json
python scripts/run_experiments.py configs/main_r1_eps1258_retuned_repeats.json --gpus 0
python scripts/summarize_results.py results/main_r1_eps1258_retuned_repeats/results.csv \
  --seed --out results/main_r1_eps1258_retuned_repeats/summary
```

Fresh tuning runs 1,632 jobs, selecting learning rate, batch size, and SparseGNN edge-retention probability by validation. It generates 1,400 repeat jobs without overwriting the frozen primary-table config. Superseded epsilon-pair and MLP-only configs are retained in the Git-ignored `old_configs/` directory; their historical results remain unchanged.

## Ablation studies

`configs/sparse_ablation.json` combines expansion-radius, edge-retention-probability, and outgoing-degree-cap ablations for SparseSAGE and SparseGIN on Yelp (`saint-yelp`), Twitch (`twitch-allbut2`), and MAG (`mag-allbut2`). Each one-factor setting is crossed with batch sizes `{256,1024}` and learning rates `{0.01,0.001}`, matching the main tuning grids. Epsilon 8 and seed 0 remain fixed. DP-GNN and ProGAP are excluded; the old depth config is archived in `old_configs/`.

The primary configuration keeps epsilon fixed at **8** (240 tuning runs). `configs/sparse_ablation_eps125.json` repeats the identical grid at epsilon **1, 2, and 5** (720 tuning runs), for **960 tuning runs** across both configurations. Its default output root is `results/sparse_ablation_eps125`. Both configurations use seed 0; each has separate repeat-selection settings.

SparseGNN uses two message-passing layers at radius 1/2 and three at radius 3. After edge thinning, incoming sampling retains at most 20, 10, and 5 neighbors per expanded node on successive hops (5 thereafter). Thus radius 3 has at most 1,221 nodes per rooted subgraph, and this comparison changes both expansion and architecture depth. Use a fresh output root; historical fixed-cap/fixed-depth radius-2/3 runs are not interchangeable with the new results. Radius-1 settings remain unchanged.

The initial campaign in `results/sparse_ablation_products_reddit_mag_tune` used Products, Reddit, and MAG with an outgoing-cap anchor of 10 and was stopped. Its radius/probability sweeps are not results for the corrected cap-5 anchor. Preserve those artifacts and use a fresh root for the revised study. `configs/sparse_ablation_repeat_selection.json` is prepared for `results/sparse_ablation_yelp_twitch_mag_k5_tune`; it selects only learning rate and batch size before generating seeds 1–5, and requires all revised tuning results to be complete.

```bash
python scripts/run_experiments.py configs/sparse_ablation.json --gpus auto \
  --out-dir results/sparse_ablation_yelp_twitch_mag_k5_tune
python scripts/run_experiments.py configs/sparse_ablation_eps125.json --gpus auto
```

The shared anchor is radius 1, probability 0.1, and cap 5. The outgoing-degree cap matches the main experiments. The probability sweep covers `{0.05,0.1,0.25,0.5,1}` and the degree-cap sweep covers `{5,10,20,40}`. Counting the anchor only once gives 10 settings × 3 datasets × 2 methods × 2 batch sizes × 2 learning rates = 240 runs. After completion, write all candidate results:

```bash
python scripts/summarize_results.py results/sparse_ablation_yelp_twitch_mag_k5_tune/results.csv \
  --bootstrap --out results/sparse_ablation_yelp_twitch_mag_k5_tune/summary
python scripts/summarize_results.py results/sparse_ablation_eps125/results.csv \
  --bootstrap --out results/sparse_ablation_eps125/summary
```

This summary retains all learning-rate/batch-size candidates. Compare validation scores within each dataset, method, radius, probability, and cap; do not select on test scores or collapse different ablation points into one winner.

The tuning runs use seed 0; stored confidence intervals describe test-node bootstrap uncertainty rather than training-seed variation. Plot only the final five-seed repeat root with `sparse_ablation.py`, not this tuning sweep. Historical outputs remain unchanged.

The completed epsilon-8 tuning study generated `configs/sparse_ablation_repeats.json` using `make_repeat_config.py` and `configs/sparse_ablation_repeat_selection.json`. This freezes 60 validation-best batch-size/LR choices, one per dataset/method/ablation point, and expands seeds 1–5 into **300 final runs** (150 SparseSAGE and 150 SparseGIN). Epsilon remains 8 and weight decay remains 0.0. Results use `results/sparse_ablation_yelp_twitch_mag_k5_repeats`, separate from seed-0 tuning.

For a fresh reproduction, generate the config only if it does not already exist; the generator refuses to overwrite it:

```bash
python scripts/make_repeat_config.py configs/sparse_ablation_repeat_selection.json
```

Run the generated config on all eight GPUs; add `--resume` when continuing an existing output root:

```bash
python scripts/run_experiments.py configs/sparse_ablation_repeats.json --gpus 0,1,2,3,4,5,6,7
```

After all final runs complete, summarize training-seed variation without selecting again or pooling seed-0 tuning:

```bash
python scripts/summarize_results.py results/sparse_ablation_yelp_twitch_mag_k5_repeats/results.csv \
  --seed --out results/sparse_ablation_yelp_twitch_mag_k5_repeats/summary
```

Render the completed epsilon-8 repeats as radius, probability, and degree-cap line panels:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_yelp_twitch_mag_k5_repeats --epsilon 8
```

Outputs are `figures_eps8/ablation_eps8.png` and `.pdf` under the repeat root, plus per-run and aggregate CSVs. Dataset colors identify Yelp, Twitch, and MAG; SAGE lines are solid and GIN lines dotted. Error bars show **±1 standard error across seeds 1–5**, computed as sample SD divided by sqrt(5), not node-bootstrap uncertainty or a 95% confidence interval. The renderer requires all 300 runs and fixed hyperparameters within each five-seed cohort. It performs no further selection and refuses to overwrite an existing figure directory; use a fresh `--out-dir` to render again.

For epsilon 1/2/5, `configs/sparse_ablation_eps125_repeat_selection.json` selects batch size/LR separately within each epsilon/dataset/method/ablation point. It generates `configs/sparse_ablation_eps125_repeats.json`: 180 configurations × seeds 1–5 = **900 final runs**, with weight decay 0.0, under `results/sparse_ablation_eps125_repeats`.

The following chain starts repeats automatically only after all 720 tuning runs succeed and the standard generator validates their results. Any failed stage prevents subsequent stages from starting. Run once with fresh destinations; the generator refuses to overwrite a generated config.

```bash
python scripts/run_experiments.py configs/sparse_ablation_eps125.json --gpus 0,1,2,3,4,5,6,7 &&
python scripts/make_repeat_config.py configs/sparse_ablation_eps125_repeat_selection.json &&
python scripts/run_experiments.py configs/sparse_ablation_eps125_repeats.json --gpus 0,1,2,3,4,5,6,7
```

After completion, summarize the 900 repeat runs separately from seed-0 tuning:

```bash
python scripts/summarize_results.py results/sparse_ablation_eps125_repeats/results.csv \
  --seed --out results/sparse_ablation_eps125_repeats/summary
```

Plot each completed epsilon cohort separately with the same renderer, for example:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_eps125_repeats --epsilon 2
```

Use `--epsilon 1` or `--epsilon 5` for the other budgets; each has its own `figures_eps<EPSILON>` output directory.

### Remaining datasets: seed-0 ablations

Two separate configs extend the same ablation grid to the remaining datasets at epsilon `{1,2,5,8}`, with both SparseSAGE and SparseGIN and all four batch-size/LR candidates:

- `configs/sparse_ablation_remaining_seed0.json`: Arxiv, Products, Reddit, and Facebook; **1,280 runs**.
- `configs/sparse_ablation_amazon_seed0.json`: Amazon only; **320 runs**.

Both use training seed 0 only, 20 epochs, hidden width 128, dropout 0.5, and weight decay 0.0. The 1,000 test-node bootstrap resamples are retained for consistency with the completed studies; they are not training-seed repeats or across-seed standard errors.

```bash
python scripts/run_experiments.py configs/sparse_ablation_remaining_seed0.json --gpus auto
python scripts/run_experiments.py configs/sparse_ablation_amazon_seed0.json --gpus auto
```

The default output roots are `results/sparse_ablation_remaining_seed0` and `results/sparse_ablation_amazon_seed0`. Add `--dry-run` to preview without launching training.

After the non-Amazon tuning completes, `configs/sparse_ablation_remaining_repeat_selection.json` selects batch size/LR by validation independently for each dataset/model/epsilon/ablation point. It generates 320 selected configurations repeated on seeds 1–5, totaling **1,600 runs**:

```bash
python scripts/make_repeat_config.py configs/sparse_ablation_remaining_repeat_selection.json
python scripts/run_experiments.py configs/sparse_ablation_remaining_repeats.json --gpus auto
```

The selector refuses to overwrite an existing generated config. Repeat results go to `results/sparse_ablation_remaining_repeats`; Amazon is excluded.

All roots share the runner's result schema. A combined seed-0 figure should select batch size/LR by validation separately within each dataset/method/epsilon/ablation point, then plot the corresponding test score without across-seed error bars. Use the existing Yelp/Twitch/MAG tuning roots for the matching seed-0 cohort, not their five-seed repeat means. The current `sparse_ablation.py` renderer still requires five seeds on Yelp/Twitch/MAG; plotting these additional roots requires extending its input, dataset, and aggregation handling.

## Numerical experiments

Generate the main privacy-accounting comparison with:

```bash
python numerics/compare.py
```

The figure is saved as `comparison.png`, `comparison.pdf`, and `comparison.svg` under `numerics/figures/main_comparison/`, alongside the numerical data. Its three horizontal panels show $\epsilon(T)$ at $r=1$, $\delta(\epsilon)$ at $r=1$, and $\epsilon(r)$ for $r\in\{1,2,3\}$. The outer panels use logarithmic $\epsilon$; the middle uses logarithmic $\delta$. All three panels include matching lower-pair curves. Defaults compare $p_2\in\{0.25,0.5,0.75\}$ against group privacy with $\sigma=5$, $T=1000$, and fixed $\delta=10^{-5}$ where applicable, with RDP included in the first two panels only. See `numerics/README.md` for parameter overrides and CSV outputs.

Generate all appendix figures with:

```bash
python numerics/run_all.py
```

You can also generate them individually:

```bash
python numerics/compare_expanded.py
python numerics/noise.py
python numerics/root_sampling.py
python numerics/degree.py
```

These commands write figures and data under `numerics/figures/expanded/`, `noise/`, `root_sampling/`, and `degree/`, respectively. The scripts expose their numerical parameters through `--help`; the full sweeps can take several hours.
