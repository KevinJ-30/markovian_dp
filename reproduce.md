# Reproducing the experiments

Run these commands from the repository directory after following the [installation and dataset instructions](README.md). The examples use GPU 0; replace `--gpus 0` with the devices available to you. The numerical experiments run on the CPU.

The two frozen final-paper run configurations are:

| Configuration | Scope | Runs |
| --- | --- | ---: |
| `main_r1_eps1258_repeats.json` | Main comparison on seven datasets; private epsilon targets 1, 2, 5, 8 | 1,085 |
| `sparse_ablation_eps1258_repeats.json` | SparseSAGE/SparseGIN ablations on Products, FB-100, Arxiv at epsilon 1, 2, 5, 8 | 1,200 |

Both use validation-selected hyperparameters and training seeds 1–5.

## Runner usage

- Replace `--gpus 0` with your GPU IDs, such as `--gpus 0,1`.
- Use `--dry-run` to preview jobs without training.
- Add `--resume` to continue an interrupted run. After fixing failed jobs, use `--resume --retry-failed`.
- Results are written to `results/<configuration-name>/`; use `--out-dir` for a different destination.

## Main experiments

Run the main comparison, then summarize the completed runs:

```bash
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0 --dry-run
python scripts/run_experiments.py configs/main_r1_eps1258_repeats.json --gpus 0
python scripts/summarize_results.py results/main_r1_eps1258_repeats/results.csv \
  --seed --out results/main_r1_eps1258_repeats/summary
```

This writes `summary.csv` and `summary.md` with mean test scores and sample standard deviations across seeds 1–5. Do not add `--best` or `--best-validation`; the configurations are already selected.

Without `--seed`, the summarizer reports each run's test point estimate without uncertainty. Individual runs do not compute confidence intervals.

### Using your own validation-selected configuration

Tune on seed 0, select by validation score, then run the selected settings on seeds 1–5:

```bash
python scripts/run_experiments.py configs/tuning/main_r1_eps1258_tune.json --gpus 0 &&
python scripts/make_repeat_config.py configs/tuning/main_r1_eps1258_selection.json &&
python scripts/run_experiments.py results/main_r1_eps1258_tune/repeats.json --gpus 0
```

Replace `results/main_r1_eps1258_repeats` with `results/main_r1_eps1258_retuned_repeats` in the summary command above. The generated config is separate from the frozen config; the selector requires complete tuning and refuses to overwrite an existing `repeats.json`.

## Ablation studies

Run the Products, FB-100, and Arxiv ablations, then summarize:

```bash
python scripts/run_experiments.py configs/sparse_ablation_eps1258_repeats.json --gpus 0 --dry-run
python scripts/run_experiments.py configs/sparse_ablation_eps1258_repeats.json --gpus 0
python scripts/summarize_results.py results/sparse_ablation_eps1258_repeats/results.csv \
  --seed --out results/sparse_ablation_eps1258_repeats/summary
```

After the multi-seed ablation runs complete, render each epsilon:

```bash
for epsilon in 1 2 5 8; do
  python scripts/sparse_ablation.py \
    --ofat-root results/sparse_ablation_eps1258_repeats \
    --epsilon "$epsilon"
done
```

Figures are saved under `results/sparse_ablation_eps1258_repeats/figures_eps<EPSILON>/` as `ablation_sd_eps<EPSILON>.png` and `.pdf`, alongside the plotted CSV data. Error bars show ±1 sample standard deviation across five seeds (ddof=1). All runs for the selected epsilon must be complete. To render again, supply a fresh `--out-dir`; existing figure directories are not overwritten.

### Using your own validation-selected configuration

Tune batch size/LR on seed 0 independently for each ablation point, then run seeds 1–5:

```bash
python scripts/run_experiments.py configs/tuning/sparse_ablation_eps1258_tune.json --gpus 0 &&
python scripts/make_repeat_config.py configs/tuning/sparse_ablation_eps1258_selection.json &&
python scripts/run_experiments.py results/sparse_ablation_eps1258_tune/repeats.json --gpus 0
```

Use `results/sparse_ablation_eps1258_retuned_repeats` in the summary and plotting commands above. The selector requires complete tuning and refuses to overwrite an existing `repeats.json`; the frozen config is unchanged.

## Numerical experiments

Generate the main privacy-accounting comparison with:

```bash
python numerics/compare.py
```

Outputs: `numerics/figures/main_comparison/comparison.{png,pdf,svg}` and numerical data.

Generate all appendix figures with:

```bash
python numerics/run_all.py
```

Outputs are under `numerics/figures/{expanded,noise,root_sampling,degree}/`. See [numerics/README.md](numerics/README.md) for individual commands and options.
