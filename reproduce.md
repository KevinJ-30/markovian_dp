# Reproducing the experiments

Run these commands from the repository directory after following the [installation and dataset instructions](README.md). The examples use GPU 0; replace `--gpus 0` with the devices available to you. The numerical experiments run on the CPU.

## Main experiments

The main configuration evaluates SparseGNN and the baselines on Arxiv, Products, Reddit, Yelp, Amazon, Twitch, Facebook, and MAG. It tunes learning rate and batch size, along with the edge-retention probability for SparseGNN, using training seed 0. SparseGNN uses expansion radius 1, and ProGAP is evaluated at depths 1, 3, and 5.

Twitch predicts the binary mature-content label and reports AUROC. Facebook predicts the six year cohorts from 2004 through 2009 using the `fb100-year-6` school split. Yelp and Amazon report micro-F1, and the other datasets report accuracy. The runners select checkpoints using validation performance.

First, preview the grid and run the tuning experiments:

```bash
python scripts/run_experiments.py configs/main_r1_tune.json --gpus 0 --dry-run
python scripts/run_experiments.py configs/main_r1_tune.json --gpus 0
```

Results are saved in `results/main_r1_tune/`. Add `--resume` to continue an interrupted run, or `--resume --retry-failed` to retry failed jobs after addressing their errors.

Once tuning is complete, select configurations by validation score and generate runs for seeds 1 through 5:

```bash
python scripts/make_repeat_config.py configs/main_r1_repeat_selection.json
python scripts/run_experiments.py configs/main_r1_repeats.json --gpus 0
```

The selection settings choose over learning rate, batch size, and SparseGNN edge-retention probability. They keep the three ProGAP depths separate. The generated configuration is saved as `configs/main_r1_repeats.json`, and the repeated runs write to `results/main_r1_repeats/`.

Summarize the repeated runs with:

```bash
python scripts/summarize_results.py results/main_r1_repeats/results.csv \
  --seed --out results/main_r1_repeats/summary
```

This writes `summary.csv` and `summary.md` with mean test scores and sample standard deviations across seeds 1 through 5. The tuning seed is not included in these statistics, and test scores are not used to choose configurations.

## Ablation studies

The SparseGNN ablation configuration varies expansion radius, edge-retention probability, and outgoing-degree cap on Arxiv, Yelp, and Twitch. The depth configuration adds DP-GNN and ProGAP comparisons. Run both studies with:

```bash
python scripts/run_experiments.py configs/sparse_ablation.json --gpus 0
python scripts/run_experiments.py configs/depth_ablation.json --gpus 0
```

After they finish, write the result tables:

```bash
python scripts/summarize_results.py results/sparse_ablation/results.csv \
  --bootstrap --out results/sparse_ablation/summary
python scripts/summarize_results.py results/depth_ablation/results.csv \
  --bootstrap --out results/depth_ablation/summary
```

Render the depth, probability, and degree-cap panels with:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation \
  --depth-root results/depth_ablation \
  --out-dir results/ablation_figures
```

The output directory contains `ablation_sage.png`, `ablation_gin.png`, their PDF versions, and the plotted data. Choose a new output directory when rendering again. These ablations use seed 0; the stored confidence intervals describe test-node bootstrap uncertainty rather than variation across training seeds.

## Numerical experiments

Generate the main privacy-accounting comparison with:

```bash
python numerics/compare.py
```

The figure is saved as `numerics/figures/comparison.png` and `comparison.pdf`, alongside the numerical data.

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
