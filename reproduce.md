# Reproduction of Results in the Main Paper

To reproduce the results in the main paper, you can use the following commands. 

## Main Experiments

The first step is to run all of the primary experiments, which is summarized in the following bash script and ablates over all relevant parameters.

```bash

```

The second step is to reproduce a markdown table that contains the main results of all the experiments.

```bash

```

This should render a Markdown table in ```enter me later```, which can be used to directly create the primary table in the experiments section of the main paper.

## Ablation Studies

The 60-run one-factor study varies expansion depth `r={1,2,3}`, edge-retention
probability `p2={0.05,0.1,0.25,0.5,1}`, and outgoing-degree cap
`K_out={5,10,20,40}`. Other factors stay at `r=1,p2=0.5,K_out=10`, with a shared
anchor run. It uses three datasets, SAGE/GIN backends, epsilon 8, and training
seed 0; there are no interaction experiments or independent-seed replicates.

Run all configurations and render them in one self-contained output folder:

```bash
PYTHON=/path/to/environment/bin/python \
  OUT_ROOT=results/sparse_ablation_ofat \
  bash scripts/sparse_ablation_paper.sh
```

For sequential execution, use `scripts/sparse_ablation_ofat.sh` instead.
Both accept `--dry-run`. All accepted configurations are available in
`OUT_ROOT/runs/`; figures and plotting CSVs go into `OUT_ROOT/figures/`.
The queued pathway also writes `OUT_ROOT/summary.csv` and `summary.md`.

Render the already-completed study without rerunning experiments:

```bash
python scripts/sparse_ablation.py \
  --ofat-root results/sparse_ablation_ofat_supervised_20260925
```

This produces one chart per backend (`ablation_sage`, `ablation_gin`, PNG/PDF),
with three horizontal parameter panels and three dataset-colored bars at each
parameter label. A shared legend identifies datasets; there is no title.
The y-axis is "Test metric": accuracy for ogbn-arxiv,
micro-F1 for Yelp, and AUROC for Twitch, with the stored 95% node-bootstrap
intervals. Resource measurements stay in `per_run.csv` and `curves.csv`.
Use `--out-dir OUT_ROOT/figures_rebuilt` for a fresh reconstruction directory.
See [scripts/README.md](scripts/README.md#sparseexpand-paper-ablations) for
the fixed settings, evidence layout, and interpretation.

## Numerical Experiments

To reproduce the numerical experiments used in the main paper, you can use the following command.

```bash
python numerics/compare.py
```

This will render the combined `numerics/figures/comparison.png`,
`comparison.pdf`, and `comparison.svg`, alongside numerical CSVs and metadata.


### Appendix Figures

Run from `markovian_dp/` with Python and `numpy`, `scipy`, `matplotlib`, and
`dp-accounting` installed:

```bash
python numerics/run_all.py
```

This computes and renders all four assembled figures below. Each of the rows are a different subgraph depth, such that `r = 1, 2, 3`. 

| Figure | Individual command | Columns |
| --- | --- | --- |
| Expanded comparison (3 × 4) | `python numerics/compare_expanded.py` | ε(T) through T = 1000; δ(ε) at T = 1, 100, 1000 |
| Noise dependence (3 × 3) | `python numerics/noise.py` | ε versus σ at T = 1, 100, 1000 |
| Root-sampling dependence (3 × 3) | `python numerics/root_sampling.py` | ε versus p₁ at T = 1, 100, 1000 |
| Degree dependence (3 × 3) | `python numerics/degree.py` | ε versus K at T = 1, 100, 1000 |

Defaults are p₁ = 0.01, p₂ ∈ {0.1, 0.25, 0.5, 0.75}, K = 5, σ = 2,
N = 100,000, and δ = 10⁻⁵, except for the parameter being swept. Noise uses
10 equally spaced values from 1 to 10. Root sampling uses 10 logarithmically
spaced batch sizes from 100 to 10,000, rounded to integers, with p₁ = batch/N.
Degree uses integers 1 through 8. Composition horizons stay fixed during sweeps.
The PLD discretization interval is 10⁻³, and the expanded privacy profiles use
ε ∈ [0.1, 10].

Each script writes one PNG/PDF/SVG figure, numerical CSVs, and `parameters.json`
(including actual grids, source hashes, and package versions) under
`numerics/figures/{expanded,noise,root_sampling,degree}/`. 