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

We also provide a set of scripts to run, create Markdown tables for all ablation studies, and render plots for all ablation studies in relevant folder(s).

First, we consider the ablation study over the out-degree cap. 

```bash

```

Now, we consider the ablation study over the model depth.

```bash

```

Finally, we consider the ablation study over the sparsification parameter $p_2$, which can be run using the following command.

```bash


```


This should render a Markdown table in ```enter me later```, which can be used 

## Numerical Experiments

To reproduce the numerical experiments used in the main paper, you can use the following command.

```bash
python numerics/compare.py
```

This will render the combined `numerics/figures/comparison.png` and
`comparison.pdf`, alongside numerical CSVs and metadata.


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

The default parameters chosen are p₁ = 0.01, p₂ ∈ {0.1, 0.25, 0.5, 0.75}, K = 5, σ = 2,
N = 100,000, and δ = 10⁻⁵, except for the parameter being swept. Noise uses
10 equally spaced values from 1 to 10. Root sampling uses 10 logarithmically
spaced batch sizes from 100 to 10,000, rounded to integers, with p₁ = batch/N.
Degree uses integers 1 through 8. Composition horizons stay fixed during sweeps.
The PLD discretization interval is 10⁻³, and the expanded privacy profiles use
ε ∈ [0.1, 10].

Each script writes one PNG/PDF figure, numerical CSVs, and `parameters.json`
(including actual grids, source hashes, and package versions) under
`numerics/figures/{expanded,noise,root_sampling,degree}/`. 