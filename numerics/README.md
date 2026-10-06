# Numerical figures

Run these CPU experiments from the repository directory after following the [installation instructions](../README.md#installation).

## Main comparison

Generate the main privacy-accounting figure with:

```bash
python numerics/compare.py
```

The three panels show $\epsilon$ versus composition steps $T$ at $r=1$, $\delta(\epsilon)$ at $r=1$, and $\epsilon$ versus radius $r$. The outer panels use logarithmic $\epsilon$; the middle uses logarithmic $\delta$. All three panels include matching lower-pair curves, drawn as dashed lines in the corresponding upper-bound colors. RDP is shown in the first two panels only; its radius values remain in `radius.csv`.

Outputs are saved as `numerics/figures/main_comparison/comparison.{png,pdf,svg}`, alongside `composition.csv`, `radius.csv`, `curves.csv`, `pair_weights.csv`, and `parameters.json`. Zero-composition values remain in the CSV but are omitted from the logarithmic plot.

## Appendix figures

Generate all four appendix figures with:

```bash
python numerics/run_all.py
```

These cover expanded composition/privacy profiles and sweeps over noise, root sampling, and degree. Every appendix panel includes matching lower-pair curves, and every epsilon y-axis uses logarithmic scaling. The root-sampling sweep also uses a logarithmic $p_1$ x-axis. Figures in PNG, PDF, and SVG format, numerical CSVs, and parameters are saved under `numerics/figures/{expanded,noise,root_sampling,degree}/`. See the [reproduction guide](../reproduce.md#numerical-experiments) for individual commands.

Panel titles use lowercase $r$ for expansion depth. For label-only changes, re-render from the saved CSVs and retain the original `parameters.json`; the generation commands above recompute the numerical data.

Full sweeps can take several hours. Building and composing discretized privacy-loss distributions is expensive, especially at larger radii, degrees, and root-sampling probabilities.

## Parameters

All figures compare $p_2\in\{0.25,0.5,0.75\}$ against group privacy and Daigavane et al. (RDP). RDP conversion uses orders $1.1,1.2,\ldots,9.9$. The main comparison defaults to $\sigma=5$, $T=1000$, $\delta=10^{-5}$, and radii $\{1,2,3\}$; use `--sigma`, `--steps`, `--delta`, `--radii`, and `--p2` to override them. Appendix defaults retain $\sigma=2$ except when sweeping noise. Use `--out-dir` to write elsewhere or `--help` to see each script's options. Running a command again replaces its existing outputs.
