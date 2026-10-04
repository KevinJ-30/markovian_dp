# Numerical figures

Run these CPU experiments from the repository directory after following the [installation instructions](../README.md#installation).

## Main comparison

Generate the main privacy-accounting figure with:

```bash
python numerics/compare.py
```

The three panels show $\epsilon$ versus composition steps $T$ at $r=1$, $\delta(\epsilon)$ at $r=1$, and $\epsilon$ versus radius $r$. The outer panels use logarithmic $\epsilon$; the middle uses logarithmic $\delta$ and includes lower-pair curves. RDP is shown in the first two panels only; its radius values remain in `radius.csv`.

Outputs are saved as `numerics/figures/comparison.{png,pdf,svg}`, alongside `composition.csv`, `radius.csv`, `curves.csv`, `pair_weights.csv`, and `parameters.json`. Zero-composition values remain in the CSV but are omitted from the logarithmic plot.

To compare $\sigma=5$ with the default $\sigma=2$ without replacing the main outputs:

```bash
python numerics/compare.py --sigma 5 --out-dir numerics/figures/sigma5
```

## Appendix figures

Generate all four appendix figures with:

```bash
python numerics/run_all.py
```

These cover expanded composition/privacy profiles and sweeps over noise, root sampling, and degree. Every appendix panel includes matching lower-pair curves, and every epsilon y-axis uses logarithmic scaling. The root-sampling sweep also uses a logarithmic $p_1$ x-axis. Figures in PNG, PDF, and SVG format, numerical CSVs, and parameters are saved under `numerics/figures/{expanded,noise,root_sampling,degree}/`. See the [reproduction guide](../reproduce.md#numerical-experiments) for individual commands.

Full sweeps can take several hours. Building and composing discretized privacy-loss distributions is expensive, especially at larger radii, degrees, and root-sampling probabilities.

## Parameters

All figures compare $p_2\in\{0.25,0.5,0.75\}$ against group privacy and Daigavane et al. (RDP). RDP conversion uses orders $1.1,1.2,\ldots,9.9$. The main comparison defaults to $T=1000$, $\delta=10^{-5}$, and radii $\{1,2,3\}$; use `--steps`, `--delta`, `--radii`, and `--p2` to override them. Use `--out-dir` to write elsewhere or `--help` to see each script's options. Running a command again replaces its existing outputs.
