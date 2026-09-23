r"""Plot four delta(epsilon) comparisons; run with `python numerics/compare.py`.

Normalize the clipping threshold to C=1, and use the same absolute Gaussian
standard deviation sigma for every curve. M = sum(K**ell, ell=0,...,r).

Our pair, imported unchanged from src.privacy.accounting (current chi=1 law):
    J = Bernoulli(p1) + sum_{ell=1}^r Binomial(K**ell, b_ell), independently,
    b_ell = p1*p2**ell / (1-p1+p1*p2**ell),
    P = sum_j Pr(J=j) Normal(-j, sigma**2),
    Q = sum_j Pr(J=j) Normal(+j, sigma**2).
At p2=1, J is Binomial(M,p1): this is the requested group-privacy baseline,
not the generic group-privacy inequality discussed in Daigavane's Remark 3.
We compose the pair's pessimistically discretized PLD, not its mixture weights.

Daigavane et al., Theorem 1: https://arxiv.org/abs/2111.15521
    H ~ Hypergeometric(N, M, m), with m/N = p1,
    R_alpha = log E exp(2*alpha*(alpha-1)*H**2/sigma**2) / (alpha-1).
This is also the Renyi divergence of the count-revealing joint Gaussian pair
    P_D(j,x) = Pr(H=j)*Normal(x; -j, sigma**2),
    Q_D(j,x) = Pr(H=j)*Normal(x; +j, sigma**2).
The plotted Daigavane curve uses t*R_alpha followed by Google's RDP-to-DP
conversion, as in the original repository, NOT a hidden-count mixture PLD.
https://github.com/google-research/google-research/blob/master/
    differentially_private_gnns/privacy_accountants.py
Upstream's sensitivity-relative multiplier is sigma/(2*M), not sigma.
We evaluate more Renyi orders than upstream's 1.1,...,9.9 grid; this tightens
numerical evaluation of the same theorem. A finite order grid remains an upper
bound. Their fixed-size root batches differ from our Bernoulli root sampling;
these are matched-parameter bound comparisons, not identical mechanisms.

The imported chi=1 pair is evaluated as supplied; these plots do not establish
its graph-level privacy assumptions. No training code or accountant is changed.
Outputs: four individual figures and one overview (PNG/PDF), curves.csv,
pair_weights.csv, and parameters.json. curves.csv retains raw_delta as well as
delta clipped to [0,1], the mathematical range, because floating-point PLD
construction/composition can return tiny negative tails or values above one.
Requires numpy, scipy, matplotlib, dp-accounting, and the imported modules.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dp_accounting.rdp.rdp_privacy_accountant import compute_delta
from scipy.special import logsumexp
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.privacy.accounting import mixture_gaussian_pld, sparsegnn_mixture_weights

RADII = (1, 2)
STEPS = (1, 1000)
# Include the upstream grid, plus near-one and larger orders.
ORDERS = np.unique(np.concatenate((
    1 + np.geomspace(1e-3, 1, 100), np.arange(1.1, 10, 0.1),
    np.linspace(2, 64, 497), np.geomspace(64, 1024, 100),
)))


def daigavane_rdp(population, batch_size, max_terms, sigma, orders=ORDERS):
    """Theorem 1, with C=1 and actual noise standard deviation sigma."""
    counts = np.arange(max(0, batch_size - (population - max_terms)),
                       min(max_terms, batch_size) + 1)
    log_probs = hypergeom.logpmf(counts, population, max_terms, batch_size)
    log_probs -= logsumexp(log_probs)
    exponent = (2 * orders[:, None] * (orders[:, None] - 1)
                * counts[None, :] ** 2 / sigma**2)
    return logsumexp(log_probs + exponent, axis=1) / (orders - 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p1", type=float, default=0.01)
    parser.add_argument("--p2", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Gaussian noise standard deviation divided by C")
    parser.add_argument("--population", type=int, default=10000,
                        help="Daigavane population N; N*p1 must be an integer")
    parser.add_argument("--grid", type=float, default=1e-4,
                        help="pessimistic PLD discretization interval")
    parser.add_argument("--delta-min", type=float, default=1e-8)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-max", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    if not 0 < args.p1 <= 1 or any(not 0 <= p < 1 for p in args.p2):
        parser.error("require 0 < p1 <= 1 and 0 <= p2 < 1 (p2=1 is the group baseline)")
    if args.degree < 1 or args.population < sum(args.degree**ell for ell in range(4)):
        parser.error("require degree >= 1 and population >= 1+K+K^2+K^3")
    if not np.isfinite(args.sigma) or args.sigma <= 0 or not np.isfinite(args.grid) or args.grid <= 0:
        parser.error("sigma and grid must be finite and positive")
    if not 0 < args.delta_min < 1 or args.points < 2:
        parser.error("require 0 < delta-min < 1 and points >= 2")
    if not (np.isfinite(args.epsilon_min) and np.isfinite(args.epsilon_max)
            and 0 <= args.epsilon_min < args.epsilon_max):
        parser.error("require finite 0 <= epsilon-min < epsilon-max")
    if not np.isclose(args.population * args.p1, round(args.population * args.p1), rtol=0, atol=1e-8):
        parser.error("population*p1 must be an integer for the fixed-size Daigavane batch")
    args.p2 = sorted(set(args.p2))
    return args


def draw_panel(ax, epsilon, curves, radius, steps, args):
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
    ax.plot(epsilon, curves[0], color="0.35", linestyle=":", linewidth=2,
            label="Daigavane et al. (RDP)")
    ax.plot(epsilon, curves[1], color="black", linestyle="--", linewidth=1.8,
            label=r"Group privacy ($p_2=1$)")
    for index, (p2, delta) in enumerate(zip(args.p2, curves[2:])):
        ax.plot(epsilon, delta, color=colors[index % len(colors)], linewidth=1.8,
                label=rf"Ours ($p_2={p2:g}$)")
    ax.set(xlabel=r"$\epsilon$", ylabel=r"$\delta(\epsilon)$",
           title=rf"$r={radius},\ t={steps}$", xlim=(args.epsilon_min, args.epsilon_max),
           ylim=(args.delta_min, 1))
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.grid(which="major", color="0.88", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 15,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    batch_size = round(args.population * args.p1)
    parameters = {**vars(args), "out_dir": str(args.out_dir), "batch_size": batch_size,
                  "radii": RADII, "steps": STEPS, "chi": 1, "orders": ORDERS.tolist(),
                  "root_sampling": {"daigavane": "fixed-size without replacement",
                                    "ours_and_group": "Bernoulli"},
                  "noise_convention": "sigma = noise_std / clipping_norm",
                  "group_definition": "our Gaussian-mixture pair at p2=1",
                  "composition": "pessimistic connect-the-dots PLD; tail_mass_truncation=1e-15",
                  "sources": ["https://arxiv.org/abs/2111.15521",
                              "https://github.com/google-research/google-research/blob/master/differentially_private_gnns/privacy_accountants.py"],
                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__).resolve(), ROOT / "src/privacy/accounting.py",
                                                 ROOT / "src/privacy/privacy_loss.py")},
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "matplotlib", "dp-accounting")}}
    (args.out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    overview, axes = plt.subplots(2, 2, figsize=(11, 8))
    caption = (rf"$p_1={args.p1:g},\ K={args.degree},\ \sigma/C={args.sigma:g}$"
               + f"; Daigavane N={args.population:,}, m={batch_size:,}")
    with (args.out_dir / "curves.csv").open("w", newline="") as curves_file, \
         (args.out_dir / "pair_weights.csv").open("w", newline="") as pairs_file:
        writer = csv.writer(curves_file)
        writer.writerow(["r", "t", "method", "p2", "epsilon", "delta", "raw_delta"])
        pair_writer = csv.writer(pairs_file)
        pair_writer.writerow(["r", "p2", "j", "probability", "P_mean", "Q_mean", "noise_std"])
        for row, radius in enumerate(RADII):
            max_terms = sum(args.degree**ell for ell in range(radius + 1))
            base_rdp = daigavane_rdp(args.population, batch_size, max_terms, args.sigma)
            plds = []
            for p2 in [1.0, *args.p2]:
                weights = sparsegnn_mixture_weights(args.p1, p2, radius, args.degree,
                                                    args.degree, union_safe=False)
                pair_writer.writerows((radius, p2, j, float(prob), -j, j, args.sigma)
                                      for j, prob in enumerate(weights))
                print(f"Building pair: r={radius}, p2={p2:g}, support={len(weights)}", flush=True)
                plds.append(mixture_gaussian_pld(weights, args.sigma, args.grid))
            for col, steps in enumerate(STEPS):
                composed = [pld if steps == 1 else pld.self_compose(steps) for pld in plds]
                epsilon = np.linspace(args.epsilon_min, args.epsilon_max, args.points)
                raw_curves = [np.array([compute_delta(ORDERS, steps * base_rdp, value)[0]
                                    for value in epsilon]),
                          *(pld.get_delta_for_epsilon(epsilon) for pld in composed)]
                curves = [np.clip(delta, 0, 1) for delta in raw_curves]
                for method, p2, delta, raw_delta in zip(
                        ["daigavane", "group", *(["ours"] * len(args.p2))],
                        ["", 1.0, *args.p2], curves, raw_curves):
                    writer.writerows((radius, steps, method, p2, float(e), float(d), float(raw))
                                     for e, d, raw in zip(epsilon, delta, raw_delta))
                draw_panel(axes[row, col], epsilon, curves, radius, steps, args)
                figure, ax = plt.subplots(figsize=(6.8, 5.4))
                draw_panel(ax, epsilon, curves, radius, steps, args)
                figure.legend(*ax.get_legend_handles_labels(), loc="lower center",
                              bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=9)
                figure.suptitle(caption, fontsize=11)
                # figure.text(0.5, 0.01, r"Linear $\epsilon$ axis; lower $\delta$ is tighter.",
                #             ha="center", fontsize=9)
                figure.tight_layout(rect=(0, 0.21, 1, 0.95))
                for extension in ("png", "pdf"):
                    figure.savefig(args.out_dir / f"delta_r{radius}_t{steps}.{extension}", dpi=180)
                plt.close(figure)
                print(f"Saved r={radius}, t={steps}; epsilon range [{epsilon[0]:g}, {epsilon[-1]:g}]", flush=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    overview.suptitle(caption)
    overview.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                    ncol=3, fontsize=10)
    overview.text(0.5, 0.01, r"Linear $\epsilon$ axis; lower $\delta$ is tighter.",
                  ha="center", fontsize=10)
    overview.tight_layout(rect=(0, 0.14, 1, 0.95))
    for extension in ("png", "pdf"):
        overview.savefig(args.out_dir / f"comparison.{extension}", dpi=180)
    plt.close(overview)
    print(f"Figures and numerical data: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
