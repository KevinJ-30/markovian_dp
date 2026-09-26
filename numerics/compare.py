r"""Plot epsilon(T) and composed delta(epsilon); run `python numerics/compare.py`.

We clip to C = 1. 

Our pair: 
    J = Bernoulli(p1) + sum_{ell=1}^r Binomial(K**ell, b_ell), independently,
    b_ell = p1*p2**ell / (1-p1+p1*p2**ell),
    P = sum_j Pr(J=j) Normal(-j, sigma**2),
    Q = sum_j Pr(J=j) Normal(+j, sigma**2).
At p2=1, J is Binomial(M,p1): this is the requested group-privacy baseline.
We compose the pair's pessimistically discretized PLD.

Group privacy:

This is just our pair with p2=1; see our main paper for more details.

Daigavane et al., Theorem 1: https://arxiv.org/abs/2111.15521
    H ~ Hypergeometric(N, M, m), with m/N = p1,
    R_alpha = log E exp(2*alpha*(alpha-1)*H**2/sigma**2) / (alpha-1).
This is also the Renyi divergence of the count-revealing joint Gaussian pair
    P_D(j,x) = Pr(H=j)*Normal(x; -j, sigma**2),
    Q_D(j,x) = Pr(H=j)*Normal(x; +j, sigma**2).
The plotted Daigavane curve uses T*R_alpha followed by Google's RDP-to-DP
conversion, as in the original repository, NOT a hidden-count mixture PLD.
https://github.com/google-research/google-research/blob/master/
    differentially_private_gnns/privacy_accountants.py


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
from dp_accounting.rdp.rdp_privacy_accountant import compute_delta, compute_epsilon
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.privacy.accounting import mixture_gaussian_pld, sparsegnn_mixture_weights

RADII = (1, 3)
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


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p1", type=float, default=0.01)
    parser.add_argument("--p2", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian noise standard deviation divided by C")
    parser.add_argument("--population", type=int, default=100000,
                        help="Daigavane population N; N*p1 must be an integer")
    parser.add_argument("--grid", type=float, default=1e-3,
                        help="pessimistic PLD discretization interval")
    parser.add_argument("--delta-min", type=float, default=1e-8)
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="fixed delta for epsilon versus composition iterations")
    parser.add_argument("--steps", type=int, default=1000,
                        help="composition horizon for both kinds of panel")
    parser.add_argument("--iteration-points", type=int, default=100,
                        help="number of composition checkpoints, plus T=0")
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-max", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    return parser


def parse_args(parser=None):
    parser = argument_parser() if parser is None else parser
    args = parser.parse_args()
    
    if not 0 < args.p1 <= 1 or any(not 0 <= p < 1 for p in args.p2):
        parser.error("require 0 < p1 <= 1 and 0 <= p2 < 1 (p2=1 is the group baseline)")
    if args.degree < 1 or args.population < sum(args.degree**ell for ell in range(4)):
        parser.error("require degree >= 1 and population >= 1+K+K^2+K^3")
    if not np.isfinite(args.sigma) or args.sigma <= 0 or not np.isfinite(args.grid) or args.grid <= 0:
        parser.error("sigma and grid must be finite and positive")
    if not 0 < args.delta_min < 1 or args.points < 2:
        parser.error("require 0 < delta-min < 1 and points >= 2")
    if not 0 < args.delta < 1 or args.steps < 1 or args.iteration_points < 2:
        parser.error("require 0 < delta < 1, steps >= 1, and iteration-points >= 2")
    if not (np.isfinite(args.epsilon_min) and np.isfinite(args.epsilon_max)
            and 0 <= args.epsilon_min < args.epsilon_max):
        parser.error("require finite 0 <= epsilon-min < epsilon-max")
    if not np.isclose(args.population * args.p1, round(args.population * args.p1), rtol=0, atol=1e-8):
        parser.error("population*p1 must be an integer for the fixed-size Daigavane batch")
    args.p2 = sorted(set(args.p2))
    return args


def epsilon_for_delta(pld, delta):
    """Invert discretized PLD."""

    pmfs = [pld._pmf_remove] if pld._symmetric else [pld._pmf_remove, pld._pmf_add]
    epsilons = []
    for pmf in pmfs:
        dense = pmf.to_dense_pmf()
        if dense._infinity_mass > delta:
            return float("inf")
        losses = (np.arange(dense.size) + dense._lower_loss) * dense._discretization

        def excess(epsilon):
            start = np.searchsorted(losses, epsilon, side="right")
            return (dense._infinity_mass
                    + np.dot(-np.expm1(epsilon - losses[start:]), dense._probs[start:])
                    - delta)

        if excess(0.0) <= 0:
            epsilons.append(0.0)
        else:
            epsilons.append(brentq(excess, 0.0, max(0.0, losses[-1]), xtol=1e-10))
    return max(epsilons)


def composition_profiles(plds, base_rdp, iterations, delta):
    """Evaluate checkpoints."""
    curves = np.zeros((len(plds) + 1, len(iterations)))
    for index, steps in enumerate(iterations):
        if steps == 0:
            continue
        composed = [pld if steps == 1 else pld.self_compose(int(steps)) for pld in plds]
        curves[0, index] = compute_epsilon(ORDERS, steps * base_rdp, delta)[0]
        curves[1:, index] = [epsilon_for_delta(pld, delta) for pld in composed]
    return curves, composed


def draw_curves(ax, x, curves, args):
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00"]
    ax.plot(x, curves[0], color="0.2", linestyle=":", linewidth=3, alpha=1,
            label="Daigavane et al. (RDP)")
    ax.plot(x, curves[1], color="black", linestyle="--", linewidth=2.8, alpha=1,
            label=r"Group privacy ($p_2=1$)")
    for index, (p2, values) in enumerate(zip(args.p2, curves[2:])):
        ax.plot(x, values, color=colors[index % len(colors)], linewidth=2.8, alpha=1,
                label=rf"Ours ($p_2={p2:g}$)")


def draw_panel(ax, x, curves, radius, args, *, composition=False):
    draw_curves(ax, x, curves, args)
    if composition:
        delta_label = matplotlib.ticker.ScalarFormatter(useMathText=True).format_data(args.delta)
        ax.set(xlabel=r"Composition iterations $T$", ylabel=r"$\epsilon$",
               title=rf"$R={radius},\ \delta={delta_label}$",
               xlim=(0, args.steps), ylim=(0, None))
    else:
        ax.set(xlabel=r"$\epsilon$", ylabel=r"$\delta(\epsilon)$",
               title=rf"$R={radius},\ T={args.steps}$",
               xlim=(args.epsilon_min, args.epsilon_max), ylim=(args.delta_min, 1))
        ax.set_yscale("log")
    ax.set_xscale("linear")
    ax.grid(which="major", color="0.88", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def set_plot_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 18,
                         "axes.labelsize": 22, "axes.titlesize": 21,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    batch_size = round(args.population * args.p1)
    iterations = np.unique(np.concatenate(([0], np.rint(
        np.linspace(1, args.steps, min(args.steps, args.iteration_points))).astype(int))))
    parameters = {**vars(args), "out_dir": str(args.out_dir), "batch_size": batch_size,
                  "radii": RADII, "iterations": iterations.tolist(), "chi": 1, "orders": ORDERS.tolist(),
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
    overview, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    with (args.out_dir / "curves.csv").open("w", newline="") as curves_file, \
         (args.out_dir / "composition.csv").open("w", newline="") as composition_file, \
         (args.out_dir / "pair_weights.csv").open("w", newline="") as pairs_file:
        writer = csv.writer(curves_file)
        writer.writerow(["r", "t", "method", "p2", "epsilon", "delta", "raw_delta"])
        composition_writer = csv.writer(composition_file)
        composition_writer.writerow(["r", "t", "method", "p2", "delta", "epsilon"])
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
            methods = ["daigavane", "group", *(["ours"] * len(args.p2))]
            probabilities = ["", 1.0, *args.p2]
            epsilon_curves, composed = composition_profiles(plds, base_rdp, iterations, args.delta)
            for method, p2, values in zip(methods, probabilities, epsilon_curves):
                composition_writer.writerows(
                    (radius, int(t), method, p2, args.delta, float(value))
                    for t, value in zip(iterations, values))

            epsilon = np.linspace(args.epsilon_min, args.epsilon_max, args.points)
            raw_curves = [np.array([compute_delta(ORDERS, args.steps * base_rdp, value)[0]
                                   for value in epsilon]),
                          *(pld.get_delta_for_epsilon(epsilon) for pld in composed)]
            curves = [np.clip(delta, 0, 1) for delta in raw_curves]
            for method, p2, delta, raw_delta in zip(methods, probabilities, curves, raw_curves):
                writer.writerows((radius, args.steps, method, p2, float(e), float(d), float(raw))
                                 for e, d, raw in zip(epsilon, delta, raw_delta))
            panels = [
                (iterations, epsilon_curves, True),
                (epsilon, curves, False),
            ]
            for col, (x, values, is_composition) in enumerate(panels):
                panel_label = ("(a)", "(b)", "(c)", "(d)")[2 * row + col]
                ax = axes[2 * row + col]
                draw_panel(ax, x, values, radius, args, composition=is_composition)
                ax.set_title(f"{panel_label} {ax.get_title()}")
    handles, labels = axes[0].get_legend_handles_labels()
    overview.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                    ncol=6, fontsize=17)
    overview.subplots_adjust(left=0.04, right=0.98, bottom=0.19, top=0.77, wspace=0.32)
    for extension in ("png", "pdf", "svg"):
        overview.savefig(args.out_dir / f"comparison.{extension}", dpi=180,
                         bbox_inches="tight", pad_inches=0.15)
    plt.close(overview)
    print(f"Figures and numerical data: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
