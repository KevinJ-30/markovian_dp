"""Shared setup for the appendix figures; accounting and styles live in compare.py."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import compare
from compare import plt

RADII = (1, 2, 3)
HORIZONS = (1, 100, 1000)


def parse_args(name):
    parser = compare.argument_parser()
    parser.description = f"Render the {name or 'complete'} appendix privacy comparison."
    parser.set_defaults(out_dir=Path(__file__).resolve().parent / "figures" / name)
    parser.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    parser.add_argument("--sweep-points", type=int, default=10,
                        help="number of noise/root-sampling points (degree uses integers 1..8)")
    args = compare.parse_args(parser)
    if any(radius < 1 for radius in args.radii) or len(set(args.radii)) != len(args.radii):
        parser.error("radii must be distinct positive integers")
    if args.sweep_points < 2:
        parser.error("sweep-points must be at least two")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def methods(args):
    return [("daigavane", ""), ("group", 1.0), *[("ours", p2) for p2 in args.p2]]


def build_accountants(args, radius, *, p1=None, sigma=None, degree=None):
    p1 = args.p1 if p1 is None else float(p1)
    sigma = args.sigma if sigma is None else float(sigma)
    degree = args.degree if degree is None else int(degree)
    max_terms = sum(degree**ell for ell in range(radius + 1))
    batch_size = round(args.population * p1)
    if max_terms > args.population:
        raise ValueError("population must cover 1+K+...+K^r for every plotted configuration")
    if not 0 < batch_size <= args.population or abs(args.population * p1 - batch_size) > 1e-8:
        raise ValueError("population*p1 must be a positive integer no larger than population")
    base_rdp = compare.daigavane_rdp(args.population, batch_size, max_terms, sigma)
    plds = []
    for p2 in [1.0, *args.p2]:
        weights = compare.sparsegnn_mixture_weights(p1, p2, radius, degree, degree,
                                                   union_safe=False)
        print(f"Building pair: r={radius}, K={degree}, p1={p1:g}, p2={p2:g}, "
              f"sigma={sigma:g}, support={len(weights)}", flush=True)
        started = time.perf_counter()
        plds.append(compare.mixture_gaussian_pld(weights, sigma, args.grid))
        print(f"Built pair in {time.perf_counter() - started:.1f}s", flush=True)
    return base_rdp, plds


def new_figure(args, ncols):
    compare.set_plot_style()
    return plt.subplots(len(args.radii), ncols, squeeze=False,
                        figsize=(6 * ncols, 4.8 * len(args.radii) + 1))


def save_figure(figure, axes, args, name):
    for index, ax in enumerate(axes.flat):
        ax.set_title(f"({chr(ord('a') + index)}) {ax.get_title()}", pad=18)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.out_dir / f"{name}.{extension}", dpi=180,
                       bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    print(f"Figure and numerical data: {args.out_dir}", flush=True)


def write_parameters(args, **extra):
    sources = [Path(__file__).resolve(), Path(compare.__file__).resolve(),
               Path(sys.argv[0]).resolve(), compare.ROOT / "src/privacy/accounting.py",
               compare.ROOT / "src/privacy/privacy_loss.py"]
    if "sweep" in sys.modules:
        sources.append(Path(sys.modules["sweep"].__file__).resolve())
    parameters = {
        **vars(args), "out_dir": str(args.out_dir), "radii": args.radii,
        "batch_size": round(args.population * args.p1), "chi": 1,
        "orders": compare.ORDERS.tolist(),
        "root_sampling": {"daigavane": "fixed-size without replacement",
                          "ours_and_group": "Bernoulli"},
        "noise_convention": "sigma = noise_std / clipping_norm",
        "group_definition": "our Gaussian-mixture pair at p2=1",
        "composition": "pessimistic connect-the-dots PLD; tail_mass_truncation=1e-15",
        "source_sha256": {str(path.relative_to(compare.ROOT)):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "matplotlib", "dp-accounting")},
        **extra,
    }
    (args.out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
