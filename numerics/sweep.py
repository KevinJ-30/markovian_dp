"""Shared numerical accountant sweeps for the appendix figures."""

import csv

import matplotlib.ticker as ticker
import numpy as np

import appendix
import compare


SWEEPS = {
    "sigma": ("noise", r"Noise multiplier $\sigma$"),
    "p1": ("root_sampling", r"Root-sampling probability $p_1$"),
    "degree": ("degree", r"Degree bound $K$"),
}


def run_sweep(parameter):
    """Sweep one accountant parameter at each radius and fixed horizon."""
    name, xlabel = SWEEPS[parameter]
    args = appendix.parse_args(name)
    metadata = {"swept_parameter": parameter, "horizons": list(appendix.HORIZONS)}
    if parameter == "sigma":
        values = np.linspace(1, 10, args.sweep_points)
    elif parameter == "p1":
        batch_sizes = np.unique(np.rint(np.geomspace(
            0.001 * args.population, 0.1 * args.population,
            args.sweep_points)).astype(int))
        if np.any(batch_sizes <= 0) or np.any(batch_sizes > args.population):
            raise ValueError("root-sampling sweep requires positive batch sizes <= population")
        values = batch_sizes / args.population
        metadata["batch_sizes"] = batch_sizes.tolist()
    else:
        values = np.arange(1, 9)
    metadata["swept_values"] = values.tolist()

    appendix.write_parameters(args, **metadata)
    fig, axes = appendix.new_figure(args, len(appendix.HORIZONS))
    methods = appendix.methods(args)
    delta_label = ticker.ScalarFormatter(useMathText=True).format_data(args.delta)
    with (args.out_dir / "curves.csv").open("w", newline="") as curves_file:
        writer = csv.writer(curves_file)
        writer.writerow(["r", "t", "parameter", "value", "method", "p2", "delta", "epsilon"])
        for row, radius in enumerate(args.radii):
            curves = np.empty((len(appendix.HORIZONS), len(methods), len(values)))
            for point, value in enumerate(values):
                value = value.item()
                print(f"Evaluating sweep: r={radius}, {parameter}={value:g}", flush=True)
                base_rdp, plds = appendix.build_accountants(args, radius, **{parameter: value})
                for column, steps in enumerate(appendix.HORIZONS):
                    epsilons = curves[column, :, point]
                    epsilons[0] = compare.compute_epsilon(
                        compare.ORDERS, steps * base_rdp, args.delta)[0]
                    for index, pld in enumerate(plds, start=1):
                        composed = pld if steps == 1 else pld.self_compose(int(steps))
                        epsilons[index] = compare.epsilon_for_delta(composed, args.delta)
                    if not np.all(np.isfinite(epsilons)):
                        raise ValueError(
                            f"non-finite epsilon at r={radius}, T={steps}, {parameter}={value:g}")
                    writer.writerows(
                        (radius, steps, parameter, value, method, p2, args.delta, float(epsilon))
                        for (method, p2), epsilon in zip(methods, epsilons))
            for column, steps in enumerate(appendix.HORIZONS):
                ax = axes[row, column]
                compare.draw_curves(ax, values, curves[column], args)
                ax.set(xlabel=xlabel, ylabel=rf"$\epsilon\ (\delta={delta_label})$",
                       title=rf"$R={radius},\ T={steps}$",
                       ylim=(0, None))
                ax.set_yscale("linear")
                ax.set_xscale("log" if parameter == "p1" else "linear")
                if len(values) > 1:
                    ax.set_xlim(values[0], values[-1])
                if parameter == "p1":
                    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
                    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                elif parameter == "degree":
                    ax.set_xticks(values)
                ax.grid(which="major", color="0.88", linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
    appendix.save_figure(fig, axes, args, name)
