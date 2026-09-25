"""Plot epsilon(T) and delta(epsilon) at T=1, 100, 1000 for each radius.

Run from the project root with ``python numerics/compare_expanded.py``.
"""

import argparse
import csv
import time

import numpy as np

import appendix
import compare


def main():
    args = appendix.parse_args("expanded")
    iterations = np.unique(np.concatenate(([0], np.rint(
        np.linspace(1, args.steps, min(args.steps, args.iteration_points))).astype(int))))
    evaluated_iterations = np.union1d(iterations, appendix.HORIZONS)
    iteration_indices = {int(t): index for index, t in enumerate(iterations)}
    horizon_columns = {t: column for column, t in enumerate(appendix.HORIZONS, start=1)}
    epsilon = np.linspace(args.epsilon_min, args.epsilon_max, args.points)
    methods = appendix.methods(args)
    appendix.write_parameters(
        args, radii=args.radii, horizons=list(appendix.HORIZONS),
        iterations=evaluated_iterations.tolist(), composition_iterations=iterations.tolist())
    figure, axes = appendix.new_figure(args, 1 + len(appendix.HORIZONS))

    with (args.out_dir / "composition.csv").open("w", newline="") as composition_file, \
         (args.out_dir / "curves.csv").open("w", newline="") as curves_file:
        composition_writer = csv.writer(composition_file)
        composition_writer.writerow(["r", "t", "method", "p2", "delta", "epsilon"])
        curves_writer = csv.writer(curves_file)
        curves_writer.writerow(["r", "t", "method", "p2", "epsilon", "delta", "raw_delta"])

        for row, radius in enumerate(args.radii):
            print(f"Building expanded row: r={radius}", flush=True)
            base_rdp, plds = appendix.build_accountants(args, radius)
            started = time.perf_counter()
            print(f"Composing expanded row: r={radius}", flush=True)
            epsilon_curves = np.zeros((len(methods), len(iterations)))
            for steps in evaluated_iterations:
                steps = int(steps)
                if steps == 0:
                    continue
                iteration_index = iteration_indices.get(steps)
                horizon_column = horizon_columns.get(steps)
                composed_rdp = steps * base_rdp
                raw_curves = (np.empty((len(methods), len(epsilon)))
                              if horizon_column is not None else None)
                if iteration_index is not None:
                    epsilon_curves[0, iteration_index] = compare.compute_epsilon(
                        compare.ORDERS, composed_rdp, args.delta)[0]
                if raw_curves is not None:
                    raw_curves[0] = [compare.compute_delta(
                        compare.ORDERS, composed_rdp, value)[0] for value in epsilon]

                # Compose each accountant only once at this checkpoint, and release
                # it after extracting both profiles rather than retaining all PLDs.
                for method_index, pld in enumerate(plds, start=1):
                    composed = pld if steps == 1 else pld.self_compose(steps)
                    if iteration_index is not None:
                        epsilon_curves[method_index, iteration_index] = compare.epsilon_for_delta(
                            composed, args.delta)
                    if raw_curves is not None:
                        raw_curves[method_index] = composed.get_delta_for_epsilon(epsilon)
                    del composed

                if iteration_index is not None and not np.all(
                        np.isfinite(epsilon_curves[:, iteration_index])):
                    raise ValueError(f"Non-finite epsilon profile for r={radius}, T={steps}")
                if raw_curves is not None:
                    if not np.all(np.isfinite(raw_curves)):
                        raise ValueError(f"Non-finite delta profile for r={radius}, T={steps}")
                    curves = np.clip(raw_curves, 0, 1)
                    for (method, p2), values, raw_values in zip(methods, curves, raw_curves):
                        curves_writer.writerows(
                            (radius, steps, method, p2, float(e), float(d), float(raw))
                            for e, d, raw in zip(epsilon, values, raw_values))
                    panel_args = argparse.Namespace(**vars(args))
                    panel_args.steps = steps
                    compare.draw_panel(axes[row, horizon_column], epsilon, curves,
                                       radius, panel_args)

            for (method, p2), values in zip(methods, epsilon_curves):
                composition_writer.writerows(
                    (radius, int(t), method, p2, args.delta, float(value))
                    for t, value in zip(iterations, values))
            compare.draw_panel(axes[row, 0], iterations, epsilon_curves,
                               radius, args, composition=True)
            print(f"Composed expanded row: r={radius} in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)

    appendix.save_figure(figure, axes, args, "expanded")


if __name__ == "__main__":
    main()
