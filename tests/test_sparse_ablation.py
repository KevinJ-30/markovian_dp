"""Five-seed ablation aggregation and selected-result integrity."""
import csv
import json
import math
from pathlib import Path

import pytest

from scripts import sparse_ablation as analysis


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def save_output(directory, result, config):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "config.json").write_text(json.dumps(config))
    write_csv(directory / "result.csv", [result])


def ordinary_run(root, ordinal=0, *, protocol="ogbn-products", method="sparse_sage",
                 radius=1, p2=.1, cap=5, seed=1, epsilon=8):
    dataset, metric, binary, multilabel, strategy = analysis.TASKS[protocol]
    score = (seed - 1) / 10 if method == "sparse_sage" else 0.
    parameters = {
        "evaluate_every": 2, "hidden": 128, "r": radius, "p2": p2,
        "K_out": cap, "K_in": 10, "layers": 3 if radius == 3 else 2,
        "incoming_sampling_caps": [20, 10, 5][:radius], "weight_decay": 0.,
        "cap_seed": 20000 + seed, "K_in_achieved": 100 + seed,
    }
    result = {
        "protocol": protocol, "dataset": dataset, "method": method, "metric": metric,
        "status": "completed", "target_epsilon": epsilon, "epsilon": epsilon - .01,
        "delta": .01, "seed": seed, "lr": .001 if radius == 1 else .01,
        "requested_batch_size": 1024 if radius == 1 else 256,
        "batch_size": 100, "effective_batch_size": 100, "epochs": 20,
        "parameters": parameters, "hidden": 128, "dropout": .5, "weight_decay": 0.,
        "split": f"{strategy}:seed0", "split_strategy": strategy, "split_seed": 0,
        "domain_split": None, "domain_split_id": None, "device": "cpu",
        "test_metric": score, "validation_metric": 1 - score,
        "selection": {"split": "validation", "metric": metric, "step": 2,
                      "validation_score": 1 - score},
    }
    config = {**result, "train_nodes": 100,
              "partitions": {"test": {"nodes": 10, "evaluated_nodes": 10, "edges": seed * cap}},
              "task": {"primary_metric": metric, "binary": binary, "multilabel": multilabel,
                       "regression": False}}
    directory = root / "runs" / f"{ordinal:04d}" / "attempts" / "1" / "output"
    save_output(directory, result, config)
    indexed = {"status": "completed", "run_id": f"{ordinal:04d}", "attempt": 1,
               "output_dir": str(directory), "protocol": protocol, "method": method,
               "target_epsilon": epsilon, "test_metric": score, "validation_metric": 1 - score}
    return indexed, result, config, directory


def complete_ofat(root, epsilon=8):
    rows = []
    settings = [(r, .1, 5) for r in (1, 2, 3)]
    settings += [(1, p, 5) for p in (.05, .25, .5, 1.)]
    settings += [(1, .1, cap) for cap in (10, 20, 40)]
    for protocol in ("ogbn-products", "fb100-year-6", "ogbn-arxiv"):
        for method in ("sparse_sage", "sparse_gin"):
            for radius, p2, cap in settings:
                for seed in (1, 2, 3, 4, 5):
                    indexed, _, _, _ = ordinary_run(
                        root, len(rows), protocol=protocol, method=method,
                        radius=radius, p2=p2, cap=cap, seed=seed, epsilon=epsilon)
                    rows.append(indexed)
    write_csv(root / "results.csv", rows)
    return rows


def test_seed_means_and_standard_deviations_keep_each_ablation_point(tmp_path):
    complete_ofat(tmp_path)
    rows, signatures = analysis.read_results(tmp_path, 8)
    assert set(signatures) == {"ogbn-products", "fb100-year-6", "ogbn-arxiv"}
    points = analysis.aggregate_seeds(rows)
    assert len(points) == 60
    for point in points:
        assert point["n"] == 5 and point["seeds"] == [1, 2, 3, 4, 5]
        if point["method"] == "sparse_sage":
            assert point["test_mean"] == pytest.approx(.2)
            assert point["validation_mean"] == pytest.approx(.8)
            assert point["test_sd"] == pytest.approx(math.sqrt(.025))
        else:
            assert point["test_mean"] == point["test_sd"] == 0.
    curves = analysis.relationship_curves(points)
    assert len(curves) == 72
    products_sage = [row for row in curves if row["protocol"] == "ogbn-products"
                     and row["method"] == "sparse_sage"]
    assert [row["curve_value"] for row in products_sage if row["curve_parameter"] == "r"] == [1, 2, 3]
    assert [row["curve_value"] for row in products_sage if row["curve_parameter"] == "p2"] == [.05, .1, .25, .5, 1.]
    assert [row["curve_value"] for row in products_sage if row["curve_parameter"] == "K_out"] == [5, 10, 20, 40]


def test_rendered_error_bars_span_sample_sd_not_standard_error(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    complete_ofat(tmp_path)
    rows, _ = analysis.read_results(tmp_path, 8)
    curves = analysis.relationship_curves(analysis.aggregate_seeds(rows))
    figures = []
    close = plt.close

    def record_closed_figure(figure=None):
        if isinstance(figure, Figure):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(plt, "close", record_closed_figure)
    analysis.draw_figures(curves, tmp_path)
    axis = figures[-1].axes[0]
    sage_bars = axis.containers[0].lines[2][0].get_segments()
    gin_bars = axis.containers[1].lines[2][0].get_segments()
    for segment in sage_bars:
        assert segment[:, 1] == pytest.approx((.2 - math.sqrt(.025), .2 + math.sqrt(.025)))
    for segment in gin_bars:
        assert segment[:, 1] == pytest.approx((0., 0.))


def test_moved_repeat_root_resolves_recorded_outputs(tmp_path):
    indexed = complete_ofat(tmp_path)
    original = Path("/previous/location/experiment")
    for row in indexed:
        row["output_dir"] = str(original / Path(row["output_dir"]).relative_to(tmp_path))
    (tmp_path / "state.json").write_text(json.dumps({"root": str(original)}))
    write_csv(tmp_path / "results.csv", indexed)
    rows, _ = analysis.read_results(tmp_path, 8)
    assert all(Path(row["actual_output_path"]).is_relative_to(tmp_path) for row in rows)
    assert rows[0]["test_metric"] == 0.


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "split", "metric", "seed0", "layers", "caps"])
def test_incomplete_ambiguous_or_conflicting_inputs_are_rejected(tmp_path, mutation):
    indexed = complete_ofat(tmp_path)
    if mutation == "missing":
        indexed.pop()
    elif mutation == "duplicate":
        indexed.append(indexed[0].copy())
    elif mutation == "failed":
        indexed[0]["status"] = "failed"
    else:
        directory = Path(indexed[0]["output_dir"])
        result = json.loads((directory / "result.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        if mutation in ("layers", "caps"):
            field, wrong = ("layers", 3) if mutation == "layers" else ("incoming_sampling_caps", [10])
            result["parameters"][field] = config["parameters"][field] = wrong
        else:
            field = {"split": "domain_split_id", "metric": "metric", "seed0": "seed"}[mutation]
            result[field] = config[field] = 0 if mutation == "seed0" else "conflicting"
        save_output(directory, result, config)
    write_csv(tmp_path / "results.csv", indexed)
    with pytest.raises(ValueError):
        analysis.read_results(tmp_path, 8)


@pytest.mark.parametrize("field", ["lr", "batch_size", "clip"])
def test_mixed_hyperparameters_cannot_be_averaged_as_seed_variation(tmp_path, field):
    cohort = []
    for seed in range(1, 6):
        indexed, _, _, _ = ordinary_run(tmp_path, seed, seed=seed)
        row, _ = analysis.read_run(tmp_path, tmp_path / "results.csv", indexed)
        cohort.append(row)
    if field == "clip":
        cohort[0]["parameters"]["clip"] = .5
    else:
        cohort[0][field] *= 2
    with pytest.raises(ValueError, match="mixed repeat configurations"):
        analysis.aggregate_seeds(cohort)


def test_epsilon_selection_does_not_pool_budgets(tmp_path):
    indexed = complete_ofat(tmp_path, epsilon=2)
    indexed.append({"target_epsilon": 8, "status": "pending"})
    write_csv(tmp_path / "results.csv", indexed)
    rows, _ = analysis.read_results(tmp_path, 2)
    assert {row["epsilon"] for row in rows} == {2}
    assert len(rows) == 300


def test_selected_output_cannot_escape_root(tmp_path):
    root = tmp_path / "input"
    indexed, _, _, _ = ordinary_run(tmp_path / "external-run")
    with pytest.raises(ValueError):
        analysis.read_run(root, root / "results.csv", indexed, str(root))
