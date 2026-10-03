"""Paper-curve loading from ordinary selected-result indexes."""
import csv
import json
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


def ordinary_run(root, ordinal=0, *, protocol="ogbn-arxiv", method="sparse_sage",
                 radius=1, p2=.5, cap=10, ci=True):
    dataset, metric, binary, multilabel, strategy = analysis.TASKS[protocol]
    sparse = method.startswith("sparse_")
    parameters = {"evaluate_every": 2, "hidden": 128}
    if sparse:
        parameters.update(r=radius, p2=p2, K_out=cap, K_in=10, layers=2)
    else:
        parameters.update(max_degree=5)
        parameters["depth" if method == "progap" else "radius"] = radius
    result = {
        "protocol": protocol, "dataset": dataset, "method": method, "metric": metric,
        "status": "completed", "target_epsilon": 8., "epsilon": 7.99,
        "delta": .01, "seed": 0, "lr": .01, "requested_batch_size": 256,
        "batch_size": 100, "effective_batch_size": 100, "epochs": 20,
        "parameters": parameters, "hidden": 128, "dropout": .5,
        "split": f"{strategy}:seed0", "split_strategy": strategy, "split_seed": 0,
        "domain_split": None, "domain_split_id": None, "device": "cpu",
        "test_metric": 0., "validation_metric": 0.,
        "selection": {"split": "validation", "metric": metric, "step": 2,
                      "validation_score": 0.},
    }
    if ci:
        result["test_confidence_intervals"] = {
            "confidence_level": .95, "method": "percentile", "resampling_unit": "node",
            "n_resamples": 1000, "seed": 0, "n_observations": 10,
            "metrics": {metric: {"lower": 0., "upper": .1, "valid_resamples": 1000}},
        }
    config = {**result, "train_nodes": 100, "partitions": {"test": {"nodes": 10}},
              "task": {"primary_metric": metric, "binary": binary, "multilabel": multilabel,
                       "regression": False}}
    directory = root / "runs" / f"{ordinal:04d}" / "attempts" / "1" / "output"
    save_output(directory, result, config)
    indexed = {"status": "completed", "run_id": f"{ordinal:04d}", "attempt": 1,
               "output_dir": str(directory), "protocol": protocol, "method": method,
               "test_metric": 0., "validation_metric": 0.}
    return indexed, result, config, directory


def complete_ofat(root):
    rows = []
    settings = [(r, .5, 10) for r in (1, 2, 3)]
    settings += [(1, p, 10) for p in (.05, .1, .25, 1.)]
    settings += [(1, .5, cap) for cap in (5, 20, 40)]
    for protocol in ("ogbn-arxiv", "saint-yelp", "twitch-allbut2"):
        for method in ("sparse_sage", "sparse_gin"):
            for radius, p2, cap in settings:
                indexed, _, _, _ = ordinary_run(root, len(rows), protocol=protocol, method=method,
                                                radius=radius, p2=p2, cap=cap)
                rows.append(indexed)
    write_csv(root / "results.csv", rows)
    return rows


def test_complete_plain_index_preserves_zero_metrics_and_missing_resources(tmp_path):
    complete_ofat(tmp_path)
    rows, signatures = analysis.read_results(tmp_path, "ofat")
    assert set(signatures) == {"ogbn-arxiv", "saint-yelp", "twitch-allbut2"}
    assert all(row["test_metric"] == 0. and row["validation_metric"] == 0. for row in rows)
    assert all(row["peak_rss_bytes"] is None and row["mean_nodes"] is None for row in rows)
    curves = analysis.relationship_curves(rows)
    arxiv_sage = [row for row in curves if row["protocol"] == "ogbn-arxiv"
                 and row["method"] == "sparse_sage"]
    assert [row["curve_value"] for row in arxiv_sage if row["curve_parameter"] == "p2"] == [.05, .1, .25, .5, 1.]
    assert [row["curve_value"] for row in arxiv_sage if row["curve_parameter"] == "K_out"] == [5, 10, 20, 40]


def test_historical_summary_resolves_moved_absolute_outputs_without_state(tmp_path):
    indexed = complete_ofat(tmp_path)
    for row in indexed:
        relative = Path(row.pop("output_dir")).relative_to(tmp_path)
        row["result_csv"] = str(Path("/previous/location/experiment") / relative / "result.csv")
    (tmp_path / "results.csv").unlink()
    write_csv(tmp_path / "summary.csv", indexed)
    rows, _ = analysis.read_results(tmp_path, "ofat")
    assert all(Path(row["actual_output_path"]).is_relative_to(tmp_path) for row in rows)
    assert rows[0]["test_metric"] == 0.


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "split", "metric"])
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
        field = "domain_split_id" if mutation == "split" else "metric"
        result[field] = config[field] = "conflicting"
        save_output(directory, result, config)
    write_csv(tmp_path / "results.csv", indexed)
    with pytest.raises(ValueError):
        analysis.read_results(tmp_path, "ofat")


def test_only_bar_panels_require_bootstrap_intervals(tmp_path):
    indexed, _, _, _ = ordinary_run(tmp_path, radius=2, ci=False)
    row, _ = analysis.read_run(tmp_path, tmp_path / "results.csv", indexed, "ofat")
    assert row["bootstrap_ci_lower"] is None
    indexed, _, _, _ = ordinary_run(tmp_path, 1, method="progap", radius=3, ci=False)
    row, _ = analysis.read_run(tmp_path, tmp_path / "results.csv", indexed, "depth-baselines")
    assert row["r"] == 3 and row["bootstrap_ci_upper"] is None
    indexed, _, _, _ = ordinary_run(tmp_path, 2, radius=1, ci=False)
    with pytest.raises(ValueError, match="bootstrap CI"):
        analysis.read_run(tmp_path, tmp_path / "results.csv", indexed, "ofat")


def test_selected_output_cannot_escape_root(tmp_path):
    root = tmp_path / "input"
    indexed, _, _, _ = ordinary_run(tmp_path / "external-run")
    with pytest.raises(ValueError):
        analysis.read_run(root, root / "results.csv", indexed, "ofat", str(root))
