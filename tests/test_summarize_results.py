"""Runner CSV aggregation and scientifically honest uncertainty."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_results.py"


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**changes):
    parameters = changes.pop("parameters", {})
    row = {
        "dataset": "fixture", "protocol": "fixture", "method": "sparse_sage",
        "metric": "accuracy", "dp": True, "target_epsilon": 1,
        "target_delta": 1e-5, "seed": 0, "lr": .01, "batch_size": 256,
        "epochs": 20, "hidden": 128, "dropout": .5, "weight_decay": 0,
        "split": "native:seed0", "domain_split": "", "domain_split_id": "",
        "test_metric": .5, "validation_metric": "", "status": "completed",
        **changes,
    }
    row["parameters"] = json.dumps({
        **{key: row[key] for key in ("lr", "batch_size", "epochs", "hidden", "dropout", "weight_decay")},
        "seed": row["seed"], "cap_seed": 20000 + row["seed"], **parameters,
    })
    return row


def _ci(lower, upper, metric="accuracy"):
    return json.dumps({"confidence_level": .95, "metrics": {
        metric: {"lower": lower, "upper": upper, "valid_resamples": 1000},
    }})


def _invoke(tmp_path, *arguments, name="summary"):
    prefix = tmp_path / name
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments), "--out", str(prefix)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    return result, prefix


def _summary(tmp_path, *arguments, name="summary"):
    result, prefix = _invoke(tmp_path, *arguments, name=name)
    assert result.returncode == 0, result.stderr
    with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, prefix.with_suffix(".md").read_text(encoding="utf-8"), result.stderr


def _missing(value):
    return value.strip().lower() in {"", "n/a", "na", "none", "null"}


def test_asymmetric_bootstrap_preserves_point_endpoints_and_primary_metric(tmp_path):
    point, lower, upper = .27288, .23627287853577372, .31114808652246256
    source = _csv(tmp_path / "results.csv", [_row(
        protocol="fixture-protocol", dataset="fixture-dataset", metric="r2",
        test_metric=point, test_confidence_intervals=_ci(lower, upper, metric="r2"),
    )])
    rows, _, _ = _summary(tmp_path, source, "--bootstrap")
    assert len(rows) == 1
    row = rows[0]
    assert (row["dataset"], row["method"], row["metric"]) == (
        "fixture-protocol", "SparseGNN-SAGE", "r2",
    )
    assert float(row["value"]) == pytest.approx(point)
    assert float(row["ci_lower"]) == lower
    assert float(row["ci_upper"]) == upper
    assert float(row["confidence_level"]) == .95
    assert float(row["uncertainty"]) == pytest.approx(upper - point)


def test_bootstrap_best_keeps_completed_runs_privacy_regimes_and_splits(tmp_path):
    common = {"regime": "first", "domain_split_id": "split-a"}
    source = _csv(tmp_path / "runs.csv", [
        _row(**common, status="failed", test_metric=.99),
        _row(**common, test_metric=.4),
        _row(**common, seed=1, lr=.02, test_metric=.6,
             test_confidence_intervals=_ci(.55, .65)),
        _row(**common, target_epsilon=2, test_metric=.5),
        _row(**common, method="mlp", target_epsilon="", dp=False, test_metric=.7),
        _row(**{**common, "domain_split_id": "split-b"}, test_metric=.8),
        _row(**{**common, "regime": "second"}, test_metric=.3),
    ])
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"groups": [
        {"name": name, "files": [source.name], "where": {"regime": name}}
        for name in ("first", "second")
    ]}), encoding="utf-8")
    rows, markdown, stderr = _summary(tmp_path, "--groups", groups, "--bootstrap", "--best")
    assert sorted(float(row["value"]) for row in rows) == [.3, .5, .6, .7, .8]
    assert {row["selection"] for row in rows} == {"best_test"}
    assert [float(row["value"]) for row in rows if row["group"] == "second"] == [.3]
    winner = next(row for row in rows if float(row["value"]) == .6)
    assert float(winner["epsilon"]) == 1
    assert float(winner["ci_lower"]) == .55
    assert float(winner["ci_upper"]) == .65
    other_private = next(row for row in rows if float(row["value"]) == .5)
    nonprivate = next(row for row in rows if float(row["value"]) == .7)
    assert float(other_private["epsilon"]) == 2
    assert _missing(other_private["uncertainty"])
    assert _missing(other_private["ci_lower"]) and _missing(other_private["ci_upper"])
    assert nonprivate["epsilon"] == "non-private"
    assert "test" in (markdown + stderr).lower()
    assert "bias" in (markdown + stderr).lower()


def test_seed_sample_sd_deduplicates_and_best_selects_configuration_mean(tmp_path):
    rows = [
        _row(seed=0, lr=.01, test_metric=.95, parameters={"sigma": 3}),
        _row(seed=1, lr=.01, test_metric=.05, parameters={"sigma": 4}),
        _row(seed=0, lr=.02, test_metric=.6, parameters={"sigma": 2}),
        _row(seed=1, lr=.02, test_metric=.8, parameters={"sigma": 6}),
        _row(seed=0, lr=.03, test_metric=.4, parameters={"sigma": 5}),
    ]
    source = _csv(tmp_path / "seeds.csv", rows + [rows[2]])
    duplicate = _csv(tmp_path / "duplicate.csv", rows)
    summaries, _, _ = _summary(tmp_path, source, duplicate, "--seed")
    assert len(summaries) == 3
    by_value = {round(float(row["value"]), 8): row for row in summaries}
    assert set(by_value) == {.4, .5, .7}
    assert int(by_value[.7]["n"]) == 2
    assert float(by_value[.7]["uncertainty"]) == pytest.approx(math.sqrt(.02))
    assert int(by_value[.4]["n"]) == 1
    assert _missing(by_value[.4]["uncertainty"])
    assert "N/A" in by_value[.4]["display"]
    best, _, _ = _summary(tmp_path, source, duplicate, "--seed", "--best", name="best")
    assert len(best) == 1
    assert float(best[0]["value"]) == pytest.approx(.7)
    assert float(best[0]["uncertainty"]) == pytest.approx(math.sqrt(.02))
    assert int(best[0]["n"]) == 2
    assert best[0]["configuration"] == by_value[.7]["configuration"]
    assert best[0]["selection"] == "best_test"


def test_seed_conflicts_are_errors_not_extra_replicates(tmp_path):
    source = _csv(tmp_path / "conflicting.csv", [
        _row(seed=3, test_metric=.4), _row(seed=3, test_metric=.9),
    ])
    result, _ = _invoke(tmp_path, source, "--seed")
    assert result.returncode != 0
    assert "seed" in result.stderr.lower()
    assert "conflict" in result.stderr.lower() or "duplicate" in result.stderr.lower()


def test_groups_resolve_relative_recursive_globs_filter_and_deduplicate(tmp_path):
    directory = tmp_path / "configuration"
    _csv(directory / "data/nested/results.csv", [
        _row(lr="0.0100", batch_size="256.0", seed=0, test_metric=.2),
        _row(lr="0.01", batch_size=256, seed=1, test_metric=.6),
        _row(lr="0.1", batch_size=256, seed=0, test_metric=.99),
    ])
    groups = directory / "groups.json"
    groups.write_text(json.dumps({"groups": [{
        "name": "small learning rate", "files": ["data/**/*.csv", "data/nested/results.csv"],
        "where": {"lr": [.01], "batch_size": 256},
    }]}), encoding="utf-8")
    rows, _, _ = _summary(tmp_path, "--groups", groups, "--bootstrap")
    assert len(rows) == 2  # The overlapping glob must not duplicate bootstrap runs.
    assert sorted(float(row["value"]) for row in rows) == [.2, .6]
    assert {row["group"] for row in rows} == {"small learning rate"}
    seed_rows, _, _ = _summary(tmp_path, "--groups", groups, "--seed", name="seed")
    assert len(seed_rows) == 1
    assert int(seed_rows[0]["n"]) == 2
    assert float(seed_rows[0]["value"]) == pytest.approx(.4)


@pytest.mark.parametrize("where", [{"misspelled_lr": .01}, {"lr": 999}])
def test_invalid_group_filters_are_actionable_errors(tmp_path, where):
    source = _csv(tmp_path / "results.csv", [_row()])
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"groups": [{
        "name": "requested regime", "files": [source.name], "where": where,
    }]}), encoding="utf-8")
    result, _ = _invoke(tmp_path, "--groups", groups, "--bootstrap")
    assert result.returncode != 0
    assert "requested regime" in result.stderr
    assert "misspelled_lr" in result.stderr or "match" in result.stderr.lower()


def test_explicit_metric_never_relabels_primary_score(tmp_path):
    source = _csv(tmp_path / "results.csv", [_row(test_metric=.9)])
    result, _ = _invoke(tmp_path, source, "--bootstrap", "--metric", "auroc")
    assert result.returncode != 0
    assert "no usable final results" in result.stderr

    _csv(source, [_row(test_metric=.9, test_auroc=.7)])
    rows, _, _ = _summary(tmp_path, source, "--bootstrap", "--metric", "auroc")
    assert rows[0]["metric"] == "auroc"
    assert float(rows[0]["value"]) == pytest.approx(.7)


def test_progap_depth_and_stage_count_are_distinct_settings(tmp_path):
    source = _csv(tmp_path / "progap.csv", [
        _row(method="progap", parameters={"depth": 2, "stages": 3}, seed=0, test_metric=.6),
        _row(method="progap", parameters={"depth": 2, "stages": 3}, seed=1, test_metric=.8),
    ])
    rows, _, _ = _summary(tmp_path, source, "--seed")
    assert len(rows) == 1
    assert rows[0]["method"] == "ProGAP"
    assert int(rows[0]["n"]) == 2
    assert float(rows[0]["value"]) == pytest.approx(.7)
    assert float(rows[0]["uncertainty"]) == pytest.approx(math.sqrt(.02))


@pytest.mark.parametrize("validation", [(0.8, 0.6), (0.8, 0.8)])
def test_best_validation_precedes_test_and_breaks_ties_by_run_index(tmp_path, validation):
    source = _csv(tmp_path / "screen.csv", [
        _row(batch_size=1024, validation_metric=validation[1], test_metric=.95, run_index=9),
        _row(batch_size=256, validation_metric=validation[0], test_metric=.4, run_index=2),
    ])
    rows, _, _ = _summary(tmp_path, source, "--bootstrap", "--best-validation")
    assert len(rows) == 1
    assert float(rows[0]["value"]) == .4
    assert float(rows[0]["validation_value"]) == .8
    assert rows[0]["selection"] == "best_validation"
    assert rows[0]["seeds"] == "0" and int(rows[0]["n"]) == 1
    assert _missing(rows[0]["uncertainty"])


def test_validation_mean_uses_exact_test_cohort_and_unique_seeds(tmp_path):
    first = _row(lr=.01, seed=0, test_metric=.2, validation_metric=.6, run_index=8)
    source = _csv(tmp_path / "seeds.csv", [
        first, first,
        _row(lr=.01, seed=1, test_metric=.4, validation_metric=.8, run_index=1),
        _row(lr=.01, seed=2, test_metric="nan", validation_metric=0, run_index=0),
        _row(lr=.02, seed=0, test_metric=.8, validation_metric=.65, run_index=3),
        _row(lr=.02, seed=1, test_metric=.9, validation_metric=.65, run_index=4),
    ])
    rows, _, _ = _summary(tmp_path, source, "--seed", "--best-validation")
    assert len(rows) == 1
    assert float(rows[0]["value"]) == pytest.approx(.3)
    assert float(rows[0]["validation_value"]) == pytest.approx(.7)
    assert float(rows[0]["uncertainty"]) == pytest.approx(math.sqrt(.02))
    assert rows[0]["seeds"] == "0;1"
    assert int(rows[0]["n"]) == 2


def test_validation_seed_tie_uses_lowest_index_in_whole_cohort(tmp_path):
    source = _csv(tmp_path / "tie.csv", [
        _row(lr=.01, seed=0, test_metric=.8, validation_metric=.7, run_index=2),
        _row(lr=.01, seed=1, test_metric=.9, validation_metric=.7, run_index=3),
        _row(lr=.02, seed=0, test_metric=.2, validation_metric=.6, run_index=8),
        _row(lr=.02, seed=1, test_metric=.4, validation_metric=.8, run_index=1),
    ])
    rows, _, _ = _summary(tmp_path, source, "--seed", "--best-validation")
    assert float(rows[0]["value"]) == pytest.approx(.3)
    assert float(rows[0]["validation_value"]) == pytest.approx(.7)


@pytest.mark.parametrize("validation", ["", "nan", "inf", "-inf"])
def test_best_validation_requires_finite_score_for_every_usable_seed(tmp_path, validation):
    source = _csv(tmp_path / "missing.csv", [
        _row(seed=0, validation_metric=.8),
        _row(seed=1, validation_metric=validation),
    ])
    result, prefix = _invoke(tmp_path, source, "--seed", "--best-validation")
    assert result.returncode == 2
    assert not prefix.with_suffix(".csv").exists()


def test_conflicting_validation_for_duplicate_seed_is_rejected(tmp_path):
    source = _csv(tmp_path / "duplicate.csv", [
        _row(validation_metric=.4), _row(validation_metric=.9),
    ])
    result, _ = _invoke(tmp_path, source, "--seed", "--best-validation")
    assert result.returncode == 2


def test_best_selectors_are_mutually_exclusive(tmp_path):
    source = _csv(tmp_path / "source.csv", [_row(validation_metric=.5)])
    result, _ = _invoke(tmp_path, source, "--seed", "--best", "--best-validation")
    assert result.returncode == 2


def test_validation_metric_override_does_not_relabel_primary_selection(tmp_path):
    source = _csv(tmp_path / "metrics.csv", [_row(
        test_auroc=.8, validation_metric=.9,
        selection=json.dumps({"metric": "accuracy", "validation_score": .9}),
    )])
    result, _ = _invoke(tmp_path, source, "--bootstrap", "--best-validation", "--metric", "auroc")
    assert result.returncode == 2


def test_standalone_csv_validation_ties_are_deterministic_without_run_indices(tmp_path):
    entries = [_row(lr=.01, test_metric=.3, validation_metric=.7),
               _row(lr=.02, test_metric=.9, validation_metric=.7)]
    source = _csv(tmp_path / "results.csv", entries)
    forward, _, _ = _summary(tmp_path, source, "--seed", "--best-validation", name="forward")
    _csv(source, list(reversed(entries)))
    backward, _, _ = _summary(tmp_path, source, "--seed", "--best-validation", name="backward")
    assert forward[0]["configuration"] == backward[0]["configuration"]
    assert forward[0]["value"] == backward[0]["value"]
