"""CLI contracts for final-run selection and scientifically honest uncertainty."""
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
    return {
        "dataset": "fixture", "model": "gnn", "aggr": "mean",
        "metric": "accuracy", "dp": True, "target_epsilon": 1,
        "target_delta": 1e-5, "seed": 0, "lr": .01, "test_acc": .5,
        **changes,
    }


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


def test_asymmetric_bootstrap_preserves_point_endpoints_and_primary_alias(tmp_path):
    point, lower, upper = .27288, .23627287853577372, .31114808652246256
    source = _csv(tmp_path / "results.csv", [_row(
        protocol="fixture-protocol", dataset="fallback", metric="r2",
        target_epsilon="", sigma=5, T=2000, step=2000, test_acc=point,
        selection=json.dumps({"step": 650}),
        test_confidence_intervals=_ci(lower, upper, metric="r2"),
    )])
    rows, markdown, _ = _summary(tmp_path, source, "--bootstrap")
    assert len(rows) == 1
    row = rows[0]
    assert (row["dataset"], row["method"], row["metric"]) == (
        "fixture-protocol", "SparseGNN-SAGE", "r2",
    )
    assert row["epsilon"].lower() == "unknown"
    assert float(row["value"]) == pytest.approx(point)
    assert float(row["ci_lower"]) == lower
    assert float(row["ci_upper"]) == upper
    assert float(row["confidence_level"]) == .95
    assert float(row["uncertainty"]) == pytest.approx(upper - point)
    assert "±" in row["display"]
    assert r"\pm" in markdown
    assert "envelope" in markdown.lower() or "asymmetric" in markdown.lower()


def test_bootstrap_best_uses_final_runs_and_keeps_privacy_regimes_and_splits(tmp_path):
    common = {"regime": "first", "domain_split_id": "split-a", "T": 2}
    source = _csv(tmp_path / "runs.csv", [
        _row(**common, target_epsilon=0, step=1, test_acc=.99),
        _row(**common, target_epsilon=0, step=2, test_acc=.4,
             selection=json.dumps({"step": 1})),
        _row(**common, target_epsilon=0, seed=1, lr=.02, step=2, test_acc=.6,
             selection=json.dumps({"step": 1}), test_confidence_intervals=_ci(.55, .65)),
        _row(**common, target_epsilon=1, step=2, test_acc=.5),
        _row(**common, target_epsilon="", dp=False, epsilon_context=1,
             step=2, test_acc=.7),
        _row(**{**common, "domain_split_id": "split-b"}, target_epsilon=0,
             step=2, test_acc=.8),
        _row(**{**common, "regime": "second"}, target_epsilon=0,
             step=2, test_acc=.3),
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
    assert float(winner["epsilon"]) == 0
    assert float(winner["ci_lower"]) == .55
    assert float(winner["ci_upper"]) == .65
    other_private = next(row for row in rows if float(row["value"]) == .5)
    nonprivate = next(row for row in rows if float(row["value"]) == .7)
    assert float(other_private["epsilon"]) == 1
    assert nonprivate["epsilon"] == "non-private"
    assert "test" in (markdown + stderr).lower()
    assert "bias" in (markdown + stderr).lower()


def test_seed_sample_sd_deduplicates_and_best_selects_configuration_mean(tmp_path):
    rows = [
        _row(seed=0, lr=.01, test_acc=.95, sigma=3),
        _row(seed=1, lr=.01, test_acc=.05, sigma=4),
        _row(seed=0, lr=.02, test_acc=.6, sigma=2),
        _row(seed=1, lr=.02, test_acc=.8, sigma=6),
        _row(seed=0, lr=.03, test_acc=.4, sigma=5),
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
        _row(seed=3, test_acc=.4), _row(seed=3, test_acc=.9),
    ])
    result, _ = _invoke(tmp_path, source, "--seed")
    assert result.returncode != 0
    assert "seed" in result.stderr.lower()
    assert "conflict" in result.stderr.lower() or "duplicate" in result.stderr.lower()


def test_groups_resolve_relative_recursive_globs_filter_and_deduplicate(tmp_path):
    directory = tmp_path / "configuration"
    _csv(directory / "data/nested/results.csv", [
        _row(lr="0.0100", batch_size="256.0", seed=0, test_acc=.2),
        _row(lr="0.01", batch_size=256, seed=1, test_acc=.6),
        _row(lr="0.1", batch_size=256, seed=0, test_acc=.99),
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


def test_missing_ci_and_uncalibrated_private_configs_remain_visible(tmp_path):
    source = _csv(tmp_path / "unknown.csv", [
        _row(target_epsilon="", sigma=3, test_acc=.2),
        _row(target_epsilon="", sigma=5, test_acc=.8),
    ])
    rows, markdown, stderr = _summary(tmp_path, source, "--bootstrap", "--best")
    assert sorted(float(row["value"]) for row in rows) == [.2, .8]
    assert {row["epsilon"].lower() for row in rows} == {"unknown"}
    assert len({row["configuration"] for row in rows}) == 2
    for row in rows:
        assert _missing(row["uncertainty"])
        assert _missing(row["ci_lower"]) and _missing(row["ci_upper"])
        assert "N/A" in row["display"]
    assert "N/A" in markdown
    warnings = stderr.lower()
    assert "epsilon" in warnings
    assert "bootstrap" in warnings or "confidence" in warnings or " ci" in warnings


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


@pytest.mark.parametrize("declared", ["", "accuracy"])
def test_explicit_metric_never_relabels_primary_score(tmp_path, declared):
    source = _csv(tmp_path / "results.csv", [_row(metric=declared, test_acc=.9)])
    result, _ = _invoke(tmp_path, source, "--bootstrap", "--metric", "auroc")
    assert result.returncode != 0
    assert "no usable final results" in result.stderr

    _csv(source, [_row(metric=declared, test_acc=.9, test_auroc=.7)])
    rows, _, _ = _summary(tmp_path, source, "--bootstrap", "--metric", "auroc")
    assert rows[0]["metric"] == "auroc"
    assert float(rows[0]["value"]) == pytest.approx(.7)


def test_progap_depth_and_stage_count_are_distinct_settings(tmp_path):
    source = _csv(tmp_path / "progap.csv", [
        _row(method="progap", family="progap", depth=2, stages=3, seed=0, test_acc=.6),
        _row(method="progap", family="progap", depth=2, stages=3, seed=1, test_acc=.8),
    ])
    rows, _, _ = _summary(tmp_path, source, "--seed")
    assert len(rows) == 1
    assert rows[0]["method"] == "ProGAP"
    assert int(rows[0]["n"]) == 2
    assert float(rows[0]["value"]) == pytest.approx(.7)
    assert float(rows[0]["uncertainty"]) == pytest.approx(math.sqrt(.02))
