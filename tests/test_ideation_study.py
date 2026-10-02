"""Single-seed selection must not leak final test scores into grid selection."""
import csv

import pytest

from scripts.full_matrix_records import _summarize_evidence


@pytest.mark.parametrize("validation", [(0.8, 0.6), (0.8, 0.8)])
def test_single_seed_screen_selects_validation_and_breaks_ties_by_request(tmp_path, validation):
    accepted, results = {}, []
    for ordinal, (batch, val_score, test_score) in enumerate(zip((256, 1024), validation, (0.4, 0.95))):
        attempt = tmp_path / str(1 - ordinal)
        output = attempt / "output"
        output.mkdir(parents=True)
        source = output / "result.csv"
        row = {
            "protocol": "fb100-gender-1", "dataset": "facebook100-gender",
            "method": "sparse_gin", "architecture": "gin_mean", "metric": "accuracy",
            "dp": True, "target_epsilon": 8, "target_delta": 1 / 4762,
            "seed": 0, "lr": 0.01, "batch_size": batch, "test_acc": test_score,
            "validation_metric": val_score, "split": "domain:seed0",
        }
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        key = str(ordinal)
        accepted[key] = ({"attempt_dir": str(attempt)}, {})
        results.append({"accepted": True, "result_csv": str(source), "ordinal": ordinal,
                        "protocol": row["protocol"], "method": row["method"], "epsilon": 8,
                        "validation_metric": val_score, "bootstrap_resamples": 0})
    rows, _ = _summarize_evidence({"purpose": "ideation", "accepted": accepted, "results": results})
    assert len(rows) == 1
    chosen = rows[0]
    assert chosen["sources"] == results[0]["result_csv"]
    assert chosen["value"] == 0.4
    assert chosen["selection"] == "best_validation"
    assert chosen["n"] == 1 and chosen["seeds"] == "0"
    assert chosen["uncertainty"] == "N/A" and chosen["uncertainty_type"] == "none"
