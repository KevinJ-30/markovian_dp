"""Selection boundaries for the two-stage experiment configuration generator."""
import csv
import json

import pytest

from scripts import make_repeat_config as repeat
from scripts import run_experiments as runner


@pytest.fixture
def study(tmp_path):
    source = {
        "name": "tuning", "gpus": [4, 5],
        "defaults": {"dataset": "ogbn-arxiv", "epochs": 20, "seed": 0,
                     "mlp_hidden": 128, "gnn_hidden": 128},
        "grid": {"batch_size": [256, 1024], "lr": [.01, .001]},
        "runs": [
            {"parameters": {"method": "mlp"}},
            {"parameters": {"method": "progap", "epsilon": 2},
             "grid": {"progap_depth": [1, 3, 5]}},
            {"parameters": {"method": "sparse_gin", "gin_pooling": "mean", "sparse_radius": 1},
             "grid": {"epsilon": [2, 8], "p2": [.1, .5]}},
        ],
    }
    (tmp_path / "tuning.json").write_text(json.dumps(source))
    jobs = runner.expand_runs(source)
    for index, job in enumerate(jobs):
        p = job["parameters"]
        if p["method"] == "progap":
            best_lr = {1: .01, 3: .001, 5: None}[p["progap_depth"]]
            score = .7 if best_lr is None else float(p["lr"] == best_lr and p["batch_size"] == 1024)
        elif p["method"] == "sparse_gin":
            score = float(p["lr"] == .001 and p["p2"] == .5 and p["batch_size"] == 1024)
        else:
            score = float(p["lr"] == .01 and p["batch_size"] == 256)
        output = tmp_path / str(index)
        output.mkdir()
        job.update(status="completed", output_dir=str(output))
        row = {"status": "completed", "method": p["method"], "protocol": p["dataset"],
               "seed": 0, "metric": "accuracy", "validation_metric": score,
               "test_metric": 1 - score}
        (output / "result.json").write_text(json.dumps(row))
        with (output / "result.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    (tmp_path / "state.json").write_text(json.dumps({"jobs": jobs}))
    settings = {"source_config": "tuning.json", "results_dir": ".",
                "output_config": "repeats.json", "name": "repeats",
                "seeds": [1, 2], "select_over": ["lr", "batch_size", "p2"]}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(settings))
    return path, jobs


def test_validation_winners_keep_depths_and_epsilons_separate(study):
    path, _ = study
    output, selected, count = repeat.generate(path)
    jobs = runner.expand_runs(runner.load_config(output))
    assert (selected, count) == (6, 12)
    assert {j["parameters"]["seed"] for j in jobs} == {1, 2}
    parameters = [j["parameters"] for j in jobs if j["parameters"]["seed"] == 1]
    pg = {p["progap_depth"]: (p["lr"], p["batch_size"]) for p in parameters if p["method"] == "progap"}
    assert pg == {1: (.01, 1024), 3: (.001, 1024), 5: (.01, 256)}
    sparse = [p for p in parameters if p["method"] == "sparse_gin"]
    assert {p["epsilon"] for p in sparse} == {2, 8}
    assert all((p["lr"], p["batch_size"], p["p2"]) == (.001, 1024, .5) for p in sparse)
    mlp = next(p for p in parameters if p["method"] == "mlp")
    assert (mlp["lr"], mlp["batch_size"]) == (.01, 256)
    assert "epsilon" not in mlp


@pytest.mark.parametrize("problem", ["incomplete", "different_config", "nonfinite"])
def test_invalid_tuning_results_do_not_publish_repeats(study, problem):
    path, jobs = study
    if problem == "incomplete":
        jobs[0]["status"] = "failed"
    elif problem == "different_config":
        jobs[0]["parameters"]["gnn_hidden"] = 64
    else:
        result = json.loads((path.parent / "0" / "result.json").read_text())
        result["validation_metric"] = float("nan")
        (path.parent / "0" / "result.json").write_text(json.dumps(result))
    (path.parent / "state.json").write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError):
        repeat.generate(path)
    assert not (path.parent / "repeats.json").exists()


def test_additional_seeds_cannot_include_the_tuning_seed(study):
    path, _ = study
    with pytest.raises(ValueError, match="exclude the tuning seed"):
        repeat.generate(path, seeds=[0, 1])
    assert not (path.parent / "repeats.json").exists()
