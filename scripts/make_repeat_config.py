#!/usr/bin/env python3
"""Generate repeat jobs from a complete, single-seed unified-runner study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_experiments as runner


def generate(settings_path: Path, *, results_dir=None, output=None, seeds=None):
    settings_path = Path(settings_path).resolve()
    settings = json.loads(settings_path.read_text())
    base = settings_path.parent
    source = runner.load_config(base / settings["source_config"])
    expected = runner.expand_runs(source)
    root = Path(results_dir) if results_dir is not None else base / settings["results_dir"]
    destination = Path(output) if output is not None else base / settings["output_config"]
    seeds = list(seeds if seeds is not None else settings["seeds"])
    selection_keys = set(settings["select_over"])
    known_keys = set().union(*(runner.scientific_parameters(j["parameters"]) for j in expected))
    if not selection_keys or selection_keys - (known_keys - {"seed"}):
        raise ValueError("select_over must name scientific parameters other than seed")
    tuning_seeds = {j["parameters"]["seed"] for j in expected}
    if len(tuning_seeds) != 1:
        raise ValueError("source_config must contain exactly one tuning seed")
    if any(seed in tuning_seeds for seed in seeds):
        raise ValueError("repeat seeds must exclude the tuning seed")

    state = json.loads((root / "state.json").read_text())
    saved = {job["id"]: job for job in state["jobs"]}
    if len(saved) != len(state["jobs"]) or set(saved) != {j["id"] for j in expected}:
        raise ValueError("results do not match the jobs in source_config")
    complete = sum(job["status"] == "completed" for job in saved.values())
    if complete != len(expected):
        raise ValueError(f"tuning is incomplete: {complete}/{len(expected)} jobs completed")

    winners = {}
    for planned in expected:
        job = saved[planned["id"]]
        parameters = runner.scientific_parameters(job["parameters"])
        if parameters != runner.scientific_parameters(planned["parameters"]):
            raise ValueError(f"{job['id']}: results differ from source_config parameters")
        _, row = runner.read_completed(job)
        group = json.dumps({k: v for k, v in parameters.items()
                            if k not in selection_keys and k != "seed"}, sort_keys=True)
        score = float(row["validation_metric"])
        previous = winners.get(group)
        if previous is not None and row["metric"] != previous[2]:
            raise ValueError(f"{job['id']}: inconsistent validation metrics within a selection group")
        # Exact ties keep the first candidate in source-config expansion order.
        if previous is None or score > previous[0]:
            winners[group] = (score, job, row["metric"])

    blocks = []
    for _, job, _ in winners.values():
        parameters = dict(job["parameters"])
        parameters.pop("seed")
        block = {"parameters": parameters}
        if job.get("resources"):
            block["resources"] = job["resources"]
        blocks.append(block)
    config = {"name": settings["name"], "grid": {"seed": seeds}, "runs": blocks}
    if "gpus" in source:
        config["gpus"] = source["gpus"]
    repeats = runner.expand_runs(config)
    # Never overwrite a previously generated or hand-edited configuration.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return destination, len(winners), len(repeats)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("settings", type=Path, help="Repeat-selection JSON; its paths are settings-relative")
    cli.add_argument("--results-dir", type=Path, help="Override the completed study root (cwd-relative)")
    cli.add_argument("--out", type=Path, help="Override the output JSON (cwd-relative; must not exist)")
    cli.add_argument("--seeds", type=int, nargs="+", help="Override the additional training seeds")
    args = cli.parse_args(argv)
    try:
        path, selected, jobs = generate(args.settings, results_dir=args.results_dir,
                                        output=args.out, seeds=args.seeds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        cli.error(str(error))
    print(f"Selected {selected} validation-best configurations; wrote {jobs} repeat jobs to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
