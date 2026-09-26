#!/usr/bin/env python
"""Canonical graph ablations and fresh-root sequential execution (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("ogbn-arxiv", "saint-yelp", "twitch-allbut2")
METHODS = ("sparse_sage", "sparse_gin")
BASELINE_METHODS = ("dp_gnn_sage", "dp_gnn_gin", "progap")
EPSILONS = (8,)
SEEDS = (0,)
RADII = (1, 2, 3)
P2_VALUES = (0.05, 0.1, 0.25, 0.5, 1.0)
DEGREE_CAPS = (5, 10, 20, 40)
ANCHOR = {"r": 1, "p2": 0.5, "K_out": 10}
ACCOUNTING_POLICY = (
    "Current SparseGNN calibration uses chi=1 and union_safe=False; these runs "
    "do not establish a union-safe accounting claim. Every configuration is "
    "recalibrated at target epsilon=8."
)
BASELINE_ACCOUNTING_POLICY = (
    "Each depth is calibrated independently at target epsilon=8. DP-GNN uses "
    "radius-dependent bounded-degree sensitivity; ProGAP composes depth NAP "
    "releases and depth+1 native DP-SGD stages. Epsilon is per run, not a "
    "composed guarantee for the study."
)
UNCERTAINTY = (
    "The 95% intervals are stored node-bootstrap intervals (1000 resamples, "
    "bootstrap seed 0) for each validation-selected checkpoint, not uncertainty "
    "across training seeds. Training seed is fixed at 0. No test-based selection."
)


def configurations(study: str = "ofat") -> list[dict[str, Any]]:
    """Return 60 distinct SparseExpand OFAT cells or 27 baseline depth cells."""
    if study == "ofat":
        methods = METHODS
        settings = list(dict.fromkeys(
            [(r, ANCHOR["p2"], ANCHOR["K_out"]) for r in RADII]
            + [(ANCHOR["r"], p2, ANCHOR["K_out"]) for p2 in P2_VALUES]
            + [(ANCHOR["r"], ANCHOR["p2"], cap) for cap in DEGREE_CAPS]
        ))
    elif study == "depth-baselines":
        methods = BASELINE_METHODS
        settings = [(depth, None, None) for depth in RADII]
    else:
        raise ValueError(f"unknown ablation study: {study!r}")
    return [
        {"protocol": protocol, "method": method, "epsilon": epsilon,
         "lr": 0.01, "batch_size": 256, "epochs": 20, "seed": seed,
         "p2": p2, "r": radius, "K_out": cap}
        for protocol in PROTOCOLS
        for method in methods
        for epsilon in EPSILONS
        for radius, p2, cap in settings
        for seed in SEEDS
    ]


def run_relative_path(config: dict[str, Any]) -> str:
    """Return a deterministic, study-independent path unique to a configuration."""
    regime = f"lr{config['lr']:g}_b{config['batch_size']}_e{config['epochs']}"
    if config["method"].startswith("sparse_"):
        regime += f"_r{config['r']}_p2{config['p2']:g}_Kout{config['K_out']}"
    elif config["method"].startswith("dp_gnn_"):
        regime += f"_r{config['r']}"
    elif config["method"] == "progap":
        regime += f"_depth{config['r']}"
    else:
        raise ValueError(f"unknown ablation method: {config['method']!r}")
    return str(Path("runs") / config["protocol"] / config["method"]
               / f"eps{config['epsilon']:g}" / regime / f"seed{config['seed']}")


def worker_command(config: dict[str, Any], run_dir: Path, python: str,
                   device: str) -> list[str]:
    """Build the exact worker argv for either a sequential run or queue attempt."""
    command = [
        str(python), "-u", "-B", str(REPO_ROOT / "scripts/full_matrix_run.py"),
        "--dataset", config["protocol"], "--method", config["method"],
        "--lr", str(config["lr"]), "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]), "--seed", str(config["seed"]),
        "--dropout", "0.5", "--mlp-hidden", "64", "--gnn-hidden", "128",
        "--device", device, "--out-dir", str(run_dir),
        "--bootstrap-resamples", "1000", "--epsilon", str(config["epsilon"]),
    ]
    if config["method"].startswith("sparse_"):
        return command + [
            "--p2", str(config["p2"]), "--sparse-radius", str(config["r"]),
            "--sparse-degree-cap", str(config["K_out"]),
        ]
    if config["method"].startswith("dp_gnn_"):
        return command + ["--dpgnn-radius", str(config["r"])]
    if config["method"] != "progap":
        raise ValueError(f"unknown ablation method: {config['method']!r}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.full_matrix_records import PROGAP_PYTHON

    return command + [
        "--progap-depth", str(config["r"]), "--progap-python", PROGAP_PYTHON,
    ]


def _reject_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"refusing symlink in output path: {component}")


def _fresh_root(value: Path) -> Path:
    value = value.expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    # Do not resolve(): that would hide an occupied or dangling symlink.
    root = Path(os.path.abspath(value))
    _reject_symlinks(root)
    if root.exists():
        raise FileExistsError(f"output root is occupied: {root}; use a fresh OUT_ROOT")
    for parent in root.parents:
        if any((parent / marker).exists() or (parent / marker).is_symlink()
               for marker in ("manifest.json", "campaign_manifest.json",
                              "queue_state.json", "requests.json", "runs")):
            raise ValueError(f"refusing output nested inside an existing campaign: {parent}")
    return root


def source_hashes() -> dict[str, str]:
    paths = [REPO_ROOT / "scripts" / name for name in (
        "full_matrix_run.py", "full_matrix_runtime.py", "full_matrix_queue.py",
        "full_matrix_records.py", "sparse_ablation_grid.py", "sparse_ablation_ofat.sh",
        "sparse_ablation_paper.sh",
    )]
    paths.append(REPO_ROOT / "scripts/sparse_ablation.py")
    paths += [path for source in (REPO_ROOT / "src", REPO_ROOT / "third_party/ProGAP")
              for path in source.rglob("*.py")
              if not any(part.startswith(".") or part == "__pycache__"
                         for part in path.relative_to(source).parts)]
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _snapshot_sources(root: Path, hashes: dict[str, str]) -> None:
    for relative, digest in hashes.items():
        content = (REPO_ROOT / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"source changed while preparing the manifest: {relative}")
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)


def _environment() -> dict[str, Any]:
    # Distribution metadata never imports torch or other training packages.
    installed = sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    )
    return {
        "executable": sys.executable,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "installed": installed,
        "environment": {name: os.environ.get(name) for name in (
            "PYTHON", "DEVICE", "OUT_ROOT", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }


def fixed_parameters(study: str = "ofat") -> dict[str, Any]:
    """Return manifest invariants without assigning sparse-only knobs to baselines."""
    common = {"hidden": 128, "dropout": 0.5, "split_seed": 0,
              "bootstrap_resamples": 1000, "bootstrap_confidence": 0.95,
              "bootstrap_seed": 0}
    if study == "ofat":
        return {**common, "layers": 2, "K_in": 10, "chi": 1, "union_safe": False}
    if study == "depth-baselines":
        return {**common, "dpgnn_max_degree": 5, "progap_max_degree": 5}
    raise ValueError(f"unknown ablation study: {study!r}")


def make_manifest(root: Path, python: str, device: str,
                  study: str = "ofat") -> dict[str, Any]:
    """Snapshot sources in a fresh owned root and return the unpublished manifest.

    The caller owns root creation/locking and must publish manifest.json before
    launching any workers. Existing manifests and source snapshots are refused.
    """
    configs = configurations(study)
    root = Path(os.path.abspath(root))
    _reject_symlinks(root)
    if not root.is_dir():
        raise ValueError(f"manifest root must already be an owned directory: {root}")
    if not Path(python).is_absolute():
        raise ValueError("manifest interpreter must be an absolute path")
    if (root / "manifest.json").exists() or (root / "manifest.json").is_symlink():
        raise FileExistsError(f"manifest already exists: {root / 'manifest.json'}")
    hashes = source_hashes()
    environment = _environment()
    (root / "source_snapshot").mkdir(exist_ok=False)
    _snapshot_sources(root, hashes)
    fixed = fixed_parameters(study)
    return {
        "schema_version": 1, "study": study, "python": str(python), "device": device,
        "source_sha256": hashes,
        "configurations": [{**config, "run_dir": run_relative_path(config)} for config in configs],
        "invocations": [
            {"run_dir": run_relative_path(config),
             "argv": worker_command(config, root / run_relative_path(config), python, device),
             "log": str(Path("logs") / Path(run_relative_path(config)).relative_to("runs")) + ".log"}
            for config in configs
        ],
        "fixed": fixed,
        "accounting_policy": ACCOUNTING_POLICY if study == "ofat" else BASELINE_ACCOUNTING_POLICY,
        "uncertainty": UNCERTAINTY,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot": "source_snapshot",
            "driver_argv": list(getattr(sys, "orig_argv", [sys.executable, *sys.argv])),
            "repository": str(REPO_ROOT), "out_root": str(root),
            **environment,
        },
    }


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    # Each invocation owns one new log. Failures preserve it and all worker output.
    with log.open("xb") as stream:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    return process.returncode


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", default="ofat", choices=("ofat", "depth-baselines"))
    cli.add_argument("--ofat-root", type=Path,
                     help="completed immutable SparseExpand root for baseline comparison")
    cli.add_argument("--out-root", type=Path,
                     help="new output root, relative to the repository if not absolute")
    cli.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    cli.add_argument("--dry-run", action="store_true",
                     help="print exact commands; no writes, training, or training imports")
    args = cli.parse_args(argv)
    if args.study == "depth-baselines" and args.ofat_root is None:
        cli.error("--study depth-baselines requires --ofat-root")
    if args.study == "ofat" and args.ofat_root is not None:
        cli.error("--ofat-root is supported only by --study depth-baselines")
    try:
        root = _fresh_root(args.out_root or Path(
            os.environ.get("OUT_ROOT", f"results/sparse_ablation_{args.study}")))
        analysis = [sys.executable, "-B", str(REPO_ROOT / "scripts/sparse_ablation.py"),
                    "--ofat-root", str(args.ofat_root or root)]
        if args.study == "depth-baselines":
            analysis += ["--depth-root", str(root)]
        if args.dry_run:
            configs = configurations(args.study)
            for config in configs:
                print(shlex.join(worker_command(
                    config, root / run_relative_path(config), sys.executable, args.device)))
            print(shlex.join(analysis))
            print(f"Dry run: {len(configs)} training runs; no experiments executed.")
            return 0

        # Exclusive creation rejects races as well as existing/dangling roots.
        _reject_symlinks(root)
        root.mkdir(parents=True, exist_ok=False)
        manifest = make_manifest(root, sys.executable, args.device, args.study)
        configs = manifest["configurations"]
        invocations = manifest["invocations"]
        hashes = manifest["source_sha256"]
        # Publish the complete planned registry before the first worker starts.
        with (root / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Manifest: {root / 'manifest.json'} ({len(configs)} planned runs)", flush=True)
        for index, invocation in enumerate(invocations, start=1):
            if source_hashes() != hashes:
                raise RuntimeError("source changed after manifest creation; refusing mixed implementations")
            run_dir = root / invocation["run_dir"]
            log = root / invocation["log"]
            _reject_symlinks(run_dir)
            _reject_symlinks(log)
            if run_dir.exists():
                raise FileExistsError(f"run directory is occupied: {run_dir}; no overwrite or retry")
            print(shlex.join(invocation["argv"]), flush=True)
            print(f"[{index}/{len(configs)}] Log: {log}", flush=True)
            returncode = _run(invocation["argv"], log)
            if returncode:
                print(f"Worker exited {returncode}; stopping without retries. Evidence: {log}",
                      file=sys.stderr, flush=True)
                return returncode if returncode > 0 else 128 - returncode
            if source_hashes() != hashes:
                raise RuntimeError("source changed during training; preserved outputs must not be accepted")
        print(f"Completed {len(configs)} training runs. Raw results: {root / 'runs'}", flush=True)
        print(shlex.join(analysis), flush=True)
        return subprocess.run(analysis, cwd=REPO_ROOT, check=False).returncode
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{cli.prog}: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; existing outputs are preserved. No automatic resume or retries.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
