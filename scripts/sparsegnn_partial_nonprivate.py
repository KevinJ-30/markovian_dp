"""Preserve user-cancelled non-private runs without inventing completed metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.sparsegnn_initial_tuning.policy import proc_identity
from results.sparsegnn_initial_tuning.study_common import HERE, atomic_json, read_json, scientific_key, sha256, utc_now

CANCELLED_STATUSES = {"cancelled_partial", "cancelled_unstarted"}
NONPRIVATE = {"mlp", "graphsage"}


def _stopped(launch):
    try:
        identity = proc_identity(launch["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return True
    return identity["start_ticks"] != launch["start_ticks"] or identity["state"] == "Z"


def capture_partial(folder: Path, cell: dict) -> dict:
    import torch

    if cell["method"] not in NONPRIVATE:
        raise ValueError("Partial cancellation is restricted to non-private runs")
    config, launch, exited = (read_json(folder / name) for name in ("config.json", "launch.json", "exit.json"))
    if scientific_key(config) != scientific_key(cell):
        raise ValueError("Partial attempt config differs from requested scientific cell")
    if exited.get("status") != "interrupted" or not exited.get("owned_process_exited") or not _stopped(launch):
        raise ValueError("Partial attempt has not confirmed interrupted owned-process exit")
    history = read_json(folder / "history.json") if (folder / "history.json").exists() else []
    progress = read_json(folder / "progress.json") if (folder / "progress.json").exists() else {}
    complete_epochs = [row for row in history if row.get("epoch_complete")]
    fields = {
        "scientific_key": scientific_key(cell), "status": "cancelled_partial", "accepted": False,
        "completed_epochs": max((row["epoch"] for row in complete_epochs), default=0),
        "requested_epochs": cell["epochs"], "updates_completed": progress.get("updates_completed", 0),
        "roots_total": progress.get("roots_total"), "process_wall_seconds": exited["process_wall_seconds"],
        "validation_metric": None, "test_metric": None, "selected_checkpoint_epoch": None,
        "selected_checkpoint_step": None, "partial_result": True,
        "metric_scope": "Saved validation-selected checkpoint only; no final test evaluation was performed",
        "progress_scope": "Persisted completed epochs and last progress update; in-flight updates may be absent",
    }
    checkpoint_path = folder / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if scientific_key(checkpoint["cell"]) != scientific_key(cell):
            raise ValueError("Saved checkpoint belongs to a different scientific run")
        fields.update(validation_metric=checkpoint["validation"]["score"],
                      selected_checkpoint_epoch=checkpoint["epoch"],
                      selected_checkpoint_step=checkpoint["step"])
    names = ["config.json", "launch.json", "exit.json", "history.json", "progress.json", "checkpoint.pt"]
    fields["artifact_sha256"] = {name: sha256(folder / name) for name in names if (folder / name).exists()}
    return fields


def verify_cancelled(root: Path, cell: dict, disposition: dict) -> list[str]:
    errors = []
    try:
        if cell.get("method") not in NONPRIVATE:
            raise ValueError("User non-private cancellation cannot resolve a private run")
        directive = root / "nonprivate_stop.json"
        if not read_json(directive).get("reason") or disposition.get("cancellation_directive_sha256") != sha256(directive):
            raise ValueError("User cancellation directive is missing or changed")
        status, folder_name = disposition["status"], disposition.get("folder")
        if status == "cancelled_unstarted":
            if folder_name is not None:
                raise ValueError("Unstarted cancellation cannot have a launched attempt")
        elif status == "cancelled_partial":
            folder = Path(folder_name)
            partial = read_json(folder / "partial.json")
            if disposition.get("partial_sha256") != sha256(folder / "partial.json"):
                raise ValueError("Partial outcome seal changed")
            if partial != capture_partial(folder, cell):
                raise ValueError("Partial outcome differs from preserved actual attempt evidence")
        else:
            raise ValueError("Unknown user cancellation status")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    return errors


def cancel_remaining(root: Path = HERE) -> dict:
    root = Path(root)
    registry = root / "phases/compare"
    requests, dispositions = read_json(registry / "requests.json"), read_json(registry / "dispositions.json")
    directive_hash = sha256(root / "nonprivate_stop.json")
    snapshot = registry / "dispositions_before_nonprivate_cancellation.json"
    if not snapshot.exists():
        atomic_json(snapshot, dispositions)
    for cell in requests:
        key = scientific_key(cell)
        prior = dispositions.get(key, {})
        if cell["method"] not in NONPRIVATE:
            if prior.get("status") != "completed":
                raise ValueError("All private runs must be complete before final cancellation reporting")
            continue
        if prior.get("status") == "completed":
            continue
        if prior.get("status") in CANCELLED_STATUSES:
            errors = verify_cancelled(root, cell, prior)
            if errors:
                raise ValueError(errors)
            continue
        folder = Path(prior["folder"]) if prior.get("folder") else None
        if folder is None:
            status, seal = "cancelled_unstarted", None
        else:
            partial = capture_partial(folder, cell)
            atomic_json(folder / "partial.json", partial)
            status, seal = "cancelled_partial", sha256(folder / "partial.json")
        dispositions[key] = {**prior, "status": status, "requested_cell": cell, "resolved_cell": cell,
                             "folder": str(folder) if folder else None, "utc": utc_now(),
                             "reason": "User cancelled remaining non-private training; not a completed 100-epoch run",
                             "cancellation_directive_sha256": directive_hash, "partial_sha256": seal}
    atomic_json(registry / "dispositions.json", dispositions)
    counts = {status: sum(row["status"] == status for row in dispositions.values())
              for status in {row["status"] for row in dispositions.values()}}
    atomic_json(root / "nonprivate_cancellation_summary.json", {"utc": utc_now(), "counts": counts,
                "directive_sha256": directive_hash, "partial_test_policy": "No final test scores fabricated or evaluated"})
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    from results.sparsegnn_initial_tuning.queue import file_lock
    root = parser.parse_args().root
    with file_lock(root / "locks/scheduler.lock", nonblocking=True):
        print(json.dumps(cancel_remaining(root), sort_keys=True))
