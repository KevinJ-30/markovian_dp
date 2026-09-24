"""Run the user-approved 10/20-epoch-selected final comparison.

Training sources stay frozen so existing, verified tuning runs remain reusable.
This controller changes only eligibility and the execution deadline, not models,
privacy accounting, optimization, or validation-selected checkpoint semantics.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.sparsegnn_initial_tuning import policy, queue as scheduler
from results.sparsegnn_initial_tuning.search import comparison_cells, comparison_slots
from results.sparsegnn_initial_tuning.study_common import HERE, atomic_json, read_json, sha256, utc_now
from scripts.sparsegnn_final_reports import summarize
from scripts.sparsegnn_final_verification import build_selection, verify_comparison, verify_selection

TIMEOUT_SECONDS = 3600.0


class FinalSweep(scheduler.StudyQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_sources = {
            path: sha256(path) for path in (
                Path(__file__),
                ROOT / "scripts/sparsegnn_final_verification.py",
                ROOT / "scripts/sparsegnn_final_reports.py",
            )
        }

    def assert_frozen(self):
        super().assert_frozen()
        if any(sha256(path) != expected for path, expected in self.execution_sources.items()):
            self.stop.set()
            raise RuntimeError("Final sweep execution or verification source changed")

    def publish(self):
        return summarize(self.root)

    def run(self):
        self.preflight()
        self.require_resolved_phase("smoke")
        selected_path = self.root / "selected.json"
        if selected_path.exists():
            selected = read_json(selected_path)
            evidence = verify_selection(self.root, require_frozen=False)
            if not evidence["accepted"]:
                raise RuntimeError(f"Frozen revised selection failed verification: {evidence['errors']}")
        else:
            selected = build_selection(self.root)
            atomic_json(selected_path, selected)
        if selected["implementation_hash"] != self.revision:
            raise RuntimeError("Selected evidence belongs to a different scientific implementation")
        selected_hash = {"sha256": sha256(selected_path)}
        frozen_path = self.phase_dir / "selected_sha256.json"
        if frozen_path.exists() and read_json(frozen_path) != selected_hash:
            raise RuntimeError("Refusing to change frozen comparison winners")
        atomic_json(frozen_path, selected_hash)
        slots = comparison_slots(selected, self.prepared)
        if len(slots) != 192:
            raise RuntimeError("Final comparison must retain all 192 presentation slots")
        atomic_json(self.root / "comparison_slots.json", slots)
        atomic_json(self.root / "final_execution.json", {
            "timeout_seconds": TIMEOUT_SECONDS,
            "termination_grace_seconds": policy.TERM_GRACE,
            "selection_scope": selected["selection_scope"],
            "selected_sha256": selected_hash["sha256"],
            "controller_sha256": sha256(Path(__file__)),
            "verification_sha256": sha256(ROOT / "scripts/sparsegnn_final_verification.py"),
            "reports_sha256": sha256(ROOT / "scripts/sparsegnn_final_reports.py"),
            "scientific_implementation_hash": self.revision,
            "command": [sys.executable, *sys.argv],
            "started_utc": utc_now(),
        })
        resolved = self.run_batch(comparison_cells(selected, self.prepared))
        verification = verify_comparison(self.root)
        with self.publication_lock:
            self.publish()
        self.event("QUEUE_COMPLETE", status="resolved" if resolved and verification["complete"] else "failed",
                   resolved=resolved and verification["complete"],
                   fully_executed=verification["fully_executed"], errors=verification["errors"])
        return resolved and verification["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    # The original scheduler resolves these globals at launch. Changing only
    # its deadline keeps historical scientific source hashes and keys intact.
    scheduler.HARD_LIMIT = TIMEOUT_SECONDS
    scheduler.HardDeadline = partial(policy.HardDeadline, hard_limit=TIMEOUT_SECONDS)
    with scheduler.file_lock(HERE / "locks/scheduler.lock", nonblocking=True):
        queue = FinalSweep("compare", [int(gpu) for gpu in args.gpus.split(",")])
        def stop(signum, frame):
            queue.stop.set()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if not queue.run():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
