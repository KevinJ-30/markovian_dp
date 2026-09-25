#!/usr/bin/env python3
"""Prepare, supervise, resume and report the frozen full-matrix campaign.

The scientific registry and acceptance rules live in full_matrix_records; GPU
queries and ownership-checked process lifetime live in full_matrix_runtime.
This controller never adjusts a scientific parameter to make a job fit.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable

# In particular, dry-run must not create local import-cache artifacts.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import full_matrix_records as records
from scripts import full_matrix_runtime as runtime

GIB = 1024 ** 3
TERMINAL = frozenset({"completed", "recovered_committed", "failed", "timeout",
                      "invalid", "ci_unavailable", "ownership_unverified"})
ACCEPTED = frozenset({"completed", "recovered_committed"})
ATTEMPT_NAME = re.compile(r"attempt_([1-9][0-9]*)\Z")
REPORT_SECONDS = 60.0


def _initial_state() -> dict[str, Any]:
    return {"status": "pending", "prompt_ooms": 0, "not_before": 0.0,
            "last_gpu": None, "attempt_number": 0, "reason": None}


def _epoch(outcome: dict, fallback: float) -> float:
    value = outcome.get("finished_epoch")
    if value is not None:
        return float(value)
    value = outcome.get("ended_utc")
    if value:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    return fallback


def round_robin_requests(requests: list[dict], protocols: list[str]) -> list[dict]:
    """Interleave protocols without disturbing each protocol's shell order."""
    groups: dict[str, deque] = {name: deque() for name in protocols}
    for row in sorted(requests, key=lambda row: row["ordinal"]):
        groups.setdefault(row["protocol"], deque()).append(row)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    return ordered


class CampaignQueue:
    """One owning thread per authorized GPU, with atomic request claims.

    ``clock`` provides time()/monotonic(). The optional GPU, runner and resource
    callables support controlled lifecycle tests; their defaults are always the
    real runtime. ``policy`` is an in-memory test override, not a CLI grid knob.
    Construction has no filesystem, GPU-query or training side effects.
    """

    def __init__(self, root: Path, *, device: str = "cuda", gpus: str = "auto",
                 resume: bool = False, policy: dict | None = None,
                 purpose: str = "campaign", clock: Any = time,
                 gpu_resolver: Callable | None = None,
                 sampler_factory: Callable | None = None,
                 gpu_probe: Callable | None = None,
                 runner: Callable | None = None,
                 host_available: Callable | None = None,
                 disk_available: Callable | None = None):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if purpose not in {"campaign", "smoke"}:
            raise ValueError("purpose must be campaign or smoke")
        if purpose == "campaign" and device != "cuda":
            raise ValueError("the training campaign requires CUDA; CPU is smoke-only")
        self.root = Path(root).expanduser().absolute()
        self.device, self.gpus, self.resume, self.purpose = device, gpus, resume, purpose
        self.clock = clock
        self._policy_override = dict(policy or {})
        self.policy = {**runtime.DEFAULT_POLICY, **self._policy_override}
        self._resolve_gpus = gpu_resolver or runtime.resolve_gpus
        self._sampler_factory = sampler_factory or runtime.GpuSampler
        self._gpu_probe = gpu_probe or runtime.gpu_snapshot
        self._runner = runner or runtime.run_process
        self._host_available = host_available or runtime.host_available_bytes
        self._disk_available = disk_available or (lambda path: shutil.disk_usage(path).free)
        self.mutex = threading.RLock()
        self._event_mutex = threading.Lock()
        self._identity_mutex = threading.Lock()
        self.stop = threading.Event()
        self._shutdown = threading.Event()
        self._report_wakeup = threading.Event()
        self._report_stop = threading.Event()
        self._stop_reason: str | None = None
        self.errors: list[str] = []
        self.state: dict[str, dict] = {}
        self.requests: list[dict] = []
        self.by_key: dict[str, dict] = {}
        self.manifest: dict = {}
        self.manifest_sha256 = ""
        self.active: dict[str, dict] = {}
        self.reservations: dict[str, int] = {}
        self.gpu_rows: dict[str, dict] = {}
        self.sampler = None
        self._retry: deque[str] = deque()
        self._availability: dict[str, str] = {}
        self._retry_preference: dict[str, float] = {}
        self._quarantined_leases: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []
        self._report_thread: threading.Thread | None = None
        self._source_thread: threading.Thread | None = None
        self._old_signals: dict[int, Any] = {}
        self._lease_root = REPO_ROOT / "results" / ".full_matrix_gpu_locks"

    def event(self, name: str, **fields: Any) -> None:
        payload = {"event": name, "utc": runtime.utc_now(), **fields}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        with self._event_mutex:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
            print(encoded, end="", flush=True)

    def request_stop(self, reason: str = "interrupted") -> None:
        with self.mutex:
            # An integrity failure must not be downgraded by a later signal.
            if self._stop_reason is None or reason == "source_changed":
                self._stop_reason = reason
            self.stop.set()
        self._report_wakeup.set()

    def _persist_locked(self) -> None:
        runtime.atomic_json(self.root / "queue_state.json", self.state)

    def _lease_path(self, uuid: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", uuid):
            raise ValueError(f"unsafe GPU UUID: {uuid!r}")
        return self._lease_root / f"{uuid}.lock"

    def _quarantine_path(self, uuid: str) -> Path:
        return self._lease_path(uuid).with_suffix(".quarantine.json")

    def _quarantine(self, uuid: str | None, folder: Path, reason: str) -> None:
        if uuid is None:
            return
        self._lease_root.mkdir(parents=True, exist_ok=True)
        path = self._quarantine_path(uuid)
        row = {"gpu_uuid": uuid, "campaign_root": str(self.root),
               "attempt_dir": str(folder), "reason": reason, "utc": runtime.utc_now()}
        # Never erase another scheduler's ownership blocker.
        if path.exists():
            previous = runtime.read_json(path)
            if previous.get("attempt_dir") != str(folder):
                row["previous_blocker"] = previous
        runtime.atomic_json(path, row)

    def _clear_quarantine(self, uuid: str | None, folder: Path) -> None:
        if uuid is None:
            return
        path = self._quarantine_path(uuid)
        if path.exists():
            row = runtime.read_json(path)
            if row.get("campaign_root") == str(self.root) and row.get("attempt_dir") == str(folder):
                if row.get("previous_blocker"):
                    runtime.atomic_json(path, row["previous_blocker"])
                else:
                    path.unlink()

    def _check_identity(self) -> None:
        # Never invoked on a thread currently supervising an active process.
        with self._identity_mutex:
            manifest, requests = records.load_campaign(
                self.root, check_sources=True, check_prepared=False)
            if runtime.sha256(self.root / "campaign_manifest.json") != self.manifest_sha256:
                raise RuntimeError("campaign manifest changed after admission")
            if manifest != self.manifest or requests != self.requests:
                raise RuntimeError("sealed manifest or requests changed after admission")

    def _check_before_launch(self) -> bool:
        try:
            self._check_identity()
        except Exception as error:
            self.errors.append(f"source/integrity check: {error}")
            self.request_stop("source_changed")
            self.event("SOURCE_CHANGED", reason=str(error))
            return False
        return not self.stop.is_set()

    def _load(self) -> None:
        self.manifest, self.requests = records.load_campaign(
            self.root, check_sources=True, check_prepared=True)
        if self.manifest.get("purpose") != self.purpose:
            raise ValueError(f"{self.purpose} queue rejects {self.manifest.get('purpose')!r} manifest")
        if self.manifest.get("device") != self.device:
            raise ValueError("queue device differs from the sealed manifest")
        expected = 336 if self.purpose == "campaign" else (20 if self.device == "cpu" else 10)
        if len(self.requests) != expected or self.manifest.get("expected_requests") != expected:
            raise ValueError(f"{self.purpose} requires exactly {expected} requests")
        self.by_key = {row["request_key"]: row for row in self.requests}
        if len(self.by_key) != len(self.requests):
            raise ValueError("duplicate request keys in sealed registry")
        self.order = [row["request_key"] for row in round_robin_requests(
            self.requests, self.manifest["protocols"])]
        self.manifest_sha256 = runtime.sha256(self.root / "campaign_manifest.json")
        self.policy = {**runtime.DEFAULT_POLICY,
                       **runtime.read_json(self.root / "execution_policy.json"),
                       **self._policy_override}
        for name, value in self.policy.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid execution policy {name}={value!r}")
        for name in ("poll_seconds", "hard_seconds", "oom_burst_attempts"):
            if self.policy[name] <= 0:
                raise ValueError(f"execution policy {name} must be positive")
        if int(self.policy["oom_burst_attempts"]) != self.policy["oom_burst_attempts"]:
            raise ValueError("oom_burst_attempts must be an integer")
        attempt_root = self.root / "attempts"
        if attempt_root.exists() and any(attempt_root.iterdir()) and not self.resume:
            raise ValueError("training attempts already exist; explicit --resume is required")
        self.state = {key: _initial_state() for key in self.by_key}
        if self.resume:
            self._recover()
        with self.mutex:
            self._persist_locked()
        self._retry = deque(key for key in self.order if self.state[key]["status"] == "retry_pending")

    def _attempts(self) -> list[tuple[str, int, Path]]:
        rows = []
        root = self.root / "attempts"
        if not root.exists():
            return rows
        for parent in root.iterdir():
            if parent.is_symlink() or not parent.is_dir() or parent.name not in self.by_key:
                raise ValueError(f"unexpected request evidence path: {parent}")
            for folder in parent.iterdir():
                match = ATTEMPT_NAME.fullmatch(folder.name)
                if folder.is_symlink() or not folder.is_dir() or match is None:
                    raise ValueError(f"unexpected attempt evidence path: {folder}")
                rows.append((parent.name, int(match.group(1)), folder))
        return sorted(rows, key=lambda row: (self.by_key[row[0]]["ordinal"], row[1]))

    def _validate_binding(self, key: str, number: int, folder: Path) -> None:
        binding = runtime.read_json(folder / "request.json")
        expected = {**self.by_key[key], "campaign_manifest_path": str(self.root / "campaign_manifest.json"),
                    "campaign_manifest_sha256": self.manifest_sha256, "attempt_number": number}
        if binding != expected:
            raise ValueError(f"attempt request binding differs from sealed registry: {folder}")

    def _verify(self, request: dict, folder: Path) -> dict:
        try:
            result = records.verify_attempt(self.root, request, folder)
        except Exception as error:
            result = {"accepted": False, "status": "invalid", "reason": str(error),
                      "request_key": request["request_key"], "ci_available": False}
        return result

    def _acceptance(self, request: dict, folder: Path, outcome: dict) -> dict:
        outcome = dict(outcome)
        outcome["accepted"] = False
        if not outcome.get("owned_process_exited"):
            outcome.update(status="ownership_unverified", reason=outcome.get("reason") or
                           "owned process-tree cleanup was not established")
            return outcome
        status = outcome.get("status")
        if status in ACCEPTED:
            if status == "completed" and outcome.get("returncode") != 0:
                outcome.update(status="invalid", reason="normal completion requires OS return code zero")
                return outcome
            verification = self._verify(request, folder)
            outcome["verification"] = verification
            outcome["accepted"] = bool(verification.get("accepted"))
            if not outcome["accepted"]:
                outcome.update(status=verification.get("status", "invalid"),
                               reason=verification.get("reason", "scientific verification failed"))
                if outcome["status"] not in {"invalid", "ci_unavailable"}:
                    outcome["status"] = "invalid"
        return outcome

    def _transition(self, state: dict, outcome: dict, number: int) -> None:
        status = outcome["status"]
        state.update(status=status, attempt_number=number, reason=outcome.get("reason"),
                     last_gpu=outcome.get("gpu_uuid") or state.get("last_gpu"))
        if outcome.get("accepted"):
            state.update(prompt_ooms=0, not_before=0.0)
        elif status == "oom":
            burst = int(state["prompt_ooms"]) + 1
            if burst >= int(self.policy["oom_burst_attempts"]):
                state.update(status="retry_deferred", prompt_ooms=0,
                             not_before=_epoch(outcome, self.clock.time()) + self.policy["oom_backoff_seconds"])
            else:
                state.update(status="retry_pending", prompt_ooms=burst, not_before=0.0)
        elif status != "interrupted":
            state["not_before"] = 0.0

    def _recover(self) -> None:
        accepted_keys: set[str] = set()
        for key, number, folder in self._attempts():
            self._validate_binding(key, number, folder)
            exit_path = folder / "exit.json"
            outcome = runtime.read_json(exit_path) if exit_path.exists() else None
            recovered = outcome is None or not outcome.get("owned_process_exited", False)
            if recovered:
                if (folder / "launch.json").exists():
                    previous = outcome
                    outcome = runtime.recover_process(
                        folder, grace=self.policy["termination_grace_seconds"])
                    if previous is not None:
                        outcome["previous_exit"] = previous
                else:
                    if (folder / "process.log").exists() or (folder / "output").exists():
                        admission_path = folder / "admission.json"
                        admission = runtime.read_json(admission_path) if admission_path.exists() else {}
                        gpu = admission.get("gpu") or {}
                        outcome = {"status": "ownership_unverified", "returncode": None,
                                   "owned_process_exited": False, "gpu_uuid": gpu.get("uuid"),
                                   "reason": "spawn evidence exists without process ownership identity"}
                    else:
                        outcome = {"status": "interrupted", "returncode": None,
                                   "owned_process_exited": True, "ended_utc": runtime.utc_now(),
                                   "reason": "controller exited before spawning this attempt"}
            outcome = self._acceptance(self.by_key[key], folder, outcome)
            if key in accepted_keys:
                raise ValueError(f"attempts exist after an accepted request: {key}")
            if outcome.get("accepted"):
                accepted_keys.add(key)
            uuid = outcome.get("gpu_uuid")
            if not outcome.get("owned_process_exited"):
                self._quarantine(uuid, folder, outcome["reason"])
                self.reservations[key] = max(self._estimated_host(self.by_key[key]),
                                             int(outcome.get("peak_rss_bytes") or 0))
                if self.purpose == "smoke":
                    self.request_stop("ownership_unverified")
            else:
                self._clear_quarantine(uuid, folder)
                self._cleanup_temporary(folder)
            self._transition(self.state[key], outcome, number)
            outcome["retry_state"] = dict(self.state[key])
            runtime.atomic_json(exit_path, outcome)
            if recovered:
                self.event("ATTEMPT_RECOVERED", request_key=key, attempt_number=number,
                           status=outcome["status"], reason=outcome.get("reason"))

    def _estimated_host(self, request: dict) -> int:
        prepared = self.manifest["prepared"][request["protocol"]]
        return max(2 * GIB, 4 * int(prepared["tensor_bytes"]), int(prepared["preflight_rss_bytes"]))

    def _resource_snapshot(self, request: dict) -> dict:
        available = int(self._host_available())
        disk = int(self._disk_available(self.root))
        required = self._estimated_host(request)
        reserved = sum(self.reservations.values())
        return {"host_available_bytes": available, "disk_free_bytes": disk,
                "reserved_host_bytes": reserved, "required_host_bytes": required,
                "host_reserve_bytes": self.policy["host_reserve_bytes"],
                "disk_reserve_bytes": self.policy["disk_reserve_bytes"],
                "admissible": available - reserved - required >= self.policy["host_reserve_bytes"]
                and disk >= self.policy["disk_reserve_bytes"]}

    def _stable_idle(self, uuid: str) -> bool:
        snapshot = self.sampler.snapshot(uuid) if self.sampler is not None else None
        if not snapshot or snapshot.get("error") or not snapshot.get("idle"):
            return False
        if snapshot.get("consecutive_idle", 0) < 2:
            return False
        observed = snapshot.get("observed_monotonic")
        return observed is not None and self.clock.monotonic() - float(observed) <= max(
            15.0, 3 * self.policy["poll_seconds"])

    def _other_idle(self, uuid: str) -> bool:
        busy = {row["gpu_uuid"] for row in self.active.values()}
        return any(other != uuid and other not in busy and self._stable_idle(other)
                   and not self._quarantine_path(other).exists() for other in self.gpu_rows)

    def _ready_key(self, uuid: str | None) -> str | None:
        now = self.clock.time()
        ordered = list(self._retry) + self.order
        seen = set()
        for key in ordered:
            if key in seen or key in self.active:
                continue
            seen.add(key)
            state = self.state[key]
            if state["status"] in TERMINAL or state["status"] == "running":
                continue
            if float(state["not_before"]) > now:
                continue
            if (uuid is not None and state["status"] == "retry_pending"
                    and state["last_gpu"] == uuid and self._other_idle(uuid)):
                # Give another idle owning thread one polling turn, never bind
                # a retry indefinitely to a device whose lease may be occupied.
                since = self._retry_preference.setdefault(key, self.clock.monotonic())
                if self.clock.monotonic() - since < self.policy["poll_seconds"]:
                    continue
            return key
        return None

    def _allocate_locked(self, request: dict, uuid: str | None, gpu: dict | None,
                         resources: dict) -> dict:
        key = request["request_key"]
        parent = self.root / "attempts" / key
        parent.mkdir(parents=True, exist_ok=True)
        number = self.state[key]["attempt_number"] + 1
        folder = parent / f"attempt_{number}"
        folder.mkdir(exist_ok=False)
        binding = {**request, "campaign_manifest_path": str(self.root / "campaign_manifest.json"),
                   "campaign_manifest_sha256": self.manifest_sha256, "attempt_number": number}
        runtime.atomic_json(folder / "request.json", binding)
        (folder / "tmp").mkdir()
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid or "",
                       "PYTHONPATH": str(REPO_ROOT), "OMP_NUM_THREADS": "2",
                       "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "TMPDIR": str(folder / "tmp")}
        command = records.worker_command(request, folder, device=self.device)
        admission = {"request_key": key, "attempt_number": number,
                     "campaign_manifest_sha256": self.manifest_sha256,
                     "request_sha256": runtime.sha256(folder / "request.json"),
                     "requests_sha256": self.manifest["requests_sha256"],
                     "source_fingerprint": self.manifest["source_fingerprint"],
                     "prepared_fingerprint": request["prepared_fingerprint"],
                     "prepared_manifest_sha256": request["prepared_manifest_sha256"],
                     "gpu": gpu, "resources": resources, "policy": self.policy,
                     "cwd": str(REPO_ROOT), "argv": command,
                     "controlled_environment": {name: environment[name] for name in (
                         "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "TMPDIR")},
                     "admitted_utc": runtime.utc_now(),
                     "reservation_policy": "advisory flock and NVIDIA observation, not exclusive reservation"}
        runtime.atomic_json(folder / "admission.json", admission)
        attempt = {"request": request, "folder": folder, "attempt_number": number,
                   "gpu_uuid": uuid, "gpu": gpu, "command": command, "environment": environment}
        self.reservations[key] = resources["required_host_bytes"]
        self.active[key] = attempt
        self.state[key].update(status="running", attempt_number=number, last_gpu=uuid,
                               not_before=0.0, reason=None)
        self._retry = deque(candidate for candidate in self._retry if candidate != key)
        self._retry_preference.pop(key, None)
        self._persist_locked()
        return attempt

    def _availability_event(self, uuid: str, state: str, **fields: Any) -> None:
        if self._availability.get(uuid) != state:
            self._availability[uuid] = state
            self.event("GPU_AVAILABILITY" if uuid != "cpu" else "CPU_AVAILABILITY",
                       gpu_uuid=None if uuid == "cpu" else uuid, availability=state, **fields)

    def _claim(self, uuid: str | None, gpu: dict | None) -> dict | None:
        with self.mutex:
            if self.stop.is_set() or (self.purpose == "smoke" and self.active):
                return None
            if uuid is not None and any(row["gpu_uuid"] == uuid for row in self.active.values()):
                return None
            key = self._ready_key(uuid)
            if key is None:
                return None
            request = self.by_key[key]
            try:
                resources = self._resource_snapshot(request)
            except (OSError, ValueError, RuntimeError) as error:
                self._availability_event(uuid or "cpu", "resource_query_unavailable", reason=str(error))
                return None
            if not resources["admissible"]:
                self._availability_event(uuid or "cpu", "waiting_resources", resources=resources)
                return None
            return self._allocate_locked(request, uuid, gpu, resources)

    def _sample(self, key: str, sample: dict) -> None:
        rss = int(sample.get("tree_rss_bytes", sample.get("peak_rss_bytes", 0)) or 0)
        with self.mutex:
            if key in self.reservations:
                self.reservations[key] = max(self.reservations[key], rss)

    @staticmethod
    def _cleanup_temporary(folder: Path) -> None:
        # These paths are exclusively attempt-local; no prepared evidence is
        # deleted. A caller must first establish complete owned-tree cleanup.
        temporary = folder / "tmp"
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)

    def _finish(self, attempt: dict, outcome: dict) -> bool:
        request, folder = attempt["request"], attempt["folder"]
        key, number = request["request_key"], attempt["attempt_number"]
        outcome = {**outcome, "gpu_uuid": attempt["gpu_uuid"], "finished_epoch": self.clock.time()}
        outcome = self._acceptance(request, folder, outcome)
        if self._stop_reason == "source_changed" and outcome.get("owned_process_exited"):
            outcome.update(status="invalid", accepted=False, reason="source_changed")
        elif self.stop.is_set() and outcome["status"] == "oom":
            outcome.update(status="interrupted", accepted=False, reason=self._stop_reason)
        cleaned = bool(outcome.get("owned_process_exited"))
        if not cleaned:
            self._quarantine(attempt["gpu_uuid"], folder, outcome["reason"])
            if self.purpose == "smoke":
                self.request_stop("ownership_unverified")
        else:
            self._cleanup_temporary(folder)
        with self.mutex:
            self._transition(self.state[key], outcome, number)
            outcome["retry_state"] = dict(self.state[key])
            runtime.atomic_json(folder / "exit.json", outcome)
            self.active.pop(key, None)
            if cleaned:
                self.reservations.pop(key, None)
            if self.state[key]["status"] == "retry_pending":
                self._retry.appendleft(key)
            self._persist_locked()
        self.event("ATTEMPT_TERMINAL", request_key=key, attempt_number=number,
                   attempt_dir=str(folder), gpu_uuid=attempt["gpu_uuid"],
                   status=outcome["status"], accepted=outcome["accepted"],
                   reason=outcome.get("reason"), retry_state=outcome["retry_state"])
        self._report_wakeup.set()
        return cleaned

    def _execute(self, attempt: dict, lease: Any = None) -> bool:
        self.event("ATTEMPT_STARTED", request_key=attempt["request"]["request_key"],
                   attempt_number=attempt["attempt_number"], attempt_dir=str(attempt["folder"]),
                   gpu_uuid=attempt["gpu_uuid"])
        try:
            outcome = self._runner(
                attempt["command"], attempt["folder"], attempt["environment"], self.stop,
                policy=self.policy, lease_fd=lease.fileno() if lease is not None else None,
                gpu=attempt["gpu"], sampler=self.sampler,
                on_sample=lambda sample: self._sample(attempt["request"]["request_key"], sample))
        except Exception as error:
            self.event("SUPERVISOR_ERROR", attempt_dir=str(attempt["folder"]), reason=str(error))
            try:
                outcome = runtime.recover_process(
                    attempt["folder"], grace=self.policy["termination_grace_seconds"])
                outcome["supervisor_error"] = str(error)
                if outcome.get("owned_process_exited") and outcome["status"] != "recovered_committed":
                    outcome.update(status="failed", reason=f"supervisor exception: {error}")
            except Exception as cleanup_error:
                outcome = {"status": "ownership_unverified", "returncode": None,
                           "owned_process_exited": False,
                           "reason": f"supervisor exception: {error}; recovery: {cleanup_error}"}
        attempt["owned_process_exited"] = bool(outcome.get("owned_process_exited"))
        return self._finish(attempt, outcome)

    def _worker(self, uuid: str | None) -> None:
        try:
            while not self.stop.is_set() and not self._shutdown.is_set():
                with self.mutex:
                    pending = self._ready_key(uuid) is not None
                    allowed = not (self.purpose == "smoke" and self.active)
                if not pending or not allowed:
                    self._shutdown.wait(min(1.0, self.policy["poll_seconds"]))
                    continue
                if uuid is None:
                    if not self._check_before_launch():
                        return
                    attempt = self._claim(None, None)
                    if attempt is not None:
                        self._execute(attempt)
                    else:
                        self._shutdown.wait(self.policy["poll_seconds"])
                    continue
                if self._quarantine_path(uuid).exists():
                    self._availability_event(uuid, "quarantined")
                    self._shutdown.wait(self.policy["poll_seconds"])
                    continue
                if not self._stable_idle(uuid):
                    self._availability_event(uuid, "waiting_two_idle_samples")
                    self._shutdown.wait(self.policy["poll_seconds"])
                    continue
                lease_context = runtime.file_lock(self._lease_path(uuid), nonblocking=True)
                try:
                    lease = lease_context.__enter__()
                except BlockingIOError:
                    self._availability_event(uuid, "cooperative_lease_busy")
                    self._shutdown.wait(self.policy["poll_seconds"])
                    continue
                release_lease = True
                attempt = None
                try:
                    if not self._check_before_launch():
                        return
                    # A fresh, bounded query is the final GPU check before claim
                    # and spawn. Once running, foreign contention never evicts us.
                    try:
                        gpu = self._gpu_probe(uuid)
                    except Exception as error:
                        gpu = {"uuid": uuid, "error": str(error), "idle": False}
                    if not gpu or gpu.get("uuid") != uuid or gpu.get("error"):
                        self._availability_event(uuid, "query_unavailable",
                                                 reason=(gpu or {}).get("error", "missing target GPU"))
                    elif not gpu.get("idle"):
                        self._availability_event(uuid, "busy_at_admission")
                    elif self._quarantine_path(uuid).exists():
                        self._availability_event(uuid, "quarantined")
                    else:
                        attempt = self._claim(uuid, gpu)
                        if attempt is not None:
                            self._availability_event(uuid, "running")
                            release_lease = False
                            release_lease = self._execute(attempt, lease)
                            if not release_lease:
                                self._quarantined_leases[uuid] = lease_context
                                return
                finally:
                    if attempt is not None and attempt.get("owned_process_exited"):
                        release_lease = True
                    if release_lease:
                        lease_context.__exit__(None, None, None)
                    else:
                        self._quarantined_leases[uuid] = lease_context
                        if attempt is not None:
                            self._quarantine(uuid, attempt["folder"],
                                             "owned process cleanup not confirmed by controller")
                self._shutdown.wait(self.policy["poll_seconds"])
        except Exception as error:
            self.errors.append(f"owning thread {uuid or 'cpu'}: {error}")
            try:
                self.event("CONTROLLER_ERROR", gpu_uuid=uuid, reason=str(error))
            finally:
                self.request_stop("controller_error")

    def _source_loop(self) -> None:
        while not self._shutdown.is_set() and not self.stop.is_set():
            if not self._check_before_launch():
                return
            self._shutdown.wait(self.policy["poll_seconds"])

    def _report_loop(self) -> None:
        while True:
            self._report_wakeup.wait(REPORT_SECONDS)
            self._report_wakeup.clear()
            try:
                records.write_reports(self.root)
            except Exception as error:
                self.errors.append(f"report publication: {error}")
                try:
                    self.event("REPORT_ERROR", reason=str(error))
                finally:
                    self.request_stop("report_error")
                return
            if self._report_stop.is_set():
                return

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum, frame):
            # No filesystem or lock acquisition from a Python signal handler.
            if self._stop_reason is None:
                self._stop_reason = "interrupted"
            self.stop.set()
            self._report_wakeup.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signals[signum] = signal.signal(signum, handler)

    def _start(self) -> None:
        if self.device == "cuda":
            rows = self._resolve_gpus(self.gpus)
            self.gpu_rows = {row["uuid"]: row for row in rows}
            if len(self.gpu_rows) != len(rows):
                raise ValueError("GPU resolver returned duplicate UUIDs")
            runtime.atomic_json(self.root / "gpu_inventory.json", {
                "observed_utc": runtime.utc_now(), "allowed_gpus": rows,
                "requested": self.gpus, "authorization_environment": {
                    name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
                "reservation": "advisory flock plus idle observations; unrelated users are never preempted"})
            self.sampler = self._sampler_factory(list(self.gpu_rows), poll_seconds=self.policy["poll_seconds"])
            self.sampler.start()
            if not self.gpu_rows:
                self.event("NO_AUTHORIZED_GPUS", reason="waiting without expanding inherited authorization")
        self._report_thread = threading.Thread(target=self._report_loop, name="campaign-reports", daemon=True)
        self._source_thread = threading.Thread(target=self._source_loop, name="campaign-integrity", daemon=True)
        self._report_thread.start()
        self._source_thread.start()
        self._report_wakeup.set()
        banner = "SMOKE_READY" if self.purpose == "smoke" else "CAMPAIGN_READY"
        print(banner, flush=True)
        self.event("CONTROLLER_READY", purpose=self.purpose, device=self.device,
                   expected_requests=len(self.requests), allowed_gpu_uuids=list(self.gpu_rows),
                   resume=self.resume, policy=self.policy)
        devices = list(self.gpu_rows) if self.device == "cuda" else [None]
        for uuid in devices:
            thread = threading.Thread(target=self._worker, args=(uuid,),
                                      name=f"campaign-{uuid or 'cpu'}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _close(self) -> None:
        self._shutdown.set()
        for thread in self._threads:
            thread.join()
        if self._source_thread is not None:
            self._source_thread.join()
        if self.sampler is not None:
            self.sampler.close()
        if self._report_thread is not None:
            self._report_stop.set()
            self._report_wakeup.set()
            self._report_thread.join()
        for signum, handler in self._old_signals.items():
            signal.signal(signum, handler)
        self._old_signals.clear()

    def run(self) -> int:
        if not self.root.is_dir() or not (self.root / "campaign_manifest.json").is_file():
            raise ValueError("run requires an existing, sealed campaign root")
        with runtime.file_lock(self.root / "campaign.lock", nonblocking=True):
            self._load()
            if self.stop.is_set():
                records.write_reports(self.root)
                return 1
            self._install_signals()
            try:
                self._start()
                heartbeat = self.clock.monotonic()
                while not self.stop.is_set():
                    with self.mutex:
                        finished = not self.active and all(row["status"] in TERMINAL for row in self.state.values())
                    if finished:
                        break
                    if self.clock.monotonic() - heartbeat >= REPORT_SECONDS:
                        with self.mutex:
                            counts = dict(Counter(row["status"] for row in self.state.values()))
                            reservations = dict(self.reservations)
                            running = list(self.active)
                        self.event("HEARTBEAT", counts=counts, active_requests=running,
                                   host_reservations_bytes=reservations)
                        heartbeat = self.clock.monotonic()
                    self.stop.wait(0.2)
            except BaseException:
                self.request_stop("controller_error")
                raise
            finally:
                self._close()
            with self.mutex:
                self._persist_locked()
                accepted = sum(row["status"] in ACCEPTED for row in self.state.values())
            self.event("CONTROLLER_STOPPED", reason=self._stop_reason or "queue_exhausted",
                       accepted_count=accepted, expected_requests=len(self.requests),
                       quarantined_gpus=list(self._quarantined_leases), errors=self.errors)
            if self.errors or self.stop.is_set():
                return 1
            verification = records.verify_campaign(self.root)
            return 0 if verification.get("fully_executed") else 1


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run", help="print the sealed 336-command grid without preparing or querying GPUs")
    prepare = sub.add_parser("prepare", help="prepare and seal a fresh production campaign")
    prepare.add_argument("--out-root", type=Path, required=True)
    internal = sub.add_parser("_prepare-protocol", help=argparse.SUPPRESS)
    internal.add_argument("--out-root", type=Path, required=True)
    internal.add_argument("--dataset", required=True)
    run = sub.add_parser("run", help="supervise a sealed production queue")
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--gpus", default="auto", help="auto or comma-separated authorized UUIDs/indices")
    for name in ("report", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--out-root", type=Path, required=True)
    smoke = sub.add_parser("smoke", help="prepare and execute isolated deterministic worker fixtures")
    smoke.add_argument("--out-root", type=Path, required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda"), required=True)
    smoke.add_argument("--gpus", default="auto")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    try:
        if args.command == "dry-run":
            rows = records.enumerate_grid()
            for row in rows:
                print(shlex.join([str(part) for part in row["argv"]]))
            print(json.dumps({"configurations": len(rows),
                              "per_protocol": dict(Counter(row["protocol"] for row in rows)),
                              "per_method": dict(Counter(row["method"] for row in rows))}, sort_keys=True))
            return 0
        root = args.out_root.expanduser().absolute()
        if args.command == "prepare":
            result = records.prepare_campaign(root, purpose="campaign", device="cuda")
        elif args.command == "_prepare-protocol":
            result = records.prepare_protocol(args.dataset, root)
        elif args.command == "run":
            return CampaignQueue(root, gpus=args.gpus, resume=args.resume).run()
        elif args.command == "smoke":
            records.prepare_campaign(root, purpose="smoke", device=args.device)
            return CampaignQueue(root, device=args.device, gpus=args.gpus, purpose="smoke").run()
        elif args.command == "report":
            result = records.write_reports(root)
        else:
            result = records.verify_campaign(root)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            if result.get("corrupt") or result.get("errors"):
                return 2
            return 0 if result.get("fully_executed") else 1
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"{args.command}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
