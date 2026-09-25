"""Bounded resource observations and ownership-safe campaign worker supervision.

Adapted from the archived campaign's queue/policy primitives. GPU observations
and flock leases are advisory: neither reserves a device against other users.
This module deliberately imports neither torch nor the scientific worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator

ROOT = Path(__file__).absolute().parents[1]
MAIN_PYTHON = "/usr/scratch/asaha92/envs/graph_subsampling/bin/python"
GIB = 1024 ** 3
NVIDIA_TIMEOUT = 5.0
DEFAULT_POLICY = {
    "poll_seconds": 5.0,
    "hard_seconds": 3600.0,
    "termination_grace_seconds": 10.0,
    "oom_burst_attempts": 3,
    "oom_backoff_seconds": 300.0,
    "host_reserve_bytes": 20 * GIB,
    "disk_reserve_bytes": 50 * GIB,
}
_CONTROLLED_ENV = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "PYTHONPATH",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "TMPDIR",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: str | Path, value: Any) -> None:
    """Publish a complete JSON artifact, including durable directory metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def file_lock(path: str | Path, *, nonblocking: bool = False, shared: bool = False) -> Iterator[Any]:
    """Yield the locked file; pass ``file.fileno()`` to an inheriting worker.

    Close rather than explicitly LOCK_UN: an inherited descriptor must retain
    the cooperative lease after its controller exits or crashes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if nonblocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(stream.fileno(), flags)
        yield stream


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def host_available_bytes() -> int:
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) < 0:
                    break
                return int(fields[1]) * 1024
    raise RuntimeError("MemAvailable unavailable; refusing unmeasured host admission")


def process_rss(pid: int) -> dict[str, int]:
    """Return measured VmRSS/VmHWM bytes, never substitute zero for unreadable RSS."""
    values = {}
    try:
        with Path(f"/proc/{int(pid)}/status").open() as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM"}:
                    fields = value.split()
                    if len(fields) != 2 or fields[1] != "kB":
                        raise RuntimeError(f"invalid process memory field for PID {pid}")
                    values[key] = int(fields[0]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return values


def _nvidia_query(kind: str, fields: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=NVIDIA_TIMEOUT,
    )
    return [[field.strip() for field in row] for row in csv.reader(io.StringIO(result.stdout))
            if row and any(field.strip() for field in row)]


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _unavailable_gpu(uuid: str, reason: str) -> dict[str, Any]:
    return {
        "uuid": uuid, "index": None, "name": None,
        "memory_total_mib": None, "memory_used_mib": None, "memory_free_mib": None,
        "utilization_gpu": None, "driver_version": None, "compute_processes": [],
        "idle": False, "error": reason, "observed_at": utc_now(),
        "observed_monotonic": time.monotonic(),
    }


def gpu_inventory() -> dict[str, dict[str, Any]]:
    """Query all GPUs and compute PIDs with bounded calls; any uncertainty is busy.

    An unavailable inventory is an empty mapping, not authorization to launch.
    A process-query failure marks every otherwise discovered GPU unavailable.
    """
    try:
        rows = _nvidia_query(
            "gpu", "uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    inventory = {}
    for row in rows:
        if len(row) != 8 or not row[0].startswith("GPU-") or not row[1].isdigit():
            return {}
        uuid, index, name, total, used, free, utilization, driver = row
        if uuid in inventory:
            return {}
        snapshot = {
            "uuid": uuid, "index": int(index), "name": name,
            "memory_total_mib": _number(total), "memory_used_mib": _number(used),
            "memory_free_mib": _number(free), "utilization_gpu": _number(utilization),
            "driver_version": driver, "compute_processes": [], "idle": False,
            "observed_at": utc_now(), "observed_monotonic": time.monotonic(),
        }
        metrics = [snapshot[key] for key in (
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu",
        )]
        if (any(value is None for value in metrics) or not name or not driver
                or snapshot["memory_total_mib"] == 0
                or (snapshot["utilization_gpu"] is not None and snapshot["utilization_gpu"] > 100)):
            snapshot["error"] = "GPU measurements incomplete or invalid"
        inventory[uuid] = snapshot
    try:
        processes = _nvidia_query("compute-apps", "pid,gpu_uuid,used_gpu_memory")
        for row in processes:
            if len(row) != 3 or not row[0].isdigit() or int(row[0]) <= 0 or row[1] not in inventory:
                raise ValueError(f"unresolved compute process observation: {row!r}")
            inventory[row[1]]["compute_processes"].append({
                "pid": int(row[0]), "gpu_uuid": row[1], "used_memory_mib": _number(row[2]),
            })
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        for snapshot in inventory.values():
            snapshot["error"] = f"compute process query unavailable: {error}"
    observed_at, observed_monotonic = utc_now(), time.monotonic()
    for snapshot in inventory.values():
        snapshot.update(observed_at=observed_at, observed_monotonic=observed_monotonic)
        snapshot["idle"] = bool(
            not snapshot.get("error") and not snapshot["compute_processes"]
            and snapshot["memory_used_mib"] <= 1024 and snapshot["utilization_gpu"] <= 5
        )
    return inventory


def gpu_snapshot(uuid: str) -> dict[str, Any]:
    inventory = gpu_inventory()
    # Public callers use UUIDs. Accepting physical indices is useful only before
    # authorization resolution and never changes the inherited visibility mask.
    if str(uuid).isdigit():
        return next((row for row in inventory.values() if row["index"] == int(uuid)),
                    _unavailable_gpu(str(uuid), "GPU missing or inventory query failed"))
    return inventory.get(str(uuid), _unavailable_gpu(str(uuid), "GPU missing or inventory query failed"))


def _resolve_tokens(value: str, inventory: dict[str, dict], *, label: str) -> list[str]:
    if not value.strip() or value.strip().lower() in {"none", "void", "-1"}:
        return []
    result = []
    for token in value.split(","):
        token = token.strip()
        if token.isdigit():
            matches = [uuid for uuid, row in inventory.items() if row["index"] == int(token)]
        elif token.startswith("GPU-"):
            matches = [uuid for uuid in inventory if uuid.startswith(token)]
        else:
            raise RuntimeError(f"cannot resolve {label} GPU token {token!r}")
        if len(matches) != 1 or matches[0] in result:
            raise RuntimeError(f"ambiguous, missing, or duplicate {label} GPU token {token!r}")
        result.append(matches[0])
    return result


def _cuda_visible_uuids() -> list[str]:
    """Resolve logical CUDA order in a child under the original environment.

    The child must have exited before any device is admitted. Interpreter paths
    stay absolute without resolving virtual-environment symlinks.
    """
    code = (
        "import json,torch,uuid\n"
        "rows=[]\n"
        "for index in range(torch.cuda.device_count()):\n"
        " value=torch.cuda.get_device_properties(index).uuid\n"
        " if isinstance(value,bytes): value='GPU-'+str(uuid.UUID(bytes=value))\n"
        " value=str(value)\n"
        " if not value.startswith('GPU-'): value='GPU-'+value\n"
        " rows.append(value)\n"
        "print(json.dumps(rows))\n"
    )
    probe = subprocess.run(
        [MAIN_PYTHON, "-c", code], cwd=ROOT, env=dict(os.environ),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    result = json.loads(probe.stdout)
    if not isinstance(result, list) or not all(isinstance(uuid, str) for uuid in result):
        raise RuntimeError("CUDA visibility probe returned an invalid UUID list")
    return result


def resolve_gpus(spec: str = "auto") -> list[dict[str, Any]]:
    """Intersect requested GPUs with inherited authorization, never expand it."""
    masks = {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
             if key in os.environ}
    # Explicitly empty masks authorize nothing, including when a caller asks for
    # an explicit GPU. Do not initialize CUDA just to discover this condition.
    if any(not value.strip() or value.strip().lower() in {"none", "void", "-1"}
           for value in masks.values()):
        if spec != "auto" and spec.strip():
            raise RuntimeError("inherited GPU visibility explicitly authorizes no devices")
        return []
    inventory = gpu_inventory()
    if not inventory:
        raise RuntimeError("NVIDIA inventory unavailable; GPU authorization cannot be resolved")
    allowed = list(inventory)
    restricted = False
    for name, value in masks.items():
        if name == "NVIDIA_VISIBLE_DEVICES" and value.strip().lower() == "all":
            continue
        if name == "CUDA_VISIBLE_DEVICES" and any(token.strip().isdigit() for token in value.split(",")):
            try:
                selected = _cuda_visible_uuids()
            except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
                raise RuntimeError(f"CUDA numeric visibility cannot be resolved: {error}") from error
            if len(set(selected)) != len(selected) or any(uuid not in inventory for uuid in selected):
                raise RuntimeError("CUDA visibility probe returned missing or duplicate UUIDs")
        else:
            selected = _resolve_tokens(value, inventory, label=name)
        allowed = [uuid for uuid in selected if uuid in allowed]
        restricted = True
    allocation = any(key.startswith(("SLURM_", "PBS_", "LSB_", "LSF_")) for key in os.environ)
    if spec == "auto":
        if allocation and not restricted:
            raise RuntimeError("allocation manager detected without GPU restriction; supply a verified explicit allow-list")
        selected = allowed
    else:
        selected = _resolve_tokens(spec, inventory, label="requested")
        unauthorized = set(selected).difference(allowed)
        if unauthorized:
            raise RuntimeError(f"requested GPUs outside inherited authorization: {sorted(unauthorized)}")
    return [copy.deepcopy(inventory[uuid]) for uuid in selected]


class GpuSampler:
    """One bounded NVIDIA observer, independent of all active job deadlines."""

    def __init__(self, uuids, poll_seconds: float = 5.0):
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and positive")
        self.uuids = tuple(dict.fromkeys(str(uuid) for uuid in uuids))
        self.poll_seconds = float(poll_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshots: dict[str, dict] = {}
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("GPU sampler already started")
        self._thread = threading.Thread(target=self._run, name="matrix-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                inventory = gpu_inventory()
                error = "GPU disappeared or NVIDIA query failed"
            except Exception as exception:
                inventory, error = {}, f"NVIDIA observation failed: {exception}"
            with self._lock:
                for uuid in self.uuids:
                    snapshot = inventory.get(uuid, _unavailable_gpu(uuid, error))
                    previous = self._snapshots.get(uuid, {})
                    recent = started - previous.get("observed_monotonic", -math.inf) <= self.poll_seconds + 2 * NVIDIA_TIMEOUT
                    snapshot["consecutive_idle"] = (
                        previous.get("consecutive_idle", 0) + 1 if snapshot["idle"] and recent
                        else 1 if snapshot["idle"] else 0
                    )
                    self._snapshots[uuid] = snapshot
            self._stop.wait(max(0.01, self.poll_seconds - (time.monotonic() - started)))

    def snapshot(self, uuid: str) -> dict[str, Any] | None:
        with self._lock:
            snapshot = copy.deepcopy(self._snapshots.get(str(uuid)))
        if snapshot is not None and time.monotonic() - snapshot["observed_monotonic"] > self.poll_seconds + 2 * NVIDIA_TIMEOUT:
            snapshot.update(idle=False, consecutive_idle=0, error="stale GPU observation")
        return snapshot

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * NVIDIA_TIMEOUT + 2)
            if self._thread.is_alive():
                raise RuntimeError("GPU sampler failed to stop after bounded NVIDIA queries")


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not value:
        raise RuntimeError("host boot identity unavailable")
    return value


def proc_identity(pid: int) -> dict[str, Any]:
    pid = int(pid)
    if pid <= 0:
        raise ValueError("PID must be positive")
    path = Path(f"/proc/{pid}")
    text = (path / "stat").read_text()
    closing = text.rfind(")")
    if closing < 0:
        raise RuntimeError(f"invalid process stat for PID {pid}")
    fields = text[closing + 2:].split()
    command = (path / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    # Read stat again to avoid combining a recycled PID's argv with old ticks.
    after = (path / "stat").read_text()
    after_fields = after[after.rfind(")") + 2:].split()
    if fields[19] != after_fields[19]:
        raise ProcessLookupError(f"PID {pid} changed while reading its identity")
    return {
        "pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]),
        "pgid": int(fields[2]), "session_id": int(fields[3]), "state": after_fields[0],
        "command": [part.decode(errors="replace") for part in command if part],
    }


def _process_table() -> dict[int, dict]:
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = proc_identity(int(path.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if row["state"] not in {"Z", "X"}:
                rows[row["pid"]] = row
    return rows


def group_members(pgid: int) -> list[dict]:
    return [row for row in _process_table().values() if row["pgid"] == int(pgid)]


class OwnershipError(RuntimeError):
    """An attempt's process tree cannot be proved safe to signal or release."""


def _out_dir(command: list[str]) -> str:
    values = []
    for index, token in enumerate(command):
        if token == "--out-dir":
            if index + 1 >= len(command):
                raise OwnershipError("missing --out-dir value")
            values.append(command[index + 1])
        elif token.startswith("--out-dir="):
            values.append(token.partition("=")[2])
    if len(values) != 1:
        raise OwnershipError("exactly one --out-dir is required for process ownership")
    return str(Path(values[0]).absolute())


def _assert_launch_binding(launch: dict) -> None:
    if launch.get("boot_id") != _boot_id():
        raise OwnershipError("host boot identity mismatch; refusing to signal")
    pid, pgid, session = int(launch["pid"]), int(launch["pgid"]), int(launch["session_id"])
    if pid <= 1 or pgid != pid or session != pid or pgid == os.getpgrp():
        raise OwnershipError("worker is not the recorded isolated session leader")
    expected = str(Path(launch["folder"]).absolute() / "output")
    if _out_dir(launch["command"]) != expected or launch.get("output_dir", expected) != expected:
        raise OwnershipError("attempt output path mismatch; refusing to signal")
    if launch.get("observed_command") not in (launch["command"], []):
        raise OwnershipError("launch observed argv does not match requested worker argv")


def _matches(current: dict, recorded: dict) -> bool:
    return all(current.get(key) == recorded.get(key) for key in (
        "pid", "start_ticks", "pgid", "session_id", "command",
    ))


def _same_lifetime(current: dict, recorded: dict) -> bool:
    return all(current.get(key) == recorded.get(key) for key in (
        "pid", "start_ticks", "pgid", "session_id",
    ))


def validate_owned_process(launch: dict[str, Any]) -> dict[str, Any]:
    _assert_launch_binding(launch)
    current = proc_identity(int(launch["pid"]))
    recorded = {**launch, "command": launch["observed_command"]}
    if (current["state"] in {"Z", "X"} or not _matches(current, recorded)
            or current["command"] != launch["command"]):
        raise OwnershipError("leader PID/start ticks/argv/session mismatch; refusing to signal")
    return current


def _recorded_members(launch: dict) -> dict[int, dict]:
    members = {int(row["pid"]): row for row in launch.get("observed_members", [])}
    members[int(launch["pid"])] = {
        key: launch[key] for key in ("pid", "start_ticks", "pgid", "session_id", "state") if key in launch
    }
    members[int(launch["pid"])]["command"] = launch["observed_command"]
    return members


def _owned_snapshot(launch: dict, recorded: dict[int, dict]) -> dict[int, dict]:
    """Expand only from a currently proved leader or a previously proved child."""
    _assert_launch_binding(launch)
    table = _process_table()
    pid, pgid = int(launch["pid"]), int(launch["pgid"])
    root = table.get(pid)
    if root is not None and (not _matches(root, recorded[pid]) or root["command"] != launch["command"]):
        # A process can clear cmdline while exiting, before stat exposes Z.
        # Re-observe this lifetime rather than treating the stale scan as reuse.
        if not _same_lifetime(root, recorded[pid]):
            raise OwnershipError("leader identity changed or PID was recycled")
        try:
            current = proc_identity(pid)
            # Linux may clear argv before exposing the zombie state. Wait only
            # for that empty-argv transition; never trust or signal it as live.
            settling_deadline = time.monotonic() + 1.0
            while (not current["command"] and current["state"] not in {"Z", "X"}
                   and _same_lifetime(current, recorded[pid]) and time.monotonic() < settling_deadline):
                time.sleep(0.01)
                current = proc_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            table.pop(pid)
        else:
            if not _same_lifetime(current, recorded[pid]):
                raise OwnershipError("leader identity changed or PID was recycled")
            if current["state"] in {"Z", "X"}:
                table.pop(pid)
            elif _matches(current, recorded[pid]) and current["command"] == launch["command"]:
                table[pid] = current
            else:
                raise OwnershipError("leader argv changed while still alive")
    for saved_pid in recorded:
        if saved_pid not in table:
            try:
                current = proc_identity(saved_pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as error:
                raise OwnershipError(f"recorded PID {saved_pid} is unreadable") from error
            if current["state"] not in {"Z", "X"}:
                raise OwnershipError(f"recorded PID {saved_pid} changed during observation")
    owned = {member_pid: row for member_pid, row in table.items()
             if member_pid in recorded and _matches(row, recorded[member_pid])}
    group = [row for row in table.values() if row["pgid"] == pgid]
    if group:
        anchors = [row for row in owned.values() if row["pgid"] == pgid and row["session_id"] == pgid]
        if not anchors or any(row["session_id"] != pgid for row in group):
            raise OwnershipError("remaining process group has no recorded live ownership witness")
        for row in group:
            if row["pid"] in recorded and not _same_lifetime(row, recorded[row["pid"]]):
                raise OwnershipError("recorded group member lifetime changed")
            # A child may be observed between fork and exec (notably ProGAP).
            # A still-verified group witness proves that its new argv belongs
            # to our session; persist the new argv before it can be a witness.
            owned[row["pid"]] = row
    # Capture children which create their own process groups/sessions as well.
    # A causal live PPID connection is needed before they may become witnesses.
    changed = True
    while changed:
        changed = False
        for member_pid, row in table.items():
            if member_pid not in owned and row["ppid"] in owned:
                if row["start_ticks"] < owned[row["ppid"]]["start_ticks"]:
                    raise OwnershipError("descendant precedes its recorded parent")
                owned[member_pid] = row
                changed = True
    recorded.update(owned)
    return owned


def _persist_members(launch: dict, recorded: dict[int, dict]) -> None:
    atomic_json(Path(launch["folder"]) / "process_identities.json", {
        "boot_id": launch["boot_id"], "leader_pid": launch["pid"],
        "observed_at": utc_now(), "members": sorted(recorded.values(), key=lambda row: row["pid"]),
    })


def _signal_owned(launch: dict, recorded: dict[int, dict], signum: int) -> bool:
    owned = _owned_snapshot(launch, recorded)
    if not owned:
        return False
    _persist_members(launch, recorded)
    pgid = int(launch["pgid"])
    anchors = [row for row in owned.values() if row["pgid"] == pgid]
    if anchors:
        # Revalidate the actual witness immediately before killpg. A surviving
        # witness holds this PGID in the recorded session even after leader exit.
        anchor = anchors[0]
        try:
            current = proc_identity(anchor["pid"])
        except (FileNotFoundError, ProcessLookupError):
            return _signal_owned(launch, recorded, signum)
        if current["state"] in {"Z", "X"}:
            return _signal_owned(launch, recorded, signum)
        if not _matches(current, anchor):
            raise OwnershipError("group witness changed before signalling")
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass
    for row in owned.values():
        if row["pgid"] == pgid:
            continue
        try:
            current = proc_identity(row["pid"])
            if current["state"] in {"Z", "X"}:
                continue
            if not _matches(current, row):
                raise OwnershipError("escaped descendant identity changed before signalling")
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                descriptor = os.pidfd_open(row["pid"])
                try:
                    if not _matches(proc_identity(row["pid"]), row):
                        raise OwnershipError("descendant PID recycled before pidfd signalling")
                    signal.pidfd_send_signal(descriptor, signum)
                finally:
                    os.close(descriptor)
            else:
                raise OwnershipError("pidfd signalling unavailable for escaped descendant")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return True


def terminate_owned(process: Any, launch: dict[str, Any], *, grace: float = 10.0) -> tuple[str, int | None]:
    """Terminate our verified tree, including a child surviving an exited leader.

    A missing proof raises OwnershipError. Callers must then quarantine their
    cooperative lease; an unrelated or recycled PID is never signalled.
    ``process=None`` is used for recovery, where an OS exit status is unknown.
    """
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("termination grace must be finite and nonnegative")
    recorded = _recorded_members(launch)
    owned = _owned_snapshot(launch, recorded)
    used_signal = "already_exited"
    if owned:
        _signal_owned(launch, recorded, signal.SIGTERM)
        used_signal = "SIGTERM"
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            if not _owned_snapshot(launch, recorded):
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        if _owned_snapshot(launch, recorded):
            _signal_owned(launch, recorded, signal.SIGKILL)
            used_signal = "SIGKILL"
        deadline = time.monotonic() + max(1.0, min(10.0, grace))
        while _owned_snapshot(launch, recorded):
            if process is not None:
                process.poll()
            if time.monotonic() >= deadline:
                raise OwnershipError("owned descendants survived bounded SIGKILL cleanup")
            time.sleep(0.05)
    _persist_members(launch, recorded)
    returncode = None
    if process is not None:
        try:
            returncode = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as error:
            raise OwnershipError("leader could not be reaped after process-tree cleanup") from error
    return used_signal, returncode


class _AttemptObserver:
    """Move /proc scanning, telemetry I/O and callbacks off the deadline thread."""

    def __init__(self, launch, gpu, sampler, on_sample, poll_seconds):
        self.launch, self.gpu, self.sampler = launch, gpu, sampler
        self.on_sample, self.poll_seconds = on_sample, poll_seconds
        self.recorded = _recorded_members(launch)
        self.peak_rss: int | None = None
        self.peak_gpu: float | None = None
        self.contention = False
        self.error: str | None = None
        self.ownership_error: str | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=f"matrix-observe-{launch['pid']}", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        last_sample, last_persist = -math.inf, -math.inf
        try:
            while not self.stop.is_set():
                with self.lock:
                    owned = _owned_snapshot(self.launch, self.recorded)
                    now = time.monotonic()
                    if now - last_persist >= 1.0:
                        _persist_members(self.launch, self.recorded)
                        last_persist = now
                if now - last_sample >= self.poll_seconds:
                    self._sample(owned, now)
                    last_sample = now
                self.stop.wait(0.2)
        except OwnershipError as error:
            self.ownership_error = str(error)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def _sample(self, owned, now):
        measured, missing = [], []
        for pid in owned:
            values = process_rss(pid)
            if "VmRSS" in values:
                measured.append(values["VmRSS"])
            else:
                missing.append(pid)
        rss = sum(measured) if not missing else None
        if rss is not None:
            self.peak_rss = max(self.peak_rss or 0, rss)
        gpu = self.sampler.snapshot(self.gpu["uuid"]) if self.sampler is not None and self.gpu else None
        gpu_memory, contention_pids = None, []
        if gpu is not None and not gpu.get("error"):
            processes = gpu.get("compute_processes", [])
            ours = [row.get("used_memory_mib") for row in processes if row["pid"] in owned]
            if all(value is not None for value in ours):
                gpu_memory = sum(ours)
                self.peak_gpu = max(self.peak_gpu or 0, gpu_memory)
            contention_pids = [row["pid"] for row in processes if row["pid"] not in owned]
            self.contention |= bool(contention_pids)
        try:
            available = host_available_bytes()
            host_error = None
        except (OSError, RuntimeError, ValueError) as error:
            available, host_error = None, str(error)
        sample = {
            "observed_at": utc_now(), "observed_monotonic": now,
            "elapsed_seconds": now - self.launch["started_monotonic"],
            "tree_rss_bytes": rss, "host_rss_bytes": rss, "peak_rss_bytes": self.peak_rss,
            "rss_unmeasured_pids": missing, "owned_pids": sorted(owned),
            "gpu": gpu, "owned_gpu_memory_mib": gpu_memory,
            "peak_gpu_memory_mib": self.peak_gpu, "contention_pids": contention_pids,
            "host_available_bytes": available, "host_error": host_error,
            "memory_observation": "sampled", "gpu_utilization_scope": "whole_gpu_not_exclusive",
        }
        append_jsonl(Path(self.launch["folder"]) / "resources.jsonl", sample)
        if self.on_sample is not None:
            self.on_sample(sample)

    def close(self, timeout: float = 2.0) -> bool:
        self.stop.set()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()


_CUDA_OOM = re.compile(r"CUDA\s+(?:error:\s*)?out of memory|(?:torch\.cuda\.)?OutOfMemoryError", re.IGNORECASE)
_HOST_OOM = re.compile(
    r"\bMemoryError(?:\s*:|\s*$)|DefaultCPUAllocator[^\n]*(?:not enough memory|can.t allocate memory)"
    r"|CPUAllocator[^\n]*out of memory|std::bad_alloc|Cannot allocate memory",
    re.IGNORECASE,
)


def classify_failure(folder: str | Path, returncode: int | None) -> dict[str, Any]:
    """Only unsuccessful workers can be OOM; SIGKILL alone is not evidence."""
    folder = Path(folder)
    attribution = {"kind": "runtime_error", "source": "exit_status", "returncode": returncode,
                   "cause_attribution": "unknown", "evidence": []}
    if returncode == 0:
        attribution.update(kind=None, source="normal_exit")
        return attribution
    if returncode is None:
        attribution["source"] = "unobserved_exit"
        return attribution
    try:
        error = read_json(folder / "output" / "worker_error.json")
        attribution["worker_error"] = error
        if error.get("kind") in {"cuda_oom", "host_oom"}:
            attribution.update(kind=error["kind"], source="worker_error.json", cause_attribution="observed")
            return attribution
    except FileNotFoundError:
        pass
    except (ValueError, TypeError, AttributeError, OSError) as error:
        attribution["worker_error_read_error"] = str(error)
    try:
        with (folder / "process.log").open("r", encoding="utf-8", errors="replace") as stream:
            overlap = ""
            for chunk in iter(lambda: stream.read(65536), ""):
                text = overlap + chunk
                cuda, host = _CUDA_OOM.search(text), _HOST_OOM.search(text)
                if cuda or host:
                    match = cuda or host
                    attribution.update(
                        kind="cuda_oom" if cuda else "host_oom", source="process.log",
                        cause_attribution="observed", evidence=[text[max(0, match.start() - 120):match.end() + 300]],
                    )
                    return attribution
                overlap = text[-512:]
    except FileNotFoundError:
        pass
    if returncode == 86:
        attribution.update(kind="oom", source="worker_oom_exit_86", cause_attribution="observed")
    return attribution


def _policy(policy: dict | None) -> dict:
    result = {**DEFAULT_POLICY, **(policy or {})}
    for key in ("poll_seconds", "hard_seconds", "termination_grace_seconds"):
        if key == "hard_seconds" and result[key] is None:
            continue
        value = float(result[key])
        if not math.isfinite(value) or value < 0 or (key != "termination_grace_seconds" and value == 0):
            raise ValueError(f"{key} has invalid duration")
        result[key] = value
    return result


def _exit_record(launch, *, status, returncode, owned, reason, observer=None, signal_used=None):
    return {
        "status": status, "returncode": returncode,
        "started_utc": launch["started_utc"], "ended_utc": utc_now(),
        "wall_seconds": max(0.0, time.monotonic() - launch["started_monotonic"]),
        "owned_process_exited": owned,
        "peak_rss_bytes": observer.peak_rss if observer is not None else None,
        "peak_gpu_memory_mib": observer.peak_gpu if observer is not None else None,
        "contention_observed": observer.contention if observer is not None else False,
        "memory_observation": "sampled", "reason": reason,
        "gpu_uuid": launch.get("gpu_uuid"), "signal": signal_used,
        "os_exit_observed": returncode is not None,
        "telemetry_error": observer.error if observer is not None else None,
        "scientific_verification_pending": status == "completed",
    }


def run_process(command, folder, environment, stop, *, policy=None, lease_fd=None,
                gpu=None, sampler=None, on_sample=None) -> dict[str, Any]:
    """Launch one attempt and enforce its wall deadline independently of telemetry.

    The caller has just rechecked GPU idleness while holding its lease. ``stop``
    is a threading.Event. Scientific verification is deliberately the caller's
    responsibility even after a returncode-zero, ownership-clean exit.
    """
    policy = _policy(policy)
    folder = Path(folder).absolute()
    command = [str(value) for value in command]
    if not folder.is_dir():
        raise ValueError("attempt directory must already exist")
    if _out_dir(command) != str(folder / "output"):
        raise ValueError("worker --out-dir must be this attempt's fresh output directory")
    if (folder / "output").exists() or (folder / "launch.json").exists():
        raise FileExistsError("attempt launch/output evidence already exists")
    environment = {str(key): str(value) for key, value in environment.items()}
    tmp = folder / "tmp"
    tmp.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(tmp)
    launch = {
        "command": command, "folder": str(folder), "output_dir": str(folder / "output"),
        "cwd": str(ROOT), "environment": {key: environment[key] for key in _CONTROLLED_ENV if key in environment},
        "boot_id": _boot_id(), "gpu": gpu, "gpu_uuid": gpu.get("uuid") if gpu else None,
        "logical_device": "cuda:0" if gpu else "cpu", "policy": policy,
        "started_utc": utc_now(), "started_at": time.time(), "started_monotonic": time.monotonic(),
        "controller_pid": os.getpid(), "lease_inherited": lease_fd is not None,
    }
    for name in ("admission", "request"):
        path = folder / f"{name}.json"
        if path.exists():
            launch[name] = read_json(path)
            launch[f"{name}_sha256"] = sha256(path)
    if "request" in launch:
        for key in ("request_key", "campaign_manifest_sha256", "source_fingerprint", "prepared_fingerprint", "attempt_number"):
            if key in launch["request"]:
                launch[key] = launch["request"][key]
    process, observer = None, None
    status, reason, signal_used = "failed", "worker not started", None
    cleanup, returncode = True, None
    with (folder / "process.log").open("x", encoding="utf-8") as log:
        if stop.is_set():
            launch["spawned"] = False
            atomic_json(folder / "launch.json", launch)
            record = _exit_record(launch, status="interrupted", returncode=None, owned=True, reason="stop requested before spawn")
            atomic_json(folder / "exit.json", record)
            return record
        try:
            # The deadline begins immediately before Popen, including worker
            # imports, calibration, nested stages, final metrics and publication.
            launch.update(started_monotonic=time.monotonic(), started_at=time.time(), started_utc=utc_now())
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(() if lease_fd is None else (int(lease_fd),)),
            )
            cleanup = False
            identity = proc_identity(process.pid)
            launch.update(identity)
            launch.update(command=command, observed_command=identity["command"], spawned=True,
                          observed_members=[identity])
            # exec can briefly expose an empty /proc cmdline after Popen's
            # close-on-exec handshake. Do not freeze that transitional argv.
            deadline = (launch["started_monotonic"] + policy["hard_seconds"]
                        if policy["hard_seconds"] is not None else None)
            identity_deadline = time.monotonic() + 1.0
            if deadline is not None:
                identity_deadline = min(deadline, identity_deadline)
            while not identity["command"] and identity["state"] not in {"Z", "X"} and process.poll() is None:
                if time.monotonic() >= identity_deadline:
                    raise OwnershipError("initial worker argv remained unavailable")
                time.sleep(0.001)
                observed = proc_identity(process.pid)
                if not _same_lifetime(observed, identity):
                    raise OwnershipError("initial worker lifetime changed")
                identity = observed
            launch.update(identity)
            launch.update(command=command, observed_command=identity["command"],
                          observed_members=[identity])
            atomic_json(folder / "launch.json", launch)
            observer = _AttemptObserver(launch, gpu, sampler, on_sample, policy["poll_seconds"])
            observer.start()
            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    status, reason = "timeout", "hard_runtime_limit"
                    break
                if stop.is_set():
                    status, reason = "interrupted", "controller stop requested"
                    break
                if observer.ownership_error:
                    status, reason = "ownership_unverified", observer.ownership_error
                    break
                if observer.error:
                    status, reason = "failed", f"runtime telemetry failed: {observer.error}"
                    break
                returncode = process.poll()
                if returncode is not None:
                    status = "completed" if returncode == 0 else "failed"
                    reason = "normal OS exit; scientific verification pending" if returncode == 0 else "worker exited nonzero"
                    break
                stop.wait(0.2 if deadline is None else min(0.2, max(0.001, deadline - now)))
        except BaseException as error:
            status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            reason = f"supervisor {type(error).__name__}: {error}"
            if process is None:
                launch.update(spawned=False, spawn_error=reason)
                atomic_json(folder / "launch.json", launch)
        finally:
            observer_closed = True
            if observer is not None:
                observer_closed = observer.close(timeout=10.0 if returncode is not None else 2.0)
                if observer_closed:
                    launch["observed_members"] = list(observer.recorded.values())
            if process is not None:
                try:
                    if "pid" not in launch:
                        raise OwnershipError("worker started but initial process identity could not be captured")
                    if not observer_closed:
                        raise OwnershipError("process observer did not stop; ownership evidence is still changing")
                    signal_used, returncode = terminate_owned(process, launch, grace=policy["termination_grace_seconds"])
                    cleanup = True
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    status, cleanup = "ownership_unverified", False
                    reason = f"{reason}; cleanup unverified: {error}"
                    returncode = process.poll()
            if not cleanup:
                status = "ownership_unverified"
            attribution = classify_failure(folder, returncode)
            if status == "failed" and returncode not in (0, None) and attribution["kind"] in {"oom", "cuda_oom", "host_oom"}:
                status, reason = "oom", f"{attribution['kind']} observed via {attribution['source']}"
            atomic_json(folder / "error_attribution.json", attribution)
            record = _exit_record(
                launch, status=status, returncode=returncode, owned=cleanup,
                reason=reason, observer=observer, signal_used=signal_used,
            )
            record["error_attribution"] = attribution
            atomic_json(folder / "exit.json", record)
    return record


def _committed_marker(folder: Path) -> tuple[bool, str | None]:
    marker_path = folder / "output" / "worker_exit.json"
    if not marker_path.exists():
        return False, "worker success marker absent"
    try:
        marker = read_json(marker_path)
        request = read_json(folder / "request.json")
        if marker.get("status") != "completed":
            raise ValueError("marker status is not completed")
        for key in ("request_key", "campaign_manifest_sha256"):
            if not marker.get(key) or marker[key] != request.get(key):
                raise ValueError(f"marker {key} binding mismatch")
        hashes = marker["artifact_sha256"]
        if set(hashes) != {"config.json", "result.json", "result.csv"}:
            raise ValueError("marker artifact set mismatch")
        for name, expected in hashes.items():
            if sha256(folder / "output" / name) != expected:
                raise ValueError(f"marker artifact hash mismatch: {name}")
        return True, None
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, f"invalid committed result: {error}"


def recover_process(folder: str | Path, *, grace: float = 10.0) -> dict[str, Any]:
    """Clean an interrupted launch without inventing an unobserved OS exit code."""
    folder = Path(folder).absolute()
    launch = read_json(folder / "launch.json")
    if str(Path(launch["folder"]).absolute()) != str(folder):
        raise OwnershipError("recovery launch folder binding mismatch")
    same_boot = launch.get("boot_id") == _boot_id()
    cleanup, signal_used, cleanup_error = False, None, None
    try:
        if not launch.get("spawned", True):
            cleanup = True
        elif not same_boot:
            # A different boot ID alone could be corrupted evidence. Require
            # the recorded launch to predate this boot before declaring that
            # no old process survives, without touching its recycled PID.
            boot_time = None
            with Path("/proc/stat").open() as stream:
                for line in stream:
                    if line.startswith("btime "):
                        boot_time = int(line.split()[1])
                        break
            if (boot_time is None or not launch.get("boot_id")
                    or not isinstance(launch.get("started_at"), (int, float))
                    or not math.isfinite(launch["started_at"])
                    or launch["started_at"] >= boot_time):
                raise OwnershipError("boot mismatch does not establish a previous-boot launch")
            cleanup, signal_used = True, "previous_boot_exited"
        else:
            try:
                evidence = read_json(folder / "process_identities.json")
            except FileNotFoundError:
                evidence = None
            if evidence is not None:
                if evidence.get("boot_id") != launch["boot_id"] or evidence.get("leader_pid") != launch["pid"]:
                    raise OwnershipError("persisted descendant identity binding mismatch")
                launch["observed_members"] = evidence["members"]
            signal_used, _ = terminate_owned(None, launch, grace=grace)
            cleanup = True
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        cleanup_error = str(error)
    committed, marker_error = _committed_marker(folder)
    status = "ownership_unverified" if not cleanup else "recovered_committed" if committed else "interrupted"
    reason = cleanup_error or ("committed worker artifacts recovered; OS status unknown; scientific verification pending"
                               if committed else marker_error)
    peak_rss, peak_gpu, contention = None, None, False
    try:
        with (folder / "resources.jsonl").open() as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except ValueError:
                    continue
                if sample.get("peak_rss_bytes") is not None:
                    peak_rss = max(peak_rss or 0, sample["peak_rss_bytes"])
                if sample.get("peak_gpu_memory_mib") is not None:
                    peak_gpu = max(peak_gpu or 0, sample["peak_gpu_memory_mib"])
                contention |= bool(sample.get("contention_pids"))
    except FileNotFoundError:
        pass
    record = {
        "status": status, "returncode": None, "started_utc": launch.get("started_utc"),
        "ended_utc": utc_now(), "wall_seconds": None,
        "recovery_elapsed_seconds": max(0.0, time.monotonic() - launch["started_monotonic"])
        if same_boot and "started_monotonic" in launch else None,
        "owned_process_exited": cleanup, "peak_rss_bytes": peak_rss,
        "peak_gpu_memory_mib": peak_gpu, "contention_observed": contention,
        "memory_observation": "sampled", "reason": reason, "gpu_uuid": launch.get("gpu_uuid"),
        "signal": signal_used, "os_exit_observed": False, "recovered": True,
        "scientific_verification_pending": status == "recovered_committed",
        "marker_error": marker_error,
    }
    atomic_json(folder / "exit.json", record)
    return record
