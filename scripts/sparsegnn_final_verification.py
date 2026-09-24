"""Verify the approved 10/20-epoch final sweep without changing frozen runners."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from results.sparsegnn_initial_tuning.search import (
    comparison_cells,
    comparison_slots,
    select_winners,
    tuning_cells,
)
from results.sparsegnn_initial_tuning.study_common import (
    EPSILONS,
    PROTOCOL_IDS,
    atomic_json,
    digest_json,
    implementation_hash,
    scientific_key,
    sha256,
    utc_now,
)
from results.sparsegnn_initial_tuning.verification_common import read_object, verify_attempt
from results.sparsegnn_initial_tuning.verify_complete import TERMINAL, failure_evidence

ORIGINAL_IMPLEMENTATION_HASH = "fa61179a549faeeecc0adaca2f564776f0f9db3c87e0da772ce85cf37fec5bc1"
SELECTION_SCOPE = {
    "epochs": [10, 20],
    "requested_candidates_per_pair": 16,
    "requested_total": 384,
    "original_requested_total": 576,
    "excluded_epochs": [25],
    "reason": "User requested final sweep using completed 10- and 20-epoch tuning only",
}
_SELECTION_QUALIFICATION = (
    "Best test-selected configuration in the complete revised 16-point grid of "
    "10- and 20-epoch candidates; all 25-epoch candidates are archived and excluded, "
    "including completed runs. Exploratory test selection; within-run checkpoints "
    "remain validation-selected."
)
_INVALID_EVIDENCE = (OSError, ValueError, TypeError, KeyError, AttributeError)


def _object(path: Path, kind: type = dict):
    value = read_object(path)
    if not isinstance(value, kind):
        raise ValueError(f"{path}: expected {kind.__name__}")
    return value


def _folder(root: Path, value) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _prepared(root: Path) -> dict:
    prepared = _object(root / "prepared.json")
    if prepared.get("implementation_hash") not in (None, ORIGINAL_IMPLEMENTATION_HASH):
        raise ValueError("prepared implementation revision differs from frozen historical revision")
    if implementation_hash() != ORIGINAL_IMPLEMENTATION_HASH:
        raise ValueError("frozen campaign/training source implementation hash changed")
    return {**prepared, "implementation_hash": ORIGINAL_IMPLEMENTATION_HASH}


def _registry(root: Path, phase: str) -> tuple[list, dict, list[str]]:
    directory = root / "phases" / phase
    requests = _object(directory / "requests.json", list)
    dispositions = _object(directory / "dispositions.json")
    keys = []
    for cell in requests:
        if not isinstance(cell, dict):
            raise ValueError("requested scientific cell must be an object")
        key = scientific_key(cell)
        if cell.get("phase") != phase:
            raise ValueError(f"{key}: requested cell has the wrong phase")
        if cell.get("implementation_hash") != ORIGINAL_IMPLEMENTATION_HASH:
            raise ValueError(f"{key}: requested implementation revision changed")
        if cell.get("requested_epochs") != cell.get("epochs"):
            raise ValueError(f"{key}: requested epoch budget differs from resolved epoch budget")
        if cell.get("scientific_key", key) != key:
            raise ValueError(f"{key}: recorded scientific key does not match cell")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate requested scientific keys")
    if set(dispositions) - set(keys):
        raise ValueError("dispositions include unrequested scientific keys")
    return requests, dispositions, keys


def _request_evidence(root: Path, cell: dict, disposition: dict, accepted: dict,
                      hash_cache: dict, *, include_result: bool = False) -> dict:
    """Apply the original phase gates, additionally requiring explicit acceptance."""
    key = scientific_key(cell)
    errors, verification, result, folder = [], None, None, None
    status = "invalid_disposition"
    try:
        if not isinstance(disposition, dict):
            raise ValueError("disposition must be an object")
        status = disposition.get("status", "pending")
        if not isinstance(status, str):
            raise ValueError("disposition status must be a string")
        folder = _folder(root, disposition.get("folder"))
        requested = disposition.get("requested_cell")
        resolved = disposition.get("resolved_cell") or cell
        if not isinstance(requested, dict) or scientific_key(requested) != key:
            errors.append("disposition requested cell does not match registry")
        if not isinstance(resolved, dict) or scientific_key(resolved) != key:
            errors.append("disposition changed requested scientific configuration")
        if status == "completed":
            if folder is None:
                errors.append("completed disposition lacks attempt folder")
            else:
                verification = verify_attempt(folder, cell, root=root, hash_cache=hash_cache)
                errors.extend(verification.get("errors", []))
                if verification.get("accepted") is not True and not verification.get("errors"):
                    errors.append("attempt did not pass original artifact verification")
                entry = accepted.get(key, {})
                if not isinstance(entry, dict):
                    raise ValueError("accepted index entry must be an object")
                if _folder(root, entry.get("folder")) != folder or entry.get("scientific_key") != key:
                    errors.append("completed artifact is missing from exact-key accepted index")
                if not verification.get("artifact_sha256") or entry.get("artifact_sha256") != verification.get("artifact_sha256"):
                    errors.append("accepted index artifact hashes changed or are absent")
                if include_result and not errors:
                    result = _object(folder / "result.json")
        elif status in {"cancelled_partial", "cancelled_unstarted"}:
            from scripts.sparsegnn_partial_nonprivate import verify_cancelled
            errors.extend(verify_cancelled(root, cell, disposition))
        elif status in TERMINAL:
            errors.extend(failure_evidence(disposition, cell, root))
        else:
            errors.append(f"request unresolved: {status}")
    except _INVALID_EVIDENCE as exc:
        errors.append(f"invalid disposition evidence: {exc}")
        if not isinstance(status, str):
            status = "invalid_disposition"
    return {
        "cell": cell, "scientific_key": key, "status": status,
        "accepted": status == "completed" and not errors,
        "folder": str(folder) if folder else None,
        "result": result, "verification": verification,
        "disposition": disposition, "errors": errors,
    }


def _build_selection(root: Path, hash_cache: dict) -> dict:
    prepared = _prepared(root)
    requests, dispositions, keys = _registry(root, "tune")
    expected_keys = {scientific_key(cell) for cell in tuning_cells(prepared)}
    if len(requests) != 576 or set(keys) != expected_keys:
        raise ValueError("original 576-cell tuning registry must remain intact and match the frozen design")
    eligible = [cell for cell in requests if cell.get("epochs") in (10, 20)]
    pairs = Counter((cell["protocol"], cell["epsilon"]) for cell in eligible)
    expected_pairs = {(protocol, epsilon): 16 for protocol in PROTOCOL_IDS for epsilon in EPSILONS}
    if len(eligible) != 384 or pairs != expected_pairs:
        raise ValueError("selection requires exactly 384 eligible requests, 16 per protocol/epsilon pair")
    accepted = _object(root / "accepted.json")
    rows, errors = [], []
    # Deliberately never inspect the dispositions/artifacts of archived 25-epoch runs.
    # Their terminality is not a prerequisite for this explicitly revised boundary.
    for cell in eligible:
        key = scientific_key(cell)
        row = _request_evidence(root, cell, dispositions.get(key, {}), accepted,
                                hash_cache, include_result=True)
        rows.append(row)
        if not row["accepted"]:
            errors.append(f"{key}: eligible candidate is not verified completed ({row['status']})")
        errors.extend(f"{key}: {error}" for error in row["errors"])
    if errors:
        raise ValueError("eligible tuning evidence rejected: " + "; ".join(errors))
    selected = select_winners(rows, eligible)
    winners = selected.get("winners", [])
    if len(winners) != 24:
        raise ValueError("deterministic selection did not produce all 24 winner slots")
    for winner in winners:
        if (winner.get("completed_candidates") != 16 or winner.get("registered_candidates") != 16
                or winner.get("excluded") or not winner.get("cell")
                or winner["cell"].get("epochs") not in (10, 20)):
            raise ValueError("deterministic selection did not accept the full revised 16-point grid")
        # Only denominator/scope annotations change. Original selection and tie order do not.
        winner.update(status="complete_grid", completed_candidates=16, requested_candidates=16,
                      registered_candidates=16, qualification=_SELECTION_QUALIFICATION)
    selected["implementation_hash"] = ORIGINAL_IMPLEMENTATION_HASH
    selected["selection_scope"] = {**SELECTION_SCOPE, "epochs": [10, 20], "excluded_epochs": [25]}
    return selected


def build_selection(root: Path) -> dict:
    """Recompute all 24 winners; raise ValueError rather than select a partial grid.

    This does not write selected.json or any registry. Original verify_attempt may
    refresh attempt verification.json while preserving its accepted artifact seals.
    """
    return _build_selection(Path(root).resolve(), {})


def _verify_selection(root: Path, *, require_frozen: bool, hash_cache: dict) -> dict:
    errors, selected_hash = [], None
    try:
        selected_path = root / "selected.json"
        selected_hash = sha256(selected_path)
        selected = _object(selected_path)
        compare = root / "phases" / "compare"
        frozen_path = compare / "selected_sha256.json"
        comparison_exists = (compare / "requests.json").exists() or (root / "comparison_slots.json").exists()
        if require_frozen or comparison_exists or frozen_path.exists():
            frozen = _object(frozen_path)
            if frozen.get("sha256") != selected_hash:
                errors.append("frozen selected.json hash changed or is absent")
        expected = _build_selection(root, hash_cache)
        for field in ("selection_basis", "selection_policy", "implementation_hash", "selection_scope"):
            if selected.get(field) != expected[field]:
                errors.append(f"selected.json has incorrect {field}")
        winners = selected.get("winners")
        if not isinstance(winners, list) or any(not isinstance(winner, dict) for winner in winners):
            raise ValueError("selected winners must be a list of objects")
        observed = {(winner["protocol"], winner["epsilon"]): winner for winner in winners}
        if len(winners) != 24 or len(observed) != 24:
            errors.append("selected.json must contain exactly 24 unique winner slots")
        for winner in expected["winners"]:
            pair = (winner["protocol"], winner["epsilon"])
            saved = observed.get(pair, {})
            for field, value in winner.items():
                if field == "folder":
                    differs = _folder(root, saved.get(field)) != _folder(root, value)
                else:
                    differs = saved.get(field) != value
                if differs:
                    errors.append(f"frozen winner differs from verified deterministic selection: {pair[0]}/{pair[1]}/{field}")
        if sha256(selected_path) != selected_hash:
            errors.append("selected.json changed during verification")
    except _INVALID_EVIDENCE as exc:
        errors.append(f"cannot verify revised frozen selection: {exc}")
    return {
        "accepted": not errors, "errors": errors, "verified_utc": utc_now(),
        "selected_sha256": selected_hash, "requested": 384,
        "original_requested": 576, "winner_slots": 24,
        "selection_scope": {**SELECTION_SCOPE, "epochs": [10, 20], "excluded_epochs": [25]},
    }


def verify_selection(root: Path, *, require_frozen: bool = True, hash_cache: dict | None = None) -> dict:
    """Independently recompute eligible winners and check saved selection/evidence.

    require_frozen=False permits checking selected.json before comparison is
    registered; once comparison or its hash exists the frozen hash is mandatory.
    """
    return _verify_selection(Path(root).resolve(), require_frozen=require_frozen,
                             hash_cache={} if hash_cache is None else hash_cache)


def _slot_identity(root: Path, slot: dict) -> tuple:
    if not isinstance(slot, dict):
        raise ValueError("comparison slot must be an object")
    cell = slot.get("cell")
    return (
        slot.get("protocol"), slot.get("epsilon_context"), slot.get("method"),
        slot.get("aggregation"), slot.get("status"), slot.get("selected_key"),
        slot.get("selection_status"), slot.get("selection_basis"),
        _folder(root, slot.get("selected_folder")), digest_json(slot.get("selected_grid_values")),
        scientific_key(cell) if cell is not None else None,
        cell.get("phase") if cell is not None else None,
        cell.get("requested_epochs") if cell is not None else None,
    )


def verify_comparison(root: Path, *, hash_cache: dict | None = None) -> dict:
    """Persist comparison coverage, keeping terminal resolution distinct from success."""
    root = Path(root).resolve()
    registry = root / "phases" / "compare"
    hash_cache = {} if hash_cache is None else hash_cache
    design_errors, rows, counts = [], [], Counter()
    requested, presentation_slots, unavailable_slots = 0, None, 0
    requests, dispositions, keys, accepted = [], {}, [], {}
    selection = _verify_selection(root, require_frozen=True, hash_cache=hash_cache)
    design_errors.extend(selection["errors"])
    try:
        requests, dispositions, keys = _registry(root, "compare")
        requested = len(requests)
        prepared = _prepared(root)
        selected = _object(root / "selected.json")
        if sha256(root / "selected.json") != selection["selected_sha256"]:
            design_errors.append("selected.json changed after selection verification")
        expected = comparison_cells(selected, prepared)
        expected_keys = {scientific_key(cell) for cell in expected}
        if set(keys) != expected_keys:
            design_errors.append(
                f"comparison registry differs from frozen design: missing {len(expected_keys-set(keys))}, "
                f"extra {len(set(keys)-expected_keys)}"
            )
        slots = _object(root / "comparison_slots.json", list)
        presentation_slots = len(slots)
        expected_slots = comparison_slots(selected, prepared)
        if len(slots) != 192 or len(expected_slots) != 192:
            design_errors.append("comparison must preserve exactly 192 presentation slots")
        if Counter(_slot_identity(root, slot) for slot in slots) != Counter(
                _slot_identity(root, slot) for slot in expected_slots):
            design_errors.append("comparison presentation slots differ from frozen selected settings/evidence")
        unavailable_slots = sum(slot.get("cell") is None for slot in slots)
        if unavailable_slots:
            design_errors.append("complete revised selection cannot produce unavailable comparison slots")
    except _INVALID_EVIDENCE as exc:
        design_errors.append(f"cannot establish exact frozen comparison coverage: {exc}")
    try:
        accepted = _object(root / "accepted.json")
    except _INVALID_EVIDENCE as exc:
        design_errors.append(f"accepted index unavailable: {exc}")
    errors = list(design_errors)
    for cell, key in zip(requests, keys):
        row = _request_evidence(root, cell, dispositions.get(key, {}), accepted, hash_cache)
        counts[row["status"]] += 1
        if row["errors"]:
            counts["invalid_or_unresolved"] += 1
            errors.extend(f"{key}: {error}" for error in row["errors"])
        rows.append({name: row[name] for name in ("scientific_key", "status", "folder", "errors")})
    try:
        if sha256(root / "selected.json") != selection["selected_sha256"]:
            message = "selected.json changed during comparison verification"
            design_errors.append(message)
            errors.append(message)
    except _INVALID_EVIDENCE as exc:
        message = f"cannot recheck frozen selection hash: {exc}"
        design_errors.append(message)
        errors.append(message)
    payload = {
        "phase": "compare", "verified_utc": utc_now(), "complete": not errors,
        "fully_executed": not errors and counts["completed"] == requested and not unavailable_slots,
        "requested": requested, "presentation_slots": presentation_slots,
        "unavailable_slots": unavailable_slots, "counts": dict(counts), "errors": errors,
        "design_errors": design_errors, "invalid_keys": [row["scientific_key"] for row in rows if row["errors"]],
        "requests": rows, "selection": selection,
        "definition": (
            "complete means every fixed comparison request has a terminal disposition with valid evidence; "
            "fully_executed additionally requires every requested run and all 192 presentation slots to have "
            "verified full-schedule success. Failures, timeouts, blocked requests and user-cancelled partial "
            "or unstarted non-private runs are never successful training. All 384 eligible 10/20-epoch tuning candidates must be verified completed. The original "
            "576-cell tuning ledger is retained, but archived 25-epoch dispositions are excluded from this boundary."
        ),
    }
    atomic_json(registry / "verification.json", payload)
    return payload
