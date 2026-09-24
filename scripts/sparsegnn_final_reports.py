"""Evidence-linked final reports for the completed 10/20-epoch selection scope.

The original 576-request ledger and frozen scientific implementation stay intact.
Publication reads, but never writes, selected.json or any experiment artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.sparsegnn_initial_tuning import summarize as original
from results.sparsegnn_initial_tuning.search import comparison_slots, tuning_cells
from results.sparsegnn_initial_tuning.study_common import atomic_json, protocols, scientific_key, utc_now
from results.sparsegnn_initial_tuning.verification_common import finite

HERE = ROOT / "results" / "sparsegnn_initial_tuning"
ELIGIBLE_EPOCHS = (10, 20)
CANDIDATES_PER_PAIR = 16
ORIGINAL_REQUESTED = 576
ELIGIBLE_REQUESTED = 384
WINNER_SLOTS = 24
PRESENTATION_SLOTS = 192
ACTIVE_TIMEOUT_SECONDS = 3600
EPSILONS = original.EPSILONS
PRIVATE_BASELINES = original.PRIVATE_BASELINES
METHOD_LABELS = original.METHOD_LABELS
SELECTION_SCOPE = {
    "epochs": [10, 20],
    "requested_candidates_per_pair": 16,
    "requested_total": 384,
    "original_requested_total": 576,
    "excluded_epochs": [25],
    "reason": "User requested final sweep using completed 10- and 20-epoch tuning only",
}
QUALIFICATIONS = (
    "**Accounting and selection limitations.** SparseGNN uses an **experimental repository "
    "path-bound χ=1 estimate, not an established ordinary-degree union-safe DP guarantee** "
    "(`repository_path_bound_chi1`, `union_safe=false`). DPAR retains its repository/upstream "
    "separate PPR+SGD composition qualification, not an independently established node-level "
    "privacy guarantee. Epsilon/delta describe individual runs, not a composed guarantee for "
    "the revised 16-candidate test-selected sweep, the original 24-candidate requests, repeated "
    "releases, or the campaign. Validation/test and feature preprocessing use the fixed/public "
    "evaluation assumptions. Configuration selection is exploratory and uses test_metric; "
    "checkpoints within each run remain first-strict-maximum validation selected, followed by "
    "one restored-checkpoint test evaluation (ProGAP retains stage-wise validation checkpoints). "
    "Scores are single-seed observations (seed 0), not means across runs or confidence intervals. "
    "Selection favors tuned SparseGNN-SAGE relative to untuned baselines; no significance or "
    "unbiased generalization is claimed. Non-private references run **100 epochs**; private "
    "comparisons inherit **10 or 20 epochs** (ProGAP per stage). Settings match B/learning rate, "
    "not epoch counts or update counts. The active final-comparison wall deadline is **3600 "
    "seconds per new run**, including loading, calibration and evaluation; historical tuning "
    "durations and launch policies are retained, not retroactively changed."
)


def _winner_map(selected: dict) -> dict:
    return {(winner["protocol"], float(winner["epsilon"])): winner
            for winner in selected.get("winners", [])}


def _winner_epochs(winner: dict):
    return (winner.get("cell") or winner.get("grid_values") or {}).get("epochs")


def _complete_selection(winner: dict) -> bool:
    return (winner.get("status") == "complete_grid"
            and all(winner.get(field) == CANDIDATES_PER_PAIR for field in
                    ("completed_candidates", "requested_candidates", "registered_candidates"))
            and _winner_epochs(winner) in ELIGIBLE_EPOCHS)


def rank_datasets(comparison_rows: list[dict], selected: dict) -> list[dict]:
    """Rank unrounded SAGE margins; accepted is supplied by artifact verification.

    Minimal score rows use protocol, epsilon_context, method, aggregation (Sparse),
    test_metric and accepted. Winners require complete_grid, all three candidate
    counts equal to 16, and cell.epochs (or grid_values.epochs) equal to 10 or 20.
    Optional selection_evidence_current=False withholds a rank. GIN/non-private
    scores never determine eligibility or the primary score. All twelve protocols
    remain visible, including those with no usable comparison rows.
    """
    winners = _winner_map(selected)
    winner_counts = Counter((row["protocol"], float(row["epsilon"]))
                            for row in selected.get("winners", []))
    groups = defaultdict(dict)
    duplicates = set()
    for row in comparison_rows:
        pair = row["protocol"], float(row["epsilon_context"])
        method = original.method_id(row)
        if method in groups[pair]:
            duplicates.add(pair)
        groups[pair][method] = row
    scope = selected.get("selection_scope")
    scope_valid = scope is None or scope == SELECTION_SCOPE
    rankings = []
    for protocol in protocols():
        identifier = protocol["id"]
        reasons, contexts = [], []
        if not scope_valid:
            reasons.append("selection scope differs from the revised 384-cell 10/20-epoch scope")
        for epsilon in EPSILONS:
            pair = identifier, epsilon
            group = groups[pair]
            winner = winners.get(pair, {})
            if not _complete_selection(winner) or winner_counts[pair] != 1:
                reasons.append(f"epsilon {epsilon:g}: tuning is not verified 16/16 complete_grid at epochs 10/20")
            if pair in duplicates:
                reasons.append(f"epsilon {epsilon:g}: duplicate comparison rows")

            def score(method):
                row = group.get(method, {})
                value = row.get("test_metric")
                valid = (row.get("accepted") is True
                         and row.get("selection_evidence_current", True) is True
                         and row.get("status") in (None, "completed")
                         and finite(value, minimum=0) and value <= 1)
                if method not in ("mlp", "graphsage") and row.get("epochs") is not None:
                    valid = valid and row["epochs"] in ELIGIBLE_EPOCHS
                if method == "sparse_mean":
                    for field in ("scientific_key", "test_metric"):
                        if field in winner and field in row and winner[field] != row[field]:
                            valid = False
                return value if valid else None

            sage, gin = score("sparse_mean"), score("sparse_gin")
            baseline_scores = {method: score(method) for method in PRIVATE_BASELINES}
            missing = [method for method, value in baseline_scores.items() if value is None]
            if sage is None:
                reasons.append(f"epsilon {epsilon:g}: missing verified SparseGNN-SAGE score")
            if missing:
                reasons.append(f"epsilon {epsilon:g}: missing private baselines: {', '.join(missing)}")
            strongest = max(PRIVATE_BASELINES, key=lambda method: baseline_scores[method]) if not missing else None
            best = baseline_scores[strongest] if strongest else None
            margin = sage - best if sage is not None and best is not None else None
            gin_margin = gin - best if gin is not None and best is not None else None
            mlp, nonprivate_sage = score("mlp"), score("graphsage")
            contexts.append({
                "epsilon": epsilon, "sparse_sage_test": sage, "sparse_gin_test": gin,
                "private_baseline_scores": baseline_scores,
                "strongest_private_baseline": METHOD_LABELS[strongest] if strongest else None,
                "strongest_private_test": best, "margin": margin,
                "margin_percentage_points": 100 * margin if margin is not None else None,
                "gin_margin": gin_margin,
                "gin_margin_percentage_points": 100 * gin_margin if gin_margin is not None else None,
                "nonprivate_mlp_gap": sage - mlp if sage is not None and mlp is not None else None,
                "nonprivate_sage_gap": sage - nonprivate_sage if sage is not None and nonprivate_sage is not None else None,
                "runtime_seconds": {method: row.get("process_wall_seconds") for method, row in group.items()},
                "evidence": {method: {"folder": row.get("folder"), "scientific_key": row.get("scientific_key"),
                                      "accepted": row.get("accepted")} for method, row in group.items()},
            })
        margins = [context["margin"] for context in contexts]
        gin_margins = [context["gin_margin"] for context in contexts]
        suitability = .5 * sum(margins) if all(value is not None for value in margins) else None
        gin_suitability = .5 * sum(gin_margins) if all(value is not None for value in gin_margins) else None
        rankings.append({
            "protocol": identifier, "dataset": protocol["dataset"], "metric": original.metric_for(identifier),
            "status": "unranked" if reasons else "definitive", "rank": None, "equal_rank": False,
            "suitability": suitability,
            "suitability_percentage_points": 100 * suitability if suitability is not None else None,
            "gin_suitability": gin_suitability,
            "gin_suitability_percentage_points": 100 * gin_suitability if gin_suitability is not None else None,
            "reasons": reasons, "epsilon_contexts": contexts,
            "selection_candidates_per_pair": CANDIDATES_PER_PAIR,
            "selection_epochs": list(ELIGIBLE_EPOCHS), "excluded_epochs": [25],
        })
    definitive = sorted((row for row in rankings if row["status"] == "definitive"),
                        key=lambda row: (-row["suitability"], row["protocol"]))
    ties = Counter(row["suitability"] for row in definitive)
    previous, rank = None, 0
    for position, row in enumerate(definitive, 1):
        if previous is None or row["suitability"] != previous:
            rank = position
        row.update(rank=rank, equal_rank=ties[row["suitability"]] > 1)
        previous = row["suitability"]
    return definitive + sorted((row for row in rankings if row["status"] != "definitive"),
                               key=lambda row: row["protocol"])


def _report_slots(root: Path, selected: dict, prepared: dict, warnings: list[str]) -> list[dict]:
    # Excluded epochs cannot become downstream report winners, even in a bad frozen file.
    eligible = {**selected, "winners": [winner if _winner_epochs(winner) in ELIGIBLE_EPOCHS else
                {**winner, "status": "unavailable", "cell": None,
                 "reason": "No eligible frozen 10/20-epoch winner; 25 epochs is archived/excluded"}
                for winner in selected.get("winners", [])]}
    expected = comparison_slots(eligible, prepared)
    persisted = original.load(root / "comparison_slots.json", None, warnings)
    if persisted is None:
        warnings.append("comparison_slots.json is absent; expected presentation slots shown without implying dispatch")
        return expected
    indexed = defaultdict(list)
    for slot in persisted:
        indexed[original._slot_identity(slot)].append(slot)
    if len(persisted) != PRESENTATION_SLOTS or len(indexed) != PRESENTATION_SLOTS:
        warnings.append("comparison registry does not contain 192 unique presentation slots")
    slots = []
    for expected_slot in expected:
        identity = original._slot_identity(expected_slot)
        matches = indexed[identity]
        reason = None
        if len(matches) != 1:
            reason = "presentation request missing or duplicated in persisted registry"
        else:
            slot = matches[0]
            cell, expected_cell = slot.get("cell"), expected_slot.get("cell")
            key = scientific_key(cell) if cell else None
            expected_key = scientific_key(expected_cell) if expected_cell else None
            if (key != expected_key or slot.get("selected_key") != expected_slot.get("selected_key")
                    or slot.get("selection_status") != expected_slot.get("selection_status")):
                reason = "comparison request differs from frozen eligible selection"
        if reason:
            warnings.append(f"{identity}: {reason}")
            slots.append({**expected_slot, "cell": None, "status": "unavailable", "reason": reason})
        else:
            slots.append(matches[0])
    return slots


def _render_results(root: Path, specs: list[dict], tuning: list[dict], comparison: list[dict],
                    selected: dict, counts: dict, warnings: list[str]) -> str:
    winners = _winner_map(selected)
    lines = ["# SparseGNN final matched comparison", f"Evidence snapshot: {utc_now()}.", QUALIFICATIONS,
             "**Revised selection:** 384 eligible tuning requests, 16 per protocol/epsilon, using only completed "
             "10- and 20-epoch runs. All 192 original 25-epoch requests are **archived and excluded from "
             "selection and ranking even when completed**. Their observed outcomes and scores remain in the "
             "full 576-row ledger and the tables below; exclusion is not an execution status. There are 24 "
             "winner slots and 192 final comparison presentation slots (12 protocols × 2 epsilons × 8 methods). "
             "This does not claim that all 576 original tuning requests completed.",
             f"**Outcome: {counts['outcome']}**. Frozen eligible winners: {counts['selected_winners']}/24; "
             f"verified eligible tuning: {counts['selection_eligible_accepted']}/384; verified original-ledger "
             f"tuning: {counts['tuning_accepted']}/576; verified comparison slots: "
             f"{counts['comparison_accepted_slots']}/192. Terminal coverage and full successful execution "
             "are distinct. Missing, timed-out, failed, blocked or unverified results receive no accepted "
             "score and no zero imputation.",
             "User-cancelled non-private rows are explicitly partial or unstarted, never completed 100-epoch "
             "results. Partial rows show the saved validation-selected checkpoint's validation score and "
             "persisted epochs/updates. Their test score is unavailable: no final test evaluation was performed. "
             "Private-only suitability ranking is unaffected by cancelled non-private baselines.",
             "Unrounded machine-readable evidence: [tuning](tuning.json), [frozen selections](selected.json), "
             "[comparisons](comparison.json), [ranking](ranking.json), [attempt ledger](attempts.json), "
             "[timing](timing.csv), [comparison verification](phases/compare/verification.json).",
             "Coverage: " + original.fmt(counts)]
    for protocol in specs:
        identifier = protocol["id"]
        domain = protocol.get("domain_split")
        description = ("Native-split graph-disjoint inductive: native role masks with separate induced "
                       "role-context graphs, not official full-graph benchmark numbers.")
        if domain:
            description = (f"All-but-two domain graph-disjoint protocol: train {original.fmt(domain['train'])}; "
                           f"validation {original.fmt(domain['val'])}; test {original.fmt(domain['test'])}. "
                           "Each held-out domain is scored in its complete own-domain context.")
        if identifier == "mag-allbut2":
            description += " All 20 logits are trained; label 19 is excluded only from scores, including full scored CN validation and DE test."
        lines.extend([f"## {identifier}",
                      f"Loader/release: `{protocol['dataset']}`. Primary metric: **{original.metric_for(identifier)}**. {description}"])
        for epsilon in EPSILONS:
            winner = winners.get((identifier, epsilon), {})
            tuning_group = [row for row in tuning if row["protocol"] == identifier and row.get("epsilon") == epsilon]
            lines.extend([
                f"### Original tuning ledger — epsilon {epsilon:g}",
                f"Frozen selection status: **{winner.get('status', 'not frozen')}**; completed eligible candidates "
                f"{winner.get('completed_candidates', 0)}/16. Only 10/20 epochs are eligible; the eight "
                "25-epoch rows below remain visible but cannot win. Configuration selection basis: "
                "**test_metric**. " + original.link(root, winner.get("folder"), "frozen selected evidence"),
                original.table(
                    ["B", "p2", "LR", "Epochs requested / completed", "Updates planned / completed", "Validation", "Test",
                     "Selection eligibility", "Selection", "Status / reason", "Process seconds", "Evidence"],
                    [[row.get("batch_size"), row.get("p2"), row.get("learning_rate"),
                      [row.get("requested_epochs"), row.get("completed_epochs")],
                      [row.get("steps"), row.get("updates_completed")], row.get("validation_metric"), row.get("test_metric"),
                      "eligible (10/20)" if row["selection_eligible"] else "archived/excluded (25)",
                      "**TEST-SELECTED WINNER**" if row["selected"] else "",
                      [row["status"], row.get("reason")], row.get("process_wall_seconds"),
                      original.link(root, row.get("folder"))] for row in tuning_group]),
                f"### Matched comparison — epsilon context {epsilon:g}",
            ])
            group = [row for row in comparison if row["protocol"] == identifier and row["epsilon_context"] == epsilon]
            lines.append(original.table(
                ["Method", "Width", "Requested B / actual APPR M", "LR / p2", "Epochs / stages", "Completed epochs",
                 "Updates planned / completed", "Budget target / achieved", "Validation / test", "Process seconds",
                 "Status / selection / reason", "Evidence"],
                [[row["report_method"], row.get("hidden_size"), [row.get("batch_size"), row.get("supervised_appr_roots")],
                  [row.get("learning_rate"), row.get("p2")],
                  ["100 (nonprivate override)" if row.get("method") in {"mlp", "graphsage"} else row.get("epochs"), row.get("stages")],
                  row.get("completed_epochs"), [row.get("steps"), row.get("updates_completed")],
                  "nonprivate" if row.get("method") in {"mlp", "graphsage"} else [row.get("epsilon_target"), row.get("epsilon_estimate")],
                  [row.get("validation_metric"), row.get("test_metric")], row.get("process_wall_seconds"),
                  [row["status"], row.get("selection_status"), row.get("reason")],
                  original.link(root, row.get("folder"))] for row in group]))
        lines.append("DPAR's requested B is distinct from the supervised APPR population M≤70. ProGAP executes "
                     "three stages with E epochs per stage. All methods use the same task/split contexts. A "
                     "non-private run may serve both epsilon-context slots only with identical effective settings, "
                     "split, seed, implementation and verified artifacts; presentation-slot counts are not physical-run counts.")
    if warnings:
        lines.extend(["## Evidence warnings", *[f"- {warning}" for warning in warnings]])
    return "\n\n".join(lines) + "\n"


def _render_ranking(root: Path, rankings: list[dict]) -> str:
    lines = ["# Exploratory dataset suitability ranking", QUALIFICATIONS,
             "For each epsilon, margin = test(SparseGNN-SAGE) − max(test(ProGAP), test(DPAR), "
             "test(DP-GNN-SAGE), test(DP-MLP)). Suitability = 0.5 × (margin at epsilon 2 + margin at "
             "epsilon 8), using unrounded scores. Margins below are primary-score percentage points "
             "(100 × margin). GIN is diagnostic only; its performance never changes the primary ranking. "
             "Exact ties share competition ranks and sort by protocol ID for presentation. Definitive "
             "coverage requires all **16/16 eligible 10/20-epoch candidates at both epsilons** and verified "
             "SparseGNN-SAGE plus all four private baselines at both epsilons. The 25-epoch candidates "
             "are excluded regardless of success. Missing competitors are never rewarded.",
             "Accuracy, micro-F1 and AUROC margins are compared only as the requested heuristic: equal "
             "numerical margins do not imply equal statistical difficulty. Non-private gaps compare "
             "100-epoch references with selected 10/20-epoch private runs, not equal-epoch contrasts. "
             "'Definitive' refers only to evidence coverage within this revised scope, not statistical certainty."]
    for title, definitive in (("Definitive coverage (still exploratory)", True),
                              ("Provisional / unranked — no definitive ordering", False)):
        subset = [row for row in rankings if (row["status"] == "definitive") == definitive]
        lines.extend(["## " + title, original.table(
            ["Rank", "Protocol", "Metric", "Suitability pp", "GIN diagnostic pp", "Epsilon 2 margin pp / strongest private",
             "Epsilon 8 margin pp / strongest private", "Reasons / evidence"],
            [[str(row["rank"]) + (" (equal rank)" if row["equal_rank"] else "") if row["rank"] is not None else "unranked",
              row["protocol"], row["metric"], row["suitability_percentage_points"], row["gin_suitability_percentage_points"],
              *[[context["margin_percentage_points"], context["strongest_private_baseline"]] for context in row["epsilon_contexts"]],
              "; ".join(row["reasons"]) + " " + original.link(root, "comparison.json", "comparison evidence")]
             for row in subset])])
    lines.extend(["## Non-private gaps and runtime (not used for rank)", original.table(
        ["Protocol / epsilon", "SAGE − non-private MLP pp", "SAGE − non-private SAGE pp", "Method runtimes seconds", "Evidence"],
        [[f"{row['protocol']} / {context['epsilon']:g}",
          100 * context["nonprivate_mlp_gap"] if context["nonprivate_mlp_gap"] is not None else None,
          100 * context["nonprivate_sage_gap"] if context["nonprivate_sage_gap"] is not None else None,
          context["runtime_seconds"], "; ".join(original.link(root, entry["folder"], method)
                                               for method, entry in context["evidence"].items())]
         for row in rankings for context in row["epsilon_contexts"]])])
    return "\n\n".join(lines) + "\n"


# cached_sha256 keys include inode, size, mtime and ctime. Explicit final
# verification still defaults to fresh hashes; publication reuses stable bytes.
_PUBLICATION_HASH_CACHE: dict = {}


def summarize(root: Path = HERE) -> dict:
    """Publish all original requests and final slots without modifying frozen selection."""
    from scripts.sparsegnn_final_verification import verify_comparison, verify_selection

    root = Path(root)
    warnings: list[str] = []
    hash_cache = _PUBLICATION_HASH_CACHE
    selection_verification = verify_selection(root, require_frozen=True, hash_cache=hash_cache)
    comparison_verification = verify_comparison(root, hash_cache=hash_cache)
    warnings.extend(f"selection verification: {error}" for error in selection_verification.get("errors", []))
    warnings.extend(f"comparison verification: {error}" for error in comparison_verification.get("errors", []))
    specs = protocols()
    prepared = original.load(root / "prepared.json", {"protocols": {}}, warnings)
    expected_tuning = tuning_cells(prepared)
    records = original.collect_phase_rows(root, "tune", hash_cache=hash_cache)
    indexed = defaultdict(list)
    for record in records:
        indexed[record["scientific_key"]].append(record)
    if len(records) != ORIGINAL_REQUESTED or len(indexed) != ORIGINAL_REQUESTED:
        warnings.append(f"original tuning registry has {len(records)} rows/{len(indexed)} unique keys; ledger denominator remains 576")
    tuning_records = []
    for cell in expected_tuning:
        key = scientific_key(cell)
        matches = indexed[key]
        tuning_records.append(matches[0] if len(matches) == 1 else {
            "cell": cell, "scientific_key": key, "status": "missing_request" if not matches else "invalid_evidence",
            "accepted": False, "folder": None, "result": None,
            "reason": "original grid request missing or duplicated in persisted registry",
        })
    tuning = [original.report_row(root, record, "tune") for record in tuning_records]
    selected = original.load(root / "selected.json", {"winners": []}, warnings)
    winners = _winner_map(selected)
    selection_current = selection_verification.get("accepted") is True
    design_errors = comparison_verification.get("design_errors", [])
    ranking_current = selection_current and not design_errors
    for row in tuning:
        winner = winners.get((row["protocol"], float(row["epsilon"])), {})
        eligible = row.get("epochs") in ELIGIBLE_EPOCHS
        row.update(selection_eligible=eligible,
                   selection_exclusion_reason=None if eligible else "25-epoch candidate archived/excluded by revised selection scope",
                   selected=eligible and row["scientific_key"] == winner.get("scientific_key"),
                   selection_status=winner.get("status", "unavailable"),
                   selection_candidates_per_pair=CANDIDATES_PER_PAIR,
                   selection_evidence_current=selection_current)
    slots = _report_slots(root, selected, prepared, warnings)
    compare_records = original.collect_phase_rows(root, "compare", hash_cache=hash_cache)
    invalid_keys = set(comparison_verification.get("invalid_keys", []))
    request_errors = {row["scientific_key"]: row.get("errors", [])
                      for row in comparison_verification.get("requests", []) if row.get("errors")}
    checked_records = [
        {**record, "accepted": False,
         "reason": "; ".join(request_errors.get(record["scientific_key"], [])) or "comparison registry verification rejected this request"}
        if record["scientific_key"] in invalid_keys else record for record in compare_records
    ]
    comparison = original.collect_comparison_rows(root, slots, selected, checked_records, records)
    for row in comparison:
        if row["status"] == "cancelled_partial" and row["scientific_key"] not in invalid_keys:
            partial = original.load(Path(row["folder"]) / "partial.json", {}, warnings)
            for field in ("completed_epochs", "updates_completed", "roots_total", "process_wall_seconds",
                          "validation_metric", "test_metric", "selected_checkpoint_epoch",
                          "selected_checkpoint_step", "partial_result", "metric_scope", "progress_scope"):
                row[field] = partial.get(field)
            row["accepted"] = False
        winner = winners.get((row["protocol"], float(row["epsilon_context"])), {})
        row.update(selection_eligible=_winner_epochs(winner) in ELIGIBLE_EPOCHS,
                   requested_candidates=CANDIDATES_PER_PAIR,
                   current_verified_tuning_candidates=winner.get("completed_candidates", 0) if selection_current else 0,
                   selection_evidence_current=ranking_current and _complete_selection(winner),
                   selection_qualification="test-selected complete 16-point grid; epochs 10/20 only, 25 archived/excluded"
                   if selection_current and _complete_selection(winner) else "no currently verified complete 16-point selection",
                   active_timeout_seconds=ACTIVE_TIMEOUT_SECONDS)
    ranking_selection = {**selected, "winners": [dict(winner) for winner in selected.get("winners", [])]}
    if not ranking_current:
        for winner in ranking_selection["winners"]:
            winner["status"] = "unavailable"
    rankings = rank_datasets(comparison, ranking_selection)
    smoke = (original.collect_phase_rows(root, "smoke", hash_cache=hash_cache)
             if (root / "phases" / "smoke" / "requests.json").exists() else [])
    attempts, timings = original.collect_attempt_ledger(root, [*smoke, *records, *compare_records])
    partial_rows = {row["scientific_key"]: row for row in comparison
                    if row["status"] == "cancelled_partial" and row.get("partial_result")}
    for row in attempts:
        row["selection_eligible"] = row.get("phase") == "tune" and row.get("epochs") in ELIGIBLE_EPOCHS
        partial = partial_rows.get(row["scientific_key"])
        if partial and row.get("folder") == partial.get("folder"):
            row.update({field: partial.get(field) for field in
                        ("status", "completed_epochs", "updates_completed", "roots_total",
                         "selected_checkpoint_epoch", "validation_metric", "test_metric", "metric_scope")})
    eligible_rows = [row for row in tuning if row["selection_eligible"]]
    excluded_rows = [row for row in tuning if not row["selection_eligible"]]
    complete = comparison_verification.get("complete") is True
    fully_executed = comparison_verification.get("fully_executed") is True
    if fully_executed and ranking_current:
        outcome = "revised-scope comparison fully executed"
    elif complete and ranking_current:
        outcome = "campaign resolved with gaps"
    else:
        outcome = "comparison in progress or unresolved; inspect dispositions and evidence warnings"
    private_rows = [row for row in comparison if row["method"] not in {"mlp", "graphsage"}]
    if complete and all(row["accepted"] for row in private_rows) and partial_rows:
        outcome = "private comparison complete; remaining non-private training cancelled by user"
    counts = {
        "tuning_requested": ORIGINAL_REQUESTED, "tuning_accepted": sum(row["accepted"] for row in tuning),
        "selection_eligible_requested": ELIGIBLE_REQUESTED,
        "selection_eligible_accepted": sum(row["accepted"] for row in eligible_rows),
        "selection_excluded_requested": ORIGINAL_REQUESTED - ELIGIBLE_REQUESTED,
        "selection_excluded_accepted": sum(row["accepted"] for row in excluded_rows),
        "winner_slots": WINNER_SLOTS,
        "selected_winners": sum(_complete_selection(winner) for winner in selected.get("winners", [])) if selection_current else 0,
        "selection_verified": selection_current,
        "comparison_slots": PRESENTATION_SLOTS,
        "comparison_accepted_slots": sum(row["accepted"] for row in comparison),
        "comparison_physical_requests": len(compare_records),
        "comparison_complete": complete, "comparison_fully_executed": fully_executed,
        "active_timeout_seconds": ACTIVE_TIMEOUT_SECONDS,
        "tuning_statuses": dict(Counter(row["status"] for row in tuning)),
        "selection_eligible_statuses": dict(Counter(row["status"] for row in eligible_rows)),
        "selection_excluded_statuses": dict(Counter(row["status"] for row in excluded_rows)),
        "comparison_statuses": dict(Counter(row["status"] for row in comparison)),
        "private_comparison_accepted": sum(row["accepted"] for row in private_rows),
        "private_comparison_requested": len(private_rows),
        "physical_comparison_statuses": dict(Counter(row["status"] for row in compare_records)),
        "definitively_ranked_datasets": sum(row["status"] == "definitive" for row in rankings),
        "outcome": outcome,
    }
    now = utc_now()
    for filename, rows in (("tuning", tuning), ("comparison", comparison), ("ranking", rankings),
                           ("attempts", attempts), ("timing", timings)):
        atomic_json(root / f"{filename}.json", {
            "generated_utc": now, "qualification": QUALIFICATIONS, "selection_scope": SELECTION_SCOPE,
            "counts": counts, "rows": rows, "warnings": warnings,
        })
        original.write_csv(root / f"{filename}.csv", rows)
    original.atomic_text(root / "RESULTS.md", _render_results(root, specs, tuning, comparison, selected, counts, warnings))
    original.atomic_text(root / "RANKING.md", _render_ranking(root, rankings))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps({"status": "published", "root": str(args.root), "counts": summarize(args.root)}), flush=True)


if __name__ == "__main__":
    main()
