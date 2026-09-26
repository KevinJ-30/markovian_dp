#!/usr/bin/env bash
# Supervise a fresh study, validate it, then reconstruct comparative figures.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/sparse_ablation_paper.sh [--dry-run]

Without OFAT_ROOT, run the 60-cell epsilon8/seed0 one-factor SparseExpand
ablation. With OFAT_ROOT, reuse that immutable completed study and run only
the 27 DP-GNN SAGE/GIN radius1,2,3 and ProGAP depth1,2,3 baselines. Both modes
use the existing GPU queue, validate outputs, and reconstruct bootstrap-CI
figures. No p2*cap interactions.

Environment:
  PYTHON       Training/analysis interpreter [python]
  GPUS         Authorized queue GPU selector [auto]
  OFAT_ROOT    Optional completed 60-run SGNN root; never written by this launcher
  OUT_ROOT     New result root [results/sparse_ablation_ofat, or
               results/sparse_ablation_depth when OFAT_ROOT is supplied]
  Figures and plotting data are written to OUT_ROOT/figures.

Paths are relative to the repository. The queue may resume its matching root;
raw attempts are preserved. Diagnose failures before using the queue's explicit
--retry-failed command. Existing figure directories are never overwritten.
--dry-run prints all 60 or 27 worker commands plus queue/report/render commands
without creating artifacts or launching training.
HELP
}
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (($#)); then usage >&2; exit 2; fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-auto}"
OFAT_ROOT="${OFAT_ROOT:-}"
study=ofat
default_root="$ROOT/results/sparse_ablation_ofat"
if [[ -n "$OFAT_ROOT" ]]; then
    study=depth-baselines
    default_root="$ROOT/results/sparse_ablation_depth"
    [[ "$OFAT_ROOT" = /* ]] || OFAT_ROOT="$ROOT/$OFAT_ROOT"
    if [[ ! -f "$OFAT_ROOT/manifest.json" ]]; then
        printf 'OFAT_ROOT requires a completed study manifest: %s\n' "$OFAT_ROOT" >&2
        exit 2
    fi
fi
OUT_ROOT="${OUT_ROOT:-$default_root}"
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$ROOT/$OUT_ROOT"
if [[ -e "$OUT_ROOT/figures" || -L "$OUT_ROOT/figures" ]]; then
    printf 'Refusing existing figure directory: %s\n' "$OUT_ROOT/figures" >&2
    exit 2
fi
queue=("$PYTHON" -B "$ROOT/scripts/full_matrix_queue.py" "--ablation-$study"
    --batch-size 256 --out-root "$OUT_ROOT" --gpus "$GPUS")
report=("$PYTHON" -B "$ROOT/scripts/full_matrix_queue.py" "--ablation-$study"
    --batch-size 256 --out-root "$OUT_ROOT" --report-only)
analysis=("$PYTHON" -B "$ROOT/scripts/sparse_ablation.py"
    --ofat-root "${OFAT_ROOT:-$OUT_ROOT}")
if [[ -n "$OFAT_ROOT" ]]; then
    analysis+=(--depth-root "$OUT_ROOT")
fi
print_command() { printf '%q ' "$@"; printf '\n'; }
if (( DRY_RUN )); then
    preview=("$PYTHON" -B "$ROOT/scripts/sparse_ablation_grid.py"
        --study "$study" --device cuda --out-root "$OUT_ROOT" --dry-run)
    if [[ -n "$OFAT_ROOT" ]]; then
        preview+=(--ofat-root "$OFAT_ROOT")
    fi
    "${preview[@]}"
    print_command "${queue[@]}"
    print_command "${report[@]}"
    print_command "${analysis[@]}"
else
    print_command "${queue[@]}"
    "${queue[@]}"
    print_command "${report[@]}"
    "${report[@]}"
    print_command "${analysis[@]}"
    "${analysis[@]}"
fi
