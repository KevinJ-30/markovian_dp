#!/usr/bin/env bash
# Supervise the fixed OFAT batch, validate it, then reconstruct paper figures.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/sparse_ablation_paper.sh [--dry-run]

Run the 60-cell epsilon8/seed0 one-factor ablation through the existing GPU
queue, validate all outputs, then reconstruct bootstrap-CI figures. No p2*cap
interactions. Use either this pathway or sparse_ablation_ofat.sh, not both.

Environment:
  PYTHON       Training/analysis interpreter [python]
  GPUS         Authorized queue GPU selector [auto]
  OUT_ROOT     New result root [results/sparse_ablation_ofat]
  Figures and plotting data are written to OUT_ROOT/figures.

Paths are relative to the repository. The queue may resume its matching root;
raw attempts are preserved. Diagnose failures before using the queue's explicit
--retry-failed command. Existing figure directories are never overwritten.
--dry-run prints the 60 worker commands and reconstruction commands only.
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
OUT_ROOT="${OUT_ROOT:-$ROOT/results/sparse_ablation_ofat}"
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$ROOT/$OUT_ROOT"
if [[ -e "$OUT_ROOT/figures" || -L "$OUT_ROOT/figures" ]]; then
    printf 'Refusing existing figure directory: %s\n' "$OUT_ROOT/figures" >&2
    exit 2
fi
queue=("$PYTHON" -B "$ROOT/scripts/full_matrix_queue.py" --ablation-ofat
    --batch-size 256 --out-root "$OUT_ROOT" --gpus "$GPUS")
report=("$PYTHON" -B "$ROOT/scripts/full_matrix_queue.py" --ablation-ofat
    --batch-size 256 --out-root "$OUT_ROOT" --report-only)
analysis=("$PYTHON" -B "$ROOT/scripts/sparse_ablation.py"
    --ofat-root "$OUT_ROOT")
print_command() { printf '%q ' "$@"; printf '\n'; }
if (( DRY_RUN )); then
    PYTHON="$PYTHON" OUT_ROOT="$OUT_ROOT" DEVICE=cuda bash "$ROOT/scripts/sparse_ablation_ofat.sh" --dry-run
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
