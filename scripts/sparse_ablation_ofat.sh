#!/usr/bin/env bash
# Sequential SparseExpand OFAT, with one fresh output directory per configuration.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/sparse_ablation_ofat.sh [--dry-run]

Run the fixed 60-configuration epsilon-8 SparseExpand OFAT study sequentially.
No interactions, seed replicates, tuning, or retries. After training, validate
all outputs and render the two backend comparison charts under OUT_ROOT/figures.
--dry-run prints shell-escaped worker commands and the count without creating
files, importing training packages, loading data, or allocating a GPU.

Environment:
  PYTHON    Python executable with training dependencies [python]
  DEVICE    Explicit training device [cuda]
  OUT_ROOT  New result root [<repository>/results/sparse_ablation_ofat]

Fixed: ogbn-arxiv, saint-yelp, twitch-allbut2; sparse_sage and sparse_gin;
epsilon8, seed0, lr0.01, batch256, epochs20, hidden128, dropout0.5, layers2.
Anchor: radius1, p2=0.5, outgoing degree cap10. Vary one factor at a time:
  radius       1 2 3
  p2           0.05 0.1 0.25 0.5 1
  outgoing cap 5 10 20 40
The shared anchor runs once. K_in=10 remains bookkeeping for outgoing-only
preprocessing. Every configuration is separately calibrated under the current
chi=1, union_safe=False policy, not a union-safe accounting claim.

Validation selects the checkpoint, never test performance. Stored 95% test
intervals use 1000 node-bootstrap resamples with bootstrap seed0; they are not
uncertainty across training seeds. Per-run logs and original outputs are kept.
The complete manifest and source snapshot are saved before the first worker.
Existing roots, symlinks, and roots nested inside known campaigns are refused.
Relative OUT_ROOT paths are relative to the repository, not the caller's cwd.
HELP
}

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (($#)); then
    usage >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/sparse_ablation_ofat}"
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$ROOT/$OUT_ROOT"
command=("$PYTHON" -B "$ROOT/scripts/sparse_ablation_grid.py"
    --study ofat --device "$DEVICE" --out-root "$OUT_ROOT")
if (( DRY_RUN )); then
    command+=(--dry-run)
fi
exec "${command[@]}"
