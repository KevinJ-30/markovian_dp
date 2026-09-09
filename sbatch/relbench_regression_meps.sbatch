#!/bin/bash
#SBATCH -Jrelreg
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # rel-amazon is multi-GB; memory, not
#SBATCH -t8:00:00                         # compute, is the binding constraint
#SBATCH -oReport-relreg-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── RelBench: scope, then DP-GNN vs DP-blind at MATCHED epsilon ──
#
# Generalizes relamazon_scope.sbatch (binary_gnn/AUROC) and ppi_matched_eps.sbatch
# (the blind-arm-first design) into one script parameterized by dataset/task/
# model/metric, so both regression and classification RelBench candidates
# reuse the same STAGE0/1/2 shape:
#
#   sbatch --export=ALL,DS_NAME=rel-f1,TASK=driver-position \
#       sbatch/relbench_regression_meps.sbatch                        # regression, MAE
#   sbatch --export=ALL,DS_NAME=rel-hm,TASK=user-churn,MODEL=binary_gnn,METRIC=test_auroc \
#       sbatch/relbench_regression_meps.sbatch                        # classification, AUROC
#
# DEFAULTS point at rel-f1/driver-position on purpose (2026-09-09: rel-amazon/
# rel-hm hit a download issue on ICE that there wasn't time to chase tonight;
# rel-f1 has run successfully here before -- see results/relbench/relf1_*).
# ENV_SCRIPT lets the same file run locally: ENV_SCRIPT=scripts/_local_env.sh
# bash sbatch/relbench_regression_meps.sbatch (the #SBATCH lines are just
# comments to plain bash).
#
# THE QUESTION THIS ANSWERS, same as ppi_matched_eps: does the graph survive
# matched-epsilon noise on a regression target, i.e. does --model regression_gnn
# at r=$R beat the SAME mechanism at r=0 (graph-blind) once both are calibrated
# to the same (p1, T, delta) -> epsilon.
#
# WHY r DEFAULTS TO 2.  relbench_data.py: a row root reaches its entity at
# depth 1 and that entity's history at depth 2 -- r=1 sees no history at all,
# so r=1 is not a meaningful "graph" arm here (unlike PPI, where r=1 already
# reaches real neighbours). r=0 stays the blind arm for the same reason it was
# on PPI/rel-amazon: no expansion, collapses to a plain subsampled Gaussian.
#
# WHY SCOPE FIRST.  rel-trial showed a large K_out is not sufficient; a task
# needs enough TRAINING ROWS that a usable batch keeps p1 small, because
# epsilon is superlinear in p1. STAGE 0 prices the grid before spending any
# wallclock, and aborts if eps=1 is unreachable at a usable p1.
#
# P2_LIST is deliberately narrow (dense vs. one sparse point) for a first probe
# on an unproven task; widen once STAGE 2 shows the frontier clears the r=0
# blind arm at an affordable epsilon.

set -u
cd ${SLURM_SUBMIT_DIR:-.}

source "${ENV_SCRIPT:-scripts/_ice_env.sh}"

DS_NAME=${DS_NAME:-rel-f1}
TASK=${TASK:-driver-position}
MODEL=${MODEL:-regression_gnn}
METRIC=${METRIC:-test_mae}
DS="relbench:${DS_NAME}/${TASK}"
OUT_ROOT=${OUT_ROOT:-results/relbench/${DS_NAME//-/}_${TASK//-/}_meps}
R=${R:-2}
T=${T:-500}
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="1 2 4 8"
P2_LIST=${P2_LIST:-"1.0 0.1"}

mkdir -p "$OUT_ROOT" results/logs

$PY -c 'import relbench' 2>/dev/null || {
  echo "FATAL: relbench is not importable via \$PY (${PY:-python})" >&2
  echo "       fix: on ICE, source \$VENV/bin/activate && pip install relbench==2.1.2" >&2
  exit 1
}

echo "=== STAGE 0: scope $DS $(date) ==="
if ! $PY scripts/relbench_scope.py --dataset "$DS_NAME" --task "$TASK" \
        --batch 512 --T $T --eps $EPS_LIST --r $R --grid $GRID \
        > "$OUT_ROOT/scope.env" 2> "$OUT_ROOT/scope.txt"; then
  cat "$OUT_ROOT/scope.txt" >&2
  echo "FATAL: scoping failed; nothing downstream can run without its p1/K" >&2
  echo "       (if this is a bad --task name, check RelBench's task catalog" >&2
  echo "       for $DS_NAME -- this script does not guess task names)" >&2
  exit 1
fi
cat "$OUT_ROOT/scope.txt"
eval "$(cat "$OUT_ROOT/scope.env")"
: "${SCOPE_P1:?scope.env did not define SCOPE_P1}"
echo "  -> p1=$SCOPE_P1  K_in=$SCOPE_K_IN  K_out=$SCOPE_K_OUT  n=$SCOPE_N_NODES"

COMMON=(--dataset "$DS" --direction in --model "$MODEL" --aggr mean
        --inductive --p1 "$SCOPE_P1" --num_layers 2 --hidden 256 --clip 1.0
        --dropout 0.0 --weight_decay 0.0 --roots_from train
        --seeds $SEEDS --T $T --track_every 25
        --K_in "$SCOPE_K_IN" --K_out "$SCOPE_K_OUT")

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out"
}

echo "=== STAGE 1: references $(date) ==="
run_cell "$OUT_ROOT/nodp_r${R}"  --p2 1.0 --r $R --optimizer adam --lr 0.01
run_cell "$OUT_ROOT/nodp_blind"  --p2 1.0 --r 0  --optimizer adam --lr 0.01

echo "=== STAGE 2a: blind arm (r=0), matched epsilon $(date) ==="
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 "$SCOPE_P1" --r 0 \
    --K "$SCOPE_K_OUT" --T $T --grid $GRID --delta_from_n "$SCOPE_N_NODES" \
    > "$OUT_ROOT/sigma_blind.txt"
cat "$OUT_ROOT/sigma_blind.txt"

grep -v '^#' "$OUT_ROOT/sigma_blind.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] blind eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/blind_eps${EPS}" \
      --p2 1.0 --r 0 --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo "=== STAGE 2b: GNN arm (r=$R), matched epsilon $(date) ==="
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 "$SCOPE_P1" \
    --r $R --K "$SCOPE_K_OUT" --T $T --grid $GRID \
    --delta_from_n "$SCOPE_N_NODES" > "$OUT_ROOT/sigma_gnn.txt"
cat "$OUT_ROOT/sigma_gnn.txt"

grep -v '^#' "$OUT_ROOT/sigma_gnn.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/gnn_p2${P2}_eps${EPS}" \
      --p2 "$P2" --r $R --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo
echo "=== $DS ($MODEL) matched-epsilon complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric "$METRIC" 2>/dev/null || true
