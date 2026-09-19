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
# DEFAULTS point at rel-f1/driver-position as the known-safe smoke test.
# rel-amazon and rel-hm both hit real bugs on 2026-09-09 (missing `pooch`
# dependency on ICE; an unhashable list-valued column in rel-amazon's product
# table crashing _encode_table) -- both are fixed now, and both datasets have
# since completed full grids; see results/relbench/relhm_itemsales_meps and
# results/relbench/relamazon_userltv_meps.
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
BATCH=${BATCH:-256}   # matches Facebook/ProGAP; PPI/arxiv stay at 512 (GraphSAGE)
# run.py writes every mechanism's primary metric into a column literally named
# test_acc regardless of what it measures (accuracy for classification, MAE
# for regression_gnn -- see run.py's CSV header) -- test_mae does not exist as
# a column and silently summarizes to nothing.  test_rmse is the one metric
# that IS its own column, for a secondary read.
MODEL=${MODEL:-regression_gnn}
METRIC=${METRIC:-test_acc}
DS="relbench:${DS_NAME}/${TASK}"
OUT_ROOT=${OUT_ROOT:-results/relbench/${DS_NAME//-/}_${TASK//-/}_meps}
R=${R:-2}
ROOT=${ROOT:-row}          # 'row' = one root per prediction row (a leaf hanging
                           # off its entity, so r=1 reaches only the entity);
                           # 'entity' = root at the entity, labels aggregated.
# One optimizer for BOTH arms. The DP cells used to run sgd/0.3 while the non-DP
# cells ran adam/0.01, which confounds the privacy comparison with an optimizer
# change; _dataset_settings.sh specifies adam/0.01 everywhere for that reason.
OPT=${OPT:-adam}
LR=${LR:-0.01}
T=${T:-500}
SEEDS=${SEEDS:-3}
# The discretization floor is ~T*grid, so the right value depends on T, and on
# r: at r=2 the composed PLD support is several times wider than at r=1, and a
# 1e-5 lattice over it OOM'd a 128G job in 90s.  At T=500 a 1e-4 floor is 0.05,
# 0.6% of eps=8 -- free.  Go finer only when T is large enough to need it.
GRID=${GRID:-1e-4}
EPS_LIST=${EPS_LIST:-"1 2 4 8"}
P2_LIST=${P2_LIST:-"1.0 0.1"}
DROPOUT=${DROPOUT:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

mkdir -p "$OUT_ROOT" results/logs

$PY -c 'import relbench' 2>/dev/null || {
  echo "FATAL: relbench is not importable via \$PY (${PY:-python})" >&2
  echo "       fix: on ICE, source \$VENV/bin/activate && pip install relbench==2.1.2" >&2
  exit 1
}

echo "=== STAGE 0: scope $DS $(date) ==="
if ! $PY scripts/relbench_scope.py --dataset "$DS_NAME" --task "$TASK" \
        --batch $BATCH --T $T --eps $EPS_LIST --r $R --grid $GRID --root "$ROOT" \
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

# EPOCHS beats a raw T: p1 varies by three orders of magnitude across RelBench
# tasks, so one T is 25 epochs on rel-avito and 0.07 on rel-hm.  p1 is only
# known after scoping, hence the recompute here; STAGE 0's sigma preview used
# the provisional T and is informational either way.
if [ -n "${EPOCHS:-}" ]; then
  T=$($PY -c "print(max(1, round($EPOCHS / $SCOPE_P1)))")
  echo "  -> EPOCHS=$EPOCHS at p1=$SCOPE_P1 -> T=$T"
fi

COMMON=(--dataset "$DS" --direction in --model "$MODEL" --aggr mean
        --p1 "$SCOPE_P1" --num_layers 2 --hidden 256 --clip 1.0
        --dropout $DROPOUT --weight_decay $WEIGHT_DECAY 
        --seeds $SEEDS --T $T --track_every 25
        --K_in "$SCOPE_K_IN" --K_out "$SCOPE_K_OUT"
        --relbench_root "$ROOT")

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.experiments.sparse "${COMMON[@]}" "$@" --out_dir "$out"
}

echo "=== STAGE 1: references $(date) ==="
run_cell "$OUT_ROOT/nodp_r${R}"  --p2 1.0 --r $R --optimizer $OPT --lr $LR
run_cell "$OUT_ROOT/nodp_blind"  --p2 1.0 --r 0  --optimizer $OPT --lr $LR

echo "=== STAGE 2a: blind arm (r=0), matched epsilon $(date) ==="
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 "$SCOPE_P1" --r 0 \
    --K "$SCOPE_K_OUT" --T $T --grid $GRID --delta_from_n "$SCOPE_N_NODES" \
    > "$OUT_ROOT/sigma_blind.txt"
cat "$OUT_ROOT/sigma_blind.txt"

grep -v '^#' "$OUT_ROOT/sigma_blind.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] blind eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/blind_eps${EPS}" \
      --p2 1.0 --r 0 --dp --sigma "$SIGMA" --optimizer $OPT --lr $LR
done

echo "=== STAGE 2b: GNN arm (r=$R), matched epsilon $(date) ==="
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 "$SCOPE_P1" \
    --r $R --K "$SCOPE_K_OUT" --T $T --grid $GRID \
    --delta_from_n "$SCOPE_N_NODES" > "$OUT_ROOT/sigma_gnn.txt"
cat "$OUT_ROOT/sigma_gnn.txt"

grep -v '^#' "$OUT_ROOT/sigma_gnn.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/gnn_p2${P2}_eps${EPS}" \
      --p2 "$P2" --r $R --dp --sigma "$SIGMA" --optimizer $OPT --lr $LR
done

echo
echo "=== $DS ($MODEL) matched-epsilon complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric "$METRIC" 2>/dev/null || true
