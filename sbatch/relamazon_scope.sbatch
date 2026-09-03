#!/bin/bash
#SBATCH -A coc-ice
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                        # rel-amazon is multi-GB; memory, not
#SBATCH -t8:00:00                         # compute, is the binding constraint
#SBATCH -oReport-relamazon-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── rel-amazon/user-churn: scope first, then a minimal matched-epsilon probe ──
#
#   sbatch sbatch/relamazon_scope.sbatch
#
# WHY REL-AMAZON.  RelBench is where this mechanism should look best: a foreign
# key graph has a small out-degree, and under in-expansion epsilon is priced by
# K_out (shells n_d = K_out^d, Thm 6.4 Eq. 44), so the structure is cheap by
# construction.  rel-trial was supposed to show that and did not:
#
#     p2    sigma    eps      AUROC      (chance = 0.5)
#     1.0     2.0  432.3     0.6743
#     1.0    20.0   16.1     0.6620
#     0.1     2.0   23.1     0.6381
#     0.1    20.0    1.45    0.5531
#
# The dense arm bottoms out at eps=16 and the only cheap point is near chance.
# The cause is not K_out -- it is p1.  rel-trial has ~1.4k training rows, so a
# usable batch forces p1=0.05, and epsilon is superlinear in p1 (0.013 -> 0.065
# measured 7.6x).  rel-amazon is one of the largest RelBench databases, so the
# same batch costs a far smaller p1.  That is the whole hypothesis being tested.
#
# Note those rel-trial epsilons were computed at delta=1e-5.  rel-amazon under
# delta = n^-1.01 with n in the millions is far stricter, which pushes epsilon
# UP -- the row-count advantage has to beat that too.
#
# STAGE 0 is a go/no-go: it reports node/edge/row counts and prices the grid
# BEFORE training anything.  If eps=1 is unreachable at a usable p1, stop there
# and record that as the result rather than burning the walltime.

set -u
cd $SLURM_SUBMIT_DIR

source scripts/_ice_env.sh

DS_NAME=${DS_NAME:-rel-amazon}
TASK=${TASK:-user-churn}
DS="relbench:${DS_NAME}/${TASK}"
OUT_ROOT=${OUT_ROOT:-results/relbench/relamazon_${TASK}}
T=${T:-500}
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="1 2 4 8"

mkdir -p "$OUT_ROOT" results/logs

echo "=== STAGE 0: scope $DS $(date) ==="
$PY scripts/relbench_scope.py --dataset "$DS_NAME" --task "$TASK" \
    --batch 512 --T $T --eps $EPS_LIST --r 2 --grid $GRID \
    > "$OUT_ROOT/scope.env" 2> "$OUT_ROOT/scope.txt"
cat "$OUT_ROOT/scope.txt"
eval "$(cat "$OUT_ROOT/scope.env")"
echo "  -> p1=$SCOPE_P1  K_in=$SCOPE_K_IN  K_out=$SCOPE_K_OUT  n=$SCOPE_N_NODES"

# ── STAGE 1: references.  AUROC's floor is 0.5 (chance), so a graph-blind r=0
#    run and a non-DP run bracket whatever the DP cells achieve.
COMMON=(--dataset "$DS" --direction in --model binary_gnn --aggr mean
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
run_cell "$OUT_ROOT/nodp_r2"    --p2 1.0 --r 2 --optimizer adam --lr 0.01
run_cell "$OUT_ROOT/nodp_blind" --p2 1.0 --r 0 --optimizer adam --lr 0.01

# ── STAGE 2: matched-epsilon probe.  Deliberately narrow -- two p2 values, to
#    see whether the frontier clears chance at an affordable epsilon.  Widen
#    only if it does.
echo "=== STAGE 2: matched-epsilon probe $(date) ==="
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 0.1 \
    --p1 "$SCOPE_P1" --r 2 --K "$SCOPE_K_OUT" --T $T --grid $GRID \
    --delta_from_n "$SCOPE_N_NODES" > "$OUT_ROOT/sigma.txt"
cat "$OUT_ROOT/sigma.txt"

grep -v '^#' "$OUT_ROOT/sigma.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/dp_p2${P2}_eps${EPS}" \
      --p2 "$P2" --r 2 --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo
echo "=== rel-amazon scope complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
