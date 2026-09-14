#!/bin/bash
#SBATCH -Jppi_adam
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t8:00:00
#SBATCH -oReport-ppi-adam-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── PPI matched-epsilon, Adam instead of SGD, everything else identical ──
#
#   sbatch sbatch/ppi_matched_eps_adam.sbatch
#
# THE QUESTION THIS ANSWERS.  ppi_matched_eps.sbatch (and every other DP cell
# in this repo) uses SGD, not Adam, for --dp training. The concern: Adam's
# per-parameter adaptive scaling divides by a running estimate of gradient
# magnitude, and DP noise is isotropic and added AFTER clipping -- so for a
# parameter with a small true gradient, noise dominates both the numerator
# and the moving-average denominator, and the adaptive step size ends up
# adapting to noise rather than signal. Plain SGD carries no such state to
# corrupt. PPI's own calibration already needs sigma=45.5 at eps=1 (see
# ppi_matched_eps.sbatch's header) -- high enough that this is not a
# theoretical concern. If Adam matches SGD here, standardize on Adam across
# the core engine (matching the baseline harness, which is already
# unconditionally Adam); if it regresses at low eps specifically, that is the
# answer, and SGD stays.
#
# SAME GRID, SAME CALIBRATION.  P1/T/K/HIDDEN/GRID/EPS_LIST/P2_LIST/N_NODES
# below are copied verbatim from ppi_matched_eps.sbatch, so calibrate_grid.py
# recomputes the IDENTICAL sigma per (p2, eps) cell -- this changes exactly
# one variable (optimizer + its learning rate) versus that script. lr=0.01
# is not a new choice: it's the same Adam lr already used for this model on
# this dataset in ppi_matched_eps.sbatch's own non-DP Stage-1 references.
# Reusing 0.3 (tuned for SGD) would not isolate the optimizer -- Adam's
# effective step is normalized to roughly unit gradient scale, so 0.3 there
# is a large step and would likely diverge regardless of DP noise.
#
# Compare cell-by-cell against results/ppi_matched_eps/ once both finish.

set -u
cd $SLURM_SUBMIT_DIR

source scripts/_ice_env.sh

OUT_ROOT=${OUT_ROOT:-results/ppi_matched_eps_adam}
N_NODES=56944
P1=0.0114
T=2000
K=5
HIDDEN=256
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="1 2 4 8"
P2_LIST="1.0 0.5 0.25 0.1"
ADAM_LR=0.01

mkdir -p "$OUT_ROOT" results/logs

COMMON=(--dataset ppi --direction in --model multilabel_gnn --aggr mean
        --p1 $P1 --num_layers 2 --hidden $HIDDEN --clip 1.0
        --dropout 0.0 --weight_decay 0.0 --roots_from train
        --seeds $SEEDS --T $T --optimizer adam --track_every 100)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_ppi*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out"
}

echo "=== PPI matched-epsilon sweep (Adam) $(date) ==="
echo "    hidden=$HIDDEN K=$K T=$T p1=$P1 seeds=$SEEDS delta=n^-1.01 (n=$N_NODES) lr=$ADAM_LR"

# ── A. Blind arm (r=0) first, same reasoning as ppi_matched_eps.sbatch: it is
#      the cheaper arm and the one a wallclock timeout must not skip.
echo "--- calibrating blind arm (r=0) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 $P1 --r 0 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_blind.txt"
cat "$OUT_ROOT/sigma_blind.txt"

grep -v '^#' "$OUT_ROOT/sigma_blind.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && continue
  run_cell "$OUT_ROOT/blind_eps${EPS}" \
      --p2 1.0 --r 0 --K_in $K --K_out $K --dp --sigma "$SIGMA" --lr $ADAM_LR
done

# ── B. GNN arm: r=1, sigma solved per (p2, eps) -- identical to the SGD run.
echo "--- calibrating GNN arm (r=1) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 $P1 --r 1 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_gnn.txt"
cat "$OUT_ROOT/sigma_gnn.txt"

grep -v '^#' "$OUT_ROOT/sigma_gnn.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/gnn_p2${P2}_eps${EPS}" \
      --p2 "$P2" --r 1 --K_in $K --K_out $K --dp --sigma "$SIGMA" --lr $ADAM_LR
done

echo
echo "=== PPI matched-epsilon (Adam) complete $(date) ==="
echo "compare against results/ppi_matched_eps/ (SGD, same sigmas, lr=0.3)"
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
