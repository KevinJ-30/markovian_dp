#!/usr/bin/env bash
# Facebook: does the PPI width fix transfer, and does it change the headline?
#
#   nohup caffeinate -i bash scripts/_facebook_width.sh \
#       > results/logs/fb_width.log 2>&1 &
#
# WHY.  The facebook coverage sweep -- the source of the matched-epsilon
# sparsification result, which is currently the strongest thing in the paper --
# ran at `--hidden 16` (scripts/_coverage_sweep.sh).  On PPI that exact setting
# held every DP cell below the trivial baseline across 47 configurations and
# eps from 0.65 to 464; raising width to 256 moved non-DP from 0.5245 to 0.6256
# at K=5/r=1, +10 points, at ZERO epsilon cost (the accountant never sees width,
# and per-root clipping to C fixes sensitivity regardless of it).
#
# If facebook is similarly starved, then the sparse-vs-dense gaps measured at
# hidden=16 --
#
#     eps    sparse(p2=0.1)  dense(p2=1.0)   delta
#     1.56          0.4790         0.4100    +6.9
#     0.75          0.4296         0.2993   +13.0
#     0.37          0.3643         0.2282   +13.6
#
# -- were measured on an undercapacity model, and the comparison has to be
# redone before it goes in a figure.  Stage A answers that for ~1 hour of CPU.
#
# Stage B then redoes the frontier properly at the winning width: sigma is
# SOLVED per (p2, target eps) instead of swept, so every cell sits at the same
# epsilon and a utility difference is attributable to p2 alone.
#
# delta = n^-1.01 with n = 26,406 (full |V|) = 3.42e-5.  The old coverage CSVs
# used 1e-6, so epsilons here are NOT comparable to those without re-emission.
#
# grid=1e-5: pessimistic rounding accumulates over composition, so the numerical
# floor is ~T*grid.  At T=500 that is 0.005, negligible against a 0.5 target.
# The old sweep's grid=1e-4 put the floor at 0.05, which is why its eps=0.19
# points are not trustworthy.

set -u
cd "$(dirname "$0")/.."

PY=${PY:-/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python}
OUT_ROOT=${OUT_ROOT:-results/facebook_width}
N_NODES=26406
P1=0.013          # batch 258 / 19,808 train nodes; unchanged from the sweep
T=500
K=5
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="0.5 1 2 4 8"
P2_LIST="1.0 0.5 0.25 0.1"

mkdir -p "$OUT_ROOT" results/logs

BASE=(--dataset facebook --direction in --aggr mean
      --p1 $P1 --num_layers 2 --clip 1.0 --dropout 0.0 --weight_decay 0.0
      --roots_from train --seeds "$SEEDS" --T $T --K_in $K --K_out $K
      --track_every 25)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_facebook*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${BASE[@]}" "$@" --out_dir "$out"
}

echo "=== facebook width check $(date) ==="
echo "    trivial baseline 0.2206 | hidden=16 non-DP refs: r1 0.6036, r2 0.6423"

# ── A. Non-DP width sweep.  epsilon is not involved, so this isolates capacity.
#      --p2 sweeps internally; r does not, hence the outer loop.
echo "--- A: non-DP width sweep ---"
for H in 16 64 256; do
  for R in 1 2; do
    run_cell "$OUT_ROOT/nodp_h${H}_r${R}" --hidden "$H" --r "$R" \
        --p2 1.0 0.1 --optimizer adam --lr 0.01
  done
done

# ── B. Matched-epsilon frontier at hidden=256, r=1.
#      Gate: only worth running if A shows 256 beating 16.  Run it regardless --
#      a null result at A makes B the confirmation that the published numbers
#      stand, which is equally worth having.
echo "--- B: calibrating sigma per (p2, eps) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 $P1 --r 1 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma.txt" || { echo "calibration failed"; exit 1; }
cat "$OUT_ROOT/sigma.txt"

grep -v '^#' "$OUT_ROOT/sigma.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/dp_h256_p2${P2}_eps${EPS}" --hidden 256 --r 1 \
      --p2 "$P2" --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo
echo "=== facebook width check complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
