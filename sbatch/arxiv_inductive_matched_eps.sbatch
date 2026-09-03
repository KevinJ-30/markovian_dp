#!/bin/bash
#SBATCH -Jarx_meps
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t8:00:00
#SBATCH -oReport-arxiv-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── ogbn-arxiv (inductive): width fix, then DP-GNN vs DP-MLP at matched eps ──
#
#   sbatch sbatch/arxiv_inductive_matched_eps.sbatch
#
# WHY ARXIV.  It is the friendliest graph in the suite for this mechanism and
# the current numbers do not show it.  Degree stats (measured):
#
#     dataset    mean deg   median   edges kept at K=5
#     facebook       80.2       62                  6%
#     ppi            27.9       15                 16%
#     arxiv           6.9        4                 47%
#
# So the K=5 cap barely bites on arxiv, where it destroys 94% of facebook.  Yet
# arxiv's measured graph advantage is only +3.1 points (GNN r=1 p2=1.0 0.5185 vs
# graph-blind MLP 0.4876, results/arxiv/inductive_*).  Those runs used
# hidden=16 -- the same starvation that held PPI below its trivial baseline
# until Stage 1 raised width to 256 for +10 to +14 points at zero epsilon cost.
# Stage A tests whether that transfers; Stage B then runs the real comparison.
#
# INDUCTIVE here is a genuine graph cut, unlike PPI.  PPI's splits are already
# disconnected components, so --inductive is a no-op.  arxiv is one connected
# graph, so run.py keeps only arcs with BOTH endpoints in train (run.py:331-337)
# -- SparseExpand provably cannot reach a val/test node during training -- and
# evaluates on the full edge_index so held-out nodes keep real neighbourhoods.
#
# delta = n^-1.01 with n = 169,343 (full |V|) = 5.24e-6.
# p1 = 512/90,941 train nodes = 0.0056, the same batch-512 rule as PPI.
# grid=1e-5: floor ~T*grid = 0.005 at T=500, negligible against eps=1.

set -u
cd $SLURM_SUBMIT_DIR

source scripts/_ice_env.sh

OUT_ROOT=${OUT_ROOT:-results/arxiv_matched_eps}
N_NODES=169343
P1=0.0056
T=500
K=5
SEEDS=${SEEDS:-3}
GRID=1e-5
EPS_LIST="1 2 4 8"
P2_LIST="1.0 0.5 0.25 0.1"

mkdir -p "$OUT_ROOT" results/logs

BASE=(--dataset ogbn-arxiv --direction in --aggr mean --inductive
      --p1 $P1 --num_layers 2 --clip 1.0 --dropout 0.0 --weight_decay 0.0
      --roots_from train --seeds $SEEDS --T $T --track_every 25)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_ogbn-arxiv*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${BASE[@]}" "$@" --out_dir "$out"
}

echo "=== arxiv inductive matched-epsilon $(date) ==="

# ── A. Does the PPI width fix transfer?  Non-DP, so epsilon is not involved. ──
# Gate: hidden=256 should clearly beat hidden=16's 0.5185.  If it does not, the
# +3.1 point graph advantage is real rather than an artifact of starvation, and
# Stage B is measuring a genuinely thin margin -- worth knowing before spending
# the budget on it.
echo "--- A: width check (non-DP) ---"
for H in 16 256; do
  run_cell "$OUT_ROOT/nodp_h${H}" --hidden "$H" --K_in $K --K_out $K \
      --p2 1.0 --r 1 --optimizer adam --lr 0.01
done
run_cell "$OUT_ROOT/nodp_blind_h256" --hidden 256 --K_in $K --K_out $K \
    --p2 1.0 --r 0 --optimizer adam --lr 0.01

# ── B. matched-epsilon GNN vs blind, at the width A should confirm ───────────
echo "--- B: calibrating GNN arm (r=1) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 $P2_LIST --p1 $P1 --r 1 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_gnn.txt"
cat "$OUT_ROOT/sigma_gnn.txt"

grep -v '^#' "$OUT_ROOT/sigma_gnn.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && { echo "  [skip] p2=$P2 eps=$EPS unreachable"; continue; }
  run_cell "$OUT_ROOT/gnn_p2${P2}_eps${EPS}" --hidden 256 --K_in $K --K_out $K \
      --p2 "$P2" --r 1 --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo "--- B: calibrating blind arm (r=0) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 $P1 --r 0 \
    --K $K --T $T --grid $GRID --delta_from_n $N_NODES \
    > "$OUT_ROOT/sigma_blind.txt"
cat "$OUT_ROOT/sigma_blind.txt"

grep -v '^#' "$OUT_ROOT/sigma_blind.txt" | while read -r P2 EPS SIGMA; do
  [ "$SIGMA" = "SKIP" ] && continue
  run_cell "$OUT_ROOT/blind_eps${EPS}" --hidden 256 --K_in $K --K_out $K \
      --p2 1.0 --r 0 --dp --sigma "$SIGMA" --optimizer sgd --lr 0.3
done

echo
echo "=== arxiv matched-epsilon complete $(date) ==="
echo "prior hidden=16 refs: GNN r=1 0.5185 | blind r=0 0.4876"
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
