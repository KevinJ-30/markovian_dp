# Shared matched-epsilon grid for the large inductive node-classification
# graphs (Yelp, AmazonProducts, Reddit), sourced by the per-dataset
# sbatch wrappers.  Expects DS (dataset key), NTRAIN (train node count),
# HIDDEN, DROPOUT and OUT_ROOT to be set by the caller.
#
#   arms.  The axis swept here is K, not p2, because on these graphs K is the
#   BINDING constraint: measured on Yelp at K=5, p2=0.1, r=1 the mean rooted
#   subgraph is 1.20 nodes -- the root plus 0.2 neighbours -- so the GNN has
#   degenerated to an MLP before p2 does anything.  Yelp's mean degree is 18.5
#   and Amazon's ~167, so a K=5 cap discards almost everything.
#
#   These graphs can afford a real K.  sigma for eps=8 at r=1, p2=1:
#       Amazon  K=5 -> 1.11   K=25 -> 1.64   K=50 -> 2.50
#       Yelp    K=5 -> 1.32   K=25 -> 2.90   K=50 -> 5.64
#   (on PPI-large K=25 needed sigma=76, which is the scaling argument in one
#   line: big graphs buy a neighbourhood cheaply.)
#
#     DP-MLP blind (r=0)      the genuinely graph-blind baseline (--model mlp,
#                             NOT --model gnn --r 0, which still message-passes
#                             at evaluation with untrained neighbour weights)
#     GNN r=1 K=5   p2=1.0    the old PPI-style config, for continuity
#     GNN r=1 K=25  p2=1.0    a neighbourhood that actually exists
#     GNN r=1 K=25  p2=0.5    does sparsification buy back the larger K?
#     GNN r=1 K=50  p2=0.5    the K x p2 interaction at the far end
#     non-DP ceiling          same architecture, no clip/noise
#
# sigma is solved per cell by calibrate_grid.py against the UNION-SAFE
# accountant (n_d = 2*K_out^d), so these epsilons already carry the
# union-graph correction.

set -u

: "${DS:?set DS}"; : "${NTRAIN:?set NTRAIN}"; : "${OUT_ROOT:?set OUT_ROOT}"
HIDDEN=${HIDDEN:-512}
DROPOUT=${DROPOUT:-0.1}
BATCH=${BATCH:-512}
T=${T:-500}
K=${K:-5}
SEEDS=${SEEDS:-1}
DELTA=${DELTA:-1e-6}
GRID=${GRID:-1e-4}
EPS_LIST=${EPS_LIST:-"2 8"}
TRACK_EVERY=${TRACK_EVERY:-50}

P1=$($PY -c "print(f'{$BATCH/$NTRAIN:.8f}')")

mkdir -p "$OUT_ROOT"
echo "=== $DS grid  $(date) ==="
echo "    N_train=$NTRAIN batch=$BATCH -> p1=$P1"
echo "    T=$T K=$K hidden=$HIDDEN dropout=$DROPOUT seeds=$SEEDS delta=$DELTA"
echo "    multilabel -> micro-F1 primary, micro-AUROC alongside"

COMMON="--dataset $DS --inductive --p1 $P1 --hidden $HIDDEN --clip 1.0
        --dropout $DROPOUT --weight_decay 0.0 --roots_from train
        --optimizer adam --lr 0.01 --T $T --seeds $SEEDS
        --track_every $TRACK_EVERY"

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $(basename "$out")  $*"
  $PY -u -m src.sparse.run $COMMON "$@" --out_dir "$out"
}

# (K, p2) cells for the GNN arm, swept at r=1.  See the header for why K.
CELLS=${CELLS:-"5:1.0 25:1.0 25:0.5 50:0.5"}

# ── non-DP ceilings (no privacy constraint) ──
run_cell "$OUT_ROOT/nodp_gnn" --model multilabel_gnn --aggr mean --p2 1.0 --r 1 \
    --num_layers 2 --K_in 25 --K_out 25
run_cell "$OUT_ROOT/nodp_mlp" --model mlp --p2 1.0 --r 0 --num_layers 2 \
    --K_in 5 --K_out 5

# ── DP-MLP blind arm: r=0, so K is irrelevant to its accounting ──
echo "--- calibrating blind arm (r=0) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 $P1 --r 0 \
    --K 5 --T $T --grid $GRID --delta $DELTA > "$OUT_ROOT/sigma_r0.txt"
cat "$OUT_ROOT/sigma_r0.txt"
grep -v '^#' "$OUT_ROOT/sigma_r0.txt" | while read -r P2 EPS SG; do
  [ "$SG" = "SKIP" ] && continue
  run_cell "$OUT_ROOT/dpmlp_eps${EPS}" --model mlp --p2 1.0 --r 0 \
      --num_layers 2 --K_in 5 --K_out 5 --dp --sigma "$SG"
done

# ── DP-GNN arms: one calibration call per (K, p2) cell ──
for cell in $CELLS; do
  K=${cell%%:*}; P2=${cell##*:}
  echo "--- calibrating GNN r=1 K=$K p2=$P2 ---"
  $PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 "$P2" --p1 $P1 --r 1 \
      --K "$K" --T $T --grid $GRID --delta $DELTA \
      > "$OUT_ROOT/sigma_K${K}_p2${P2}.txt"
  cat "$OUT_ROOT/sigma_K${K}_p2${P2}.txt"
  grep -v '^#' "$OUT_ROOT/sigma_K${K}_p2${P2}.txt" | while read -r PP EPS SG; do
    [ "$SG" = "SKIP" ] && { echo "  [skip] K=$K p2=$PP eps=$EPS unreachable"; continue; }
    run_cell "$OUT_ROOT/gnn_K${K}_p2${PP}_eps${EPS}" --model multilabel_gnn \
        --aggr mean --p2 "$PP" --r 1 --num_layers 2 \
        --K_in "$K" --K_out "$K" --dp --sigma "$SG"
  done
done

echo
echo "=== $DS complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
