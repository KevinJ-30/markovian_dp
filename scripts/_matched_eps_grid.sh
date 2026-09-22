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
GRID=${GRID:-1e-3}
EPS_LIST=${EPS_LIST:-"2 8"}
TRACK_EVERY=${TRACK_EVERY:-50}
# Depth/sparsification for the non-DP GNN ceiling.  Defaults are the UNCAPPED
# architecture ceiling (r=2, p2=1.0): what the model can do with no privacy
# constraint at all.  Set NODP_R/NODP_P2 to match the DP cells instead when the
# question is "what did the NOISE cost", as opposed to "what did privacy cost".
NODP_R=${NODP_R:-2}
NODP_P2=${NODP_P2:-1.0}
# Single-label graphs (saint-flickr, saint-reddit) need --model gnn; PPI, Yelp
# and Amazon are multi-hot and need multilabel_gnn -- a different loss and a
# different metric.  Wrong value is a shape error at best, a silently
# meaningless micro-F1 at worst.
GNN_MODEL=${GNN_MODEL:-multilabel_gnn}
# Degree cap for the non-DP ceiling.  NODP_K=none removes it entirely, which is
# what you want to know the real headroom: Amazon's mean degree is ~167, so the
# old hardcoded K=25 kept ~15% of each neighbourhood and the "ceiling" came in
# 16 points under GraphSAINT's published SAGE number for the same graph.
NODP_K=${NODP_K:-25}
if [ "$NODP_K" = "none" ]; then NODP_CAP=""
else NODP_CAP="--K_in $NODP_K --K_out $NODP_K"; fi

P1=$($PY -c "print(f'{$BATCH/$NTRAIN:.8f}')")

# Prefer EPOCHS to a raw T: a fixed T is a different amount of training on every
# graph (T=500 at B=512 is 0.2 epochs on Amazon and 25 on rel-avito), so cells
# are not comparable across datasets.  T = epochs / p1.  Raw T still works if
# EPOCHS is unset.
if [ -n "${EPOCHS:-}" ]; then
  T=$($PY -c "print(max(1, round($EPOCHS / $P1)))")
  echo "    EPOCHS=$EPOCHS at p1=$P1 -> T=$T"
fi
# The PLD floor is ~T*grid, so the usable grid depends on T. The 1e-3
# default is faster and less memory-intensive; set GRID explicitly when a
# smaller discretization floor is required.
if [ "$T" -gt 3000 ] && [ "$GRID" != "1e-3" ]; then
  echo "    [warn] T=$T with grid=$GRID: fine grids at large T OOM the" >&2
  echo "           accountant.  Consider raising BATCH to cut T." >&2
fi

mkdir -p "$OUT_ROOT"
echo "=== $DS grid  $(date) ==="
echo "    N_train=$NTRAIN batch=$BATCH -> p1=$P1"
echo "    T=$T K=$K hidden=$HIDDEN dropout=$DROPOUT seeds=$SEEDS delta=$DELTA"
echo "    multilabel -> micro-F1 primary, micro-AUROC alongside"

# C does not scale the SNR (noise is sigma*C, signal is clipped at C) but it
# does set the clipping bias.  Never swept.
CLIP=${CLIP:-1.0}
COMMON="--dataset $DS --p1 $P1 --hidden $HIDDEN --clip $CLIP
        --dropout $DROPOUT --weight_decay 0.0 
        --optimizer adam --lr 0.01 --T $T --seeds $SEEDS
        --track_every $TRACK_EVERY"

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $(basename "$out")  $*"
  $PY -u -m src.experiments.sparse $COMMON "$@" --out_dir "$out"
}

# GNN cells as K:p2:r. r is per-cell; one hop did not converge to useful
# utility on the measured large graphs, while r=2 did.
#
# r=2 is what costs K_out^2 in the shells, and p2 is what buys it back --
# measured sigma for eps=8 at r=2, T=3000:
#     PPI-large  K=5  p2=1.0 -> 102.25   p2=0.1 ->  8.20   (12.5x)
#     Amazon     K=5  p2=1.0 ->   3.67   p2=0.1 ->  1.07
#     Amazon     K=10 p2=1.0 ->  13.23   p2=0.1 ->  1.68
# and at p2=0.1 larger K stays reachable on Amazon: K=15 -> 4.25, K=20 -> 9.41.
# (K=25 at p2=1.0 is NOT reachable -- the PLD grid blows up with the component
# count when sigma is large; aggressive p2 keeps the mixture concentrated.)
#
# Full sigma table for eps=8, r=2, T=3000 (the default cells below):
#              K=5            K=10           K=15
#   p2=0.1   yelp 1.28      yelp  3.36     yelp  9.90
#            amz  1.07      amz   1.68     amz   4.25
#   p2=0.5   yelp 6.71      yelp 29.28     yelp 66.50
#            amz  2.92      amz  12.52     amz  28.45
#   p2=1.0   yelp 8.53          --             --
#            amz  3.67
#
# Note how the K x p2 INTERACTION behaves: at K=5 dropping p2 from 0.5 to 0.1
# saves 2.7x on Amazon, but at K=10 it saves 7.5x and at K=15 6.7x.  The larger
# the degree cap, the more sparsification is worth -- which is the mechanism
# claim, and it is only legible at r=2.
# which is the composite-subsampling claim, only visible at r=2.
CELLS=${CELLS:-"5:0.1:2 5:0.5:2 5:1.0:2 10:0.1:2 10:0.5:2 15:0.1:2 15:0.5:2"}

# ── non-DP ceilings (no privacy constraint) ──
#
# NODP selects whether this invocation runs the non-private GNN ceiling; set it
# to "none" to skip it. Splitting the grid across separate sbatch jobs otherwise
# makes every job recompute the same ceiling: run_cell's skip-if-CSV-exists only
# helps once a run has finished, so concurrent jobs can race on one output.
# Run the ceiling once in its own job, then pass NODP=none to the cell jobs.
#
# NOTE ${VAR:-default} substitutes on empty as well as unset, so NODP="" gets
# the default rather than nothing -- hence the explicit "none" sentinel.
NODP=${NODP:-"gnn"}
for _arm in $NODP; do
  case $_arm in
    gnn)  run_cell "$OUT_ROOT/nodp_gnn" --model "$GNN_MODEL" --aggr mean \
              --p2 "$NODP_P2" --r "$NODP_R" --num_layers 2 $NODP_CAP ;;
    none) echo "  [skip] non-DP ceiling (NODP=none)" ;;
    *)    echo "  [warn] unknown NODP arm '$_arm' (expected gnn or none)" >&2 ;;
  esac
done

# ── DP-GNN arms: one calibration call per (K, p2) cell ──
for cell in $CELLS; do
  K=$(echo "$cell" | cut -d: -f1)
  P2=$(echo "$cell" | cut -d: -f2)
  R=$(echo "$cell" | cut -d: -f3)
  echo "--- calibrating GNN r=$R K=$K p2=$P2 ---"
  $PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 "$P2" --p1 $P1 --r "$R" \
      --K "$K" --T $T --grid $GRID --delta $DELTA \
      > "$OUT_ROOT/sigma_K${K}_p2${P2}_r${R}.txt"
  cat "$OUT_ROOT/sigma_K${K}_p2${P2}_r${R}.txt"
  grep -v '^#' "$OUT_ROOT/sigma_K${K}_p2${P2}_r${R}.txt" | while read -r PP EPS SG; do
    [ "$SG" = "SKIP" ] && { echo "  [skip] K=$K p2=$PP r=$R eps=$EPS unreachable"; continue; }
    run_cell "$OUT_ROOT/gnn_K${K}_p2${PP}_r${R}_eps${EPS}" --model "$GNN_MODEL" \
        --aggr mean --p2 "$PP" --r "$R" --num_layers 2 \
        --K_in "$K" --K_out "$K" --dp --sigma "$SG"
  done
done

echo
echo "=== $DS complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
