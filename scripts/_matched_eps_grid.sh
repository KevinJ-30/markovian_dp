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
# Schedule is specified in EPOCHS, not steps.  A fixed T is a different amount
# of data on every dataset -- at the old T=500 with B=512, PPI-large saw 5.7
# passes over its training nodes and Amazon 0.20, a 28x difference that reads
# as a utility difference.  T is derived below so every dataset trains equally.
EPOCHS=${EPOCHS:-10}
K=${K:-5}
SEEDS=${SEEDS:-1}
DELTA=${DELTA:-1e-6}
EPS_LIST=${EPS_LIST:-"2 8"}

# ACCOUNTANT DISCRETIZATION.  `grid` is dp_accounting's value_discretization_
# interval: the PLD lives on a lattice of this spacing and rounds PESSIMISTICALLY
# to stay an upper bound, so the error accumulates under composition at roughly
# T*grid.  That is safe (epsilon is never under-reported) but it is paid for in
# noise, and the overpayment is NOT uniform -- it is a fixed ADDITIVE epsilon, so
# it eats a far larger share of an eps=2 budget than an eps=8 one.
#
# The old default of 1e-4 was correct when T was a hardcoded 500 (floor 0.05).
# It is not correct now that T = epochs/p1 and p1 = BATCH/NTRAIN, because T grows
# with the graph -- which means the penalty falls hardest on exactly the large
# graphs this suite exists to measure.  Measured on amazon at 10 epochs, eps=2:
#
#     grid=1e-4  floor 2.45  ->  sigma 6.996   (floor EXCEEDS the eps=2 target)
#     grid=1e-6  floor 0.025 ->  sigma 2.963   (2.4x less noise, same guarantee)
#
# 1e-6 holds the floor under ~2.5% of an eps=2 budget through ~20 epochs and
# costs 80-550s per calibration call, which is a one-time per-cell cost.  Beyond
# ~30 epochs it starts to bind again and wants to go finer, but calibration time
# grows as 1/grid, so check `calibrate_grid.py`'s stderr warning rather than
# assuming.  That warning fires whenever T*grid > min(eps)/10; it goes to the
# slurm log, and it is not decoration.
GRID=${GRID:-1e-6}

# ONE EVALUATION PER EPOCH.  1/p1 = NTRAIN/BATCH steps is one pass over the root
# pool, so this tracks once per epoch on every dataset.  A CONSTANT here is a
# different number of evaluations at every T: at the old fixed 50, amazon at 10
# epochs would run 490 full-graph evaluations over a 262.8M-arc graph, which is
# the dominant cost of the job and the thing the sbatch headers warn about.
# summarize_sweep.py then reports both the best tracked checkpoint (selected on
# VALIDATION) and the final one; the final is the matched-epsilon number, since
# every cell reaches step T at its calibrated target.
TRACK_EVERY=${TRACK_EVERY:-$(( NTRAIN / BATCH ))}

P1=$($PY -c "print(f'{$BATCH/$NTRAIN:.8f}')")
# T = epochs/p1. Derived here rather than inside run.py because calibrate_grid.py
# solves sigma out of process and must be given the SAME T; one derivation, so
# the two cannot disagree.
T=${T:-$($PY -c "print(max(1, round($EPOCHS/$P1)))")}

mkdir -p "$OUT_ROOT"
echo "=== $DS grid  $(date) ==="
echo "    N_train=$NTRAIN batch=$BATCH -> p1=$P1"
echo "    epochs=$EPOCHS -> T=$T"
echo "    K=$K hidden=$HIDDEN dropout=$DROPOUT seeds=$SEEDS delta=$DELTA"
echo "    multilabel -> micro-F1 primary, micro-AUROC alongside"

COMMON="--dataset $DS --inductive --p1 $P1 --expect_pool_size $NTRAIN
        --hidden $HIDDEN --clip 1.0
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

# GNN cells as K:p2:r.  r IS PER-CELL -- the previous grid hardcoded --r 1 for
# every cell, which is why its GNN arms all lost: measured non-privately on
# PPI-large at K=25, 34 epochs, r=1 sits 8 points BELOW the graph-blind MLP
# (0.4542 vs 0.5330) and does not improve with more steps, while r=2 beats it
# by 29 points (0.8227).  One hop is not enough on these graphs.
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
# Selectable because nodp_gnn is the most expensive cell in the whole grid: it
# is the only one at r=2 with p2=1.0, so on a high-degree graph its rooted
# subgraphs reach 1 + 2K + (2K)^2 nodes (1,275 at K=25) over the full arc set.
# On Amazon (262.8M arcs) that dominates the job, and it is worth skipping when
# the question is DP-MLP vs SparseExpand rather than "what is the graph worth".
#   NODP="mlp"       just the graph-blind ceiling
#   NODP=""          neither
NODP=${NODP:-"gnn mlp"}
for _c in $NODP; do
  case $_c in
    gnn) run_cell "$OUT_ROOT/nodp_gnn" --model multilabel_gnn --aggr mean \
             --p2 1.0 --r 2 --num_layers 2 --K_in 25 --K_out 25 ;;
    mlp) run_cell "$OUT_ROOT/nodp_mlp" --model mlp --p2 1.0 --r 0 \
             --num_layers 2 --K_in 5 --K_out 5 ;;
    *)   echo "  [warn] unknown NODP entry '$_c' (expected gnn or mlp)" >&2 ;;
  esac
done

# ── DP-MLP blind arm: r=0, so K is irrelevant to its accounting ──
# EPS_LIST="" runs no DP arms at all, which is how a job asks for the non-DP
# ceilings alone.  Guarded because `--eps` with no value is an argparse error,
# and set -u would otherwise leave a stale empty sigma_r0.txt behind.
if [ -n "${EPS_LIST// /}" ]; then
echo "--- calibrating blind arm (r=0) ---"
$PY scripts/calibrate_grid.py --eps $EPS_LIST --p2 1.0 --p1 $P1 --r 0 \
    --K 5 --T $T --grid $GRID --delta $DELTA > "$OUT_ROOT/sigma_r0.txt"
cat "$OUT_ROOT/sigma_r0.txt"
grep -v '^#' "$OUT_ROOT/sigma_r0.txt" | while read -r P2 EPS SG; do
  [ "$SG" = "SKIP" ] && continue
  run_cell "$OUT_ROOT/dpmlp_eps${EPS}" --model mlp --p2 1.0 --r 0 \
      --num_layers 2 --K_in 5 --K_out 5 --dp --sigma "$SG"
done
else
  echo "--- EPS_LIST empty: skipping all DP arms ---"
fi

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
    run_cell "$OUT_ROOT/gnn_K${K}_p2${PP}_r${R}_eps${EPS}" --model multilabel_gnn \
        --aggr mean --p2 "$PP" --r "$R" --num_layers 2 \
        --K_in "$K" --K_out "$K" --dp --sigma "$SG"
  done
done

echo
echo "=== $DS complete $(date) ==="
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
