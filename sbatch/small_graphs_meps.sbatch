#!/bin/bash
#SBATCH -Jsmall_meps
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t8:00:00
#SBATCH -oReport-small-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── PPI and Flickr: DP-MLP vs SparseExpand at matched epsilon ──
#
#   DS=ppi          sbatch sbatch/small_graphs_meps.sbatch
#   DS=saint-flickr sbatch sbatch/small_graphs_meps.sbatch
#
# WHY r=1 AND NOT r=2.  These graphs are SMALL, so p1 = B/N_train is large
# (0.0029 at B=128 against yelp's 0.00095), and epsilon is paid per step at that
# rate.  Measured on PPI at B=128, 10 epochs, delta=1e-6:
#
#     arm                          sigma @ eps=2   sigma @ eps=8   run here?
#     blind (r=0)                           1.77            1.08   yes
#     K=5  p2=0.5  r=1  (~3.5 nodes)        9.25            2.74   yes
#     K=25 p2=0.1  r=1  (~3.5 nodes)        9.25            2.74   yes
#     K=5  p2=1.0  r=1  (~6   nodes)       16.97            4.92   no
#     K=25 p2=0.5  r=1  (~13  nodes)       40.09           11.62   no
#     K=5  p2=0.5  r=2  (~10  nodes)       73.94           21.44   no
#
# r=2 costs 8x the noise of r=1 at the same K and p2, because the shells go as
# K_out^d while path retention only decays as p2^d.  sigma=74 is not a training
# run, so the r=2 and K=25 rows are recorded above as a RESULT rather than
# spent as cluster time.  This is the small-graph mirror of the paper's claim:
# big graphs buy a neighbourhood cheaply, small ones cannot afford one.
#
# WHY BATCH 128.  At fixed epochs, halving the batch halves p1, and epsilon
# falls faster than the extra compositions add it back -- measured 29% less
# sigma at B=128 than B=256, for 2x the steps on graphs this small.
#
# AT r=1, sigma IS A FUNCTION OF K*p2 ALONE.  The mixture is
# Binomial(2K, p1*p2), and at p1~0.003 that is deep in the Poisson limit where
# only the product matters.  Verified: K=5/p2=0.5 and K=25/p2=0.1 both give
# sigma = 9.250000 at eps=2 and 2.736328 at eps=8, to every printed digit.
#
# The two DP-GNN cells below exploit that to run a CONTROLLED ABLATION.  Both
# have K*p2 = 2.5, so they cost the SAME epsilon, carry the SAME noise, and
# retain the SAME expected number of neighbours.  They differ only in how those
# 2.5 neighbours are chosen:
#
#     K=5  p2=0.5   cap hard to 5 candidates, then keep half
#     K=25 p2=0.1   cap loosely to 25 candidates, then keep a tenth
#
# PPI's mean degree is ~14, so K=25 is effectively uncapped and the second cell
# is "sparsify the real neighbourhood" against the first's "truncate it, then
# sparsify gently".  The paper argues p2 is the cheap knob and K the expensive
# one; at r=1 they are interchangeable in epsilon, so this asks whether they are
# also interchangeable in UTILITY.  Any gap between these two cells is a
# property of the sampling mechanism, not of the privacy budget.
#
# SIGMA IS SOLVED IN-PROCESS by --target_epsilon, against the p1 and T that
# run.py itself derives from --batch_size and --epochs.  That is deliberate:
# the external calibrate_grid.py path needs N_train passed through the shell,
# and a stale value there silently miscalibrates every cell.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

DS=${DS:-ppi}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-10}
HIDDEN=${HIDDEN:-256}
SEEDS=${SEEDS:-1}
DELTA=${DELTA:-1e-6}
# T*grid is the accountant's discretization floor.  T ~ 3500 here, so 1e-5 puts
# it at 0.035 -- under 2% of the eps=2 budget.  1e-6 is NOT affordable at r=2:
# it failed to calibrate a single cell in 10 minutes.
GRID=${GRID:-1e-5}
EPS_LIST=${EPS_LIST:-"2 8"}

case $DS in
  ppi)
    # 24 disjoint graphs split 20/2/2 -> natively inductive, so no --inductive.
    # 121-way multilabel: BCE + micro-F1, reported alongside micro-AUROC.
    GNN_MODEL="multilabel_gnn"; INDUCTIVE=(); DROPOUT=0.0 ;;
  saint-flickr)
    # 7-way single-label: cross-entropy + accuracy.  GNNMechanism.evaluate
    # reports accuracy only -- AUROC is not defined for its head.
    GNN_MODEL="gnn"; INDUCTIVE=(--inductive); DROPOUT=0.2 ;;
  *)
    echo "FATAL: unknown DS=$DS (expected ppi or saint-flickr)" >&2; exit 1 ;;
esac

OUT_ROOT=${OUT_ROOT:-results/${DS}_small_meps}
mkdir -p "$OUT_ROOT"

echo "=== $DS matched-epsilon $(date) ==="
echo "    batch=$BATCH epochs=$EPOCHS hidden=$HIDDEN seeds=$SEEDS"
echo "    delta=$DELTA grid=$GRID eps=$EPS_LIST model=$GNN_MODEL"

COMMON=(--dataset "$DS" "${INDUCTIVE[@]}" --direction in
        --batch_size "$BATCH" --epochs "$EPOCHS"
        --num_layers 2 --hidden "$HIDDEN" --clip 1.0
        --dropout "$DROPOUT" --weight_decay 0.0 --roots_from train
        --optimizer adam --lr 0.01 --seeds "$SEEDS"
        --accounting_grid "$GRID" --track_every 100)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $(basename "$out")"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $(basename "$out")  $*"
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out"
}

# ── non-DP ceilings.  No clip, no noise, no calibration -- the cheapest runs
#    here, and without them the DP numbers have nothing to be read against.
#    The GNN ceiling is UNCAPPED on purpose: it answers "what is the graph
#    worth at all", so it must not be handicapped by the degree cap that only
#    exists to price epsilon.
run_cell "$OUT_ROOT/nodp_gnn" --model "$GNN_MODEL" --aggr mean --p2 1.0 --r 2
run_cell "$OUT_ROOT/nodp_mlp" --model mlp --p2 1.0 --r 0

# ── DP arms.  --target_epsilon makes run.py solve sigma per cell.
for EPS in $EPS_LIST; do
  DP=(--dp --target_epsilon "$EPS" --target_delta "$DELTA")

  # Graph-blind baseline: --model mlp, NOT --model gnn --r 0, which would still
  # message-pass at evaluation with untrained neighbour weights.
  run_cell "$OUT_ROOT/dpmlp_eps${EPS}" --model mlp --p2 1.0 --r 0 \
      --K_in 5 --K_out 5 "${DP[@]}"

  # SparseExpand, one hop.  MATCHED PAIR: both have K*p2 = 2.5, so both get the
  # same solved sigma and the same expected neighbour count.  Only the route to
  # it differs -- hard cap vs aggressive sparsification.  See the header.
  run_cell "$OUT_ROOT/gnn_K5_p20.5_r1_eps${EPS}" --model "$GNN_MODEL" --aggr mean \
      --p2 0.5 --r 1 --K_in 5 --K_out 5 "${DP[@]}"

  run_cell "$OUT_ROOT/gnn_K25_p20.1_r1_eps${EPS}" --model "$GNN_MODEL" --aggr mean \
      --p2 0.1 --r 1 --K_in 25 --K_out 25 "${DP[@]}"
done

echo
echo "=== $DS complete $(date) ==="
# ppi reports micro_f1 in the *_acc columns (multilabel); flickr reports
# accuracy there.  Both are the mechanism's own metric_name -- see the `metric`
# column in the CSV.
$PY scripts/summarize_sweep.py "$OUT_ROOT" --metric test_acc 2>/dev/null || true
