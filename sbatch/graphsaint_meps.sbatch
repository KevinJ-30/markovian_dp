#!/bin/bash
#SBATCH -Jmeps
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t8:00:00
#SBATCH -oReport-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── One matched-epsilon driver for every GraphSAINT inductive graph ──
#
#   DS=amazon EPOCHS=10 BATCH=512 OUT_ROOT=results/x sbatch --export=ALL $0
#
# GPU: sbatch flags OVERRIDE the #SBATCH lines above, so there is no separate
# GPU copy of this file -- ask for one at submit time:
#
#   DS=amazon ... sbatch -p ice-gpu --gres=gpu:H200:1 --export=ALL $0
#
# src/experiments/sparse.py selects cuda whenever it is visible, so nothing
# needs a device flag.  Do NOT add --mem-per-gpu: Slurm rejects it alongside
# the --mem above.
#
# WHAT A GPU ACTUALLY BUYS.  The DP cells use the padded-batch path
# (PaddedGNNStack + per-sample gradients) and speed up.  The non-DP ceilings
# run one forward per root through the unbatched path and stay CPU-bound, so a
# ceiling-only job gains little.
#
# Every knob below is overridable; see scripts/_matched_eps_grid.sh for the
# rest (EPOCHS, BATCH, GRID, EPS_LIST, CELLS, NODP, NODP_R, NODP_P2, NODP_K,
# TRACK_EVERY, SEEDS).

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

# NTRAIN is the training-node count the sampling rate is defined against:
# p1 = BATCH / NTRAIN.  Wrong NTRAIN silently mis-prices epsilon, so these are
# measured from the released files rather than the GraphSAINT paper's Table 1
# (they disagree for Yelp and Amazon -- see the note in the git history).
# HIDDEN/DROPOUT follow GraphSAINT Appendix C.3.
DS=${DS:?set DS (amazon|yelp|ppi-large|saint-reddit|saint-flickr)}
case $DS in
  amazon)        : "${NTRAIN:=1255968}"; : "${HIDDEN:=512}"; : "${DROPOUT:=0.1}"
                 : "${GNN_MODEL:=multilabel_gnn}" ;;
  yelp)          : "${NTRAIN:=537635}";  : "${HIDDEN:=512}"; : "${DROPOUT:=0.1}"
                 : "${GNN_MODEL:=multilabel_gnn}" ;;
  ppi-large)     : "${NTRAIN:=44904}";   : "${HIDDEN:=512}"; : "${DROPOUT:=0.0}"
                 : "${GNN_MODEL:=multilabel_gnn}" ;;
  saint-reddit)  : "${NTRAIN:=153939}";  : "${HIDDEN:=128}"; : "${DROPOUT:=0.1}"
                 : "${GNN_MODEL:=gnn}" ;;          # single-label, 41 communities
  saint-flickr)  : "${NTRAIN:?saint-flickr NTRAIN is not pinned yet -- pass it}"
                 : "${HIDDEN:=256}"; : "${DROPOUT:=0.2}"
                 : "${GNN_MODEL:=gnn}" ;;          # single-label, 7 classes
  *) echo "FATAL: unknown DS=$DS" >&2; exit 1 ;;
esac
export NTRAIN HIDDEN DROPOUT GNN_MODEL

OUT_ROOT=${OUT_ROOT:-results/${DS}_meps}

echo "=== $DS  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "  (no GPU requested/visible; running on CPU)"
$PY -c "import torch; print(f'  torch cuda={torch.cuda.is_available()}')"

source scripts/_matched_eps_grid.sh
