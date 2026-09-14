#!/bin/bash
#SBATCH -Jamz_meps
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t24:00:00
#SBATCH -oReport-amazon-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── AmazonProducts (GraphSAINT): matched-epsilon grid ──
#
#   sbatch sbatch/amazon_matched_eps.sbatch
#
# DATA.  Needs $GRAPHSAINT_DATA_ROOT/amazon/:
#     adj_full.npz  adj_train.npz  feats.npy  role.json  labels.npy
# (labels.npy is preferred over the 523 MB class_map.json -- same content, far
# faster to load.)  Extract with scripts/setup_graphsaint.sh; ~3.7 GB on disk.
#
# STATS (measured from the files):
#   1,569,960 nodes / 262,769,508 arcs / 200 feat / 107 labels (multilabel)
#   split .80/.05/.15 by node = 1,255,968 train.
#   NOTE the paper's Table 1 says 1,598,960 nodes and .85/.05/.10 -- the
#   released files disagree on both.  We report what we run on.
#
# WHY 180G.  This is the memory-critical job.  edge_index alone is
# 262.8M arcs x 2 x int64 = 4.2 GB, and train_edge_index is another ~3.4 GB.
# Add features (1.26 GB), labels (672 MB), the per-seed capped copy, and the
# CSR adjacency, and loading alone is ~15 GB.  The full-graph evaluation
# forward over 262.8M arcs x 200 features is the real spike -- it exceeds
# base_mechanism's 250M arc-feature threshold so it takes the CSR path, but
# that still materializes large intermediates.  180G is deliberately generous;
# if it lands comfortably, drop it and re-submit.
#
# WHY NOT GPU.  Expansion and the adjacency build are CPU-side, and the
# multilabel head runs one forward per root rather than a batched one, so the
# GPU would idle between tiny transfers.  Moving a 4.2 GB edge_index plus
# features onto a card for the eval forward is also tight.  Fix the per-root
# batching first; then a GPU is worth requesting for evaluation.
#
# IF IT OOMs.  In order of preference: raise --track_every (fewer full-graph
# evaluations), then drop --hidden to 256, then cut T.  Do not cut K or p2 to
# save memory -- those change the mechanism.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

DS=amazon
NTRAIN=1255968
HIDDEN=512          # GraphSAINT Appendix C.3 uses 512 for Amazon
DROPOUT=0.1         # ditto
OUT_ROOT=${OUT_ROOT:-results/amazon_matched_eps}
TRACK_EVERY=100     # fewer full-graph evals than the default 50

source scripts/_matched_eps_grid.sh
