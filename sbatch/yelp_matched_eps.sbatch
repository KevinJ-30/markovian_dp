#!/bin/bash
#SBATCH -Jyelp_meps
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t8:00:00
#SBATCH -oReport-yelp-meps-%j.out
#SBATCH --mail-type=END,FAIL
#
# ── Yelp (GraphSAINT): matched-epsilon grid ──
#
#   sbatch sbatch/yelp_matched_eps.sbatch
#
# DATA.  Needs the GraphSAINT release files at $GRAPHSAINT_DATA_ROOT/yelp/:
#     adj_full.npz  adj_train.npz  feats.npy  role.json  class_map.json
# Extract with scripts/setup_graphsaint.sh; ~2.2 GB on disk.  NOT in git.
#
# STATS (measured from the files, not the paper -- they disagree):
#   716,847 nodes / 13,237,972 arcs / 300 feat / 100 labels (multilabel)
#   split .75/.15/.10 by node = 537,635 train.  NOTE the paper's Table 1 says
#   .75/.10/.15; the released role.json has val and test the other way round.
#
# RESOURCES.  CPU, not GPU: SparseExpand and the CSR adjacency build run on CPU,
# and the multilabel head does one forward per root rather than a batched one,
# so a GPU would sit idle between tiny transfers.  Memory is dominated by the
# full-graph EVALUATION forward (13.2M arcs x 300 features), which is why 64G
# rather than the 32G the training itself needs.
#
# COST.  Training is ~500 steps x 512 roots of tiny (K=5) subgraphs -- cheap.
# The expensive part is evaluation at every --track_every 50, over the full
# 716k-node graph.  If this times out, raise track_every before raising -t.

set -u
cd $SLURM_SUBMIT_DIR
source scripts/_ice_env.sh

export GRAPHSAINT_DATA_ROOT=${GRAPHSAINT_DATA_ROOT:-/storage/ice1/8/1/kjacob7/graphsaint}

DS=yelp
NTRAIN=537635
HIDDEN=${HIDDEN:-512}      # GraphSAINT Appendix C.3 uses 512 for Yelp
DROPOUT=${DROPOUT:-0.1}    # ditto
OUT_ROOT=${OUT_ROOT:-results/yelp_matched_eps}

source scripts/_matched_eps_grid.sh
