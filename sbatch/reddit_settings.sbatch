#!/bin/bash
#SBATCH -Jreddit_dp
#SBATCH -A gts-pli77                      # CHECK THIS — see note below
#SBATCH -q embers                         # free QoS.  inferno spends credit.
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                         # Reddit: 114.6M arcs before capping
#SBATCH -t4:00:00                         # allocation is billed, not usage —
                                          # request the minimum that fits
#SBATCH -oReport-reddit-dp-%j.out
#SBATCH --mail-type=END,FAIL
#
# ACCOUNT: the -A string above is a guess.  Confirm the real one with
#     sacctmgr show assoc user=$USER format=account,qos
# and fix it before submitting; a wrong account fails fast rather than
# charging, but -A <anything>-paid would be real money.
#
# CPU node on purpose.  This workload is launch-latency bound, not FLOP bound:
# ~96% of step time is a Python loop over per-root subgraphs of a few nodes
# each, which a GPU does not accelerate.  To try one anyway, add
#     #SBATCH --gres=gpu:L40S:1
# — naming the type matters, since the allocator defaults to V100.
#
# embers is PREEMPTIBLE.  Cells skip themselves when their output CSV exists,
# so resubmitting the same script resumes.  One setting per submission, so a
# preemption loses at most one arm:
#     sbatch --export=SETTING=transductive sbatch/reddit_settings.sbatch
#     sbatch --export=SETTING=inductive    sbatch/reddit_settings.sbatch
#
# transductive: full graph in training; test nodes reachable by expansion.  The
#   DP guarantee still covers every node (Theorem 6.4 substitutes any s), but
#   test accuracy no longer measures generalization to unseen nodes.  This is
#   the setting GAP/ProGAP report, so it is the comparable arm.
# inductive: train on the train-induced subgraph; expansion cannot reach
#   val/test.  Honest generalization, impoverished training graph.

set -e
cd $SLURM_SUBMIT_DIR

PROJECT=/storage/project/r-pli77-0/kjacob7
SCRATCH=${SCRATCH:-$HOME/scratch}

SETTING=${SETTING:-transductive}
FLAG=""
[ "$SETTING" = "inductive" ] && FLAG="--inductive"

# Scratch is wiped every 60 days.  Fine for the dataset (re-downloadable) and
# for the venv (rebuildable), but nothing here is the only copy of anything.
export REDDIT_DATA_ROOT=$SCRATCH/data/Reddit
source $SCRATCH/venvs/markovian/bin/activate
export OMP_NUM_THREADS=1

# Results go to project storage, which persists.
OUT_ROOT=$PROJECT/results/reddit

PY="python -u"
DELTA=1e-6
COMMON="--dataset reddit --direction in --aggr mean --p1 0.01 --r 1 \
        --num_layers 2 --T 2000 --K_in 5 --K_out 5 --clip 1.0 \
        --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 \
        --track_every 50"

run_cell() {   # out_dir, then extra flags
  local OUT=$1; shift
  if [ -s "$OUT/sparse_gnn_reddit_dp_results.csv" ] || \
     [ -s "$OUT/sparse_gnn_reddit_results.csv" ]; then
    echo "  [skip] $OUT already complete"; return 0
  fi
  $PY -m src.sparse.run $COMMON $FLAG "$@" --out_dir $OUT
}

echo "=== $SETTING: non-DP reference ==="
run_cell $OUT_ROOT/${SETTING}_nodp --p2 1.0 --lr 0.01

for P2 in 1.0 0.1; do
  for SIGMA in 5.0 20.0; do
    OUT=$OUT_ROOT/${SETTING}_p${P2}_s${SIGMA}
    echo "=== $SETTING DP: p2=$P2 sigma=$SIGMA ==="
    run_cell $OUT --dp --p2 $P2 --sigma $SIGMA --lr 0.3
    $PY -m src.sparse.compute_epsilon \
        --csv $OUT/sparse_gnn_reddit_dp_results.csv --delta $DELTA || true
  done
done

echo "=== REDDIT $SETTING COMPLETE ==="
