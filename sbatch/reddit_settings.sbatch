#!/bin/bash
#SBATCH -Jreddit_dp
#SBATCH -A gts-smussmann3                 # free account; -paid actually costs
#SBATCH -q embers                         # free QoS.  inferno spends the $70/mo
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                         # Reddit: 114.6M arcs before capping
#SBATCH -t4:00:00                         # allocation is billed, not usage —
                                          # request the minimum that fits
#SBATCH -oReport-reddit-dp-%j.out
#SBATCH --mail-type=END,FAIL
#
# CPU node on purpose.  This workload is launch-latency bound, not FLOP bound:
# ~96% of step time goes to a Python loop over per-root subgraphs averaging a
# few nodes each, which a GPU does not accelerate.  Spending L40S hours here
# would cost more and run no faster.  If you want to try a GPU anyway, add:
#     #SBATCH --gres=gpu:L40S:1
# (the allocator defaults to V100 unless the type is named).
#
# embers is PREEMPTIBLE.  Runs flush a CSV row per seed and skip cells whose
# output already exists, so a killed job is resumed by resubmitting the same
# script.  Keep the walltime short for the same reason.
#
# ONE SETTING PER SUBMISSION so a preemption loses at most one arm:
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

SETTING=${SETTING:-transductive}
FLAG=""
[ "$SETTING" = "inductive" ] && FLAG="--inductive"

# Dataset is ~2 GB, under the 10 GB shared-folder limit, and persists there —
# scratch is wiped every 60 days and would need re-downloading.
export REDDIT_DATA_ROOT=/storage/project/r-smussmann3-0/shared/datasets/Reddit
# Use the shared venv; do not create new ones (they are large).
source /storage/project/r-smussmann3-0/shared/shared_venvs/default_venv/bin/activate
export OMP_NUM_THREADS=1                  # per-process; cells are run serially
                                          # here but this avoids thread thrash

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
run_cell results/reddit/${SETTING}_nodp --p2 1.0 --lr 0.01

for P2 in 1.0 0.1; do
  for SIGMA in 5.0 20.0; do
    OUT=results/reddit/${SETTING}_p${P2}_s${SIGMA}
    echo "=== $SETTING DP: p2=$P2 sigma=$SIGMA ==="
    run_cell $OUT --dp --p2 $P2 --sigma $SIGMA --lr 0.3
    $PY -m src.sparse.compute_epsilon \
        --csv $OUT/sparse_gnn_reddit_dp_results.csv --delta $DELTA || true
  done
done

chmod -R 777 results/reddit 2>/dev/null || true
echo "=== REDDIT $SETTING COMPLETE ==="
