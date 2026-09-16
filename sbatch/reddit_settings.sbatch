#!/bin/bash
#SBATCH -Jreddit_dp
#SBATCH -A gts-pli77                      # free account (-paid costs money)
#SBATCH -q embers                         # free QoS.  inferno spends credit.
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                         # Reddit: 114.6M arcs before capping
#SBATCH -t6:00:00                         # allocation is billed, not usage —
                                          # request the minimum that fits
#SBATCH -oReport-reddit-dp-%j.out
#SBATCH --mail-type=END,FAIL
#
# ACCOUNT: gts-pli77 is confirmed present for this user.  Never swap in
# gts-pli77-paid — that one bills real money.  gts-pli77-ideas_l40s holds the
# L40S allocation, which this job deliberately does not use.
#
# CPU node on purpose.  This workload is launch-latency bound, not FLOP bound:
# ~96% of step time is a Python loop over per-root subgraphs of a few nodes
# each, which a GPU does not accelerate.  To try one anyway, add
#     #SBATCH --gres=gpu:L40S:1
# — naming the type matters, since the allocator defaults to V100.
#
# embers is PREEMPTIBLE. Cells skip themselves when their output CSV exists,
# so resubmitting the same script resumes. SparseGNN always trains on the
# train-induced graph and evaluates on the separate test graph.

set -e
cd $SLURM_SUBMIT_DIR

PROJECT=/storage/project/r-pli77-0/kjacob7
SCRATCH=${SCRATCH:-$HOME/scratch}


# Scratch is wiped every 60 days.  Fine for the dataset (re-downloadable) and
# for the venv (rebuildable), but nothing here is the only copy of anything.
export REDDIT_DATA_ROOT=$SCRATCH/data/Reddit
source $SCRATCH/venvs/markovian/bin/activate
export OMP_NUM_THREADS=1

# Results go to project storage, which persists.
OUT_ROOT=$PROJECT/results/reddit/inductive

PY="python -u"
DELTA=1e-6
COMMON="--dataset reddit --direction in --aggr mean --p1 0.01 --r 1 \
        --num_layers 2 --T 1000 --K_in 5 --K_out 5 --clip 1.0 \
        --dropout 0.0 --weight_decay 0.0 --seeds 3 \
        --track_every 25"

run_cell() {   # out_dir, then extra flags
  local OUT=$1; shift
  if [ -s "$OUT/sparse_gnn_reddit_dp_results.csv" ] || \
     [ -s "$OUT/sparse_gnn_reddit_results.csv" ]; then
    echo "  [skip] $OUT already complete"; return 0
  fi
  $PY -m src.experiments.sparse $COMMON "$@" --out_dir $OUT
}

echo "=== Reddit inductive: non-DP reference ==="
run_cell $OUT_ROOT/nodp --p2 1.0 --lr 0.01

for P2 in 1.0 0.1; do
  for SIGMA in 5.0 20.0; do
    OUT=$OUT_ROOT/p${P2}_s${SIGMA}
    echo "=== Reddit inductive DP: p2=$P2 sigma=$SIGMA ==="
    run_cell $OUT --dp --p2 $P2 --sigma $SIGMA --lr 0.3
    $PY -m src.experiments.compute_epsilon \
        --csv $OUT/sparse_gnn_reddit_dp_results.csv --delta $DELTA || true
  done
done

echo "=== REDDIT INDUCTIVE COMPLETE ==="
