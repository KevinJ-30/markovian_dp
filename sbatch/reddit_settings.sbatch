#!/bin/bash
#SBATCH -Jreddit_settings
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G                # Reddit: 114.6M arcs before capping
#SBATCH -t12:00:00
#SBATCH -oReport-reddit-settings-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# Reddit under three graph-access settings, so the utility cost of the
# inductive restriction is separated from the cost of DP noise.
#
#   transductive   full graph during training; test nodes are reachable by
#                  expansion.  The DP guarantee still covers every node
#                  (Theorem 6.4 substitutes ANY s), but test accuracy no longer
#                  measures generalization to unseen nodes.  This is the
#                  setting GAP / ProGAP report, so it is the comparable one.
#   inductive      train on the train-induced subgraph; expansion cannot reach
#                  val/test.  Honest generalization, impoverished training graph
#                  (the same surgery costs Flickr 76% and arxiv 68% of arcs).
#
# PPI and RelBench supply the third arm — natively inductive, no surgery — and
# run elsewhere.
#
# Every run is tracked every 50 steps, so one run sweeps epsilon continuously
# and eps in {1, 2, 4, 8, 16} (GAP/ProGAP's grid) can be read off directly
# rather than targeted with separate sigma values.

cd $SLURM_SUBMIT_DIR
export REDDIT_DATA_ROOT=/storage/ice1/8/1/kjacob7/pyg_data/Reddit
export OMP_NUM_THREADS=1

PY="python -u"
DELTA=1e-6
COMMON="--dataset reddit --direction in --aggr mean --p1 0.01 --r 1 \
        --num_layers 2 --T 2000 --K_in 5 --K_out 5 --clip 1.0 \
        --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 \
        --track_every 50"

for SETTING in transductive inductive; do
  FLAG=""
  [ "$SETTING" = "inductive" ] && FLAG="--inductive"

  echo "=== $SETTING: non-DP reference ==="
  $PY -m src.sparse.run $COMMON $FLAG --p2 1.0 --lr 0.01 \
      --out_dir results/reddit/${SETTING}_nodp

  for P2 in 1.0 0.1; do
    for SIGMA in 5.0 20.0; do
      OUT=results/reddit/${SETTING}_p${P2}_s${SIGMA}
      echo "=== $SETTING DP: p2=$P2 sigma=$SIGMA ==="
      $PY -m src.sparse.run $COMMON $FLAG --dp --p2 $P2 --sigma $SIGMA \
          --lr 0.3 --out_dir $OUT
      $PY -m src.sparse.compute_epsilon \
          --csv $OUT/sparse_gnn_reddit_dp_results.csv --delta $DELTA
    done
  done
done

echo "=== summarize ==="
for d in results/reddit/*/; do python scripts/summarize_sweep.py $(dirname $d) 2>/dev/null | head -1; done
echo "=== REDDIT SETTINGS COMPLETE ==="
