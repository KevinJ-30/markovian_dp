#!/bin/bash
#SBATCH -Je2_reddit
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t24:00:00
#SBATCH -oReport-e2-reddit-%j.out
#SBATCH --mail-type=END,FAIL

# E2 — Privacy/utility frontier on Reddit.
# Methods: DP-MLP (= MLP + Algo2 + bins=1), Algo 2, Algo 3 (p=0.3).
# Grid: bins {4,8,16} x sigma {0.5,0.7,1.0,1.5,2.0,4.0} x q {0.03,0.05,0.10,0.20,0.40}.
# Seeds: 3. Accountant: PRV.
# No graph sparsification — kept separate from sparsification experiments.
# (Note: without a degree cap, per-node sensitivity for node-DP is bounded
# only by the gradient clip norm; the algorithmic argument still holds, but
# this is a different operating point than the sparsified runs.)
#
# Reddit is memory-heavy at large q — q=0.40 with 232k nodes can stress 80GB.
# If OOM: drop q=0.40 first, then q=0.20.

cd "$SLURM_SUBMIT_DIR"

export REDDIT_DATA_ROOT=/storage/ice1/8/1/kjacob7/pyg_data/Reddit
echo "Reddit data root: $REDDIT_DATA_ROOT"

SIGMAS="0.5 0.7 1.0 1.5 2.0 4.0"
SEEDS=3
OUT=results/e2_frontier

for Q in 0.03 0.05 0.10 0.20 0.40; do
  echo "=== q=$Q : Algo 2 / Algo 3 sweep ==="
  srun python run.py \
    --dataset reddit \
    --task node \
    --algo 2 3 \
    --num-bins 4 8 16 \
    --subsample-prob 0.3 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 200 --patience 10 \
    --tag "e2_reddit_q${Q//./}" \
    --output-dir "$OUT"

  echo "=== q=$Q : DP-MLP (bins=1) ==="
  srun python run.py \
    --dataset reddit \
    --task node \
    --model mlp \
    --algo 2 \
    --num-bins 1 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 200 --patience 10 \
    --tag "e2_reddit_dpmlp_q${Q//./}" \
    --output-dir "$OUT"
done
