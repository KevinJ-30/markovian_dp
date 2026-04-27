#!/bin/bash
#SBATCH -Je2_bluesky
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-e2-bluesky-%j.out
#SBATCH --mail-type=END,FAIL

# E2 — Privacy/utility frontier on Bluesky (GraphBench, temporal-split inductive).
# Methods: DP-MLP (= MLP + Algo2 + bins=1), Algo 2, Algo 3 (p=0.3).
# Grid: bins {4,8,16} x sigma {0.5,0.7,1.0,1.5,2.0,4.0} x q {0.03,0.05,0.10,0.20,0.40}.
# Seeds: 1. Accountant: PRV.
# No graph sparsification — kept separate from sparsification experiments (E3).
#
# Bluesky temporal split is naturally inductive (see e1_bluesky.sbatch header).
# Cutoffs default to 60th/80th percentile of edge_time; override via
# BLUESKY_T_TRAIN / BLUESKY_T_VAL.

cd "$SLURM_SUBMIT_DIR"

export BLUESKY_DATA_ROOT=/storage/ice1/8/1/kjacob7/graphbench/bluesky
echo "Bluesky data root: $BLUESKY_DATA_ROOT"

SIGMAS="0.5 0.7 1.0 1.5 2.0 4.0"
SEEDS=1
OUT=results/e2_frontier

for Q in 0.03 0.05 0.10 0.20 0.40; do
  echo "=== q=$Q : Algo 2 / Algo 3 sweep ==="
  srun python run.py \
    --dataset bluesky \
    --task node \
    --algo 2 3 \
    --num-bins 4 8 16 \
    --subsample-prob 0.3 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 200 --patience 10 \
    --tag "e2_bluesky_q${Q//./}" \
    --output-dir "$OUT"

  echo "=== q=$Q : DP-MLP (bins=1) ==="
  srun python run.py \
    --dataset bluesky \
    --task node \
    --model mlp \
    --algo 2 \
    --num-bins 1 \
    --single-phase --q "$Q" --steps-per-epoch 10 \
    --dp --noise-multiplier $SIGMAS \
    --accountant prv \
    --seeds $SEEDS \
    --converge --max-epochs 200 --patience 10 \
    --tag "e2_bluesky_dpmlp_q${Q//./}" \
    --output-dir "$OUT"
done
