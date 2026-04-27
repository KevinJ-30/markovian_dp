#!/bin/bash
#SBATCH -Je1_bluesky
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-e1-bluesky-%j.out
#SBATCH --mail-type=END,FAIL

# E1 — No-DP utility on Bluesky (GraphBench, temporal-split inductive).
# Methods: GCN baseline, MLP baseline, Algo 2, Algo 3 (p=0.3).
# Bins: 2, 4, 8, 16. Seeds: 3.
#
# Bluesky has a temporal split: training graph is the snapshot at t<=t_train,
# eval graph is the snapshot at t<=t_val. By construction this is inductive:
# test-time nodes/edges did not exist at training time. The trainer's
# evaluate() method picks up data.eval_edge_index automatically.
#
# Cutoffs default to the 60th/80th percentile of edge_time. Override via
# BLUESKY_T_TRAIN / BLUESKY_T_VAL env vars if needed.

cd "$SLURM_SUBMIT_DIR"

export BLUESKY_DATA_ROOT=/storage/ice1/8/1/kjacob7/graphbench/bluesky
echo "Bluesky data root: $BLUESKY_DATA_ROOT"

srun python run.py \
  --dataset bluesky \
  --task node \
  --baseline gcn mlp \
  --algo 2 3 \
  --num-bins 2 4 8 16 \
  --subsample-prob 0.3 \
  --seeds 3 \
  --converge --max-epochs 200 --patience 10 \
  --tag e1_bluesky \
  --output-dir results/e1_no_dp
