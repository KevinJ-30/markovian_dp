#!/bin/bash
#SBATCH -Je1_reddit
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-e1-reddit-%j.out
#SBATCH --mail-type=END,FAIL

# E1 — No-DP utility on Reddit.
# Methods: GCN baseline, MLP baseline, Algo 2, Algo 3 (p=0.3).
# Bins: 2, 4, 8, 16. Seeds: 3.
# No graph sparsification — kept separate from sparsification experiments.

cd "$SLURM_SUBMIT_DIR"

export REDDIT_DATA_ROOT=/storage/ice1/8/1/kjacob7/pyg_data/Reddit
echo "Reddit data root: $REDDIT_DATA_ROOT"

srun python run.py \
  --dataset reddit \
  --task node \
  --baseline gcn mlp \
  --algo 2 3 \
  --num-bins 2 4 8 16 \
  --subsample-prob 0.3 \
  --seeds 3 \
  --converge --max-epochs 200 --patience 10 \
  --tag e1_reddit \
  --output-dir results/e1_no_dp
