#!/bin/bash
#SBATCH -Je1_arxiv
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t4:00:00
#SBATCH -oReport-e1-arxiv-%j.out
#SBATCH --mail-type=END,FAIL

# E1 — No-DP utility on ogbn-arxiv.
# Methods: GCN baseline, MLP baseline, Algo 2, Algo 3 (p=0.3).
# Bins: 2, 4, 8, 16. Seeds: 3 (large-dataset budget).

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-arxiv
echo "OGB data root: $OGB_DATA_ROOT"

srun python run.py \
  --dataset ogbn-arxiv \
  --task node \
  --baseline gcn mlp \
  --algo 2 3 \
  --num-bins 2 4 8 16 \
  --subsample-prob 0.3 \
  --seeds 3 \
  --epochs 200 \
  --tag e1_arxiv \
  --output-dir results/e1_no_dp
