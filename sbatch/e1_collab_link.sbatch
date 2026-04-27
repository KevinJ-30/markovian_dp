#!/bin/bash
#SBATCH -Je1_collab_lp
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t4:00:00
#SBATCH -oReport-e1-collab-lp-%j.out
#SBATCH --mail-type=END,FAIL

# E1 — Non-DP link prediction on ogbl-collab.
# Algo 2 / 3 only (--baseline is not supported for --task link in run.py).
# Bins: 2, 4, 8, 16. Metrics: Hits@K (reported as train/test columns in JSONL).

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbl-collab
echo "OGB data root: $OGB_DATA_ROOT"

srun python run.py \
  --dataset ogbl-collab \
  --task link \
  --algo 2 3 \
  --num-bins 2 4 8 16 \
  --subsample-prob 0.3 \
  --seeds 2 \
  --epochs 200 \
  --tag e1_collab_link \
  --output-dir results/e1_no_dp
