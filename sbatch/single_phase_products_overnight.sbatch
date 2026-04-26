#!/bin/bash
#SBATCH -Jsp_products
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-sp-products-%j.out
#SBATCH --mail-type=END,FAIL

# Single-phase DP on ogbn-products (overnight, compact sweep).

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products
echo "OGB data root: $OGB_DATA_ROOT"

srun python run.py \
  --dataset ogbn-products \
  --task node \
  --algo 2 3 \
  --num-bins 4 8 16 \
  --single-phase --q 0.10 --steps-per-epoch 10 \
  --dp --noise-multiplier 1.0 2.0 \
  --accountant prv \
  --max-in-degree 100 \
  --seeds 1 \
  --converge --max-epochs 250 --patience 12 \
  --tag sp_products_q10_deg100 \
  --output-dir results/single_phase_overnight
