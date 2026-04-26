#!/bin/bash
#SBATCH -Jsp_collab_lp
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t8:00:00
#SBATCH -oReport-sp-collab-lp-%j.out
#SBATCH --mail-type=END,FAIL

# Single-phase DP link prediction on ogbl-collab.

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbl-collab
echo "OGB data root: $OGB_DATA_ROOT"

srun python run.py \
  --dataset ogbl-collab \
  --task link \
  --algo 2 3 \
  --num-bins 4 8 16 \
  --single-phase --q 0.10 --steps-per-epoch 10 \
  --dp --noise-multiplier 0.7 1.0 2.0 \
  --accountant prv \
  --seeds 1 \
  --converge --max-epochs 250 --patience 12 \
  --tag sp_collab_link_q10 \
  --output-dir results/single_phase_overnight
