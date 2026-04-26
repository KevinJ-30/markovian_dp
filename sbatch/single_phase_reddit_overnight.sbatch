#!/bin/bash
#SBATCH -Jsp_reddit
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-sp-reddit-%j.out
#SBATCH --mail-type=END,FAIL

# Single-phase DP on Reddit with degree cap for stability.

cd "$SLURM_SUBMIT_DIR"

export REDDIT_DATA_ROOT=/storage/ice1/8/1/kjacob7/pyg_data/Reddit
echo "Reddit data root: $REDDIT_DATA_ROOT"

srun python run.py \
  --dataset reddit \
  --task node \
  --algo 2 3 \
  --num-bins 8 \
  --single-phase --q 0.05 --steps-per-epoch 10 \
  --dp --noise-multiplier 1.0 2.0 \
  --accountant prv \
  --max-in-degree 100 \
  --seeds 1 \
  --converge --max-epochs 200 --patience 10 \
  --tag sp_reddit_q05_deg100 \
  --output-dir results/single_phase_overnight
