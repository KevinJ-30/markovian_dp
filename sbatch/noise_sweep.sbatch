#!/bin/bash
#SBATCH -Jdp_noise_sweep                  # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=64G                # Memory per GPU
#SBATCH -t3:00:00                         # 3 hours walltime
#SBATCH -oReport-noise-%j.out             # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── Noise sweep: accuracy vs epsilon (privacy-utility tradeoff) ──
# Sweeps noise multipliers with RDP accountant to get epsilon values.
# Produces the main privacy-utility curves for the paper.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda

srun python run.py \
    --dataset cora citeseer pubmed \
    --baseline gcn mlp \
    --algo 1 2 3 \
    --num-bins 4 8 16 \
    --dp --noise-multiplier 0.1 0.5 1.0 2.0 4.0 8.0 \
    --accountant \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag noise_sweep \
    --output-dir results/noise_sweep
