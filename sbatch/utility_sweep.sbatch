#!/bin/bash
#SBATCH -Jdp_utility_sweep                # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=64G                # Memory per GPU
#SBATCH -t2:00:00                         # 2 hours walltime
#SBATCH -oReport-utility-%j.out           # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── Utility sweep: all algos x bins, NO DP ──
# Shows how much accuracy each partitioning strategy costs
# without any noise — isolates the structural effect of subgraphing.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda

srun python run.py \
    --dataset cora citeseer pubmed \
    --algo 1 2 3 \
    --num-bins 2 4 8 16 \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag utility \
    --output-dir results/utility
