#!/bin/bash
#SBATCH -Jdp_baselines                    # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=64G                # Memory per GPU
#SBATCH -t0:30:00                         # 30 min walltime
#SBATCH -oReport-baselines-%j.out         # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── GCN + MLP baselines on all small datasets (5 seeds, converge) ──
# Use this to establish non-private accuracy reference points.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda

srun python run.py \
    --dataset cora citeseer pubmed \
    --baseline gcn mlp \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag baselines \
    --output-dir results/baselines
