#!/bin/bash
#SBATCH -Jmarkovian_dp_benchmark         # Job name
#SBATCH -N1 --ntasks-per-node=1          # 1 node, 1 task
#SBATCH --gres=gpu:H200:1               # 1x H200 GPU
#SBATCH --mem-per-gpu=128G              # Memory per GPU
#SBATCH -t4:00:00                        # 4 hours walltime
#SBATCH -oReport-%j.out                  # Output file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail notifications

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda

# Install dependencies (first run only — pip caches after that)
pip install --user torch torch-geometric ogb matplotlib numpy

# Run the full benchmark
srun python run_benchmark.py --converge --max-epochs 500 --patience 20
