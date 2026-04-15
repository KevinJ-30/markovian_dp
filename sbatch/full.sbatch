#!/bin/bash
#SBATCH -Jdp_full                         # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=128G               # 128G for ogbn-products
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-full-%j.out              # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── Full sweep: everything in one job ──
# Run overnight to get all results at once.

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda

OUTPUT=results/full

# ── 1. Baselines (small datasets) ──
echo "=== Baselines (small) ==="
srun python run.py \
    --dataset cora citeseer pubmed \
    --baseline gcn mlp \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag full_baselines \
    --output-dir $OUTPUT

# ── 2. Utility sweep (no DP) ──
echo "=== Utility sweep ==="
srun python run.py \
    --dataset cora citeseer pubmed \
    --algo 1 2 3 \
    --num-bins 2 4 8 16 \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag full_utility \
    --output-dir $OUTPUT

# ── 3. Noise sweep (DP) ──
echo "=== Noise sweep ==="
srun python run.py \
    --dataset cora citeseer pubmed \
    --algo 1 2 3 \
    --num-bins 4 8 16 \
    --dp --noise-multiplier 0.1 0.5 1.0 2.0 4.0 8.0 \
    --accountant \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag full_noise \
    --output-dir $OUTPUT

# ── 4. Poisson (DP + subsampling) ──
echo "=== Poisson ==="
srun python run.py \
    --dataset cora citeseer pubmed \
    --algo 2 3 \
    --num-bins 4 8 16 \
    --poisson --q-epoch 0.5 --q-step 0.5 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 5 \
    --converge --max-epochs 500 --patience 20 \
    --tag full_poisson \
    --output-dir $OUTPUT

# ── 5. ogbn-products ──
echo "=== ogbn-products ==="
srun python run.py \
    --dataset ogbn-products \
    --baseline gcn mlp \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --batch-size 1024 --num-neighbors 15 10 \
    --tag full_ogbn \
    --output-dir $OUTPUT

echo "=== Done ==="
