#!/bin/bash
#SBATCH -Jdp_single_phase                 # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=128G               # 128G for ogbn-products
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-single_phase-%j.out      # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── Single-phase (independent) Poisson on ogbn-products ──
#
# --single-phase: fresh independent Bernoulli(q) sample per step (matches
# standard DP-SGD / Opacus accounting). Compare against the correlated
# two-phase scheme in poisson.sbatch.
#
# Early stopping: --converge --patience 15 --max-epochs 300.
# Seeds: 3 (matches poisson.sbatch).
#
# q sweep mirrors effective per-step rates from the two-phase sweep:
#   q=0.03  ↔ two-phase (0.1, 0.3)
#   q=0.10  ↔ two-phase (1.0, 0.1)
#   q=0.30  ↔ aggressive

cd $SLURM_SUBMIT_DIR

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products
echo "OGB data root: $OGB_DATA_ROOT"

# ── q = 0.03 (low rate, conservative privacy cost) ──
echo "=== Single-phase: q=0.03 ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --single-phase --q 0.03 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 2 \
    --converge --max-epochs 300 --patience 15 \
    --tag single_phase_q03 \
    --output-dir results/single_phase

# ── q = 0.10 (moderate rate) ──
echo "=== Single-phase: q=0.10 ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --single-phase --q 0.10 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 2 \
    --converge --max-epochs 300 --patience 15 \
    --tag single_phase_q10 \
    --output-dir results/single_phase

# ── q = 0.30 (aggressive rate) ──
echo "=== Single-phase: q=0.30 ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --single-phase --q 0.30 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 2 \
    --converge --max-epochs 300 --patience 15 \
    --tag single_phase_q30 \
    --output-dir results/single_phase

echo "=== Done ==="
