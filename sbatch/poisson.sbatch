#!/bin/bash
#SBATCH -Jdp_poisson                      # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=128G               # 128G for ogbn-products
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-poisson-%j.out           # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── Poisson sweep on ogbn-products ──
#
# Part A: Rate sweep (fixed steps_per_epoch=10)
#   Tests how subsampling rate affects privacy-utility tradeoff.
#
# Part B: Steps-per-epoch sweep (fixed rate q_epoch=0.1, q_step=0.3)
#   Tests whether 10 steps/epoch is enough for convergence at aggressive rates.
#   Expected node appearances per epoch: steps * q_epoch * q_step
#     10 steps → 0.3x    (undercoverage)
#     20 steps → 0.6x
#     50 steps → 1.5x    (slight overcoverage, but more composition → higher eps)

cd $SLURM_SUBMIT_DIR

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products
echo "OGB data root: $OGB_DATA_ROOT"

# ── Baselines (for plot reference lines) ──
#echo "=== Baselines ==="
#srun python run.py \
#    --dataset ogbn-products \
#    --baseline gcn mlp \
 #   --seeds 3 \
  #  --converge --max-epochs 300 --patience 15 \
  #  --batch-size 1024 --num-neighbors 15 10 \
  #  --tag poisson_baselines \
  #  --output-dir results/poisson

# ═══════════════════════════════════════════════════════════════════
# Part A: Rate sweep (steps_per_epoch=10)
# ═══════════════════════════════════════════════════════════════════

# ── Rate 1: q_epoch=0.5, q_step=0.5 (effective 25%) ──
echo "=== Rate: (0.5, 0.5) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.5 --q-step 0.5 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q50_q50 \
    --output-dir results/poisson

# ── Rate 2: q_epoch=0.3, q_step=0.3 (effective 9%) ──
echo "=== Rate: (0.3, 0.3) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.3 --q-step 0.3 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q30_q30 \
    --output-dir results/poisson

# ── Rate 3: q_epoch=0.1, q_step=0.3 (effective 3%) ──
echo "=== Rate: (0.1, 0.3) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.1 --q-step 0.3 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q10_q30 \
    --output-dir results/poisson

# ── Rate 4: q_epoch=1.0, q_step=0.1 (effective 10%, step-only) ──
echo "=== Rate: (1.0, 0.1) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 1.0 --q-step 0.1 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q100_q10 \
    --output-dir results/poisson

# ═══════════════════════════════════════════════════════════════════
# Part B: Steps-per-epoch sweep at (0.1, 0.3)
# ═══════════════════════════════════════════════════════════════════

# ── 20 steps/epoch (0.6x expected coverage) ──
echo "=== Steps: 20 at (0.1, 0.3) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.1 --q-step 0.3 --steps-per-epoch 20 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q10_q30_steps20 \
    --output-dir results/poisson

# ── 50 steps/epoch (1.5x expected coverage) ──
echo "=== Steps: 50 at (0.1, 0.3) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.1 --q-step 0.3 --steps-per-epoch 50 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --tag poisson_q10_q30_steps50 \
    --output-dir results/poisson

echo "=== Done ==="
