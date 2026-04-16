#!/bin/bash
#SBATCH -Jdp_ogbn                         # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                # 1x H200 GPU
#SBATCH --mem-per-gpu=128G               # 128G — ogbn-products is large
#SBATCH -t6:00:00                         # 8 hours walltime
#SBATCH -oReport-ogbn-%j.out              # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── ogbn-products: large-scale validation (~2.4M nodes) ──
# Proves the approach scales beyond Planetoid-sized graphs.
#
# Structure (4 runs to decompose accuracy drops):
#   1. Baselines          — GCN/MLP reference (NeighborLoader)
#   2. Utility (no sub)   — partitioning cost alone (full-batch, 1 step/epoch)
#   3. Utility (Poisson)  — partitioning + subsampling cost (no noise)
#   4. DP + Poisson       — full private pipeline

cd $SLURM_SUBMIT_DIR

#module load anaconda3
#module load cuda

# Pre-download dataset to scratch if OGB_DATA_ROOT is set
if [ -n "$OGB_DATA_ROOT" ]; then
    echo "OGB data root: $OGB_DATA_ROOT"
fi

# ── 1. Baselines ──
echo "=== Baselines ==="
srun python run.py \
    --dataset ogbn-products \
    --baseline gcn mlp \
    --seeds 3 \
    --converge --max-epochs 300 --patience 15 \
    --batch-size 1024 --num-neighbors 15 10 \
    --tag ogbn_baselines \
    --output-dir results/ogbn

# ── 2. Utility: partitioning only (no DP, no Poisson) ──
# Full-batch: 1 step/epoch with all training nodes.
# Slow but gives the pure partitioning cost.
#echo "=== Utility (full-batch) ==="
#srun python run.py \
#    --dataset ogbn-products \
#    --algo 2 3 \
#    --num-bins 4 8 16 32 \
#    --seeds 3 \
#    --converge --max-epochs 300 --patience 15 \
#    --tag ogbn_utility_fullbatch \
#    --output-dir results/ogbn

# ── 3. Utility: partitioning + Poisson subsampling (no DP) ──
# Shows how much accuracy Poisson costs on top of partitioning.
#echo "=== Utility (Poisson) ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 4 8 16 32 \
    --poisson --q-epoch 0.1 --q-step 0.3 --steps-per-epoch 10 \
    --seeds 1 \
    --converge --max-epochs 300 --patience 15 \
    --tag ogbn_utility_poisson \
    --output-dir results/ogbn

# ── 4. DP + Poisson (full private pipeline) ──
echo "=== DP + Poisson ==="
srun python run.py \
    --dataset ogbn-products \
    --algo 2 3 \
    --num-bins 8 16 32 \
    --poisson --q-epoch 0.1 --q-step 0.3 --steps-per-epoch 10 \
    --dp --noise-multiplier 0.5 1.0 2.0 4.0 \
    --accountant \
    --seeds 1 \
    --converge --max-epochs 300 --patience 15 \
    --tag ogbn_dp_poisson \
    --output-dir results/ogbn

echo "=== Done ==="
