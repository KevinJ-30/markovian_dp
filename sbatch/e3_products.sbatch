#!/bin/bash
#SBATCH -Je3_products
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-e3-products-%j.out
#SBATCH --mail-type=END,FAIL

# E3 — Sparsified-DP utility frontier on ogbn-products.
#
# Distinct paradigm from E1/E2:
#   * Graph is sparsified to bounded max in-degree D (--max-in-degree).
#   * Per-NODE gradient clipping (not per-bin).
#   * Single Gaussian release per step with std
#         sigma * C * (1 + D + D^2 + ... + D^L)
#     where L = --gnn-layers (default 2 to match SubgraphGCN).
# Standard node-DP bound from Daigavane et al. 2021. Compare against E2.
#
# Products is in between arxiv (smallest) and reddit (largest). Run after
# arxiv has validated the per-node grad path; before reddit if wall-time
# budget is tight.

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products
echo "OGB data root: $OGB_DATA_ROOT"

SIGMAS="0.5 0.7 1.0 1.5 2.0 4.0"
SEEDS=1
OUT=results/e3_sparsified_dp

for D in 10 25; do
  for Q in 0.03 0.05 0.10; do
    echo "=== D=$D q=$Q : Algo 2 / Algo 3 sweep (sparsified-dp) ==="
    srun python run.py \
      --dataset ogbn-products \
      --task node \
      --paradigm sparsified-dp \
      --max-in-degree "$D" \
      --gnn-layers 2 \
      --algo 2 3 \
      --num-bins 4 8 \
      --subsample-prob 0.3 \
      --single-phase --q "$Q" --steps-per-epoch 10 \
      --dp --noise-multiplier $SIGMAS \
      --accountant prv \
      --seeds $SEEDS \
      --converge --max-epochs 250 --patience 12 \
      --tag "e3_products_D${D}_q${Q//./}" \
      --output-dir "$OUT"
  done
done
