#!/bin/bash
#SBATCH -Je3_arxiv
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=96G
#SBATCH -t8:00:00
#SBATCH -oReport-e3-arxiv-%j.out
#SBATCH --mail-type=END,FAIL

# E3 — Sparsified-DP utility frontier on ogbn-arxiv.
#
# Distinct paradigm from E1/E2:
#   * Graph is sparsified to bounded max in-degree D (--max-in-degree).
#   * Per-NODE gradient clipping (not per-bin).
#   * Single Gaussian release per step with std
#         sigma * C * (1 + D + D^2 + ... + D^L)
#     where L = --gnn-layers (default 2 to match SubgraphGCN).
# This is the standard node-DP bound from Daigavane et al. 2021: removing one
# node changes the loss summand of at most (1+D+...+D^L) others, each with
# clipped gradient L2 <= C, so the summed-gradient sensitivity is bounded.
#
# Compare against E2 (same dataset, same sigma/q grid) — E2 uses per-bin clip
# with degree-agnostic noise, this run uses per-node clip with degree-aware
# noise. Same accountant input (sigma); only the sensitivity claim differs.
#
# arxiv first because per-node grads cost ~|active| backwards per bin per step;
# this is the smallest of the three target datasets and the right place to
# validate the path before scaling to products / reddit.

cd "$SLURM_SUBMIT_DIR"

export OGB_DATA_ROOT=/storage/ice1/8/1/kjacob7/ogb_data/ogbn-arxiv
echo "OGB data root: $OGB_DATA_ROOT"

SIGMAS="0.5 1.0 2.0"
SEEDS=1
OUT=results/e3_sparsified_dp

for D in 10 25; do
  for Q in 0.05 0.10; do
    echo "=== D=$D q=$Q : Algo 2 / Algo 3 sweep (sparsified-dp) ==="
    srun python run.py \
      --dataset ogbn-arxiv \
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
      --converge --max-epochs 300 --patience 15 \
      --tag "e3_arxiv_D${D}_q${Q//./}" \
      --output-dir "$OUT"
  done
done
