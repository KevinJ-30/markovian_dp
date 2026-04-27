#!/bin/bash
#SBATCH -Jbaselines_node
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t8:00:00
#SBATCH -oReport-baselines-node-%j.out
#SBATCH --mail-type=END,FAIL

# GCN + MLP node baselines only (no --algo → no binning / subgraph runs).
# Planetoid full-graph; ogbn-arxiv / reddit / ogbn-products use NeighborLoader (GCN)
# and node minibatches (MLP) on large graphs. See scripts/run_baselines_node.sh.
#
# Submit from repo root: sbatch sbatch/baselines_node.sbatch
# Override roots (optional): sbatch --export=ALL,OGB_ROOT_ARXIV=/path/...,OGB_ROOT_PRODUCTS=/path/...,REDDIT_ROOT=/path/... sbatch/baselines_node.sbatch

cd "$SLURM_SUBMIT_DIR"

# Same scratch layout as e1_*/e2_*/single_phase_* sbatch files (read by scripts/run_baselines_node.sh).
export OGB_ROOT_ARXIV="${OGB_ROOT_ARXIV:-/storage/ice1/8/1/kjacob7/ogb_data/ogbn-arxiv}"
export OGB_ROOT_PRODUCTS="${OGB_ROOT_PRODUCTS:-/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products}"
export REDDIT_ROOT="${REDDIT_ROOT:-/storage/ice1/8/1/kjacob7/pyg_data/Reddit}"
echo "OGB_ROOT_ARXIV=$OGB_ROOT_ARXIV"
echo "OGB_ROOT_PRODUCTS=$OGB_ROOT_PRODUCTS"
echo "REDDIT_ROOT=$REDDIT_ROOT"

srun bash scripts/run_baselines_node.sh
