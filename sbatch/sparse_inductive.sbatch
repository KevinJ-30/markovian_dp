#!/bin/bash
#SBATCH -Jsparse_ind                      # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                 # 1x H200 GPU
#SBATCH --mem-per-gpu=128G                # Reddit is large
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-sparse-ind-%j.out        # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── SparseGNN inductive ladder on Reddit ──
#
# Non-private ceiling -> sparsification-only -> clip+noise -> post-hoc epsilon.
# All private runs use --direction in.
#
# Reddit (232k nodes) is the reason this lives on the cluster;
# For local experiments, invoke python -m src.experiments.sparse directly.
#
# IMPORTANT: --direction in is the corrected orientation (manuscript v35 §5-6).
# Anything recorded before that fix used out-expansion, where the root received
# no messages during training; do not pool those CSVs with these.

cd $SLURM_SUBMIT_DIR

export OGB_DATA_ROOT=${SCRATCH:-$HOME/scratch}/data/ogbn-arxiv
export REDDIT_DATA_ROOT=${SCRATCH:-$HOME/scratch}/data/Reddit
export FLICKR_DATA_ROOT=${SCRATCH:-$HOME/scratch}/data/Flickr

DELTA=1e-6

# ── Reddit: single-label, convert to inductive via the train-induced subgraph ──

echo "=== Reddit Stage 0: GCN ceiling ==="
srun python -u -m src.experiments.sparse \
    --dataset reddit --direction in \
    --p2 1.0 --r 2 --p1 0.002 --T 500 --lr 0.01 --seeds 3 \
    --out_dir results/ppi/ceiling_reddit

echo "=== Reddit Stage 1: sparsification sweep ==="
srun python -u -m src.experiments.sparse \
    --dataset reddit --direction in --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 1 --p1 0.002 --T 500 --lr 0.01 --seeds 3 \
    --out_dir results/arxiv/reddit_stage1

echo "=== Reddit Stage 2: DP sigma sweep ==="
srun python -u -m src.experiments.sparse \
    --dataset reddit --direction in --dp --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 1 --sigma 2.0 5.0 10.0 20.0 \
    --clip 1.0 --lr 1.0 --p1 0.002 --T 500 --seeds 3 \
    --out_dir results/arxiv/reddit_stage2

echo "=== Reddit Stage 3: post-hoc epsilon ==="
srun python -u -m src.experiments.compute_epsilon \
    --csv results/arxiv/reddit_stage2/sparse_gnn_reddit_dp_results.csv \
    --delta $DELTA

echo "=== Done ==="
