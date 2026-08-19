#!/bin/bash
#SBATCH -Jsparse_ind                      # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task
#SBATCH --gres=gpu:H200:1                 # 1x H200 GPU
#SBATCH --mem-per-gpu=128G                # Reddit + PPI are large
#SBATCH -t8:00:00                         # 8 hours walltime
#SBATCH -oReport-sparse-ind-%j.out        # Output file
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail notifications

# ── SparseGNN inductive ladder on the big homogeneous graphs ──
#
# Stage 0 baselines -> Stage 1 sparsification-only -> Stage 2 clip+noise ->
# Stage 3 post-hoc epsilon (Theorem 6.4, since these runs use --direction in).
#
# Reddit (232k nodes) and PPI (57k nodes, 1.6M edges, 121-way multilabel) are the
# reason this lives on the cluster; ogbn-arxiv and Flickr run fine on a laptop
# via scripts/ladder_stage01.sh.
#
# IMPORTANT: --direction in is the corrected orientation (manuscript v35 §5-6).
# Anything recorded before that fix used out-expansion, where the root received
# no messages during training; do not pool those CSVs with these.

cd $SLURM_SUBMIT_DIR

export OGB_DATA_ROOT=/storage/project/r-smussmann3-0/shared/datasets/ogbn-arxiv
export REDDIT_DATA_ROOT=/storage/project/r-smussmann3-0/shared/datasets/Reddit
export FLICKR_DATA_ROOT=/storage/project/r-smussmann3-0/shared/datasets/Flickr
export PPI_DATA_ROOT=/storage/project/r-smussmann3-0/shared/datasets/PPI

DELTA=1e-6

# ── Reddit: single-label, convert to inductive via the train-induced subgraph ──
echo "=== Reddit Stage 0: MLP baseline ==="
srun python -u -m src.sparse.run \
    --dataset reddit --inductive --direction in --model mlp \
    --r 0 --p2 1.0 --p1 0.002 --T 500 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/ppi/mlp_reddit

echo "=== Reddit Stage 0: GCN ceiling ==="
srun python -u -m src.sparse.run \
    --dataset reddit --inductive --direction in \
    --p2 1.0 --r 2 --p1 0.002 --T 500 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/ppi/ceiling_reddit

echo "=== Reddit Stage 1: sparsification sweep ==="
srun python -u -m src.sparse.run \
    --dataset reddit --inductive --direction in --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 1 --p1 0.002 --T 500 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/arxiv/reddit_stage1

echo "=== Reddit Stage 2: DP sigma sweep ==="
srun python -u -m src.sparse.run \
    --dataset reddit --inductive --direction in --dp --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 1 --sigma 2.0 5.0 10.0 20.0 \
    --clip 1.0 --lr 1.0 --p1 0.002 --T 500 --seeds 3 \
    --roots_from train --out_dir results/arxiv/reddit_stage2

echo "=== Reddit Stage 3: post-hoc epsilon ==="
srun python -u -m src.sparse.compute_epsilon \
    --csv results/arxiv/reddit_stage2/sparse_gnn_reddit_dp_results.csv \
    --delta $DELTA

# ── PPI: natively inductive (24 disjoint graphs), 121-way multilabel -> micro-F1 ──
# --inductive is deliberately absent: the splits are already separate components,
# so a training root's expansion cannot reach a val/test node at any r.
echo "=== PPI Stage 0-1 ==="
srun python -u -m src.sparse.run \
    --dataset ppi --direction in --model mlp \
    --r 0 --p2 1.0 --p1 0.01 --T 1000 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/ppi/mlp_ppi

srun python -u -m src.sparse.run \
    --dataset ppi --direction in --model multilabel_gnn \
    --p2 1.0 --r 2 --p1 0.01 --T 1000 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/ppi/ceiling_ppi

srun python -u -m src.sparse.run \
    --dataset ppi --direction in --model multilabel_gnn --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 2 --p1 0.01 --T 1000 --lr 0.01 --seeds 3 \
    --roots_from train --out_dir results/ppi/stage1_ppi

echo "=== PPI Stage 2-3: DP + epsilon ==="
srun python -u -m src.sparse.run \
    --dataset ppi --direction in --model multilabel_gnn --dp --K_in 5 \
    --p2 1.0 0.5 0.25 0.1 --r 2 --sigma 2.0 5.0 10.0 20.0 \
    --clip 1.0 --lr 1.0 --p1 0.01 --T 1000 --seeds 3 \
    --roots_from train --out_dir results/ppi/stage2_ppi

srun python -u -m src.sparse.compute_epsilon \
    --csv results/ppi/stage2_ppi/sparse_gnn_ppi_dp_results.csv \
    --delta $DELTA

echo "=== Done ==="
