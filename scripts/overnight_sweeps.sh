#!/bin/zsh
# Overnight SparseGNN sweeps on ogbn-arxiv (CPU).
#
# Run with (lid-closed recipe):
#   sudo pmset -a disablesleep 1
#   nohup caffeinate -is ./scripts/overnight_sweeps.sh > results/overnight.log 2>&1 &
#   # morning: sudo pmset -a disablesleep 0
#
# Estimated total: ~4-5 h.  Each block writes its own out_dir and finishes
# with post-hoc Theorem 4 epsilon, so partial completion still yields usable
# results in order of priority.

set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

run_eps() {
    $PY -m src.sparse.compute_epsilon --csv "$1" --delta 1e-6 || true
}

echo "=== [1/5] Headline re-run: r=1 K=5, 5 seeds, + sigma=20 point ==="
$PY -m src.sparse.run --dataset ogbn-arxiv --dp --K_in 5 \
    --p1 0.005 --p2 1.0 0.5 0.25 0.1 --r 1 --sigma 2.0 5.0 10.0 20.0 \
    --T 500 --clip 1.0 --lr 1.0 --seeds 5 --roots_from train \
    --out_dir results/overnight_headline
run_eps results/overnight_headline/sparse_gnn_ogbn-arxiv_dp_results.csv

echo "=== [2/5] r=2: does deeper expansion pay for its epsilon cost? ==="
$PY -m src.sparse.run --dataset ogbn-arxiv --dp --K_in 5 \
    --p1 0.005 --p2 0.5 0.25 0.1 --r 2 --sigma 10.0 20.0 \
    --T 500 --clip 1.0 --lr 1.0 --seeds 3 --roots_from train \
    --out_dir results/overnight_r2
run_eps results/overnight_r2/sparse_gnn_ogbn-arxiv_dp_results.csv

echo "=== [3/5] K_in=10: looser degree cap ==="
$PY -m src.sparse.run --dataset ogbn-arxiv --dp --K_in 10 \
    --p1 0.005 --p2 0.25 0.1 --r 1 --sigma 10.0 20.0 \
    --T 500 --clip 1.0 --lr 1.0 --seeds 3 --roots_from train \
    --out_dir results/overnight_k10
run_eps results/overnight_k10/sparse_gnn_ogbn-arxiv_dp_results.csv

echo "=== [4/5] Non-DP ceilings for the r=2 and K_in=10 graphs ==="
$PY -m src.sparse.run --dataset ogbn-arxiv --K_in 5 \
    --p1 0.005 --p2 0.5 0.25 0.1 --r 2 --T 500 --lr 0.01 --seeds 2 \
    --roots_from train --out_dir results/overnight_r2
$PY -m src.sparse.run --dataset ogbn-arxiv --K_in 10 \
    --p1 0.005 --p2 0.25 0.1 --r 1 --T 500 --lr 0.01 --seeds 2 \
    --roots_from train --out_dir results/overnight_k10

echo "=== [5/5] CiteSeer DP re-run with batch normalization fix ==="
$PY -m src.sparse.run --dataset citeseer --dp --K_in 5 \
    --p1 0.3 --p2 1.0 0.25 0.1 --r 1 --sigma 2.0 5.0 \
    --T 300 --clip 1.0 --lr 1.0 --seeds 3 --roots_from train \
    --out_dir results/overnight_citeseer
run_eps results/overnight_citeseer/sparse_gnn_citeseer_dp_results.csv

echo "=== overnight sweeps complete ==="
