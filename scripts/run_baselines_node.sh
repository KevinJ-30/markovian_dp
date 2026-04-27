#!/usr/bin/env bash
# Run GCN + MLP node-classification baselines only (no --algo => no binning / subgraph DP).
# Large graphs (>100k nodes): GCN uses NeighborLoader train+eval; MLP uses node minibatches.
#
# Usage (from repo root, on login or compute):
#   bash scripts/run_baselines_node.sh
#
# Override scratch roots if needed:
#   OGB_ROOT_ARXIV=... OGB_ROOT_PRODUCTS=... REDDIT_ROOT=... bash scripts/run_baselines_node.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OGB_ROOT_ARXIV="${OGB_ROOT_ARXIV:-/storage/ice1/8/1/kjacob7/ogb_data/ogbn-arxiv}"
OGB_ROOT_PRODUCTS="${OGB_ROOT_PRODUCTS:-/storage/ice1/8/1/kjacob7/ogb_data/ogbn-products}"
REDDIT_ROOT="${REDDIT_ROOT:-/storage/ice1/8/1/kjacob7/pyg_data/Reddit}"

COMMON_LARGE=(--task node --baseline gcn mlp --batch-size 1024 --num-neighbors 15 10 --seeds 2 --converge --max-epochs 300 --patience 15 --output-dir results/baselines_node)

echo "=== Planetoid (full-graph GCN/MLP) ==="
python run.py \
  --dataset cora citeseer pubmed \
  --task node --baseline gcn mlp \
  --seeds 2 --epochs 200 \
  --tag baselines_planetoid \
  --output-dir results/baselines_node

echo "=== ogbn-arxiv ==="
export OGB_DATA_ROOT="$OGB_ROOT_ARXIV"
echo "OGB_DATA_ROOT=$OGB_DATA_ROOT"
python run.py \
  --dataset ogbn-arxiv \
  "${COMMON_LARGE[@]}" \
  --tag baselines_arxiv

echo "=== Reddit ==="
unset OGB_DATA_ROOT
export REDDIT_DATA_ROOT="$REDDIT_ROOT"
echo "REDDIT_DATA_ROOT=$REDDIT_DATA_ROOT"
python run.py \
  --dataset reddit \
  "${COMMON_LARGE[@]}" \
  --tag baselines_reddit

echo "=== ogbn-products ==="
unset REDDIT_DATA_ROOT
export OGB_DATA_ROOT="$OGB_ROOT_PRODUCTS"
echo "OGB_DATA_ROOT=$OGB_DATA_ROOT"
python run.py \
  --dataset ogbn-products \
  "${COMMON_LARGE[@]}" \
  --tag baselines_products

echo "Done. Results under results/baselines_node/"
