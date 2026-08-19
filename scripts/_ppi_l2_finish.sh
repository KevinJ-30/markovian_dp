#!/bin/zsh
# Finish the PPI L=2 DP sweep: the two remaining p2=0.1 r=2 cells, then merge
# all fragments and attach epsilon.  Run detached under caffeinate so neither
# idle sleep nor the Claude session lifecycle can kill it:
#
#   nohup caffeinate -i ./scripts/_ppi_l2_finish.sh > results/ppi_L2_rerun_c.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
SCRATCH=/private/tmp/claude-501/-Users-kevinjacob-markovian-dp-copy/101f6985-da05-419f-9855-65ff68fe6783/scratchpad

echo "=== PPI L=2 stage2 FINISH: p2=0.1 r=2 cells $(date) ==="
$PY -u -m src.sparse.run --dataset ppi --direction in --dp \
  --model multilabel_gnn --aggr mean \
  --p1 0.01 --p2 0.1 --r 2 --num_layers 2 --T 2000 --sigma 5.0 20.0 \
  --clip 1.0 --lr 1.0 --K_in 5 --K_out 5 \
  --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
  --out_dir results/ppi_L2_rerun_c

echo "=== merge fragments $(date) ==="
$PY $SCRATCH/merge_ppi_dp2.py

echo "=== epsilon $(date) ==="
$PY -u -m src.sparse.compute_epsilon \
  --csv results/ppi_L2_rerun/sparse_gnn_ppi_dp_results.csv --delta 1e-6

echo "=== PPI L=2 COMPLETE $(date) ==="
