#!/bin/zsh
# Full rel-f1 driver-top3 ladder through the fixed pipeline:
# stage1 (non-DP p2 sweep) + stage2 (DP p2 x sigma grid) + epsilon.
# r=2/L=2 (a row root needs 2 hops to reach its driver's history);
# K_in=20 K_out=3 so Thm 6.4 shells are 3^d.
#
#   nohup caffeinate -i ./scripts/_relf1_ladder.sh > results/relf1_ladder.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
OUT=results/relf1_ladder

echo "=== rel-f1 stage1 (non-DP p2 sweep) $(date) ==="
$PY -u -m src.sparse.run --dataset relbench-f1-top3 --inductive --direction in \
  --model binary_gnn --aggr mean \
  --p1 0.05 --p2 1.0 0.5 0.25 0.1 --r 2 --num_layers 2 --T 900 \
  --lr 0.01 --K_in 20 --K_out 3 --dropout 0.0 --weight_decay 0.0 \
  --roots_from train --seeds 3 --out_dir $OUT

echo "=== rel-f1 stage2 (DP grid) $(date) ==="
$PY -u -m src.sparse.run --dataset relbench-f1-top3 --inductive --direction in --dp \
  --model binary_gnn --aggr mean \
  --p1 0.05 --p2 1.0 0.5 0.1 --r 2 --num_layers 2 --T 900 \
  --sigma 5.0 10.0 20.0 --clip 1.0 --lr 1.0 --K_in 20 --K_out 3 \
  --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 3 --out_dir $OUT

echo "=== rel-f1 epsilon $(date) ==="
$PY -u -m src.sparse.compute_epsilon \
  --csv $OUT/sparse_gnn_relbench-f1-top3_dp_results.csv --delta 1e-6

echo "=== REL-F1 LADDER COMPLETE $(date) ==="
