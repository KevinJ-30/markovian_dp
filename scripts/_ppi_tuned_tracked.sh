#!/bin/zsh
# Tracked DP tuning round on the PPI headline cell (p2=0.1, r=1, L=2, K=5,
# p1=0.01, T=2000 cap, checkpoints every 50 steps).  Sweeps the two levers the
# LR sweep pointed at: lower lr {0.1, 0.3} and SGD momentum {0.0, 0.9} at
# sigma=5, plus the winners' grid at sigma=20.  Each checkpoint row gets its
# own eps(t) from compute_epsilon, so every run yields a full frontier curve.
#
#   nohup caffeinate -i ./scripts/_ppi_tuned_tracked.sh > results/ppi_tuned.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

run_cell() {  # sigma lr momentum
  local SIG=$1 LR=$2 M=$3
  local OUT=results/ppi_tuned/s${SIG}_lr${LR}_m${M}
  echo "=== sigma=$SIG lr=$LR momentum=$M $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
    --model multilabel_gnn --aggr mean \
    --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma $SIG \
    --clip 1.0 --lr $LR --momentum $M --K_in 5 --K_out 5 \
    --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
    --track_every 50 --out_dir $OUT
  $PY -u -m src.sparse.compute_epsilon \
    --csv $OUT/sparse_gnn_ppi_dp_results.csv --delta 1e-6 --grid 1e-3 \
    | tail -4
}

run_cell 5.0 0.3 0.0
run_cell 5.0 0.3 0.9
run_cell 5.0 0.1 0.0
run_cell 5.0 0.1 0.9
run_cell 20.0 0.3 0.9
run_cell 20.0 0.1 0.9

echo "=== summary: best test F1 at final step per cell $(date) ==="
$PY - <<'EOF'
import csv, glob, os
print(f"{'sigma':>6} {'lr':>5} {'mom':>5} {'best_t':>7} {'best_test':>10} {'final_test':>11}")
for d in sorted(glob.glob('results/ppi_tuned/s*')):
    rows = list(csv.DictReader(open(f'{d}/sparse_gnn_ppi_dp_results.csv')))
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r['step']), []).append(float(r['test_acc']))
    means = {t: sum(v)/len(v) for t, v in by_step.items()}
    best_t = max(means, key=means.get)
    r0 = rows[0]
    print(f"{r0['sigma']:>6} {r0['lr']:>5} {r0['momentum']:>5} "
          f"{best_t:>7} {means[best_t]:>10.4f} {means[max(means)]:>11.4f}")
EOF
echo "=== TUNED TRACKED COMPLETE $(date) ==="
