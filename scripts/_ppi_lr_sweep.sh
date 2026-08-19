#!/bin/zsh
# DP learning-rate x steps sweep on the PPI headline cell
# (p2=0.1, r=1, L=2, sigma=5, K=5, p1=0.01).  Epsilon depends only on
# (p1, p2, r, sigma, T, K):  T=2000 -> eps=2.57,  T=500 -> eps=1.18  (d=1e-6).
#
#   nohup caffeinate -i ./scripts/_ppi_lr_sweep.sh > results/ppi_lr_sweep.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

for T in 500 2000; do
  for LR in 0.3 1.0 3.0 10.0; do
    # lr=1.0 / T=2000 is the already-measured reference cell (0.4244); skip.
    if [[ "$LR" == "1.0" && "$T" == "2000" ]]; then continue; fi
    echo "=== lr=$LR T=$T $(date) ==="
    $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
      --model multilabel_gnn --aggr mean \
      --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T $T --sigma 5.0 \
      --clip 1.0 --lr $LR --K_in 5 --K_out 5 \
      --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
      --out_dir results/ppi_lr_sweep/lr${LR}_T${T}
  done
done

echo "=== summary $(date) ==="
$PY - <<'EOF'
import csv, glob, os
print(f"{'lr':>6} {'T':>6} {'test_mean':>10}")
print(f"{1.0:>6} {2000:>6} {'0.4244':>10}   (reference, ppi_L2_rerun)")
for d in sorted(glob.glob('results/ppi_lr_sweep/lr*_T*')):
    tag = os.path.basename(d)
    lr, T = tag[2:].split('_T')
    rows = list(csv.DictReader(open(f'{d}/sparse_gnn_ppi_dp_results.csv')))
    m = sum(float(r['test_acc']) for r in rows) / len(rows)
    print(f"{lr:>6} {T:>6} {m:>10.4f}")
EOF
echo "=== LR SWEEP COMPLETE $(date) ==="
