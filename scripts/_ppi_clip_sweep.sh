#!/bin/zsh
# Clipping-norm sweep on the PPI headline cell — the free lever.
#
# EPSILON DOES NOT DEPEND ON C.  The base mechanism is N(G(y), sigma^2 C^2 I)
# with ||g0|| <= C, so sensitivity and noise scale together and the dominating
# pair is stated in units of C (accounting.py takes sigma, never clip).
# Lowering C therefore cuts the injected noise at IDENTICAL epsilon; the only
# cost is bias from clipping gradients that exceeded C.
#
# Measured on PPI at init: per-root gradient norms are median 0.199, p90 0.434,
# max 0.706 — so C=1.0 clips NOTHING and simply inflates the noise ~5x.
#
# Waits for the lr/momentum job to finish, adopts its best (lr, momentum), then
# sweeps C.  Launch with:
#   nohup caffeinate -i ./scripts/_ppi_clip_sweep.sh > results/ppi_clip.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

echo "=== waiting for results/ppi_tuned.log to complete $(date) ==="
while ! grep -q "TUNED TRACKED COMPLETE" results/ppi_tuned.log 2>/dev/null; do
  sleep 60
done
echo "=== lr/momentum job done, picking its best cell $(date) ==="

BEST=$($PY - <<'EOF'
import csv, glob
best = (None, None, -1)
for d in sorted(glob.glob('results/ppi_tuned/s5.0_*')):
    rows = list(csv.DictReader(open(f'{d}/sparse_gnn_ppi_dp_results.csv')))
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r['step']), []).append(float(r['test_acc']))
    peak = max(sum(v)/len(v) for v in by_step.values())
    if peak > best[2]:
        best = (rows[0]['lr'], rows[0]['momentum'], peak)
print(f"{best[0]} {best[1]}")
EOF
)
LR=${BEST%% *}; MOM=${BEST##* }
echo "adopting lr=$LR momentum=$MOM"

for C in 1.0 0.5 0.2 0.1; do
  OUT=results/ppi_clip/c${C}
  echo "=== clip=$C (lr=$LR mom=$MOM sigma=5, eps identical across all C) $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
    --model multilabel_gnn --aggr mean \
    --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma 5.0 \
    --clip $C --lr $LR --momentum $MOM --K_in 5 --K_out 5 \
    --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
    --track_every 50 --out_dir $OUT
  $PY -u -m src.sparse.compute_epsilon \
    --csv $OUT/sparse_gnn_ppi_dp_results.csv --delta 1e-6 --grid 1e-3 | tail -3
done

echo "=== summary: clip vs utility (all at the same epsilon) $(date) ==="
$PY - <<'EOF'
import csv, glob
print(f"{'clip':>6} {'best_t':>7} {'best_test':>10} {'final_test':>11}")
for d in sorted(glob.glob('results/ppi_clip/c*'),
                key=lambda p: -float(p.rsplit('/c', 1)[1])):
    rows = list(csv.DictReader(open(f'{d}/sparse_gnn_ppi_dp_results.csv')))
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r['step']), []).append(float(r['test_acc']))
    means = {t: sum(v)/len(v) for t, v in by_step.items()}
    bt = max(means, key=means.get)
    print(f"{rows[0]['clip']:>6} {bt:>7} {means[bt]:>10.4f} {means[max(means)]:>11.4f}")
print("\ntrivial baseline = 0.4608;  non-DP reference at p2=0.1 r=1 = 0.5485")
EOF
echo "=== CLIP SWEEP COMPLETE $(date) ==="
