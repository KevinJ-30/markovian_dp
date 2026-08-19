#!/bin/zsh
# Joint (clip, lr) sweep at FIXED NOISE — the experiment the first clip sweep
# should have been.
#
# The DP update is   lr * (S + sigma*C*z) / B,  S = sum of clipped per-root
# gradients.  Varying C at fixed lr changes BOTH terms, so that sweep was
# confounded: it read as "small C is bad" when small C also shrank every step.
#
# Holding the product lr*C CONSTANT fixes the noise contribution exactly.  Then,
# while C stays above the typical gradient norm (no clipping), S is unchanged
# and the SIGNAL term grows as 1/C — so smaller C should be strictly better
# until clipping bias bites.  Probe data (results/ppi_gradnorm.log) shows norms
# GROW during training (median 0.19 -> 0.27 by step 250, 10.7% already above
# 0.5), which is why C <= 0.5 at fixed lr lost signal.
#
# lr*C = 0.3 throughout, matching the best cell so far (C=1.0, lr=0.3 -> 0.4412).
#
#   nohup caffeinate -i ./scripts/_ppi_clip_lr_coupled.sh > results/ppi_clip_lr.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

run_cell() {  # clip lr
  local C=$1 LR=$2
  local OUT=results/ppi_clip_lr/c${C}_lr${LR}
  echo "=== clip=$C lr=$LR (lr*C=0.3, identical noise, identical eps) $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
    --model multilabel_gnn --aggr mean \
    --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --sigma 5.0 \
    --clip $C --lr $LR --momentum 0.0 --K_in 5 --K_out 5 \
    --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
    --track_every 50 --out_dir $OUT
}

run_cell 0.5 0.6
run_cell 0.2 1.5
run_cell 0.1 3.0
run_cell 0.05 6.0

echo "=== summary: fixed-noise clip sweep $(date) ==="
$PY - <<'EOF'
import csv, glob
rows0 = list(csv.DictReader(open(
    'results/ppi_clip/c1.0/sparse_gnn_ppi_dp_results.csv')))
def curve(rows):
    by = {}
    for r in rows:
        by.setdefault(int(r['step']), []).append(float(r['test_acc']))
    return {t: sum(v)/len(v) for t, v in by.items()}
print(f"{'clip':>6} {'lr':>5} {'best_t':>7} {'best_test':>10} {'final_test':>11}")
m = curve(rows0); bt = max(m, key=m.get)
print(f"{'1.0':>6} {'0.3':>5} {bt:>7} {m[bt]:>10.4f} {m[max(m)]:>11.4f}   (reference)")
for d in sorted(glob.glob('results/ppi_clip_lr/c*_lr*'),
                key=lambda p: -float(p.rsplit('/c', 1)[1].split('_lr')[0])):
    rows = list(csv.DictReader(open(f'{d}/sparse_gnn_ppi_dp_results.csv')))
    m = curve(rows); bt = max(m, key=m.get)
    print(f"{rows[0]['clip']:>6} {rows[0]['lr']:>5} {bt:>7} {m[bt]:>10.4f} "
          f"{m[max(m)]:>11.4f}")
print("\ntrivial baseline = 0.4608;  non-DP at p2=0.1 r=1 = 0.5485")
print("eps is identical for every row (depends on sigma, not C or lr):")
print("  t=250 -> 0.81   t=1000 -> 1.74   t=2000 -> 2.57   (delta=1e-6)")
EOF
echo "=== COUPLED SWEEP COMPLETE $(date) ==="
