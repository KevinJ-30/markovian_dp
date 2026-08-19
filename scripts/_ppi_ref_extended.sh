#!/bin/zsh
# Extend the reference config (p1=0.01, B=449, sigma=5, lr=0.3, C=1.0) to
# T=6000 so its privacy-utility curve reaches eps ~5.5.  Without this the
# batch sweep's "eps <= 5" row compares a large-batch checkpoint at eps=4.12
# against a reference that simply stops at eps=2.57 — not a like-for-like
# comparison.  Tracked every 50 steps.
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

echo "=== waiting for the batch sweep to finish $(date) ==="
while ! grep -q "BATCH SWEEP COMPLETE" results/ppi_batch.log 2>/dev/null; do sleep 60; done

echo "=== reference extended to T=6000 $(date) ==="
$PY -u -m src.sparse.run --dataset ppi --direction in --dp \
  --model multilabel_gnn --aggr mean \
  --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 6000 --sigma 5.0 \
  --clip 1.0 --lr 0.3 --momentum 0.0 --K_in 5 --K_out 5 \
  --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
  --track_every 50 --out_dir results/ppi_ref_ext
$PY -u -m src.sparse.compute_epsilon \
  --csv results/ppi_ref_ext/sparse_gnn_ppi_dp_results.csv --delta 1e-6 | tail -2

echo "=== curve: reference out to eps~5.5 $(date) ==="
$PY - <<'EOF'
import csv
rows = list(csv.DictReader(open(
    'results/ppi_ref_ext/sparse_gnn_ppi_dp_results_with_eps.csv')))
by, eps = {}, {}
for r in rows:
    t = int(r['step']); by.setdefault(t, []).append(float(r['test_acc'])); eps[t] = float(r['epsilon'])
print(f"{'step':>6} {'eps':>7} {'test':>8}")
for t in sorted(by):
    if t % 500 == 0:
        print(f"{t:>6} {eps[t]:>7.3f} {sum(by[t])/len(by[t]):>8.4f}")
best = max(by, key=lambda t: sum(by[t])/len(by[t]))
print(f"\nbest {sum(by[best])/len(by[best]):.4f} at t={best}, eps={eps[best]:.3f}")
print("trivial baseline 0.4608")
EOF
echo "=== REF EXTENDED COMPLETE $(date) ==="
