#!/bin/zsh
# Matched-optimizer non-DP reference.
#
# Every non-DP number so far used Adam while every DP number used SGD, so the
# "cost of privacy" gap silently included the cost of changing optimizer.  This
# runs the SAME non-DP configuration under SGD across a range of lr, giving a
# reference that differs from its DP counterpart ONLY by clipping and noise.
#
# Reported alongside, not instead of, the Adam ceiling: Adam answers "best
# achievable non-privately", SGD answers "what did privacy actually cost".
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

echo "=== waiting for the K sweep to finish $(date) ==="
while ! grep -q "K SWEEP COMPLETE" results/ppi_k.log 2>/dev/null; do sleep 60; done

# Sparsified config (p2=0.1) — the arm the DP runs are compared against.
for LR in 0.03 0.1 0.3 1.0; do
  echo "=== non-DP SGD lr=$LR (p2=0.1, K=5, r=1, L=2) $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in \
    --model multilabel_gnn --aggr mean --optimizer sgd \
    --p1 0.01 --p2 0.1 --r 1 --num_layers 2 --T 2000 --lr $LR --momentum 0.0 \
    --K_in 5 --K_out 5 --dropout 0.0 --weight_decay 0.0 \
    --roots_from train --seeds 2 --track_every 50 \
    --out_dir results/ppi_optmatch/sgd_lr${LR}
done

echo "=== summary $(date) ==="
$PY - <<'EOF'
import csv, glob
def best(path, key='test_acc'):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r['step']), []).append(float(r[key]))
    m = {t: sum(v)/len(v) for t, v in by.items()}
    bt = max(m, key=m.get); return bt, m[bt]
print("non-DP, p2=0.1 K=5 r=1 L=2 — optimizer comparison")
print(f"{'optimizer':>12} {'lr':>6} {'best_f1':>9} {'best_auroc':>11}")
for d in sorted(glob.glob('results/ppi_optmatch/sgd_lr*')):
    f = glob.glob(f'{d}/*_results.csv')[0]
    rows = list(csv.DictReader(open(f)))
    _, f1 = best(f); _, au = best(f, 'test_auroc')
    print(f"{'sgd':>12} {rows[0]['lr']:>6} {f1:>9.4f} {au:>11.4f}")
print(f"{'adam':>12} {'0.01':>6} {0.5485:>9.4f} {0.7816:>11.4f}   (existing reference)")
print("\nDP (sgd, lr=0.3, sigma=5, eps~2.6): 0.4408 f1 / 0.6863 auroc")
EOF
echo "=== OPT MATCH COMPLETE $(date) ==="
