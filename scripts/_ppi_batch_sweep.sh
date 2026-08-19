#!/bin/zsh
# Batch size (p1) vs steps (T) at matched epsilon — the lever the clip and lr
# sweeps ruled in by elimination.
#
# Per-step SNR is  signal/noise ~ (B * mean_grad / B) / (sigma*C*sqrt(D)/B),
# i.e. proportional to B: the summed signal grows with the batch while the
# noise is added ONCE per step regardless of batch size.  Sampling more roots
# per step therefore buys signal-to-noise, at the cost of a larger per-step
# epsilon.  Accounting says the trade is favourable when sigma rises with B:
#
#   p1=0.01 B=449  T=2000 sigma=5   -> eps 2.57   (current reference, 0.4412)
#   p1=0.05 B=2245 T=400  sigma=10  -> eps 2.79   ~2.5x better per-step SNR
#   p1=0.1  B=4491 T=200  sigma=10  -> eps 4.07
#
# Every config is tracked every 25 steps, so utility can be read off at MATCHED
# epsilon rather than at a fixed step count.  lr is swept too: the update is
# lr*(S + sigma*C*z)/B, so a bigger B shrinks the step and may want a larger lr.
#
#   nohup caffeinate -i ./scripts/_ppi_batch_sweep.sh > results/ppi_batch.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

run_cell() {  # p1 sigma T lr
  local P1=$1 SIG=$2 T=$3 LR=$4
  local OUT=results/ppi_batch/p${P1}_s${SIG}_lr${LR}
  echo "=== p1=$P1 sigma=$SIG T=$T lr=$LR $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
    --model multilabel_gnn --aggr mean \
    --p1 $P1 --p2 0.1 --r 1 --num_layers 2 --T $T --sigma $SIG \
    --clip 1.0 --lr $LR --momentum 0.0 --K_in 5 --K_out 5 \
    --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
    --track_every 25 --out_dir $OUT
  $PY -u -m src.sparse.compute_epsilon \
    --csv $OUT/sparse_gnn_ppi_dp_results.csv --delta 1e-6 | tail -2
}

run_cell 0.05 10.0 800 0.3
run_cell 0.05 10.0 800 1.0
run_cell 0.1  15.0 400 0.3
run_cell 0.1  15.0 400 1.0

echo "=== summary: utility at MATCHED epsilon $(date) ==="
$PY - <<'EOF'
import csv, glob
def curve(path):
    rows = list(csv.DictReader(open(path)))
    by, eps = {}, {}
    for r in rows:
        t = int(r['step'])
        by.setdefault(t, []).append(float(r['test_acc']))
        eps[t] = float(r['epsilon'])
    return {t: (sum(v)/len(v), eps[t]) for t, v in by.items()}, rows[0]

def at_eps(c, target):
    """best utility among checkpoints with eps <= target (None if none)."""
    ok = [(u, e, t) for t, (u, e) in c.items() if e <= target]
    return max(ok) if ok else None

paths = [('results/ppi_clip/c1.0/sparse_gnn_ppi_dp_results_with_eps.csv',
          'p1=0.01 sigma=5 lr=0.3 (reference)')]
for d in sorted(glob.glob('results/ppi_batch/p*')):
    f = f'{d}/sparse_gnn_ppi_dp_results_with_eps.csv'
    rows = list(csv.DictReader(open(f)))
    paths.append((f, f"p1={rows[0]['p1']} sigma={rows[0]['sigma']} lr={rows[0]['lr']}"))

for target in (1.0, 2.6, 5.0):
    print(f"\n--- best test F1 at eps <= {target} ---")
    for f, label in paths:
        c, _ = curve(f)
        hit = at_eps(c, target)
        if hit:
            u, e, t = hit
            print(f"  {label:<38} {u:.4f}  (t={t}, eps={e:.2f})")
        else:
            print(f"  {label:<38} —  (min eps {min(e for _, e in c.values()):.2f})")
print("\ntrivial baseline = 0.4608;  non-DP at p2=0.1 r=1 = 0.5485")
EOF
echo "=== BATCH SWEEP COMPLETE $(date) ==="
