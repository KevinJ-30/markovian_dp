#!/bin/zsh
# K vs p2 along an ISO-EPSILON diagonal: is it better to keep few neighbours
# densely, or many neighbours sparsely?
#
# At r=1 the substitution pair has q_1 = p2 and shell size n_1 = K_out, so the
# mixture weight is Binomial(K, p1*p2) and epsilon depends on (K, p2) almost
# entirely through the PRODUCT K*p2.  Verified numerically at sigma=5, T=2000,
# delta=1e-6:
#
#     (K=5,p2=1.0) (K=10,p2=0.5) (K=20,p2=0.25) (K=50,p2=0.1)  -> eps 12.17 each
#
# The expected rooted-subgraph size is 1 + K*p2, so compute cost is matched
# along the diagonal as well.  Privacy and cost being held fixed, this isolates
# a pure utility question, and the mechanism's central claim predicts an answer:
# degree capping discards a fixed random K-of-720 arcs ONCE, before training,
# and that information never comes back; sparsification redraws every step, so
# a large-K/small-p2 run eventually sees the whole neighbourhood.  If the claim
# holds, utility should rise along the diagonal toward large K.
#
# Both non-DP and DP arms are run so "the graph helps" is separated from
# "the noise hurts".
#
#   nohup caffeinate -i ./scripts/_ppi_k_sweep.sh > results/ppi_k.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

cell() {  # K p2 dp
  local K=$1 P2=$2 DP=$3
  local OUT=results/ppi_k/K${K}_p${P2}_dp${DP}
  local FLAGS=(--lr 0.01)
  [[ "$DP" == "1" ]] && FLAGS=(--dp --sigma 5.0 --clip 1.0 --lr 0.3)
  echo "=== K=$K p2=$P2 dp=$DP $(date) ==="
  $PY -u -m src.sparse.run --dataset ppi --direction in \
    --model multilabel_gnn --aggr mean \
    --p1 0.01 --p2 $P2 --r 1 --num_layers 2 --T 2000 \
    --K_in $K --K_out $K $FLAGS --momentum 0.0 \
    --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
    --track_every 50 --out_dir $OUT
  if [[ "$DP" == "1" ]]; then
    $PY -u -m src.sparse.compute_epsilon \
      --csv $OUT/sparse_gnn_ppi_dp_results.csv --delta 1e-6 | tail -2
  fi
}

for spec in "5 1.0" "10 0.5" "20 0.25" "50 0.1"; do
  cell ${=spec} 0
done
for spec in "5 1.0" "10 0.5" "20 0.25" "50 0.1"; do
  cell ${=spec} 1
done

echo "=== summary: iso-epsilon K/p2 diagonal $(date) ==="
$PY - <<'EOF'
import csv, glob, os
def best(path, key='test_acc'):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r['step']), []).append(float(r[key]))
    means = {t: sum(v)/len(v) for t, v in by.items()}
    bt = max(means, key=means.get)
    return bt, means[bt], means[max(means)]

print("all DP cells share eps = 12.17 (sigma=5, T=2000, delta=1e-6)\n")
for dp in (0, 1):
    print(f"--- {'DP (sigma=5)' if dp else 'non-DP'} ---")
    print(f"{'K':>4} {'p2':>6} {'K*p2':>6} {'best_f1':>9} {'best_auroc':>11} {'final_f1':>9}")
    for K, p2 in ((5, 1.0), (10, 0.5), (20, 0.25), (50, 0.1)):
        d = f"results/ppi_k/K{K}_p{p2}_dp{dp}"
        f = glob.glob(f"{d}/*_results.csv")
        if not f:
            print(f"{K:>4} {p2:>6}  (missing)"); continue
        _, bf1, ff1 = best(f[0])
        try:
            _, bau, _ = best(f[0], 'test_auroc')
        except (KeyError, ValueError):
            bau = float('nan')
        print(f"{K:>4} {p2:>6} {K*p2:>6.1f} {bf1:>9.4f} {bau:>11.4f} {ff1:>9.4f}")
    print()
print("references:  all-ones 0.4608 f1 / 0.4955 auroc")
print("             true ceiling (uncapped, p2=1) 0.6998 f1 / 0.8925 auroc")
EOF
echo "=== K SWEEP COMPLETE $(date) ==="
