#!/bin/zsh
# PPI with GCN aggregation, matched cell-for-cell against the mean arm.
#
# Both are valid mechanisms — Assumption 6.3 only asks that g0 be a bounded
# function of the rooted subgraph — so epsilon is identical.  What differs is
# whether the rooted computation equals full-graph inference: mean's normalizer
# reads only the TARGET's in-degree, which SparseExpand always materializes,
# while gcn's reads the SOURCE's degree, which is truncated at the subgraph
# boundary.  This measures what that costs on PPI.
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python

for AGGR in gcn mean; do
  for P2 in 1.0 0.1; do
    OUT=results/ppi/aggr/${AGGR}_p${P2}_nodp
    [[ -s $OUT/sparse_gnn_ppi_results.csv ]] || {
      echo "=== aggr=$AGGR p2=$P2 non-DP $(date) ==="
      $PY -u -m src.sparse.run --dataset ppi --direction in \
        --model multilabel_gnn --aggr $AGGR \
        --p1 0.01 --p2 $P2 --r 1 --num_layers 2 --T 2000 --lr 0.01 \
        --K_in 5 --K_out 5 --dropout 0.0 --weight_decay 0.0 \
        --roots_from train --seeds 2 --track_every 50 --out_dir $OUT
    }
    OUT=results/ppi/aggr/${AGGR}_p${P2}_dp
    [[ -s $OUT/sparse_gnn_ppi_dp_results.csv ]] || {
      echo "=== aggr=$AGGR p2=$P2 DP sigma=5 $(date) ==="
      $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
        --model multilabel_gnn --aggr $AGGR \
        --p1 0.01 --p2 $P2 --r 1 --num_layers 2 --T 2000 --sigma 5.0 \
        --clip 1.0 --lr 0.3 --K_in 5 --K_out 5 --dropout 0.0 --weight_decay 0.0 \
        --roots_from train --seeds 2 --track_every 50 --out_dir $OUT
    }
  done
done

echo "=== summary: gcn vs mean $(date) ==="
$PY - <<'EOF'
import csv, glob, os
def best(path, key='test_acc'):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r['step']), []).append(float(r[key]))
    m = {t: sum(v)/len(v) for t, v in by.items()}
    b = max(m, key=m.get); return m[b]
print(f"{'cell':<20} {'gcn':>9} {'mean':>9} {'gcn-mean':>10}")
for p2 in ('1.0', '0.1'):
    for tag in ('nodp', 'dp'):
        row = {}
        for aggr in ('gcn', 'mean'):
            g = glob.glob(f'results/ppi/aggr/{aggr}_p{p2}_{tag}/*_results.csv')
            if g: row[aggr] = best(g[0])
        if len(row) == 2:
            print(f"p2={p2:<5} {tag:<10} {row['gcn']:>9.4f} {row['mean']:>9.4f} "
                  f"{row['gcn']-row['mean']:>+10.4f}")
EOF
echo "=== GCN ARM COMPLETE $(date) ==="
