#!/usr/bin/env bash
# ogbn-arxiv (inductive): does r=2 restore the GNN's edge over blind?
#
#   nohup caffeinate -i bash scripts/_arxiv_r2_diag.sh \
#       > results/logs/arxiv_r2_diag.log 2>&1 &
#
# WHY.  arxiv's matched-epsilon frontier (results/arxiv_matched_eps, complete)
# showed the opposite of facebook and PPI: dense beat every sparser setting at
# every eps, and the graph-blind arm tied or beat the GNN for eps>=2.  The
# leading suspect is r=1 itself -- the measured mean rooted-subgraph size at
# r=1 is only 2.46 nodes on arxiv, versus 4.52 on PPI and 5.55 on facebook, so
# there may simply not be enough graph at r=1 for sparsification to trade away.
#
# This is deliberately non-DP and r=1 vs r=2 ONLY: it tests whether the GNN's
# non-private edge over blind (currently +2.1 pts at hidden=256, r=1) grows at
# r=2, before spending any budget on a wider DP re-sweep.  If it does not grow,
# r=1 is not the explanation and the anomaly needs a different diagnosis.
#
# hidden=256, K=5, p1, T match the matched-epsilon sweep's settings exactly --
# so r=1 and r=0 are ALREADY on disk (results/arxiv_matched_eps/nodp_h256 and
# nodp_blind_h256, pulled from ICE) and are not rerun here.  This script adds
# the one new cell, r=2, and reports it against those existing numbers.

set -u
cd "$(dirname "$0")/.."

PY=${PY:-/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python}
OUT_ROOT=${OUT_ROOT:-results/arxiv_r2_diag}
P1=0.0056          # matches arxiv_inductive_matched_eps.sbatch: 512/90,941 train
T=500
K=5
SEEDS=${SEEDS:-3}

mkdir -p "$OUT_ROOT" results/logs

BASE=(--dataset ogbn-arxiv --direction in --aggr mean --inductive
      --p1 $P1 --hidden 256 --K_in $K --K_out $K --num_layers 2
      --clip 1.0 --dropout 0.0 --weight_decay 0.0 --roots_from train
      --seeds "$SEEDS" --T $T --optimizer adam --lr 0.01)

run_cell() {   # out_dir, then extra flags
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_ogbn-arxiv*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out  $*"
  $PY -u -m src.sparse.run "${BASE[@]}" "$@" --out_dir "$out"
}

echo "=== arxiv r=2 diagnostic (non-DP) $(date) ==="
echo "    prior refs (hidden=256): GNN r=1 0.5427 | blind r=0 0.5213"

run_cell "$OUT_ROOT/gnn_r2" --p2 1.0 --r 2

echo
echo "=== complete $(date) ==="
$PY -c "
import csv
f = '$OUT_ROOT/gnn_r2/sparse_gnn_ogbn-arxiv_results.csv'
rows = list(csv.DictReader(open(f)))
T = max(int(float(r['step'])) for r in rows)
acc = [float(r['test_acc']) for r in rows if int(float(r['step'])) == T]
r2 = sum(acc) / len(acc)
r1, blind = 0.5427, 0.5213
print(f'  r=1  {r1:.4f}  (edge over blind: {r1-blind:+.4f})')
print(f'  r=2  {r2:.4f}  (edge over blind: {r2-blind:+.4f})')
print(f'  blind    {blind:.4f}')
print()
if r2 - blind > (r1 - blind) * 1.1:
    print('  -> edge GROWS at r=2: r=1 starvation is a plausible explanation')
elif r2 - blind < (r1 - blind) * 0.9:
    print('  -> edge SHRINKS at r=2: r=1 is not the explanation, look elsewhere')
else:
    print('  -> edge roughly UNCHANGED: r=1 is not the explanation')
"
