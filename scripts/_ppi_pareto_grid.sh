#!/bin/zsh
# Full p2 x sigma grid on PPI with tracking, so the frontier is the Pareto
# envelope over CONFIGURATIONS rather than one configuration's trajectory.
#
# Tracking already sweeps epsilon within a run (40 checkpoints each), so only
# the mechanism parameters need enumerating.  r=1 only: r=2 was established to
# cost K_out^2 in epsilon for no utility gain.
set -e
cd "/Users/kevinjacob/markovian_dp copy"
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
for P2 in 1.0 0.5 0.25 0.1; do
  for SIG in 2.0 5.0 20.0; do
    OUT=results/ppi/pareto/p${P2}_s${SIG}
    [[ -s $OUT/sparse_gnn_ppi_dp_results_with_eps.csv ]] && continue
    echo "=== p2=$P2 sigma=$SIG $(date) ==="
    $PY -u -m src.sparse.run --dataset ppi --direction in --dp \
      --model multilabel_gnn --aggr mean \
      --p1 0.01 --p2 $P2 --r 1 --num_layers 2 --T 2000 --sigma $SIG \
      --clip 1.0 --lr 0.3 --momentum 0.0 --K_in 5 --K_out 5 \
      --dropout 0.0 --weight_decay 0.0 --roots_from train --seeds 2 \
      --track_every 50 --out_dir $OUT
    $PY -u -m src.sparse.compute_epsilon \
      --csv $OUT/sparse_gnn_ppi_dp_results.csv --delta 1e-6 | tail -2
  done
done
echo "=== PARETO GRID COMPLETE $(date) ==="
