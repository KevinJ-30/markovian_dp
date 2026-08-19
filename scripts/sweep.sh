#!/bin/zsh
# Tuning sweeps: vary one axis at a time around a fixed base configuration.
#
#   ./scripts/sweep.sh <axis> [dataset]
#   ./scripts/sweep.sh lr ppi
#
# Axes:
#   lr         learning rate x steps
#   momentum   SGD momentum (post-processing; no privacy cost)
#   clip       clipping norm C at fixed lr, and at fixed lr*C (fixed noise)
#   batch      root-sampling rate p1 with sigma raised to hold epsilon
#   k          degree cap along the iso-epsilon K/p2 diagonal (K*p2 fixed)
#   optimizer  non-DP SGD reference, so the DP gap excludes the optimizer change
#
# Every DP cell is tracked and given per-checkpoint epsilon, so cells are
# compared at matched epsilon rather than at a fixed step count.  Note epsilon
# is independent of clip, momentum, lr, and the optimizer — only p1, p2, r, K,
# sigma, and T move it.
#
#   nohup caffeinate -i ./scripts/sweep.sh lr > results/logs/sweep_lr.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python
AXIS=${1:?usage: sweep.sh lr|momentum|clip|batch|k|optimizer [dataset]}
DS=${2:-ppi}
source scripts/_dataset_settings.sh $DS
OUT_ROOT=results/${TAG}/sweep_${AXIS}

# Base cell: best known DP configuration for this dataset.
BASE_P2=0.1; BASE_R=1; BASE_SIGMA=5.0; BASE_CLIP=1.0; BASE_LR=0.3

cell() {  # name, then extra flags overriding the base
  local NAME=$1; shift
  echo "=== $AXIS/$NAME $(date) ==="
  $PY -u -m src.sparse.run --dataset $DS --direction in $MODEL $REG \
      --p1 $P1 --p2 $BASE_P2 --r $BASE_R --num_layers $L --T $T \
      --K_in ${CAP[2]} --K_out ${CAP[4]} \
      --clip $BASE_CLIP --lr $BASE_LR --momentum 0.0 \
      --roots_from train --seeds 2 --track_every 50 \
      --out_dir $OUT_ROOT/$NAME "$@"
  local CSV=$OUT_ROOT/$NAME/sparse_gnn_${TAG}_dp_results.csv
  [[ -f $CSV ]] && $PY -u -m src.sparse.compute_epsilon --csv $CSV --delta $DELTA | tail -2
  return 0
}

case $AXIS in
  lr)
    for LR in 0.03 0.1 0.3 1.0 3.0; do
      cell lr$LR --dp --sigma $BASE_SIGMA --lr $LR
    done ;;
  momentum)
    for M in 0.0 0.9; do
      for LR in 0.1 0.3; do
        cell m${M}_lr${LR} --dp --sigma $BASE_SIGMA --lr $LR --momentum $M
      done
    done ;;
  clip)
    # Fixed lr: shrinking C shrinks the whole step, so this axis is confounded.
    for C in 1.0 0.5 0.2 0.1; do
      cell fixedlr_c$C --dp --sigma $BASE_SIGMA --clip $C
    done
    # Fixed lr*C = 0.3: noise held constant, so only the signal term moves.
    for pair in "0.5 0.6" "0.2 1.5" "0.1 3.0"; do
      set -- ${=pair}
      cell fixednoise_c$1 --dp --sigma $BASE_SIGMA --clip $1 --lr $2
    done ;;
  batch)
    # Larger batches raise per-step SNR but cost epsilon; sigma compensates.
    for spec in "0.05 10.0 800" "0.1 15.0 400"; do
      set -- ${=spec}
      for LR in 0.3 1.0; do
        cell p$1_s$2_lr$LR --dp --p1 $1 --sigma $2 --T $3 --lr $LR
      done
    done ;;
  k)
    # K*p2 held fixed => epsilon fixed (at r=1) and expected subgraph size fixed.
    for spec in "5 1.0" "10 0.5" "20 0.25" "50 0.1"; do
      set -- ${=spec}
      cell K$1_p$2_nodp --p2 $2 --K_in $1 --K_out $1 --lr $LR_NONDP
      cell K$1_p$2_dp   --dp --sigma $BASE_SIGMA --p2 $2 --K_in $1 --K_out $1
    done ;;
  optimizer)
    for LR in 0.03 0.1 0.3 1.0; do
      cell sgd_lr$LR --optimizer sgd --lr $LR
    done ;;
  *)
    echo "unknown axis: $AXIS" >&2; exit 2 ;;
esac

echo "=== summary $(date) ==="
$PY scripts/summarize_sweep.py $OUT_ROOT
echo "=== SWEEP $AXIS COMPLETE $(date) ==="
