#!/usr/bin/env bash
# sigma-COVERAGE sweep over the (L, r) depth matrix, for any dataset.
#
#   DS=facebook SEEDS=2 bash scripts/_coverage_sweep.sh
#   DS=flickr   SEEDS=2 bash scripts/_coverage_sweep.sh
#   DS=ppi      SEEDS=2 T=1000 bash scripts/_coverage_sweep.sh
#
# This is deliberately NOT a target-epsilon calibration.  The union-graph degree
# question (Assumption 5.2 bounds g v g', we cap only g) is unresolved and may
# relabel epsilon by ~2x, so calibrating sigma to fixed budgets now would weld
# today's accounting into the runs.  Instead: sweep sigma widely with
# --track_every, so each run traces a whole epsilon curve and consecutive sigma
# windows overlap.  Coverage is then continuous and the target-eps points can be
# selected post-hoc under whatever accounting we settle on.
#
# Coverage at r=1 (facebook params, T=500, tracked from step 25):
#   sigma    eps as accounted now (K=5)   eps union-safe (K=10)
#       2          6.02 -  23.79                11.07 - 51.91
#       5          1.38 -   7.21                 2.66 - 14.83
#      10          0.64 -   3.30                 1.23 -  6.53
#      20          0.31 -   1.56                 0.58 -  3.00
#      40          0.15 -   0.75                 0.28 -  1.42
#      80          0.07 -   0.37                 0.13 -  0.69
# Both bracket {0.5, 1, 2, 4, 8}.  r=2 is ~5x pricier, hence the sigma=160 arm;
# even so r=2 may not reach eps=0.5 union-safe -- that is itself a result.
#
# Depth matrix: (L=1,r=1), (L=2,r=1), (L=2,r=2).  --r sweeps internally but
# --num_layers does not, so L needs its own invocation.  Note
# _dataset_settings.sh pins L=2 for the ladder scripts because L=1 measured
# model capacity rather than sparsification on PPI; the L=1 arm here is the
# "faithful receptive field" (L == r) comparison, so expect it to be
# capacity-limited.
#
# Re-running skips cells whose CSV already exists, so it resumes after a laptop
# sleep or a Ctrl-C.

set -u

DS=${DS:?set DS=<dataset>, e.g. DS=facebook}
# The project's deps live in the PytorchEnv conda env (torch 2.8, PyG 2.7,
# xgboost, sklearn); the base python does not have the full stack.
# --no-capture-output so progress streams instead of appearing at the end.
PY=${PY:-"conda run --no-capture-output -n PytorchEnv python"}

# _dataset_settings.sh assigns T and SEEDS unconditionally, so stash any env
# override before sourcing and reapply after.
_T_ENV=${T:-}
_SEEDS_ENV=${SEEDS:-}

source "$(dirname "$0")/_dataset_settings.sh" "$DS"

T=${_T_ENV:-$T}
SEEDS=${_SEEDS_ENV:-2}
DELTA=${DELTA:-1e-6}
SIGMAS=${SIGMAS:-"2 5 10 20 40 80 160"}
P2=${P2:-"1.0 0.1"}
OUT_ROOT=${OUT_ROOT:-results/coverage_$TAG}

# ${arr[@]+"${arr[@]}"} — macOS ships bash 3.2, where expanding an empty array
# under `set -u` is an "unbound variable" error.  INDUCTIVE is empty for the
# natively-inductive and the transductive datasets alike.
COMMON=(--dataset "$DS" --direction in "${MODEL[@]}"
        ${INDUCTIVE[@]+"${INDUCTIVE[@]}"}
        --p1 "$P1" "${CAP[@]}" --clip "$CLIP" --hidden 16 "${REG[@]}"
        --roots_from train --seeds "$SEEDS" --T "$T" --track_every 25)

echo "=== coverage sweep: $DS ==="
echo "  p1=$P1  T=$T  seeds=$SEEDS  cap=${CAP[*]}  inductive=${INDUCTIVE[*]:-no}  lr_dp=$LR_DP"
echo "  sigma: $SIGMAS"
echo "  p2:    $P2"
echo "  out:   $OUT_ROOT"
echo

run_cell() {
  local out=$1; shift
  if compgen -G "$out/sparse_gnn_*_results.csv" > /dev/null; then
    echo "  [skip] $out"; return 0
  fi
  mkdir -p "$out"
  echo "  [run ] $out"
  # shellcheck disable=SC2086
  $PY -u -m src.sparse.run "${COMMON[@]}" "$@" --out_dir "$out" || return 1
  local csv
  csv=$(ls "$out"/sparse_gnn_*_dp_results.csv 2>/dev/null | head -1)
  [ -n "${csv:-}" ] && $PY -u -m src.sparse.compute_epsilon --csv "$csv" \
      --delta "$DELTA" | tail -4
  return 0
}

# shellcheck disable=SC2086
for depth in "1 1" "2 1" "2 2"; do
  set -- $depth; L_=$1; R_=$2
  echo "--- L=$L_  r=$R_ ---"
  run_cell "$OUT_ROOT/nodp_L${L_}_r${R_}" \
      --num_layers "$L_" --r "$R_" --p2 $P2 --optimizer adam --lr "$LR_NONDP"
  run_cell "$OUT_ROOT/dp_L${L_}_r${R_}" \
      --num_layers "$L_" --r "$R_" --p2 $P2 --optimizer sgd --lr "$LR_DP" \
      --dp --sigma $SIGMAS
done

echo
echo "done.  results under $OUT_ROOT/"
