#!/bin/bash
#SBATCH -Jrel_trial
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=128G
#SBATCH -t12:00:00
#SBATCH -oReport-rel-trial-%j.out

# rel-trial study-outcome / study-adverse, hubs dropped and replicated.
# Arms: MLP, non-DP ceiling, non-DP matched, DP at eps 2/8. delta = 1/n_train_nodes.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Not scripts/_ice_env.sh: its accountant check fails since chi=1.
SCRATCH=${SCRATCH:-$([ -d "$HOME/scratch" ] && echo "$HOME/scratch" || echo "$HOME")}
if [ -z "${VENV:-}" ]; then
  for cand in "$SCRATCH/venvs/markovian" "$SCRATCH/markovian_env" "$SCRATCH/venvs/markovian_env"; do
    [ -f "$cand/bin/activate" ] && { VENV="$cand"; break; }
  done
fi
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
PY=${PY:-python}
export PYTHONNOUSERSITE=1
export RELBENCH_CACHE_DIR=${RELBENCH_CACHE_DIR:-$SCRATCH/data/relbench_cache}
export HF_HOME=${HF_HOME:-$SCRATCH/data/hf_cache}
$PY -c "import torch, torch_geometric, dp_accounting, relbench; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "FATAL: missing dependency" >&2; exit 1; }

EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-1024}
K=${K:-10}
P2=${P2:-0.5}
R=${R:-2}
HIDDEN=${HIDDEN:-128}
LR=${LR:-0.01}
EPS_LIST=${EPS_LIST:-"2 8"}
HUBS_LIST=${HUBS_LIST:-"drop replicate"}
OUT_ROOT=${OUT_ROOT:-results/rel_trial}

run() {  # keep going on failure
  local label=$1; shift
  echo; echo "=== $label  $(date) ==="
  "$@" || echo "!!! FAILED: $label (exit $?)"
}

for TASK in study-outcome study-adverse; do
  DS="relbench:rel-trial/$TASK"
  if [ "$TASK" = study-outcome ]; then MODEL=binary_gnn; else MODEL=regression_gnn; fi
  OUT=$OUT_ROOT/$TASK
  mkdir -p "$OUT"
  echo; echo "################ $DS ################"

  # MLP also writes info.json.
  run "$TASK mlp" $PY -u scripts/rel_trial_mlp.py --dataset "$DS" \
      --out_dir "$OUT" --epochs "$EPOCHS" --batch_size "$BATCH" --hidden "$HIDDEN"
  [ -f "$OUT/info.json" ] || { echo "no info.json for $TASK; skipping"; continue; }
  read -r NROWS DELTA < <($PY -c "import json; i=json.load(open('$OUT/info.json')); print(i['n_train_rows'], repr(i['delta']))")
  P1=$($PY -c "print(f'{min(1.0, $BATCH/$NROWS):.8f}')")
  T=$($PY -c "print(max(1, round($EPOCHS / $P1)))")
  echo "  n_train_rows=$NROWS p1=$P1 T=$T delta=$DELTA"

  COMMON=(--dataset "$DS" --model "$MODEL" --relbench_root entity --aggr mean
          --direction in --p1 "$P1" --T "$T" --num_layers 2 --hidden "$HIDDEN"
          --optimizer adam --lr "$LR" --weight_decay 0 --seeds 1
          --eval_every "$T")

  for HUBS in $HUBS_LIST; do
    O=$OUT/$HUBS
    run "$TASK $HUBS nodp ceiling (p2=1, r=$R, uncapped)" \
        $PY -u -m src.experiments.sparse "${COMMON[@]}" --relbench_hubs "$HUBS" \
        --p2 1.0 --r "$R" --out_dir "$O/nodp_ceiling"
    run "$TASK $HUBS nodp matched (p2=$P2, r=$R, K=$K)" \
        $PY -u -m src.experiments.sparse "${COMMON[@]}" --relbench_hubs "$HUBS" \
        --p2 "$P2" --r "$R" --K_in "$K" --K_out "$K" --out_dir "$O/nodp_matched"
    for EPS in $EPS_LIST; do
      run "$TASK $HUBS dp eps=$EPS" \
          $PY -u -m src.experiments.sparse "${COMMON[@]}" --relbench_hubs "$HUBS" \
          --p2 "$P2" --r "$R" --K_in "$K" --K_out "$K" --dp --clip 1.0 \
          --target_epsilon "$EPS" --target_delta "$DELTA" --accounting_grid 1e-3 \
          --out_dir "$O/dp_eps$EPS"
    done
  done
done

echo; echo "=== summary  $(date) ==="
$PY - "$OUT_ROOT" <<'SUMMARY'
import csv, glob, json, os, sys
root = sys.argv[1]
for task in sorted(os.listdir(root)):
    tdir = os.path.join(root, task)
    mlp = os.path.join(tdir, 'mlp_result.json')
    if os.path.exists(mlp):
        r = json.load(open(mlp))
        print(task, 'mlp', {k: round(v, 4) for k, v in r.items()
                            if k.startswith(('validation_', 'test_')) and isinstance(v, float)})
    for path in sorted(glob.glob(os.path.join(tdir, '*', '*', '*.csv'))):
        arm = os.path.relpath(os.path.dirname(path), tdir)
        for row in csv.DictReader(open(path)):
            keep = {k: row[k] for k in row
                    if k.startswith(('val', 'test', 'calibrated', 'sigma', 'trivial')) and row[k] != ''}
            print(task, arm, keep)
SUMMARY
