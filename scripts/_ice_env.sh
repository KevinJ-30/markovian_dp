# Shared environment setup for the ICE sbatch jobs.  Source, do not execute.
#
#   source scripts/_ice_env.sh
#
# Fails LOUDLY and early.  An 8-hour job that dies on its first import an hour
# into the queue is worse than one that never starts, so every dependency the
# sweeps need is checked here before any training begins.
#
# The $SCRATCH VARIABLE is not exported on ICE (verified 2026-09-03: `ls
# $SCRATCH/...` expanded to `/...`), but the DIRECTORY exists at ~/scratch and
# already holds venvs/, markovian_env/, data/, ogb_data/, pyg_data/ and the repo
# itself.  So resolve it rather than trusting the variable.  Note the /scratch
# mount visible in df is node-local and does not persist between jobs -- it is
# not this directory.
#
# Override any of these from the sbatch script or the submit line:
#   SCRATCH     base dir (default: $SCRATCH if set, else ~/scratch, else $HOME)
#   VENV        venv to activate (default: first candidate below that exists)
#   PY          python to use (default: whatever `python` resolves to after that)
#   DATA_ROOT   dataset cache location (default: $SCRATCH/data)

if [ -z "${SCRATCH:-}" ]; then
  if [ -d "$HOME/scratch" ]; then SCRATCH="$HOME/scratch"; else SCRATCH="$HOME"; fi
fi

# Both scratch/venvs/markovian and scratch/markovian_env exist in that tree; try
# the likely layouts in order and report which one was taken, so a job that ends
# up on the wrong interpreter is obvious in the log rather than silent.
if [ -z "${VENV:-}" ]; then
  for cand in "$SCRATCH/venvs/markovian" "$SCRATCH/markovian_env" \
              "$SCRATCH/venvs/markovian_env"; do
    [ -f "$cand/bin/activate" ] && { VENV="$cand"; break; }
  done
fi
if [ -n "${VENV:-}" ]; then
  if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  else
    echo "FATAL: VENV=$VENV has no bin/activate" >&2
    exit 1
  fi
else
  echo "WARNING: no venv found under $SCRATCH; using ambient python" >&2
fi

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-$SCRATCH/data}
CACHE_ROOT=${CACHE_ROOT:-$SCRATCH/.cache_markovian}

# ~/.local/lib/python3.10/site-packages holds a BROKEN torch (verified
# 2026-09-03: libtorch_global_deps.so missing).  User-site can shadow a venv, so
# a job that picked it up would die on `import torch` with a confusing dlopen
# error rather than anything about this project.  Shut user-site out entirely.
export PYTHONNOUSERSITE=1

mkdir -p "$DATA_ROOT" "$CACHE_ROOT"

export PPI_DATA_ROOT=${PPI_DATA_ROOT:-$DATA_ROOT/PPI}
export TORCH_HOME=$CACHE_ROOT/torch
export MPLCONFIGDIR=$CACHE_ROOT/mpl
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export RELBENCH_CACHE_DIR=${RELBENCH_CACHE_DIR:-$DATA_ROOT/relbench_cache}
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "--- env ---"
echo "  SCRATCH    = $SCRATCH"
echo "  VENV       = ${VENV:-<none>}"
echo "  PY         = $(command -v $PY)  ($($PY --version 2>&1))"
echo "  DATA_ROOT  = $DATA_ROOT"
echo "  CACHE_ROOT = $CACHE_ROOT"
echo "  threads    = $OMP_NUM_THREADS"

# Preflight.  dp_accounting is the one most likely to be missing: it is only
# needed by the accountant, so a torch-only env passes every other check and
# then fails at the first calibration call.
# -u and a per-module line: a cold torch/torch_geometric import off the shared
# NFS home can take minutes on a login node, and a silent wait is
# indistinguishable from a hang.
$PY -u - <<'PREFLIGHT' || { echo "FATAL: preflight failed (see above)" >&2; exit 1; }
import importlib, sys, time
missing = []
for mod in ('numpy', 'scipy', 'dp_accounting', 'torch', 'torch_geometric'):
    t0 = time.time()
    print(f"  importing {mod:<16}", end='', flush=True)
    try:
        importlib.import_module(mod)
        print(f" ok ({time.time() - t0:.1f}s)")
    except Exception as exc:
        print(f" FAILED ({type(exc).__name__})")
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    print("missing/broken imports:", file=sys.stderr)
    for m in missing:
        print("   ", m, file=sys.stderr)
    sys.exit(1)
import torch
print(f"  torch      = {torch.__version__} (cuda={torch.cuda.is_available()})")
PREFLIGHT

# The accountant must import and reproduce a known value, or every epsilon this
# job reports is suspect.  7.2143 is the facebook cell
# (p1=0.013, p2=1, r=1, K=5, sigma=5, T=500, delta=1e-6, grid=1e-4).
$PY -u - <<'REGRESSION' || { echo "FATAL: accountant regression failed" >&2; exit 1; }
import sys
sys.path.insert(0, '.')
from src.sparse.accounting import sparsegnn_substitution_epsilon as EPS
e = EPS(p1=0.013, p2=1.0, r=1, K_in=5, K_out=5, sigma=5.0,
        steps=500, delta=1e-6, direction='in', grid=1e-4)
ok = abs(e - 7.2143) < 1e-3
print(f"  accountant = {e:.4f} (expect 7.2143) {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
REGRESSION
echo "-----------"
