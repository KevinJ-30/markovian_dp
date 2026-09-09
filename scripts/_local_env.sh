# Local (non-cluster) environment setup, mirroring scripts/_ice_env.sh's
# preflight but pointed at whatever conda env on THIS machine actually has
# relbench + torch + torch_geometric together (checked 2026-09-09: PytorchEnv).
#
#   source scripts/_local_env.sh
#
# Override PY to point at a different interpreter; RelBench's own default
# cache dir (~/Library/Caches/relbench on macOS) is left alone deliberately —
# that is where the already-downloaded rel-f1/rel-hm databases live, and this
# script exists specifically to avoid triggering any new downloads.

PY=${PY:-/Users/kevinjacob/anaconda3/envs/PytorchEnv/bin/python3}

echo "--- env (local) ---"
echo "  PY = $PY  ($($PY --version 2>&1))"

$PY -u - <<'PREFLIGHT' || { echo "FATAL: preflight failed (see above)" >&2; exit 1; }
import importlib, sys, time
missing = []
for mod in ('numpy', 'scipy', 'dp_accounting', 'torch', 'torch_geometric', 'relbench'):
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
