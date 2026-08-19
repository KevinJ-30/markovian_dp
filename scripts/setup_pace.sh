#!/bin/bash
# One-time PACE setup for this project.  Run on an INTERACTIVE node, not login:
#   salloc -A <your-account> -q embers --mem=16GB --time=1:00:00
#   bash setup_pace.sh
set -e
SCRATCH=${SCRATCH:-$HOME/scratch}
PROJECT=/storage/project/r-pli77-0/kjacob7

echo "== 1. confirm your charge account and available QoS =="
sacctmgr show assoc user=$USER format=account,qos --noheader || true

echo "== 2. check outbound network (compute nodes are sometimes firewalled) =="
if ! timeout 20 python -c "import urllib.request as u; u.urlopen('https://pypi.org/simple/', timeout=15)" 2>/dev/null; then
  echo "  NO PyPI ACCESS from this node."
  echo "  Run steps 2-3 from the LOGIN node (they are I/O, not compute), then"
  echo "  rerun this script here to finish."
  exit 1
fi
echo "  ok"

echo "== 3. venv in scratch (wiped every 60 days; rebuild with this script) =="
mkdir -p $SCRATCH/venvs
python -m venv $SCRATCH/venvs/markovian
source $SCRATCH/venvs/markovian/bin/activate
python -m pip install --upgrade pip
# Verbose on purpose: torch is ~2 GB and a silent install looks like a hang.
python -m pip install torch torch_geometric ogb dp_accounting opacus scipy \
    pandas matplotlib pytest

echo "== 4. pre-stage Reddit into scratch (compute nodes may lack outbound net) =="
mkdir -p $SCRATCH/data
python - <<'PY'
import os
from torch_geometric.datasets import Reddit
root = os.path.join(os.environ.get('SCRATCH', os.path.expanduser('~/scratch')), 'data', 'Reddit')
ds = Reddit(root=root)
d = ds[0]
print(f"Reddit ready: {d.num_nodes:,} nodes, {d.edge_index.size(1):,} arcs, "
      f"{ds.num_features} features, {ds.num_classes} classes")
PY

echo "== 5. verify the pipeline imports and the test suite passes =="
cd $PROJECT/markovian_dp 2>/dev/null || cd $PROJECT
python -m pytest tests/ -q | tail -2

echo "== setup complete =="
