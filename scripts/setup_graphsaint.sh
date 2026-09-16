#!/bin/bash
# Extract the GraphSAINT benchmark data into the layout src/data/datasets.py expects.
#
#   ./scripts/setup_graphsaint.sh <dir-with-zips> [dest]
#
# <dir-with-zips>  where the Google Drive exports live (e.g. ~/Downloads)
# dest             default: $GRAPHSAINT_DATA_ROOT, else data/graphsaint
#
# Google Drive splits a folder download into -001, -002, ... parts, each a
# separate zip holding a SUBSET of the files -- not a multi-part archive.  So
# every part must be extracted into the same destination; unzip -o over all of
# them is correct and idempotent.
#
# Expected result, per dataset:
#     <dest>/<name>/adj_full.npz  adj_train.npz  feats.npy  role.json
#                   class_map.json          (+ labels.npy for amazon)
#
# Sizes, extracted:  ppi-large 38M   reddit 1.2G   yelp 2.2G   amazon 3.7G
# These are NOT committed -- `data/` is in .gitignore and the total is ~7 GB.

set -euo pipefail

SRC=${1:?usage: setup_graphsaint.sh <dir-with-zips> [dest]}
DEST=${2:-${GRAPHSAINT_DATA_ROOT:-data/graphsaint}}

mkdir -p "$DEST"
shopt -s nullglob

found=0
for z in "$SRC"/ppi-large-*.zip "$SRC"/flickr-*.zip "$SRC"/reddit-*.zip \
         "$SRC"/yelp-*.zip "$SRC"/amazon-*.zip; do
    echo "  extracting $(basename "$z")"
    unzip -oq "$z" -d "$DEST"
    found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
    echo "no GraphSAINT zips found in $SRC" >&2
    echo "expected names like ppi-large-<timestamp>-1-001.zip" >&2
    exit 1
fi

echo
echo "=== verifying $DEST ==="
rc=0
for d in "$DEST"/*/; do
    name=$(basename "$d")
    missing=()
    for f in adj_full.npz adj_train.npz feats.npy role.json; do
        [ -f "$d/$f" ] || missing+=("$f")
    done
    # labels come from either file
    if [ ! -f "$d/class_map.json" ] && [ ! -f "$d/labels.npy" ]; then
        missing+=("class_map.json|labels.npy")
    fi
    if [ ${#missing[@]} -eq 0 ]; then
        echo "  OK       $name  ($(du -sh "$d" | cut -f1))"
    else
        echo "  MISSING  $name: ${missing[*]}" >&2
        rc=1
    fi
done

echo
echo "point the loader at it with:  export GRAPHSAINT_DATA_ROOT=$(cd "$DEST" && pwd)"
echo "then e.g.:  python -m src.experiments.sparse --dataset ppi-large --model multilabel_gnn ..."
exit $rc
