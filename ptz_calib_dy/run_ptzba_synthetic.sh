#!/bin/bash
# Offline stage (PTZ-BA) on the Synthetic dataset - mirrors PTZ-Calib/run_ptzba_synthetic.sh
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA=${DATA:-../PTZ-Calib/data/synthetic}
OUT=${OUT:-output-synthetic-offline}
EVAL_DIR=${EVAL_DIR:-../PTZ-Calib/scripts}

for i in 01 02 03 04 05 06 07 08 09 10; do
    $PY scripts/run_ptz_ba.py \
        -i "$DATA/offline/scene_$i" \
        -f "$DATA/offline_matches/scene_$i" \
        -a "$DATA/offline/scene_$i/scene_$i.json" \
        -o "$OUT"
done

for i in 01 02 03 04 05 06 07 08 09 10; do
    echo "--- scene_$i ---"
    $PY "$EVAL_DIR/eval_synthetic.py" --pred "$OUT/scene_$i.json" --gt "$DATA/gt/scene_$i.json"
done

$PY "$EVAL_DIR/agg_synthetic.py" --pred_dir "$OUT" --gt_dir "$DATA/gt" --label "synthetic offline (python)"
