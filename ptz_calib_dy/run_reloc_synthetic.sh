#!/bin/bash
# Online stage (PTZ-Reloc) on the Synthetic dataset - mirrors PTZ-Calib/run_reloc_synthetic.sh
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA=${DATA:-../PTZ-Calib/data/synthetic}
REF=${REF:-output-synthetic-offline}
OUT=${OUT:-output-synthetic-online}
EVAL_DIR=${EVAL_DIR:-../PTZ-Calib/scripts}

for i in 01 02 03 04 05 06 07 08 09 10; do
    $PY scripts/run_ptz_reloc.py \
        --ref_images "$DATA/offline/scene_$i" \
        --ref_features "$DATA/offline_matches/scene_$i" \
        --ref_params "$REF/scene_$i.json" \
        --test_images "$DATA/online/scene_$i" \
        --test_features "$DATA/online_matches/scene_$i" \
        --output "$OUT"
done

for i in 01 02 03 04 05 06 07 08 09 10; do
    echo "--- scene_$i ---"
    $PY "$EVAL_DIR/eval_synthetic.py" --pred "$OUT/scene_$i.json" --gt "$DATA/gt/scene_$i.json"
done

$PY "$EVAL_DIR/agg_synthetic.py" --pred_dir "$OUT" --gt_dir "$DATA/gt" --label "synthetic online (python)"
