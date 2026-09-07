#!/bin/bash
# Online stage (PTZ-Reloc) on WorldCup14 - mirrors PTZ-Calib/run_reloc_worldcup14.sh
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA=${DATA:-../PTZ-Calib/data/worldcup14}
REF=${REF:-output-worldcup14-offline}
OUT=${OUT:-output-worldcup14-online}
EVAL_DIR=${EVAL_DIR:-../PTZ-Calib/scripts}

run() {  # run <ref_cam> <test_cam>
    $PY scripts/run_ptz_reloc.py \
        --ref_images "$DATA/offline/$1" \
        --ref_features "$DATA/offline_matches/$1" \
        --ref_params "$REF/$1.json" \
        --test_images "$DATA/online/$2" \
        --test_features "$DATA/online_matches/$2" \
        --output "$OUT"
}

run GER_ARG ESP_CHI
run GER_ARG FRA_GER
run GER_POR SUI_FRA
run NED_ARG ARG_SUI
run NED_ARG BRA_CRO
run NED_ARG URU_ENG
run USA_GER CRO_MEX

for cam in ARG_SUI BRA_CRO CRO_MEX ESP_CHI FRA_GER SUI_FRA URU_ENG; do
    $PY "$EVAL_DIR/eval_worldcup.py" --gt_dir "$DATA/gt/test" --pred "$OUT/$cam.json"
done
