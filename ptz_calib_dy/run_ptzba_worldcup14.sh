#!/bin/bash
# Offline stage (PTZ-BA) on WorldCup14 - mirrors PTZ-Calib/run_ptzba_worldcup14.sh
set -e
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA=${DATA:-../PTZ-Calib/data/worldcup14}
OUT=${OUT:-output-worldcup14-offline}

for cam in GER_ARG GER_POR NED_ARG USA_GER; do
    $PY scripts/run_ptz_ba.py \
        -i "$DATA/offline/$cam" \
        -f "$DATA/offline_matches/$cam" \
        -a "$DATA/offline/$cam/$cam.json" \
        -o "$OUT"
done
