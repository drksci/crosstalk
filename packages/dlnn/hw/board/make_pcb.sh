#!/usr/bin/env bash
# usage: hw/board/make_pcb.sh tile LAYER | carrier      (full flow: place, dsn, route, ses; log in hw/board/pcb/<board>/flow.log)
# run long jobs as:  nohup hw/board/make_pcb.sh tile 1 > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export KICAD9_FOOTPRINT_DIR=${KICAD9_FOOTPRINT_DIR:-/usr/share/kicad/footprints}
PY=hw/.venv-eda/bin/python
if [[ "$1" == tile ]]; then S=hw/board/make_pcb_tile.py; A="--layer $2"; D=hw/board/pcb/tile_L$2; else S=hw/board/make_pcb_carrier.py; A=""; D=hw/board/pcb/carrier; fi
mkdir -p $D
for st in place dsn route ses; do
  echo "$(date +%T) stage $st" >> $D/flow.log
  $PY $S $A $st 2>&1 | grep -v "PgmOrNull\|leak" >> $D/flow.log
done
echo "$(date +%T) done" >> $D/flow.log
