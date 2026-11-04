#!/usr/bin/env bash
# M3a: generate tile (layers 1-3 + default tile/) and carrier, run ERC, export netlists, run count checks, export SVG/PDF.
# Usage: hw/board/lib/build_m3a.sh [--no-export]      (~1 min)
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=hw/.venv-eda/bin/python
B=hw/board
ART=docs/artifacts/board/schematic
MODEL=${MODEL:-models/tdlnet_b_w8_aware2_fro_ternary.npz}
export KICAD9_FOOTPRINT_DIR=${KICAD9_FOOTPRINT_DIR:-/usr/share/kicad/footprints}
rc=0
rm -f docs/artifacts/data/board/tapmap.json
erc() { # dir sch
  ( cd "$1" && kicad-cli sch erc --exit-code-violations --severity-all -o erc.rpt "$2" >/dev/null; r=$?; echo "ERC $1/$2: $(grep 'ERC messages' erc.rpt)  (exit $r)"; exit $r ) || rc=1
}
nets() {
  ( cd "$1" && kicad-cli sch export netlist --format kicadsexpr -o "${2%.kicad_sch}.net" "$2" >/dev/null 2>err.txt && \
    kicad-cli sch export netlist --format spice -o "${2%.kicad_sch}.cir" "$2" >/dev/null 2>>err.txt; \
    echo "netlists $1: kicadsexpr+spice exit $? ; stderr: $(tr '\n' ' ' < err.txt)"; rm -f err.txt )
}
$PY $B/gen_tile.py --layer 2 --model $MODEL --out $B/tile | tail -1
erc $B/tile tile.kicad_sch; nets $B/tile tile.kicad_sch
for L in 1 2 3; do
  $PY $B/gen_tile.py --layer $L --model $MODEL --out $B/tile/layer$L | tail -1
  erc $B/tile/layer$L tile.kicad_sch; nets $B/tile/layer$L tile.kicad_sch
  $PY $B/lib/check_schematic.py $B/tile/layer$L --layer $L --model $MODEL --json $B/tile/layer$L/counts.json > $B/tile/layer$L/counts.txt; cat $B/tile/layer$L/counts.txt | grep -E "^layer|DIFF|crossbar|LCSC present"
done
$PY $B/gen_carrier.py | tail -1
erc $B/carrier carrier.kicad_sch; nets $B/carrier carrier.kicad_sch
if [[ "${1:-}" != "--no-export" ]]; then
  mkdir -p $ART
  for d in tile/layer1 tile/layer2 tile/layer3 carrier; do
    n=$(echo $d | tr / _); o=$ART/$n; mkdir -p $o
    f=$(ls $B/$d/*.kicad_sch | grep -E "/(tile|carrier)\.kicad_sch$")
    kicad-cli sch export pdf -o $o/$n.pdf $f >/dev/null
    [[ $d == tile/layer1 || $d == tile/layer3 ]] && continue
    kicad-cli sch export svg --no-background-color -o $o $f >/dev/null
    kicad-cli sch export bom -o $o/$n-bom.csv --fields 'Reference,Value,Footprint,LCSC,MPN,${DNP}' --group-by Value,LCSC,Footprint,'${DNP}' --exclude-dnp $f >/dev/null 2>&1 || true
  done
  du -sh $ART
fi
exit $rc
