#!/usr/bin/env bash
# P0 demo pipeline: schematic -> ERC -> netlists -> ngspice -> PCB -> freerouting -> DRC -> fab outputs.
# Needs ./setup_eda.sh to have run (KiCad 9, hw/.venv-eda, freerouting jar). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv-eda/bin/python
export KICAD9_FOOTPRINT_DIR=${KICAD9_FOOTPRINT_DIR:-/usr/share/kicad/footprints}
t() { local s=$(date +%s.%N); "$@"; printf '[time] %s: %.1f s\n' "$1 ${2:-}" "$(echo "$(date +%s.%N) - $s" | bc)"; }
$PY gen_demo.py > /dev/null 2>&1
kicad-cli sch erc --exit-code-violations -o erc.rpt demo.kicad_sch && echo "ERC: $(grep 'ERC messages' erc.rpt)"
kicad-cli sch export netlist --format spice -o demo.cir demo.kicad_sch > /dev/null
kicad-cli sch export netlist --format kicadsexpr -o demo.net demo.kicad_sch > /dev/null
$PY flatten_spice.py demo.cir models/lc_parts.lib demo_flat.cir
python3 - <<'PYEOF'
t = open('demo_flat.cir').read().replace('.end\n', '') + """
.control
run
meas tran t_n1 when v(/N1)=0.5 rise=1
meas tran t_n2 when v(/N2)=0.5 rise=1
wrdata tran_ngspice.txt v(/N1) v(/N2) v(/T1) v(/T2) v(/OUT)
.endc
.end
"""
open('demo_ng.cir', 'w').write(t)
PYEOF
ngspice -b demo_ng.cir 2>&1 | grep -E '^t_n[12] ' || true
$PY netlist_to_pcb.py demo.net demo_raw.kicad_pcb 2>&1 | grep -v 'assert "PgmOrNull' || true
t $PY make_pcb.py place dsn
t $PY make_pcb.py route
t $PY make_pcb.py ses
kicad-cli pcb drc --exit-code-violations -o drc.rpt demo.kicad_pcb
echo "DRC: $(grep -E 'Found [0-9]+ (DRC violations|unconnected)' drc.rpt | tr '\n' ' ')"
mkdir -p fab/gerbers fab/svg fab/render
kicad-cli pcb export gerbers -o fab/gerbers/ --board-plot-params demo.kicad_pcb > /dev/null
kicad-cli pcb export drill -o fab/gerbers/ --format excellon --excellon-separate-th --generate-map --map-format gerberx2 demo.kicad_pcb > /dev/null
kicad-cli pcb export pos --format csv --units mm --side both -o fab/demo_pos.csv demo.kicad_pcb > /dev/null
$PY jlc_export.py demo.kicad_pcb fab/demo_pos.csv fab
kicad-cli pcb export svg --layers F.Cu,In1.Cu,In2.Cu,B.Cu,F.Silkscreen,Edge.Cuts --mode-multi --page-size-mode 2 --exclude-drawing-sheet -o fab/svg/ demo.kicad_pcb > /dev/null
kicad-cli pcb export glb --subst-models --include-tracks --include-zones --include-silkscreen --include-soldermask --force -o fab/demo.glb demo.kicad_pcb > /dev/null
kicad-cli pcb export step --subst-models --force -o fab/demo.step demo.kicad_pcb > /dev/null
kicad-cli pcb render --side top --quality basic -w 1600 -h 1000 -o fab/render/top.png demo.kicad_pcb > /dev/null
kicad-cli pcb render --side bottom --quality basic -w 1600 -h 1000 -o fab/render/bottom.png demo.kicad_pcb > /dev/null
kicad-cli pcb render --side top --quality basic --perspective --rotate '-45,0,45' --zoom 0.9 -w 1600 -h 1000 -o fab/render/perspective.png demo.kicad_pcb > /dev/null
(cd fab/gerbers && rm -f ../demo_gerbers.zip && zip -q ../demo_gerbers.zip *.gbr *.drl *.gbrjob)
echo "fab outputs: $(find fab -type f | wc -l) files"
