"""JLCPCB BOM + CPL from a .kicad_pcb (via kiutils) and the kicad-cli pos csv.
usage: jlc_export.py board.kicad_pcb pos.csv out_dir
LCSC part numbers come from lcsc_map.json ({"<value>|<footprint>": "Cxxxxx"}); unmapped parts stay blank
(P0 spike: only parts whose C-number is known are mapped; M1a owns the real part selection)."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
from kiutils.board import Board

pcb, pos, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
lcsc = json.loads((Path(__file__).parent / "lcsc_map.json").read_text())
groups = defaultdict(list)
for fp in Board.from_file(pcb).footprints:
    ref = fp.properties.get("Reference", "?") ; val = fp.properties.get("Value", "")
    if ref.startswith(("J",)):      # headers are DNP/hand-fit in the spike
        continue
    name = fp.libId.split(":")[-1]
    groups[(val, name)].append(ref)
with open(out / "demo_bom_jlc.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
    for (val, name), refs in sorted(groups.items(), key=lambda kv: kv[1][0]):
        refs.sort(key=lambda r: (r[0], int(r[1:])))
        w.writerow([val, ",".join(refs), name, lcsc.get(f"{val}|{name}", "")])
n = 0
with open(pos) as fi, open(out / "demo_cpl_jlc.csv", "w", newline="") as fo:
    r = csv.DictReader(fi); w = csv.writer(fo); w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
    for row in r:
        if row["Ref"].startswith("J"):
            continue
        w.writerow([row["Ref"], f'{float(row["PosX"]):.4f}mm', f'{float(row["PosY"]):.4f}mm',
                    "Top" if row["Side"] == "top" else "Bottom", f'{float(row["Rot"]):.0f}']); n += 1
print("BOM lines", len(groups), "CPL rows", n)
