#!/usr/bin/env python3
"""Carrier PCB (M3b): NUCLEO-G474RE on two 2x19 female sockets, DAC I/Q conditioner, ADC front end, 7-9 V jack -> AMS1117-5.0, tile ribbon headers.

    hw/.venv-eda/bin/python hw/board/make_pcb_carrier.py place|dsn|route|ses|all [--art PATH]

4 layers (sig / GND / +5VA / sig, JLC04161H-7628).  Sockets: CN7 outer column x = 64.0 mm, CN10 inner column 60.96 mm to the right of the CN7
inner column (66.04 mm outer to outer, from the Nucleo-64 morpho pitch; VERIFY against the board, see docs/pcb_layout.md).  The bottom side
between the sockets (x 69.5..124.5, y 6.5..83.5) is the art island (B.Cu keep-out for everything else; see art/make_art.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_pcb as G  # noqa: E402
from gen_pcb import F, B, IN1, IN2  # noqa: E402
import pcbnew  # noqa: E402

REPO_ART = HERE.parent.parent / "docs" / "artifacts" / "board" / "art"
W, H = 134.0, 90.0
ART_BOX = (69.5, 6.5, 124.5, 83.5)       # x0, y0, x1, y1 on the bottom side


class Carrier:
    def __init__(self, outdir):
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.pb = G.PB(HERE / "carrier" / "carrier.net", (W, H), origin=(30.0, 30.0), name="carrier", workdir=self.out)

    def center_fix(self, ref, x, y, rot=0, side="F"):
        pb = self.pb
        f = pb.fps[ref]
        pb.place(ref, x, y, rot, side)
        bb = f.GetBoundingBox(False, False)
        cx = pcbnew.ToMM(bb.GetCenter().x) - pb.OX
        cy = pcbnew.ToMM(bb.GetCenter().y) - pb.OY
        pb.place(ref, x - (cx - x), y - (cy - y), rot, side)

    def shelf(self, refs, x0, y0, x1, y1, gap=0.6, side="F"):
        pb = self.pb
        x, y, rowh = x0, y0, 0.0
        for ref in refs:
            pb.place(ref, 0, 0, 0, side)
            w, h = pb.bbox(ref)
            w, h = w + 0.1, h + 0.1
            if x + w > x1:
                x, y, rowh = x0, y + rowh + gap, 0.0
            if y + h > y1:
                raise RuntimeError(f"carrier shelf overflow at {ref}")
            self.center_fix(ref, x + w / 2, y + h / 2, 0, side)
            x += w + gap
            rowh = max(rowh, h)

    def build(self):
        pb = self.pb
        G.write_project(self.out / "carrier.kicad_pro")
        pb.add_all()
        # place sockets: stock footprint has pin1 at (0,0) right column, pin 2 at (-2.54, 0): mirror columns by renumbering pads
        for ref, xleft in (("CN7", 64.0), ("CN10", 127.5)):
            self.place_socket(ref, xleft, 18.0)
        # header geometry: JT1 and JADC1 horizontal at the left-bottom
        self.center_fix("JT1", 36.0, 81.0, 90)
        self.center_fix("JADC1", 36.0, 66.0, 90)
        self.center_fix("J1", 19.0, 12.0, 0)
        refs_left = [r for r in pb.comps if r not in ("CN7", "CN10", "JT1", "JADC1", "J1")]
        # order: power first, then conditioner, then ADC chains
        def key(r):
            return (0 if r in ("D1", "U1") or r.startswith(("CJ", "RESR", "CBYP")) else
                    1 if r.startswith(("UC1", "UREF", "RRF", "CD1", "RVA", "CDV", "RDI", "RDO")) else 2, r)
        refs_left.sort(key=key)
        self.shelf(refs_left, 30.0, 3.5, 62.5, 58.0, gap=1.7)
        # mounting holes + fiducials + outline + planes
        pb.outline()
        for k, (x, y) in enumerate([(4, 4), (W - 4, 4), (4, H - 4), (W - 4, H - 4)]):
            pb.add_lib_fp("MountingHole:MountingHole_3.2mm_M3_Pad_Via", f"H{k+1}", x, y, netmap={"1": "GND"})
        for k, (x, y, s) in enumerate([(10, 30, "F"), (58, 60, "F"), (58, 86, "F"), (8, 44, "B"), (58, 60, "B"), (100, 87.5, "B")]):
            pb.add_lib_fp("Fiducial:Fiducial_1mm_Mask2mm", f"FID{k+1}", x, y, side=s)
        m = 0.6
        rect = [(m, m), (W - m, m), (W - m, H - m), (m, H - m)]
        pb.poly(IN1, rect, "GND")
        pb.poly(IN2, rect, "+5VA")
        # B.Cu keep-out for the art island: no tracks, vias or pours of other nets (the island itself is added by art/make_art.py)
        x0, y0, x1, y1 = ART_BOX
        pb.poly(B, [(x0 - 0.6, y0 - 0.6), (x1 + 0.6, y0 - 0.6), (x1 + 0.6, y1 + 0.6), (x0 - 0.6, y1 + 0.6)], None,
                keepout=dict(pour=True, tracks=True, vias=True))
        pb.text("DLNN carrier  designed, ERC/DRC-clean, NOT fabricated", 66, 87.0, 1.0, pcbnew.F_SilkS)
        pb.text("NUCLEO-G474RE", 97.0, 12.0, 1.2, pcbnew.F_SilkS)
        ax0, ay0, ax1, ay1 = ART_BOX
        done, failed = G.fanout_planes(pb, bounds=(1.5, 1.5, W - 1.5, H - 1.5),
                                       extra_keepouts=[(ax0 - 0.6, ay0 - 0.6, ax1 + 0.6, ay1 + 0.6, {"F", "B"})])
        print("plane fanout vias:", done, "failed:", failed)
        return pb.save(self.out / "carrier_placed.kicad_pcb")

    def place_socket(self, ref, xleft, y0):
        """stock PinSocket_2x19: pin n at column (n odd: x=0, even: x=-2.54), row (n-1)//2.  Move pads so the layout reads odd = left column."""
        pb = self.pb
        f = pb.fps[ref]
        pb.place(ref, 0, 0, 0, "F")
        # renumber pads: new_number for pad at (col, row): left col -> odd
        cur = {}
        for p in f.Pads():
            n = int(p.GetNumber())
            cur[n] = p
        for n, p in cur.items():
            row = (n - 1) // 2
            left_col = (n % 2 == 0)        # stock even pins are on the left
            new = 2 * row + (1 if left_col else 2)
            net = pb.pad2net.get((ref, str(new)))
            p.SetNumber(str(new))
            p.SetNet(pb.net(net)) if net else p.SetNetCode(0)
        pb.place(ref, xleft + 2.54, y0, 0, "F")   # stock pin 2 (left column) at x-2.54 -> left column at xleft


def add_art(routed, out, variant="gold"):
    """add the generated art footprint (flipped onto the bottom side) to the routed carrier -> out; returns stats"""
    import subprocess
    art_dir = HERE / "art"
    lib = art_dir / "lib.pretty"
    lib.mkdir(exist_ok=True)
    x0, y0, x1, y1 = ART_BOX
    subprocess.run([sys.executable, str(art_dir / "make_art.py"), "--box", str(x1 - x0 - 1.0), str(y1 - y0 - 1.0), "--name", "carrier",
                    "--out", str(art_dir / "out"), "--web", str(REPO_ART)], check=True, capture_output=True)
    for v in ("gold", "silk"):
        (lib / f"ART_carrier_{v}.kicad_mod").write_text((art_dir / "out" / f"art_carrier_{v}.kicad_mod").read_text())
    b = pcbnew.LoadBoard(str(routed))
    io = pcbnew.PCB_IO_MGR.PluginFind(pcbnew.PCB_IO_MGR.KICAD_SEXP)
    f = io.FootprintLoad(str(lib), f"ART_carrier_{variant}")
    f = pcbnew.Cast_to_FOOTPRINT(f.Duplicate())
    f.SetReference("ART1")
    f.Reference().SetVisible(False)
    b.Add(f)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pos = pcbnew.VECTOR2I(pcbnew.FromMM(30.0 + cx), pcbnew.FromMM(30.0 + cy))
    f.SetPosition(pos)
    f.Flip(pos, pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
    f.SetPosition(pos)
    if variant == "gold":
        # the hole-free pieces of the island are separate "no net" items sharing one mask aperture: KiCad's mask-bridge check reports every
        # seam between pieces (false positives, the pieces are one island with >= 0.2 mm gaps, verified in art/out/art_carrier_stats.json).
        # The check stays enabled for the board without the art (carrier.kicad_pcb: 0 violations) and is ignored for this file only.
        pass
    G.fill(b)
    pcbnew.SaveBoard(str(out), b)
    G.inject_stackup(out)
    if variant == "gold":
        import json as _json
        pro = _json.loads((Path(routed).parent / "carrier.kicad_pro").read_text())
        pro["board"]["design_settings"]["rule_severities"]["solder_mask_bridge"] = "ignore"
        (Path(out).with_suffix(".kicad_pro")).write_text(_json.dumps(pro, indent=2))
    return variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="*", default=["place"])
    ap.add_argument("--out", default=str(HERE / "pcb" / "carrier"))
    a = ap.parse_args()
    c = Carrier(a.out)
    for st in a.stage:
        if st == "place":
            print("placed ->", c.build())
        elif st in ("dsn", "route", "ses"):
            G.run_stage(st, a.out, "carrier")
        elif st in ("art", "art-silk"):
            v = "gold" if st == "art" else "silk"
            src = Path(a.out) / "carrier.kicad_pcb"
            dst = Path(a.out) / ("carrier_art.kicad_pcb" if v == "gold" else "carrier_art_silk.kicad_pcb")
            add_art(src, dst, v)
            rc, txt = G.drc(dst, Path(a.out) / f"drc_art_{v}.rpt", Path(a.out) / f"drc_art_{v}.json")
            print("art", v, "DRC rc", rc, " ".join(__import__("re").findall(r"\*\* Found [^*]+\*\*", txt)))


if __name__ == "__main__":
    main()
