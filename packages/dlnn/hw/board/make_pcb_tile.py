#!/usr/bin/env python3
"""TILE-8 PCB (M3b): netlist -> scripted placement + scripted regular routing -> freerouting for the rest -> DRC -> fab outputs.

    hw/.venv-eda/bin/python hw/board/make_pcb_tile.py --layer 2 place        # build + place + scripted copper + DSN
    ...                                                  --layer 2 route      # freerouting (run under nohup, see make_pcb_tile.sh)
    ...                                                  --layer 2 ses        # import SES, fill zones, DRC

Geometry (all numbers in mm, derived from the stock footprints: L_1812 courtyard 5.99 x 3.99, C_0603 1.55 x 3.05 rotated):
  * ladder block per line: 7 rows x 8 sections (one row = one tap interval), section cell 7.7 mm (L 5.99 + C 1.55 in line),
    row pitch 4.2 mm, rows alternate direction (boustrophedon) so taps sit at the row ends; tap quads in the end gutters.
  * lines 0-3: blocks on the TOP side (2 x 2, inductor axes along X, GND plane In1 directly below);
    lines 4-7: blocks on the BOTTOM side, rows along Y (inductor axes orthogonal to the top ones), under the top rows only,
    their gutters outside the top blocks; In2 carries a GND island under those rows.
  * GND of every ladder capacitor is a scripted bus track in the gap between rows, stitched to the plane in the gutters.
  * crossbar: 64 columns (tap) x 16 rows (SN_o/SP_o) matrix of 0402, pitch 1.3 x 2.3 mm: resistor axis along Y, pad 2 on the
    row track (F.Cu), pad 1 -> via -> column bus (B.Cu).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_pcb as G  # noqa: E402
from gen_pcb import F, B, IN1, IN2  # noqa: E402
import pcbnew  # noqa: E402

REPO = HERE.parent.parent
CELL, ROWP, RW = 7.7, 4.2, 61.6          # section cell, row pitch, row width (8 cells)
GUT = 9.5
W, H = 180.0, 142.0
LAYER_NET = {1: "layer1", 2: "layer2", 3: "layer3"}


class Tile:
    def __init__(self, layer, outdir, model):
        self.layer = layer
        self.claim_clean = layer == 1 or bool(int(__import__("os").environ.get("TILE_CLAIM_CLEAN", "0")))
        self.n_lines = 2 if layer == 1 else 8
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.model = Path(model)
        self.sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        net = HERE / "tile" / LAYER_NET[layer] / "tile.net"
        self.pb = G.PB(net, (W, H), origin=(40.0, 40.0), name=f"tile_L{layer}", workdir=self.out)
        self.log = []

    # ------------------------------------------------------------------ block geometry
    def block_T(self, line):
        """returns (kind, T(u,v)->(x,y), dirmap, side, layer) for the ladder block of `line`"""
        if line < 4:
            col, row = line // 2, line % 2
            x0 = 12 + 80.6 * col + GUT
            y0 = 17.0 + 30.0 * row
            return ("top", lambda u, v: (x0 + u, y0 + v), lambda d: d, "F", F)
        b = line - 4
        xb = [22.5, 54.1, 103.0, 134.6][b]
        y0 = 16.4
        return ("bot", lambda u, v: (xb + v, y0 + u), lambda d: (d[1], d[0]), "B", B)

    def place_fit(self, ref, x, y, side, want_ext=None, want1=None, dirmap=None):  # want_ext in BOARD axes
        """place a part trying the four rotations: either pad 1 toward want1 (2-pad parts) or bbox extents ~ want_ext"""
        pb = self.pb
        if want1 is not None:
            return pb.place_oriented(ref, x, y, dirmap(want1) if dirmap else want1, side)
        best = None
        for rot in (0, 90, 180, 270):
            pb.place(ref, x, y, rot, side)
            w, h = pb.bbox(ref)
            err = abs(w - want_ext[0]) + abs(h - want_ext[1])
            if best is None or err < best[0] - 1e-6:
                best = (err, rot)
        pb.place(ref, x, y, best[1], side)
        return best[1]

    def ladder_block(self, i):
        pb = self.pb
        kind, T, dm, side, lay = self.block_T(i)
        gnd_pads = []        # (u, v_pad2, v_bus_row) for GND tracks
        rowv = lambda j: 2.1 + ROWP * j
        loc = lambda ref, n: self.inv(i, pb.pad_xy(ref, n))
        for s in range(1, 57):
            j, m = (s - 1) // 8, (s - 1) % 8
            east = (j % 2 == 0)
            ue = lambda a: a if east else RW - a
            uL = ue(CELL * m + 3.0)
            self.place_fit(f"L{i}_{s}", *T(uL, rowv(j)), side, want1=((-1, 0) if east else (1, 0)), dirmap=dm)
            if s <= 55:
                uC = ue(CELL * m + 6.9)
                self.place_fit(f"C{i}_{s}", *T(uC, rowv(j)), side, want1=(0, 1), dirmap=dm)
        # ---- scripted chain: node s between L_s.2 and L_{s+1}.1 with C_s
        w = 0.25
        for s in range(1, 56):
            j = (s - 1) // 8
            a = loc(f"L{i}_{s}", "2")
            b = loc(f"L{i}_{s+1}", "1")
            c1 = loc(f"C{i}_{s}", "1")
            c2 = loc(f"C{i}_{s}", "2")
            netn = pb.pad2net[(f"C{i}_{s}", "1")]
            if (s % 8) != 0:        # in-row node: straight track at the C pad-1 level through L.2 -> C.1 -> L.1(next)
                v = c1[1]
                pb.track(netn, lay, [self.xy(T, a[0], v), self.xy(T, b[0], v)], w)
            else:                   # turn node: L_s.2 -> C.1 along the row, then vertical to L_{s+1}.1 in the next row
                v = c1[1]
                pb.track(netn, lay, [self.xy(T, a[0], v), self.xy(T, c1[0], v), self.xy(T, c1[0], b[1])], w)
            # GND: pad 2 -> north to the bus line above this row
            vb = rowv(j) - 2.1
            pb.track("GND", lay, [self.xy(T, c2[0], c2[1]), self.xy(T, c2[0], vb)], 0.2)
            gnd_pads.append((j, c2[0]))
        # ---- GND buses above each row, stitched to the plane at the end that has no turn track
        self.gnd_buses = getattr(self, "gnd_buses", {})
        for j in range(7):
            us = [u for (jj, u) in gnd_pads if jj == j]
            if not us:
                continue
            vb = rowv(j) - 2.1
            if j == 0 or j % 2 == 0 and j != 0:
                pass
            # connect side: row j even -> its turn track (from row j-1) sits at the west end -> connect east; j odd -> west
            if j == 0:
                conn_w = True
            else:
                conn_w = (j % 2 == 1)
            if conn_w:
                u_end, via_u = -1.4, -1.4
                ua, ub = via_u, max(us) + 0.2
            else:
                via_u = RW + 1.4
                ua, ub = min(us) - 0.2, via_u
            # the bus must not run into the turn track end: restrict to the span of its own caps
            lo, hi = (via_u, max(us) + 0.1) if conn_w else (min(us) - 0.1, via_u)
            pb.track("GND", lay, [self.xy(T, lo, vb), self.xy(T, hi, vb)], 0.3)
            pb.via("GND", *self.xy(T, via_u, vb), 0.6, 0.3)
        return T, dm, side, lay

    def xy(self, T, u, v):
        return T(u, v)

    def inv(self, line, xy):
        kind, T, dm, side, lay = self.block_T(line)
        x, y = xy
        x0, y0 = T(0, 0)
        return (x - x0, y - y0) if kind == "top" else (y - y0, x - x0)

    # ------------------------------------------------------------------ generic shelf packers
    def _center_fix(self, ref, x, y, side, rot=None):
        pb = self.pb
        f = pb.fps[ref]
        pb.place(ref, x, y, f.GetOrientationDegrees() if rot is None else rot, side)
        bb = f.GetBoundingBox(False, False)
        cx = pcbnew.ToMM(bb.GetCenter().x) - pb.OX
        cy = pcbnew.ToMM(bb.GetCenter().y) - pb.OY
        pb.place(ref, x - (cx - x), y - (cy - y), f.GetOrientationDegrees(), side)

    def via_points(self):
        to = pcbnew.ToMM
        return [(to(v.GetPosition().x) - self.pb.OX, to(v.GetPosition().y) - self.pb.OY) for v in self.pb.b.GetTracks()
                if v.Type() == pcbnew.PCB_VIA_T]

    def shelf_board(self, refs, x0, y0, x1, y1, side="F", gap=0.5, rot=0, name="", avoid_vias=False):
        """pack parts in rows inside the board-coordinate box; raises on overflow"""
        pb = self.pb
        vias = self.via_points() if avoid_vias else []
        x, y, rowh = x0, y0, 0.0
        for ref in refs:
            pb.place(ref, 0, 0, rot, side)
            w, h = pb.bbox(ref)
            w, h = w + 0.1, h + 0.1
            if x + w > x1 + 1e-6:
                x, y, rowh = x0, y + rowh + gap, 0.0
            if y + h > y1 + 1e-6:
                raise RuntimeError(f"shelf overflow {name} at {ref}: box {x0,y0,x1,y1}")
            while vias and any(x - 0.6 < vx < x + w + 0.6 and y - 0.6 < vy < y + h + 0.6 for vx, vy in vias):
                x += 0.4
                if x + w > x1 + 1e-6:
                    x, y, rowh = x0, y + rowh + gap, 0.0
                if y + h > y1 + 1e-6:
                    raise RuntimeError(f"shelf overflow {name} at {ref}: box {x0,y0,x1,y1}")
            self._center_fix(ref, x + w / 2, y + h / 2, side, rot)
            x += w + gap
            rowh = max(rowh, h)
        return y + rowh

    def shelf_local(self, i, refs, u0, v0, u1, v1, gap=0.45):
        kind, T, dm, side, lay = self.block_T(i)
        pb = self.pb
        u, v, rowh = u0, v0, 0.0
        for ref in refs:
            pb.place(ref, 0, 0, 0, side)
            w, h = pb.bbox(ref)
            eu, ev = (w, h) if kind == "top" else (h, w)
            eu, ev = eu + 0.1, ev + 0.1
            if u + eu > u1 + 1e-6:
                u, v, rowh = u0, v + rowh + gap, 0.0
            if v + ev > v1 + 1e-6:
                raise RuntimeError(f"gutter overflow line {i} at {ref}")
            self._center_fix(ref, *T(u + eu / 2, v + ev / 2), side, 0)
            u += eu + gap
            rowh = max(rowh, ev)

    def gutters(self, i):
        pb = self.pb
        kind, T, dm, side, lay = self.block_T(i)
        sw = (lambda a: a) if kind == "top" else (lambda a: (a[1], a[0]))
        for name, ucen, vcen in (("A", -6.4, 14.0), ("B", RW + 6.4, 16.5)):
            ref = f"UT{name}{i}"
            self.place_fit(ref, *T(ucen, vcen), side, want_ext=sw((5.7, 7.75)))
            self._center_fix(ref, *T(ucen, vcen), side)
            self.shelf_local(i, [f"CDT{name}{i}"], ucen - 2.8, vcen + 4.3, ucen + 2.8, vcen + 6.5)
        self.shelf_local(i, [f"RS{i}", f"C{i}_0_1", f"C{i}_0_2", f"RP{i}", f"CP{i}"], -9.4, -1.0, -2.6, 9.6)
        self.shelf_local(i, [f"C{i}_56_1", f"C{i}_56_2", f"RT{i}"], RW + 2.6, 23.0, RW + 9.4, 30.0)

    # ------------------------------------------------------------------ crossbar matrix
    XM0, YM0, PX, PY = 18.0, 94.0, 1.3, 2.6

    def matrix(self):
        pb = self.pb
        n_col = 8 * self.n_lines
        self.col_x = {}
        for o in range(8):
            for pol, rowoff, sumnet in (("P", 0, f"SN{o}"), ("N", 1, f"SP{o}")):
                r = 2 * o + rowoff
                y = self.YM0 + self.PY * r
                for i in range(self.n_lines):
                    for k in range(8):
                        c = 8 * i + k
                        x = self.XM0 + self.PX * c
                        ref = f"RX{pol}_{o}_{i}_{k}"
                        pb.place_oriented(ref, x, y, (0, -1), "F")
                        self.col_x[c] = (x, f"T{i}_{k}")
                        yp1 = y - 0.51
                        pb.track(f"T{i}_{k}", F, [(x, yp1), (x, y - 1.18)], 0.2)
                        pb.via(f"T{i}_{k}", x, y - 1.18, 0.5, 0.25)
                x_end = self.XM0 + self.PX * (n_col - 1)
                pb.track(sumnet, F, [(self.XM0, y + 0.51), (x_end + 2.2, y + 0.51)], 0.2)
        y_last = self.YM0 + self.PY * 15 - 1.18
        for c, (x, tn) in self.col_x.items():
            pb.track(tn, B, [(x, self.YM0 - 6.0), (x, y_last)], 0.2)
        self.mx_end = self.XM0 + self.PX * (n_col - 1) + 2.2

    # ------------------------------------------------------------------ channels, misc, connectors
    @staticmethod
    def chan_of(ref):
        m = re.fullmatch(r"(RFP|RFN|CFP|CFN|RB|RO|CBF|QD|CDA|CH)(\d)", ref) or re.fullmatch(r"(RLD1_|RLD2_|RDA1_|RDA2_|RDA3_|RDA4_)(\d)", ref)
        if m:
            return int(m.group(2))
        m = re.fullmatch(r"(UCS|UT|UDA|CDCS|CDDA)(\d)", ref)
        if m and m.group(1) in ("UCS", "UT", "UDA", "CDCS", "CDDA"):
            return int(m.group(2)) * 2
        m = re.fullmatch(r"(USH|CDSH)(\d)", ref)
        if m:
            return int(m.group(2)) * 4
        return None

    def channels(self):
        pb = self.pb
        groups = {0: [], 1: []}
        for ref in pb.comps:
            o = self.chan_of(ref)
            if o is not None:
                groups[o // 4].append((o, 0 if ref[0] == "U" else 1, ref))
        for g in (0, 1):
            refs = [r for _, _, r in sorted(groups[g], key=lambda t: (t[0], t[1], t[2]))]
            x0 = 104.0 + 29.5 * g
            self.shelf_board(refs, x0, 92.0, x0 + 29.0, 132.0, "F", gap=0.4, name=f"channels{g}")

    def misc(self):
        pb = self.pb
        done = set(self.placed_refs())
        # bottom-side east: UIB1 + decap (lines 4-7 input buffers) when present
        top_refs, bot_refs = [], []
        for ref in pb.comps:
            if ref in done:
                continue
            if ref.startswith(("JIN", "JOUT", "JPS", "TPMON", "JMON")):
                continue
            if ref in ("UIB1", "CDIB1"):
                if self.n_lines > 4:
                    bot_refs.append(ref)
                continue
            top_refs.append(ref)
        self.shelf_board(top_refs, 8.0, 80.4, 177.0, 91.6, "F", gap=0.45, name="misc top", avoid_vias=True)
        if bot_refs:
            self.shelf_board(bot_refs, 165.0, 79.0, 178.0, 90.0, "B", gap=0.45, name="misc bottom")

    def placed_refs(self):
        out = []
        for ref in self.pb.comps:
            if self.chan_of(ref) is not None or ref.startswith(("RX", )):
                out.append(ref)
            elif re.match(r"(L|C|R|UT[AB])\d_", ref) or re.match(r"(UTA|UTB|CDTA|CDTB|RP|CP|RS|RT)\d$", ref):
                out.append(ref)
        return out

    def connectors(self):
        pb = self.pb
        pb.place("JIN1", 0, 0, 0, "F"); self._center_fix("JIN1", 9.5, 118.0, "F")
        pb.place("JOUT1", 0, 0, 0, "F"); self._center_fix("JOUT1", 170.5, 118.0, "F")
        pb.place("JPS1", 0, 0, 90, "F"); self._center_fix("JPS1", 118.0, 135.0, "F", 90)
        pb.place("JMON1", 0, 0, 0, "F"); self._center_fix("JMON1", 176.5, 52.0, "F")
        pb.place("TPMON1", 0, 0, 0, "F"); self._center_fix("TPMON1", 176.5, 62.0, "F")
        for r in ("JIN1", "JOUT1", "JPS1", "JMON1"):
            pb.fps[r].Reference().SetVisible(True)
            pb.fps[r].Reference().SetTextSize(pcbnew.VECTOR2I(G.mm(1.0), G.mm(1.0)))

    # ------------------------------------------------------------------ board furniture
    def furniture(self):
        pb = self.pb
        pb.outline()
        for k, (x, y) in enumerate([(5, 5), (175, 5), (5, 92), (175, 92)]):
            pb.add_lib_fp("MountingHole:MountingHole_3.2mm_M3_Pad_Via", f"H{k+1}", x, y, netmap={"1": "GND"})
        for k, (x, y, s) in enumerate([(14, 10, "F"), (168, 10, "F"), (92, 89.5, "F"), (14, 137, "B"), (168, 137, "B"), (92, 137, "B")]):
            pb.add_lib_fp("Fiducial:Fiducial_1mm_Mask2mm", f"FID{k+1}", x, y, side=s)
        # planes: In1 GND, In2 +5VA with GND islands under the bottom-side rows
        m = 0.6
        rect = [(m, m), (W - m, m), (W - m, H - m), (m, H - m)]
        pb.poly(IN1, rect, "GND", priority=0)
        pb.poly(IN2, rect, "+5VA", priority=0)
        if self.n_lines > 4:
            for xb in (22.5, 54.1, 103.0, 134.6)[: self.n_lines - 4]:
                pb.poly(IN2, [(xb - 1.0, 16.2), (xb + 29.6, 16.2), (xb + 29.6, 78.2), (xb - 1.0, 78.2)], "GND", priority=2)
        status = "designed, ERC/DRC-clean, NOT fabricated" if self.claim_clean else "PLACED REFERENCE, NOT ROUTED, NOT fabricated"
        title = f"TILE-8  layer-{self.layer} variant  model sha256 {self.sha[:12]}  {status}"
        pb.text(title, 60, 139.4, 1.1, pcbnew.F_SilkS)
        pb.text(f"DLNN TILE-8 L{self.layer}  {self.model.name}  sha {self.sha[:12]}", 60, 139.4, 1.0, pcbnew.B_SilkS, mirror=True)

    # ------------------------------------------------------------------ driver
    def build_placed(self):
        pb = self.pb
        G.write_project(self.out / f"tile_L{self.layer}.kicad_pro", extra_sev={"track_dangling": "ignore"})   # KiCad ignores DNP pads as connection points: the crossbar row tracks end on DNP pads
        pb.add_all()
        for i in range(self.n_lines):
            self.ladder_block(i)
            self.gutters(i)
        self.matrix()
        self.channels()
        self.connectors()
        self.misc()
        self.furniture()
        done, failed = G.fanout_planes(pb, bounds=(1.5, 1.5, W - 1.5, H - 1.5))
        print("plane fanout vias:", done, "failed:", failed)
        path = pb.save(self.out / f"tile_L{self.layer}_placed.kicad_pcb")
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=str(REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"))
    ap.add_argument("stage", nargs="*", default=["place"])
    a = ap.parse_args()
    out = a.out or HERE / "pcb" / f"tile_L{a.layer}"
    t = Tile(a.layer, out, a.model)
    for st in a.stage:
        if st == "place":
            t0 = time.time()
            p = t.build_placed()
            print("placed ->", p, f"{time.time()-t0:.1f}s")
        elif st in ("dsn", "route", "ses"):
            G.run_stage(st, out, f"tile_L{a.layer}")


if __name__ == "__main__":
    main()
