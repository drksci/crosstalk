"""PCB construction helpers shared by make_pcb_tile.py / make_pcb_carrier.py (M3b).

Direct pcbnew creation from the KiCad kicadsexpr netlist (the schematic stays the single source of truth).  kinet2pcb (proven in P0)
needs more than 2 minutes for 700 parts, so the same job is done here with the pcbnew API: footprints from the stock KiCad
libraries, nets from the netlist, scripted placement and routing helpers, JLC-style stackup injection, Specctra DSN -> freerouting
-> SES round trip and zone fill.  Run with hw/.venv-eda/bin/python (pcbnew from system python3.12).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KICAD9_FOOTPRINT_DIR", "/usr/share/kicad/footprints")
import pcbnew  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pcb"))
from netlist import read_net  # noqa: E402

mm = pcbnew.FromMM
FPDIR = Path("/usr/share/kicad/footprints")
FR_JAR = Path(os.environ.get("FREEROUTING_JAR", "/home/user/freerouting/freerouting/build/libs/freerouting-current-executable.jar"))
FR_JAVA = os.environ.get("FREEROUTING_JAVA", "/root/.gradle/jdks/eclipse_adoptium-25-amd64-linux.2/bin/java")

F, B = pcbnew.F_Cu, pcbnew.B_Cu
IN1, IN2 = pcbnew.In1_Cu, pcbnew.In2_Cu

# JLC04161H-7628 4-layer 1.6 mm stackup (injected as text: the python API has no stackup descriptor)
STACKUP = """(stackup
\t\t\t(layer "F.SilkS" (type "Top Silk Screen") (color "White"))
\t\t\t(layer "F.Paste" (type "Top Solder Paste"))
\t\t\t(layer "F.Mask" (type "Top Solder Mask") (color "Black") (thickness 0.01))
\t\t\t(layer "F.Cu" (type "copper") (thickness 0.035))
\t\t\t(layer "dielectric 1" (type "prepreg") (thickness 0.2104) (material "7628") (epsilon_r 4.4) (loss_tangent 0.02))
\t\t\t(layer "In1.Cu" (type "copper") (thickness 0.0152))
\t\t\t(layer "dielectric 2" (type "core") (thickness 1.065) (material "FR4") (epsilon_r 4.6) (loss_tangent 0.02))
\t\t\t(layer "In2.Cu" (type "copper") (thickness 0.0152))
\t\t\t(layer "dielectric 3" (type "prepreg") (thickness 0.2104) (material "7628") (epsilon_r 4.4) (loss_tangent 0.02))
\t\t\t(layer "B.Cu" (type "copper") (thickness 0.035))
\t\t\t(layer "B.Mask" (type "Bottom Solder Mask") (color "Black") (thickness 0.01))
\t\t\t(layer "B.Paste" (type "Bottom Solder Paste"))
\t\t\t(layer "B.SilkS" (type "Bottom Silk Screen") (color "White"))
\t\t\t(copper_finish "ENIG")
\t\t\t(dielectric_constraints no)
\t\t)"""


def inject_stackup(path):
    t = Path(path).read_text()
    t = re.sub(r"\(stackup\b.*?\n\t\t\)\n?", "", t, flags=re.S)
    i = t.index("(setup")
    j = t.index("\n", i)
    t = t[:j + 1] + "\t\t" + STACKUP + "\n" + t[j + 1:]
    Path(path).write_text(t)


def write_project(path, clearance=0.15, track=0.15, via_d=0.5, via_drill=0.25, extra_sev=None):
    """minimal .kicad_pro with the JLC-4-layer-capability rules used here (kicad-cli drc and pcbnew.LoadBoard both read it)"""
    sev = {"silk_overlap": "ignore", "silk_over_copper": "ignore", "silk_edge_clearance": "ignore",
           "lib_footprint_mismatch": "ignore", "lib_footprint_issues": "ignore", "footprint_symbol_mismatch": "ignore",
           "footprint_filters_mismatch": "ignore", "extra_footprint": "ignore", "missing_footprint": "ignore",
           "unresolved_variable": "ignore", "text_height": "ignore", "text_thickness": "ignore", "silk_over_copper": "ignore"}
    sev.update(extra_sev or {})
    pro = {
        "meta": {"filename": Path(path).name, "version": 3},
        "board": {"design_settings": {
            "rules": {"min_clearance": clearance, "min_copper_edge_clearance": 0.3, "min_hole_clearance": 0.25,
                      "min_hole_to_hole": 0.25, "min_through_hole_diameter": 0.2, "min_track_width": 0.09,
                      "min_via_annular_width": 0.1, "min_via_diameter": 0.45, "min_microvia_diameter": 0.2,
                      "min_microvia_drill": 0.1, "min_silk_clearance": 0.0, "min_text_height": 0.8,
                      "min_text_thickness": 0.08, "min_resolved_spokes": 2, "min_connection_width": 0.15,
                      "solder_mask_clearance": 0.0, "solder_mask_min_width": 0.1},
            "rule_severities": sev, "defaults": {"board_outline_line_width": 0.1}}},
        "net_settings": {"classes": [{"name": "Default", "clearance": clearance, "track_width": track, "via_diameter": via_d,
                                      "via_drill": via_drill, "diff_pair_gap": 0.25, "diff_pair_width": 0.2, "pcb_color": "rgba(0,0,0,0.000)",
                                      "schematic_color": "rgba(0,0,0,0.000)", "wire_width": 6, "bus_width": 12, "line_style": 0,
                                      "microvia_diameter": 0.3, "microvia_drill": 0.1}],
                         "meta": {"version": 3}, "net_colors": None, "netclass_assignments": None, "netclass_patterns": []},
    }
    Path(path).write_text(json.dumps(pro, indent=2))


class PB:
    """a board under construction"""

    def __init__(self, netlist, size, origin=(30.0, 30.0), name="board", workdir=None, copper_layers=4):
        self.comps, self.netpins = read_net(netlist)
        self.W, self.H = size
        self.OX, self.OY = origin
        self.dir = Path(workdir or ".")
        self.name = name
        self.b = pcbnew.BOARD()
        self.b.SetCopperLayerCount(copper_layers)
        if copper_layers == 4:        # inner layers are planes: stops freerouting from routing signals on them (which would slot the planes)
            self.b.SetLayerType(pcbnew.In1_Cu, pcbnew.LT_POWER)
            self.b.SetLayerType(pcbnew.In2_Cu, pcbnew.LT_POWER)
        bds = self.b.GetDesignSettings()
        nc = bds.m_NetSettings.GetDefaultNetclass()
        nc.SetClearance(mm(0.15)); nc.SetTrackWidth(mm(0.15)); nc.SetViaDiameter(mm(0.5)); nc.SetViaDrill(mm(0.25))
        bds.m_MinClearance = mm(0.15)
        bds.m_TrackMinWidth = mm(0.15)
        bds.m_ViasMinSize = mm(0.45)
        bds.m_MinThroughDrill = mm(0.2)
        self.fps = {}
        self.netobj = {}
        self._cache = {}
        self.io = pcbnew.PCB_IO_MGR.PluginFind(pcbnew.PCB_IO_MGR.KICAD_SEXP)
        self.pad2net = {}
        for n, pins in self.netpins.items():
            for ref, pin in pins:
                self.pad2net[(ref, pin)] = n

    # ------------------------------------------------------------------ coordinates
    def P(self, x, y):
        return pcbnew.VECTOR2I(mm(self.OX + x), mm(self.OY + y))

    def net(self, name):
        if name is None:
            return None
        if name not in self.netobj:
            n = pcbnew.NETINFO_ITEM(self.b, name)
            self.b.Add(n)
            self.netobj[name] = n
        return self.netobj[name]

    # ------------------------------------------------------------------ footprints
    def _load(self, fpid):
        if fpid not in self._cache:
            lib, name = fpid.split(":")
            f = self.io.FootprintLoad(str(FPDIR / f"{lib}.pretty"), name)
            if f is None:
                raise RuntimeError(f"footprint {fpid}")
            self._cache[fpid] = f
        return self._cache[fpid]

    def add_all(self, strip_silk_for=("R_0402", "R_0603", "C_0402", "C_0603", "C_0805", "L_1812")):
        for ref, c in self.comps.items():
            base = self._load(c["fp"])
            f = pcbnew.Cast_to_FOOTPRINT(base.Duplicate())
            f.SetReference(ref)
            f.SetValue(c["value"])
            small = any(s in c["fp"] for s in strip_silk_for)
            f.Reference().SetLayer(pcbnew.F_Fab)       # references on the fab layer: no silkscreen clutter, still in the assembly SVGs
            f.Reference().SetVisible(True)
            f.Value().SetVisible(False)
            if small:
                for g in list(f.GraphicalItems()):
                    if g.GetLayer() == pcbnew.F_SilkS:      # silkscreen on the small passives only adds DRC noise: move to F.Fab
                        g.SetLayer(pcbnew.F_Fab)
            for p in f.Pads():
                n = self.pad2net.get((ref, p.GetNumber()))
                if n:
                    p.SetNet(self.net(n))
            if c["dnp"]:
                f.SetDNP(True)
                f.SetExcludedFromBOM(True)
                f.SetExcludedFromPosFiles(True)
                if "R_0402" in c["fp"] or "R_0603" in c["fp"] or "C_" in c["fp"]:
                    for p in f.Pads():     # no stencil aperture on unpopulated pads
                        ls = p.GetLayerSet()
                        ls.RemoveLayer(pcbnew.F_Paste)
                        p.SetLayerSet(ls)
            self.b.Add(f)
            self.fps[ref] = f

    def place(self, ref, x, y, rot=0.0, side="F"):
        f = self.fps[ref]
        f.SetPosition(self.P(x, y))
        if side == "B" and not f.IsFlipped():
            f.Flip(self.P(x, y), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        f.SetOrientationDegrees(rot)
        return f

    def pad(self, ref, num):
        for p in self.fps[ref].Pads():
            if p.GetNumber() == str(num):
                return p
        raise KeyError((ref, num))

    def pad_xy(self, ref, num):
        p = self.pad(ref, num).GetPosition()
        return (pcbnew.ToMM(p.x) - self.OX, pcbnew.ToMM(p.y) - self.OY)

    def pad_dir(self, ref, num, other=None):
        """vector from footprint centre to pad centre (mm)"""
        p = self.pad(ref, num).GetPosition()
        c = self.fps[ref].GetPosition()
        return (pcbnew.ToMM(p.x - c.x), pcbnew.ToMM(p.y - c.y))

    def place_oriented(self, ref, x, y, want1, side="F"):
        """place a 2-pad part so that pad 1 lies in direction want1 (unit-ish vector) from the centre"""
        wx, wy = want1
        for rot in (0, 90, 180, 270):
            self.place(ref, x, y, rot, side)
            dx, dy = self.pad_dir(ref, "1", None)
            if abs(dx * wy - dy * wx) < 1e-3 * (abs(dx) + abs(dy) + 1) and dx * wx + dy * wy > 0:
                return rot
        raise RuntimeError(f"cannot orient {ref}")

    def bbox(self, ref):
        bb = self.fps[ref].GetBoundingBox(False, False)
        return pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())

    # ------------------------------------------------------------------ copper
    def track(self, netname, layer, pts, w=0.15, locked=True):
        n = self.net(netname)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
                continue
            t = pcbnew.PCB_TRACK(self.b)
            t.SetStart(self.P(x1, y1)); t.SetEnd(self.P(x2, y2))
            t.SetWidth(mm(w)); t.SetLayer(layer); t.SetNet(n)
            t.SetLocked(locked)
            self.b.Add(t)

    def via(self, netname, x, y, d=0.5, drill=0.25, locked=True):
        v = pcbnew.PCB_VIA(self.b)
        v.SetPosition(self.P(x, y))
        v.SetViaType(pcbnew.VIATYPE_THROUGH)
        v.SetWidth(mm(d)); v.SetDrill(mm(drill))
        v.SetLayerPair(F, B)
        v.SetNet(self.net(netname))
        v.SetLocked(locked)
        self.b.Add(v)
        return v

    def poly(self, layer, pts, netname=None, priority=0, keepout=None):
        z = pcbnew.ZONE(self.b)
        z.SetLayer(layer)
        if netname:
            z.SetNet(self.net(netname))
        z.SetMinThickness(mm(0.15))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        z.SetThermalReliefGap(mm(0.25)); z.SetThermalReliefSpokeWidth(mm(0.3))
        z.SetLocalClearance(mm(0.15))
        z.SetAssignedPriority(priority)
        z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
        ol = z.Outline(); ol.NewOutline()
        for (x, y) in pts:
            ol.Append(mm(self.OX + x), mm(self.OY + y))
        if keepout:
            z.SetIsRuleArea(True)
            z.SetDoNotAllowCopperPour(keepout.get("pour", False)); z.SetDoNotAllowTracks(keepout.get("tracks", False))
            z.SetDoNotAllowVias(keepout.get("vias", False)); z.SetDoNotAllowPads(keepout.get("pads", False))
            z.SetDoNotAllowFootprints(keepout.get("fps", False))
        self.b.Add(z)
        return z

    def outline(self, pts=None, w=0.1):
        pts = pts or [(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)]
        for i in range(len(pts)):
            s = pcbnew.PCB_SHAPE(self.b)
            s.SetShape(pcbnew.SHAPE_T_SEGMENT); s.SetLayer(pcbnew.Edge_Cuts)
            s.SetStart(self.P(*pts[i])); s.SetEnd(self.P(*pts[(i + 1) % len(pts)])); s.SetWidth(mm(w))
            self.b.Add(s)

    def text(self, s, x, y, size=1.0, layer=pcbnew.F_SilkS, rot=0, mirror=False, bold=False):
        t = pcbnew.PCB_TEXT(self.b)
        t.SetText(s); t.SetLayer(layer)
        t.SetPosition(self.P(x, y)); t.SetTextSize(pcbnew.VECTOR2I(mm(size), mm(size)))
        t.SetTextThickness(mm(max(0.15, size * 0.15)))
        t.SetTextAngleDegrees(rot)
        if mirror or layer == pcbnew.B_SilkS:
            t.SetMirrored(True)
        t.SetBold(bold)
        self.b.Add(t)
        return t

    def add_lib_fp(self, fpid, ref, x, y, rot=0, side="F", netmap=None):
        base = self._load(fpid)
        f = pcbnew.Cast_to_FOOTPRINT(base.Duplicate())
        f.SetReference(ref); f.Reference().SetVisible(False); f.Value().SetVisible(False)
        self.b.Add(f)
        f.SetPosition(self.P(x, y))
        if side == "B":
            f.Flip(self.P(x, y), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        f.SetOrientationDegrees(rot)
        for p in f.Pads():
            if netmap and p.GetNumber() in netmap:
                p.SetNet(self.net(netmap[p.GetNumber()]))
        self.fps[ref] = f
        return f

    # ------------------------------------------------------------------ io
    def save(self, path=None):
        path = Path(path or self.dir / f"{self.name}.kicad_pcb")
        pcbnew.SaveBoard(str(path), self.b)
        inject_stackup(path)
        return path


def load(path):
    return pcbnew.LoadBoard(str(path))


def fill(b):
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())


def export_dsn(path_pcb, dsn):
    b = pcbnew.LoadBoard(str(path_pcb))
    ok = pcbnew.ExportSpecctraDSN(b, str(dsn))
    return ok


def import_ses(path_pcb, ses, out):
    b = pcbnew.LoadBoard(str(path_pcb))
    ok = pcbnew.ImportSpecctraSES(b, str(ses))
    fill(b)
    pcbnew.SaveBoard(str(out), b)
    inject_stackup(out)
    return ok


def run_freerouting(dsn, ses, log, passes=40, threads=4, timeout=None, extra=()):
    """blocking call; run it under nohup from the make_pcb_* scripts"""
    cmd = [FR_JAVA, "-Xmx9g", "-jar", str(FR_JAR), "--gui.enabled=false", "--router.optimizer.enabled=false", "-de", str(dsn), "-do", str(ses),
           "-mp", str(passes), *extra]       # optimizer off: it ran out of heap on the 1650-pin board and lost the routed result
    with open(log, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def drc(pcb, rpt, json_out=None, extra=()):
    cmd = ["kicad-cli", "pcb", "drc", "--exit-code-violations", "--severity-all", "--all-track-errors", "--schematic-parity" if False else "--units", "mm", "-o", str(rpt), str(pcb)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if json_out:
        subprocess.run(["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all", "--all-track-errors", "--units", "mm", "-o", str(json_out), str(pcb)],
                       capture_output=True, text=True)
    txt = Path(rpt).read_text()
    m = re.findall(r"Found (\d+) (DRC violations|unconnected pads|Footprint errors|\w+)", txt)
    return r.returncode, txt


def run_stage(stage, outdir, prefix, passes=40):
    """dsn | route | ses for <outdir>/<prefix>_placed.kicad_pcb -> <prefix>.dsn/.ses -> <prefix>.kicad_pcb (+ DRC report)"""
    outdir = Path(outdir)
    placed = outdir / f"{prefix}_placed.kicad_pcb"
    dsn, ses, routed = outdir / f"{prefix}.dsn", outdir / f"{prefix}.ses", outdir / f"{prefix}.kicad_pcb"
    t0 = time.time()
    if stage == "dsn":
        print("ExportSpecctraDSN", export_dsn(placed, dsn))
    elif stage == "route":
        rc = run_freerouting(dsn, ses, outdir / "freerouting.log", passes=passes)
        print("freerouting rc", rc, f"{time.time()-t0:.0f}s")
    elif stage == "ses":
        print("ImportSpecctraSES", import_ses(placed, ses, routed))
        rc, txt = drc(routed, outdir / "drc.rpt", outdir / "drc.json")
        if "track_dangling" in txt:
            print("trimmed dangling bus stubs:", trim_dangling(routed, outdir / "drc.json"))
            rc, txt = drc(routed, outdir / "drc.rpt", outdir / "drc.json")
        print("DRC rc", rc, " ".join(re.findall(r"\*\* Found [^*]+\*\*", txt)))


# ---------------------------------------------------------------------------------------------------------------------------
# scripted fanout of plane-net SMD pads (freerouting cannot drop vias onto planes when In1/In2 are typed as power layers)
class Rects:
    """coarse spatial hash of copper rectangles (mm, board coords) of every net"""

    def __init__(self, cell=2.0):
        self.cell, self.h, self.items = cell, {}, []

    def add(self, x0, y0, x1, y1, net, layers):
        i = len(self.items)
        self.items.append((x0, y0, x1, y1, net, layers))
        c = self.cell
        for gx in range(int(x0 // c), int(x1 // c) + 1):
            for gy in range(int(y0 // c), int(y1 // c) + 1):
                self.h.setdefault((gx, gy), []).append(i)

    def clear(self, x, y, r, net, layers):
        """True if a disc (centre x,y radius r) on the given layer set touches no rectangle of another net"""
        c = self.cell
        seen = set()
        for gx in range(int((x - r) // c), int((x + r) // c) + 1):
            for gy in range(int((y - r) // c), int((y + r) // c) + 1):
                for i in self.h.get((gx, gy), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    x0, y0, x1, y1, n, ly = self.items[i]
                    if n == net or not (ly & layers):
                        continue
                    dx = max(x0 - x, 0, x - x1)
                    dy = max(y0 - y, 0, y - y1)
                    if dx * dx + dy * dy < r * r:
                        return False
        return True


def build_rects(pb, extra_keepouts=()):
    R = Rects()
    to = pcbnew.ToMM
    for f in pb.b.GetFootprints():
        for p in f.Pads():
            bb = p.GetBoundingBox()
            ly = set()
            if p.IsOnLayer(F):
                ly.add("F")
            if p.IsOnLayer(B):
                ly.add("B")
            if p.GetAttribute() in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH):
                ly = {"F", "B"}
            R.add(to(bb.GetX()) - pb.OX, to(bb.GetY()) - pb.OY, to(bb.GetRight()) - pb.OX, to(bb.GetBottom()) - pb.OY, p.GetNetname(), frozenset(ly))
    for t in pb.b.GetTracks():
        bb = t.GetBoundingBox()
        if t.Type() == pcbnew.PCB_VIA_T:
            ly = frozenset({"F", "B"})
            bb = t.GetBoundingBox()
            R.add(to(bb.GetX()) - pb.OX, to(bb.GetY()) - pb.OY, to(bb.GetRight()) - pb.OX, to(bb.GetBottom()) - pb.OY, "VIA:" + t.GetNetname(), ly)
            continue
        else:
            ly = frozenset({"F" if t.GetLayer() == F else "B"})
        R.add(to(bb.GetX()) - pb.OX, to(bb.GetY()) - pb.OY, to(bb.GetRight()) - pb.OX, to(bb.GetBottom()) - pb.OY, t.GetNetname(), ly)
    for (x0, y0, x1, y1, ly) in extra_keepouts:
        R.add(x0, y0, x1, y1, "__keepout__", frozenset(ly))
    return R


def fanout_planes(pb, nets=("GND", "+5VA"), margin=0.2, via_d=0.5, via_drill=0.25, extra_keepouts=(), bounds=None):
    """one via + short stub next to every unconnected SMD pad of a plane net; returns (n_done, [failed refs])"""
    to = pcbnew.ToMM
    R = build_rects(pb, extra_keepouts)
    done, failed = 0, []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)]
    tracks = list(pb.b.GetTracks())
    for f in list(pb.b.GetFootprints()):
        for p in f.Pads():
            n = p.GetNetname()
            if n not in nets or p.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue
            # already tied by a (scripted) track?
            if any(t.Type() == pcbnew.PCB_TRACE_T and t.GetNetname() == n and (p.HitTest(t.GetStart()) or p.HitTest(t.GetEnd())) for t in tracks if abs(to(t.GetStart().x - p.GetPosition().x)) < 6):
                continue
            px, py = to(p.GetPosition().x) - pb.OX, to(p.GetPosition().y) - pb.OY
            lay = "F" if p.IsOnLayer(F) else "B"
            side = F if lay == "F" else B
            half = max(to(p.GetSizeX()), to(p.GetSizeY())) / 2
            ok = False
            for d in [half + 0.5, half + 0.9, half + 1.4, half + 2.0]:
                for dx, dy in dirs:
                    vx, vy = px + dx * d, py + dy * d
                    if bounds and not (bounds[0] < vx < bounds[2] and bounds[1] < vy < bounds[3]):
                        continue
                    r = via_d / 2 + margin
                    if not R.clear(vx, vy, r, n, frozenset({"F", "B"})):
                        continue
                    # stub: samples along pad centre -> via on the pad layer, half-width 0.1 + clearance
                    good = True
                    for k in range(1, 9):
                        sx, sy = px + (vx - px) * k / 8, py + (vy - py) * k / 8
                        if not R.clear(sx, sy, 0.1 + margin - 0.02, n, frozenset({lay})) and (k * d / 8 > half * 0.9):
                            good = False
                            break
                    if good:
                        pb.track(n, side, [(px, py), (vx, vy)], 0.2)
                        pb.via(n, vx, vy, via_d, via_drill)
                        R.add(vx - via_d / 2, vy - via_d / 2, vx + via_d / 2, vy + via_d / 2, "VIA:" + n, frozenset({"F", "B"}))
                        xs0, xs1 = sorted((px, vx)); ys0, ys1 = sorted((py, vy))
                        R.add(xs0 - 0.1, ys0 - 0.1, xs1 + 0.1, ys1 + 0.1, n, frozenset({lay}))
                        done += 1
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                failed.append(f"{f.GetReference()}.{p.GetNumber()}")
    return done, failed


def trim_dangling(pcb_path, drc_json, tol=0.01):
    """shorten tracks that the DRC reports with a dangling end (scripted column-bus arrival stubs that freerouting did not need):
    the dangling end is moved to the nearest connected via/pad of the same track net along the track"""
    d = json.loads(Path(drc_json).read_text())
    ids = {}
    for v in d.get("violations", []):
        if v["type"] == "track_dangling":
            for it in v["items"]:
                ids[it["uuid"]] = it["pos"]
    b = pcbnew.LoadBoard(str(pcb_path))
    to, mmf = pcbnew.ToMM, pcbnew.FromMM
    vias = [(t.GetPosition(), t.GetNetname()) for t in b.GetTracks() if t.Type() == pcbnew.PCB_VIA_T]
    n = 0
    for t in list(b.GetTracks()):
        if t.Type() != pcbnew.PCB_TRACE_T or t.m_Uuid.AsString() not in ids:
            continue
        sx, sy, ex, ey = t.GetStart().x, t.GetStart().y, t.GetEnd().x, t.GetEnd().y
        if sx != ex:
            continue                       # only the vertical buses
        cands = [p for p, nn in vias if nn == t.GetNetname() and abs(p.x - sx) < 5000 and min(sy, ey) - 5000 <= p.y <= max(sy, ey) + 5000]
        if not cands:
            continue
        top = min(cands, key=lambda p: p.y)
        if sy < ey:
            t.SetStart(pcbnew.VECTOR2I(sx, top.y))
        else:
            t.SetEnd(pcbnew.VECTOR2I(ex, top.y))
        n += 1
    fill(b)
    pcbnew.SaveBoard(str(pcb_path), b)
    inject_stackup(pcb_path)
    return n
