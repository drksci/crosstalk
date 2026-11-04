"""P0 demo PCB: place + outline + 4-layer stackup + planes, DSN export, freerouting, SES import, zone fill.
Run with hw/.venv-eda/bin/python:  make_pcb.py place|dsn|ses|all   (paths are relative to this file)."""
import os, subprocess, sys
from pathlib import Path
import pcbnew

HERE = Path(__file__).parent
os.environ.setdefault("KICAD9_FOOTPRINT_DIR", "/usr/share/kicad/footprints")
FR_JAR = Path(os.environ.get("FREEROUTING_JAR", "/home/user/freerouting/freerouting/build/libs/freerouting-current-executable.jar"))
FR_JAVA = os.environ.get("FREEROUTING_JAVA", "/root/.gradle/jdks/eclipse_adoptium-25-amd64-linux.2/bin/java")
mm = pcbnew.FromMM
W, H = 84.0, 54.0          # board size, mm
OX, OY = 100.0, 80.0       # board origin on the sheet, mm

# (x, y, rotation_deg) in board coordinates
PLACE = {
    "J1": (7, 12, 0), "R1": (18, 12, 90), "L1": (31, 12, 0), "L2": (49, 12, 0), "R2": (68, 12, 90),
    "C1": (40, 21, 90), "C2": (58, 21, 90),
    "J2": (7, 32, 0), "U1": (40, 32, 0), "U2": (58, 32, 0),
    "R3": (44, 44, 90), "R4": (50, 44, 90), "R5": (60, 44, 90), "U3": (72, 34, 0),
    "C3": (24, 36, 90), "C4": (28, 36, 90), "J3": (78, 46, 90),
}

# JLC04161H-7628 4-layer 1.6 mm stackup. pcbnew's python API does not expose the stackup descriptor,
# so the (stackup ...) S-expression is injected into the saved file (and survives load/save).
STACKUP = """(stackup
\t\t\t(layer "F.SilkS" (type "Top Silk Screen"))
\t\t\t(layer "F.Paste" (type "Top Solder Paste"))
\t\t\t(layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))
\t\t\t(layer "F.Cu" (type "copper") (thickness 0.035))
\t\t\t(layer "dielectric 1" (type "prepreg") (thickness 0.2104) (material "7628") (epsilon_r 4.4) (loss_tangent 0.02))
\t\t\t(layer "In1.Cu" (type "copper") (thickness 0.0152))
\t\t\t(layer "dielectric 2" (type "core") (thickness 1.065) (material "FR4") (epsilon_r 4.6) (loss_tangent 0.02))
\t\t\t(layer "In2.Cu" (type "copper") (thickness 0.0152))
\t\t\t(layer "dielectric 3" (type "prepreg") (thickness 0.2104) (material "7628") (epsilon_r 4.4) (loss_tangent 0.02))
\t\t\t(layer "B.Cu" (type "copper") (thickness 0.035))
\t\t\t(layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.01))
\t\t\t(layer "B.Paste" (type "Bottom Solder Paste"))
\t\t\t(layer "B.SilkS" (type "Bottom Silk Screen"))
\t\t\t(copper_finish "ENIG")
\t\t\t(dielectric_constraints no)
\t\t)"""

def inject_stackup(path):
    import re
    t = Path(path).read_text()
    t = re.sub(r"\(stackup\b.*?\n\t\t\)\n?", "", t, flags=re.S)       # drop any existing stackup
    i = t.index("(setup")
    j = t.index("\n", i)
    t = t[:j + 1] + "\t\t" + STACKUP + "\n" + t[j + 1:]
    Path(path).write_text(t)

def net_by_name(b, name):
    return b.FindNet(name)

def place():
    b = pcbnew.LoadBoard(str(HERE / "demo_raw.kicad_pcb"))
    b.SetCopperLayerCount(4)
    bds = b.GetDesignSettings()
    nc = bds.m_NetSettings.GetDefaultNetclass()
    nc.SetClearance(mm(0.2)); nc.SetTrackWidth(mm(0.25)); nc.SetViaDiameter(mm(0.6)); nc.SetViaDrill(mm(0.3))
    for f in b.GetFootprints():
        x, y, r = PLACE[f.GetReference()]
        f.SetPosition(pcbnew.VECTOR2I(mm(OX + x), mm(OY + y)))
        f.SetOrientationDegrees(r)
    # outline
    pts = [(0, 0), (W, 0), (W, H), (0, H)]
    for i in range(4):
        s = pcbnew.PCB_SHAPE(b); s.SetShape(pcbnew.SHAPE_T_SEGMENT); s.SetLayer(pcbnew.Edge_Cuts)
        s.SetStart(pcbnew.VECTOR2I(mm(OX + pts[i][0]), mm(OY + pts[i][1])))
        s.SetEnd(pcbnew.VECTOR2I(mm(OX + pts[(i + 1) % 4][0]), mm(OY + pts[(i + 1) % 4][1])))
        s.SetWidth(mm(0.1)); b.Add(s)
    # planes: In1 = GND, In2 = +5V
    for layer, net in ((pcbnew.In1_Cu, "GND"), (pcbnew.In2_Cu, "+5V")):
        z = pcbnew.ZONE(b); z.SetLayer(layer); z.SetNet(net_by_name(b, net))
        z.SetMinThickness(mm(0.2)); z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        ol = z.Outline(); ol.NewOutline()
        for (px, py) in [(0.5, 0.5), (W - .5, .5), (W - .5, H - .5), (.5, H - .5)]:
            ol.Append(mm(OX + px), mm(OY + py))
        b.Add(z)
    # silk title
    t = pcbnew.PCB_TEXT(b); t.SetText("DLNN P0 DEMO  LC ladder + taps"); t.SetLayer(pcbnew.F_SilkS)
    t.SetPosition(pcbnew.VECTOR2I(mm(OX + 42), mm(OY + 51))); t.SetTextSize(pcbnew.VECTOR2I(mm(1.2), mm(1.2))); b.Add(t)
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    pcbnew.SaveBoard(str(HERE / "demo_placed.kicad_pcb"), b)
    inject_stackup(HERE / "demo_placed.kicad_pcb")
    print("placed:", len(b.GetFootprints()), "footprints, copper layers", b.GetCopperLayerCount())

def dsn():
    b = pcbnew.LoadBoard(str(HERE / "demo_placed.kicad_pcb"))
    ok = pcbnew.ExportSpecctraDSN(b, str(HERE / "demo.dsn"))
    print("ExportSpecctraDSN ->", ok)
    return ok

def route():
    cmd = [FR_JAVA, "-jar", str(FR_JAR), "--gui.enabled=false", "-de", str(HERE / "demo.dsn"),
           "-do", str(HERE / "demo.ses"), "-mp", "30"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    (HERE / "freerouting.log").write_text(r.stdout + r.stderr)
    print("freerouting rc", r.returncode)
    return r.returncode == 0

def ses():
    b = pcbnew.LoadBoard(str(HERE / "demo_placed.kicad_pcb"))
    ok = pcbnew.ImportSpecctraSES(b, str(HERE / "demo.ses"))
    print("ImportSpecctraSES ->", ok)
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    pcbnew.SaveBoard(str(HERE / "demo.kicad_pcb"), b)

if __name__ == "__main__":
    for st in (sys.argv[1:] or ["all"]):
        if st in ("place", "all"): place()
        if st in ("dsn", "all"): dsn()
        if st in ("route", "all"): route()
        if st in ("ses", "all"): ses()
