#!/usr/bin/env python3
"""Fab outputs for a routed board (M3b): gerbers + drill zip, JLC BOM/CPL (DNP excluded), per-layer SVG, GLB, STEP, renders, DRC, ERC,
and the "CPL is the model" check for tiles.

    hw/.venv-eda/bin/python hw/board/make_fab.py tile 2|1|3     |    carrier        [--routed PATH] [--no-3d]

Output: docs/artifacts/board/{tile_L1,tile_L2,tile_L3,carrier}/ .  Run from anywhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE / "pcb"))
from netlist import read_net  # noqa: E402

ART = REPO / "docs" / "artifacts" / "board"
HAND = ("J", "CN", "H", "FID", "TP", "JP", "ART")          # THT headers / mechanical / test pads: not in the SMT BOM/CPL


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r


def is_smt_part(ref, c):
    if ref.startswith(("JIN", "JOUT", "JPS", "JMON", "JT", "JADC", "CN", "J1", "TP", "JPID")):
        return False
    return not c["fields"].get("LCSC", "").startswith("n/a")


def fab(kind, layer, routed, no3d=False, model=None, placed_only=False, full_glb=False):
    name = "carrier" if kind == "carrier" else f"tile_L{layer}"
    out = ART / name
    out.mkdir(parents=True, exist_ok=True)
    pcb = Path(routed).resolve()
    sch = HERE / ("carrier" if kind == "carrier" else f"tile/layer{layer}") / ("carrier.kicad_sch" if kind == "carrier" else "tile.kicad_sch")
    netf = sch.with_suffix(".net")
    comps, nets = read_net(netf)
    res = {"board": name, "pcb": str(pcb.relative_to(REPO)), "sha256_pcb": hashlib.sha256(pcb.read_bytes()).hexdigest()}
    if not placed_only:
        # --- gerbers + drill
        gdir = out / "gerbers"
        shutil.rmtree(gdir, ignore_errors=True)
        gdir.mkdir()
        sh(["kicad-cli", "pcb", "export", "gerbers", "-o", str(gdir) + "/", "--board-plot-params", str(pcb)])
        sh(["kicad-cli", "pcb", "export", "drill", "-o", str(gdir) + "/", "--format", "excellon", "--excellon-separate-th",
            "--generate-map", "--map-format", "gerberx2", str(pcb)])
        zp = out / f"{name}_gerbers.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(gdir.iterdir()):
                if f.suffix in (".gbr", ".drl", ".gbrjob") or f.name.endswith((".gbr", ".drl")):
                    z.write(f, f.name)
        res["gerber_files"] = len(list(gdir.glob("*.gb*"))) + len(list(gdir.glob("*.drl")))
        shutil.rmtree(gdir)         # keep only the zip
    else:
        res['routed'] = False
        res['note'] = 'PLACED ONLY (not routed): no gerbers or drill files are produced'
    # --- position file (DNP excluded) -> JLC CPL
    pos = out / f"{name}_pos_kicad.csv"
    sh(["kicad-cli", "pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both", "--exclude-dnp", "-o", str(pos), str(pcb)])
    cpl_rows, bom = [], defaultdict(list)
    for row in csv.DictReader(open(pos)):
        ref = row["Ref"]
        c = comps.get(ref)
        if c is None or not is_smt_part(ref, c) or c["dnp"]:
            continue
        cpl_rows.append([ref, f'{float(row["PosX"]):.4f}mm', f'{float(row["PosY"]):.4f}mm', "Top" if row["Side"] == "top" else "Bottom",
                         f'{float(row["Rot"]):.0f}'])
        bom[(c["value"], c["fp"].split(":")[-1], c["fields"].get("LCSC", ""))].append(ref)
    with open(out / f"{name}_cpl_jlc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
        w.writerows(sorted(cpl_rows, key=lambda r: r[0]))
    with open(out / f"{name}_bom_jlc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
        for (val, fp, lcsc), refs in sorted(bom.items(), key=lambda kv: kv[1][0]):
            w.writerow([val, ",".join(sorted(refs)), fp, lcsc])
    res.update(cpl_rows=len(cpl_rows), bom_lines=len(bom), bom_parts=sum(len(v) for v in bom.values()))
    hand = sorted(r for r, c in comps.items() if not is_smt_part(r, c) and not c["dnp"])
    res["hand_fit_or_not_assembled"] = hand
    # --- "the CPL is the model"
    if kind == "tile":
        res["cpl_is_model"] = check_cpl_model(out / f"{name}_cpl_jlc.csv", layer, model)
    # --- svg per layer
    sdir = out / "svg"
    shutil.rmtree(sdir, ignore_errors=True)
    sdir.mkdir()
    sh(["kicad-cli", "pcb", "export", "svg", "--layers", "F.Cu,In1.Cu,In2.Cu,B.Cu,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,F.Paste,B.Paste,Edge.Cuts",
        "--mode-multi", "--page-size-mode", "2", "--exclude-drawing-sheet", "-o", str(sdir) + "/", str(pcb)])
    # --- drc / erc
    sh(["kicad-cli", "pcb", "drc", "--exit-code-violations", "--severity-all", "--all-track-errors", "-o", str(out / "drc.rpt"), str(pcb)])
    sh(["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all", "--all-track-errors", "-o", str(out / "drc.json"), str(pcb)])
    sh(["kicad-cli", "sch", "erc", "--format", "json", "--severity-all", "-o", str(out / "erc.json"), str(sch)])
    sh(["kicad-cli", "sch", "erc", "--severity-all", "-o", str(out / "erc.rpt"), str(sch)])
    d = json.loads((out / "drc.json").read_text())
    e = json.loads((out / "erc.json").read_text())
    res["drc"] = dict(violations=len(d.get("violations", [])), unconnected=len(d.get("unconnected_items", [])),
                      errors=sum(1 for v in d.get("violations", []) if v["severity"] == "error"),
                      warnings=sum(1 for v in d.get("violations", []) if v["severity"] == "warning"))
    res["erc"] = dict(violations=sum(len(s.get("violations", [])) for s in e.get("sheets", [])))
    for f in sdir.glob(f"{pcb.stem}-*.svg"):
        f.rename(sdir / f.name.replace(f"{pcb.stem}-", f"{name}-", 1))
    # --- drill svg for the D3 layer viewer (kicad-cli has no drill layer in SVG export)
    res["drill_svg"] = drill_svg(pcb, sdir / f"{name}-Drill.svg")
    # --- 3d: GLB for three.js (+ STEP) and one sanity render per side (own check only)
    if not no3d:
        if kind == "carrier":     # full: copper (incl. the gold art island), pads, silk, mask, all 3D models (16 MB)
            glb_args = ["--include-tracks", "--include-pads", "--include-zones", "--include-silkscreen", "--include-soldermask", "--min-distance", "0.03mm"]
        else:                     # lite (~3-9 MB): body, mask, silk and the inductors / ICs / connectors only; copper comes from the layer SVGs.
            # the full tile GLB with tracks, pads and every 0402/0603 model is 40-50 MB per tile (40k separate meshes): make_fab.py --full-glb
            glb_args = (["--include-tracks", "--include-pads", "--include-silkscreen", "--include-soldermask", "--min-distance", "0.03mm"] if full_glb else
                        ["--fuse-shapes", "--fill-all-vias", "--component-filter", "L*,U*,J*,H*,Q*", "--include-silkscreen", "--include-soldermask"])
        sh(["kicad-cli", "pcb", "export", "glb", "--subst-models", *glb_args, "--force", "-o", str(out / f"{name}.glb"), str(pcb)])
        if not placed_only:
            sh(["kicad-cli", "pcb", "export", "step", "--subst-models", "--force", "-o", str(out / f"{name}.step"), str(pcb)])
        for side in ("top", "bottom"):
            sh(["kicad-cli", "pcb", "render", "--side", side, "--quality", "basic", "-w", "1600", "-h", "1100", "-o",
                str(out / f"sanity_{side}.png"), str(pcb)])
        res["glb_materials"] = glb_materials(out / f"{name}.glb")
        res["glb_bytes"] = (out / f"{name}.glb").stat().st_size
    shutil.copy(pcb, out / f"{name}.kicad_pcb")
    if pcb.with_suffix(".kicad_pro").exists():
        shutil.copy(pcb.with_suffix(".kicad_pro"), out / f"{name}.kicad_pro")     # DRC rules next to the board copy (kicad-cli reads them)
    (out / "fab_summary.json").write_text(json.dumps(res, indent=1))
    return res


def drill_svg(pcb, path):
    """all drilled holes (PTH pads, NPTH pads, vias) as circles in board-local mm (origin = top-left of the Edge.Cuts bounding box)"""
    os.environ.setdefault("KICAD9_FOOTPRINT_DIR", "/usr/share/kicad/footprints")
    import pcbnew
    b = pcbnew.LoadBoard(str(pcb))
    bb = b.GetBoardEdgesBoundingBox()
    ox, oy = pcbnew.ToMM(bb.GetX()), pcbnew.ToMM(bb.GetY())
    w, h = pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())
    holes = []
    for f in b.GetFootprints():
        for p in f.Pads():
            d = pcbnew.ToMM(p.GetDrillSizeX())
            if d > 0:
                holes.append((pcbnew.ToMM(p.GetPosition().x) - ox, pcbnew.ToMM(p.GetPosition().y) - oy, d, "npth" if p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH else "pth"))
    for t in b.GetTracks():
        if t.Type() == pcbnew.PCB_VIA_T:
            holes.append((pcbnew.ToMM(t.GetPosition().x) - ox, pcbnew.ToMM(t.GetPosition().y) - oy, pcbnew.ToMM(t.GetDrillValue()), "via"))
    cls = {"pth": "#e8c547", "npth": "#ff6b6b", "via": "#7fd1ff"}
    body = "".join(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{d/2:.3f}" fill="none" stroke="{cls[k]}" stroke-width="0.08" class="{k}"/>' for x, y, d, k in holes)
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.3f} {h:.3f}" width="{w:.3f}mm" height="{h:.3f}mm">'
                          f'<title>drill holes (mm, board-local; PTH / NPTH / via)</title>{body}</svg>')
    return dict(holes=len(holes), width_mm=w, height_mm=h)


def glb_materials(glb):
    import struct
    b = Path(glb).read_bytes()
    L = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + L])
    out = []
    for i, m in enumerate(j.get("materials", [])):
        pbr = m.get("pbrMetallicRoughness", {})
        c = pbr.get("baseColorFactor", [1, 1, 1, 1])
        if pbr.get("metallicFactor", 0) >= 0.9 and c[2] < 0.1:
            role = "copper_enig_gold"
        elif c[0] > 0.9 and c[1] > 0.9 and pbr.get("metallicFactor", 0) == 0:
            role = "silkscreen_white"
        elif c[3] < 1.0 and max(c[:3]) < 0.2:
            role = "soldermask_black"
        else:
            role = "other"
        out.append(dict(index=i, name=m.get("name"), role=role, glb_pbr=pbr))
    return out


def check_cpl_model(cpl, layer, model):
    """populated RXP/RXN refdes in the CPL == the model's non-zero weights (RXP: W=+1, RXN: W=-1)"""
    z = np.load(model, allow_pickle=True)
    W = z[f"L{layer-1}_W"].astype(int)           # [O, I, K]
    want = set()
    for o in range(W.shape[0]):
        for i in range(W.shape[1]):
            for k in range(W.shape[2]):
                if W[o, i, k] == 1:
                    want.add(f"RXP_{o}_{i}_{k}")
                elif W[o, i, k] == -1:
                    want.add(f"RXN_{o}_{i}_{k}")
    got = {r["Designator"] for r in csv.DictReader(open(cpl)) if r["Designator"].startswith(("RXP_", "RXN_"))}
    return dict(model=str(Path(model).name), model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(), n_nonzero_weights=len(want),
                n_cpl_crossbar=len(got), match=(want == got), missing=sorted(want - got)[:5], extra=sorted(got - want)[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("tile", "carrier"))
    ap.add_argument("layer", nargs="?", type=int, default=2)
    ap.add_argument("--routed", default=None)
    ap.add_argument("--no-3d", action="store_true")
    ap.add_argument("--placed-only", action="store_true")
    ap.add_argument("--full-glb", action="store_true")
    ap.add_argument("--model", default=str(REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"))
    a = ap.parse_args()
    routed = a.routed or (HERE / "pcb" / ("carrier/carrier.kicad_pcb" if a.kind == "carrier" else f"tile_L{a.layer}/tile_L{a.layer}.kicad_pcb"))
    res = fab(a.kind, a.layer, routed, a.no_3d, a.model, a.placed_only, a.full_glb)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
