#!/usr/bin/env python3
"""Board art (M3b): figure_source.jpg -> clean solid shapes + a coarse dot screen -> KiCad footprints (.kicad_mod), SVG and preview PNGs.

    hw/.venv-eda/bin/python hw/board/art/make_art.py --box 53 76.5 --name carrier [--out DIR]

Pipeline (all dimensions in mm after scaling to the box; the source is a 1200 x 1732 halftone whose dots are ~0.1 mm at board scale,
far below the JLC minimum feature):
  1. dark = 1 - gray; despeckle by blur (sigma 0.11 mm) + threshold -> solid silhouette, solid dark areas, the vein gaps on the arm
     and the machine-side outlines.
  2. mid-tone regions (the grey bars and discs of the machine side: broad areas of intermediate tone, found by blurring at 0.45 mm and
     opening the mid-tone mask with a 0.5 mm disc) are replaced by a deliberate dot screen: 45 degree grid, pitch 0.60 mm, dot diameter
     0.25..0.40 mm following the local tone (copper gap between dots >= 0.2 mm).
  3. morphological close/open with a 0.1 mm disc: no copper feature and no gap below 0.2 mm survives (checked with buffers at the end).
  4. contours -> polygons (shapely), polygons with holes are split along vertical cuts through each hole into hole-free polygons
     (the .kicad_mod / .kicad_pcb polygon format cannot hold holes); output coordinates rounded to 1 um.
Variants written to OUT/:
  art_<name>_gold.kicad_mod   F.Cu fill + F.Mask opening (placed flipped on the bottom: B.Cu + B.Mask = exposed ENIG), no net
  art_<name>_silk.kicad_mod   F.SilkS fill (placed flipped on the bottom: B.SilkS), the generated fallback
  art_<name>.svg, art_<name>_processed.png, art_<name>_stats.json
The footprints are authored as seen from the board side that carries them, mirror handling is done by pcbnew when they are placed on B.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiPolygon, Polygon, box as sbox
from shapely.ops import split, unary_union

HERE = Path(__file__).resolve().parent
SRC = HERE / "figure_source.jpg"


def disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def process(src, box_w, box_h, px_mm=0.02, pitch=0.60, min_feat=0.2, tone_curve=1.0):
    im = Image.open(src).convert("L")
    g = 1.0 - np.asarray(im).astype(np.float32) / 255.0
    # --- crop to the figure: largest blobs of the 2.5 px-blurred threshold
    b0 = ndi.gaussian_filter(g, 2.5) > 0.5
    lab, n = ndi.label(b0)
    sizes = ndi.sum(b0, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 400])
    ys, xs = np.where(keep)
    x0, x1, y0, y1 = max(xs.min() - 12, 0), min(xs.max() + 12, g.shape[1]), max(ys.min() - 12, 0), min(ys.max() + 12, g.shape[0])
    g = g[y0:y1, x0:x1]
    h, w = g.shape
    s_mm = min(box_w / w, box_h / h)                  # mm per source pixel
    W_mm, H_mm = w * s_mm, h * s_mm
    # --- work raster at px_mm per pixel
    nw, nh = int(round(W_mm / px_mm)), int(round(H_mm / px_mm))
    gw = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((nw, nh), Image.BICUBIC)).astype(np.float32) / 255.0
    sig_px = lambda mm_: mm_ / px_mm
    sharp = ndi.gaussian_filter(gw, sig_px(0.11)) > 0.46
    # --- mid-tone regions -> dot screen
    tone = ndi.gaussian_filter(gw, sig_px(0.45))
    mid = ((tone > 0.20) & (tone < 0.62)).astype(np.uint8)
    mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, disk(0.25 / px_mm))
    # keep the dot screen away from the edges of solid areas (those stay crisp silhouettes)
    solid = cv2.morphologyEx((tone > 0.66).astype(np.uint8), cv2.MORPH_OPEN, disk(0.6 / px_mm))
    near_solid = cv2.dilate(solid, disk(0.55 / px_mm))
    dens = ndi.gaussian_filter(sharp.astype(np.float32), sig_px(0.35))
    mid = (mid > 0) & (near_solid == 0) & (dens < 0.38)
    mid = cv2.morphologyEx(mid.astype(np.uint8), cv2.MORPH_OPEN, disk(0.2 / px_mm))
    # grow slightly so the dots reach the outlines, then fill pinholes
    mid = cv2.dilate(mid, disk(0.12 / px_mm))
    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    X, Y = xx * px_mm, yy * px_mm
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    U, V = (X * c + Y * s) / pitch, (-X * s + Y * c) / pitch
    cu, cv_ = np.round(U), np.round(V)
    # centre of the nearest dot back in mm
    Xc = (cu * c - cv_ * s) * pitch
    Yc = (cu * s + cv_ * c) * pitch
    dxy = np.hypot(X - Xc, Y - Yc)
    # dot diameter from the tone at the dot centre (area fraction ~ tone), clamped to leave >= min_feat gaps
    ti = np.clip((Yc / px_mm).astype(int), 0, nh - 1), np.clip((Xc / px_mm).astype(int), 0, nw - 1)
    tc = tone[ti]
    dia = np.clip(pitch * np.sqrt(np.clip(tc, 0.05, 1) * tone_curve * 4 / math.pi) * 0.8, 0.0, pitch - min_feat - 0.02)
    dots = (dxy <= dia / 2) & (dia >= min_feat * 1.25) & (mid > 0)
    cu_mask = (sharp & ~(mid > 0)) | dots | (sharp & (mid > 0) & ndi.binary_erosion(sharp, disk(0.05 / px_mm)))
    cu_mask = cu_mask.astype(np.uint8)
    # --- enforce the minimum feature / gap rule
    r = int(round(min_feat / 2 / px_mm))
    cu_mask = cv2.morphologyEx(cu_mask, cv2.MORPH_CLOSE, disk(r - 1))
    cu_mask = cv2.morphologyEx(cu_mask, cv2.MORPH_OPEN, disk(r))
    return cu_mask, px_mm, (W_mm, H_mm)


def to_polys(mask, px_mm, simplify=0.02):
    """binary raster -> list of shapely polygons (with holes), mm, origin top-left"""
    m = np.pad(mask, 2)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    polys = []
    for i, cnt in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        outer = cnt[:, 0, :].astype(float)
        if len(outer) < 4:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            hc = cnts[j][:, 0, :].astype(float)
            if len(hc) >= 4:
                holes.append(((hc - 2 + 0.5) * px_mm).tolist())
            j = hier[j][0]
        p = Polygon(((outer - 2 + 0.5) * px_mm).tolist(), holes)
        if not p.is_valid:
            p = p.buffer(0)
        p = p.simplify(simplify)
        for q in (p.geoms if isinstance(p, MultiPolygon) else [p]):
            if q.area > 0.0 and not q.is_empty:
                polys.append(q)
    return polys


def split_holes(poly, depth=0):
    """cut polygons that have holes along a vertical line through each hole -> hole-free pieces"""
    if not poly.interiors:
        return [poly]
    hole = max(poly.interiors, key=lambda r: Polygon(r).area)
    hx = Polygon(hole).centroid.x
    minx, miny, maxx, maxy = poly.bounds
    parts = split(poly, LineString([(hx, miny - 1), (hx, maxy + 1)]))
    out = []
    for q in parts.geoms:
        if q.area < 1e-6:
            continue
        out.extend(split_holes(q, depth + 1) if q.interiors else [q])
    return out


def hole_free(polys):
    out = []
    for p in polys:
        out.extend(split_holes(p))
    return out


def check_rules(polys, min_feat=0.2):
    u = unary_union(polys)
    eps = min_feat / 2 - 0.004
    narrow = u.difference(u.buffer(-eps).buffer(eps))
    gaps = u.buffer(eps).buffer(-eps).difference(u)
    return dict(area_mm2=u.area, bounds=u.bounds, narrow_residual_mm2=narrow.area, gap_residual_mm2=gaps.area,
                n_polygons=len(polys), n_vertices=int(sum(len(p.exterior.coords) for p in polys)))


def kicad_mod(name, polys, layers, descr, w_mm, h_mm, attr="board_only exclude_from_pos_files exclude_from_bom"):
    """footprint text; polygons in mm, centred on the box, y down (KiCad)."""
    L = []
    L.append(f'(footprint "{name}"')
    L.append('\t(version 20241229)\n\t(generator "dlnn_make_art")\n\t(generator_version "9.0")')
    L.append('\t(layer "F.Cu")')
    L.append(f'\t(descr "{descr}")')
    L.append(f'\t(tags "art logo")')
    L.append(f'\t(attr {attr})')
    L.append(f'\t(property "Reference" "ART**" (at 0 {-h_mm/2 - 2:.3f} 0) (layer "F.Fab") (hide yes) (effects (font (size 1 1) (thickness 0.15))))')
    L.append(f'\t(property "Value" "{name}" (at 0 {h_mm/2 + 2:.3f} 0) (layer "F.Fab") (hide yes) (effects (font (size 1 1) (thickness 0.15))))')
    for layer in layers:
        for p in polys:
            pts = list(p.exterior.coords)[:-1]
            if p.exterior.is_ccw is False:
                pass
            s = " ".join(f"(xy {x - w_mm/2:.3f} {y - h_mm/2:.3f})" for x, y in pts)
            L.append(f'\t(fp_poly (pts {s}) (stroke (width 0) (type solid)) (fill yes) (layer "{layer}"))')
    L.append(")")
    return "\n".join(L) + "\n"


def svg(polys, w_mm, h_mm, bg="#1b3a2a", fg="#d4a74a"):
    ps = []
    for p in polys:
        ps.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in list(p.exterior.coords)[:-1]) + " Z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w_mm:.3f} {h_mm:.3f}" width="{w_mm*8:.0f}" height="{h_mm*8:.0f}">'
            f'<rect width="100%" height="100%" fill="{bg}"/><path d="{" ".join(ps)}" fill="{fg}" stroke="{fg}" stroke-width="0.004"/></svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--box", nargs=2, type=float, required=True, metavar=("W", "H"), help="art bounding box in mm")
    ap.add_argument("--name", default="carrier")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--px", type=float, default=0.02)
    ap.add_argument("--pitch", type=float, default=0.60)
    ap.add_argument("--tone-curve", type=float, default=1.0)
    ap.add_argument("--web", default=None, help="also write the clean SVG, geometry JSON, processed PNG and before/after preview here")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mask, px, (w_mm, h_mm) = process(a.src, a.box[0], a.box[1], a.px, a.pitch, tone_curve=a.tone_curve)
    Image.fromarray((1 - mask) * 255).save(out / f"art_{a.name}_processed.png")
    polys_h = to_polys(mask, px)
    polys = hole_free(polys_h)
    stats = check_rules(polys)
    stats.update(box_mm=[w_mm, h_mm], px_mm=px, dot_pitch_mm=a.pitch, min_feature_mm=0.2)
    (out / f"art_{a.name}_stats.json").write_text(json.dumps(stats, indent=1))
    (out / f"art_{a.name}.svg").write_text(svg(polys, w_mm, h_mm))
    (out / f"art_{a.name}_gold.kicad_mod").write_text(kicad_mod(
        f"ART_{a.name}_gold", polys, ["F.Cu", "F.Mask"],
        "Gold art variant B: net-less copper island + solder-mask opening (exposed ENIG); place flipped on the bottom side", w_mm, h_mm))
    (out / f"art_{a.name}_silk.kicad_mod").write_text(kicad_mod(
        f"ART_{a.name}_silk", polys, ["F.SilkS"], "Silkscreen art variant A (fallback); place flipped on the bottom side", w_mm, h_mm))
    if a.web:
        web = Path(a.web)
        web.mkdir(parents=True, exist_ok=True)
        # clean standalone SVG (outer rings + holes, even-odd), gold on transparent
        d = []
        for p in polys_h:
            for ring in [p.exterior, *p.interiors]:
                pts = list(ring.coords)[:-1]
                d.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in pts) + "Z")
        (web / f"art_{a.name}_clean.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w_mm:.3f} {h_mm:.3f}" width="{w_mm:.2f}mm" height="{h_mm:.2f}mm">'
            f'<title>board art, gold variant B shapes (mm); reading orientation as seen from the bottom of the board</title>'
            f'<path fill="#d4a74a" fill-rule="evenodd" d="{" ".join(d)}"/></svg>')
        geo = dict(schema="dlnn-board-art/1", note="hole-free polygons in mm, origin top-left of the art box, y down, reading orientation as seen from the BOTTOM of the board; "
                   "the same polygons are the B.Cu copper island (no net) and the B.Mask opening (exposed ENIG); apply PBR gold: metalness 1, roughness 0.25, colour #d4a74a",
                   box_mm=[w_mm, h_mm], layers=["B.Cu", "B.Mask"], silk_fallback_layer="B.SilkS", stats=stats,
                   pbr=dict(color="#d4a74a", metalness=1.0, roughness=0.25),
                   polygons=[[[round(x, 3), round(y, 3)] for x, y in list(p.exterior.coords)[:-1]] for p in polys])
        (web / f"art_{a.name}_geometry.json").write_text(json.dumps(geo, separators=(",", ":")))
        Image.fromarray((1 - mask) * 255).save(web / f"art_{a.name}_processed.png", optimize=True)
        src_im = Image.open(a.src).convert("L")
        proc = Image.fromarray((1 - mask) * 255)
        hh = 900
        a1 = src_im.resize((int(src_im.width * hh / src_im.height), hh), Image.LANCZOS)
        a2 = proc.resize((int(proc.width * hh / proc.height), hh), Image.LANCZOS)
        ba = Image.new("L", (a1.width + a2.width + 30, hh), 255)
        ba.paste(a1, (0, 0)); ba.paste(a2, (a1.width + 30, 0))
        ba.save(web / f"art_{a.name}_before_after.png", optimize=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
