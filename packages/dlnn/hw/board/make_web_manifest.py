#!/usr/bin/env python3
"""docs/artifacts/board/materials.json + board_manifest.json from the make_fab.py summaries (run after make_fab.py for every board)."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "docs" / "artifacts" / "board"
MODEL = REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"
sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
ROLES = {
    "copper_enig_gold": dict(color="#d4a74a", metalness=1.0, roughness=0.25, note="ENIG: all exposed copper (pads, THT rings, the gold art island); GLB from kicad-cli carries a yellow-olive copper material, replace it"),
    "soldermask_black": dict(color="#0b0b0d", metalness=0.0, roughness=0.55, opacity=0.9, note="solder mask, black (stackup colour), semi-opaque so tracks show faintly"),
    "silkscreen_white": dict(color="#f2f2f2", metalness=0.0, roughness=0.85),
    "substrate": dict(color="#4d4a3a", metalness=0.0, roughness=0.8, note="FR-4 edge / board body"),
    "other": dict(color="#4d4d4d", metalness=0.0, roughness=0.8, note="component bodies (KiCad stock 3D models)"),
}
mat, boards = {"schema": "dlnn-board-materials/1", "label": "render of the design (not a photo; board not fabricated)", "roles": ROLES, "glb": {}}, []
mat["art"] = dict(file="art/art_carrier_geometry.json", layers=["B.Cu", "B.Mask"], role="copper_enig_gold", **ROLES["copper_enig_gold"])
ALT = {
    "carrier": "Carrier board for the NUCLEO-G474RE and the tile chain: Morpho sockets, DAC/ADC conditioning, 7-9 V LDO; the bottom side carries a half-human, half-machine figure in exposed gold (ENIG) copper.",
    "tile_L1": "Routed layer-1 TILE-8 slice (2 delay lines of 56 LC sections, 8 taps each, 64 x 16 crossbar slice populated from the model, 8 output channels).",
    "tile_L2": "Full 8-line layer-2 TILE-8 placement (448 inductors, crossbar populated from the model); placed, not routed.",
    "tile_L3": "Full 8-line layer-3 TILE-8 placement; placed, not routed.",
}
for name in ("carrier", "tile_L1", "tile_L2", "tile_L3"):
    d = ART / name
    f = d / "fab_summary.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    gm = s.get("glb_materials")
    if gm:
        mat["glb"][name] = {m["name"]: m["role"] for m in gm}
    files = sorted(str(p.relative_to(ART)) for p in d.rglob("*") if p.is_file() and p.suffix != ".png" or p.name.startswith("sanity"))
    boards.append(dict(board=name, routed=name in ("carrier", "tile_L1"), files=files, drc=s["drc"], erc=s["erc"], cpl_rows=s["cpl_rows"],
                       cpl_is_model=s.get("cpl_is_model"), glb=f"{name}/{name}.glb" if (d / f"{name}.glb").exists() else None,
                       layer_svgs=[f"{name}/svg/{p.name}" for p in sorted((d / "svg").glob("*.svg"))], crossbar_model_sha256=sha if name.startswith("tile") else None,
                       alt=ALT[name] + " Render of the design (not a photo); board not fabricated."))
manifest = dict(schema="dlnn-board-manifest/1", label="designed, ERC/DRC-clean where stated, NOT fabricated, nothing measured", model=str(MODEL.relative_to(REPO)),
                model_file_sha256=sha, firmware_readout="firmware/readout.json", materials="materials.json",
                art=dict(clean_svg="art/art_carrier_clean.svg", geometry="art/art_carrier_geometry.json", footprint_gold="art/art_carrier_gold.kicad_mod",
                         footprint_silk_fallback="art/art_carrier_silk.kicad_mod", processed_bitmap="art/art_carrier_processed.png", before_after="art/art_carrier_before_after.png",
                         note="Artwork: supplied by the project owner, used under fair use per owner."), boards=boards)
(ART / "materials.json").write_text(json.dumps(mat, indent=1))
(ART / "board_manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote materials.json, board_manifest.json:", [b["board"] for b in boards])
