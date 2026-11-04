# PCB layout (M3b): TILE-8 and carrier

Status: **designed, ERC/DRC-clean where stated, not fabricated, nothing measured.** Renders and GLBs are renders of the design, not photos.
Crossbar model: `models/tdlnet_b_w8_aware2_fro_ternary.npz` (sha256 `3405923d2c64...`), ternary taps in every layer; the float digital readout is MCU firmware data
(`docs/artifacts/board/firmware/readout.json`), not on the PCB. Artwork: supplied by the project owner, used under fair use per owner.

## Results

| Board | State | DRC (`kicad-cli pcb drc --exit-code-violations`, all severities) | Unrouted | Notes |
|---|---|---|---|---|
| carrier (134 x 90 mm, 4 layer) with gold art | **routed** | exit 0, 0 violations | 0 | 50 SMT parts in the CPL, 5 hand-fit THT parts (2 sockets, jack, 2 IDC headers) |
| carrier, silkscreen-art fallback | routed | exit 0, 0 violations | 0 | `hw/board/pcb/carrier/carrier_art_silk.kicad_pcb` (generated only) |
| **TILE-8 layer 1, reduced** (180 x 142 mm, 4 layer) | **routed** | exit 0, 0 violations | 0 | **cut line, see below**: 2 delay lines (of 8), full 8-output crossbar slice, 85 populated crossbar resistors = the model's 85 non-zero weights |
| TILE-8 layer 2, full 8 lines | **placed only** | not routed | see `drc.rpt` | placed reference: 448 inductors, 315 populated crossbar resistors, geometry DRC-clean (no clearance/courtyard/hole errors), only ratsnest left |
| TILE-8 layer 3, full 8 lines | placed only | not routed | see `drc.rpt` | 316 populated crossbar resistors |

ERC (`lib/build_m3a.sh`): 0 errors, 0 warnings on all tile variants and the carrier.
"The CPL is the model" check (`make_fab.py`, `cpl_is_model` in each `fab_summary.json`): the populated `RXP`/`RXN` refdes in the JLC CPL equal the model's non-zero weights
(`RXP` = W +1, `RXN` = W -1): layer 1 85/85, layer 2 315/315, layer 3 316/316, all `match: true`; DNP crossbar parts are excluded from the CPL, BOM and stencil paste.

## Cut line (what was cut and why)
The full 8-line tile has 2,200 placements, about 4,800 SMD pins and 660 connections left after the scripted routing. Routing the 2-line slice (700 parts, 448 connections left)
with freerouting took 22 passes, about 18 minutes on 4 cores with the optimizer off (the optimizer ran out of Java heap and lost a whole run). Following the scope note
("take the cut line early"), the routed tile deliverable is the **layer-1 slice**: lines 0 and 1 only (112 inductors, 16 taps, 8 outputs, the full 64 x 16 crossbar footprint grid cut
to 16 x 16, 2 input buffers' worth of RC), the same footprints, rules and generator as the full tile. Layers 2 and 3 are generated, placed and checked (geometry DRC, CPL = model) but not routed.
A routing attempt on the full layer-2 tile was started (same flow) and stopped after 17 minutes inside freerouting's fanout stage (4,833 SMD pins, 3,547 needing fanout, no autoroute pass completed); the layer-1 run needed 77 s for the fanout and 18 minutes in total, so the full tile is expected to need hours. Not retried: a routing run with a longer budget, or a split into two half-tiles routed separately, is the obvious next step.

## Placement strategy (`hw/board/make_pcb_tile.py`, all scripted)
* Footprints from the stock KiCad 9 libraries, nets from the kicadsexpr netlist of the M3a schematic, created directly with pcbnew (kinet2pcb needs >2 min for 700 parts and is not used;
  P0's kinet2pcb wrapper remains the documented alternative). Stackup JLC04161H-7628 injected as text (ENIG, black mask, white silk).
* Board 180 x 142 mm, **not** 160 x 100: 448 1812 inductors (courtyard 5.99 x 3.99 mm) with their 0603 capacitors in line need about 14,000 mm2 of one-sided area; the 64 x 16 crossbar
  needs a window with both sides free for its vias. M1a's 5 mm row pitch with alternating row orientation is not possible with an in-line chain (the inductor axis must follow the chain).
* Ladder block per line: 7 rows x 8 sections (one row = one tap interval, so taps sit at row ends), cell 7.7 mm (L + C in line), row pitch 4.2 mm, rows alternate direction (boustrophedon).
  Lines 0-3 on the top side (2 x 2 blocks, GND plane In1 directly below, inductor axes along X); lines 4-7 on the bottom side under the top rows, rows along Y, i.e. **inductor axes orthogonal to the
  top ones**, gutters outside the top blocks, a GND island on In2 under their rows. Tap buffer quads sit in the block end gutters: quad A = even taps (ladder nodes 0, 16, 32, 48), quad B = odd taps (8, 24, 40, 56) (schematic regrouped for this).
* Crossbar matrix: 64 columns (tap nets) x 16 rows (SN_o / SP_o), 0402 at 1.3 x 2.6 mm pitch, resistor axis along Y, pad 2 on the F.Cu row track, pad 1 -> via -> B.Cu column bus. Placed between the tap buffers
  (north) and the channel strip (east); DNP footprints keep their pads and copper (so the model is the CPL, not the copper) but have no paste aperture.
* Channels: 8 summers / BJT pairs / diff-amps / track-and-hold in two shelf-packed groups; shared quads sit with their lowest channel. Headers at the board ends (J_IN west, J_OUT east), J_PS (DNP) below the channels,
  4 mounting holes, 6 fiducials, silk title with the layer variant and the model sha.

## Routing method and results
* Scripted (locked, pre-routed): all delay-line L-C chains (55 interior nodes + 7 turn nodes per line), the GND bus tracks between rows (stitched to In1 in the gutters), the crossbar row tracks, via stubs and column buses.
* Plane-net fanout: one via + stub next to every SMD GND / +5VA pad, placed by a collision-checked script (`gen_pcb.fanout_planes`; layer 1: 144 vias). Needed because freerouting cannot drop vias onto planes once In1/In2
  are typed as power layers; with signal-typed inner layers it routed tracks on In1/In2 and slotted the planes (first runs, discarded).
* freerouting v2.5.1 via Specctra DSN -> SES (`ExportSpecctraDSN` / `ImportSpecctraSES`), Java 25, `-Xmx9g`, optimizer off, up to 40 passes. Carrier: 1 minute. Layer-1 tile: 22 passes, 0 unrouted, 18 minutes.
  Final layer-1 tile: 3,396 track segments, 787 vias, 8.0 m of F.Cu and 4.7 m of B.Cu; carrier (before the art): 449 segments, 81 vias.
* Post-processing: dangling scripted bus stubs trimmed (`gen_pcb.trim_dangling`); `track_dangling` is ignored in the tile's DRC rules because KiCad does not treat DNP pads as connection points and the crossbar row tracks end on them (all other checks on).
* Rules (JLC 4-layer capable): clearance 0.15, track >= 0.09 (0.15 default), via 0.5/0.25, hole to hole 0.25, edge clearance 0.3. Silk checks are ignored in the project file (references live on F.Fab).

## Layer usage
F.Cu: top ladders (lines 0-3), crossbar pads and row tracks, channel parts, routing. In1 = GND plane (continuous under the top rows). In2 = +5VA plane with GND islands under the bottom-side ladder rows. B.Cu: bottom ladders (lines 4-7),
crossbar column buses, routing. Mask black, finish ENIG.

## Crosstalk and layout decisions
* Row pitch 4.2 mm with parallel axes inside one line's block: dipole estimate about -40 dB (docs/board_design.md method, V_eff assumed), not demonstrated; different lines are separated by 9.5 mm gutters, top/bottom lines by orthogonal axes and the GND plane.
* The 3 mm pitch of M1a does not fit the 1812 part; no shielded alternative fits either (see board_design.md).
* Tap-lead length: up to about 15 mm of 0.15 mm trace per tap (about 1.5 pF at 0.1 pF/mm, estimate) against a 6 pF assumed buffer load: not simulated here.

## Fab outputs (`docs/artifacts/board/`)
`{carrier,tile_L1,tile_L2,tile_L3}/`: gerbers + drill zip (routed boards only), JLC BOM and CPL, kicad pos, per-layer SVGs (F/B/In1/In2 copper, mask, paste, silk, edge cuts, drill), `drc.json/.rpt`, `erc.json/.rpt`,
`fab_summary.json`, a sanity PNG per side, GLB (carrier: full, 16 MB, with the gold art island; tiles: lite ~3-9 MB with body, mask, silk, inductors, ICs and connectors; `make_fab.py --full-glb` gives 40-50 MB),
STEP (routed boards). `materials.json` (PBR values per GLB material role), `board_manifest.json`, `firmware/readout.json`, `art/` (clean SVG, geometry JSON, footprints, processed bitmap, before/after).
Regenerate: `hw/board/lib/build_m3a.sh`, `hw/board/make_pcb.sh carrier | tile N`, `make_pcb_carrier.py art`, `make_fab.py`, `make_web_manifest.py`. The EDA venv gained `shapely scikit-image scipy opencv-python-headless` for the art.

## Gold art (variant B, default) and the silkscreen fallback
`hw/board/art/make_art.py`: despeckle (blur 0.11 mm + threshold) to crisp solid shapes, vein gaps on the arm (>= 0.2 mm), a deliberate 45 degree dot screen (pitch 0.6 mm, dot 0.25-0.40 mm) on the mid-tone machine-side bars,
close/open with 0.2 mm discs (no copper feature or gap below 0.2 mm; residual narrow/gap area 1.7 / 1.3 mm2 of 1,198 mm2 at the 0.02 mm simplification), hole-free polygons (the file format cannot hold holes). Variant B: `F.Cu` fill + `F.Mask` opening in a footprint
placed flipped on the bottom (B.Cu + B.Mask, exposed ENIG), no net (island >= 0.5 mm from all other copper, B.Cu keep-out around it, nothing else on that part of B.Cu), 52.8 x 76 mm between the Nucleo sockets. Variant A: B.SilkS, same polygons, kept as a generated fallback.
DRC on the carrier with the art: 0 violations; the mask-bridge check is ignored in the art file only (it reports every seam between hole-free pieces of one aperture, 199 false positives) and enabled for the board without the art.
ENIG is required for the gold look (HASL would be silver); cost impact is an estimate in `hw/board/cost.json` (`M3b_notes`).

## Human review before ordering
1. Nucleo morpho geometry: CN7/CN10 outer-to-outer 66.04 mm and pin 1 at the top-left column of each header were taken from the Nucleo-64 pitch, not from a measured board (UM2505 drawing is only partly legible). Check against a real NUCLEO-G474RE; the Morpho socket C22373924 footprint match.
2. Courtyards and 3D: stock footprints; inductor 1812 pads and the JLC rotations of the CPL (KiCad rotation, per-part JLC offsets for the SOT-363, TSSOP, MSOP, SOIC parts and bottom-side parts are not applied): compare against JLC's preview. Orientation of the 74HCT4066, TCA9548A, MCP4728 pin 1.
3. Inductor orientation and the crosstalk estimate (parallel axes inside a line at 4.2 mm); a field simulation or a test coupon is needed before any claim.
4. Extended parts: all 15+ extended part types (inductor, TLV9064, BCM847BS (stock 4 at the reading), MCP4728, 74HCT4066, TCA9548A, LM4040, E96 values including 931 ohm (12 in stock) and 8.87 kohm (1 in stock)): live quote and stock check; the DC barrel jack has no verified LCSC number.
5. Bias path: the MCP4728-at-3.3-V plus RO offset network and R_b changes are checked by ERC only, not simulated; set `nSWBOOT0 = 0` on the Nucleo; I2C addresses 0x70 + tile ID.
6. THT soldering of the full-connection GND plane pads (no thermal relief on the plane pours: easier DRC, harder hand soldering), ESR of the AMS1117 output capacitor, 7 V supply preferred.
7. A live JLCPCB quote (4 layer, 180 x 142 mm, ENIG, black mask) and a DFM check; board size changed from the 160 x 100 mm cost assumption.
8. DNP crossbar footprints keep unpopulated pads without paste: confirm JLC's CPL/BOM handling and that the stencil has no apertures there.

## ECO-1 (board-mode C tier findings)
**Why.** The ngspice C tier on the exported netlists (`docs/circuit_extraction.md`, board mode, findings 1 and 2) found (1) line terminations `RT` to GND put the taps at about 1.25 V against the 2.5 V summer reference (one summer output 2.4 V off, outside the DAC trim range) and (2) no compensation of the line loss (symbol-sampled on-lag gain 0.33 mean).
**What changed** (`gen_tile.py`, `params.json` untouched):
* `RT{i}` now goes to `VREF25`. 2-line tile (L1): the V_cm follower (spare unit of UIB0) carries 2 x 1.25 mA. 8-line tiles (L2/L3): worst case 10 mA, so a dedicated follower (spare unit 2 of UVR1, `CB5` 10 uF) drives net `VREF25T` for the 8 `RT`; the summers' reference stays separate. (TLV9064 output-current limits were not checked against the datasheet; the buffer is the conservative choice.)
* `ECO_GAIN = 2.9` multiplies R_f, R_b and R_o (R_o follows R_b), rounded to E96; the bias current is divided by 2.9 so the DAC swing is unchanged. R_f / R_b, layers 1 / 2 / 3: 2.21k / 1.05k / 2.8k and 15k / 9.09k / 22.6k (the C tier's 2.2k / 1.07k / 2.8k and 15.2k / 9.2k / 22.8k were estimates). New LCSC numbers verified with `lib/jlc_lookup.py` (cache updated); 22.6k (C22959) had only 8 in stock.
**Results.** ERC 0 / 0 on all variants and the carrier; crossbar counts unchanged (85 / 315 / 316). L1: updated from the new netlist and fully rerouted (about 15 min, 289 unrouted items at start); first result 0 unconnected, 2 clearance errors of 1.5-2.4 um at JPID2 / RID0, fixed by a scripted nudge of the +3V3 track and the TID0 via (rule not relaxed), zones refilled: **DRC 0 / 0, 0 unconnected**. Carrier untouched (no pin change), DRC 0 / 0. L2 / L3: re-placed only, geometry DRC unchanged (ratsnest and dangling items only); one plane fanout (UIB0.4) failed in the placement script on both. Fab outputs, SVGs, GLBs and `board_manifest.json` regenerated.
**SPICE confirmation** (layer 1, `build_setups(rf_scale=1)` on the new netlist): DC tap gain 0.525 .. 0.456 and DAC trims identical to the earlier ECO-patched run, no clipped channels; on-lag line gain 0.338 x 2.9 = 0.98; calibrated model vs SPICE on 3 examples NMSE 0.0066 / 0.015 / 0.062 (layers 1 / 2 / 3), outputs within 1.0-3.0 V. Simulation of a design that is not built.
