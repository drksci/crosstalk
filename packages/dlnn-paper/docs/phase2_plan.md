# DLNN Phase 2 plan: buildable board, live browser arena, verifiable notebooks, Hackaday article

Architecture by Fable (planning pass, 2026-11-03); amendments A1-A6 merged by the orchestrator from Blake's later direction. **Amendments override the body.**

## Amendments (Blake's direction, override anything below)

- **A1 Live arena is the centrepiece.** A small DLNN, sized for 60 fps on a mid-range phone, runs fully client-side next to one or more competitors on the same continuously cycling live input. Candidates: a conventional MLP, a small 1-D CNN, a random delay reservoir with a trained readout (ties into the lineage), and a classical cumulant/feature classifier. Each shows live accuracy, its per-step compute (ops, a delay-line energy estimate and digital energy per classification, labelled estimates with cited sources) and latency.
- **A2 Engaging user input, not just dropdowns.** Pick 2-3 great interactions:
  - keyboard/spacebar keying of OOK/FSK, Morse-style
  - microphone whistle/hum via Web Audio, mapped onto a modulation
  - drawing an I/Q trajectory with mouse or finger
  - live noise/CFO/tolerance knobs
  - a "fool the network" challenge
  - a step-through mode showing each tap's contribution
- **A3 The small nets also live in the notebooks.** Training, export, widget data generation (weights, cycling streams, expected outputs) and a check that the JS forward passes match Python (run headless) all happen in a notebook. The small net is the first point on the scaling curves.
- **A4 The scale story comes from Jupyter.** The at-scale notebooks answer:
  - how many tiles/boards, inductors, area, BOM cost, power and latency
  - accuracy and energy/latency per classification versus the conventional ANN on an MCU/DSP
  - scaling with width, depth and taps
  - tolerance yield

  These feed the article's outcome charts and the verify notebook. There is one pipeline: notebook -> `docs/artifacts/data/` -> widgets + verify notebook.
- **A5 Verifiable public notebooks.** `00_verify.ipynb` prints PASS/FAIL per claim from `claims.json`; Binder/Colab runtimes; CI; a claim-to-evidence map (already in M6, kept).
- **A6 Prior art.** Never claim "first". Close prior work: pour2025 (ISCAS 2025) and merton2026 (Virginia Tech thesis) on RF recurrent nets with tapped transmission-line delay, woo2024 (ternary modulation recognition, digital), tank1987. Position the project as a reproducible, four-fidelity, open-hardware build with an explicit prior-art comparison. Our synthetic-set accuracy is not comparable to RadioML. Literature: `docs/literature.md`, `docs/references.bib`, `docs/artifacts/data/literature.json` (70 verified refs, 48 timeline events).
- **A7 Visuals are data-driven JS (Blake).** Every article visual is rendered live in the browser from exported JSON/GLB/SVG data:
  - D3 for charts and diagrams.
  - three.js for the boards, including the gold PBR material on the art.
  - p5.js where an animated or generative sketch fits better (signal flow, the mercury delay-line hook, the gold art reveal).
  - No static matplotlib/KiCad PNGs in the article; those are notebook-only. A static fallback is allowed only for no-JS readers.
  - Scope: a research article and prototype. Keep reviews minimal; aim for good enough and honestly labelled.
- **Model allocation:** Sonnet workers implement; Opus orchestrates and runs the independent reviews; Haiku does cheap mechanical checks; Fable only for genuinely hard design calls.
- **Defaults taken for Fable's questions:** licences MIT / CERN-OHL-P-2.0 / CC BY 4.0; JLCPCB 4-layer with JLC parts; accept d = 1 TDL-Net-B and the 3-tile width-8 demonstrator as "the board"; LLM moved to an appendix; RadioML only in the full path, not redistributed; public repo layout prepared locally (name `dlnn-paper`), with no push; article 6-8k words; no fabrication budget ("designed, not built").
- **Output location:** everything goes under `docs/artifacts/` and `docs/*.md` for a later publishing agent. The public-repo tree is built locally under `/home/user/dlnn-paper/` and never pushed.

## 0. Verified vs assumed (Fable's probes)

| Item | Status |
|---|---|
| apt kicad = 7.0.11; KiCad 9.0 PPA reachable (install not attempted) | verified |
| kicad-cli 9: sch erc, sch export netlist --format spice/kicadsexpr, svg/pdf/bom; pcb drc --exit-code-violations; pcb export gerbers/drill/pos/svg/step/glb/vrml; pcb render | verified from the 9.0 manual |
| ngspice 42, Java 21, Node 22.23, Python 3.13 venv (+ system 3.12 for pcbnew) | verified |
| pip: skidl 3.0.0, kicad-sch-api 0.5.6, kicad-skip 0.2.5, kinet2pcb 1.1.4, kiutils 1.4.8, kikit 1.8.1, PySpice 1.5, nbmake 1.5.5, papermill 2.7.0, onnxruntime 1.31.1 | verified (index) |
| npm: d3 7.9.1 (ISC), three 0.187.0 (MIT), eecircuit-engine 1.9.0 (MIT, 20.2 MB ngspice-wasm bundle), onnxruntime-web 1.31.1, @observablehq/plot 0.6.17 | verified |
| KiCanvas (MIT, renders .kicad_sch/.kicad_pcb in browser) | verified licence |
| freerouting jar | needs the freerouting/freerouting repo attached (orchestrator attaches) |
| LC ladder 8 lines x 56 sections (120 uH + 3 ohm, 120 pF), 130 us .tran | 0.77 s in ngspice; tap spacing 0.963-0.970 us; tap-7 -2.4 dB incl. ringing; 20 % undershoot with 50 ns edges |
| JLCPCB part pages fetchable; LCSC API / Digi-Key / Mouser / Nexperia / ST blocked; TI and Murata reachable | verified |
| RadioML 2016.10a/2018.01a CC BY-NC-SA 4.0 with known errata | verified |
| Binder base Ubuntu 22.04: ngspice 36, iverilog 11, KiCad 6 (no kicad-cli) | assumed |
| MN3007-class BBDs: boutique clones only; no tapped BBD exists | verified |

## 1. Physical redesign: LC-ladder tiles at about 1 MBd

**Chosen:** lumped LC-ladder delay lines at about 1 MBd; burned-in ternary weights as a 0402 resistor crossbar; BJT diff-pair tanh; S/H re-timing; one PCB design ("tile") for every layer; NUCLEO-G474RE controller on a small carrier. Dilation is dropped (d = 1 everywhere) and the network is retrained to the board.

Rejected options:
- **PCB stripline:** 1 m of trace per line at T_s = 1 ns.
- **BBD / S/H chains:** no taps, no JLC parts.
- **Op-amp all-pass:** 15-25 op-amps per line.
- **SAW:** bandpass, custom, no taps.
- **Single-loop time-multiplexed:** the taps stop being physical. Kept as a sidebar.

### 1.1 TILE-8

| Parameter | Value |
|---|---|
| T_s | 0.96 us nominal (8 x 120 ns), MCU-calibrated to the measured line (monitor pin on line 0, tap 7); 1 MS/s; modrec at 8 samples/symbol -> 125 kBd |
| Delay line | 56 constant-k sections, L 120 uH +/-10 % (shielded, DCR <= 8 ohm, SRF >= 5 MHz), C 120 pF C0G +/-5 % 0603, Z0 1 kohm, source- and end-terminated; group delay +8 % at 1 MHz; 20 % ringing at tap 7 with 50 ns edges -> 1.5 MHz input RC, and train on the real response |
| Delay tolerance | systematic +/-5.6 % (removed by clock calibration); random about 27 ns at tap 7 (2.8 % of T_s) |
| Lines / taps | 8 lines (layer 1: 2 populated); K = 8 taps at sections 0, 8, ..., 56, each buffered (TLV9064-class quad, 16 per tile) |
| Weights | crossbar, two 0402 10 kohm footprints per (line, tap, output) to the +/- summing nodes; W = +1 / -1 / 0 = populate + / populate - / DNP. **The CPL file is the trained model.** 1024 footprints, <= 512 populated; "0" isolation about -50 dB |
| Programmable alternative | 74HC4066 + 74HC595: 320 ICs per tile, -30 dB isolation; rejected, shown as a cost table |
| Summer | inverting differential summer; R_f sets V_sat per layer (1.6-3.6 kohm); noise about 10 uV rms vs 52 mV V_sat |
| Bias | 2 x MCP4728 12-bit I2C DAC per tile into the summing nodes, also trimming V_BE mismatch |
| tanh | BCM847BS matched dual NPN (fallback BC847BS + DAC trim), I_EE about 1 mA, 1 kohm loads, 5 V, 2 V_T = 52 mV |
| Output | difference amp (x0.1) -> track-and-hold (74HC4066 + 1 nF + buffer) at the symbol instant -> ZOH PAM at 2.5 V +/- 0.5 V into the next tile |
| Pool / readout | last tile's 8 outputs on the Nucleo ADCs at 1 MS/s (DMA) -> digital mean + readout + softmax |
| Input | Nucleo DAC1 ch1/ch2 (I/Q) at 1 MS/s, 1.5 MHz RC |
| Interconnect | 2x13 0.1" header / ribbon <= 10 cm: 8 signals interleaved with GND, 8 partial-sum pairs, +5 VA, +3.3 V, I2C, S/H clock, sync, tile ID, monitor |
| Power | 5 V analog LDO on the carrier; about 100 mA per tile |
| Stack-up | 4-layer (sig/GND/5V/sig), JLC04161H-7628, for crossbar routing and continuous ground under the inductor rows (crosstalk target -40 dB: 3 mm pitch, alternating orientation) |
| Size / count | about 160 x 100 mm; about 1,500 placements per full tile |
| Cost (estimate; a live JLC quote is needed) | tile about $150-250 assembled; layer-1 tile about $100; carrier + Nucleo about $50 -> width-8 net about $600-800 |

**TDL-Net-B:** 3 layers, K = 8, d = 1, width 8 (2->8->8->8), receptive field 22, global mean pool, 8-class digital readout, 1,152 ternary weights = 3 tiles. Dilation is not realisable on a fixed-spacing tile.

**Scaling wall:** width scales as (W/8)^2 tiles per layer via the partial-sum ports. Width 32 = 48 tiles, about 21,500 inductors, about $8-10k, so the wide net is simulation-only.

**Scaled virtual model:** width 32, 4 layers, d = 1, K = 8, on the synthetic set. RadioML 2016.10a is optional, full path only, never redistributed.

### 1.2 Tiers re-pointed at the board
- **V:** the same math on the new models.
- **E:** 8-bit fixed point + RTL, and explicitly the digital twin, with an MCU-cycle table. 1.15 GMAC/s exceeds what a 170 MHz M4 can do in real time.
- **C:** ngspice of the KiCad-exported tile netlist: .ac per tap, .dc diff pair, full-tile .tran on >= 64 examples per layer (no extrapolated layers), Monte Carlo on L and C, temperature; a calibrated behavioural model for the 4000-example set. Models:
  - TI TLV906x macro
  - inductor = L + DCR + C_p + R_p, datasheet-derived
  - generic NPN fitted to the BC847 datasheet
  - 4066 = R_on + C_off
- **R -> P (parasitics and tolerance):** a scikit-rf ABCD cascade of the real sections with parasitics, mutual inductance, ribbon/connector, and a 1,000-trial Monte Carlo (L, C, R, V_BE, clock) -> accuracy mean +/- sigma and yield. State that at 1 MHz this is not RF. The coax path is kept only for a "coax study" appendix.

## 2. EDA pipeline (headless)
1. KiCad 9 from the PPA. Fallback: KiCad 7 (no glb / pcb render; use VRML).
2. Design as code: `hw/board/gen_tile.py` uses kicad-sch-api to write a hierarchical .kicad_sch (fallback: raw S-expressions with labels). This schematic is the single source of truth -> ERC, kicadsexpr netlist (PCB), spice netlist (C tier), svg/pdf/bom. Layer variants and the weight CPL are generated from model.npz.
3. PCB: kinet2pcb -> pcbnew Python placement of the regular structures -> Specctra DSN -> freerouting -> SES. Fallback: scripted routing.
4. Outputs: DRC, gerbers, drill, pos, JLC BOM/CPL, per-layer SVG, glb, step, render PNGs. The carrier board goes through the same flow.
5. Human review before ordering. Claim only "designed, ERC/DRC-clean, not fabricated, not measured".

## 3. Article and widgets (static, vendored, offline, `docs/artifacts/`)
- **Format:** `index.html` + `assets/` (d3 7.9.1, three 0.187.0 + GLTFLoader, KiCanvas, eecircuit-engine lazy-loaded) + `data/`.
- **Constraints:** dark/light themes; mobile fallbacks; < 3 MB initial load; 60 fps.

Widgets:
- **W0 Arena (A1/A2):** the live duel/arena with user input. This is the centrepiece.
- **W1:** network/tap diagram.
- **W2:** live inference (hand-written ternary JS forward pass).
- **W3:** real ngspice-wasm unit cell.
- **W4:** V/E/C/P tier viewers.
- **W5:** tolerance/noise sliders from a precomputed grid.
- **W6:** board (KiCanvas, SVG layers, three.js GLB).
- **W7:** stitching and scaling counters.
- **W8:** lineage timeline from `literature.json`.
- **W9:** scale-story outcome charts (A4).
- **W10:** claims map.

Data exports (each with schema + run id + sha256):
- `model_*.json` (int8 W base64 [O,I,K], dilation, gain, bias, vsat, readout, meta)
- `arena_models.json`
- `examples.bin/.json` (N = 40, per-layer V traces, probs per tier)
- `tiers.json`, `sweeps_grid.json`
- `spice/*.cir` + `tap_response.json` + `diffpair_dc.json`
- `board/` (sch, pcb, glb, SVGs, gerbers zip, bom, cpl, erc, drc, cost)
- `stitching.json`, `scale_story.json`, `literature.json`
- **`claims.json`** (id, text, value, tolerance, source file/key, notebook cell tag, pytest id)

CI fails if any number in the page is not in `claims.json`.

## 4. Article content (6-8k words, about 25 min read)
1. Hook: a tile render, and "the weights are the pick-and-place file".
2. The idea (W1, W0).
3. Lineage (W8).
4. Maths.
5. From coax to board: phase-1 coax study, the option table, the scaling wall.
6. The tile (W6).
7. The four fidelities (W4).
8. Results and sweeps (W5).
9. Live demo (W2, W3).
10. Scaling and stitching (W7, W9).
11. What doesn't work: the d = 1 cost, ringing, QPSK/8PSK, tolerance yield, the LLM finding.
12. BOM and cost.
13. Reproduce: Binder/Colab, the verify notebook, the claims map.
14. References.

Put the honesty box at the top. Provide 12 static figures plus the widgets.

## 5. LLM
The LLM moves to an appendix and out of the default pipeline: it restates, it doesn't discover. Optional LLM-drafted alt text is human-reviewed and labelled.

## 6. Work breakdown

| M | Scope | Acceptance | Effort |
|---|---|---|---|
| P0 | feasibility spike: KiCad 9; kicad-sch-api -> ERC -> spice/kicadsexpr -> ngspice; kinet2pcb -> place -> freerouting -> DRC -> gerbers/glb/render; eecircuit-engine unit cell in headless Chromium; JS forward pass vs numpy | all checks pass or a documented fallback; JS = numpy <= 1e-5 on 40 examples | 4 h |
| M1a | hardware spec `docs/board_design.md`, `hw/board/params.json` (derived in a script), parts with C-numbers, cost model | ringing <= 10 % at tap 7 with the input filter; loss <= 3 dB; isolation <= -45 dB | 6 h |
| M1b | retraining d = 1 (w8/w16/v32 4-layer, 3 seeds) + arena models (A1) | **gate: w8-d1 ternary >= 0.70 mixed and >= 0.75 at >= 10 dB**; else fallback A (d = 2 variant, cost x2) or B (layer-1 physical only) | 6 h |
| M2a | C tier on the KiCad netlist | calibrated vs SPICE NMSE <= 0.02 per layer; >= 64 SPICE examples per layer; C - V <= 5 points nominal | 8 h |
| M2b | P tier + grid + web exporters | 1,000-trial MC <= 10 min; grid >= 5^5; P - V <= 3 points nominal | 8 h |
| M3a | schematic generators + ERC | ERC 0 errors; part count matches params.json | 8 h |
| M3b | PCB + fab outputs | DRC 0 errors, 0 unrouted; CPL weight count = nonzero W | 10 h |
| M4a | page core, W0 arena, W1, W2, W4 | < 3 MB initial; JS matches examples <= 1e-4; mobile Lighthouse >= 90 | 10 h |
| M4b | W3, W5 | unit cell <= 3 s; slider values within 0.5 pt of the grid | 6 h |
| M4c | W6, W7, W8, W9 | GLB <= 15 MB; KiCanvas opens both files; timeline renders | 6 h |
| M5 | article text + static figures + claims ids | every number resolves in claims.json | 8 h |
| M6 | public notebook repo (`/home/user/dlnn-paper/`, local only) | fast path <= 10 min, <= 2 GB, no torch/LLM, PASS on every claim; full path <= 2 h | 10 h |
| M7 | integration + independent Opus review | 0 unresolved high findings | 6 h |

Critical path: P0 -> M1b gate -> M3a -> M3b -> M4c -> M7.

## 7. Risks, cut lines, must-not-claim

Cut lines:
- KiCad 9 fails -> KiCad 7.
- No freerouting -> scripted routing and a 4x4 tile.
- d = 1 misses the gate -> fallback A/B.
- eecircuit-engine chokes on the op-amp macros -> passive line + level-1 BJT only.
- Binder RAM -> stored results re-checked.
- Page too heavy -> W3 becomes a SPICE replay.

**Must NOT claim:**
- any measurement
- that the board works
- vendor-accurate models (the TI macro yes; BJT and inductors are datasheet-derived)
- costs beyond "estimate pending quote"
- "analog beats digital" (only the MAC-rate observation, with caveats)
- scaling beyond the (W/8)^2 arithmetic
- that the LLM analysed anything
- RadioML as more than simulation on a dataset with known errata
- d = 1 accuracy equal to phase 1, unless measured
- "first"
