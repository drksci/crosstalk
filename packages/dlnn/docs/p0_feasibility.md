# P0 feasibility spike: results (2026-11-03)

All eight checks PASS. Three needed workarounds (listed under each check and in the fallback table). Everything is reproducible with `./setup_eda.sh` then `hw/p0/build_p0.sh`; the browser and JS checks are listed below.

| # | Check | Result | Fallback used |
|---|---|---|---|
| 1 | KiCad 9 install, `kicad-cli`, `pcbnew` in python3.12 | PASS (9.0.9) | none (KiCad 7 not needed); `add-apt-repository` and library packages needed workarounds |
| 2 | venv + pip, schematic via kicad-sch-api, ERC | PASS (0 errors, 0 warnings) | none |
| 3 | spice + kicadsexpr netlists, ngspice `.tran`, tap delays | PASS | none |
| 4 | PCB via kinet2pcb + pcbnew, 4-layer, freerouting, DRC | PASS (0 violations, 0 unrouted) | kinet2pcb workarounds (2 small patches in a wrapper); stackup injected as text |
| 5 | Fab outputs (gerber, drill, pos, JLC BOM/CPL, SVG, GLB, STEP, renders) | PASS | none |
| 6 | eecircuit-engine in headless Chromium vs ngspice 42 | PASS (max error 5.2e-9 V) | none |
| 7 | JS ternary forward pass vs numpy tier V | PASS (max error 1.9e-6, limit 1e-5) | none |
| 8 | d3 / three (+GLTFLoader) / KiCanvas vendored and rendering | PASS | none |

## Versions

| Tool | Version |
|---|---|
| KiCad (PPA `ppa:kicad/kicad-9.0-releases`) | `kicad-cli version` = **9.0.9** (`9.0.9~ubuntu24.04.1`), pcbnew 9.0.9 |
| OS / python | Ubuntu 24.04.5; system python3.12 (pcbnew); project `.venv` python 3.13 |
| pip in `hw/.venv-eda` | kicad-sch-api 0.5.6, kinet2pcb 1.1.4, kiutils 1.4.8, skidl 3.0.0 |
| ngspice (system) | 42 |
| freerouting | v2.5.1-SNAPSHOT, commit `88480e8` (shallow clone of master), jar `build/libs/freerouting-current-executable.jar` (65 MB) |
| Java | system JDK 21.0.12 cannot run it: freerouting's `build.gradle` targets Java 25 (class file 69). Gradle's foojay toolchain resolver downloaded Temurin 25.0.5 into `~/.gradle/jdks/`; that JDK runs the jar. |
| node / playwright | node 22.23.4, playwright 1.63.0 (preinstalled), chromium headless |
| npm | d3 7.9.1 (ISC), three 0.187.0 (MIT), eecircuit-engine 1.9.0 (MIT), KiCanvas official build (MIT) |

## The demo

`hw/p0/gen_demo.py` writes `hw/p0/demo.kicad_sch` (single source of truth): J1 input header, R1 1 kohm source, L1/L2 120 uH with 3 ohm DCR, C1/C2 120 pF, R2 1 kohm termination, U1/U2 unity-gain tap buffers, U3 inverting summer (R3/R4/R5 10 kohm), +/-5 V rails with 100 nF decoupling, J2 power and J3 output headers. Stock KiCad symbols (`Device`, `Amplifier_Operational:LMV321`, `Connector_Generic`, `power`) and footprints (0603 R/C, 1210 L, SOT-23-5, 2.54 mm headers). SPICE models are behavioural, in `hw/p0/models/lc_parts.lib`: `IND_DCR` (L + series R, params from the symbol's `Sim.Params`) and `OPA_IDEAL` (VCVS, gain 1e5). The headers carry `Sim.Enable=0`. The summer is therefore an ideal-VCVS summer, as allowed for the spike.

## Check 1: KiCad 9

Commands (also in `setup_eda.sh`):
```
/usr/bin/python3.12 /usr/bin/add-apt-repository -y ppa:kicad/kicad-9.0-releases
apt-get update
apt-get install -y --no-install-recommends kicad kicad-symbols kicad-footprints kicad-packages3d
```
- `add-apt-repository` fails under the default `python3` (`No module named apt_pkg`); the system python3.12 works.
- `--no-install-recommends kicad` alone installs no libraries (no `/usr/share/kicad/symbols`); the ERC then warns about every missing library. `kicad-symbols`, `kicad-footprints`, `kicad-packages3d` (needed for GLB/STEP/render) must be named explicitly. Install took about 1 to 2 minutes.
- The global library tables must be copied from `/usr/share/kicad/template/{sym,fp}-lib-table` to `~/.config/kicad/9.0/`; the headless first run only creates an empty footprint table.
- `python3.12 -c "import pcbnew"` works; the venv with `--system-site-packages` sees it too. (Harmless stderr line `assert "PgmOrNull()" failed in register_action()` appears on import.)

## Check 2: schematic and ERC

`hw/.venv-eda/bin/python hw/p0/gen_demo.py` then `kicad-cli sch erc --exit-code-violations -o hw/p0/erc.rpt hw/p0/demo.kicad_sch`: exit 0, **0 errors, 0 warnings** (`hw/p0/erc.rpt`). Notes for M3a:
- kicad-sch-api API: `sch.components.add(lib_id, reference=, value=, position=)` (not `ref=`), `sch.get_component_pin_position(ref, pin)`; component fields via `c.properties[...]` (used for `Sim.*`); positions snap to the 1.27 mm grid.
- Nets are formed by net labels and power symbols placed exactly at pin ends (no wires). Power symbols plus `PWR_FLAG` give a clean ERC; a `VPOS` label next to a `+5V` symbol triggers `multiple_net_names`, so only the power symbol is used.
- The raw S-expression fallback was not needed.

## Check 3: netlists and ngspice

- `kicad-cli sch export netlist --format spice -o demo.cir demo.kicad_sch` (valid ngspice, includes `lc_parts.lib` by absolute path) and `--format kicadsexpr -o demo.net` both succeed.
- `hw/p0/flatten_spice.py` inlines the lib and adds `Vstim /VDRV 0 PULSE(0 1 100n 5n 5n 4u 8u)` and `.tran 2n 3u`; the same `demo_flat.cir` is run by ngspice and by eecircuit-engine. ngspice batch run takes about 13 ms; `wrdata` output `hw/p0/tran_ngspice.txt` (1520 points).
- Tap delays. Step 50 % crossings (edge midpoint at 102.5 ns): N1 at 287.5 ns, N2 at 484.7 ns, so tap spacing **197.2 ns** for the step. This is longer than the 120 ns of a long ladder because the 5 ns step sees the lumped section as a low-pass with ringing (peak 0.567 V at N1; DC gain 0.5). The small-signal phase delay from an AC run at 200 kHz (`hw/p0/demo_ac.cir`: phase -0.1496 and -0.3027 rad) is **119.0 ns (R1+L1+C1)** and **121.8 ns tap to tap**, i.e. about 120 ns per section as designed (sqrt(LC) = 120 ns). M1a should specify the delay metric (group delay vs. step 50 % crossing) explicitly.

## Check 4: PCB and routing

Chain (all in `hw/p0/`, driven by `build_p0.sh`):
1. `netlist_to_pcb.py demo.net demo_raw.kicad_pcb`: kinet2pcb 1.1.4 is used, with two workarounds for KiCad 9.0.9: (a) its parser raises `Sexp isn't in a form that permits extracting a single value` on the new `(property ...)` children of `(comp)`, so they are stripped in memory first; (b) `pcbnew.FootprintLoad()` returns `NoneType` headless, so it is replaced with `PCB_IO_MGR.PluginFind(KICAD_SEXP).FootprintLoad`. Also `KICAD9_FOOTPRINT_DIR=/usr/share/kicad/footprints` must be set. 17 footprints, all nets. (Direct pcbnew-API creation remains the fallback.)
2. `make_pcb.py place`: scripted placement from a dict, 84 x 54 mm outline on Edge.Cuts, 4 copper layers via `SetCopperLayerCount(4)`, default netclass 0.2 mm clearance, 0.25 mm track, 0.6/0.3 mm via, GND zone on In1.Cu, +5V zone on In2.Cu, silk title. The python API does not expose the stackup descriptor, so the JLC04161H-7628 1.6 mm stackup (35/210.4/15.2/1065/15.2/210.4/35 um) is injected as a `(stackup ...)` S-expression into the saved file; KiCad reads and keeps it.
3. `ExportSpecctraDSN` -> freerouting -> `ImportSpecctraSES`:
   `java -jar freerouting-current-executable.jar --gui.enabled=false -de demo.dsn -do demo.ses -mp 30` (with the JDK 25 binary). freerouting log: fanout 37/37 pins (+27 vias), 0 unrouted and 0 violations after pass 1, optimiser 2 passes, job 3.0 s; whole stage 6.4 s wall including JVM start. Result: 77 track segments, 20 vias; it treated the GND and +5V zones as planes. Zones are refilled with `ZONE_FILLER` after the SES import.
4. `kicad-cli pcb drc --exit-code-violations -o drc.rpt demo.kicad_pcb`: exit 0, **0 violations, 0 unrouted** (`hw/p0/drc.rpt`). Negative control: the same DRC on the placed, unrouted board exits 5 with 31 unconnected items, so the check does bite.
- Build issues: Maven Central intermittently answered HTTP 429 through the proxy during the first Gradle runs (plugin resolution); a retry loop in `setup_eda.sh` handles it (it succeeded on the first retry here). `./gradlew assemble` produces a jar without a manifest; `executableJar` is the runnable task.

## Check 5: fab outputs (`hw/p0/fab/`)

- `gerbers/`: 11 layers incl. In1/In2, `demo-PTH.drl`, `demo-NPTH.drl`, drill maps, job file; zipped as `demo_gerbers.zip`.
- `demo_pos.csv` (kicad), `demo_cpl_jlc.csv` (Designator, Mid X, Mid Y, Layer, Rotation; headers excluded), `demo_bom_jlc.csv` (Comment, Designator, Footprint, LCSC Part #; 6 lines) made by `jlc_export.py` with kiutils. Only three C-numbers are mapped in `lcsc_map.json` (100 nF C14663, 1 kohm C21190, 10 kohm C25804, from memory, not re-verified); the rest are blank and M1a must assign and verify real parts.
- `svg/`: F.Cu, In1.Cu, In2.Cu, B.Cu, F.Silkscreen, Edge.Cuts. `demo.glb` (1.1 MB, 1362 meshes, includes the 3D models of the stock footprints), `demo.step` (412 KB).
- `render/top.png`, `bottom.png`, `perspective.png` (`kicad-cli pcb render`, 1600 x 1000, about 1.3 s each, works headless, no xvfb). I opened `top.png` and `perspective.png`: board, outline, silk, footprints with 3D bodies, and routed copper are visible.

## Check 6: eecircuit-engine in the browser

- Vendored with `npm pack eecircuit-engine@1.9.0`: `docs/artifacts/vendor/eecircuit-engine/eecircuit-engine.mjs` (20,346,804 bytes, 5.68 MB gzip, ngspice compiled to wasm and embedded) + `LICENSE` (MIT). The UMD build and the PTM model typings were not copied.
- `hw/p0/browser/eecircuit.html` + `run_eecircuit.cjs` (Playwright, tiny static server over the repo root): same `demo_flat.cir`, 1520 points, same variable set as ngspice.
- Result vs ngspice 42 (`hw/p0/compare_spice.py`, `hw/p0/compare_spice.json`): max abs error of V(N1), V(N2), V(T1), V(T2) between 1.4e-9 and 1.9e-9 V, V(OUT) 5.2e-9 V (waveform peaks 0.55 to 1.04 V); 50 % crossing times agree to better than 0.001 ps (e.g. N2: 484.70737512 ns vs 484.70737466 ns); time grids identical to 2e-15 s.
- Timing: engine start (wasm init) 362 to 384 ms, `.tran` run 48 to 60 ms, page load to result 0.73 to 0.77 s wall (local server, headless Chromium). Unit cell target (<= 3 s) is met by a wide margin; the 20 MB module must be lazy-loaded and is the cost.
- The embedded ngspice version is not reported by the package; agreement with ngspice 42 is the evidence.

## Check 7: JS forward pass

- `dlnn/viz/export_web.py`: general exporter. `model_to_json` ("dlnn-web-model/1": int8 W as base64 [O,I,K], gain/bias/readout as float32 base64, dilation, vsat, readout_from/kind, input_scale, sha256, meta) and `examples_to_json` ("dlnn-web-examples/1": x, labels, SNR, tier-V logits from `virtual.forward`, optional per-layer outputs); `modrec_examples` takes the pipeline's `stratified_subset` of the modrec test set (40 examples = one per class x SNR cell). CLI: `python -m dlnn.viz.export_web --model models/tdlnet_modrec_ternary.npz --n 40 --out docs/artifacts/data`. Wrote `docs/artifacts/data/model_tdlnet_modrec_ternary.json` (3.1 KB) and `examples_tdlnet_modrec_ternary.json` (57 KB).
- `docs/artifacts/assets/js/tdlnet.js`: dependency-free ES module (`decodeModel`, `layerForward`, `forward`, `decodeExamples`, `softmax`, `argmax`); works in node and in browsers; sparse tap lists, float32 storage, pooled and per-step readouts. Does not touch `dlnn/tiers/*` or `dlnn/model/*`.
- Result on the w8 ternary model (3 layers 8x2x8, 8x8x8, 8x8x8, dilations 1/4/16, pooled readout from layer 2): **max |JS - numpy| logit error 1.9e-6**, top-1 agreement 40/40, JS accuracy 30/40 on this subset; 0.49 ms per example in node.
- `tests/test_export_web.py` (pytest, 2 tests, about 1 s) exports to a tmp dir and runs `tests/js/check_tdlnet.mjs` through node; asserts error <= 1e-5 and 40/40 top-1 agreement.

## Check 8: vendored libraries and browser rendering

`docs/artifacts/vendor/` (sizes, gzip, sha256 in `VENDOR.json`, each folder has its `LICENSE`):

| Library | Version | Licence | Files | Bytes (gzip) |
|---|---|---|---|---|
| d3 | 7.9.1 | ISC | `d3.min.js` | 280,493 (92.5 KB) |
| three | 0.187.0 | MIT | `three.module.js` + `three.core.js` | 658,900 + 1,474,799 (129 + 290 KB) |
| three addons | 0.187.0 | MIT | GLTFLoader, BufferGeometryUtils, SkeletonUtils, OrbitControls | 120 + 39 + 12 + 41 KB |
| KiCanvas | official build from `https://kicanvas.org/kicanvas/kicanvas.js` (no version string) | MIT (LICENSE from `github.com/theacodes/kicanvas`) | `kicanvas.js` | 477,451 (112 KB) |
| eecircuit-engine | 1.9.0 | MIT | see check 6 | 20.3 MB (5.68 MB) |

- KiCanvas is not on npm (404) and `github.com` HTTPS clone is blocked at the proxy (403); the official build and the raw LICENSE file were fetched instead. Its build references `fonts.googleapis.com` (UI font) and GitHub APIs for its remote-loading feature; offline the font falls back. Two harmless 404s for a `$$:0:$$` placeholder URL appear per page.
- `hw/p0/browser/three.html` (importmap: `three` -> vendored module) + `kicanvas.html` + `run_vendor_check.cjs` (Playwright, chromium with SwiftShader WebGL). Results: three.js r187 loads `demo.glb` in 179 ms, 1362 meshes, 84 x 11.5 x 54 mm bounding box, rendered with OrbitControls; KiCanvas `<kicanvas-embed>` loaded both `demo.kicad_sch` and `demo.kicad_pcb` (`loaded: true`). Screenshots (viewed): `hw/p0/shots/three_glb.png` (board, tracks, headers, ICs visible), `hw/p0/shots/kicanvas_sch_pcb.png` (schematic sheet with symbols, and the PCB with copper layers and outline; KiCanvas shows "Click or tap to interact" overlay on the first embed).

## Fallbacks and findings for later milestones

| Item | Finding |
|---|---|
| `add-apt-repository` | use `/usr/bin/python3.12` |
| kicad libraries | name `kicad-symbols kicad-footprints kicad-packages3d` explicitly; copy template lib tables |
| kinet2pcb 1.1.4 + KiCad 9 | strip `(property ...)` and replace `FootprintLoad` (done in `hw/p0/netlist_to_pcb.py`) |
| pcbnew stackup | not in the python API; inject S-expression text (done in `make_pcb.py`) |
| freerouting | needs JDK 25 (auto-fetched by its Gradle build); `executableJar` task; Maven Central 429 retry; `--gui.enabled=false`; `-mp` deprecated in favour of `--router.autorouter.max_passes` |
| Third-party downloads | besides the approved sources, the freerouting Gradle build itself fetched Gradle 9.8.1 (services.gradle.org), Maven Central/Gradle plugin portal artefacts and the Temurin JDK 25 (via foojay/Adoptium). These are the build's own dependencies. |
| LCSC numbers | only 3 mapped, unverified; M1a must source real parts |
| Delay metric | AC phase delay 120 ns/section vs 197 ns step 50 % spacing for the 2-section demo |
| Not done in P0 | hierarchical sheets, weight-dependent CPL, 4-layer layout tuned for RF/ringing, DRC rules per JLC capability, Binder (KiCad 6, no kicad-cli) |

## Reproduce

```
./setup_eda.sh                                   # idempotent: KiCad 9, hw/.venv-eda, freerouting, vendored web libs
hw/p0/build_p0.sh                                # ERC -> netlists -> ngspice -> PCB -> freerouting -> DRC -> fab (about 20 s)
NODE_PATH=$(npm root -g) node hw/p0/browser/run_eecircuit.cjs
(cd hw/p0 && ../../.venv/bin/python compare_spice.py tran_ngspice.txt eecircuit_tran.json)
(cd hw/p0 && ngspice -b demo_ac.cir)             # phase at 200 kHz -> phase delay
NODE_PATH=$(npm root -g) node hw/p0/browser/run_vendor_check.cjs
.venv/bin/python -m dlnn.viz.export_web --model models/tdlnet_modrec_ternary.npz --n 40 --out docs/artifacts/data
.venv/bin/python -m pytest tests/test_export_web.py
```
