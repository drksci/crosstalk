# DLNN article shell (M4a)

Static, offline page: `article/index.html` + `assets/{css,js,fonts}` + `data/*.json` + `vendor/*` (d3 7.9.1 loaded lazily; three, KiCanvas and
eecircuit-engine are vendored for later widgets and are NOT loaded by this page). No CDN. ES modules and `fetch` need an HTTP origin, so
`file://` does not work; serve the folder: `python3 -m http.server 8765 --directory docs/artifacts` then open `http://localhost:8765/article/`.

## Structure
- `index.html`: top bar with sticky section nav (collapses to a "Contents" menu under 900 px), honesty box, 13 article sections (all prose is
  `[[DRAFT]]` placeholder for M5), a W0 arena block after section 1, W8/W1/W2 widgets, dashed stubs for W3-W7, W9, W10.
- `assets/css/site.css`: tokens on `:root` (light), dark override under `prefers-color-scheme` and `[data-theme]` (toggle stored in localStorage,
  try/catch). Fonts: Source Serif 4 (body), Hanken Grotesk (UI), JetBrains Mono (numbers), self-hosted woff2 under `assets/fonts/` with OFL licences.
- `assets/js/`: `main.js` (boot, nav, theme, lazy widgets), `chips.js` (evidence chips), `common.js`, `signals.js` (JS port of `dlnn/data/modrec.py`,
  FFT, PRNG), `models.js` (JS forward passes: MLP, CNN, reservoir, cumulant features, DLNN with tap tolerance), `tdlnet.js` (P0 DLNN forward),
  `arena.js` (DOM-free engine), `inputs.js` (cycle / key / mic / draw sources), `arena-ui.js` (W0 widget), `viz.js` (W8, W1, W2, references).
- `tests/js/`: `check_arena.mjs`, `arena_e2e.mjs`, `shots.mjs`, `check_tdlnet.mjs` (P0).

## Colour
Tier colours V violet / E aqua / C gold / R magenta, tap polarity + warm / - cool / 0 grey, status chips with icon + label (simulated, RTL, designed,
measured). Tier and 7-lane palettes were run through the dataviz `validate_palette.js` (adjacent pairs, both modes: pass; contrast below 3:1 on
light for aqua/gold/magenta, so each use carries a letter or label). Polarity is always paired with a +/-/0 glyph.

## Data contracts (all under `docs/artifacts/data/`)
| File | Schema | Producer |
|---|---|---|
| `arena_models.json` | `dlnn-arena-models/1`: `models[]` {id, label, params, ops, acc, provisional, data{schema-specific, base64 float32/int8}} | `python -m dlnn.viz.export_web --arena` |
| `examples_arena.json` | `dlnn-arena-examples/1`: 40 stratified test windows + numpy logits per model | same |
| `examples_gen_stats.json` | `dlnn-gen-stats/1`: per-class mean/std of 25 features of the Python generator | same |
| `energy_refs.json` | `dlnn-energy-refs/1` (copied from `models/arena/`); absent means "pending" | same (copy) |
| `claims.json` | `dlnn-claims/1`: id, text, value, unit, tolerance, fidelity, source{file,key}, notebook_cell, pytest, status | `python docs/artifacts/article/build_claims.py` |
| `literature.json` | W8 and references | lit. worker |
| `model_*.json`, `examples_tdlnet_*.json` | `dlnn-web-model/1`, `dlnn-web-examples/1` | `export_web --model` |

Regenerate everything: `.venv/bin/python -m dlnn.viz.export_web --arena --out docs/artifacts/data && python3 docs/artifacts/article/build_claims.py`.
Evidence chips: `<a data-claim="p1.w32.acc.C" data-show="value">`; unresolved ids render red.

## Widgets and how each is verified
- **W0 Arena.** Models: DLNN (`tdlnet_b_w8_ternary`), DLNN-F (float readout), MLP-S / MLP-L (**provisional**, being retrained; slot is generic, drop in a new
  npz and re-export), 1-D CNN, reservoir, cumulant. Inputs: cycling labelled stream, space-bar / touch keying (OOK, 2FSK), microphone (loudness to OOK,
  pitch to 2/4FSK; denial falls back with a message), drawn I/Q path. Knobs: SNR, CFO, DLNN tap tolerance; pause, step, speed; challenge.
  Verify: `node tests/js/check_arena.mjs` (JS vs numpy logits on 40 stored examples <= 1e-5, top-1 agreement; JS generator vs Python generator per-class
  feature means, two-sample z <= 5), `pytest tests/test_export_web.py` (runs it; numpy reference vs torch), `node tests/js/arena_e2e.mjs` (all inputs,
  step, challenge, frame rate with CPU throttle).
- **W1 taps** and **W2 inference**: same weights as W0; W2 prints the live JS-vs-numpy max difference for the chosen example.
- **W8 timeline**: from `literature.json` (48 events, 70 references); text fallback in a `<details>`.
- Screens: `node tests/js/shots.mjs` writes `article/_screens/*` (1440 and 390 px, light and dark), console errors and page weight.

## Honesty notes
Energy in the arena is a digital-equivalent arithmetic estimate from Horowitz ISSCC 2014 (45 nm); nonlinear ops are assumed 10 fp32 multiplies; MCU
figures (STM32G4) are pending. "time" is browser compute time. On accuracy the CNN and cumulant classifier beat the w8 DLNN; the page says so.
