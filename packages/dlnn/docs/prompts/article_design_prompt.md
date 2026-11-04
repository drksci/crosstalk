# Prompt: write and design the article

You are writing and designing a long-form, Hackaday-style technical article, with live interactive visuals, about a **delay-line neural network (DLNN)**.

The network is a ternary tapped-delay-line net. Its tap weights are in {-1, 0, +1}, so each weight is just a tap polarity, in the BitNet b1.58 style. It sums the taps in analog, applies a tanh soft limiter and pools with an RC stage. A small digital readout then classifies 8 radio modulation types.

The project simulates the same trained model at four fidelities:
- **V:** ideal math in numpy.
- **E:** bit-true fixed point, verified against Verilog RTL.
- **C:** SPICE circuit simulation.
- **R:** an RF / signal-integrity channel model.

There is also a manufacturable PCB design.

The audience is sceptical electronics engineers and makers. The article has to be beautiful, and it also has to be **verifiable**: every number links to the notebook or test that reproduces it.

## Sources of truth
- **Prior art:** the literature study found close prior work: pour2025 (ISCAS 2025) and merton2026 (a Virginia Tech thesis) on RF recurrent nets with tapped transmission-line delays, woo2024 on ternary weights for modulation recognition, and tank1987. Never claim "first". Position the project as a reproducible, four-fidelity, open-hardware build with an explicit prior-art comparison.
- **Accuracy context:** our accuracy is on our own synthetic set and is not comparable with RadioML benchmarks. (read before writing; never invent numbers)
All files are under `/home/user/dlnn/`:
- **Results and figures:** `docs/README.md`, `docs/results.md` (generated from `out/final-w32/compare.json` and `out/final-w8/compare.json`), and `docs/figures/` for the static figures.
- **Concept and tiers:** `docs/concept_and_math.md`, `docs/tiers.md`, `docs/rf_model.md`, `docs/circuit_extraction.md`, `docs/fixed_point_spec.md`.
- **Caveats:** `docs/honesty.md` (the claims you may and may not make), `docs/review.md` (independent audit and response) and `docs/deviations.md`.
- **Lineage:** `docs/literature.md`, `docs/references.bib` and `docs/artifacts/data/literature.json` give the lineage from delay-line memory to this project. Cite only the verified entries.
- **Phase-2 plan:** `docs/phase2_plan.md`, once it exists, has the buildable board design (delay element, symbol rate, tiles or multilayer, BOM, KiCad outputs). Wherever it isn't final, write `[[TBD: board]]`. Do not guess.
- **Notebooks:** `notebooks/*.ipynb`, plus the public "verify" notebook once it exists.

## Voice and length
- Hackaday long-read: first person plural, concrete, a bit playful, never hype. Open with the hook that 1940s computers stored bits as sound in tubes of mercury, and we use the same trick to compute.
- About 4,000-6,000 words, plus figures and widgets. Short paragraphs. Equations only where they earn their place (layer equation, tanh from the diff pair, delay = length / v_p).
- Every quantitative claim carries a small evidence chip linking to the notebook cell, test or JSON field that produces it.
- Label everything **simulated**, **emulated (RTL)**, **designed (not fabricated)** or **measured**. Nothing is measured yet, so nothing gets that label.

## Article structure
1. Hook: mercury delay lines, then the question "what if the delay line *was* the neural net?"
2. The lineage, with the interactive timeline (see widget 1): delay-line memory, transversal filters, CCD/BBD/SAW, ADALINE, TDNN, WaveNet/TCN, delay reservoirs, analog AI, BitNet.
3. The idea in one picture: taps as weights, polarity as ternary, the diff-pair tanh.
4. The maths, light: layer equation, dilation, receptive field, what passive parts cannot do (gain, nonlinearity, softmax), and the minimal hybrid.
5. Training a net that hardware can run: ternary QAT and the straight-through estimator, width 8 vs width 32, and the 8PSK collapse at width 8 told honestly.
6. Four fidelities, one model: V/E/C/R side by side, what each simulates and omits, and the results table.
7. Where it breaks: line loss, the compensation-mode finding (C 0.663 default vs 0.80 without compensation), ADC bits, isolation, tolerance sweeps.
8. From coax to a board: why metres of RG-58 at 10 ns is silly, and the buildable design `[[TBD: board]]`.
9. The board: schematic, layers, 3D, BOM and cost, how tiles chain. Labelled "designed, DRC-clean, not fabricated".
10. Scaling up, told with the Jupyter at-scale simulations (simulation only). Covers:
   - how many modules, tiles or boards the full and scaled-up nets need
   - area, part count, BOM cost, power and latency
   - accuracy and energy/latency per classification versus the traditional ANN on an MCU/DSP
   - scaling curves against width, depth, taps, tolerance and loss

   Every chart comes from the notebooks' exported data. Energy figures are labelled as estimates and cite their sources.
11. What doesn't work / what we got wrong. Include the LLM commentary experiment honestly: it restated a Python-computed list and didn't discover anything.
12. Reproduce it yourself: the public notebook, Binder/Colab badges, the verify notebook's PASS/FAIL table, and a claim-to-evidence map.
13. References: verified only.

## Visual design system
- **Aesthetic:** a modern lab notebook crossed with an oscilloscope. Warm off-white paper in light mode; deep graphite, not pure black, in dark mode. Signal traces glow subtly in dark mode only.
- **Type:** a humanist serif for body text (e.g. Source Serif 4), a grotesk for UI and labels (e.g. Inter), and a monospace for numbers, part numbers and code (e.g. JetBrains Mono). Use tabular numerals in all tables and readouts.
- **Colour:**
  - One fixed colourblind-safe colour per tier, used everywhere: in charts, chips, tables and legends.
  - Tap polarity: +1 warm, -1 cool, 0 neutral grey.
  - Status chips: simulated / RTL / designed / measured.
  - Define everything as CSS custom properties on `:root`, with a `prefers-color-scheme` dark override and a manual theme toggle.
- **Layout:** a single column of about 68ch for prose. Widgets break out to full width. Sticky section nav on desktop collapses to a top menu on phones. Must be legible at 360 px width with no horizontal scroll.
- **Motion:** purposeful only. Signals propagating along lines, probabilities easing into place. Respect `prefers-reduced-motion`.
- **Diagrams:** clean inline SVG. Direct labels instead of legends wherever possible. No chart junk.

## Interactive widgets
All data is exported from Python to JSON or binary under `docs/artifacts/data/`. The notebooks read the same files, so the page and the notebooks can never disagree.
1. **Lineage timeline (D3).** Years on the x axis, strands as lanes. Hover shows the reference; click opens the verified source. Data: `literature.json`.
2. **Network / tap explorer (D3).** The real trained weights. Hover a tap to see its polarity, physical delay, element (coax length or board part) and layer.
0. **Live duel (the centrepiece).** A small DLNN and a conventional ANN of comparable size (an MLP or 1-D CNN, trained on the same data) run side by side on the same continuously cycling input stream. Signals stream in, propagate tap by tap along the delay lines, and both nets' class probabilities update in sync.
   - Controls: pause, step, modulation, SNR, noise and tolerance injection.
   - Readouts: live accuracy counters, ops/energy-per-classification estimates and latency for each net, labelled as estimates.
   - Both forward passes run in JS from exported weights, and both are verified to match their Python versions.
3. **Live inference.** The actual ternary net runs in the browser as a small JS forward pass from the exported weights. The reader picks a modulation and an SNR, watches waveforms move through each layer, and sees the class probabilities update. Show V and E, the fixed-point result, side by side.
4. **Four-fidelity scope.** Overlaid waveforms, eye and spectrum for V/E/C/R on the same example, with residual vs V. Uses precomputed traces.
5. **Degradation lab.** Sliders for SNR, line loss, ADC bits, isolation, tap tolerance and compensation mode. Readouts show accuracy and KL per tier from the precomputed sweep grid, with points interpolated between measured grid values clearly marked as interpolated.
6. **Circuit sandbox.** The board's unit cell. Use real ngspice compiled to WebAssembly if that's feasible; otherwise use precomputed SPICE sweeps, and label which one it is. The reader edits a component value, reruns, and sees the waveform.
7. **PCB viewer.** Toggleable layer SVGs (copper, silk, mask, drill) and a three.js 3D board from the KiCad GLB export, with a BOM table and cost.
8. **Stitching diagram.** How tiles or boards chain into the full network, with connector and signal path and what's physical vs simulated.
9. **Claim-to-evidence map.** A table of every headline claim, its value, the verify-notebook cell and PASS/FAIL status.

## Engineering constraints
- **Output:** a static site in `docs/artifacts/article/`, with `index.html`, `assets/`, a `data/` link or copy, and vendored libraries at pinned versions (D3 v7, three.js, and anything else, with each licence noted). No CDN at runtime, no server, and it works offline from `file://` or any static host. Keep it portable, because another agent will move it into a different site.
- **Performance:** first paint under 1.5 s on a mid-range phone. Lazy-load the heavy widgets (3D, wasm). Total page weight under 8 MB excluding the optional wasm.
- **Accessibility:** alt text for every figure, keyboard-operable widgets, a text fallback for each widget, and WCAG AA contrast in both themes.
- **Also provide:**
  - `docs/artifacts/article/article.md`: the article text in plain Markdown, for the publishing agent.
  - `docs/artifacts/article/README.md`: how it's built, the data contracts, how to regenerate the data, and the file manifest.

## Acceptance checks (run them and report)
- Every number in the article matches its source file. Write a script that cross-checks the numbers in `article.md` against the JSON exports and fails on any mismatch.
- All widgets render with no console errors in headless Chromium (Playwright is preinstalled). Take screenshots at 1440 px and 390 px, light and dark, and review them yourself.
- No reference appears that isn't in the verified literature list.
- The honesty rules hold: nothing is labelled measured, the board is "designed, not fabricated", and the LLM is described as restating, not analysing.
