# Honesty: what this demo claims and what it does not

## What may be claimed

**Simulation-only** cross-fidelity consistency and degradation curves of a small ternary delay-line network on **synthetic** data:

* the trained ternary network, evaluated in four simulations, gives (a) the same answer as the float-precision ideal math when implemented in 8-bit fixed point (tier E agrees with V on
  almost every example, with the RTL bit-identical to the golden model on the subsets that were compared), and (b) clearly worse answers when line loss and the other channel
  non-idealities of C and R are applied, with the degradation curves in `results.md`;
* the numbers are reproducible from the commands in `reproducibility.md`.

## What is NOT claimed

| Not claimed | Why / detail |
|---|---|
| Any hardware measurement | nothing was built or measured; every circuit and RF number is simulation |
| Vendor-accurate models | the BJT diff pair is a generic textbook NPN (`hw/spice/diffpair.lib`), the coax is a generic RLGC RG-58-like line, not a datasheet or measured cable |
| EM, layout, connector, bend, packaging effects | not modelled in any tier |
| That the full network is buildable at RG-58 loss without repeaters | the `--cascade` headline for C and R is exactly that no-repeater point, and the tiers degrade there; at lower loss they recover. Not all of the degradation is line loss, though: see "C's headline depends on a modelling choice" below |
| Competitiveness with digital classifiers | not tested; the float32 reference is the same topology, not a digital state of the art |
| That the LLM analysed anything | it comments on a text card rendered from `compare.json`; it restates the card. Without the card's Python-computed divergence list it found the divergent tier in **0 of 5 injected cards** (the 1 "correct" card of 8 is a control with no divergence, and it got only 1 of 3 controls right) |

## Specific limits of each tier

**E (emulated).** The RTL (`hw/rtl/tdl_layer.v`, Icarus Verilog 12) is compared bit for bit with the numpy golden model on a **subset** (64 test examples through all three layers;
for the width-32 net also only the first 32 of 128 time steps, which is exact for those steps because the network is causal). In 32 steps the layer-3 taps k = 2..7 (lags of 32 to 112 steps) only ever see zero history,
so six of the eight layer-3 tap positions are not driven with data in the width-32 RTL check, and the "pooled sums" there are 32-step prefix pools, not the classification pools. The width-8 run
(`final-w8`, all 128 steps, same module) is the evidence for the long-tap logic. The 4000-example metrics and all sweeps run on the numpy golden model, not on RTL. The
ADC quantiser, the pooled-sum conversion and the float readout are numpy in both paths. Analog effects are not modelled. ADC full scale is the fixed value stored in the model (`input_fullscale`: train-set
abs-max rounded up to 0.5, 4.5 for the width-32 file), else the abs-max of the batch; a 99.99th-percentile rule was used only for the random-init nets of M2a.

**C (circuit).**
* ngspice ran only on a **layer-1 unit cell** (16 test examples end to end) plus the `.ac` tap transfer functions and the `.dc` diff-pair curve. The 4000-example numbers are from a
  **circuit-calibrated behavioural model** (closed-form RLC line fitted to the SPICE data, tanh fitted to the `.dc` curve), not from SPICE.
* **Layers 2 and 3 of C are extrapolated**: their long lines (up to 1.12 microseconds) use the fit from layer 1's short lines and were never simulated in SPICE.
* ngspice's LTRA line model supports only constant series R and G = 0. Consequently there is no `sqrt(f)` skin effect, low-frequency loss is overstated (DC loss of the longest layer-1 tap
  3.6 dB against 2.2 dB at 100 MHz), and the taps carry an **excess group delay of 54-110 ps** (`taps.delay_error_s` in `params/circuit_extracted.json`).
* The RC integrator is a leaky average (tau = N*Ts) and differs from the uniform mean the net was trained with; the tier's readout uses the exact digital mean.
* **Most of C's gap to V is a modelling choice, not line loss.** At 32 dB/us the default loss compensation ("layer": one ideal gain trim per layer) scores 0.663, whereas "none" (no compensation) scores 0.804 and an ideal
  per-tap repeater 0.805 (`params/circuit_diagnostics.json`, `results.md` section 2). V is 0.863, so about 14 of the 20 points of C's gap come from the default trim and about 6 from the line loss. The cause was
  measured in M4d (`circuit_extraction.md`, "Where C's gap comes from"): the layer-3 trim is 2.85 (1 / the weighted mean main-cursor gain, 0.35) and multiplies the unattenuated short taps too; on 400 test examples
  (4000 gives the same to within half a point) 72.4 % of C's layer-3 activations are in tanh saturation (|tanh| > 0.95) with the `layer` trim, against 49.5 % in V, 30.7 % with `none` and 49.8 % with per-tap repeaters.
  Method: the tier's own forward code, read-only, `scripts/saturation_check.py`. The headline stays at the default `layer` mode (not changed, not re-run); it should be read as "the default trim over-drives layer 3 at this operating point", not as "RG-58 loss costs 20 points".
  Which compensation a real build would use is not settled here, and the `none` / `tap` numbers are optimistic (no repeater noise).
  Both statements belong with the headline: the cascade operating point is not buildable without repeaters, and the default compensation mode over-drives layer 3.
* Noise is off in SPICE; temperature and bias drift are not modelled.

**R (RF).**
* Channel-level scikit-rf two-port cascades turned into impulse responses and run by numpy convolution; no EM or layout; amplifier is memoryless (no bandwidth limit, AM/PM or drift);
  re-timing between layers is ideal; crosstalk is flat and between neighbouring lines only; hybrid isolation is not separate from coupler isolation.
* **R's `snr_db` axis is the per-stage amplifier SNR** (three independent noise sources), not the input AWGN that V, E and C use. The two SNR axes are not comparable.
* **The 98 % criterion note.** The plan's acceptance "30 dB SNR, 0 dB/us, ideal couplers: top-1 agreement vs V >= 98 %" is ambiguous about where the 30 dB is injected. It failed as per-stage SNR on random-init nets,
  holds as input AWGN, and holds per-stage on the trained nets (`results.md` section 8). The code was not tuned to pass. This is the *ideal-channel* number; R's headline operating point is far
  harsher.
* With `--cascade`, R inherits C's fitted 32 dB/us loss; the SPICE `tap_H` samples are not injected into R and the plan's "V_cal" sensitivity check in V was not implemented. The fitted loss (32.000005 dB/us)
  equals the 32.0 that was typed into the netlist (the LTRA line and the closed-form fit are the same RLC line, fit residual 1.3e-13), so the cascade is plumbing and provenance, not new information; the only extracted
  numbers that are not configured inputs are the diff-pair scales (V_sat 52.02 mV vs 51.8 mV textbook, gain 1.0003).
* One amplifier gain per layer compensates the mean tap loss (long taps are weaker); a per-tap equaliser exists but is optimistic (no repeater noise). Measured in M4d with the same method as for C: at the
  cascade loss 71.3 % of R's layer-3 activations are saturated (400 examples; V 49.5 %), against 54.4 % at R's standalone 5 dB/us, and R's `tap_equalize` raises its accuracy at 32 dB/us from 0.701 to 0.810 (full set)
  while layer-3 saturation falls to 61.3 %. So R's gap at the headline also depends on its gain rule. Whether that rule is a modelling artefact was not decided, and `tap_equalize` changes more than layer 3.
* The R defaults (5 dB/us when not cascaded, 30 dB isolation, 0.5 dB / 5 degrees tap errors, reflection 0.05, -35 dB crosstalk, 30 dB stage SNR, 3 dB NF) are typical values from the plan, not tuned and not measured.

**Models.** The plan-specified **width-8 ternary net misses its target** (>= 85 % at SNR >= 10 dB; it reaches 76.3 %) and **collapses the 8PSK class**. The headline is therefore a wider net
(width 32), a deviation from the plan; both are reported (`results.md` section 4). The training recipe differs from the plan's (`deviations.md`); the width-32 model is a single seed, and the
test set was not used to pick seeds. NARMA10 is a sanity task only; its circuit tier at the no-repeater loss is poor.

**LLM.**
* Qwen2.5-1.5B Q4_K_M comments on a card. The guard checks that numbers (and units like "6-bit") appear on the card and that tier statuses are not misdescribed; it cannot check that a causal
  sentence is true. `likely_cause` text is copied from the tiers' own descriptions, not physical analysis.
* On synthetic cards, attribution is 5/5 injected cards (2/5 under the strict most-severe-metric rule) plus 3/3 controls only because the card contains a Python-computed list of material divergences; without it **0/5 injected cards**
  (the 1 of 8 that counts as correct is a control with no divergence; Qwen gets 1 of 3 controls right, i.e. it invents divergences on the others). That is restatement, not discovery. The synthetic injections are easy.
* The commentary of the final run (`final-w32`) is shown in `results.md` section 9.2 exactly as produced and was not edited. All 25 numbers in it pass the number guard, but the guard flags
  `ids_verified = false` because the divergent tier R has no finding (R appears only in the summary and in a caveat); every finding is about C and states each metric value twice in one sentence; the two caveats
  call C and R "not the real tool" although both ran their real tool (ngspice, scikit-rf) with status `ok`; and the question is not meaningful ("recover ... with more accuracy"). The guard does not catch the last
  three. Earlier Qwen commentaries on the same kind of card contained other errors (a "6-bit" or "32-bit" ADC claim for C that is not on the card, and "R is a FALLBACK" while R's status is `ok`; see
  `llm_commentary.md`, `llm_eval.md`, `build-log.md`). Different wording each time, same conclusion: the model restates the card, sometimes badly, and its text is not analysis.
* BitNet b1.58-2B (bitnet.cpp) built and ran but produced degenerate output here (0/5 injected cards), cause not investigated. It is not usable and not part of the demo.
* llama-cpp-python had to be built from source because the index wheel is musl-linked; the build is `-march=native` and tied to this CPU.

**Database.** SQLite is the default and always works. PostgreSQL was verified only against a local, throwaway PostgreSQL 16 server on this box (peer authentication over a unix socket, no password); no
hardened or remote deployment was tested.

## Known gaps

* No SPICE for layers 2-3, no skin-effect line in SPICE, no noise in SPICE.
* No repeater/noise model for long lines, so the 32 dB/us cases are pessimistic in one sense (no repeaters allowed, and the default per-layer trim over-drives layer 3) and unmodelled in another (repeaters would add noise).
* Test set of 4000 examples from one seed: differences of about a point between tiers are within what a different seed could change; no confidence intervals were computed. Width-8 training showed seed noise of about +-2 points.
* The notebooks' prose describes the headline run even when another run is loaded.
* Figure 3(c) overlays folded combiner waveforms of two different pulse shapes and tap models (it is not a true eye diagram; R's panel shows none); it visualises two channel models, not a measurement.
* The `modrec-w32-full` card (earlier run) exceeded the 1.5k-token budget; the final run's card size is reported in `results.md`.

## What I would do next

1. Repeater-aware loss model: per-tap or per-segment amplifiers with noise, and decide between C's compensation modes on physical grounds, then re-run the loss sweep.
2. Replace the constant-R LTRA line (for example by a lumped RLC ladder with `sqrt(f)` loss fitted to the scikit-rf coax), so C's long taps can be simulated in SPICE and layers 2-3 stop being extrapolated.
3. Train the width-8 and width-32 nets with several seeds and report confidence intervals; try width 16 and larger readouts to see where 8PSK comes back.
4. Quantisation- and loss-aware training: put the 32 dB/us (or a swept) line response and stage noise into the training loop, which is the obvious way to recover accuracy at C and R.
5. Evaluate the commentary on cards from real runs with injected faults rather than synthetic ones, try a larger model, and treat BitNet's failure properly (chat template versus the i2_s tensors).
6. Build one real hardware-in-the-loop data point (a short tapped coax line and one amplifier) before trusting any of the RF defaults.
