# Deviations: planned versus done

The plan is `docs/plan.md` (kept as written). This page lists, per milestone, where the work differs from it. Sources are the build logs
(`build-log.md`, `build-log-M1.md`, `-M2a`, `-M2b`, `-M2c`, `-M3`, `-M4b`); numbers are quoted from there.

## Summary of the decisions that matter

| # | Plan | What was done | Why |
|---|---|---|---|
| 1 | TDL-Net-S, width 8 (1,216 ternary weights, counting the 8x8 readout) is the headline | **Width 32 is the headline** (16,896 ternary conv weights, 17,152 counting the 8x32 readout, the same convention as 1,216 for width 8); width 8 is kept as a documented secondary and is also run through all tiers | width-8 ternary reaches 73.5 % mixed / 76.3 % at >= 10 dB and misses the 85 % target; same code at width 32 reaches 86.3 % / 89.1 % |
| 2 | Adam lr 3e-3, 20 epochs, ~3-5 min | Adam lr **3e-2**, cosine, **60 epochs**, IQ-rotation augmentation, progressive ternarisation, seeds chosen by TRAIN accuracy (width 8: 3 seeds; width 32: 1 seed) | the plan recipe underfits badly (float 62.1 %, ternary 45.1 % on the test set) |
| 3 | llama-cpp-python from the pinned wheel on the abetlen index | built **from the sdist** (`pip install --no-binary llama-cpp-python llama-cpp-python==0.3.19`, about 4 min with 2 jobs) | the index's `cp313-linux_x86_64` wheel links against musl (`libc.musl-x86_64.so.1`) and does not import on this glibc box |
| 4 | BitNet b1.58-2B as optional second LLM backend, 45-min time box | built via bitnet.cpp (clang 18, cmake) and run, but its output is **degenerate** (0/5 injected cards attributed); reported as unusable here; Qwen2.5-1.5B Q4_K_M is the working commentator | cause not investigated (time box) |
| 5 | R acceptance: "30 dB SNR, 0 dB/us, ideal couplers: top-1 agreement >= 98 %" | missed on random-init nets as per-stage SNR (93.6 %); holds under the input-AWGN reading (98.4 %); **met on the trained nets** per-stage (width 32: 0.9863; width 8: 0.9809) | the criterion text does not say where the 30 dB is injected; no code was changed |
| 6 | Circuit tier runs the full test set | C runs SPICE only on a layer-1 unit cell (16 examples) plus `.ac`/`.dc` extraction; the 4000-example numbers and layers 2-3 come from the **circuit-calibrated behavioural model** (as the plan's full-set/subset rule already stated), labelled as such | ngspice speed |
| 7 | `--cascade` feeds R and a V "V_cal" sensitivity check | only R consumes `circuit_extracted.json` (fitted loss and amplifier scales; SPICE `tap_H` samples are not injected); the fitted loss equals the number put into the netlist, so this is plumbing and provenance, not new information; **V_cal not implemented** | scope |
| 8 | Subset policy for C and R | not needed; with width 32 all four tiers run on all 4000 examples (`DEFAULT_CR_SUBSET = None`). The subset code path exists and is tested | runtime was within budget |

## M0 walking skeleton

* `scikit-rf` was not a dependency yet; `jsonschema` was added for contract validation.
* `acc_per_snr` is a `{snr: acc}` dict; `nmse` and `nmse_vs_virtual` metric ids were added for the regression task; sweeps carry an optional `nmse[]`.
* `arrays` in a tier JSON is an index (file/key/shape/dtype), with nested names flattened using `.`.
* Every tier computes the V forward itself so that `run()` keeps the four-argument signature.
* E-tier LUT range fixed at +-4 (step 1/32) with a Q.15 multiplier.
* Acceptance: NARMA10 NMSE_V 0.1497 (limit 0.40), NMSE_E 0.1555 (+3.9 % against a 10 % limit); met. C and R were stubs at that stage.

## M1 model

* Ternary readout is stored as float32 `gamma_r * exp(log_ra) * {-1,0,+1}` (digital readout, contract compatible); the float baseline uses a separate schema string `dlnn-model-float/1`, which the ternary loader refuses on purpose.
* `gamma` is exported as the input-referred gain `vsat = 1/(alpha*gamma)`, not as an output gain. The per-line gain (`--line-gain`) was tried and is not used in the shipped models.
* Plan acceptance, width 8: train time 3.4 min (limit 10), float 84.65 % mixed (limit 80), ternary 73.48 % mixed (limit 70), V reproduces torch logits to 1.9e-6 (limit 1e-4), **ternary at >= 10 dB 76.29 % against the 85 % target: not met**. The width-8 ternary net drops 8PSK entirely (0 %).
* Width 32 (`--width 32`, 1 seed, 60 epochs, 150 s ternary): ternary 86.3 % mixed / 89.1 % at >= 10 dB; float 91.9 % mixed / 95.5 % at >= 10 dB (the M1 note quoted 91.9 as the >= 10 dB figure, corrected in M4a).
* The table of what was tried for width 8 (learning rates, scaling, per-line gain, distillation, warm start, widths 16/32) is in `build-log-M1.md`.

## M2a emulated

* Validated first on a **randomly initialised** net (the trained model did not exist yet); re-run on the trained nets in M4a (E vs V top-1 agreement 0.9958 at width 8, 0.9972 at width 32).
* ADC full scale: on the random-init nets of this milestone a 99.99th-percentile rule was used (abs-max gave 0.955 mean agreement, p99.99 gave 0.9685 against the 0.95 limit); that percentile code lives only in `tests/helpers_tdlnet.py`. The shipped `emulated.adc_codes` uses the fixed full scale stored in the model (`input_fullscale`, the train-set abs-max rounded up to 0.5; 4.5 in the width-32 file), else the abs-max of the batch.
* Per-line requantisation uses the signed ratio `gain/max|gain|` and is always applied; identical for the trained nets (positive equal gains).
* The RTL is run on a subset only (work cap of 4.5e6 MACs per layer): the first 64 test examples through all three layers; for the width-32 net also only the first 32 of 128 time steps (causal, so the prefix values are exact; but in 32 steps the layer-3 taps k = 2..7, at lags of 32 to 112 steps, only ever see zero history, so the long-tap logic is exercised by the width-8 full-length RTL check, not by the width-32 one). The exact counts of each run are in `results.md`, generated from `E_rtl.json`.
* `metrics` is restricted to the contract's ids, so the RTL counts are stated in `fidelity.models` and in a sidecar `E_rtl.json`.

## M2b circuit

* ngspice 42's LTRA line does not support `G` (`Nonzero G (except RG) line not supported`), and the TXL line (`Y`) has no `.ac`. Used LTRA with constant R, G = 0: low-frequency loss is overstated (tap 7 DC -3.61 dB against -2.24 dB at 100 MHz) and the taps carry an excess group delay of 54-110 ps.
* Lines are cut so that delay = k*d*Ts exactly (v_p = 2.00e8 m/s from L and C rather than 1.98e8).
* The default loss compensation is "layer" (one ideal gain trim per layer); "tap" and "none" are diagnostics. At 32 dB/us on the trained width-32 net "none" and "tap" score 0.804 and 0.805 against 0.663 for "layer" (`params/circuit_diagnostics.json`): most of C's headline gap (about 14 of 20 points) is that modelling choice, not line loss. M4d measured why: the layer-3 trim of 2.85 over-drives the short taps and puts 72.8 % of the layer-3 activations into tanh saturation against 49.9 % in V (`circuit_extraction.md`, "Where C's gap comes from"). The circuit tier's default was deliberately not changed, so the published numbers stay as measured.
* Plan acceptance, on a random-init net: unit-cell layer-1 `.tran` NMSE vs V 0.0136 (limit 0.05, **met**), diff-pair tanh fit R^2 = 0.99999997 (limit 0.99, **met**), run time 264 s on a loaded box (limit 10 min, **met**). On the trained width-32 net the stored extraction shows variance-normalised NMSE 0.0220 (16 examples).
* NARMA10 (the reservoir sanity task): the layer-1 unit-cell NMSE exceeds the 0.05 target. The stored extraction of `out/narma10-m4a/params/circuit_extracted.json` shows 0.098 variance-normalised / 0.017 energy-normalised; an earlier relayed figure of 0.116 appears in no artefact and is not used. At the 32 dB/us no-repeater point, NARMA10 tier C NMSE is 1.19 (V 0.1497). The NARMA10 track is a sanity check, not a headline.
* RC pool: tau = N*Ts is a leaky average; it differs from the uniform mean the model was trained with by NMSE 0.026 on the random net (0.216 on the trained width-32 unit cell, `results.md`).

## M2c RF

* Acceptance "30 dB SNR, 0 dB/us, ideal couplers: agreement >= 98 %" see summary row 5.
* Default amplifier model is `tanh`, not Rapp: the model activation is tanh; Rapp (p about 1.31, max deviation 0.021) is implemented and selectable but changes the activation.
* `snr_db` of R is the per-stage amplifier SNR; V, E and C use input AWGN.
* One amplifier gain per layer compensates the mean tap loss (long taps droop); a per-tap trim (`tap_equalize`) is available but optimistic (no repeater noise modelled).
* Sweeps are monotone except two noise-level wiggles recorded in `build-log-M2c.md`.
* Nominal defaults (5 dB/us, 30 dB isolation, 0.5 dB / 5 degrees, Gamma 0.05, -35 dB crosstalk, 30 dB stage SNR, NF 3 dB) were chosen from the plan's typical values and were not tuned.

## M3 LLM

* llama-cpp-python and BitNet: summary rows 3 and 4.
* Several guard checks were added after seeing real output (scoped `metric_id=value` check, unit-bound numbers such as "6-bit", `missing_tier` and `status_claim` consistency checks). Scoring code of the eval was not changed between runs. A tolerance bug (a card integer matching anything within 0.5) was found and fixed.
* The real `modrec` card is 1,453 Qwen tokens (limit 1,500) after a character budget was introduced; the saved `llm/card.txt` of the earlier `modrec-w32-full` run is 4,494 characters (about 1,870 tokens by the approximate counter), above the card limit (M4b finding).
* Plan acceptance (>= 75 % attribution, 0 unverified numbers): 8/8 attribution with the Python-computed divergence list (5/5 injected cards, 2/5 under the strict rule, plus 3/3 controls), 0 unverified numbers after the retry step; **without** that list **0 of 5 injected cards** are attributed correctly (the one "correct" card of 8 is a control with no divergence). The result is real but the card does most of the attribution work (`llm_eval.md`).

## M4a pipeline

* Defaults: `--model` defaults to the width-32 model; a float or missing model is refused before anything is written. `--llm qwen|bitnet` builds the backend before the expensive part.
* Postgres was verified against a local PostgreSQL 16.15 server started for the purpose (apt package; cluster was down; role and database created locally); the test `test_postgres_load_roundtrip` runs only when `DLNN_PG_TEST_DSN` is set. One bug (the DDL splitter broke on a `;` inside a comment) was found and fixed on the way.
* The 25-minute budget is met: width 32, four tiers, cascade, template commentary took 10.2 min (`modrec-w32-full`); with Qwen and Postgres 12.9 min (`modrec-w32-qwen-pg`). The `final-w32` run is in `reproducibility.md`.

## M4b figures and notebooks

* Fixed on the way: figure 4 panel titles printed V's predicted class as the "label"; they now print the true label from `labels.npy`.
* Figure 3's eye compares two different pulse shapes (C trapezoid, R raised cosine); stated on the figure.
* Figure 8 distinguishes `source: template` from model output, dims unverified findings, shows the `ids_verified` state, and appends an independent audit of the text against `compare.json` (M4d: the figure audit now applies the same rules as `docs/results.md`; before M4d it only caught "fallback/stub" wording and showed "all numbers verified" for a run with `ids_verified: false`).
* The notebooks' Markdown prose describes the headline run; with another `DLNN_RUN` the numbers come from that run but some prose still quotes width 32.

## M4c final run and documentation (this milestone)

* `setup.sh` now builds llama-cpp-python straight from the sdist (no wheel attempt) and documents the optional BitNet commands in comments; `pyproject.toml`'s `llm` extra lists `diskcache`.
* Final runs `final-w32` (with Qwen) and `final-w8` (no LLM), `out/LATEST` points at `final-w32`. Figures re-exported and all seven notebooks re-executed against it. See `reproducibility.md`.
* `docs/results.md` is generated by `scripts/make_results_md.py` rather than typed.

## M4d documentation fixes after the independent review (`review.md`)

* Documentation, captions and the generated `results.md` only; no measured number and no tier default was changed, and the headline pipeline was not re-run. The C-tier default stays `layer`.
* New read-only measurement `scripts/saturation_check.py` (results in `docs/saturation_check_w32_n400.json` and `_n4000.json`): the per-layer gain trim over-drives layer 3 of C (and R shows the same layer-3 saturation); documented in `circuit_extraction.md`, `honesty.md`, `results.md` section 2.
* Figure 8 now shows `ids_verified` and an audit block; `docs/figures/` PNG and SVG files are exported with opaque backgrounds (`--no-run-dir` leaves `out/<run>/figures` untouched); figure captions for figs 1, 2, 5 and 7 and the fig 3(c) title were corrected. The `out/final-*/figures` copies written by the pipeline are the older transparent versions with the old captions.
* Notebooks rebuilt and re-executed against `final-w32` (zero error cells).

## Planned, not done

* V "V_cal" cascade consumer; injection of SPICE `tap_H` samples into R; SPICE for layers 2-3; BitNet as a working backend.
