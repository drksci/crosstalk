# Circuit-tier extraction (`params/circuit_extracted.json`, schema `dlnn-circuit-extracted/1`)

Written by `dlnn/tiers/circuit.py` into `<run_dir>/params/circuit_extracted.json` on every C-tier run; read through
`dlnn/cascade.py` (`load_extracted`, `apply_to_rf_cfg`, `line_H`, `loss_db_per_us`).

**Provenance rule.** `source` is `"ngspice"` only if ngspice computed the numbers. If ngspice is missing or fails, the C tier
writes the same schema with `source: "analytic-numpy-rlgc"` (closed-form RLC line, ideal tanh) and is labelled
`status: "fallback"`. Any consumer must propagate that source. All models are GENERIC (textbook RLC coax, textbook NPN):
no vendor model, no measurement of real hardware.

## What is simulated

| Item | Tool | Notes |
|---|---|---|
| per-tap complex `H_k(f)`, `k = 0..K-1` | ngspice `.ac` (LTRA lossy line, 50 ohm load) | tap `k` = coax of delay `k*d*Ts` (`Ts` = 10 ns, line cut so the delay is exact); `H_0 = 1` |
| diff-pair `V_out(v_in)` | ngspice `.dc` | fitted with `A*tanh(v/V_sat) + offset`; reports R^2 |
| layer-1 unit cell end-to-end | ngspice `.tran`, 16 test examples | PAM trains (peak 0.5 V, 10 ns symbols, 1 ns edges), sampled at 0.8 `Ts` |
| everything else (full test set, layers 2-3) | **circuit-calibrated behavioural model** | NOT SPICE |

Line model (`hw/spice/coax_rg58.lib`): L = 250 nH/m, C = 100 pF/m (Z0 = 50 ohm, v_p = 2.00e8 m/s), constant series R chosen for
32 dB/us at 100 MHz (R = 1.8421 ohm/m), G = 0 (LTRA does not support G). No skin-effect sqrt(f) loss/dispersion, so low-frequency
loss is overstated (DC loss of the longest layer-1 tap: -3.6 dB; real RG-58 would be about 0.1 dB).

## JSON layout

```
schema                 "dlnn-circuit-extracted/1"
source                 "ngspice" | "analytic-numpy-rlgc"
generic_model_notice   str
tool                   {name, version}                    # ngspice 42, or numpy (fallback)
ngspice_version        str
netlist_sha256         {coax_rg58.lib, diffpair.lib, tap_lines_ac, diffpair_dc, unit_cell_example0}  # sha256 hex
units                  {...}
symbol_period_s        1e-8
dilation, n_taps       layer-1 d (=1) and K (=8) of the unit cell
timing                 {t_rise_s, sample_phase}           # stimulus edge and sampling instant inside the symbol
line.model             "ltra-rlc-constant-R" | "analytic-rlc-constant-R"
line.R_ohm_per_m_nominal / L_h_per_m_nominal / C_f_per_m_nominal / G_s_per_m_nominal / loss_db_per_us_nominal_100mhz
line.fit               closed-form terminated-line fit to the SPICE H_k(f):
                         R_ohm_per_m, L_h_per_m, C_f_per_m, G_s_per_m, z0_ohm, vp_m_per_s, termination_ohm,
                         loss_db_per_us_100mhz, fit_rms_abs_err, fit_max_abs_err
line.omits             [str]
taps                   k[K], delay_nominal_s[K], length_m[K], dc_gain_db[K], gain_100mhz_db[K], delay_error_s[K]
                       (excess group delay over 5-60 MHz, positive = late), freq_hz[F] (1 Hz stands for DC, then
                       linear to 500 MHz, 1025 points), H_re[K][F], H_im[K][F]   (H = V_tap / V_source)
diffpair               A_v, vsat_v, offset_v, r2, rms_resid_v, iee_a, rl_ohm, vsat_design_v (= 2*25.9 mV),
                       vsat_scale (= vsat_v / vsat_design_v), gain_scale, offset_rel, v_t_fitted_v, v_t_ngspice_27c_v
amp                    {vsat_scale, gain_scale, offset_rel}   # what the behavioural model multiplies V_sat / gain by
loss_comp, loss_comp_gain_trim_per_layer   loss-compensation mode ("none"/"layer"/"tap") and the per-layer gain trim (1 / |W|-weighted mean main-cursor tap gain) used by the cell and the model
unit_cell_tran         null (no .tran) or {nmse, nmse_energy_normalised, nmse_per_channel, nmse_calibrated_model_vs_spice,
                       pool_*, spice_wall_s, no_trim: {...same without AGC trim...}, ...}
```

`apply_to_rf_cfg(cfg, extracted)` returns a copy of the RF cfg with `cfg["cascade_circuit"]` (`source`, `tool`, `loss_db_per_us`,
`line_rlc`, `tap_H{freq_hz,H,delay_s}`, `amp{vsat_scale,gain_scale,offset_rel,...}`, `symbol_period_s`), convenience scalars
`loss_db_per_us`, `amp_vsat_scale`, `amp_gain_scale`, and `inputs_from += ["params/circuit_extracted.json"]`.
`cascade.line_H(extracted, freqs, delay_s)` evaluates the FITTED RLC line for any tap delay (layers 2-3 have delays up to 1.12 us;
those long-line responses are an extrapolation of the layer-1 fit, not SPICE).

## Behavioural model (what the full-set numbers are)

For a layer with taps `W[o,i,k]`, dilation `d`: tap response `s_k[m]` = response of the fitted line (delay `k*d*Ts`) to one
trapezoidal symbol, sampled at `(m + 0.8) Ts` (FFT on a 1 ns grid, 4096 points). Then
`u[o,n] = trim * sum_{i,k,m} W[o,i,k] s_k[m] x[i, n-m] + bias[o]` and
`y = gain[o] * (gain_scale * tanh(u / (vsat * vsat_scale)) + offset_rel)`; layer outputs are sample-and-held between layers.
Loss sweep: R is scaled so the 100 MHz loss equals 0/5/14/32 dB/us, with the ideal per-layer gain trim re-applied.

## Where C's gap comes from: the per-layer gain trim, measured (M4d)

`--cascade` runs C at 32 dB/us, where C scores 0.663 against V's 0.863. Two things are in that 20-point gap and they are not equal.
The diagnostics in `params/circuit_diagnostics.json` show the loss-compensation mode matters more than the loss: at the same 32 dB/us the
default `layer` mode (one ideal gain trim per layer) scores 0.663, `none` (no compensation at all) 0.804 and `tap` (ideal per-tap repeater, every tap's
main cursor restored to 1) 0.805. So about 14 of the 20 points come from the default trim and about 6 from the line loss itself.
Ideal per-tap repeaters gain 0.1 point over doing nothing.

**Mechanism.** `cursor_trim` sets one scalar per layer, 1 / (|W|-weighted mean main-cursor gain of the taps). In layer 3 the tap main cursors run from 1.000
(k = 0, no line) down to 0.0169 (k = 7, -35.4 dB), so the weighted mean is only 0.35 (pulled down by the weak long taps) and the trim is 2.85 (layers 1 and 2: 1.13 and 1.51).
It multiplies the short, almost unattenuated taps by 2.85 as well, so their contribution to the tap sum is 2.85 times what the trained net expects and the
layer-3 tanh is driven well into saturation.

**Method** (`scripts/saturation_check.py`, read-only, about 10 s, nothing in `out/` is written). It uses the tier's own forward code
(`circuit.cursor_trim`, `effective_kernel`, `_conv_fft`, the fitted line and diff-pair parameters of `out/final-w32/params/circuit_extracted.json`;
for R `rf.build_channels`, `fir_apply`, `amplifier`, with the cascade configuration applied the way the tier applies it) and records the argument of the tanh,
`z = (tap sum + bias) / (V_sat * vsat_scale)`, for every (example, output line, time step) of each layer. "Saturated" means |tanh(z)| > 0.95, i.e. |z| > 1.832.
The first 400 test examples were used; the same script on all 4000 reproduces the tier's own accuracies exactly (C `layer` 0.66275, `none` 0.80425, `tap` 0.80525; R 0.70125), which validates the re-implementation.
Results are in `docs/saturation_check_w32_n400.json` and `docs/saturation_check_w32_n4000.json`.

Fraction of tanh inputs in saturation, width-32 net, 32 dB/us (400 examples; 4000 in brackets for layer 3):

| model | layer 1 | layer 2 | layer 3 | accuracy (400) |
|---|---|---|---|---|
| V (ideal) | 0.522 | 0.796 | **0.495** (0.499) | 0.8625 |
| C, `layer` (tier default) | 0.512 | 0.792 | **0.724** (0.728) | 0.6575 |
| C, `none` | 0.467 | 0.686 | **0.307** (0.311) | 0.7625 |
| C, `tap` | 0.508 | 0.782 | **0.498** (0.501) | 0.7900 |
| R, `--cascade` (32 dB/us, tier default gain rule) | 0.533 | 0.828 | **0.713** (0.717) | 0.675 |
| R, `--cascade` with `tap_equalize` | 0.533 | 0.822 | **0.613** (0.617) | 0.800 |
| R, standalone defaults (5 dB/us) | 0.525 | 0.803 | **0.544** (0.549) | 0.8475 |

Reading it: V itself saturates about half of layer 3. C `layer` pushes layer 3 to 72 %, 23 points above V, while the
`tap` mode reproduces V's 50 % almost exactly. `none` under-drives layer 3 (31 %) but still classifies better than `layer`, because the loss costs less than the over-drive does.
Layers 1 and 2 are essentially the same in `layer` and `tap`; the damage is a layer-3 effect.

**R.** Yes, it can be checked cheaply and it shows the same thing: R's per-layer rule (`comp = 1 / mean_main`, an |W|-count-weighted mean of the main-lobe gains) also leaves 71 % of layer 3 saturated at the
cascade loss (V 50 %), against 54 % at R's standalone 5 dB/us. With R's own per-tap equaliser (`tap_equalize=True`, optimistic: no repeater noise is modelled) layer-3 saturation drops to 61 % and
accuracy at 32 dB/us rises from 0.701 to 0.810 on the full set (0.675 to 0.800 on the 400). R's equalised 0.810 is close to C's `tap` 0.805.
This measures that R shares the over-driven layer 3 and that R's accuracy at 32 dB/us depends on the gain rule; it does not show whether R's rule is an artefact of the model or what a real amplifier chain
would do, and `tap_equalize` also changes the relative tap weights, so the 0.81 is not attributed to layer-3 saturation alone.

**What this does not settle.** All of this is the calibrated behavioural model (numpy), not SPICE. Whether a real build would use one gain per layer, per-tap repeaters or something else is a design
question these runs do not answer; no repeater noise is modelled in `tap` or in `tap_equalize`, so 0.805 / 0.810 are optimistic. The C tier default was not changed and the published C number is the
`layer` one, 0.663. Both facts belong in any statement of the headline: the cascade operating point is not buildable without repeaters, and the default compensation mode over-drives layer 3.

---

# Board mode (M2a): ngspice on the KiCad-exported TILE-8 netlist

Code: `dlnn/tiers/circuit_board.py` (the coax path above is untouched and is now the "coax study"). Parts: `hw/spice/parts/`.
Results: `out/P2-M2a/{aware2_fro,plain}/result.json` (+ `progress.log`, per-batch SPICE checkpoints). Browser files: `docs/artifacts/data/spice/`.
Everything is simulation of a design that is not built; nothing is measured.

## What SPICE computes, what the calibrated model computes

| Item | Tool | Notes |
|---|---|---|
| netlist | `gen_tile.py --layer L --model M` + `kicad-cli sch export netlist --format spice` | re-exported on every run; sha256 of each `tile.cir`, the schematic sheets, the model file and `gen_tile.py` are stored in `export.json` / `result.json` |
| per-tap transfer | `.ac` of the whole tile (line 0 driven), taps `T0_0..T0_7` | input RC, buffer, 56 LC sections, tap buffers |
| diff-pair curve | `.dc` of the whole bias DAC -> summer -> BJT pair (mirror tail) -> difference amp -> S/H -> buffer chain, all 8 channels | fitted per channel with `A tanh((V_dac - s)/vs) + c` (R^2 >= 0.9999999) |
| full tile | `.tran` of 3 tiles in a chain (layer 1 -> S/H samples -> layer 2 -> S/H samples -> layer 3), 64 stratified test examples x 3 layers at Q = 10, 16 at Q = 30 | about 20 s per example (3 tiles) |
| Monte Carlo | `agauss` on every inductor L and ladder C (sigma = 5 %/sqrt(3)), 10 runs, DAC re-trimmed and clock re-calibrated per draw | 2 chain examples per draw |
| temperature | 0 / 25 / 50 C, DAC trim and clock frozen at 25 C | DCR (copper tempco), 4066 R_on, BJT via `.temp` |
| 4000-example test set | **circuit-calibrated behavioural model (NOT SPICE)** | tap FIR from `.ac` (ZOH launch, IFFT), per-channel tanh from `.dc`, post-chain delay/S-H memory fitted to 4 TRAIN-set calibration transients |

Calibration steps a bring-up MCU would also do (all derived from SPICE, none from the test set): DAC trim of every channel so the zero-input output equals
`gain*tanh(bias/vsat)` (`.dc` inversion, 12 bit, no channel clipped); symbol clock T and tap-sampling phase chosen to maximise train-subset accuracy of the behavioural model.

## Part models (`hw/spice/parts/`)

| Part | Model | Provenance |
|---|---|---|
| FH FHW1812IF121JST inductor | L 120 uH (agauss in MC) + DCR 3.3 ohm + C_p 19.38 pF (SRF 3.3 MHz) + R_p 6.35 kohm (Q = 10 @ 796 kHz, datasheet MINIMUM) / 21.56 kohm (Q = 30, ASSUMED typical) | datasheet-derived lumped model, not a vendor model |
| TLV9064 / 9062 op-amp | **full-tile runs: single-pole fit** (A0 1e5, GBW 10 MHz, R_out 100 ohm, 6 pF on in+, no clamp, no noise, no slew limit) | labelled FIT, `opa_quad_fit.lib`. The TI PSpice macro (`opa_quad_ti.lib`, downloaded at run time) is supported by the deck builder but its operating point does not converge in useful time with more than about 16 macros (measured: 4 / 8 / 16 macros 0.6 / 1.1 / 5.5 s; one line plus 8 tap buffers 36 s; the full tile never) |
| BC847BS / BCM847BS | generic NPN, IS fitted to V_BE(on) 665 mV @ 2 mA, BF 300, C_cbo 2 pF, C_ebo 11 pF; VAF, RB, TF assumed | labelled generic (same model as M1a) |
| 74HCT4066 | `SW` with R_on 95 ohm @ 25 C (+0.383 ohm/K, 118 ohm @ 85 C), C_off 5 pF per terminal | R_on from the TI HC4066 datasheet (max), C_off ASSUMED; no charge injection |
| MCP4728, LM4040 | ideal sources (levels set per run) | |

## Findings from the real netlist (these change what C measures)

1. **DC level of the delay lines (ECO-1).** Each line is terminated `RT{i}` to GND. The tap nodes therefore sit at about 1.25 V while the summers' virtual ground is 2.5 V,
   so every populated crossbar resistor injects about 125 uA of DC: one summer output was 2.4 V off, 46 vsat, far outside what a 0..2.048 V bias DAC can trim.
   The run applies `RT{i}` -> `VREF25` (one net per line). The committed schematic does not have this.
2. **Tap-gain compensation (ECO-2).** `derive_params.layer_params` sizes `R_f = R_in 2 V_T/(A_in vsat)` as if the tap voltage were `A_in*x`. In the drawn board
   (buffers gain 1, matched ladder) the tap DC gain is 0.525 (tap 0) .. 0.456 (tap 7) (-5.6 .. -6.8 dB), and with the 482 kHz input RC and the line's dispersion the
   symbol-sampled on-lag gain is 0.36 .. 0.28 (mean 0.33, -9.6 dB; Q = 10). The nets trained on the M1a line FIR (`aware2*`) assume a mean on-lag gain of 1.
   Behavioural-model accuracy of aware2_fro vs the R_f/R_b/R_o multiplier (train subset, 1500 examples): x2.0 0.30, x2.5 0.71, **x2.9 0.84**, x3.3 0.86 (consistent with the P tier: 0.87 with ~9 dB, 0.17 with none).
   The board as drawn has NO compensation. Fix: multiply R_f, R_b and R_o of every layer by 2.9 (about +9.2 dB; layer R_f 768/365/976 -> about 2.2k/1.07k/2.8k, R_b 5.23k/3.16k/7.87k -> 15.2k/9.2k/22.8k, R_o the same factor so the DAC centre stays 1.024 V), or give the 64 tap buffers a gain of 2.9. The plain net (ideal-line assumption) needs only x2.
3. The M1a line FIR (`line_fir_m1a.json`) sums to 1.4-1.5 for taps 1-7 against 0.96 for tap 0, whereas the DC gains of the real tile are flat (0.46-0.53). The aware2 nets were trained on a relative tap-0 weakness the netlist does not show; this is one reason why C and V-on-FIR still differ by 2-3 points (below).
4. Post-chain (summer, pair, difference amp, 4066, buffer) lag: the behavioural model needs a 0.35-0.5 us delay between tap sampling and the S/H clock edge; the S/H clock phase must be set to `t0_eff + tau_post`.

## Results (see `out/P2-M2a/*/result.json`)

Numbers, model aware2_fro (headline, sha256 `3405923d...`, ECO-1 and R x2.9) unless stated. V = ideal tier V; V(FIR) = V on the M1a Q = 10 / 30 line FIR (the fair reference for a board-trained net).

| | Q = 10 | Q = 30 |
|---|---|---|
| SPICE examples per layer | 64 | 16 |
| NMSE calibrated vs SPICE, layers 1 / 2 / 3 (layer driven by the SPICE input) | 0.0147 / 0.0134 / **0.0266** | 0.0042 / 0.0042 / 0.0112 |
| NMSE SPICE vs V (chain) | 0.090 / 0.182 / 0.465 | 0.056 / 0.094 / 0.272 |
| NMSE SPICE vs V(FIR) | 0.041 / 0.138 / 0.352 | 0.012 / 0.035 / 0.146 |
| C accuracy, 4000 (calibrated) | 0.8505 | 0.8588 |
| V / V(FIR), 4000 | 0.655 / 0.878 | 0.655 / 0.878 |
| C - V, C - V(FIR) (points) | +19.6 / -2.8 | +20.4 / -1.9 |
| accuracy on the SPICE subset: SPICE / calibrated / V(FIR) | 0.813 / 0.797 / 0.844 (n = 64) | 0.875 / 0.875 / 0.875 (n = 16) |

Layer 3 at Q = 10 misses the 0.02 NMSE target: the residual sits on a few saturated steps (outputs flipping from +0.99 to -0.7 between samples) whose timing the linear FIR + tanh model cannot follow. The Q = 30 corner meets it.

Plain model `tdlnet_b_w8_ternary` (ECO-1, R x2.0, SPICE on 8 examples only, calibrated model on 4000): C 0.7895 (Q10) / 0.7853 (Q30) against V 0.8055: **C - V = -1.6 / -2.0 points** (acceptance <= 5: met); NMSE calibrated vs SPICE 0.0036 / 0.0027 / 0.0046 (Q10). One layer-3 channel could not be DAC-trimmed to its zero-input target.
`aware2_ternary` was not run through SPICE (scope).

Monte Carlo (10 draws, L and C): accuracy 0.8542 +- 0.0042 (min 0.8488, max 0.8625), clock T 0.946 +- 0.008 us. Temperature (DAC trim frozen at 25 C): 0.8533 / 0.8528 / 0.8490 at 0 / 25 / 50 C.
Rail check: all S/H outputs stay within 1.0-3.0 V.

Limits: no layout parasitics, no mutual inductance or crosstalk, no noise, no ribbon (tiles hand over ideal ZOH symbols at the S/H sampling instant); op-amp is a fit, not TI's macro; Q_typ = 30 and C_off are assumptions; L/C Monte Carlo sigma matches a +-5 % uniform band; the temperature coefficient of the ferrite inductance is not modelled. Parasitics and tolerance belong to the P tier.

## Notebook API (`dlnn.tiers.circuit_board`; every function takes its parameters and an output directory, no module state)

| Call | What | Time |
|---|---|---|
| `export_netlist(model_npz, layer, outdir) -> Path` | `gen_tile.py --model` + `kicad-cli` for one layer | ~5 s |
| `calibrate(model_npz, outdir, q=10.0, temp_c=25.0, T=None, rf_scale=2.0) -> {'setups','summary','info'}` | `.ac` + `.dc` of the three tiles, FIR, amplifier fits, DAC trim, writes `calibration_Q<q>.json` | ~20-30 s |
| `run_spice(setups, model, examples[n,2,N], outdir) -> {'y': [3 x (n,8,N)], ...}` | whole-tile `.tran` chain | ~20 s per example |
| `export_web(outdir, model_npz, run_id, rf_scale=2.0) -> manifest` | `unit_cell.template.cir`, `tap_response.json`, `diffpair_dc.json`, `manifest.json` (sha256) | ~1 min |
| `run_model(model_path, outroot, n_test, corners, n_mc, rf_scale, ...)` | everything above, resumable | ~50 min for 64 examples |
| `cal_forward(model, cal_models(setups), x)`, `v_forward_fir(model, x, fir=...)` | behavioural model and V on a line FIR | seconds |
| `run(model_npz, dataset, cfg, outdir)` | contract tier entry point (`TierResult`, validates) | ~1 min |

CLI: `.venv/bin/python -m dlnn.tiers.circuit_board --model models/tdlnet_b_w8_aware2_fro_ternary.npz --out out/P2-M2a/aware2_fro --n-test 64 --mc 10 --rf-scale 2.9`.
Use `rf_scale=2.9` for the M1a-FIR board nets and 2.0 for the plain net.
