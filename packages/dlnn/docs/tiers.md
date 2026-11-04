# The four tiers: what each one is, and what it is not

Everything here is **simulation on synthetic data**. No hardware was measured. All circuit and RF parameters are generic
textbook models (RG-58-like coax, textbook NPN diff pair), not vendor data. Status values and labels in the JSON are the
source of truth: a tier is `ok` only when the real tool ran; `fallback` and `stub` are never presented as the real tool.

| Tier | Tool actually used | Is it the real tool? | What is computed by it | What is calibrated / behavioural | What is not computed |
|---|---|---|---|---|---|
| **V** virtual | numpy float32 | n/a (it is the math) | ideal unit delays, exact ternary sums, exact tanh | nothing | every physical effect |
| **E** emulated | numpy bit-true fixed-point golden model + **Icarus Verilog 12** RTL (`hw/rtl/tdl_layer.v`) | yes, but RTL covers a subset | RTL: the 3 layers on the first 64 test examples, bit-compared with the golden model (0 mismatches required for `status=ok`), plus layer 1 at ADC widths 4-10 on 16 examples | the full test set and all sweeps run on the **numpy** golden model, not on the RTL | all analog effects; ADC quantiser, pooled-sum conversion and float readout are numpy in both paths |
| **C** circuit | **ngspice 42** on a layer-1 *unit cell* + circuit-calibrated behavioural model | partly | SPICE: per-tap `.ac` transfer functions, diff-pair `.dc` curve, `.tran` of the layer-1 unit cell on 16 examples | **everything on the 4000-example test set and layers 2-3 is the circuit-calibrated behavioural model (NOT SPICE)**: a closed-form RLC line fitted to the SPICE `H_k(f)` and a fitted `A*tanh(v/V_sat)+offset` | full network end-to-end in SPICE, layout parasitics, frequency-dependent R / skin effect (LTRA constant-R only), bias drift, noise |
| **R** RF | **scikit-rf 2.2** two-port cascades -> IFFT impulse responses -> numpy convolution | yes (channel level) | `skrf.media.Coaxial` lossy dispersive line, coupler, hybrid, Wilkinson combiner, mismatch, per tap path | amplifier (memoryless tanh), noise (white, one source per stage), crosstalk (flat, neighbour only), ideal re-timing | EM / layout / connectors, transistor-level amplifier, AM/PM, bandwidth limit, jitter, thermal/supply effects |

"Real tool" statements apply to what the tier JSON says: `tool.name`, `status`, `fidelity.models`, `fidelity.omits`. If
ngspice is missing or fails, C switches to a closed-form numpy line and says `status: fallback`; if scikit-rf is missing, R
does the same; if the RTL does not match the golden model or iverilog is missing, E is `fallback` ("bit-true model (no RTL)").

## V, virtual (`dlnn/tiers/virtual.py`)

Layer `y[o,n] = gain[o] * tanh((sum_{i,k} W[o,i,k] x[i,n-k*d] + bias[o]) / vsat)`, causal, zeros before `n = 0`; global
time-mean; linear readout; softmax. It reproduces the torch eval of the trained model to 2e-6 (logits). It is the reference
for every "agreement", "KL" and "delta vs V" number. `snr_db` sweep = AWGN added to the input at the stated SNR *on top of*
the dataset's own noise.

## E, emulated (`dlnn/tiers/emulated.py`, `hw/rtl/`, `docs/fixed_point_spec.md`)

8-bit absmax ADC (full scale = the fixed value stored in the model, `input_fullscale`, the train-set abs-max rounded up to 0.5; abs-max of the batch if absent), int2
ternary taps, int32 accumulate, Q0.15 scale, 256-entry tanh LUT -> int8, per-line Q.15 gain, integer pool. The RTL and the golden
model are bit-identical on 64 examples x 3 layers. Sweeps: `snr_db` (input AWGN), `adc_bits` 4-10. Not modelled: anything analog.

## C, circuit (`dlnn/tiers/circuit.py`, `hw/spice/`, `docs/circuit_extraction.md`)

* SPICE-computed: per-tap complex `H_k(f)` (`.ac`, LTRA lossy line, 50 ohm load), diff-pair `V_out(v_in)` (`.dc`, fitted with `tanh`),
  and the layer-1 unit cell `.tran` on 16 test examples (8 outputs each) compared with V.
* Calibrated behavioural: the closed-form RLC line fitted to the SPICE `H_k(f)` extrapolated to the long layer-2/3 taps, the fitted
  amplifier curve; the 4000-example numbers come from this, **not from SPICE**. Its agreement with SPICE traces is reported
  (`unit_cell_tran.nmse_calibrated_model_vs_spice`).
* Loss compensation: one ideal gain trim per layer (default `layer`; `none` and `tap` = ideal per-tap repeater are diagnostics in `params/circuit_diagnostics.json`).
  At RG-58 loss (32 dB/us) the default `layer` mode scores 0.663, `none` 0.804 and `tap` 0.805 (V 0.863): per-tap repeaters gain 0.1 point over doing nothing, so what hurts C at the headline is mostly the
  per-layer trim over-driving the short layer-3 taps (trim 2.85, 72.8 % of layer-3 activations saturated against 49.9 % in V; `circuit_extraction.md`, "Where C's gap comes from"), and only about 6 of the 20 points are line loss.
  The plan's statement that the long taps "need re-amplification" still holds as a statement about buildability; it is not what the C sweeps show as the dominant cause.
* LTRA does not support `G`, so low-frequency loss is overstated (DC loss of the longest layer-1 tap -3.6 dB vs -2.2 dB at 100 MHz).
* The RC pool in SPICE is a leaky average (tau = N*Ts), which differs from the uniform mean the model was trained with.
* Writes `params/circuit_extracted.json` (`source: "ngspice"` or `"analytic-numpy-rlgc"`).

### C, board mode (M2a, `dlnn/tiers/circuit_board.py`, details in `docs/circuit_extraction.md`)

* SPICE-computed: ngspice 42 on the KiCad-exported TILE-8 netlist of each layer (`gen_tile.py --model`, `kicad-cli` spice export, sha256 recorded): `.ac` per tap, `.dc` of the bias -> pair -> S/H chain,
  and the whole-tile `.tran` of the 3-tile chain on 64 stratified test examples per layer (layer 1 -> S/H samples -> layer 2 -> S/H samples -> layer 3; Q = 10; 16 examples at Q = 30), 10-run L/C Monte Carlo, 0/25/50 C.
* Calibrated behavioural (NOT SPICE): tap FIR from `.ac`, per-channel tanh from `.dc`, post-chain delay fitted to 4 train-set transients; runs the 4000-example set. Per-layer NMSE vs SPICE (aware2_fro, Q10) 0.015 / 0.013 / 0.027.
* Parts: inductor L + DCR + C_p + R_p (datasheet-derived, Q 10 / 30 corners); op-amp = single-pole TLV906x FIT (TI macro does not converge at tile scale); generic BC847 NPN; 4066 = R_on + C_off.
* The exported schematic needs two changes to work (ECO-1 line termination to VREF25, ECO-2 R_f/R_b/R_o x2.9 for the M1a-FIR nets); the SPICE runs apply them. Headline: aware2_fro C 0.8505 (Q10) / 0.8588 (Q30) vs V 0.655 and V on the M1a FIR 0.878; plain C 0.7895 / 0.7853 vs V 0.8055.
* Omits: layout parasitics, crosstalk, mutual inductance, noise, ribbon (that is tier P).

## R, RF (`dlnn/tiers/rf.py`, `docs/rf_model.md`)

Channel-level: each tap path is a scikit-rf cascade (coupler, mismatch junction, coax of length `k*d*Ts*v_p`, junction,
Wilkinson branch); impulse responses are run by FFT convolution over the whole set; stage noise, amplifier and 10-bit ADC in numpy.
Sweeps: `snr_db` (**per-stage amplifier SNR**, not input AWGN: the one sweep whose meaning differs from V/E; documented, kept),
`loss_db_per_us` 0/5/14/32, `isolation_db` 15-40, `tap_amp_err_db` 0-2, `adc_bits` 4-10.
Defaults (5 dB/us, 30 dB isolation, 0.5 dB / 5 deg tap errors, Gamma 0.05, -35 dB crosstalk, 30 dB stage SNR, 3 dB NF) were chosen
from the plan's typical values and are not tuned.

### Cascade (`--cascade`)

`C` runs before `R`, writes `params/circuit_extracted.json`; `R` reads it through `dlnn.cascade` (`apply_to_rf_cfg`): the **fitted
line loss** (about 32 dB/us, the RG-58 figure that was put into the SPICE line) replaces R's default 5 dB/us, and the fitted amplifier
`V_sat` / gain / offset scales are applied. The SPICE `tap_H` samples are NOT injected (they exist for the d=1 unit cell only).
`R.inputs_from = ["params/circuit_extracted.json"]` and the extraction `source` is carried in R's warnings. The plan's second cascade
consumer, a "V_cal" sensitivity check in tier V, is **not implemented**. The fitted loss equals the number that was put into the netlist (32.000005 vs 32.0 dB/us), so the cascade demonstrates plumbing and provenance,
not independent information. Consequence for reading the tables: with `--cascade` the R headline sits at 32 dB/us with no repeaters (a far harsher operating point than R's standalone defaults), and R's per-layer gain rule leaves layer 3
largely saturated there (71 % of activations against 50 % in V; with `tap_equalize` R scores 0.810 instead of 0.701; `circuit_extraction.md`). R's `loss_db_per_us`
sweep shows both operating points either way.

## Definitions that matter when reading the numbers

* **top1_agreement_vs_virtual**: fraction of test examples whose argmax equals V's argmax on the same example.
* **mean_kl_virtual_vs_tier**: mean over examples of KL(softmax(V) || softmax(tier)) in nats. Sweep `mean_kl` is KL(V on the clean input || tier at the swept value).
* **Sweep knee**: first value, walking from the best end of the axis, where accuracy is >= 0.05 below its best-end value.
* **Subsets**: when tiers C and R run on a subset (see the run's `compare.json` `run_meta`/`paired_vs_virtual`), the subset is a fixed,
  (class, SNR)-balanced selection, identical for C and R and nested in V's and E's test set. `deltas_vs_virtual` for accuracy,
  macro-F1 and accuracy per SNR is then computed against V restricted to the same examples (`paired_vs_virtual`); agreement and KL
  are always per example against V on the same example. Sweep curves of a subset tier are on the subset.
* **Float baseline**: `compare.json["reference"]`, produced by the V code path on the float32 model of identical topology. It is a
  reference, not a fifth tier: it is not ternary and cannot run through E, C or R.

## Criterion note: R agreement at 30 dB (M2c), re-measured on the trained nets (M4a)

The M2c acceptance text "at 30 dB SNR, 0 dB/us, ideal couplers: top-1 agreement vs V >= 98 %" is ambiguous about where the 30 dB is
injected. R's `snr_db` is the **per-stage** SNR (three independent noise sources, one per layer's amplifier input), whereas V and E
inject 30 dB at the **input** only. Definition used for the measurement below: `rf.forward` with `preset="ideal"` (0 dB/us, infinite
isolation, no amplitude/phase error, no mismatch or crosstalk, NF 0 dB, ideal ADC), per-stage SNR 30 dB, 4000 test examples, 3 noise seeds.

| model | R ideal, noiseless | R ideal, 40 dB per stage | **R ideal, 30 dB per stage** | V(30 dB input AWGN) vs V(clean) |
|---|---|---|---|---|
| width-32 ternary (headline) | 1.0000 | 0.9955 | **0.9863** (0.9873 / 0.9855 / 0.9860) | 0.9958 |
| width-8 ternary | 1.0000 | 0.9955 | **0.9809** (0.9828 / 0.9778 / 0.9823) | 0.9895 |

Both trained nets meet >= 98 % under the per-stage reading (random-init nets did not: 93.6 %; their logit margins were small).
The code was not changed to achieve this; the criterion is simply met on the trained nets. This is the *ideal-channel* number: the
R tier's headline operating point (5 dB/us, 30 dB isolation, 0.5 dB / 5 deg errors, mismatch, crosstalk; or 32 dB/us with `--cascade`)
is much harsher and agrees with V far less (see the build log). Script: `scripts/r_ideal30.py` (`.venv/bin/python scripts/r_ideal30.py models/tdlnet_modrec_w32_ternary.npz models/tdlnet_modrec_ternary.npz`, about 1 min).

## Extra blocks in `compare.json` (M4a; schema `dlnn-compare/1`, all optional)

`reference` (float32 baseline, `is_tier: false`), `paired_vs_virtual` (only when C/R ran on a subset), `timing` (seconds per
pipeline stage and total), `run_meta` (model file, subset size, cascade flag, load average). `deltas_vs_virtual` for accuracy, macro-F1
and accuracy-per-SNR of a subset tier uses V restricted to the same examples.

## What none of this licenses

Claims allowed: simulation-only cross-fidelity consistency and degradation curves of a small ternary delay-line network on synthetic
data. Not allowed: any hardware measurement, vendor-accurate SPICE models, EM/layout effects, "the LLM analysed", competitiveness
with digital classifiers, buildability of the full network at RG-58 loss without repeaters.
