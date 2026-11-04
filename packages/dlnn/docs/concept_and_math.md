# Concept and math: a ternary delay-line neural network

This is a **simulation study**. Nothing here was built or measured. Component values (RG-58-like coax, a textbook NPN differential pair)
are generic textbook models. See `honesty.md` for what the study does and does not show.

## 1. The idea in one paragraph

A tapped delay line computes a convolution: the signal travels down a cable, couplers sample it at several lengths, and a combiner adds the
samples. If each tap is one of "+1 (couple straight), 0 (no coupler), -1 (couple through a 180 degree hybrid)", the sum is exactly a
convolution with a **ternary** kernel, the same weight alphabet as BitNet b1.58. A saturating amplifier after the combiner, built from a
bipolar differential pair whose transfer curve is a tanh, supplies the nonlinearity. Stack three such layers with growing tap spacing
(dilation) and you have a small dilated 1-D convolutional network whose multiplications are cable lengths and coupler polarities. The demo trains
such a network (`TDL-Net`) in PyTorch on synthetic modulation recognition, then evaluates the same trained weights through four simulation
fidelities: V (ideal math), E (fixed-point and RTL), C (SPICE circuit), R (S-parameter RF channel).

## 2. The network (as trained and simulated)

Symbol period `T_s` (10 ns in all circuit and RF simulations). Layer `l` has input lines `i`, output lines `o`, `K = 8` taps, dilation `d_l`
(1, 4, 16 for layers 1-3):

```
u_o[n]  = sum_i sum_{k=0..K-1}  W[o,i,k] * x_i[n - k*d_l]          W in {-1, 0, +1}
y_o[n]  = g_o * tanh( (u_o[n] + b_o) / V_sat )                     g_o = per-line gain (1 in the shipped models), b_o = bias
```

Causal; the signal is zero before `n = 0`. After layer 3 the network takes the time mean of each line (global average pool), applies a linear readout
(ternary weights times scalars, plus a per-class bias) and a softmax over the 8 classes.

* Receptive field of the three layers: `1 + 7*(1 + 4 + 16) = 148` samples, which covers the 128-sample input.
* Two shipped sizes, same topology and code, differing only in width (lines per hidden layer): **width 8** (the plan's specification, 1,152 ternary
  conv weights, or 1,216 with the 8x8 readout) and **width 32** (16,896 ternary conv weights, or 17,152 with the 8x32 readout; the headline model; `models/train_report_w32.json` counts the conv layers only).
* Training (offline, PyTorch): ternary quantisation-aware training with the BitNet b1.58 rule `W_t = clip(round(W/gamma), -1, 1)`,
  `gamma = mean|W|` per layer, straight-through estimator, 8-bit activation fake-quantisation, progressive ternarisation. Details and the full
  list of what was tried: `build-log-M1.md`, `model_export.md`.
* Export (`dlnn-model/1`): `gamma` and a learnable amplifier gain become the input-referred gain `1/V_sat` of layer `l`; the biases become DC offsets in the units of the tap sum.

## 3. Math to hardware: tap polarity and the ternary alphabet

| Math | Physical element | Note |
|---|---|---|
| delay `k * d_l * T_s` | coax length `k * d_l * T_s * v_p` | RG-58-like coax: `v_p` = 19.8 cm/ns, so one 10 ns symbol is 1.98 m; dilation 16 means 31.7 m per tap step |
| `W = +1` | directional coupler into the combiner | |
| `W = 0` | no coupler at that tap | a real coupler leaks (finite isolation); R models the leak |
| `W = -1` | coupler through a 180 degree hybrid | sign flip, ideal and matched in R |
| `sum over (i, k)` | N-way Wilkinson combiner | passive, `-10 log10(N)` dB; `N = I*K` ports |
| `tanh` | BJT differential pair, `I = I_EE tanh(v / 2 V_T)` | `V_T` = 25.9 mV at room temperature; tier C fits `A tanh(v/V_sat) + offset` to an ngspice `.dc` sweep |
| `g_l`, `V_sat`, biases | amplifier gain and bias point | compensates the passive loss; adds noise (NF) |
| global average pool | RC integrator, tau about `N * T_s` | passive; in the unit cell it is a leaky average and differs from the uniform mean the net was trained with |
| readout + softmax | ADC and software | honestly digital |

Longest taps (arithmetic from the table, `K=8`, `T_s=10 ns`, `v_p=19.8 cm/ns`): layer 1 `7*1*10 ns = 70 ns` (about 13.9 m), layer 2 `280 ns` (about 55 m),
layer 3 `1.12 microseconds` (about 222 m). The networks have no multipliers: every product is a sign selection, every sum is analog addition.

## 4. Signal encoding

Baseband pulse-amplitude modulation, no carrier. Each of the two input lines (I, Q) carries one sample per symbol period as a pulse of amplitude proportional to the sample
(plan: about +-0.5 V, 10 ns symbols, roughly 100 MHz bandwidth). Between layers the amplifier output is re-timed (sampled at the symbol instant) and a fresh
pulse train drives the next layer's lines. In R this re-timing and pulse regeneration are ideal; in C they are sample-and-hold in the behavioural model. The input is
the synthetic complex baseband of one of eight modulations (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK): 128 samples at 8 samples per symbol, random carrier phase,
carrier frequency offset up to 1 % of the sample rate, timing offset, AWGN at 0, 5, 10, 15, 20 dB SNR. 16,000 training and 4,000 test examples are generated in-repo
(seeds in `reproducibility.md`).

## 5. Physical limits (what the simulations parametrise)

All values are the plan's typical figures or the simulation's default parameters, not measurements.

| Limit | Plan / model value | Where it shows up in the demo |
|---|---|---|
| Cable loss | RG-58 about 32 dB per microsecond at 100 MHz; LMR-400 about 14 (the plan's figure; a datasheet-based estimate would be nearer 10-11, not checked here and not used by any result); fibre about 0. The sweep axis is "dB per microsecond" so results do not depend on the medium | the longest layer-3 tap (1.12 microseconds) is about 36 dB down at 32 dB/us; the `loss_db_per_us` sweeps of C and R; `--cascade` runs C and R at 32 dB/us with no repeaters (only part of the resulting gap is line loss; see section 6) |
| Dispersion | skin effect: loss and delay vary with `sqrt(f)`; a one-sided ISI tail on a 10 ns pulse | R uses `skrf.media.Coaxial` (dispersive); C's LTRA line has constant series R (no `sqrt(f)` skin effect), so its frequency dependence differs from a real coax |
| Time-bandwidth product | the plan estimates about 10^2 to 10^3, loss-bound | the 148-sample receptive field is inside this range, but the long taps need re-amplification |
| Tap precision | coupler amplitude +-0.5 dB, phase +-5 degrees, isolation 20-30 dB (a "0" leaks 3-10 %) | R defaults (0.5 dB, 5 degrees, 30 dB) and sweeps of tap amplitude error and isolation |
| Noise | NF about 3 dB per amplifier; activations need about 5-6 effective bits | R stage noise and the `snr_db` (per-stage) sweep; E's ADC-bits sweep |
| Crosstalk | about -35 dB between coax lines | flat neighbour-line crosstalk in R |
| Combiner loss | `10 log10(N)` dB for an N-way Wilkinson | `N = I*K`: 16 in layer 1, 64 (width 8) or 256 (width 32) in layers 2-3; it enters R's required-gain report |

## 6. What a passive network cannot do, and the minimal hybrid

Passive delay lines, couplers, hybrids and combiners are linear and lossy. They cannot supply gain, a nonlinearity, normalisation or a softmax. The
smallest network that works is therefore a hybrid: passive tapped delay lines and combiners for the linear (multiply-accumulate) part, **one saturating
amplifier per line per layer** for gain and tanh, and **one ADC at the end** for the pooled readout and software linear layer plus softmax. In this study the
passive part is modelled with real line physics (C and R), the amplifier as an ideal diff pair or a memoryless tanh, and the last stage as ADC plus software.

The demo makes one consequence visible, and it has two parts. (1) With RG-58-like loss and no repeaters in the line the longest taps are much weaker than the shortest taps (about 36 dB
for the longest layer-3 tap at 32 dB/us); that operating point is not buildable, and at lower loss the gap to V closes (`results.md`, section 2). (2) With **one gain per layer**, the
default compensation of the C tier divides by the mean tap gain, which is dominated by the weak long taps, and so multiplies the unattenuated short taps by 2.85 in layer 3. That over-drives the
tanh: C scores 0.663 with this default, 0.804 with no compensation and 0.805 with ideal per-tap repeaters at the same 32 dB/us, so most of C's 20-point gap to V is that modelling choice and only
about 6 points is line loss (`circuit_extraction.md`, "Where C's gap comes from"). R uses a similar per-layer gain rule and shows the same layer-3 saturation. This is a statement about the
simulation, not a hardware prediction.

## 7. Why this is a neural net and not a delay-line LLM

The requirement was a delay-line *neural network*, and that is what this repository builds and simulates: a small ternary convolutional classifier. A delay-line
"LLM" was not built and nothing here claims one is possible. A tiny language model (Qwen2.5-1.5B Q4_K_M) appears only as a **commentator**: it reads a text card of at most about
1.5k tokens that Python renders from `compare.json` and writes a short JSON commentary, whose numbers are machine-checked against the card. It does not run, train or evaluate the
delay-line network and has not analysed anything beyond that card (`llm_commentary.md`, `llm_eval.md`).

## 8. Prior art and what is (modestly) new

Not new: tapped delay lines as FIR filters, transversal and SAW matched filters, delay-based reservoir computing (Appeltant 2011; Paquot 2012; Larger 2012; Brunner 2013),
analog and photonic accelerators, TCN/S4/Hyena long convolutions, BitNet b1.58 (Ma et al. 2024). The plan's claim of modest novelty is the explicit mapping of BitNet-style
ternary weights onto tap polarity in dilated tapped delay lines with a natively-tanh diff-pair activation, plus a reproducible four-fidelity harness whose tiers share one weight file and one set of metrics, and in which the commentary is number-checked. The harness also has a
plumbing path for handing circuit-extraction output to the RF tier (`--cascade`), but that path is **provenance and plumbing, not new information**: the line loss fitted to the
ngspice `.ac` data is the number that was typed into the netlist (fitted 32.000005 dB/us against the 32.0 put in; fit residual max |dH| = 1.3e-13, because the LTRA line and the
closed-form fit are the same RLC line), so what reaches R is the configured RG-58 figure. The only extracted quantities that are not simply the configured inputs are the diff-pair
parameters from the ngspice `.dc` sweep: V_sat = 52.02 mV against the textbook 2 V_T = 51.8 mV (scale 1.0043), gain scale 1.0003 and an offset of about 1e-9, i.e. the SPICE BJT
model agrees with the textbook formula to under half a percent. The SPICE `tap_H` samples are not used by R. These are statements about this repository, not a literature survey.
