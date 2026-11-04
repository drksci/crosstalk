# RF tier (R): channel-level S-parameter model

`dlnn/tiers/rf.py`. **This is a channel-level simulation: scikit-rf two-port cascades turned into impulse responses and run by
numpy convolution. There is no EM/layout modelling and no hardware measurement; the coax is a generic RG-58-like model, not vendor data.**
Status is `ok` only if `skrf` built the channel; otherwise a closed-form numpy RLGC line is used and the result is `fallback`.

## Signal chain (one layer)

```
x_i[n] --pulse p(t)--> line i
   tap path (o,i,k):  coupler(c, a_oik, skew) -> junction(Gamma) -> coax(l_k) -> junction(Gamma) -> hybrid sign -> Wilkinson branch(1/sqrt N)
   sum over (i,k) [+ neighbour crosstalk] -> + AWGN (stage SNR, NF) -> amplifier gain*A((u+b)/Vsat) -> ideal re-timing S/H -> pulse regeneration -> next layer
global time-mean (RC pool) -> ADC(adc_bits) -> software linear readout
```

## Equations

* Geometry: tap line length `l_k = k*d*T_s*v_p`, `T_s = 10 ns`, `v_p = 19.8 cm/ns`. `eps_r = (c/v_p)^2`, `D_out = D_in*exp(2*pi*Z0*sqrt(eps_r)/eta_0)`
  (`D_in = 0.9 mm`), so Z0 = 50 ohm.
* Line: `skrf.media.Coaxial` (Schelkunoff): `gamma(f) = sqrt((R+j w L)(G+j w C))`, `R ~ sqrt(f)/sigma^0.5` (skin effect, the dispersive part),
  `G ~ f*tan(delta)`. `sigma` and `tan(delta)` are solved so the matched-line loss at `f_ref = 100 MHz` is `loss_db_per_us` per microsecond of delay
  (dielectric share fixed at ~2 % of the 32 dB/us case, scaled with the target); loss = 0 uses `sigma = 1e30`. Loss therefore scales ~ sqrt(f) and the
  line is causal and dispersive (one-sided ISI tail).
* Tap path (skrf cascade with `**`): `S_path = Coupler(c) ** J(Gamma) ** Line(l_k) ** J(Gamma) ** Wilkinson(1/sqrt N)`;
  `J = [[Gamma, t],[t, -Gamma]]`, `t = sqrt(1-Gamma^2)` (lossless mismatch; gives the `Gamma^2 e^{-2 gamma l}` echo/ripple), `c = 10^(-coupling_db/20)`,
  `N = I*K` ports of the Wilkinson combiner (-10 log10 N dB). Hybrid (-1 taps) is an ideal matched sign flip.
* Delay trim: the designer trims each line so its group delay at `trim_mhz` (50 MHz) is exactly `k*d*T_s`:
  `H_eff(f) = S21(f) * exp(+j 2 pi f tau_g(f_trim))`, `tau_g = l * d(beta)/d(omega)`. What remains (group-delay spread, droop, echoes) is the channel degradation.
* Pulse: end-to-end raised cosine, beta = 1 (100 MHz bandwidth, zero ISI at symbol spacing), `P(f) = (T/2)(1+cos(pi f T))`, `p(0)=1`.
  Path response `g_k(t) = IFFT[P(f) H_eff(f)]`; on the grid `U = 8` points per symbol. Symbol-rate FIR taps `g_k[j] = g_k(j T_s)`.
* Tap amplitudes: `a_oik = (W_oik + (1-|W_oik|) * s_oik * 10^(-isolation_db/20)) * 10^(eps_oik/20)`, `eps ~ N(0, tap_amp_err_db^2)`,
  `s = +-1` random (a "0" tap leaks). Phase error `N(0, tap_phase_err_deg^2)` at `f_ref` is realised as a delay skew
  `delta = phi/(360 f_ref)`; first order `g(t-delta) ~ g - delta*g'` (error O((delta/T)^2) < 0.2 % for 10 degrees). Random draws are seeded and fixed, so sweeps use common random numbers.
* Crosstalk: line `i` carries `x_i + xt*(x_{i-1}+x_{i+1})`, `xt = 10^(-|crosstalk_db|/20)` (flat, before the tap).
* Amplifier gain compensation: `comp = 1 / (sum_k n_k |g_k[0]| / sum_k n_k)` (`n_k` = active taps at index `k`): one gain per layer restores the mean main-lobe tap gain
  to 1, so long taps are weaker than short taps (**droop**, needs repeaters in reality). `tap_equalize=True` instead trims every tap separately (optimistic: no extra noise modelled).
  Measured in M4d (`circuit_extraction.md`, "Where C's gap comes from"): with the one-gain-per-layer rule at the cascade loss (32 dB/us) 71 % of layer-3 activations are in tanh saturation (V: 50 %); `tap_equalize` lowers that to 61 %
  and raises accuracy from 0.701 to 0.810. Whether the per-layer rule is a modelling artefact or what a real amplifier chain would do was not decided.
  The required amplifier gain in dB is reported per layer in `warnings`.
* Layer: `u[o,n] = sum_{i,k,j} comp_k * a_oik * (g_k[j] - delta_oik g_k'[j]) * x_i[n - k d - j]` (+ crosstalk mixing), computed as an FFT convolution
  over the 4k test set. `u += N(0, sigma^2)`, `sigma^2 = P_ref * 10^((NF - SNR)/10)`, `P_ref` = mean-square amplifier-input level of the ideal V forward
  (first 256 examples; a fixed physical noise floor, not per-example scaled).
  `y = gain * (amp_gain_scale * A(z) + amp_offset_rel)`, `z = (u + bias)/(Vsat * amp_vsat_scale)`.
* AM/AM: `A = tanh` (the diff pair; default, exactly the model activation) or Rapp `A(z) = z/(1+|z|^(2p))^(1/(2p))` with `p` fitted to tanh (p ~ 1.31, max deviation 0.02).
  The Rapp curve is NOT the model activation, so selecting it adds an activation mismatch (agreement vs V drops, see build log).
* Re-timing: after the amplifier the output is sampled at the symbol instants (ideal sample&hold, zero jitter) and a fresh raised-cosine pulse train drives the next layer.
  Because sampling follows the (memoryless) amplifier, the convolution sampled at symbol spacing is exact; `fine_waveform()` computes the real fine-grid waveform
  of a layer (8 points/symbol, stored as `waveforms.<example>` for layer 1) and the tests check it matches the symbol-rate engine (relative error < 0.5 %).
* Readout: time-mean of the last layer (RC pool), uniform mid-tread ADC with full scale = max |feature| over the data (`adc_bits`), software readout.

## Parameters (`cfg`, defaults)

| key | default | meaning |
|---|---|---|
| `loss_db_per_us` | 5 | coax loss at 100 MHz, dB per us of delay (RG-58 ~32, LMR-400 ~14, 0 = lossless) |
| `isolation_db` | 30 | coupler isolation; a "0" tap leaks `10^(-iso/20)` (inf = ideal) |
| `tap_amp_err_db` / `tap_phase_err_deg` | 0.5 / 5 | sigma of per-tap amplitude (dB) / phase (deg at 100 MHz) error |
| `snr_db` | 30 | per-stage SNR at the amplifier input (None = noiseless) |
| `nf_db` | 3 | amplifier noise figure; effective stage SNR = `snr_db - nf_db` |
| `z0` | 50 | system impedance |
| `gamma_mismatch` | 0.05 | reflection coefficient at both ends of every tap line |
| `crosstalk_db` | -35 | neighbour-line crosstalk (None = off) |
| `adc_bits` | 10 | readout ADC (None = ideal) |
| `coupling_db` | 10 | coupler coupling (only sets the required amplifier gain) |
| `amp_model`, `rapp_p` | `tanh`, `fit` | AM/AM curve |
| `tap_equalize` | False | per-tap gain trim |
| `oversample`, `symbol_ns`, `vp_cm_per_ns`, `f_ref_mhz`, `trim_mhz` | 8, 10, 19.8, 100, 50 | grid / medium |
| `sweep_baseline` | `nominal` | sweeps vary one parameter around the resolved cfg; `ideal` = around `preset=ideal` (stage SNR kept) |
| `snr_sweep_mode` | `stage` | `stage`: R's `snr_db` axis is the per-stage SNR; `input`: AWGN on the input like V/E |
| `cascade_params` | None | path of `params/circuit_extracted.json`; if `dlnn/cascade.py` exists, `cascade.apply_to_rf_cfg` is applied (loss, V_sat/gain/offset scales from the SPICE fit; `inputs_from` set). SPICE `tap_H` samples are not used directly |
| `preset` | - | `"ideal"`: lossless, infinite isolation, no errors, no mismatch/crosstalk, NF 0, tanh, ideal ADC (stage `snr_db` still applies, set `snr_db=None` for noiseless) |

## Sweeps

`snr_db` (per-stage, differs from the V/E input-AWGN semantics unless `snr_sweep_mode="input"`), `loss_db_per_us` 0/5/14/32, `isolation_db` 15..40,
`tap_amp_err_db` 0..2, `adc_bits` 4..10. Each entry has `accuracy` and `mean_kl` (KL of V on the clean input vs R at the swept value). The contract
schema has no field for agreement, so `top1_agreement_vs_virtual` (and `values`) of every sweep are stored in the tier npz as
`sweep.<axis>.top1_agreement_vs_virtual` / `sweep.<axis>.values` (regression: `sweep.<axis>.nmse_vs_virtual`).

## What is NOT modelled

EM/layout/connector/bend effects; transistor-level amplifier (memoryless, no bandwidth limit, no AM/PM, no bias drift, no input/output mismatch); frequency
shape of crosstalk (NEXT/FEXT) and non-neighbour coupling; separate hybrid isolation (one `isolation_db`); temperature, ageing, ground loops, supply noise;
clock jitter and timing offset in the re-timer; DAC/pulse generator imperfections; thermal-noise accumulation along repeatered long lines; the SPICE `tap_H` samples
(only the fitted loss and amplifier scales cascade). Noise is white, injected once per stage at the amplifier input and the ADC is noiseless.
Simulation only: nothing here is a measurement of hardware.
