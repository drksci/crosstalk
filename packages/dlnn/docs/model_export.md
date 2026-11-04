# TDL-Net export (M1): torch model -> `dlnn-model/1`

The schema `dlnn-model/1` is **unchanged**; the V and E tiers load the exported ternary model without modification.

## Training-time model (`dlnn/model/tdlnet_torch.py`)

Layer l (causal, zeros before n = 0, tap k = delay k*d_l samples):

```
u[o,n]  = sum_{i,k} W_t[o,i,k] x[i, n-k*d_l]          W_t = clip(round(W/gamma), -1, 1), gamma = mean|W| (per layer)
pre     = alpha_l * gamma_l * u + b[o]                  alpha_l = learnable per-layer amplifier gain (scalar)
y[o,n]  = g[o] * tanh(pre)                              g = 1 unless --line-gain
```

Straight-through estimator on `W_eff = W + (gamma*W_t - W).detach()`. Training-time ADC: 8-bit absmax fake-quant on the
input (fixed full scale = absmax of the train set, rounded up to 0.5) and on every layer output (grid max|g|/127, i.e. the
E-tier int8 grid). During the first `soft_frac` (50 %) of training the ternarisation is ramped in
(`W_used = lam*gamma*W_t + (1-lam)*W`, lam 0 -> 1); the final epochs are fully ternary. RC pool = time mean.
Readout: ternary linear 8->8 (`gamma_r * W_t,r * exp(log_ra)`) + float bias per class.

## Mapping to the container

| torch | `model.npz` |
|---|---|
| `W_t` (tap k stored at index k; torch conv kernel is flipped in forward) | `L{l}_W` int8 [O,I,K], `L{l}_dilation` |
| `alpha*gamma` | `L{l}_vsat = 1/(alpha*gamma)` |
| `b` | `L{l}_bias = b*vsat` (i.e. `b/(alpha*gamma)`, "same units as the tap sum") |
| `g` (per line, or ones) | `L{l}_gain` [O] |
| readout `R_eff` | `readout_W` float32 [8,8] (values `gamma_r*exp(log_ra)*{-1,0,1}`); `readout_b`; `readout_kind="pooled"`, `readout_from=[2]` |
| input | `input_scale=1.0`, `input_fullscale` = ADC full scale (4.0 for the shipped model) |
| training info | `meta_json` (`arch`, dataset sha, seed, epochs) |

So `y = gain*tanh((u+bias)/vsat)` of the virtual tier equals the torch forward (without fake-quant) to float32 rounding
(`tests/test_model.py`: max |logit diff| <= 1e-4 on 256 test examples; measured value in `docs/build-log-M1.md`).
`gamma` therefore becomes the input-referred amplifier gain 1/vsat of layer l (together with the learnable alpha), not the
output gain; the output gain `g` is the optional per-line scale from the plan's risk list (default off, not used in the shipped model).

## Float baseline (`models/tdlnet_modrec_float.npz`)

Identical topology, unconstrained float taps, no fake-quant. `dlnn-model/1` stores taps as int8, so a float file can
not be in that schema. Chosen: a *separate* schema string `dlnn-model-float/1` with the same keys and `L{l}_W` float32
(`meta_json.float_baseline = true`). `load_model` (ternary loader) refuses it by design; `dlnn.model.export.load_float_model`
returns a `Model` with float W, which `dlnn.tiers.virtual.forward` evaluates unchanged (it only does `W.astype(float32)`).
E/C/R must not be run on the float file (taps are not ternary; `Model.sha256()` of it is not meaningful).
The float metrics also go to `models/train_report.json`.

## Side files
`models/*.pt` are torch state dicts used by the tests (`TDLNet.from_config(meta["arch"])`); `models/train_report.json` holds
mixed/per-SNR/per-class accuracy, seeds, restart train accuracies, seconds and the dataset sha.

# Phase 2 additions (M1b)

## TDL-Net-B and the scaled virtual model (`dlnn-model/1`, unchanged schema)
`models/tdlnet_b_w{8,16}_ternary.npz`, `models/tdlnet_b_w8_aware_ternary.npz`, `models/v32_modrec_ternary.npz` use the same container as
the phase-1 model; all layers have `L{l}_dilation = 1`, K = 8. Trained by `python -m dlnn.model.train --name <stem> --width W
[--depth D] --dilations 1,1,1 --ternary-only [--float-readout] [--line-fir models/board_aware/line_fir.json]`. The readout is the
ternary 8->8 matrix (stored as float32 `gamma_r*{-1,0,1}`) unless `--float-readout`, which stores an unconstrained float digital readout
(physically honest: the readout runs on the Nucleo). The board-aware model's `meta_json.arch.line_fir` carries the measured per-tap FIR
`[K][M]`; the V tier ignores it (ideal delays), so evaluating the aware model with the V tier is the "ideal-line" number; the torch
model (`.pt` + `arch`) applies the FIR. `W_long[o,i,m] = sum_k W[o,i,k] * h_k[m]`.

## Arena models (`models/arena/*.npz`, own schemas, float32 unless noted)
Input convention for all: x [2,128] raw I/Q float32 (as the dataset; no scaling), output 8 logits (class order `dlnn.data.modrec.CLASSES`).
- `mlp_s.npz`, `mlp_l.npz` (`dlnn-arena-mlp/1`): `fc_{i}_weight` [out,in], `fc_{i}_bias` (i = 0..n-1; torch state-dict keys with `.`->`_`),
  `layer_names`, `hidden`. Forward: h = x.flatten (I row then Q row, 256); h = relu(W h + b) for all but the last layer.
- `cnn.npz` (`dlnn-arena-cnn/1`): `cv_{i}_weight` [O,I,8], `cv_{i}_bias` (i = 0..2), `fc_weight` [8,8], `fc_bias`. Forward per layer:
  left-pad 7 zeros, cross-correlation (torch `conv1d` convention: out[n] = sum_j w[j] xpad[n+j]), ReLU; then mean over time; linear.
- `reservoir.npz` (`dlnn-arena-reservoir/1`): random ternary layers `L{l}_W` int8 [O,I,K], `L{l}_dilation`, `L{l}_bias`, `L{l}_vsat`
  (same layer equation as `dlnn-model/1` with gain 1, `y = tanh((u+bias)/vsat)`), `pool` ("mean" | "mean_sq": per-layer time mean of y
  [and of y^2]), `feat_mean`, `feat_std` (standardise the concatenated pooled features of all layers), `readout_W` [8,F], `readout_b`.
- `cumulant.npz` (`dlnn-arena-cumulant/1`): `classifier` ("logreg" | "tree"), `feature_names`, `feat_mean`/`feat_std` (applied to the
  features of `dlnn.model.arena.features`, a documented numpy function), then either `W` [F,8], `b` [8] (softmax regression) or the
  tree arrays `tree_feat` (-1 = leaf), `tree_thr`, `tree_left`, `tree_right`, `tree_prob` [nodes,8] (go left if feature <= thr).
  Feature definitions are in the docstring/`FEATURE_NAMES` of `dlnn/model/arena.py` (cumulants after x^4-line CFO compensation, spectral
  lines of x^2/x^4/x^8, amplitude, instantaneous-frequency, autocorrelation and spectrum statistics).
- `arena_report.json`: per model `params`, `ops` {mults, adds, nonlin} for one 128-sample window (a MAC = 1 mult + 1 add; ternary taps = 1
  add per non-zero tap), `acc`, `acc_ge10dB`, `acc_per_snr`, `acc_per_class`, training hyper-parameters. Also `energy_refs.json`.
