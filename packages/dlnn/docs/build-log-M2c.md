# Build log M2c (RF tier: S-parameter / channel model + degradation sweeps)

Worker M2c. Files: `dlnn/tiers/rf.py` (replaces the stub), `tests/test_rf.py`, `tests/helpers_tdlnet_rf.py`, `docs/rf_model.md`, this log.
Channel-level simulation only: no EM/layout effects, no hardware measurement.

## Environment
* `.venv/bin/pip install scikit-rf` -> `Successfully installed pandas-3.1.0 scikit-rf-2.2.0` (the only two packages this worker installed; numpy 2.5.4 / scipy 1.18.1 already present).
  `skrf.__version__ == 2.2.0`. Proxy worked; TLS verification untouched.
* Tier status is `ok` (skrf really built every tap path). The numpy RLGC fallback exists, is selected only when `import skrf` fails (status `fallback`) and is tested.

## Model used for validation
**No trained model existed** (`models/tdlnet_modrec_ternary.npz` absent at the time of every run below), so everything was validated on a
**RANDOMLY INITIALISED TDL-Net-S** (2->8 K=8 d=1, 8->8 K=8 d=4, 8->8 K=8 d=16, pooled readout; `tests/helpers_tdlnet.py` from M2a, ridge-fitted readout, V accuracy 0.45;
`tests/helpers_tdlnet_rf.py` wraps it). Agreement numbers are therefore agreement with that model's own V outputs, and this model is fragile
(a swap of tanh for the fitted Rapp curve alone drops V-vs-V agreement to 78 %, independent check in plain V code). Re-run
the acceptance script (`/tmp/claude-0/-home-user/b7e00273-2a66-5b9b-99ad-f33d102fbb7e/scratchpad/m2c/accept_rf.py`, session scratch, it switches to `models/tdlnet_modrec_ternary.npz` automatically when that file exists) before quoting these numbers for the trained model.

## Commands and actual output
`.venv/bin/python -m pytest -q tests/test_rf.py` -> `18 passed in 35.55s`.

Acceptance script (4000-example modrec test set, random model seed 0), output verbatim:
```
V acc 0.4495
V under 30 dB INPUT AWGN: agreement with V(clean) 0.9838
ideal, noiseless: agree 1.0000
ideal, 30 dB per-stage SNR: agree 0.9357
ideal, 40 dB per-stage SNR: agree 0.9805
ideal, 30 dB INPUT AWGN through RF: agree 0.9838
```
Four random models (seeds 0-3), ideal channel: noiseless agreement 1.0000 on all; 30 dB per-stage 0.9357 / 0.9397 / 0.9350 / 0.9455; 40 dB 0.9805 / 0.981 / 0.9758 / 0.98; 50 dB 0.994-0.995.

**Acceptance "30 dB SNR, 0 dB/us, ideal couplers: top-1 agreement >= 98 %" is NOT met as per-stage SNR** with this model (93.6 %; >= 98 % needs ~40 dB per stage).
With the ideal channel and no stage noise R equals V exactly (100 %, layer outputs within 2e-3), and with 30 dB *input* AWGN (the V/E convention) R agrees with V(clean) 98.4 %,
exactly V's own noise sensitivity (98.4 %). The per-stage figure is lower because three independent 30 dB noise sources are injected and the random model has small logit margins
(median top-1 probability 0.40). Whether the trained model clears 98 % at 30 dB per stage is open until M1's model exists.

Full RF run, nominal defaults, 4000 test examples: `status=ok tool=scikit-rf 2.2.0 runtime=35.4s` (with ~9 load average from other workers; 15.9 s for the `sweep_baseline=ideal` rerun, which reuses cached lines).
Headline (defaults: 5 dB/us, iso 30 dB, amp err 0.5 dB, phase err 5 deg, Gamma 0.05, xt -35 dB, stage SNR 30 dB, NF 3 dB, 10-bit ADC, tanh):
accuracy 0.397, top-1 agreement vs V 0.594, mean KL 0.2135, logit NMSE 0.416 (V accuracy 0.4495). Required amplifier gain per layer 22.2 / 28.4 / 29.5 dB.

Sweeps (nominal baseline; acc / mean_kl / top-1 agreement vs V):
```
snr_db(stage) 0,5,10,15,20,30   acc .125 .128 .183 .319 .382 .397   kl 17.1 9.32 2.80 .688 .309 .214   top1 .225 .229 .295 .476 .578 .594
loss_db_per_us 0,5,14,32        acc .423 .397 .313 .232            kl .079 .214 .783 2.42               top1 .734 .594 .393 .255
isolation_db 15..40 step 5      acc .278 .331 .380 .397 .405 .413  kl .98 .475 .287 .214 .184 .170      top1 .434 .510 .562 .594 .609 .622
tap_amp_err_db 0,.25,.5,1,2     acc .416 .406 .397 .364 .297       kl .204 .193 .214 .335 .869          top1 .660 .632 .594 .509 .364
adc_bits 4..10                  acc .237 .315 .369 .395 .398 .398 .397  kl 2.39 .627 .321 .238 .219 .215 .214  top1 .319 .478 .558 .586 .594 .592 .594
```
Monotonicity: accuracy is monotone in SNR, loss, isolation and ADC bits. Non-monotone: `tap_amp_err_db` mean KL 0.204 -> 0.193 -> 0.214 for 0 -> 0.25 -> 0.5 dB (small, random-error realisation + fragile model); ADC top-1 is .586 / .594 / .592 / .594 at 7 / 8 / 9 / 10 bits (non-monotone by 0.002, noise level).
Same sweeps around the ideal channel (`sweep_baseline=ideal`, stage SNR 30 dB kept): top-1 at loss 0/5/14/32 = .936/.725/.422/.238; isolation 15..40 = .437/.607/.736/.831/.887/.916; amp err = .936/.882/.806/.670/.476; ADC 4..10 bits = .360 .533 .713 .841 .905 .927 .934.
Per-tap equalisation (`tap_equalize=True`, ideal otherwise, noiseless) at loss 5/14/32: agreement .782/.524/.318 vs .724/.422/.241 without: dispersion (skin-effect ISI tail), not only droop, dominates.

NARMA10 (reservoir model, regression): `rf.run` ok in 4.3 s, NMSE 0.214 vs V 0.150, nmse_vs_virtual 0.053, 5 sweep axes (nmse), contract valid.

## Validation done (tests)
skrf line loss at 100 MHz equals `loss_db_per_us` (0.1 dB); loss ratio 100/25 MHz = 2 (sqrt f); one-sided ISI tail; ideal channel == V; fine-grid waveform == symbol-rate engine (rel. err < 5e-3, measured 1e-7 ideal / 1e-3..2e-3 non-ideal);
numpy fallback vs skrf (|H| within 2 % at 32 dB/us, FIR within 0.5 %); isolation leak error scales 10 dB -> factor 3.16; amp-error zero is exact; tap droop and equaliser; Rapp fit and limits; ADC levels; stage noise equals P_ref*10^((NF-SNR)/10);
tier JSON validates (5 sweep axes, top-1 arrays, waveforms); regression run; cascade hook (missing cascade -> warning; real `dlnn/cascade.py` with a synthetic `dlnn-circuit-extracted/1` JSON -> loss/amp scales applied, `inputs_from=["params/circuit_extracted.json"]`).

## Deviations / decisions
* Default `amp_model="tanh"`, not Rapp: the model activation is tanh (diff pair); Rapp (fitted p=1.31, max deviation 0.021) is implemented and selectable but is a *different* function (78 % agreement vs V alone on the random model).
* `snr_db` axis of R = per-stage amplifier SNR (V/E use input AWGN). `snr_sweep_mode="input"` gives the V/E convention.
* The contract's sweep entries cannot carry `top1_agreement_vs_virtual`; it is stored in the tier npz as `sweep.<axis>.top1_agreement_vs_virtual` (+ `sweep.<axis>.values`).
* Cascade: only the fitted loss and amplifier scales (V_sat, gain, offset) are consumed; SPICE `tap_H` samples are not injected (they exist for the layer-1 unit cell only). The hook calls `cascade.load_extracted` then `apply_to_rf_cfg`.
* `tests/test_tiers.py::test_all_tiers_valid_and_compare` (M0) hard-codes `["ok","ok","stub","stub"]` and now fails because R (and C, by M2b) are real: the file is not mine, the expected statuses need updating at integration.
* One gain per layer compensates the mean tap loss (plan: "amplifier gain compensates loss"); long taps are therefore weaker. Chosen over per-tap trim because per-tap trim would need repeaters whose noise is not modelled.

## Open problems
* Re-validate with the trained TDL-Net-S (agreement at 30 dB per stage, default-operating-point degradation, knee positions).
* Nominal defaults (5 dB/us, iso 30 dB, ...) were chosen from the plan's typical values, not tuned; on the random model they already cost ~40 points of top-1 agreement.
* No repeater/noise model for long lines; `loss_db_per_us = 32` is therefore the "no repeaters" case the plan says is not buildable.
