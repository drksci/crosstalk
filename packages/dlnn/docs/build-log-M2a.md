# Build log M2a: emulated tier (numpy fixed-point golden model + Verilog RTL)

## What was built
- `hw/rtl/tdl_layer.v`: parametrised (I, O, K, D, IN_W) tapped-delay-line layer. Shift-register delay line per input line, ternary MAC
  (2-bit taps, int32 accumulate), bias, Q.15 scale + clip, 256-entry tanh LUT (loaded through a config port), per-line Q.15 gain, integer
  pool sum. 3-stage pipeline, one sample per clock.
- `hw/rtl/tb_tdl_layer.v`: geometry via `iverilog -P`, data via `$readmemh` files named by plusargs, results via `$fwrite`.
- `dlnn/tiers/emulated.py`: golden model refactored (`layer_params`, `forward_int`, `readout_from_int`) + RTL path
  (`run_layer_rtl`, `verify_chain`, `rtl_crosscheck`, `rtl_info`); `run()` keeps the contract signature.
  The test set and all sweeps run on the numpy golden model; the RTL is run on the first 64 test examples through all 3 layers (each RTL
  layer fed with the previous RTL layer's output) and compared bit for bit, plus layer 1 at ADC widths 4-10 on 16 examples.
  `status` is `ok` only if RTL ran with zero mismatches; otherwise `fallback` with the reason in `fidelity.omits` and `warnings`.
  The contract restricts `metrics` keys to `METRIC_IDS`, so `rtl_examples_compared` / `rtl_mismatches` are stated in `fidelity.models`
  (exact counts) and written to a sidecar `out/<run>/tiers/E_rtl.json` (not part of the tier schema).
- `tests/helpers_tdlnet.py`: `make_random_tdlnet`, `save_random_tdlnet` (randomly initialised TDL-Net-S, NOT trained; see below).
- `tests/test_emulated_rtl.py`: 9 tests + 1 conditional (trained model), RTL tests skip with a reason when iverilog is missing.
- `docs/fixed_point_spec.md`.

## Model validated on
`models/tdlnet_modrec_ternary.npz` did NOT exist at the time of the runs (only `models/llm/`). Everything below is on a **randomly
initialised** TDL-Net-S (topology 2->8 K8 d1, 8->8 K8 d4, 8->8 K8 d16; random ternary taps; vsat calibrated to 1.2x std of the
pre-activation; biases ~N(0, vsat^2); float ridge readout on the time-means of the three layers' outputs, `readout_from=[0,1,2]`;
ADC full scale = 99.99th percentile of |x|). V top-1 accuracy of that net is only 45 % (far below a trained net), so E-vs-V agreement is a
weak proxy; re-run on the trained model when it appears (`test_trained_model_rtl` does that automatically).

## Results (random net seed 0, modrec test set, 4000 examples)
See the verbatim output below.

## Deviations
- Per-line requantisation uses the signed ratio `gain/max|gain|` and is always applied (identity when R = 2^15); M0 used `|gain|` and skipped
  it for `allclose` gains. Identical for positive equal gains (trained TDL-Net).
- RTL cannot be run on 4000 examples in reasonable time (iverilog ~0.5 M ternary MAC/s); work per layer is capped at 4.5e6 MACs
  (`RTL_MAC_BUDGET`), i.e. 64 x 128 samples for TDL-Net-S, but only the first 262 time steps of 1 example for the 48x48 NARMA10
  reservoir (`fidelity` says "first N of T time steps"). Causal, so the prefix is exactly what the full series computes.
- ADC quantiser, pooled-sum -> float conversion and the float readout are numpy in both paths (not RTL).
- The 'ok' status is given to a tier whose headline numbers come from the numpy golden model with RTL equality verified on a subset;
  the tier says so in `fidelity.models/omits`.

## Open problems
- tests/test_tiers.py (M0) asserts C and R are stubs; that is out of date because other workers replaced them (not caused by M2a).
- With the NARMA10 reservoir `run()` now spends ~10-20 s in iverilog (disable with `cfg["rtl"]=False`).

## Commands and actual output

`iverilog -g2012 -Wall -o x.vvp hw/rtl/tdl_layer.v hw/rtl/tb_tdl_layer.v` -> no warnings (Icarus Verilog 12.0).

`.venv/bin/python -m pytest -q tests/test_emulated_rtl.py -rs`
```
.........s
SKIPPED [1] tests/test_emulated_rtl.py:141: trained model (M1) not present
9 passed, 1 skipped in 10.60s
```

`emulated.run(random_tdlnet_seed0.npz, modrec.dataset() [4000 test], ...)` (script /tmp scratch; model from `tests/helpers_tdlnet.py`),
runtime 38.6 s on a loaded 4-core box, `contract.validate` + `contract.save` + `contract.load` OK, status `ok`:
```
accuracy 0.45  macro_f1 0.353  top1_agreement_vs_virtual 0.96225  mean_kl_virtual_vs_tier 0.00221  logit_nmse_vs_virtual 0.00412
fidelity.models[1]: Verilog RTL hw/rtl/tdl_layer.v (iverilog 12.0) run on the first 64 test examples (all 128 time steps), all 3 layers
  chained ..., 198144 values (layer outputs + pooled sums) compared, 0 mismatches [L1: 66048 / 0, L2: 66048 / 0, L3: 66048 / 0]
fidelity.models[2]: RTL-driven float readout on those 64 examples identical to golden: True
fidelity.models[3]: RTL layer 1 also checked at ADC widths 4-10 bits on 16 examples: 115584 values compared, 0 mismatches
snr_db  [0, 5, 10, 15, 20, 30]  acc [.3118 .4085 .4352 .4468 .4498 .451]   KL(V||E) [.470 .1218 .0343 .0122 .0052 .0025]
adc_bits [4..10]                acc [.4315 .432 .4535 .4535 .45 .4532 .4545]  KL(V||E) [.0797 .0603 .0076 .0021 .0022 .0005 .0003]
```
E-vs-V top-1 agreement at 8 bits, random nets seed 0-5 (ADC full scale = p99.99 of |x|): 0.971 0.957 0.973 0.968 0.968 0.975 (mean 0.9685);
with full scale = abs-max instead: 0.950 0.960 0.960 0.973 0.943 0.943 (mean 0.955). Agreement vs ADC bits (seed 0, abs-max full scale):
4: .621, 5: .724, 6: .859, 7: .957, 8: .950, 9: .974, 10: .985.

NARMA10 reservoir (1->48->48 K8, 1 example) RTL check, `rtl_crosscheck`: 41,760 values compared, 0 mismatches, first 434 of 2000 steps
(budget later lowered to 4.5e6 MACs -> ~244 steps), readout identical.

`.venv/bin/python -m pytest -q tests/test_tiers.py` (M0 tests, while C and R were already real): 4 passed, 2 failed; the failures are
`assert status == "stub"` for C and R (another worker replaced the stubs), the E assertions in those tests pass.
