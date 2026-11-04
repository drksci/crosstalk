# Output contract (schema `dlnn-tier/1`)

Single source of truth: `dlnn/contract.py` (`TierResult`, `METRIC_IDS`, `TIER_SCHEMA`, `validate`, `save`, `load`).
Every tier module exposes `run(model_npz, dataset, cfg, outdir) -> TierResult` (`dlnn/tiers/{virtual,emulated,circuit,rf}.py`).
Never present a `stub` or `fallback` result as the real tool.

## On-disk layout

```
out/<run_id>/
  model.npz                  model, see "model.npz schema"
  tiers/<V|E|C|R>.json       canonical tier record (this document)
  tiers/<tier>.npz           arrays referenced by the JSON "arrays" index
  compare.json               merged matrix (schema dlnn-compare/1)
  figures/ llm/ params/ dlnn.sqlite args.json created_at.txt labels.npy(classification)
```

`contract.save(result, run_dir)` validates before writing (invalid results are refused); `contract.load(path)` validates on read.
NaN/inf are written as `null`.

## Tier JSON fields (top level; no extra keys allowed)

| field | type | meaning |
|---|---|---|
| `schema_version` | const `"dlnn-tier/1"` | bump on breaking change |
| `run_id` | str | directory name under `out/` |
| `tier` | `V`\|`E`\|`C`\|`R` | virtual, emulated, circuit, RF |
| `status` | `ok`\|`skipped`\|`stub`\|`fallback` | `ok` = the real tool/model ran. `stub` = placeholder (V + noise), `fidelity.omits` MUST say so ("stub"/"placeholder") and be non-empty. `fallback` = a documented cheaper model replaced the real tool (omits non-empty). `skipped` = did not run; `warnings` MUST give the reason, metrics may be empty. Pipeline continues after a skip. |
| `task` | `narma10`\|`modrec` | |
| `tool` | `{name, version}` | what actually computed it, e.g. `numpy-fixedpoint`, later `ngspice`/`42`; stubs use `stub(V+AWGN)`/`stub-0` |
| `model_sha256` | str | canonical hash of the arrays in `model.npz` (`Model.sha256()`, independent of zip timestamps) |
| `dataset_sha256` | str | `dataset["sha256"]` |
| `inputs_from` | [str] | provenance, e.g. `["params/circuit_extracted.json"]` when cascaded; `[]` otherwise |
| `n_examples` | int | classification: test examples; regression: evaluated time steps (after washout) |
| `arrays` | `{name: {file, key, shape, dtype}}` | index of the sibling npz (see below) |
| `metrics` | `{metric_id: number\|null\|[number]\|{str: number}}` | keys restricted to `METRIC_IDS` |
| `sweeps` | `[{axis, values[], accuracy?[], mean_kl?[], nmse?[]}]` | `axis` in `snr_db, adc_bits, loss_db_per_us, tap_amp_err_db, isolation_db`; lists same length as `values`. `mean_kl` is KL(p_V(clean input) \|\| p_tier(swept)); `nmse` is vs target. |
| `fidelity` | `{models: [str], omits: [str]}` | what is simulated / what is NOT. Required honesty field. |
| `runtime_s` | number | wall time of `run()` |
| `warnings` | [str] | free text; stubs repeat that they are stubs |

### `METRIC_IDS`

`accuracy`, `macro_f1`, `acc_per_snr` (`{"0": acc, "5": acc, ...}`), `nmse` (regression, MSE/var(target) after washout),
`top1_agreement_vs_virtual`, `mean_kl_virtual_vs_tier` (nats), `logit_nmse_vs_virtual`, `nmse_vs_virtual` (regression: MSE(y_tier, y_V)/var(y_V)).
Classification tiers emit the first 7 (acc_per_snr only if `dataset["snr_test"]`); regression tiers emit `nmse`, `nmse_vs_virtual`.
V reports exact `0`/`1` for its own vs-virtual metrics.

### npz array names (all optional; keys use `[A-Za-z0-9_.]` only)

| key | shape | task |
|---|---|---|
| `logits`, `probs` | [N, C] float32 | classification |
| `pred` | [N] int64 | classification |
| `y_pred`, `y_true` | [n_out, T] float32 (full test series incl. washout) | regression |
| `trace_activations.L<l>` | classification [<=16, O_l, T]; regression [1, O_l, <=500] | per-layer outputs (E: dequantised) |
| `waveforms.<example_id>` | [lines, T_fine] | C/R later |

### Example (`tiers/E.json`, abbreviated)

```json
{"schema_version": "dlnn-tier/1", "run_id": "narma10-20261103-020338", "tier": "E", "status": "ok",
 "task": "narma10", "tool": {"name": "numpy-fixedpoint", "version": "2.5.4"},
 "model_sha256": "f5f93b52...", "dataset_sha256": "6de11aba...", "inputs_from": [], "n_examples": 1900,
 "arrays": {"y_pred": {"file": "E.npz", "key": "y_pred", "shape": [1, 2000], "dtype": "float32"}},
 "metrics": {"nmse": 0.1555, "nmse_vs_virtual": 0.0043},
 "sweeps": [{"axis": "adc_bits", "values": [4, 5, 6, 7, 8, 9, 10], "nmse": [0.182, 0.160, 0.156, 0.155, 0.1555, 0.154, 0.1535]}],
 "fidelity": {"models": ["bit-true model (no RTL)", "8-bit absmax ADC", "int2 ternary weights"],
              "omits": ["Verilog RTL (not yet written; this is the numpy golden model only)", "all analog effects"]},
 "runtime_s": 0.28, "warnings": []}
```

## Dataset dict (input of every `run()`)

`name`, `task` (`regression`|`classification`), `sha256`, `x_train`, `y_train`, `x_test`, `y_test`, optional `washout`, `snr_test`, `classes`.
Inputs are `[B, C, T]` float; regression targets `[1, n_out, T]`; classification `x [N,2,128]`, `y [N] int`.

## `compare.json` (schema `dlnn-compare/1`, `compare.build(run_dir)`)

`schema_version, run_id, task, model_sha256, dataset_sha256, tiers[], tier_info{tier: {status, tool, fidelity, runtime_s, warnings, inputs_from, n_examples}},
metric_matrix{metric_id: {tier: value}}, deltas_vs_virtual{metric_id: {tier: value - V}}, sweeps{axis: {tier: {values, accuracy?, mean_kl?, nmse?}}},
knees[{axis, tier, metric, knee_value|null, rule}]`.
Knee = first sweep value, walking from the best end of the axis, where accuracy is >=0.05 lower or NMSE >=10 % higher than at the best end.

## `llm/commentary.json` (schema `dlnn-commentary/1`)

`{schema_version, source: "template"|"model", backend, summary, findings[{metric_id, tiers[], observation, likely_cause}], caveats[], questions[]}`.
`source:"template"` is only legal with `backend:"null"`; NullBackend text is never model output. `llm/card.txt` is the <=1.5k-token card (`dlnn/llm/card.py`).

## `model.npz` schema `dlnn-model/1` (`dlnn/model/npz.py`; extended, not changed, by the M1 torch TDL-Net)

Generic layers `l = 0..n_layers-1`:

| key | dtype/shape | meaning |
|---|---|---|
| `schema` | str `"dlnn-model/1"` | |
| `kind` | str | `reservoir` \| `tdlnet` |
| `n_layers` | int | |
| `L{l}_W` | int8 [O, I, K] in {-1,0,+1} | ternary taps (tap polarity) |
| `L{l}_dilation` | int | tap spacing d (delay = k*d samples) |
| `L{l}_gain` | float32 [O] | per-output amplifier gain g (per-layer scale broadcast to O) |
| `L{l}_bias` | float32 [O] | added to the tap sum before tanh (same units as the sum) |
| `L{l}_vsat` | float32 scalar | tanh saturation voltage |
| `readout_W`, `readout_b` | float32 [n_out, F], [n_out] | linear readout |
| `readout_from` | int64 [n] | layers whose outputs are concatenated into the features |
| `readout_kind` | str | `per_step` (features per time step; reservoir) or `pooled` (time-mean then linear; TDL-Net, with `readout_from=[last]`) |
| `input_scale` | float32 | raw input is multiplied by this first |
| `input_fullscale` | float32 / NaN | ADC full scale (after `input_scale`) for the E tier; NaN = take from data |
| `meta_json` | str | JSON hyperparameters / training info |

Per layer: `u[o,n] = sum_{i,k} W[o,i,k] * x[i, n-k*d]` (zeros before n=0);  `y[o,n] = gain[o] * tanh((u[o,n] + bias[o]) / vsat)`.
The E-tier arithmetic is specified in the docstring of `dlnn/tiers/emulated.py`. M1 TDL-Net: write `readout_kind="pooled"` and `readout_from=[n_layers-1]`; a gain that differs per output line is handled by E via Q.15 requantisation.
