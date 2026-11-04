# Build log

<!-- ENV-BEGIN -->
## Environment

- date: 2026-11-03T06:03:06Z
- python: Python 3.13.18
- pip core + extras rf,dev,pg (numpy scipy matplotlib jsonschema nbformat nbconvert papermill ipykernel jupyter scikit-rf pandas pytest psycopg[binary]; editable dlnn): INSTALLED
- torch: already present (2.15.0+cpu)
- llama-cpp-python: already present (0.3.19)
- Qwen2.5-1.5B-Instruct Q4_K_M weights: already present (models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf)
- BitNet b1.58-2B-4T (optional second backend): not installed by this script (commands in the setup.sh comments and docs/llm_eval.md)
- jupyter kernel 'dlnn': REGISTERED
- apt ngspice: already present (/usr/bin/ngspice)
- apt iverilog: already present (/usr/bin/iverilog)
- postgresql server: not requested (WITH_PG=1 to install; --db sqlite is the default)
<!-- ENV-END -->

## M0 walking skeleton

### What was done
Contract (`dlnn/contract.py`, `docs/contract.md`), NARMA10 and modrec generators, numpy ternary-TDL reservoir (`dlnn/model/reservoir.py`, container/schema in `dlnn/model/npz.py`), real V tier (float32 numpy), real E tier (numpy bit-true fixed-point, no RTL), C and R **stubs**, compare, sweeps, NullBackend + card, figures 4-5 (light/dark, PNG+SVG), SQLite schema/loader, `run.py` (+ `dlnn/pipeline.py`), notebooks 00-06 (built by `notebooks/_build.py`), tests.

Environment first run of `./setup.sh` (before the idempotent re-run shown in the Environment block above, which reports "already present"): apt installed `ngspice` and `iverilog` successfully (both at `/usr/bin/`); pip installed all packages; Jupyter kernel `dlnn` registered. torch / llama-cpp-python intentionally not installed.

### Acceptance (actual output)

`./setup.sh` -> exit 0 (run twice; second run idempotent).

`.venv/bin/python run.py --task narma10 --llm none`:
```
run narma10-20261103-020552: model f5f93b52374e dataset 6de11aba7368
  tier V: status=ok metric=0.1497 (0.6s)
  tier E: status=ok metric=0.1555 (0.3s)
  tier C: status=stub metric=0.1667 (0.0s)
  tier R: status=stub metric=0.2146 (0.0s)
done in 3.1s -> /home/user/dlnn/out/narma10-20261103-020552
```
All four tier JSONs load through `contract.load` (validates), `compare.json` validates (`compare.load`).
**NMSE_V = 0.1497** (limit 0.40), **NMSE_E = 0.1555** (+3.9 % relative to V, limit 10 %). Reservoir: 1->48->48 lines, K=8, d=(1,2), 18,816 ternary weights, ridge lambda picked on a held-out tail of the train set; fit takes ~0.03 s. No tuning beyond choosing `drive=0.5`, width 48 from a handful of trials (NMSE 0.15-0.25 across settings).

`.venv/bin/python run.py --task modrec` -> prints `error: modrec: model not available until M1 ...`, exit code 2, nothing written.

`.venv/bin/python -m pytest -q` -> `16 passed in 3.98s`.

Notebooks: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dlnn` succeeded on all of 00-06 (00 and 06 have real content; 01-05 are labelled stubs with one true cell each).

### Deviations from the plan
- `scikit-rf` is not in the M0 deps (not needed until M2c); `jsonschema` was added (contract validation).
- `acc_per_snr` is a `{snr: acc}` dict, and `nmse` / `nmse_vs_virtual` metric ids were added for regression; sweeps carry an optional `nmse[]` besides `accuracy[]`/`mean_kl[]`.
- `arrays` in the JSON is an index (file/key/shape/dtype); nested names are flattened with `.` (e.g. `trace_activations.L0`).
- Every tier calls `virtual.forward` itself to compute the vs-virtual metrics (V is cheap), so `run()` keeps the plan's 4-argument signature.
- E tier: LUT input range fixed at +-4 (step 1/32) with a Q.15 multiplier; exact spec is the docstring in `dlnn/tiers/emulated.py` and must be matched by the M2a RTL.
- `--cascade` is accepted but ignored with a warning (M2b/M4). `--db postgres` exits with a clear "M4" message. `--llm qwen|bitnet` raise NotImplementedError (M3).

### Stubbed / not real
- C and R tiers: `status:"stub"`, V forward on an input with AWGN (35 dB / 30 dB SNR) and 0.3 / 0.5 dB flat loss. Not SPICE, not RF simulation; `fidelity.omits` says so, figures label them "STUB: not a real simulation", the LLM card and template commentary say so.
- E tier is a numpy bit-true model only (no Verilog).
- LLM "commentary" is a fixed template (`source:"template"`).
- `fig4_outputs` classification branch (modrec probability bars) is written but not yet exercised (no modrec model before M1).
- Sweep axes implemented: `snr_db` (input AWGN, all tiers) and `adc_bits` (E). The modrec SNR sweep adds noise on top of the dataset's own noise.

## M1 model: trained ternary TDL-Net (summary; details in `docs/build-log-M1.md`)

- Built `dlnn/model/{ternary,tdlnet_torch,export,train}.py`, `docs/model_export.md`, `tests/test_model.py`; torch 2.15.0+cpu installed from the PyTorch CPU index.
- `python -m dlnn.model.train --task modrec`: 202 s total. Width 8 (plan topology, 1,216 ternary weights): float 84.65 % mixed / 87.46 % at >= 10 dB; ternary **73.48 % mixed / 76.29 % at >= 10 dB**. The 85 %-at->=10 dB ternary target is **not met** at width 8 (capacity limit; per-class: the ternary width-8 net drops 8PSK to 0 %).
- Width 32 (same code, `--width 32`): float 91.9 % mixed (95.5 % at >= 10 dB, measured in M4a; the M1 note quoted 91.9 as the >= 10 dB figure); ternary **86.3 % mixed / 89.1 % at >= 10 dB**. This is the headline model of the demo (`models/tdlnet_modrec_w32_ternary.npz`); the width-8 net stays as a documented secondary (`models/tdlnet_modrec_ternary.npz`). Both are always reportable; neither result is hidden.
- V tier reproduces the torch logits to 1.9e-6 (limit 1e-4). The plan recipe (lr 3e-3, 20 epochs) underfits badly; the adopted recipe (Adam lr 3e-2, cosine, 60 epochs, rotation augmentation, progressive ternarisation, 3 seeds picked by TRAIN accuracy) is documented with the full table of what was tried.

## M2a emulated tier (summary; details in `docs/build-log-M2a.md`)

- Built `hw/rtl/{tdl_layer.v,tb_tdl_layer.v}`, `dlnn/tiers/emulated.py` (numpy bit-true golden model + RTL path), `docs/fixed_point_spec.md`, `tests/test_emulated_rtl.py`.
- iverilog 12 RTL vs numpy golden model: **0 mismatches** over all 3 layers on 64 test examples (198,144 values) plus layer 1 at ADC widths 4-10 on 16 examples; the RTL is run on this subset only (iverilog speed), the test set and all sweeps run on the numpy golden model, and the tier says so. `status=ok` requires zero mismatches.
- E vs V top-1 agreement at 8 bits on random nets: 0.9685 mean (limit 0.95). ADC full scale default = 99.99th percentile of |x| (documented decision; abs-max gives 0.955). On the trained width-8 net 0.99575; see M4a below for the width-32 net.

## M2b circuit tier (summary; details in `docs/build-log-M2b.md`)

- Built `hw/spice/{coax_rg58.lib,diffpair.lib,unit_cell.cir}`, `dlnn/tiers/circuit.py` (ngspice 42 driver and its own raw-file reader), `dlnn/cascade.py`, `docs/circuit_extraction.md`, `tests/test_circuit.py`.
- Unit cell `.tran` on 16 examples vs V: layer-1 NMSE 0.0136 (limit 0.05); diff-pair tanh fit R^2 = 0.99999997; fitted line max |dH| 1.3e-13 vs SPICE. Validated on a random-init net, 264 s on a loaded box.
- Honest limits: ngspice LTRA has no `G` (so constant-R loss, low-frequency loss overstated); the full test set is run by the circuit-calibrated behavioural model, not by SPICE, and the tier labels it so.

## M2c RF tier (summary; details in `docs/build-log-M2c.md`)

- Built `dlnn/tiers/rf.py` (scikit-rf 2.2.0 channel model), `docs/rf_model.md`, `tests/test_rf.py` (18 tests), five sweep axes.
- Acceptance "30 dB SNR, 0 dB/us, ideal couplers: top-1 agreement >= 98 %" was **missed on random-init nets as per-stage SNR** (93.6 %); it holds under the input-AWGN reading (98.4 %). Not tuned to pass; re-measured on the trained nets in M4a below.
- `snr_db` axis of R = per-stage amplifier SNR (documented, kept). Sweeps monotone except two noise-level wiggles noted in the M2c log.

## M4a pipeline integration (modrec end to end)

### What was wired
- `run.py` / `dlnn/pipeline.py`: `--task modrec`, `--model PATH` (default `models/tdlnet_modrec_w32_ternary.npz`; a float or missing file is refused before anything is written), `--float-model`, `--cascade` (C runs before R; R gets `cascade_params=<run>/params/circuit_extracted.json`, `inputs_from` is set by R/`dlnn.cascade`), `--llm none|qwen|bitnet` (backend built BEFORE the expensive part; missing GGUF/wheel = clear error, never a silent template), `--db sqlite|postgres` + `--pg-dsn`/`$DLNN_PG_DSN`, `--cr-subset N`, `--n-test N`.
- Per-stage timing (`timing.json` -> `compare.json["timing"]`), float-baseline `reference` block (labelled `is_tier: false`), `run_meta`, knees and `fidelity.omits` per tier (from the tier JSONs), optional `paired_vs_virtual` for subset tiers.
- Subset policy for C and R: a fixed stratified subset (`pipeline.stratified_subset`: first n/40 examples of each (class, SNR) cell in test-set order; identical for C and R; the leading examples coincide with the full set, so figure 4 and traces line up). **Not needed**: with width 32 on the full 4000-example test set all four tiers take 10-13 min (see below), so `DEFAULT_CR_SUBSET = None` and every tier ran on all 4000 examples (`n_examples = 4000` in all four tier JSONs). The subset code path is exercised by tests (`cr_subset=80` of 200).
- `dlnn/contract.py`: COMPARE_SCHEMA gained optional `reference`, `paired_vs_virtual`, `timing`, `run_meta`. `dlnn/compare.py`: builds them. `dlnn/db/load.py`: one code path for SQLite and PostgreSQL (psycopg 3), `pg_summary()` SELECT check; subset tiers store their true test-set example ids.
- `tests/test_tiers.py` fixed (C and R are no longer stubs; the NARMA10 end-to-end test now runs V,E and the modrec-missing test checks the new error). New `tests/test_pipeline_modrec.py` (11 tests, ~1 min: stratified subset, subset dataset hash, refused float model, 200-example run with cascade + fallback labels + paired deltas + SQLite content + LLM-unavailable guard + Postgres round trip when `DLNN_PG_TEST_DSN` is set).
- `pyproject.toml` extras: `train` (torch), `rf` (scikit-rf, pandas), `llm` (llama-cpp-python==0.3.19), `pg` (psycopg[binary]), `dev` (pytest), `all`. `setup.sh` rewritten (idempotent; torch from the PyTorch CPU index; llama-cpp-python wheel attempt, then the source build that M3 found necessary because the abetlen cp313 wheel is a musl build; Qwen GGUF via curl; optional `WITH_PG=1`; records installed / already present / FAILED in the Environment block). Second run: 5 s, all "already present". spicelib and huggingface_hub are not used (own raw-file reader, curl).

### Acceptance runs (actual commands and output)

Box: 4 cores shared with the M3 worker (BitNet eval using 2 threads), load average 2-4 during the runs.

`.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none --run-id modrec-w32-full` (width 32, **10.2 min**)
```
run modrec-w32-full: model 3e67999dbaf5 (tdlnet_modrec_w32_ternary.npz) dataset e1b8e74480bd  load avg [3.85, 3.58, 4.3]
  tier V: status=ok n=4000 metric=0.8628 (22.2s)
  tier E: status=ok n=4000 metric=0.863 top1-agree-vs-V=0.9972 (169.3s)
  tier C: status=ok n=4000 metric=0.6627 top1-agree-vs-V=0.6617 (211.1s) inputs_from=['params/circuit_extracted.json']
  tier R: status=ok n=4000 metric=0.7013 top1-agree-vs-V=0.7085 (207.0s) inputs_from=['params/circuit_extracted.json']
  reference float baseline: accuracy 0.9187 (ternary V 0.8628)
  llm: backend=null source=template numbers_verified=n/a (template) (0.0s)
  db: sqlite /home/user/dlnn/out/modrec-w32-full/dlnn.sqlite (0.1s)
timing (s): dataset+model=0.51, V=22.19, E=169.26, C=211.07, R=206.99, reference=1.66, compare=0.0, figures=3.04, llm=0.0, total_before_db=614.75
done in 614.8s (10.2 min; budget 25 min) -> /home/user/dlnn/out/modrec-w32-full
```
All four tier JSONs load through `contract.load` (validates) and `compare.json` through `compare.load`.

| width 32 (4000 examples) | accuracy | macro-F1 | top-1 agreement vs V | mean KL(V\|\|tier) nats | logit NMSE vs V | status / tool | runtime |
|---|---|---|---|---|---|---|---|
| V | 0.8628 | 0.8624 | 1 | 0 | 0 | ok, numpy-float32 | 22 s |
| E | 0.8630 | 0.8625 | 0.9972 | 7.5e-05 | 2.3e-05 | ok, numpy-fixedpoint + iverilog RTL on 64 ex | 169 s |
| C | 0.6628 | 0.6265 | 0.6618 | 1.055 | 0.2256 | ok, ngspice 42 (unit cell) + calibrated model | 211 s |
| R | 0.7013 | 0.6580 | 0.7085 | 0.766 | 0.2059 | ok, scikit-rf 2.2.0 | 207 s |
| *float32 baseline (reference, not a tier)* | 0.9187 (95.5 % at >= 10 dB) | | 0.8818 vs ternary V | | | | |

Read this honestly: E tracks V to 0.3 % top-1 (8-bit fixed point is nearly free). C and R lose 16-20 accuracy points, **because `--cascade` makes
R inherit the SPICE line loss of 32 dB/us (RG-58, no repeaters)** and C's headline is also the 32 dB/us line with one ideal gain trim per layer:
this is the operating point the plan says is not buildable without repeaters. Their own `loss_db_per_us` sweeps show the cheaper media:
C accuracy 0.863 / 0.827 / 0.751 / 0.663 and R 0.861 / 0.845 / 0.789 / 0.701 at 0 / 5 / 14 / 32 dB/us (knees: C 14, R 14 dB/us). R's other sweeps: isolation knee none (0.666 at 15 dB vs 0.704 at 40 dB), `tap_amp_err_db` knee 2 dB, per-stage SNR knee 15 dB (accuracy 0.70 at 30 dB, 0.25 at 5 dB); `adc_bits` flat. E: `adc_bits` accuracy 0.849 at 4 bits, >= 0.86 from 5 bits. V/E `snr_db` sweep knee at 5 dB (input AWGN, accuracy 0.698 at 0 dB, 0.811 at 5 dB, V and E nearly identical). Per-SNR accuracy V: 0.79 / 0.85 / 0.88 / 0.89 / 0.90 at 0-20 dB; C 0.55-0.71; R 0.59-0.74.

`.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none --model models/tdlnet_modrec_ternary.npz --run-id modrec-w8-full` (width 8, **2.8 min**, total 165.7 s)
```
  tier V: status=ok n=4000 metric=0.7348 (2.4s)
  tier E: status=ok n=4000 metric=0.7348 top1-agree-vs-V=0.9958 (27.2s)
  tier C: status=ok n=4000 metric=0.54 top1-agree-vs-V=0.5960 (95.4s) inputs_from=['params/circuit_extracted.json']
  tier R: status=ok n=4000 metric=0.4785 top1-agree-vs-V=0.4990 (35.5s) inputs_from=['params/circuit_extracted.json']
  reference float baseline: accuracy 0.8465 (ternary V 0.7348)
```
Width 8: V 0.7348 (76.3 % at >= 10 dB, drops 8PSK), E 0.7348, C 0.540, R 0.479; mean KL C 0.491, R 0.880. The width-8 net is more fragile: R at 5 dB/us (loss sweep) 0.707, at 32 dB/us 0.479. Results are reported next to, not instead of, the headline.

`LLM + Postgres`: `DLNN_PG_DSN="dbname=dlnn host=/var/run/postgresql" .venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm qwen --db postgres --run-id modrec-w32-qwen-pg` -> **12.9 min** (LLM stage 106.9 s; Qwen2.5-1.5B Q4_K_M, 2 threads, with the box also running the M3 BitNet evaluation):
```
  llm: backend=qwen source=model numbers_verified=False (106.9s)
  db: postgres loaded run modrec-w32-qwen-pg; SELECT counts {'runs': 1, 'tier_results': 4, 'sweeps': 57, 'predictions': 16000, 'llm_outputs': 1}; tiers [['C','ok',4000,237.261],['E','ok',4000,169.8017],['R','ok',4000,220.0991],['V','ok',4000,26.7546]]
```
The verification guard did its job: it retried once and still flagged one number (a "6-bit ADC" claim for C that is not on the card) as unverified (`verified:false`, to be shown dimmed). The commentary is still weak as analysis: its summary says "R stays close to V" while the card shows R agreement 0.709 and KL 0.766 (the guard checks numbers, not attribution; see `docs/llm_eval.md`). The card does not mention the float reference, the 32 dB/us cascade operating point or subsets (card code is M3's, not changed).

### R agreement at 30 dB per stage, on the trained nets (M2c criterion re-measured)
`scripts/r_ideal30.py` (R tier, `preset=ideal`: 0 dB/us, ideal couplers, no mismatch/crosstalk, NF 0, per-stage SNR 30 dB, 4000 examples, 3 noise seeds): **width 32 0.9863** (0.9873 / 0.9855 / 0.9860), **width 8 0.9809**. The >= 98 % criterion is met on both trained nets (the random-init nets missed it at 93.6 %). Noiseless ideal R = V exactly (1.0000). The definition note (per-stage vs input SNR) is in `docs/tiers.md`; no code was changed for this.

### PostgreSQL status: verified against a live server
PostgreSQL 16.15 (Ubuntu apt package, already installed, cluster `16/main` was down) started with `pg_ctlcluster 16 main start`; role `root` (superuser, peer auth over the unix socket, no password) and database `dlnn` created; `pip install 'psycopg[binary]'` -> psycopg 3.3.6. `--db postgres --pg-dsn ...` loaded the runs above and a `SELECT` read them back (counts above); `psql`: accuracy from `predictions` per tier V 0.8628, E 0.8630, C 0.6628, R 0.7013, equal to the tier JSONs. One bug found and fixed on the way: the DDL splitter broke on a `;` inside a comment. The test `test_postgres_load_roundtrip` runs when `DLNN_PG_TEST_DSN` is set. This is a throwaway local server, not a hardened deployment.

### Tests
`DLNN_PG_TEST_DSN="dbname=dlnn host=/var/run/postgresql" .venv/bin/python -m pytest -q -rs` -> `112 passed in 484.36s (0:08:04)` (no skips). Without the env var one test (`test_postgres_load_roundtrip`) skips with reason "no PostgreSQL test server: set DLNN_PG_TEST_DSN". NARMA10 sanity: `run.py --task narma10 --llm none` -> all four tiers `ok` in 43.8 s (V 0.1497, E 0.1555, C 1.19 at 32 dB/us with no repeaters, R 0.2137 NMSE).

### Not done / open
- The plan's "V_cal" cascade consumer in tier V is not implemented; only R consumes `circuit_extracted.json` (loss + amplifier scales; SPICE `tap_H` samples unused).
- With `--cascade` the R headline is the 32 dB/us operating point (see above), not R's standalone defaults; both appear in R's loss sweep.
- `dlnn/viz/figures.py` fig 4 titles say "label N" but print V's predicted class, not the true label (not my file; not changed). Figures 1-3, 6-8, executed notebooks 01-05 for modrec, and the notebook builder were not part of M4a.
- E's runtime is dominated by the iverilog cross-check and SNR/ADC sweeps (169 s); C's by ngspice plus the behavioural model.

## M4c final run and documentation set

- `setup.sh` / `pyproject.toml`: llama-cpp-python is now built straight from the pinned sdist (`pip install --no-binary llama-cpp-python llama-cpp-python==0.3.19`, needs cmake and a C/C++ compiler, about 4 min first time; the wheel attempt was removed because the index wheel is musl-linked), `diskcache` listed in the `llm` extra, optional bitnet.cpp commands documented in comments (not run by default). Idempotence verified: second run 5 s, exit 0, `pip freeze` identical, GGUF mtime/size unchanged.
- Final runs (commands in `docs/reproducibility.md`): `final-w32` (Qwen, all four tiers, cascade, 4000 examples) 681.0 s = 11.4 min, V 0.8628 / E 0.8630 / C 0.6627 / R 0.7013, float reference 0.9187; LLM stage 187.7 s, `numbers_verified=True`, `ids_verified=False` (tier R has no finding); `final-w8` (no LLM) 124.7 s, V 0.7348 / E 0.7348 / C 0.5400 / R 0.4785. `out/LATEST` = `final-w32`.
- Figures re-exported (`dlnn.viz.export --run final-w32 --llm-run final-w32 --secondary final-w8`, 64 files), stale `docs/figures/template/` removed; all seven notebooks re-executed with `DLNN_RUN=out/final-w32`, zero error outputs.
- Docs written: README.md (top level and docs/), concept_and_math, architecture (+ `figures/architecture.svg`), results (generated by `scripts/make_results_md.py`), reproducibility, honesty, deviations; banner added to plan.md.
- Tests: 131 passed, 1 skipped (PG test without DSN; passes with the DSN).

