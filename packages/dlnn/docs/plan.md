> **Original plan; see deviations.md for what changed.**

# Delay-Line Neural Network (DLNN): four-fidelity simulation demo, technical plan

Authored by Fable (planning pass). Supersedes the earlier 433 MHz OOK-receiver plan (discarded).
Constraints from the user: Jupyter notebooks are the modelling/simulation surface; docs are local Markdown under `docs/`; nothing is published anywhere (no Cloudflare, no GitHub push); the tiny LLM is only a commentator on the side-by-side outputs; the system being modelled is a **neural net**, not a language model.

## 0. Sandbox probe results (verified at planning time)

| Item | Status |
|---|---|
| torch 2.15.0+cpu cp313 | resolvable via `--index-url https://download.pytorch.org/whl/cpu` |
| scipy, scikit-rf 2.2, matplotlib, plotly | resolvable |
| ngspice 42, iverilog 12 | in apt (`apt-get -s install` ok; running as root) |
| llama-cpp-python | cp313 wheel **0.3.19** on `abetlen.github.io/llama-cpp-python/whl/cpu`; MUST pin `==0.3.19` (unpinned = source build). Install not yet run |
| HF `microsoft/bitnet-b1.58-2B-4T-gguf`, `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | reachable |
| bitnet.cpp build | assumed; needs clang>=18 + cmake>=3.22 (present); unverified |
| GitHub release binaries | NOT reachable (403); do not rely on |
| numpy 2.5.3, pandas, 4 cores, 15 GB RAM | present |

All network goes via the proxy (see /root/.ccr/README.md). Never disable TLS verification.

## 1. Core idea and math

**Chosen: ternary tapped-delay-line convolution network ("TDL-Net")**, plus a random-tap delay reservoir used only as the M0 walking-skeleton model.

Layer l with input lines i, output lines o, K taps, dilation d_l, symbol period T_s:

```
u_o[n] = sum_i sum_k W[o,i,k] * x_i[n - k*d_l]        W in {-1, 0, +1}
y_o[n] = g_l * tanh(u_o[n] / V_sat)                   g_l = per-layer scale (amplifier gain)
```

| Math | Physical element | Note |
|---|---|---|
| delay k·d_l·T_s | coax length = k·d_l·T_s·v_p (RG-58: 19.8 cm/ns) | dilation = longer line |
| W = +1 / 0 / -1 | coupler tap into combiner / no tap / tap through 180° hybrid | BitNet b1.58 ternary maps 1:1 to tap polarity |
| sum over (i,k) | N-way Wilkinson combiner | passive, -10·log10(N) dB |
| tanh | BJT diff pair: I = I_EE·tanh(v/2V_T), V_T = 25.9 mV | natively tanh |
| g_l, V_sat | amplifier gain / bias | compensates loss, adds NF |
| global average pool | RC integrator, τ ≈ N·T_s | passive |
| readout linear + softmax | ADC + software | honestly digital |

Signal encoding: baseband PAM pulse train per channel (amplitude = sample, ±0.5 V, root-raised-cosine, 10 ns symbol, ~100 MHz BW). No carrier.

Training: ternary QAT offline in PyTorch, BitNet b1.58 rule `W_t = clip(round(W/γ), -1, 1)`, γ = mean|W| per layer, straight-through estimator; activations absmax-quantised to 8 bits during training (ADC in the loss). γ becomes amplifier gain g_l. Bias = DC offset at amplifier input.

Physical limits (coax, T_s = 10 ns): time-bandwidth product ~10^2-10^3 (loss-bound); RG-58 loss ≈ 32 dB/µs at 100 MHz (LMR-400 ≈ 14, fibre ≈ 0), so the layer-3 dilation-16 tap (1.12 µs) needs re-amplification; sweep axis is "dB per µs" so results are medium-independent; skin-effect dispersion smears 10 ns pulses ~1-2 ns/100 m; NF 3 dB per amp; activations need ~5-6 ENOB; coupler amplitude ±0.5 dB, phase ±5°, isolation 20-30 dB (a "0" leaks 3-10 %); crosstalk -35 dB coax. Passive-only cannot do gain, nonlinearity, normalisation, softmax; minimal hybrid = passive tapped lines + one saturating amp per line per layer + one ADC at the end.

## 2. Task and model

**Headline task: 8-class modulation recognition** on synthetic I/Q (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK), 128 samples at 8 sps, random phase, CFO ±1 % of fs, timing offset, AWGN at SNR in {0,5,10,15,20} dB; 16k train / 4k test, generated in-repo, deterministic, sha256-recorded. **Sanity task: NARMA10** (1 in, 1 out, NMSE).

**TDL-Net-S:** input 2 lines (I,Q) -> L1 2->8, K=8, d=1 -> L2 8->8, K=8, d=4 -> L3 8->8, K=8, d=16 -> RC average pool -> ternary linear 8->8 -> softmax. Receptive field 1+7·(1+4+16)=148 >= 128. 128+512+512+64 = **1,216 ternary weights** + 4 scales + 32 biases.

Baseline: identical topology float32 (equal size). Optional float MLP on flattened samples at equal parameter count.

Recipe: Adam 3e-3, cosine decay, 20 epochs, batch 256, ~3-5 min on 4 cores. Export `model.npz` (int8 ternary weights, γ, biases, V_sat) consumed by all tiers.

Targets: float >= 92 % at >=10 dB and >= 80 % mixed-SNR; ternary >= 85 % at >=10 dB and within 7 points of float; NARMA10 NMSE <= 0.15 float, <= 0.25 ternary.

## 3. The four tiers

| Tier | Delay-line model | Tool | Simulates | Not simulated | Fallback |
|---|---|---|---|---|---|
| V virtual | ideal unit delays, exact ternary sums, exact tanh | numpy float32 (mirrors torch eval) | the math | everything physical | none |
| E emulated | shift registers; int8 input (8-bit absmax ADC), int2 weights, int32 accumulate, 256-entry tanh LUT -> int8, per-layer Q0.15 scale, pool = sum >> log2 N | numpy fixed-point golden model + **iverilog** RTL `tdl_layer.v` + testbench | quantisation, rounding, saturation, bit-exact arithmetic | analog effects | numpy fixed-point only, labelled "bit-true model (no RTL)" |
| C circuit | lossy RLGC transmission lines (ngspice LTRA / `Y` TXL; RG-58 L=250 nH/m, C=100 pF/m), ideal ±1 VCVS taps into a summing node, BJT diff-pair tanh, RC integrator | **ngspice 42** batch + raw-file reader; runs on a *unit cell* (one 8-tap line + combiner + amp) | per-tap impulse response (loss, delay error, ISI), true diff-pair transfer, bias drift, RC pool | layout parasitics, frequency-dependent R, full network end-to-end | numpy closed-form RLGC line, labelled "analytic" |
| R RF | `skrf.media.Coaxial` lossy dispersive lines per tap, coupler 3-ports (coupling, isolation), 180° hybrid, Wilkinson N-way, amplifier (gain, NF, Rapp AM/AM), mismatch Γ, crosstalk, AWGN per stage | **scikit-rf** builds H_{oik}(f) -> IFFT -> impulse responses; numpy convolution runs the full set | channel-level loss, dispersion, reflections, noise, saturation, leakage | transistor-level, EM | numpy RLGC + lumped params (shared with C fallback) |

**Full-set vs subset rule.** V, E, R run all 4k test examples. C runs SPICE end-to-end on layer 1 for 16 examples (acceptance of extraction), plus `.ac` tap-path extraction and `.dc` diff-pair extraction; the full test set is then run by the **circuit-calibrated behavioural model**, labelled exactly that. Nothing is called a "SPICE result" unless SPICE computed it.

**Cascade (`cascade.py`):** C `.ac` -> per-tap complex H_k(f) and `.dc` -> (V_sat, g) -> `params/circuit_extracted.json` -> consumed by R (replacing analytic coax loss when `--cascade`) and by V as a "V_cal" sensitivity check. E consumes nothing from C/R. Provenance in `inputs_from`.

**Common output contract** (`out/<run>/tiers/<tier>.json`, canonical; arrays in sibling `.npz`): `schema_version, run_id, tier, tool{name,version}, model_sha256, dataset_sha256, inputs_from[], n_examples, arrays{logits[N,C], probs[N,C], pred[N], trace_activations{layer:[16,8,T]}, waveforms{example_id:[lines,T_fine]}}, metrics{accuracy, macro_f1, acc_per_snr[], top1_agreement_vs_virtual, mean_kl_virtual_vs_tier, logit_nmse_vs_virtual, nmse}, sweeps[{axis: snr_db|adc_bits|loss_db_per_us|tap_amp_err_db|isolation_db, values[], accuracy[], mean_kl[]}], fidelity{models[], omits[]}, runtime_s, warnings[], status` where status in ok|skipped|stub|fallback. `compare.py` merges tiers into `compare.json`, the single input to figures, docs tables and the LLM card. A failed tier emits `status:"skipped"` and the pipeline continues. Stubs/fallbacks must be labelled; never present them as the real tool.

## 4. The small LLM: commentator only

Reads a <=1.5k-token card rendered from `compare.json` (per-tier accuracy, agreement, KL, sweep knee points computed in Python, each tier's `fidelity.omits`) and returns grammar-constrained JSON `{summary, findings:[{metric_id, tiers, observation, likely_cause}], caveats, questions}`. Not the reference model, not a teacher.

Backend: **Qwen2.5-1.5B-Instruct Q4_K_M via `llama-cpp-python==0.3.19`**. **BitNet b1.58-2B-4T** via bitnet.cpp as optional second backend, time-boxed 45 min. `NullBackend` renders a template (clearly labelled as template, never as model output).

Guardrails: `verify.py` extracts every number in the output and matches to card values within rounding; mismatches -> one retry listing the mismatch, then flagged "unverified" and dimmed; metric/tier ids validated against the contract. Eval: 8 synthetic cards with injected divergences (6-bit ADC, 30 dB/µs loss, 15 dB isolation, mis-biased diff pair, wrong dilation); target >= 75 % correct metric+tier attribution, 0 unverified numbers.

## 5. Visuals (Jupyter-first)

matplotlib for all exported figures (PNG + SVG, light and dark `.mplstyle`, transparent background, `dataviz` skill palette); plotly optional inside notebooks. Figures: (1) network/tap diagram (lines as bars, tap polarity markers, dilation lengths in metres); (2) signal trace strip: same example's line voltages after each layer, V/E/C/R overlaid with residual vs V; (3) tap-path impulse responses and |H(f)| from C and R, eye diagram at layer-1 combiner; (4) per-tier class-probability bars for 4 examples with KL; (5) accuracy and mean-KL vs SNR / ADC bits / loss per µs / isolation, tiers as series; (6) confusion-matrix diff per tier vs V; (7) prediction-agreement heatmap tier x tier; (8) LLM commentary panel, unverified dimmed.

## 6. Layout (project root /home/user/dlnn)

```
dlnn/
  pyproject.toml     # numpy scipy scikit-rf matplotlib jupyter nbconvert papermill nbformat; extras: train(torch), llm(llama-cpp-python==0.3.19), pg(psycopg)
  setup.sh           # apt ngspice iverilog; venv + pip; model download (HF); records installed vs failed
  run.py             # python run.py --task modrec|narma10 --tiers V,E,C,R --cascade --llm qwen|bitnet|none --db sqlite|postgres
  dlnn/contract.py   # TierResult, METRIC_IDS, JSON schema, validate()
  dlnn/data/{modrec.py,narma10.py}
  dlnn/model/{tdlnet_torch.py,ternary.py,export.py,reservoir.py}
  dlnn/tiers/{virtual.py,emulated.py,circuit.py,rf.py}   # run(model_npz, dataset, cfg, outdir) -> TierResult
  dlnn/{cascade.py,compare.py,sweeps.py}
  dlnn/llm/{backend.py,card.py,prompt.py,verify.py,eval.py}
  dlnn/viz/{figures.py,styles/{light,dark}.mplstyle}
  dlnn/db/{schema.sql,load.py}
  hw/spice/{unit_cell.cir,coax_rg58.lib,diffpair.lib}  hw/rtl/{tdl_layer.v,tb_tdl_layer.v}
  notebooks/00_overview 01_task_and_model 02_virtual 03_emulated 04_circuit 05_rf 06_compare_and_llm  (.ipynb, executed with saved outputs)
  docs/{plan,quickstart,concept_and_math,tiers,results,llm_commentary,reproducibility,honesty,build-log}.md
  out/<run_id>/{model.npz,tiers/*.json,*.npz,params/,compare.json,figures/,llm/}
```

Postgres (optional, SQLite default, same DDL; JSON remains canonical): runs, tier_results, sweeps, predictions, llm_outputs.

## 7. Milestones

| M | Owns | Acceptance (numeric) | Parallel |
|---|---|---|---|
| **M0 walking skeleton** | contract, data generators, `reservoir.py` (random ternary TDL + ridge readout on NARMA10, numpy only), V and E (numpy) real, C and R as V+AWGN stubs labelled `stub`, compare, NullBackend, figures 4-5, SQLite, notebook stubs, setup.sh | `python run.py --task narma10 --llm none` -> 4 valid tier JSONs, compare.json validates, NMSE_V <= 0.40, E within 10 % of V | gate |
| **M1 model** | `tdlnet_torch.py, ternary.py, export.py`, modrec generator final, baseline | `python -m dlnn.model.train --task modrec` <= 10 min: float >= 80 % mixed-SNR, ternary >= 70 % mixed-SNR and >= 85 % at >=10 dB; V tier reproduces torch logits to 1e-4 | with M2a/b/c, M3 |
| **M2a emulated** | `emulated.py`, `hw/rtl/*`, fixed-point spec | iverilog tb vs numpy golden 100 % bit-exact on 64 examples x layer 1; E vs V top-1 agreement >= 95 % at 8 bits; ADC sweep 4-10 bits | yes |
| **M2b circuit** | `circuit.py`, `hw/spice/*`, extraction, `cascade.py` | ngspice unit-cell `.tran` layer-1 activations vs V on 16 examples NMSE <= 0.05 (noise off); diff-pair tanh fit R² >= 0.99; <= 10 min | yes |
| **M2c RF** | `rf.py`, `sweeps.py` | at 30 dB SNR, 0 dB/µs, ideal couplers: top-1 agreement vs V >= 98 %; sweeps on 4 axes; accuracy monotone in SNR | yes |
| **M3 LLM** | `dlnn/llm/*` | `python -m dlnn.llm.eval` >= 75 % attribution, 0 unverified numbers; BitNet time-boxed | yes (works on M0 stub output) |
| **M4 integration** | `--cascade`, figures 1-3, 6-8, executed notebooks, docs, Postgres loader | `python run.py --task modrec --tiers V,E,C,R --cascade --llm qwen` end-to-end <= 25 min; all notebooks execute via nbconvert; `--db postgres` loads a run | after M1-M3 |

## 8. Novelty and honesty

Prior art: tapped-delay-line transversal filters and SAW matched filters, delay-based reservoir computing (Appeltant 2011; Paquot 2012; Larger 2012; Brunner 2013), analog in-memory and photonic accelerators, BitNet b1.58 (Ma et al. 2024), TCN/S4/Hyena long convolutions. Modestly new here: explicit mapping of BitNet-style ternary weights onto tap polarity in dilated tapped delay lines with a natively-tanh diff-pair activation, and a reproducible four-fidelity harness with SPICE-extracted parameters cascading into RF/behavioural tiers plus verified LLM commentary. Not new: tapped delay lines as FIR filters, delay reservoirs.

May claim: simulation-only cross-fidelity consistency and degradation curves of a tiny ternary delay-line network on synthetic data.
Must NOT claim: any hardware measurement; vendor-accurate SPICE models (generic BJT, RLGC coax); EM/layout effects; that the LLM "analysed" anything (it comments on a card); competitiveness with digital classifiers; that the full network is buildable at RG-58 loss without repeaters.

Risks / cut lines: ternary accuracy too low -> per-output-line scale; ngspice LTRA slow/unstable -> `Y` TXL -> analytic lines; iverilog mismatch -> tighten rounding spec, keep numpy fixed-point as E; llama-cpp wheel fails -> source build -> NullBackend; BitNet build -> cut first. Cut order: BitNet -> Postgres -> crosstalk/mismatch params -> eye diagram -> NARMA10 track (keep M0 only).

## 9. Defaults taken (questions for Blake, not blockers)

Coax RG-58 at 10 ns symbols (loss/µs sweep covers other media); modrec + NARMA10; Qwen first, BitNet time-boxed; Postgres optional (SQLite default); full run <= 25 min.
