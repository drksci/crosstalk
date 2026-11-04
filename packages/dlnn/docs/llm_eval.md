# LLM commentary evaluation (M3)

Reproduce: `.venv/bin/python -m dlnn.llm.eval --backend qwen --ablate` then
`.venv/bin/python -m dlnn.llm.eval --backend bitnet --ablate --merge` (about 20 min each on the shared 4-core box,
2 threads). Raw outputs of every card are in `models/llm/eval_report.json`.

## Setup

* 8 **synthetic** cards (`dlnn/llm/synth.py`, invented numbers shaped like modrec results, never written to `out/`):
  `adc6` (6-bit ADC, tier E), `loss30` (30 dB/us line loss, tier C), `iso15` (15 dB coupler isolation, tier R; tier C is a
  fallback), `vsat` (mis-biased diff pair / V_sat shift, tier C), `dil` (layer-3 dilation 8 instead of 16, tier R),
  `ctrl1`, `ctrl3` (no divergence), `ctrl2` (no divergence; C and R are stubs).
* Scoring is fixed (see `dlnn/llm/eval.py`; it was not changed between runs): an injected card is **correct** iff some
  finding has a `metric_id` that the card flags as material for the injected tier, lists the injected tier, and lists no
  tier other than {injected tier, V}. **Strict** additionally requires the single most severe metric. A control card is
  correct iff no finding or summary claims a divergence for a non-V tier (keyword rule `claims_divergence`).
  "stub flagged" = for cards containing stub/fallback tiers, the text names every such tier and says stub/fallback/placeholder.
* Numbers: `raw` = unverified numbers in the first model answer (plain mismatches, `metric=value` or unit-bound claims such
  as "32-bit" that the card does not support); `after retry (flagged)` = still unverified after the single retry (marked
  `verified:false`, never hidden); `exposed as verified` = unverified numbers sitting in text that is still marked verified
  (0 by construction, re-checked independently). "id/term failures" = unknown identifiers, a tier on the divergence list
  without a finding, or an `ok` tier called a stub/fallback.
* Qwen2.5-1.5B-Instruct Q4_K_M, llama-cpp-python 0.3.19, 2 threads, temperature 0.1, seed 1234, JSON-schema grammar.

## Results

<!-- eval:begin -->

### Summary

| backend | card section | attribution (all 8) | injected 5 (loose / strict) | controls 3 | stub flagged | numbers | unverified raw | unverified after retry (flagged) | exposed as verified | id/term failures raw / final | retried | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen (model) | with divergence list | 8/8 = 100 % | 5/5 / 2/5 | 3/3 | 2/2 | 48 | 0 | 0 | 0 | 4 / 4 | 2 | 9.12 |
| qwen (model) | without divergence list | 1/8 = 12 % | 0/5 / 0/5 | 1/3 | 1/2 | 60 | 5 | 5 | 0 | 19 / 19 | 5 | 9.05 |
| bitnet (model) | with divergence list | 3/8 = 38 % | 0/5 / 0/5 | 3/3 | 0/2 | 3 | 6 | 3 | 0 | 5 / 5 | 6 | 10.4 |
| bitnet (model) | without divergence list | 3/8 = 38 % | 0/5 / 0/5 | 3/3 | 0/2 | 1 | 1 | 1 | 0 | 0 / 0 | 1 | 10.43 |

### Per card

**qwen, divergence list on**

| card | injected | correct | strict | numbers (raw bad / final bad) | wall s |
|---|---|---|---|---|---|
| adc6: 6-bit ADC in the emulated tier | E | True | True | 16 (0 / 0) | 88.2 |
| loss30: 30 dB/us line loss in the circuit tier | C | True | False | 8 (0 / 0) | 120.6 |
| iso15: 15 dB coupler isolation in the RF tier (circuit tier is a fallback) | R | True | False | 7 (0 / 0) | 132.4 |
| vsat: mis-biased diff pair (V_sat shift) in the circuit tier | C | True | True | 11 (0 / 0) | 64.0 |
| dil: wrong dilation (layer 3 tap spacing 8 instead of 16) in the RF tier | R | True | False | 6 (0 / 0) | 49.3 |
| ctrl1: control: no divergence | none | True | True | 0 (0 / 0) | 26.2 |
| ctrl2: control: no divergence, circuit and RF tiers are stubs | none | True | True | 0 (0 / 0) | 31.5 |
| ctrl3: control: no divergence, different seed | none | True | True | 0 (0 / 0) | 26.6 |

**qwen, divergence list off**

| card | injected | correct | strict | numbers (raw bad / final bad) | wall s |
|---|---|---|---|---|---|
| adc6: 6-bit ADC in the emulated tier | E | False | False | 0 (0 / 0) | 118.9 |
| loss30: 30 dB/us line loss in the circuit tier | C | False | False | 4 (0 / 0) | 102.1 |
| iso15: 15 dB coupler isolation in the RF tier (circuit tier is a fallback) | R | False | False | 0 (0 / 0) | 120.5 |
| vsat: mis-biased diff pair (V_sat shift) in the circuit tier | C | False | False | 4 (0 / 0) | 102.6 |
| dil: wrong dilation (layer 3 tap spacing 8 instead of 16) in the RF tier | R | False | False | 4 (0 / 0) | 92.7 |
| ctrl1: control: no divergence | none | False | False | 32 (5 / 5) | 159.0 |
| ctrl2: control: no divergence, circuit and RF tiers are stubs | none | True | True | 8 (0 / 0) | 56.5 |
| ctrl3: control: no divergence, different seed | none | False | False | 8 (0 / 0) | 114.0 |

**bitnet, divergence list on**

| card | injected | correct | strict | numbers (raw bad / final bad) | wall s |
|---|---|---|---|---|---|
| adc6: 6-bit ADC in the emulated tier | E | False | False | 0 (0 / 0) | 74.5 |
| loss30: 30 dB/us line loss in the circuit tier | C | False | False | 0 (0 / 0) | 73.5 |
| iso15: 15 dB coupler isolation in the RF tier (circuit tier is a fallback) | R | False | False | 0 (0 / 0) | 78.1 |
| vsat: mis-biased diff pair (V_sat shift) in the circuit tier | C | False | False | 0 (0 / 0) | 75.0 |
| dil: wrong dilation (layer 3 tap spacing 8 instead of 16) in the RF tier | R | False | False | 0 (0 / 0) | 76.2 |
| ctrl1: control: no divergence | none | True | True | 0 (0 / 0) | 33.9 |
| ctrl2: control: no divergence, circuit and RF tiers are stubs | none | True | True | 3 (6 / 3) | 85.2 |
| ctrl3: control: no divergence, different seed | none | True | True | 0 (0 / 0) | 33.2 |

**bitnet, divergence list off**

| card | injected | correct | strict | numbers (raw bad / final bad) | wall s |
|---|---|---|---|---|---|
| adc6: 6-bit ADC in the emulated tier | E | False | False | 0 (0 / 0) | 30.3 |
| loss30: 30 dB/us line loss in the circuit tier | C | False | False | 0 (0 / 0) | 30.1 |
| iso15: 15 dB coupler isolation in the RF tier (circuit tier is a fallback) | R | False | False | 0 (0 / 0) | 116.9 |
| vsat: mis-biased diff pair (V_sat shift) in the circuit tier | C | False | False | 0 (0 / 0) | 29.8 |
| dil: wrong dilation (layer 3 tap spacing 8 instead of 16) in the RF tier | R | False | False | 0 (0 / 0) | 29.9 |
| ctrl1: control: no divergence | none | True | True | 0 (0 / 0) | 29.5 |
| ctrl2: control: no divergence, circuit and RF tiers are stubs | none | True | True | 1 (1 / 1) | 60.9 |
| ctrl3: control: no divergence, different seed | none | True | True | 0 (0 / 0) | 29.3 |

<!-- eval:end -->

## Reading the results honestly

**Acceptance targets (plan section 7): >= 75 % attribution and 0 unverified numbers after retry/flagging.** Qwen with the
card as designed: 8/8 attribution (that is 5/5 injected cards, only 2/5 under the strict rule, plus 3/3 controls), 0 unverified numbers raw, 0 after the retry step, 0 exposed as verified. That number is
real but flattering, for these reasons:

* **The card does most of the attribution work.** It contains a Python-computed "MATERIAL DIVERGENCES FROM V" section
  (tier, diverging metrics, the tier's own models line, thresholds printed in the card). Qwen copies it. The strict column
  (the single most severe metric) is 2/5 because Qwen often picks `accuracy` or `top1_agreement_vs_virtual` instead of the
  first metric on the line; the loose rule (any metric the card flags as material for that tier) is what the plan's
  target measures.
* **Without that list the 1.5B model cannot find the divergent tier: 0 of 5 injected cards** (the "1/8" in the table is a control with no
  divergence, `ctrl2`, scored correct for not claiming one; the other two controls are wrong): it reproduces the shape of the
  in-context example ("E and R diverge ... C stays close") whatever the card says, and it gets only 1/3 controls right. This
  ablation is the real measure of what the small model contributes: fluent restatement and caveat wording, not discovery.
  It also produced 5 unverified numbers (kept, flagged `verified:false`, none shown as verified).
* **The guard flags what the model copies from the example.** With the list, Qwen still wrote "C is a FALLBACK" or "R is a
  FALLBACK" on cards where that tier has `status=ok` (the in-context example has a fallback tier). The `status_claim`
  check caught all of these (4 failures in 2 cards, `loss30` and `iso15`; still present after the one retry, therefore
  flagged, with `ids_verified:false`). On the real `modrec-w32-full` card
  (`models/llm/example_real_commentary.json`) it wrote "C's own models line lists a 32-bit absmax ADC" for a SPICE circuit
  tier; "32" does occur on the card (32 dB/us), so a plain number check passes, and only the unit-bound check
  ("32 bit" never occurs on the card) flags it. The attribution there is right (C and R named with `top1_agreement_vs_virtual`),
  the `likely_cause` text is wrong and is shown as unverified.
* **What the guard cannot catch**: a causal sentence whose numbers are all on the card and attached to the right units but
  that is still not supported by the tier's models line ("quantisation is the candidate cause" for C). `likely_cause` is
  copied text, not physical analysis. Do not read it as analysis.
* The synthetic injections are easy: one tier differs per card, by a lot; the real divergence on `modrec-w32-full` is
  large too (it comes from 32 dB/us of loss).
* Cards are 906 to 1119 real Qwen tokens (real modrec card 1453; limit 1500; digits tokenise one at a time). The prompt
  including system rules and the in-context example is about 2.4k tokens. Prefill 20 to 30 s, decode about 9 tok/s on 2
  threads (the box load was 5 to 10 from other workers; an earlier contended run measured 1 to 2 tok/s); 25 to 130 s per card.

## BitNet b1.58-2B-4T (optional backend, time-boxed)

Status: **built and runs, but unusable for this task in my setup (3/8 = the 3 controls only, achieved by writing almost
nothing).** Timeline: box started 02:15 UTC; clone, download, build and smoke test were done at 02:56, inside the 45
minutes; the evaluation runs afterwards were background jobs.

1. `git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git models/llm/bitnet.cpp` works (git over the
   proxy is reachable; the bundled llama.cpp fork is `isHuangXin/llama.cpp` at 390c307, build id b9918).
2. `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` (1.19 GB) downloaded with curl through the proxy.
3. `cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLAMA_BUILD_TOOLS=ON
   -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF`, then
   `cmake --build build --target llama-cli -j2` (about 20 min under load). `setup_env.py` is not needed for i2_s on x86.
4. This fork's `llama-cli` rejects `-no-cnv` ("please use llama-completion instead"), so `--target llama-completion` was
   built too. It supports `-jf` (JSON-schema grammar). `BitNetBackend` (subprocess, 2 threads, same prompt/guard, BitNet
   "System:/User:/Assistant:" `<|eot_id|>` format) is in `dlnn/llm/backend.py`.
5. Speed is fine (about 10 tok/s decode). Quality is not: the log prints `ftype : Q1_0` for the i2_s file and most outputs
   are 30-token fragments such as "E is not modelled by R" or "E has not evidence of V" (0 findings in all 16 runs; one run
   hit the 899-token limit), 0/5 injected cards attributed, and the few numbers it wrote ("E is a 5-bit, R is a 2-bit",
   "E=0.0003") were flagged. I did not find out whether the cause is the chat format, the newer llama.cpp fork mishandling
   the i2_s tensors, or the model being too weak for a 2.4k-token structured prompt. Not pursued, per the time box.

`--llm bitnet` works mechanically (`get_backend("bitnet")`) but must not be used for the demo; Qwen is the model backend.
