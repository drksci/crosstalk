# LLM commentary (M3)

A tiny local language model writes a short, grounded commentary on the side-by-side results of the four fidelity
tiers (V virtual, E emulated, C circuit, R RF). **It is a commentator only.** It does not simulate anything, it is not
the reference model, and it has not "analysed" anything beyond a text card of at most ~1.5k tokens that Python rendered
from `compare.json`. Every number it writes is checked against that card.

```
compare.json --card.py--> card.txt --prompt.py--> LLM (grammar-constrained JSON) --verify.py--> commentary.json
                                                      ^---- ONE retry listing the mismatches ----|
```

## Files

| file | role |
|---|---|
| `dlnn/llm/card.py` | `render_card(compare, rank=True)`: tier status, tool, `fidelity.models` and `fidelity.omits` (stub/fallback tiers are flagged `<<< STUB / PLACEHOLDER`), metrics with deltas vs V, `acc_per_snr`, a Python-computed *material divergences* list (thresholds printed in the card), sweep knees (from `compare.py`), warnings. |
| `dlnn/llm/prompt.py` | system rules, one in-context example (an invented regression card with a different defect than any eval card), JSON schema whose `metric_id`/`tiers` enums are restricted to the ids on the card. |
| `dlnn/llm/backend.py` | `NullBackend` (template, `source:"template"`), `GuardedBackend` (generate, verify, one retry, flag), `QwenBackend` (llama-cpp-python), `BitNetBackend` (optional), `get_backend(name)`. |
| `dlnn/llm/verify.py` | pure functions: number extraction, matching against the card, id validation, sentence-level flagging, retry message. |
| `dlnn/llm/synth.py`, `dlnn/llm/eval.py` | 8 synthetic cards with injected divergences, scorer, `python -m dlnn.llm.eval`. |

## Output (`llm/commentary.json`, schema `dlnn-commentary/1` plus extra keys)

`summary, findings[{metric_id, tiers, observation, likely_cause, verified}], caveats, questions`, plus
`source` (`"model"` for Qwen/BitNet, `"template"` only for the null backend, which says so in its own text),
`backend`, `numbers_verified` (false if anything is still unverified after the retry), `summary_verified`,
`caveats_verified[]`, `questions_verified[]`, `unverified[]` (where/kind/token/sentence), `verification`
(raw vs final counts, whether a retry happened), `llm` (model name, GGUF file and sha256, prompt sha256, seed,
temperature, decoding mode, tokens/s, threads) and `raw_first_answer` (the pre-retry answer when a retry happened).
Viewers should dim anything with `verified:false`.

## Guard rules (`verify.py`)

* every number in the text must equal some number printed in the card within `max(1 %, half a unit of the card's
  printed precision)`, or be a correct rounding of it to >= 2 decimals; sign is ignored; `91 %` matches `0.91`;
  integers printed in the card must match exactly; the number of tiers is the only derived number allowed;
* identifier-like tokens (`top1_agreement_vs_virtual`) are removed before number extraction; snake_case tokens that do
  not occur in the card are flagged `unknown_id`;
* scoped checks: `metric_id=0.75` (also "of/is/at") must match a value the card prints for that metric, and unit-bound
  numbers ("6-bit", "30 dB/us", "60 mV") must occur with the same unit in the card ("32-bit" is rejected when the card only has
  "8-bit" and "32 dB/us");
* consistency: every tier on the card's divergence list needs a finding (`missing_tier`), and an `ok` tier may not be called a
  stub/fallback/placeholder (`status_claim`); these set `ids_verified:false`, not `numbers_verified:false`;
* `metric_id` and `tiers` must be valid under the contract AND present in the card (the decoding grammar already
  enforces this, `verify` re-checks);
* one retry: the mismatch list is appended and the model rewrites the JSON; the retry replaces the first answer
  unless it has more failures; whatever still fails is flagged `verified:false` and `numbers_verified:false`.

Limitation: outside the `metric_id=value` and unit patterns a number is only checked for *being on the card*, not for
being attached to the right tier. The model can still misattribute a real number or write a `likely_cause` that its
numbers do not support; the eval table counts attribution separately and `docs/llm_eval.md` lists what the guard cannot see.

## Backends

* **Qwen2.5-1.5B-Instruct Q4_K_M** (`models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`, ~1.1 GB, from
  `Qwen/Qwen2.5-1.5B-Instruct-GGUF`) via `llama-cpp-python==0.3.19`, `n_threads=2`, `temperature=0.1`, `seed=1234`,
  grammar from a JSON schema. Select with `--llm qwen`.
* **BitNet b1.58-2B-4T** (optional): see `docs/llm_eval.md` for what happened.
* **NullBackend** (`--llm none`): deterministic template. Always labelled `source:"template"`.

### Installing llama-cpp-python 0.3.19 on this box

`pip install llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` installs a
`cp313-linux_x86_64` wheel whose `.so` files link against `libc.musl-x86_64.so.1` (a musl build mislabelled as plain
linux); `import llama_cpp` then fails on this glibc system. Working path used here: build the pinned sdist from source
(about 4 minutes with 2 jobs):

```
CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS="-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF" \
  .venv/bin/pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python==0.3.19"
```

Weights: `curl -L -o models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf`
(through the configured proxy; the sha256 is in `models/llm/*.sha256`).

## Wiring into the pipeline

`dlnn/pipeline.py` already calls `llm_backend.get_backend(llm).comment(cmp, card)`; `get_backend("qwen")` and
`get_backend("bitnet")` now exist, so `python run.py --llm qwen` works without further changes. If the GGUF or the
wheel is missing, `get_backend` raises `BackendUnavailable` (it never silently falls back to the template).

## What this does and does not show

It shows that a 1.5B quantised model can restate a structured comparison with its numbers machine-checked. It does not
show physical insight: `likely_cause` strings are copied from the tiers' own `models`/`omits` text on the card. Results
and failure counts are in `docs/llm_eval.md`.

## Real example (`out/modrec-w32-full`, Qwen, 2 threads)

Card: `models/llm/example_real_card.txt` (1453 Qwen tokens). Full output: `models/llm/example_real_commentary.json`.

* Summary: "C diverges from V in accuracy, macro_f1, top1_agreement_vs_virtual, and logit_nmse_vs_virtual; R diverges in ..."
* Finding 1: metric `top1_agreement_vs_virtual`, tier C, "C has top1_agreement_vs_virtual=0.662 (-0.338 versus V)." (numbers
  match the card); Finding 2: same metric, tier R, 0.709 (-0.291).
* Both `likely_cause` sentences claim "a 32-bit absmax ADC" (copied from the in-context example's shape; not in C's or R's
  models line). The guard flagged both findings `verified:false`, the top-level `numbers_verified` is false, one retry did
  not fix it. The correct reading of that card is: C and R diverge strongly from V, E matches V, and both lossy tiers model
  32 dB/us of line loss; the model did not say that.
