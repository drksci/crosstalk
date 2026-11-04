# Build log M3 (tiny-LLM commentary with number verification)

Worker log, UTC 2026-11-03. Commands are as run; outputs are copied from the terminal.

## 1. llama-cpp-python 0.3.19 (plan item 1)

```
$ .venv/bin/pip install llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  Downloading llama_cpp_python-0.3.19-cp313-cp313-linux_x86_64.whl (13.5 MB)
  Successfully installed diskcache-5.6.3 llama-cpp-python-0.3.19
$ .venv/bin/python -c "import llama_cpp"
OSError: libc.musl-x86_64.so.1: cannot open shared object file: No such file or directory
$ readelf -d .venv/.../llama_cpp/lib/libllama.so | grep NEEDED
 ... libstdc++.so.6, libgcc_s.so.1, libc.musl-x86_64.so.1
```

**Deviation / problem.** The wheel tagged `linux_x86_64` on that index is built against musl; it cannot load on this glibc
(Ubuntu 2.39) box. I briefly tried `apt-get install musl` (then removed it again: mixing two libcs in one process is not
something to rely on) and fell back to building the **same pinned version from the sdist**:

```
$ CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS="-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF" \
    .venv/bin/pip install --no-binary llama-cpp-python --no-cache-dir "llama-cpp-python==0.3.19" --force-reinstall --no-deps -v
  Building wheel for llama-cpp-python (pyproject.toml): finished with status 'done'
  Successfully installed llama-cpp-python-0.3.19
$ .venv/bin/python -c "import llama_cpp;print(llama_cpp.__version__)"
0.3.19
```

About 4 minutes with 2 jobs (gcc 13, `-march=native`, so the wheel is tied to this CPU). `setup.sh` / `pyproject.toml` (not mine)
should say: `pip install --no-binary llama-cpp-python llama-cpp-python==0.3.19` when the index wheel fails to import.
I also briefly installed `huggingface_hub` (not needed, uninstalled with its dependencies); final new pip packages:
`llama-cpp-python==0.3.19` and its dependency `diskcache==5.6.3`.

## 2. Weights

```
$ curl -sSL -C - -o models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf   # 1117320736 bytes
$ sha256sum models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
```

BitNet GGUF: `microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf` (1187801280 bytes) into `models/llm/bitnet/`.

## 3. Files written

`dlnn/llm/{card,prompt,verify,backend,synth,eval}.py`, `tests/test_llm.py`, `docs/llm_commentary.md`, `docs/llm_eval.md`,
`models/llm/{*.gguf,*.sha256,eval_report.json,example_real_card.txt,example_real_commentary.json,bitnet.cpp/,bitnet/}`.
`NullBackend` is unchanged (still `source:"template"`).

## 4. Pipeline hook: no change needed

`dlnn/pipeline.py` already does `llm_backend.get_backend(llm).comment(cmp, card)`; `backend.get_backend` (mine) now returns
`QwenBackend` for `"qwen"` and `BitNetBackend` for `"bitnet"`, so `python run.py --task modrec ... --llm qwen` works. The only
suggested edit outside my files is cosmetic: `run.py` help text says "(M3)". One caveat for the orchestrator:
`get_backend` raises `BackendUnavailable` (not a silent template fallback) if the GGUF or the wheel is missing; `pipeline.py` may
want to catch it and print a warning. Loading Qwen takes ~2 s, one commentary ~50 to 130 s on 2 shared threads.

## 5. Iterations on card and prompt (scoring code was never changed)

| step | observation | change |
|---|---|---|
| first run, free-form rules, no example | Qwen mislabelled R as "circuit", listed all four tiers in one finding, repeated sentences, ~92 s/card | card gets tier names in the TIERS lines, in-context example (invented regression card), rule "one finding per tier on the divergence list" |
| number tolerance bug | `half a unit of printed precision` let a card integer ("V=1", "8") match anything within 0.5, so 0.333 verified | integers printed in the card must match exactly; model rounding accepted only for >= 2 decimals |
| real tokenizer vs estimate | real `modrec-w32-full` card was 1794 Qwen tokens (digits tokenise one by one, ~2.4 chars/token), over the 1.5k limit; my estimate said 1405 | char budget 3500, `CHARS_PER_TOKEN=2.4`, acc_per_snr printed only for tiers that are not within 0.02 of V, thresholds line shortened, models/omits clipped; test uses the real tokenizer (1453 tokens for the real card, 906 to 1119 for the synthetic ones) |
| guard added after seeing real output | verified numbers but wrong semantics: "E models a 5-bit ADC" (8-bit on card), "R is a FALLBACK" (status ok), missing tier on the divergence list, "32-bit ADC" (32 only occurs as "32 dB/us") | scoped `metric_id=value` check, unit-bound number check, `missing_tier` and `status_claim` checks (reported as id/consistency failures, retry message lists them) |

## 6. BitNet (time box 02:15 to 03:00 UTC)

See `docs/llm_eval.md`. Short version: clone OK, `cmake` + clang-18 build OK (llama-cli, then llama-completion), runs at ~10 to
11 tok/s on 2 threads with grammar-constrained JSON, but outputs on the cards are degenerate; reported as unusable here.

```
$ git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git models/llm/bitnet.cpp     # ok
$ cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
$ cmake --build build --target llama-cli -j2          # [100%] Built target llama-cli
$ build/bin/llama-cli ... -no-cnv                     # "--no-conversation is not supported by llama-cli, please use llama-completion instead"
$ cmake --build build --target llama-completion -j2   # [100%] Built target llama-completion
$ llama-completion -m bitnet/ggml-model-i2_s.gguf -t 2 -c 2048 -n 60 --temp 0.1 --seed 1234 -no-cnv --no-warmup -jf schema.json -p "System: ..."
{ "greeting": "hello" } [end of text]      common_perf_print: eval time = 803.02 ms / 11 runs (13.70 tokens per second)
```

## 7. Final numbers

From `python -m dlnn.llm.eval --backend qwen --ablate` then `--backend bitnet --ablate --merge` (full table in
`docs/llm_eval.md`, raw outputs in `models/llm/eval_report.json`):

```
| backend | card section | attribution (all 8) | injected 5 (loose / strict) | controls 3 | stub flagged | numbers | unverified raw | after retry (flagged) | exposed | id/term raw / final | retried | decode tok/s |
| qwen   | with divergence list    | 8/8 = 100 % | 5/5 / 2/5 | 3/3 | 2/2 | 48  | 0 | 0 | 0 | 4 / 4   | 2 | 9.12  |
| qwen   | without divergence list | 1/8 = 12 %  | 0/5 / 0/5 | 1/3 | 1/2 | 60  | 5 | 5 | 0 | 19 / 19 | 5 | 9.05  |
| bitnet | with divergence list    | 3/8 = 38 %  | 0/5 / 0/5 | 3/3 | 0/2 | 3   | 6 | 3 | 0 | 5 / 5   | 6 | 10.4  |
| bitnet | without divergence list | 3/8 = 38 %  | 0/5 / 0/5 | 3/3 | 0/2 | 1   | 1 | 1 | 0 | 0 / 0   | 1 | 10.43 |
$ .venv/bin/python -m pytest -q tests/test_llm.py        # see section 8
```

The run order was: three card/prompt iterations, then a rerun of the whole eval after each guard change (status/missing-tier
checks, unit-bound check), because the guard decides which answers get a retry. The Qwen figures did not change between the
last two reruns.

## 8. Open problems

* Qwen 1.5B cannot discover divergent tiers itself (ablation 12 %); the product value is the verified restatement.
* `likely_cause` text is not physical analysis and the guard cannot validate its logic (see `docs/llm_eval.md`).
* BitNet gives degenerate output here (cause not investigated).
* `llama-cpp-python==0.3.19` has to be built from the sdist on glibc systems (the index wheel is musl-linked).
* In-context example leaks phrasing ("C is a FALLBACK", "ADC") into outputs; the guard flags it but a retry rarely fixes it.
