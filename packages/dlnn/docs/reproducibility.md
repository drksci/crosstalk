# Reproducibility

Everything is simulation on synthetic data. The numbers below were recorded on the box that produced the final runs (2026-11-03).

## Hardware and OS

| item | value |
|---|---|
| CPU | Intel Xeon (virtualised), **4 cores** (`nproc` = 4), shared with other jobs during development; nothing else of note ran during the final runs |
| RAM | 15 GB |
| OS | Ubuntu 24.04.5 LTS, glibc 2.39, kernel 6.18 (Firecracker VM) |
| load during the final runs | `final-w32`: load average 0.38 / 0.56 / 1.05 at start, 1.79 / 1.45 / 1.26 at the end (from `run_meta.json`); `final-w8`: 1.46 / 1.40 / 1.25 at start, 1.35 / 1.40 / 1.27 at the end; `uptime` just before the first final run: 0.38, 0.56, 1.05; after both: 0.81, 1.15, 1.20 |

Runtimes in other docs (10.2 min and 12.9 min for width 32) were measured while other jobs shared the box and are slower than the final runs below.

## Software versions

| component | version |
|---|---|
| Python | 3.13.18 |
| numpy / scipy / matplotlib / pandas | 2.5.4 / 1.18.1 / 3.11.2 / 3.1.0 |
| torch (CPU) | 2.15.0+cpu (training only) |
| scikit-rf | 2.2.0 |
| ngspice | 42 |
| Icarus Verilog (iverilog) | 12.0 (stable) |
| llama-cpp-python | 0.3.19, **built from the sdist** (index wheel is musl-linked and fails to import), diskcache 5.6.3 |
| gcc / cmake / clang | 13.3.0 / 3.28.3 / 18.1.3 (clang only for the optional bitnet.cpp build) |
| jupyter / nbconvert / ipykernel / pytest / jsonschema | 1.1.1 / 7.17.2 / 7.4.0 / 9.2.0 / 4.26.0 |
| PostgreSQL (only for `--db postgres`; not used by the final runs) | 16.15 server, psycopg 3.3.6 |
| LLM | Qwen2.5-1.5B-Instruct Q4_K_M, `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes |

`./setup.sh` records installed / already present / failed per component in the Environment block of `docs/build-log.md`. A re-run on this box took 5 s and left `pip freeze` unchanged and
the GGUF untouched (checked for M4c).

## Hashes

| artefact | sha256 |
|---|---|
| Qwen GGUF | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| dataset (generator output, 16k train / 4k test) | `e1b8e74480bdb325cc7e77066c545331f771fe20bf577e9e40ccd036af92b1d3` |
| width-32 ternary model, canonical (`model_sha256` in tier JSONs, `Model.sha256()`) | `3e67999dbaf59454d25d3a6c7272cd17d407dd322f4a37287b056626a6c514c0` |
| width-32 ternary model, file `models/tdlnet_modrec_w32_ternary.npz` (`sha256sum`) | `99a720e0df76f85b11c1004dd778ee5db40e05843b63aaa8b7330add3a0601e9` |
| width-8 ternary model, canonical | `44a6c540d1c84c0877c18bfd5d17a7f6973823cea3a7655aa07ad0ffc3043ace` |
| width-8 ternary model, file `models/tdlnet_modrec_ternary.npz` | `744e2c121de99c48ea8bad96d0f92f429c6005af83413a36860aedb82248e14e` |
| width-32 float baseline file | `b66a42e534939e7cbfd9328566c48c955f0c41e079f1aee202b2c0993ecdb9fb` |
| width-8 float baseline file | `af8799146b647d992ce48fca437a81427e3ddccf9408b852369102957717c773` |

(The canonical hash covers the arrays only and is independent of zip timestamps; the file hash is the plain `sha256sum`.)

## Seeds

| what | seed |
|---|---|
| dataset: train / test generation | 20260101 / 20260102 (disjoint) |
| training, width 32 | base seed 0, 1 restart, 60 epochs (`models/train_report_w32.json`) |
| training, width 8 | base seed 0, 3 restarts, restart chosen by TRAIN accuracy (the chosen seed is recorded in `models/train_report.json`) |
| sweeps (input AWGN) | `seed` argument of `sweeps.snr_sweep` (default 0), per-point offset |
| R tier random tap errors | seeded and fixed in the RF config (common random numbers across sweep points) |
| LLM | temperature 0.1, seed 1234, JSON-schema grammar, 2 threads |
| `scripts/r_ideal30.py` | noise seeds 0, 1, 2 |

## Commands (as run)

```bash
cd /home/user/dlnn
./setup.sh                                   # idempotent check, 5 s, exit 0

# final runs
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm qwen --db sqlite --out out --run-id final-w32
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none --db sqlite --out out --run-id final-w8 \
    --model models/tdlnet_modrec_ternary.npz
printf 'final-w32' > out/LATEST                  # run.py writes LATEST itself; final-w8 ran last, so it was pointed back by hand

# agreement criterion table (section 8 of results.md), output saved next to compare.json
.venv/bin/python scripts/r_ideal30.py models/tdlnet_modrec_w32_ternary.npz models/tdlnet_modrec_ternary.npz | tee out/final-w32/r_ideal30.txt

# figures (docs/figures/ and docs/figures/w8/), notebooks, results tables
.venv/bin/python -m dlnn.viz.export --run final-w32 --llm-run final-w32 --secondary final-w8
export DLNN_RUN=out/final-w32 DLNN_LLM_RUN=out/final-w32
for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=dlnn --ExecutePreprocessor.timeout=900 $n; done
.venv/bin/python scripts/make_results_md.py       # docs/results.md and the headline block of docs/README.md

.venv/bin/python -m pytest -q
```

Retraining the models is not part of the final run (the shipped `models/*.npz` were used): `.venv/bin/python -m dlnn.model.train --task modrec` (width 8, about 3.4 min) and
`... --width 32` (about 4 min), see `build-log-M1.md`. Rebuilding the optional bitnet.cpp backend: commands in the comments of `setup.sh` and in `llm_eval.md`.

## Runtimes of the final runs (seconds per stage, from `compare.json["timing"]`)

| stage | final-w32 (with Qwen) | final-w8 (no LLM) |
|---|---|---|
| dataset + model | 0.68 | 0.47 |
| V | 9.02 | 2.23 |
| E (numpy fixed point, sweeps, iverilog RTL on a subset) | 159.48 | 27.41 |
| C (ngspice unit cell + calibrated model, sweeps) | 92.43 | 50.62 |
| R (scikit-rf channel, five sweeps) | 217.51 | 32.35 |
| float reference | 1.65 | 0.91 |
| figures (written by the pipeline) | 11.48 | 10.64 |
| LLM stage (Qwen2.5-1.5B, 2 threads, includes one retry) | 187.71 | 0 (template) |
| **total (before database load)** | **680.97 (11.4 min)** | **124.66 (2.1 min)** |

The 25-minute budget of the plan is met for both. Per-tier runtimes in the tier JSONs (`runtime_s`) are slightly different from the stage times because the stages also include saving.
Post-run steps: `scripts/r_ideal30.py` 35 s; figure export 41 s; the seven notebooks 4, 6, 4, 25, 10, 34 and 6 s (`nbconvert --execute`).

## Notebooks

All seven (`00_overview` ... `06_compare_and_llm`) were re-executed against `out/final-w32` with kernel `dlnn`; every code cell ran and **no cell produced an error output**
(checked by reading the saved outputs with nbformat).

## Tests

`.venv/bin/python -m pytest -q` (after the final runs, with the PostgreSQL server `16/main` online but `DLNN_PG_TEST_DSN` unset): **131 passed, 1 skipped in 233.6 s**. The skip is
`test_postgres_load_roundtrip` ("no PostgreSQL test server: set DLNN_PG_TEST_DSN"). With
`DLNN_PG_TEST_DSN="dbname=dlnn host=/var/run/postgresql" .venv/bin/python -m pytest -q tests/test_pipeline_modrec.py -k postgres`: 2 passed (30 s), i.e. the Postgres round trip passes
against the local PG 16 server; that is the only Postgres verification there is.

## Things that make a re-run differ from the saved one

* The Qwen output is deterministic given the card and the build (temperature 0.1, fixed seed) but depends on the llama.cpp build (`-march=native`) and thread count; a different build or machine can
  give different text. The card itself comes from `compare.json`, so any change in the tiers changes the card.
* Runtimes depend on load (4 shared cores).
* ngspice, iverilog and scikit-rf versions are those above; other versions were not tested.
