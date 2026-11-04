# Quickstart

Every command below was run on this box (4 cores, 15 GB, Python 3.13) as part of M4a; outputs are in `docs/build-log.md`.
Simulation only, synthetic data, no hardware measurement.

## 1. Environment

```bash
./setup.sh                  # idempotent; records installed / failed components in docs/build-log.md (ENV block)
WITH_LLM=0 ./setup.sh       # skip llama-cpp-python and the 1.1 GB Qwen weights
WITH_PG=1 ./setup.sh        # also apt-install a local PostgreSQL server
source .venv/bin/activate   # or call .venv/bin/python directly
```

System tools: `ngspice` (tier C), `iverilog` (tier E RTL). Extras in `pyproject.toml`: `train` (torch), `rf` (scikit-rf, pandas),
`llm` (llama-cpp-python==0.3.19), `pg` (psycopg), `dev` (pytest).

## 2. Models (already shipped in `models/`)

| file | what | accuracy (test set, 4000) |
|---|---|---|
| `models/tdlnet_modrec_w32_ternary.npz` | **headline**: width-32 ternary TDL-Net | 86.3 % mixed SNR, 89.1 % at >= 10 dB |
| `models/tdlnet_modrec_w32_float.npz` | its float32 baseline (reference only) | 91.9 % mixed SNR, 95.5 % at >= 10 dB |
| `models/tdlnet_modrec_ternary.npz` | the plan-spec width-8 net (documented secondary) | 73.5 % mixed, 76.3 % at >= 10 dB, **drops 8PSK entirely** |

Retrain: `.venv/bin/python -m dlnn.model.train --task modrec` (about 3.4 min, width 8; see `docs/build-log-M1.md`).

## 3. Run the pipeline

```bash
# headline: modulation recognition, all four tiers, C -> R cascade, template commentary, SQLite
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none

# the width-8 secondary model
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none --model models/tdlnet_modrec_ternary.npz

# real LLM commentary (Qwen2.5-1.5B via llama-cpp-python, about 1 min extra), verified numbers
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm qwen

# PostgreSQL instead of SQLite (needs a server and psycopg; see section 5)
.venv/bin/python run.py --task modrec --tiers V,E --db postgres --pg-dsn "dbname=dlnn host=/var/run/postgresql"

# NARMA10 sanity task (regression)
.venv/bin/python run.py --task narma10 --llm none
```

Useful flags: `--tiers V,E` (any subset; V is always run), `--cr-subset N` (tiers C and R on a fixed stratified N-example subset,
0 = full test set), `--float-model PATH`, `--n-test N` (small generated test set, multiple of 40), `--run-id`, `--out`.

Output: `out/<run_id>/` with `tiers/{V,E,C,R}.json (+ .npz)`, `compare.json`, `params/circuit_extracted.json`, `reference.json`,
`timing.json`, `figures/`, `llm/{card.txt,commentary.json}`, `dlnn.sqlite`, `model.npz`, `args.json`. `out/LATEST` names the last run.
`compare.json["reference"]` is the float32 baseline (labelled a reference, not a tier); `compare.json["timing"]` has seconds per stage.

## 4. Look at the results

```bash
.venv/bin/python - <<'EOF'
import json
from dlnn.pipeline import latest_run
run = latest_run()
c = json.load(open(run / "compare.json"))
for m in ("accuracy", "top1_agreement_vs_virtual", "mean_kl_virtual_vs_tier"):
    print(m, c["metric_matrix"][m])
print("float baseline:", c["reference"]["accuracy"], " total seconds:", c["timing"]["total_s"])
EOF
sqlite3 out/<run_id>/dlnn.sqlite "select tier,status,n_examples,runtime_s from tier_results"   # if sqlite3 is installed
```

Notebooks `notebooks/00..06_*.ipynb` (built by `notebooks/_build.py`) execute with
`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dlnn notebooks/<name>.ipynb`.

## 5. PostgreSQL

SQLite is the default and always works. For `--db postgres` you need a reachable server and `psycopg` (`pip install 'psycopg[binary]'`).
A local throwaway server on this box (Ubuntu 24.04, PostgreSQL 16 from apt):

```bash
apt-get install -y postgresql                # or WITH_PG=1 ./setup.sh
pg_ctlcluster 16 main start
su postgres -c "createuser -s $(whoami) && createdb -O $(whoami) dlnn"     # peer auth over the unix socket, no password
export DLNN_PG_DSN="dbname=dlnn host=/var/run/postgresql"
```

The loader creates the same five tables as `dlnn/db/schema.sql` (`runs, tier_results, sweeps, predictions, llm_outputs`) and replaces
the run if it exists. Keep credentials out of files: pass the DSN with `--pg-dsn` or `$DLNN_PG_DSN`.

## 6. Tests

```bash
.venv/bin/python -m pytest -q                       # whole suite (see build log for timings and skips)
DLNN_PG_TEST_DSN="dbname=dlnn host=/var/run/postgresql" .venv/bin/python -m pytest -q tests/test_pipeline_modrec.py   # + live Postgres test
```
