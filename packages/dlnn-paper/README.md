# dlnn-paper: a delay-line neural network, simulated at four fidelities (reproducibility repo)

A research prototype. A ternary tapped-delay-line network for 8-class modulation recognition, simulated at four fidelities (ideal math V, bit-true fixed point + Verilog RTL E, SPICE-calibrated circuit C, RF/parasitics R and P), plus the designed (**not built**) TILE-8 board. **Everything here is simulation of a design; nothing is a measurement.** The accuracy on the synthetic set is not comparable to RadioML or any published benchmark.

The notebooks in `notebooks/` *are* the pipeline: they run the simulations themselves and write the data files the article reads (`docs/artifacts/data/`). `00_verify` re-checks every claim of the article.

| notebook | what it does | runtime* |
|---|---|---|
| [`01_small_nets_and_arena`](notebooks/01_small_nets_and_arena.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/01_small_nets_and_arena.ipynb) | trains the small DLNN and the arena baselines with lite settings (20 epochs, 1 seed; the shipped models used 60 epochs, 3 restarts), exports the arena data from the shipped models, checks the browser JavaScript forward passes against Python in `node` (skipped without node) | about 4 min (234 s) |
| [`02_spice_board`](notebooks/02_spice_board.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/02_spice_board.ipynb) | ngspice on the exported KiCad tile netlists (C tier, board mode): `.ac`/`.dc` of all three tiles at Q = 10 and 30, the SPICE widget data (`spice/*.json`), a whole-tile `.tran` of 3 test examples, the behavioural C tier on 4000 examples | about 1 min (72 s) |
| [`03_parasitics_montecarlo`](notebooks/03_parasitics_montecarlo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/03_parasitics_montecarlo.ipynb) | P tier (lumped parasitics + tolerances; **not RF at about 1 MHz**) through its API: `build_line`, `run_nominal`, a 50-trial Monte Carlo, four sweeps, the gain-trim finding F1, a 3^5 slider grid; compared with the stored 1000-board / 5^5 run | about 1 min (65 s) |
| [`04_scale_story`](notebooks/04_scale_story.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/04_scale_story.ipynb) | regenerates the cost model; tiles, inductors, area, cost for W in 8..64, D in 3, 4, 6; accuracy against size; labelled energy/latency estimates vs a conventional ANN; writes `scale_story.json`, `board_cost.json` | 2 s |
| [`00_verify`](notebooks/00_verify.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/00_verify.ipynb) | loads `docs/artifacts/data/claims.json` and checks every claim: re-read from its source, and (where the code can do it in minutes) recomputed from the shipped models; PASS/FAIL table with tolerances; no torch training | about 2.5 min (147 s) |

\* wall time of `jupyter nbconvert --execute` on a 4-core Linux box, fresh Python 3.12 venv, ngspice 42, run one after the other with nothing else running (see "Tested how"). `01` is the longest because it trains.

Replace `<owner>` in the badges and in the Colab setup cells (`https://github.com/<owner>/dlnn-paper`) once the repository is published; nothing has been pushed.

## Run it

Colab: open a notebook with the badge. The first cell (guarded by `if 'google.colab' in sys.modules`) runs `apt-get -qq install ngspice iverilog`, clones the repo and `pip install -r requirements.txt`. Run the notebooks in any order; `00_verify` is meant to be last.

Locally (Ubuntu/Debian, Python 3.12):

```bash
sudo apt-get install ngspice iverilog nodejs      # nodejs only for the JS check in notebook 01
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
for n in 01_small_nets_and_arena 02_spice_board 03_parasitics_montecarlo 04_scale_story 00_verify; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/$n.ipynb; done
```

Environment variables: `DLNN_LITE_EPOCHS` (01, default 20), `DLNN_TRAN_N` (02, number of whole-tile SPICE transients, default 3, about 20 s each), `DLNN_FULL=1` (03: also run the long batch: 1000 boards on 800 examples and the 5^5 grid, about 10 minutes, and overwrite `docs/artifacts/data/sweeps_grid.json`), `DLNN_VERIFY_CR_N` (00, size of the fresh C/R subset, default 200, `0` skips it), `DLNN_VERIFY_STRICT=1` (00 raises on any FAIL).

## Layout

| path | content | licence |
|---|---|---|
| `dlnn/` | the package: data generator, models, tiers V/E/C/R/P, web exporters; `dlnn/paper.py` is the notebooks' bookkeeping | MIT |
| `hw/board/` | board generators (`gen_tile.py`, `derive_params.py`, `cost_model.py`, ...), `params.json`, `cost.json`, `parts.csv`, **exported tile netlists** `tile/layer{1,2,3}/tile.cir` | MIT (code), CERN-OHL-P-2.0 (netlists, data) |
| `hw/spice/`, `hw/rtl/` | datasheet-derived/generic SPICE models, Verilog of the fixed-point layer | CERN-OHL-P-2.0 |
| `models/` | only the models the article uses (`.npz`, ternary weights), reports; `expected/hashes.json` has their sha256 | MIT |
| `docs/artifacts/` | the article (`article/`), its widgets (`assets/`), `board/` (viewer files), **`data/` (what the notebooks write and the article reads)**, `data/evidence/` (stored run files the claims point to, e.g. `out/final-w32/compare.json`, and per-example predictions) | CC BY 4.0 (text), MIT (JS) |
| `expected/` | `data/` = snapshot of the article data before the notebooks run (they are compared against it), `hashes.json`, `make_hashes.py` | |
| `tests/js/` | headless node checks of the JS forward passes | MIT |
| `LICENSE`, `LICENSE-HARDWARE`, `LICENSE-TEXT`, `LICENSES.md` | MIT, CERN-OHL-P-2.0, CC BY 4.0 | |

Not included (too large, or needs tools Colab does not have): KiCad schematics/PCB files, Gerbers, the `.pt` training checkpoints, the LLM commentary, `out/` run directories. The tile netlists are the committed `kicad-cli` exports (their sha256 are checked against `docs/artifacts/data/spice/manifest.json`), so no KiCad is needed.

## Claim -> notebook map

`00_verify` covers all claims. "Recomputed" = recomputed from models/data/code; "re-read" = read again from the source file (or evaluated from its formula); the notebook in the second column is the one that *produces* the data the claim reads, where there is one.

| claim ids | what | produced by | in `00_verify` |
|---|---|---|---|
| `p1.w32.*`, `p1.w8.*` (accuracy of tiers V/E/C/R, agreement, NMSE, RTL mismatches, float reference) | phase-1 coax study, 4000 test examples | batch pipeline `python -m dlnn.pipeline` (not a notebook; stored run in `data/evidence/out/final-*`) | V, E, float recomputed on 4000 examples; RTL re-run on 4 examples; C and R from stored per-example predictions + fresh 200-example subset |
| `p1.w32.acc.C_nocomp`, `p1.coax.*` | loss-compensation diagnostics, coax numbers | batch pipeline | re-read |
| `arena.*` | accuracy, SNR >= 10 dB accuracy, operations, parameters of the DLNN and the competitors | `01` (training lite + export + JS check); shipped models from `dlnn.model.arena` | recomputed with the independent numpy forward passes |
| `d1.*`, `aware.*` | TDL-Net-B reports; board-aware training on the ngspice M1a line FIR | `dlnn.model.train` (not run here) | recomputed with tier V (ideal and on the M1a FIR) |
| `p.nom.*`, `p.v.*`, `p.p_minus_v` | P tier nominal accuracies, board as designed (finding F1) | `03` | recomputed (`run_nominal`, 4000 examples) |
| `p.mc.*`, `p.xt.*`, `p.trim.*`, `p.agree` | 1000-board Monte Carlo, crosstalk cases, gain trim | batch run `python -m dlnn.tiers.parasitics` (`03` with `DLNN_FULL=1`) | re-read; `03` compares a lite re-run with them |
| `line.*`, `xbar.*`, `pair.*`, `slice.*`, `power.*`, `tile.*` | derived hardware parameters | `hw/board/derive_params.py` (+ ngspice acceptance results) | recomputed by re-running `derive_params.py` |
| `cost.*`, `scale.*` | BOM/PCB/assembly estimates, tiles, inductors, area | `04` | recomputed by re-running `cost_model.py` |
| `en.*`, `loss.*` | energy per classification, line-loss arithmetic | `04` (energy section) | formula over source values |
| `board.*`, `ind.*`, `llm.*` | fabrication-output counts, PCB-routing notes, inductor options, LLM-commentary scores | KiCad/kicad-cli runs, documents in `docs/` (not reproducible without KiCad) | re-read |

## Tested how, and caveats

* **Tested in a Colab-like environment, not on Colab itself.** Ubuntu 24.04, root, a fresh `python3.12` venv built from `requirements.txt` only (no other packages), `ngspice` 42 and `iverilog` 12 from apt, node 22, `jupyter nbconvert --execute` against the local copy (the Colab setup cell is skipped there). Colab's apt `ngspice` may be older (Ubuntu 22.04 ships 36); the notebooks compare their SPICE-derived numbers with stored references at tolerances that allow last-digit differences, but an older ngspice has not been tried. Colab's preinstalled numpy/torch versions will be replaced by the pinned ones (one `pip install` of a few hundred MB, CPU torch).
* **Lite versus full.** The article's numbers come from long batch runs (the 4000-example coax pipeline, 60-epoch trainings with 3 restarts, 1000 Monte Carlo boards, a 5^5 grid, 64 whole-tile SPICE transients). The notebooks re-run lite versions and compare them with the stored full runs within sampling tolerance; lite outputs go to `docs/artifacts/data/lite/`, `.../p_tier/`, `.../spice/tile_tran.json` and never replace the article's full-run files unless `DLNN_FULL=1`.
* **Provenance of the SPICE numbers.** C (board mode) numbers on 4000 examples come from a *circuit-calibrated behavioural model* fitted to ngspice results, not from SPICE itself; only the `.ac`/`.dc` results and the 3-example `.tran` are SPICE. All part models are generic or datasheet-derived lumped models; no vendor macro is shipped.
* **P tier is not RF** (about 1 MHz, lumped, scikit-rf only as a cascade engine) and models a board that is designed, not built.
* All prices are estimates from single readings of JLC part pages (2026-11-03) and need a live quote; the energy figures are order-of-magnitude estimates from a design power estimate and 45 nm / datasheet numbers.
* The model files in `models/` are the trained weights (ternary). The training recipe in `dlnn/model/train.py` is deterministic given the seed, but CPU/BLAS differences can change the last digits of a retrained model; that is why `00_verify` checks the shipped weights, not retrained ones.
