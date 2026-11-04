# DLNN: ternary delay-line neural network, four-fidelity simulation

A tapped coaxial delay-line network with ternary taps (tap polarity = BitNet-style {-1, 0, +1} weights) and a differential-pair tanh,
trained on synthetic modulation recognition and simulated at four fidelities (V ideal math, E fixed point + Verilog RTL, C ngspice circuit,
R scikit-rf RF channel), with a small local LLM that comments on the side-by-side results.

**Simulation only, synthetic data, no hardware measurement.** C and R are 16 to 20 points below V at the headline `--cascade` operating point
(32 dB/us loss, no repeaters). Two things cause that: the operating point is not buildable without repeaters, **and** C's default per-layer gain trim over-drives layer 3
(C scores 0.663 by default, 0.804 with no compensation, 0.805 with ideal per-tap repeaters at the same loss), so most of C's gap is a modelling choice, not line loss.
The numbers, the caveats and the failures are in the docs.

**Start here: [`docs/README.md`](docs/README.md)** (index, results, honesty page, reproducibility).

## Quickstart (commands run on this box: 4 cores, 15 GB, Ubuntu 24.04, Python 3.13)

```bash
./setup.sh                                  # idempotent; first run builds llama-cpp-python from source (~4 min, needs cmake + a C/C++ compiler)
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm qwen --db sqlite --out out --run-id final-w32   # about 13 min
.venv/bin/python run.py --task modrec --tiers V,E,C,R --cascade --llm none --model models/tdlnet_modrec_ternary.npz --run-id final-w8   # width-8 secondary, about 3 min
.venv/bin/python scripts/make_results_md.py # regenerate docs/results.md from out/final-w32 and out/final-w8
.venv/bin/python -m pytest -q               # test suite
```

Use `--llm none` for the template commentary (no Qwen, no GGUF needed; `WITH_LLM=0 ./setup.sh` skips the build and the 1.1 GB download).
Notebooks: `notebooks/00..06_*.ipynb` (executed; see `docs/notebooks.md`). Full list of commands and versions: `docs/reproducibility.md`.
