# AGENTS.md

Guide for agents working in this repository. Humans: start at [README.md](README.md).

## Commands

| Task | Command |
|---|---|
| set up Delay Tensor | `uv sync --all-packages` |
| set up benches too | `uv sync --all-packages --group bench` (ngspice via `NGSPICE_LIBRARY_PATH`) |
| run an experiment | `uv run --all-packages delay-tensor {classify,xor,mnist} --config packages/delay-tensor/configs/<name>.yaml --seeds 1234,1235,1236` |
| Delay Tensor + bench tests | `uv run --all-packages pytest` (bench tests skip without the group) |
| lint | `uv run --all-packages ruff check packages/delay-tensor benches` |
| DLNN setup and tests | `cd packages/dlnn && ./setup.sh && .venv/bin/python -m pytest -q` |
| DLNN article numbers | `packages/dlnn-paper/notebooks/00_verify.ipynb` |

Long runs (MNIST, DLNN `run.py --cascade`) take minutes. Do not re-run them to confirm a number that a results file already records; read the file.

## Invariants

1. **Readout parity.** Raw and tensor arms share readout class, C, split, seeds and batch boundaries. A change that breaks this invalidates every comparison.
2. **The reservoir is fixed.** Nothing in `DelayTensor` is fitted; the mask comes from `mask_seed`.
3. **One recurrence.** It lives only in `packages/delay-tensor/src/delay_tensor/reservoir.py`; `benches/netlist.py` must stay equation-for-equation equivalent (checked to ~1e-15).
4. **PySpice is GPL-3.0.** Only `benches/spice_runner.py` imports it. Never import it from the package.
5. **Extracted systems must be stable.** The loader refuses spectral radius above 1 + 1e-6; do not loosen it.
6. **The CPL is the model.** Populated resistors in the DLNN fab output must equal the model's non-zero weights (85 / 315 / 316); `tapmap.json` is the map.
7. **Status words are exact.** Simulated, emulated, designed, pending. Nothing is fabricated or measured; do not write that it is.

## Where truth lives

| Fact | Source |
|---|---|
| Delay Tensor run results | `results/<run_id>/metrics.json` (gitignored; regenerate with the CLI) |
| Experiment definitions | `packages/delay-tensor/configs/*.yaml` |
| DLNN printed numbers | `packages/dlnn-paper/docs/artifacts/data/claims.json` |
| DLNN tier results | `packages/dlnn-paper/docs/artifacts/data/evidence/` |
| What is and is not established | `packages/dlnn-paper/docs/honesty.md` |
| History and dates | `docs/story.md`, `docs/timeline.json` |

## Writing

Docs are third person, Australian English, plain. Every number carries its conditions (seeds, Q, tier). When two sources disagree, say so and name both.
