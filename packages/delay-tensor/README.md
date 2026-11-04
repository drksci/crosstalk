# delay-tensor

Clean-room re-implementation of the Stage-1 simulated experiment from
`../../archive/packages/phase-lattice-sim`: does a fixed coupled-line reservoir
turn a raw waveform into features that a linear readout classifies better than
the raw waveform itself?

```text
raw path    : waveform samples --------------------------------> linear readout -> accuracy
reservoir   : waveform samples -> coupled delay tensor -> traces -> linear readout -> accuracy
mlp         : waveform samples --------------------------------> MLPClassifier -> accuracy   (mnist only)
```

Both linear paths use the identical readout class, hyperparameters and batch
boundaries, so the comparison measures feature generation, not classifier
capacity. The MLP arm is deliberately different — it is the yardstick that
says what a conventional model reaches on the same split, which is what turns
a gap between the two linear arms into a claim somebody can argue with. It
appears in reports only for experiments that have one.

## What changed from the archive version

| Archive (`phase-lattice-sim`)             | This package                                            |
| ----------------------------------------- | ------------------------------------------------------- |
| PyTorch `>=2.2,<2.3` (~2 GB)              | NumPy + scikit-learn (reservoir is frozen; no autograd) |
| Single seed, single accuracy number       | N seeds, mean ± std, per-seed CSV                       |
| `dict` config with ad-hoc validation      | Frozen dataclasses, typed, validated on load            |
| 4 MB report with inlined minified Plotly  | Static self-contained HTML + inline SVG (tens of KB)    |
| Hand-rolled KiCad/ngspice subprocess code | PySpice/ngspice equivalence bench, confined to `benches/` |
| One-line stub experiments                 | `classify`, `xor`, and full MNIST three ways            |

## Quick start

Use `uv` from the repository root:

```bash
uv sync --all-packages
uv run --all-packages delay-tensor classify \
  --config packages/delay-tensor/configs/tensor_8x64.yaml \
  --seeds 1234,1235,1236 \
  --out results/
uv run --all-packages delay-tensor mnist \
  --config packages/delay-tensor/configs/mnist_8x784.yaml
uv run --all-packages pytest
```

Each run writes a self-contained directory:

```text
results/<run_id>/
  config.yaml       # exact config used
  metrics.json      # aggregate + per-seed metrics
  runs.csv          # one row per seed
  observations.txt  # plain-text summary of what happened and what to do next
  report.html       # charts, confusion matrices, example traces, config echo
```

## Layout

```text
packages/delay-tensor/
  configs/            YAML tensor configs
  src/delay_tensor/  package source
  tests/              unit + end-to-end tests
  docs/               design notes and tooling notes
```

## Scope

Simulation only, with one deliberate exception. The archive's KiCad coupon
generation stays dropped, but the ngspice path was picked up as a *bench*:
`benches/01-equivalence.ipynb` checks the tensor's update equations against a
PySpice/ngspice netlist to 5.274e-16 over 2,048 stages. PySpice is GPL-3.0, so
it is imported only by `benches/` and never by this package — the wheel stays
NumPy/scikit-learn. See `docs/tools.md` for that, and for the recommended
actively maintained tools if schematics or web visualisation come back.
