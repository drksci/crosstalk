# delay-tensor

Clean-room re-implementation of the Stage-1 simulated experiment from
`../../archive/packages/phase-lattice-sim`: does a fixed coupled-line reservoir
turn a raw waveform into features that a linear readout classifies better than
the raw waveform itself?

```text
raw path    : waveform samples --------------------------------> linear readout -> accuracy
reservoir   : waveform samples -> coupled delay tensor -> traces -> linear readout -> accuracy
```

Both paths use the identical readout family and hyperparameters, so the
comparison measures feature generation, not classifier capacity.

## What changed from the archive version

| Archive (`phase-lattice-sim`)             | This package                                            |
| ----------------------------------------- | ------------------------------------------------------- |
| PyTorch `>=2.2,<2.3` (~2 GB)              | NumPy + scikit-learn (reservoir is frozen; no autograd) |
| Single seed, single accuracy number       | N seeds, mean ± std, per-seed CSV                       |
| `dict` config with ad-hoc validation      | Frozen dataclasses, typed, validated on load            |
| 4 MB report with inlined minified Plotly  | Static self-contained HTML + inline SVG (tens of KB)    |
| Hand-rolled KiCad/ngspice subprocess code | Removed (simulation-only scope)                         |
| One-line stub experiments                 | `classify` and `xor` both implemented                   |

## Quick start

Use `uv` from the repository root:

```bash
uv sync --all-packages
uv run --all-packages delay-tensor classify \
  --config packages/delay-tensor/configs/tensor_8x64.yaml \
  --seeds 1234,1235,1236 \
  --out results/
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

Simulation only. The archive's KiCad coupon and ngspice netlist generation were
dropped deliberately; see `docs/tools.md` for the recommended, actively
maintained tools if that path is picked up again, including web-based
visualisation options.
