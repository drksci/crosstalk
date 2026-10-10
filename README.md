# delay-tensors

Delay-line and coupled-line reservoir experiments.

```text
delay-tensors/
  packages/delay-tensor/   the simulated experiment (active)
  archive/                  the original version, kept as reference
  results/                  generated run outputs (gitignored)
```

`archive/` is untouched: it is the original `dllm` workspace, with its own
virtualenv, and it stays as the reference for what the experiment used to be.
The active code is `packages/delay-tensor/`, a clean-room rebuild of the same
Stage-1 question — see its [README](packages/delay-tensor/README.md) and
[design notes](packages/delay-tensor/docs/design.md).

## Quick start

```bash
uv sync --all-packages

# six-class waveform classification
uv run --all-packages delay-tensor classify \
  --config packages/delay-tensor/configs/tensor_8x64.yaml \
  --seeds 1234,1235,1236

# temporal XOR: the experiment where the reservoir actually earns its keep
uv run --all-packages delay-tensor xor \
  --config packages/delay-tensor/configs/tensor_8x64.yaml

# checks
uv run --all-packages pytest
uv run --all-packages ruff check packages/delay-tensor
uv run --all-packages ruff format --check packages/delay-tensor
```

Each run writes `results/<run_id>/` containing `report.html`, `metrics.json`,
`runs.csv`, `observations.txt` and the exact `config.yaml` used.

Latest results on `tensor_8x64` (3 seeds):

| experiment | raw baseline | reservoir | delta |
| --- | --- | --- | --- |
| classify | 0.847 ± 0.016 | 0.861 ± 0.022 | +0.014 ± 0.006 |
| xor | 0.724 ± 0.024 | 0.995 ± 0.009 | +0.271 ± 0.024 |

Classification is a near-tie; temporal XOR is where separated events need to
be mixed, and there the reservoir is decisive. Those are the honest numbers.

## Tooling

`packages/delay-tensor/docs/tools.md` records the survey of
[github.com/topics/electronics](https://github.com/topics/electronics) and
[circuit-simulation](https://github.com/topics/circuit-simulation): lighter
actively maintained alternatives for SPICE (PySpice, scikit-rf), schematics
(tscircuit, skidl, atopile), and web visualisation (inline SVG, uPlot,
tscircuit, CircuitJS, Wokwi).

## Version control

This directory is not a git repository, so nothing here is committed yet. Say
the word and it can be initialised (the archived `archive/` carries its own
`.git` history and should be left alone or handled separately).
