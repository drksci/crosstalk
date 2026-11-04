# Design notes

Stage-1 simulated experiment, rebuilt. The archived original lives in
`../../archive/packages/phase-lattice-sim`; this package is the clean-room
version and does not import from it.

## What the experiment asks

> Does a fixed coupled-line reservoir turn a raw waveform into features that a
> simple linear readout classifies better than the raw waveform itself?

Both paths are deliberately identical apart from the feature generator:

```text
raw path    : waveform ----------------------------> LinearReadout -> accuracy
reservoir   : waveform -> DelayTensor ----> LinearReadout -> accuracy
```

Same readout class, same hyperparameters, same split, same seed. Any accuracy
difference is a statement about features.

## Decisions

**NumPy, not PyTorch.** The reservoir is frozen — `train_masks=False` in the
archive meant nothing in the tensor ever received a gradient — so the only
trained object is one linear layer. scikit-learn's logistic regression fits
that in one deterministic pass and removes roughly two gigabytes of
dependencies plus a hand-written Adam loop.

**Replicate over seeds.** One seed on a 576-sample synthetic dataset moves by
several accuracy points. Every run reports mean ± sample standard deviation
over seeds, a per-seed CSV, and a verdict that checks the delta against the
spread before calling it a lift.

**Typed config, validated once.** Frozen dataclasses per section, unknown
sections and unknown keys rejected at load time, and unmodelled knobs
(`diagonal_strength`, `jitter_std_slots`) refused with a message instead of
being silently ignored. `loop_period_ns` must equal `slots * slot_dt_ns`
because they are the same quantity spelled twice.

**The mask does something.** The archive drew masks as `0.01 * randn`, where
`sigmoid` sits flat at 0.5 and `mask_mode: modulated` was a no-op. Masks are
now standard normal, so modulation spans roughly 0.11 to 0.88 while keeping
the same mean coupling.

**Second-neighbour coupling is implemented.** The archive's config exposed
`second_neighbour_strength` and the backend never read it.

**The full slot window always runs.** A short input leaves the remaining slots
evolving on decay alone, which is what the tensor does when the launch stops,
rather than zero-padding the state; input longer than the slot window is
truncated and said so.

**Static report.** The archive inlined a minified Plotly build (3.6 MB on
Plotly's own benchmark table) and produced multi-megabyte pages for four
charts. Reports here render charts as inline SVG and pull nothing from the
network, so a run stays readable in ten years. See `tools.md` for what to use
if interactivity becomes worth the bytes.

**Simulation only.** The archive generated a KiCad coupon and an ngspice
netlist per run and, when `kicad-cli`/`ngspice` were absent, rendered the
report section as "pending" forever. That path is out of scope here; `tools.md`
records what to reach for if it comes back.

## Layout

```text
src/delay_tensor/
  config.py        typed config, YAML load/save, validation
  datasets.py      seeded synthetic tasks + normalisation + stratified split
  mnist.py         IDX fetch/cache/parse + raster_scan into slot clocks
  reservoir.py     the simulated substrate (frozen, vectorised)
  readout.py       LinearReadout + StreamingLinearReadout, shared by both paths
  metrics.py       accuracy, confusion matrix, spread
  experiments/
    base.py        replicate loop, aggregation, example collection
    classify.py    six-class waveform classification
    xor.py         temporal XOR
    mnist.py       full MNIST: raw pixels vs tensor features vs an MLP
  svg.py           inline SVG charts (no plotting dependency)
  report.py        run directory: metrics.json, runs.csv, observations, report.html
  cli.py           delay-tensor classify|xor|mnist
```

SPICE equivalence lives outside the package, in `benches/` at the repository
root, because PySpice is GPL-3.0 and this package is not. See `tools.md`.

## Adding an experiment

An experiment is a seeded dataset factory plus `run_replicates`:

```python
def run_my_task(config, *, seeds, ...):
    def dataset_factory(seed):
        return make_my_dataset(seed=seed, ...)
    return run_replicates(
        experiment="my-task",
        config=config,
        dataset_factory=dataset_factory,
        seeds=seeds,
        parameters={...},
    )
```

Register it in `cli.py` with a subparser. Reports, metrics, CSV and
observations come along for free.

`mnist` is the exception that shows where the seam is: it does not use
`run_replicates` at all. MNIST's split is canonical rather than stratified,
its readout is the streaming one rather than `LinearReadout`, and its feature
matrix is a disk memmap rather than a `DatasetFactory`'s arrays. It reuses
`SeedRecord` / `Summary` / `TraceExample` — including the optional third arm —
and writes its own loop. An experiment joins `base.py` at the data model, not
at the replicate loop, which is the reason those two are separate files.

## Known limits

- Single band only (`tensor.bands` must be 1); no cross-band/diagonal coupling.
- Timing jitter is not modelled (`loss.jitter_std_slots` must be 0).
- The reservoir is a physical metaphor, not a circuit solver: no S-parameters
  and no propagation delay beyond the slot clock. Its *equations* are checked
  against ngspice by `benches/01-equivalence.ipynb` (max |error| 5.274e-16
  over 2,048 stages), but the package does not simulate a circuit.
- The synthetic tasks are sanity checks, not benchmarks. A reservoir that
  wins them has earned the next experiment, not a conclusion.
