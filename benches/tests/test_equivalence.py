"""ngspice cross-check: the generated netlist solves to DelayTensor.run().

These tests are the reason ``benches/netlist.py`` exists. They need a working
ngspice + PySpice, which live in a separate dependency group, so they skip
cleanly when that stack is missing rather than failing the default suite.

Run them with the bench group::

    uv run --all-packages --group bench pytest benches/tests -v
"""

from __future__ import annotations

import numpy as np
import pytest

from delay_tensor.config import (
    CouplingConfig,
    DelayTensorConfig,
    LossConfig,
    NonlinearityConfig,
    ReadoutConfig,
    TensorConfig,
)
from delay_tensor.reservoir import DelayTensor
from netlist import build_netlist

spice = pytest.importorskip("PySpice.Spice.Parser")
from spice_runner import solve_states  # noqa: E402


def _config(**overrides) -> DelayTensorConfig:
    """A small but non-trivial tensor; sections can be overridden per test."""
    section = overrides.pop("tensor", TensorConfig(wires=4, slots=6, slot_dt_ns=20.0, loop_period_ns=120.0))
    return DelayTensorConfig(
        name="bench_test",
        tensor=section,
        loss=overrides.pop("loss", LossConfig(decay=0.95, noise_std=0.0)),
        **overrides,
    )


def _roundtrip(config: DelayTensorConfig, waveforms: np.ndarray, seed: int = 0) -> float:
    """Push the same waveforms through NumPy and ngspice; return max |diff|."""
    reservoir = DelayTensor(config)
    reference = reservoir.run(waveforms)
    netlist = build_netlist(config, waveforms, reservoir.masks, reservoir.drive_profile)

    wires, slots = config.tensor.wires, config.tensor.slots
    got, timings = solve_states(netlist, waveforms.shape[0], wires, slots)

    assert timings["solve_s"] < 60.0, "equivalence bench should stay interactive"
    # The readout path must agree too, not just the raw traces.
    np.testing.assert_allclose(
        reservoir._readout_features(got), reference.features, rtol=1e-9, atol=1e-9
    )
    return float(np.abs(got - reference.traces).max())


@pytest.mark.parametrize("batch", [1, 3])
def test_equivalence_default_config(batch: int) -> None:
    config = _config(loss=LossConfig(decay=0.995, noise_std=0.0))
    rng = np.random.default_rng(42)
    waveforms = rng.uniform(-1.0, 1.0, size=(batch, config.tensor.slots + 5))
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_matches_full_geometry() -> None:
    """The shipped 8x64 config, with a waveform shorter than the window."""
    config = _config(
        tensor=TensorConfig(wires=8, slots=64, slot_dt_ns=20.0, loop_period_ns=1280.0),
        loss=LossConfig(decay=0.995, noise_std=0.0),
    )
    rng = np.random.default_rng(7)
    waveforms = rng.uniform(-1.0, 1.0, size=(2, 64))
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_short_input_runs_tail_at_zero_drive() -> None:
    config = _config()
    rng = np.random.default_rng(3)
    waveforms = rng.uniform(-1.0, 1.0, size=(2, 2))  # 2 samples into a 6-slot window
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_additive_mask_mode() -> None:
    config = _config(
        coupling=CouplingConfig(mask_mode="additive", mask_gain=0.05),
        loss=LossConfig(decay=0.9, noise_std=0.0),
    )
    rng = np.random.default_rng(11)
    waveforms = rng.uniform(-1.0, 1.0, size=(3, config.tensor.slots))
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_second_neighbour_coupling() -> None:
    config = _config(
        tensor=TensorConfig(wires=6, slots=8, slot_dt_ns=20.0, loop_period_ns=160.0),
        coupling=CouplingConfig(second_neighbour_strength=0.03),
        loss=LossConfig(decay=0.95, noise_std=0.0),
    )
    rng = np.random.default_rng(11)
    waveforms = rng.uniform(-1.0, 1.0, size=(2, config.tensor.slots))
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_identity_nonlinearity_rightmost_wire() -> None:
    config = _config(
        coupling=CouplingConfig(mask_mode="additive", mask_gain=0.05),
        loss=LossConfig(decay=0.99, noise_std=0.0),
        nonlinearity=NonlinearityConfig(type="identity", location="state"),
        readout=ReadoutConfig(traces="rightmost_wire"),
    )
    rng = np.random.default_rng(5)
    waveforms = rng.uniform(-1.0, 1.0, size=(2, config.tensor.slots))
    assert _roundtrip(config, waveforms) < 1e-9


def test_equivalence_zero_coupling() -> None:
    config = _config(
        coupling=CouplingConfig(neighbour_strength=0.0, mask_gain=0.0),
        loss=LossConfig(decay=0.8, noise_std=0.0),
    )
    rng = np.random.default_rng(9)
    waveforms = rng.uniform(-1.0, 1.0, size=(1, config.tensor.slots))
    assert _roundtrip(config, waveforms) < 1e-9
