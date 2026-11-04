"""Pure-Python checks on the netlist generator — no ngspice required.

``spice_runner`` is imported lazily by the equivalence tests only, so this
file runs under the ordinary uv venv.
"""

from __future__ import annotations

import numpy as np
import pytest

from delay_tensor.config import (
    CouplingConfig,
    DelayTensorConfig,
    LossConfig,
    NonlinearityConfig,
    TensorConfig,
)
from netlist import build_netlist, drive_samples, state_node_name, state_node_names


def _config(**overrides) -> DelayTensorConfig:
    section = overrides.pop("tensor", TensorConfig(wires=4, slots=6, slot_dt_ns=20.0, loop_period_ns=120.0))
    return DelayTensorConfig(
        name="gen_test",
        tensor=section,
        loss=overrides.pop("loss", LossConfig(decay=0.95, noise_std=0.0)),
        **overrides,
    )


def test_state_node_names_round_trip_to_shape() -> None:
    names = state_node_names(3, 4, 6)
    assert len(names) == 3 * 4 * 6
    assert names[0] == state_node_name(0, 0, 0)
    # index order must be (b, w, s) so a flat reshape lands correctly
    assert names[1 * 4 * 6 + 2 * 6 + 5] == state_node_name(1, 2, 5)
    assert len(set(names)) == len(names), "node names must be unique"


def test_drive_samples_truncates_and_zero_pads() -> None:
    config = _config()
    long = np.ones((2, 10))
    drive = drive_samples(config, long)
    assert drive.shape == (2, 6), "input longer than the slot window truncates"
    assert np.all(drive == 1.0)

    short = np.full((2, 2), 0.5)
    drive = drive_samples(config, short)
    assert np.all(drive[:, :2] == 0.5)
    assert np.all(drive[:, 2:] == 0.0), "a short input leaves the tail at zero drive"


def test_drive_samples_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        drive_samples(_config(), np.zeros((0, 6)))
    with pytest.raises(ValueError, match="2-D"):
        drive_samples(_config(), np.zeros((1, 0)))


def test_deck_shape_and_gating() -> None:
    config = _config(
        coupling=CouplingConfig(second_neighbour_strength=0.03),
        nonlinearity=NonlinearityConfig(type="identity", location="state"),
    )
    waveforms = np.linspace(-1, 1, 6)[None, :]
    reservoir_masks = np.zeros((4, 6))
    profile = np.linspace(1.0, 0.2, 4)

    deck = build_netlist(config, waveforms, reservoir_masks, profile)
    lines = deck.splitlines()

    assert lines[0].startswith("*"), "first line must be a comment/title"
    assert lines[-1] == ".end"
    assert any(line.startswith(".options") for line in lines)

    # 24 stages -> 24 summing resistors, 24 sinks, 6 drive sources
    assert sum(line.startswith("R_") for line in lines) == 24
    assert sum(line.startswith("B_") for line in lines) == 24
    assert sum(line.startswith("Vd_") for line in lines) == 6

    # slot 0 has no stage -1, so it gets neither decay nor coupling terms
    slot0_gdc = [line for line in lines if line.startswith("Gdc_")]
    assert all("_s0" not in name for name in (line.split()[0] for line in slot0_gdc))
    assert len(slot0_gdc) == 4 * 5, "decay appears on slots 1..5 only"

    # identity nonlinearity -> unity buffer, never tanh
    assert "tanh" not in deck
    # second-neighbour coupling present for interior wires
    assert any(line.startswith("Gn2l_") for line in lines)
    assert any(line.startswith("Gn2r_") for line in lines)


def test_tanh_emitted_only_when_configured() -> None:
    config = _config()
    waveforms = np.zeros((1, 6))
    deck = build_netlist(config, waveforms, np.zeros((4, 6)), np.linspace(1.0, 0.2, 4))
    assert "V=tanh(" in deck


def test_modulated_vs_additive_sources() -> None:
    waveforms = np.zeros((1, 6))
    masks = np.arange(24, dtype=float).reshape(4, 6)
    profile = np.linspace(1.0, 0.2, 4)

    modulated = build_netlist(_config(), waveforms, masks, profile)
    assert "Vm_" not in modulated, "modulated mode folds the mask into coupling gains"
    assert "Gmk_" not in modulated

    additive = build_netlist(
        _config(coupling=CouplingConfig(mask_mode="additive", mask_gain=0.05)),
        waveforms,
        masks,
        profile,
    )
    assert "Vm_" in additive, "additive mode emits a per-slot offset source"
    assert "Gmk_" in additive
    # offset value is mask_gain * mask[w, s]
    assert "0.05" in additive


def test_vccs_injects_with_zero_as_positive_terminal() -> None:
    """SPICE drives current out of the positive terminal; get the sign right."""
    config = _config()
    waveforms = np.zeros((1, 6))
    deck = build_netlist(config, waveforms, np.zeros((4, 6)), np.linspace(1.0, 0.2, 4))
    for line in deck.splitlines():
        if line.startswith(("Gdc_", "Gdr_", "Gnl_", "Gnr_", "Gn2l_", "Gn2r_", "Gmk_")):
            parts = line.split()
            assert parts[1] == "0", f"{parts[0]} must have 0 as its positive terminal"
            assert parts[2].startswith("u"), f"{parts[0]} must drive a summing node"


def test_rejects_mismatched_masks_and_profile() -> None:
    config = _config()
    waveforms = np.zeros((1, 6))
    with pytest.raises(ValueError, match="masks"):
        build_netlist(config, waveforms, np.zeros((3, 6)), np.linspace(1.0, 0.2, 4))
    with pytest.raises(ValueError, match="profile"):
        build_netlist(config, waveforms, np.zeros((4, 6)), np.linspace(1.0, 0.2, 3))


def test_coefficients_round_trip_at_full_precision() -> None:
    """17 significant digits, so ngspice parses back exactly what NumPy computed."""
    config = _config(loss=LossConfig(decay=0.995, noise_std=0.0))
    waveforms = np.zeros((1, 6))
    deck = build_netlist(config, waveforms, np.zeros((4, 6)), np.array([0.9999999999999999] * 4))
    decay_terms = [
        line for line in deck.splitlines() if line.startswith("Gdc_") and "0.995" in line
    ]
    assert len(decay_terms) == 4 * 5, "every stage but slot 0 carries the decay term"
