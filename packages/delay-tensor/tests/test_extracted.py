"""Checks on ``coupling.topology: extracted`` -- the macromodel update.

:func:`test_extracted_update_matches_a_hand_written_iteration` is the
decisive one: it re-derives the update from ``(A, B)`` and the mask by hand
and requires the reservoir's traces to agree to machine precision. The two
topologies have to be interchangeable inside one loop, so a claim that the
extracted path is "just the physical model" has to survive being written out
step by step rather than asserted in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from delay_tensor.config import (
    ConfigError,
    CouplingConfig,
    DelayTensorConfig,
    LossConfig,
    NonlinearityConfig,
    ReadoutConfig,
    TensorConfig,
    load_config,
)
from delay_tensor.reservoir import DelayTensor

WIRES = 4
SLOTS = 12


def stable_model(order: int = WIRES, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A strictly stable ``(A, B)``: a passive line's model cannot grow."""
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(order, order)) * 0.1
    a_mat /= max(1.05, float(np.abs(np.linalg.eigvals(a_mat)).max()))
    return a_mat, rng.normal(size=(order, 1))


def write_model(path, a_mat: np.ndarray, b_mat: np.ndarray) -> str:
    np.savez(
        path,
        A=a_mat,
        B=b_mat,
        C=np.eye(a_mat.shape[0]),
        D=np.zeros((a_mat.shape[0], b_mat.shape[1])),
    )
    return str(path)


def extracted_config(model_path: str, **overrides) -> DelayTensorConfig:
    return DelayTensorConfig(
        tensor=TensorConfig(wires=WIRES, slots=SLOTS, slot_dt_ns=20.0, loop_period_ns=SLOTS * 20.0),
        coupling=overrides.pop(
            "coupling",
            CouplingConfig(topology="extracted", neighbour_strength=0.0, model_path=model_path),
        ),
        loss=overrides.pop("loss", LossConfig(noise_std=0.0)),
        nonlinearity=overrides.pop("nonlinearity", NonlinearityConfig()),
        readout=overrides.pop("readout", ReadoutConfig()),
    )


def test_extracted_update_matches_a_hand_written_iteration(tmp_path) -> None:
    a_mat, b_mat = stable_model()
    path = write_model(tmp_path / "model.npz", a_mat, b_mat)
    reservoir = DelayTensor(extracted_config(path))

    rng = np.random.default_rng(1)
    waveforms = rng.standard_normal((3, SLOTS))
    result = reservoir.run(waveforms)

    mask = reservoir.masks
    state = np.zeros((3, WIRES))
    traces = np.zeros((3, WIRES, SLOTS))
    for slot in range(SLOTS):
        term = np.outer(waveforms[:, slot], b_mat[:, 0])
        state = state @ a_mat.T + (1.0 / (1.0 + np.exp(-mask[:, slot])))[None, :] * term
        state = np.tanh(state)
        traces[:, :, slot] = state

    np.testing.assert_allclose(result.traces, traces, rtol=1e-12, atol=1e-12)
    assert result.features.shape == (3, WIRES * SLOTS)


def test_mask_gates_the_input_and_not_the_dynamics(tmp_path) -> None:
    """With no input the mask has nothing to gate, so the run is just ``A x``.

    That is only true if the mask is on the input path: applied to ``A`` it
    would still reshape every slot.
    """
    a_mat, b_mat = stable_model()
    path = write_model(tmp_path / "model.npz", a_mat, b_mat)
    reservoir = DelayTensor(extracted_config(path))
    result = reservoir.run(np.zeros((2, SLOTS)))

    state = np.zeros((2, WIRES))
    traces = np.zeros((2, WIRES, SLOTS))
    for slot in range(SLOTS):
        state = np.tanh(state @ a_mat.T)
        traces[:, :, slot] = state

    np.testing.assert_allclose(result.traces, traces, atol=1e-12)


def test_health_reports_the_models_own_decay(tmp_path) -> None:
    a_mat, b_mat = stable_model()
    path = write_model(tmp_path / "model.npz", a_mat, b_mat)
    result = DelayTensor(extracted_config(path)).run(np.ones((1, SLOTS)))

    radius = float(np.abs(np.linalg.eigvals(a_mat)).max())
    assert result.health["topology"] == "extracted"
    # not the configured loss.decay, which does not apply in this mode
    np.testing.assert_allclose(result.health["decay"], radius, rtol=1e-12)


def test_deterministic_without_noise(tmp_path) -> None:
    a_mat, b_mat = stable_model()
    path = write_model(tmp_path / "model.npz", a_mat, b_mat)
    reservoir = DelayTensor(extracted_config(path))
    x = np.random.default_rng(0).standard_normal((4, SLOTS))
    np.testing.assert_allclose(reservoir.run(x).features, reservoir.run(x).features)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_extracted_requires_a_model_path() -> None:
    with pytest.raises(ConfigError, match=r"requires .*model_path"):
        CouplingConfig(topology="extracted", neighbour_strength=0.0).validate()


def test_extracted_refuses_leftover_neighbour_coupling() -> None:
    """The default is 0.08, so this is the easy mistake -- and it must not pass silently."""
    with pytest.raises(ConfigError, match="neighbour_strength"):
        CouplingConfig(topology="extracted", model_path="m.npz").validate()


def test_model_path_is_rejected_without_the_extracted_topology() -> None:
    with pytest.raises(ConfigError, match="only applies to topology 'extracted'"):
        CouplingConfig(model_path="m.npz").validate()


def test_load_config_resolves_model_path_against_the_config_file(tmp_path) -> None:
    """A config and its model travel together, whatever the working directory."""
    a_mat, b_mat = stable_model()
    (tmp_path / "runs").mkdir()
    write_model(tmp_path / "runs" / "model.npz", a_mat, b_mat)
    config_file = tmp_path / "runs" / "run.yaml"
    config_file.write_text(
        "tensor:\n"
        f"  wires: {WIRES}\n"
        f"  slots: {SLOTS}\n"
        "  slot_dt_ns: 20.0\n"
        f"  loop_period_ns: {SLOTS * 20.0}\n"
        "coupling:\n"
        "  topology: extracted\n"
        "  neighbour_strength: 0.0\n"
        "  model_path: model.npz\n"
        "loss:\n"
        "  noise_std: 0.0\n",
        encoding="utf-8",
    )

    config = load_config(config_file)
    assert config.coupling.model_path == str((tmp_path / "runs" / "model.npz").resolve())
    # and it loads from an unrelated working directory
    assert DelayTensor(config).run(np.ones((1, SLOTS))).features.shape == (1, WIRES * SLOTS)


# --------------------------------------------------------------------------
# what a bad model looks like
# --------------------------------------------------------------------------


def test_missing_model_file(tmp_path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        DelayTensor(extracted_config(str(tmp_path / "nope.npz")))


def test_model_with_the_wrong_state_dimension(tmp_path) -> None:
    a_mat, b_mat = stable_model(order=WIRES + 2)
    path = write_model(tmp_path / "model.npz", a_mat, b_mat)
    with pytest.raises(ConfigError, match=r"tensor\.wires"):
        DelayTensor(extracted_config(path))


def test_model_that_grows_without_bound(tmp_path) -> None:
    """A passive line's macromodel cannot have spectral radius above one."""
    path = write_model(tmp_path / "model.npz", np.eye(WIRES) * 1.5, np.ones((WIRES, 1)))
    with pytest.raises(ConfigError, match="spectral radius"):
        DelayTensor(extracted_config(path))


def test_model_without_the_state_matrices(tmp_path) -> None:
    np.savez(tmp_path / "model.npz", C=np.eye(WIRES), D=np.zeros((WIRES, 1)))
    with pytest.raises(ConfigError, match="missing"):
        DelayTensor(extracted_config(str(tmp_path / "model.npz")))
