"""Checks on the macromodel fit -- pure NumPy, no ngspice, no scikit-rf.

The centrepiece is :func:`test_round_trip_recovers_the_impulse_response`.
Ho-Kalman is an exact result: fed the true Markov parameters of a rank-``r``
system it recovers a realization that reproduces them, so a synthetic system
with a known answer is a decisive check. If that test passes the algebra is
right; if it fails, no amount of matching on real data would make the model
trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from extract import block_hankel, fit_macromodel, markov_from_transient, save_macromodel
from transient import resample_uniform


def _stable_system(order: int, outputs: int, inputs: int, *, seed: int = 0):
    """A random but strictly stable system, so its powers decay."""
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(order, order)) * 0.4
    a_mat /= max(1.05, float(np.abs(np.linalg.eigvals(a_mat)).max()))
    return (
        a_mat,
        rng.normal(size=(order, inputs)),
        rng.normal(size=(outputs, order)),
        rng.normal(size=(outputs, inputs)) * 0.01,
    )


def _markov(a_mat, b_mat, c_mat, d_mat, steps: int) -> np.ndarray:
    """Exact Markov parameters: ``h[0] = D``, ``h[k] = C A^(k-1) B``."""
    out = np.zeros((steps, c_mat.shape[0], b_mat.shape[1]))
    out[0] = d_mat
    # read before advancing: h[1] = C B, h[2] = C A B, ...
    power = np.eye(a_mat.shape[0])
    for k in range(1, steps):
        out[k] = c_mat @ power @ b_mat
        power = a_mat @ power
    return out


def test_round_trip_recovers_the_impulse_response() -> None:
    order, outputs, inputs = 6, 3, 2
    a_mat, b_mat, c_mat, d_mat = _stable_system(order, outputs, inputs)
    markov = _markov(a_mat, b_mat, c_mat, d_mat, steps=3 * order + 4)

    model = fit_macromodel(markov, order=order)

    assert model.order == order
    assert model.outputs == outputs and model.inputs == inputs
    assert model.max_error(markov) < 1e-12, "an exact system must be recovered exactly"
    assert model.spectral_radius() < 1.0


def test_round_trip_holds_for_every_system_size() -> None:
    for outputs, inputs in ((1, 1), (4, 1), (1, 4), (3, 3)):
        a_mat, b_mat, c_mat, d_mat = _stable_system(5, outputs, inputs, seed=outputs * 10 + inputs)
        markov = _markov(a_mat, b_mat, c_mat, d_mat, steps=24)
        assert fit_macromodel(markov, order=5).max_error(markov) < 1e-12


def test_tolerance_drops_a_direction_below_the_noise_floor() -> None:
    """A sixth direction an order of magnitude down must not become a state."""
    order, outputs, inputs = 4, 2, 2
    a_mat, b_mat, c_mat, d_mat = _stable_system(order, outputs, inputs)
    markov = _markov(a_mat, b_mat, c_mat, d_mat, steps=20)

    weak = _stable_system(1, outputs, inputs, seed=9)
    markov = markov + 1e-8 * _markov(*weak, steps=20)

    model = fit_macromodel(markov, tolerance=1e-6)
    assert model.order == order
    assert model.max_error(markov) < 1e-6


def test_order_above_the_true_rank_is_clamped() -> None:
    a_mat, b_mat, c_mat, d_mat = _stable_system(3, 2, 2)
    markov = _markov(a_mat, b_mat, c_mat, d_mat, steps=16)
    assert fit_macromodel(markov, order=99).order <= markov.shape[0]


def test_block_hankel_stacks_lags_along_both_axes() -> None:
    steps, outputs, inputs = 7, 2, 3
    markov = np.arange(steps * outputs * inputs, dtype=float).reshape(steps, outputs, inputs)
    hankel = block_hankel(markov, blocks=3, offset=1)

    assert hankel.shape == (outputs * 3, inputs * 3)
    np.testing.assert_allclose(hankel[:outputs, :inputs], markov[1])
    np.testing.assert_allclose(hankel[:outputs, inputs : 2 * inputs], markov[2])
    np.testing.assert_allclose(hankel[outputs : 2 * outputs, :inputs], markov[2])
    np.testing.assert_allclose(hankel[2 * outputs :, 2 * inputs :], markov[5])


def test_block_hankel_refuses_more_blocks_than_data() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        block_hankel(np.zeros((4, 2, 2)), blocks=4, offset=1)


def test_fit_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="markov must be"):
        fit_macromodel(np.zeros((5, 4)))
    with pytest.raises(ValueError, match="at least 2"):
        fit_macromodel(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="identically zero"):
        fit_macromodel(np.zeros((8, 2, 2)))


def test_impulse_response_rejects_zero_steps() -> None:
    a_mat, b_mat, c_mat, d_mat = _stable_system(3, 2, 2)
    model = fit_macromodel(_markov(a_mat, b_mat, c_mat, d_mat, steps=12), order=3)
    with pytest.raises(ValueError, match="steps"):
        model.impulse_response(0)


# --------------------------------------------------------------------------
# from a transient record, and back out to the package
# --------------------------------------------------------------------------


def test_markov_from_transient_recovers_a_known_impulse_response() -> None:
    """Convolve broadband noise with a known kernel; deconvolution must invert it.

    The launch is noise rather than a shaped pulse on purpose. A windowed or
    trapezoidal launch has *exact* spectral nulls, and no deconvolution can
    recover frequencies the source never excited -- see
    :func:`test_markov_from_transient_degrades_gracefully_at_spectral_nulls`.
    Noise has energy everywhere, so nothing is blanked and the kernel comes
    back exactly.
    """
    n, dt = 512, 10e-12
    times = np.arange(n) * dt

    delay, second, third = 20, 25, 32
    kernel = np.zeros(n)
    kernel[delay], kernel[second], kernel[third] = 1.0, -0.4, 0.15
    other = np.zeros(n)
    other[delay + 3] = 0.7

    # The record is circular for the FFT but the convolution is linear, so both
    # ends must be quiet: zeroing the first 40 and last 40 makes the wrap land
    # where the source is zero, which is what makes the recovery exact.
    source = np.random.default_rng(7).uniform(-1.0, 1.0, n)
    source[:40] = 0.0
    source[-40:] = 0.0
    responses = np.stack(
        [
            np.convolve(source, kernel)[:n],
            np.convolve(source, other)[:n],
        ],
        axis=1,
    )

    _, grid = resample_uniform(times, source, dt=dt)
    h = markov_from_transient(times, source, responses, dt=dt)
    # the resample spans [t0, t_last), so it holds one sample fewer than the input
    assert h.shape == (grid.size, 2, 1)

    np.testing.assert_allclose(h[delay, 0, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(h[second, 0, 0], -0.4, atol=1e-9)
    np.testing.assert_allclose(h[third, 0, 0], 0.15, atol=1e-9)
    np.testing.assert_allclose(h[delay + 3, 1, 0], 0.7, atol=1e-9)
    # nothing before the delay, and no cross-contamination between outputs
    assert np.abs(h[:delay, :, 0]).max() < 1e-9
    assert np.abs(h[third + 1 :, 0, 0]).max() < 1e-9


def test_markov_from_transient_degrades_gracefully_at_spectral_nulls() -> None:
    """A windowed launch has exact nulls; the answer must stay bounded, not explode.

    Zeroing a bin the source cannot reach is the only honest answer -- dividing
    by it would return numerical noise as though it were measurement. The cost
    is visible: a kernel with a flat spectrum weights every bin equally, so
    losing three of them shifts the recovered spike by a few percent.
    """
    n, dt = 512, 10e-12
    times = np.arange(n) * dt
    kernel = np.zeros(n)
    kernel[20], kernel[25], kernel[32] = 1.0, -0.4, 0.15
    source = np.zeros(n)
    source[10:60] = np.hanning(50)  # has exact spectral nulls
    response = np.convolve(source, kernel)[:n]

    h = markov_from_transient(times, source, response, dt=dt)

    assert np.all(np.isfinite(h)), "a null must be blanked, never returned as noise"
    assert abs(h[20, 0, 0] - 1.0) < 0.1, (
        f"nulls cost accuracy but must stay bounded, got {h[20, 0, 0]:.4f}"
    )


def test_markov_from_transient_rejects_mismatched_inputs() -> None:
    times = np.arange(64) * 1e-11
    with pytest.raises(ValueError, match="source_wave"):
        markov_from_transient(times, np.zeros(63), np.zeros((64, 2)), dt=1e-11)
    with pytest.raises(ValueError, match="samples"):
        markov_from_transient(times, np.zeros(64), np.zeros((63, 2)), dt=1e-11)
    with pytest.raises(ValueError, match="identically zero"):
        markov_from_transient(times, np.zeros(64), np.ones((64, 2)), dt=1e-11)


def test_a_saved_model_is_readable_by_the_reservoir(tmp_path) -> None:
    """The whole hand-off: simulated line -> fit -> file -> running reservoir."""
    from delay_tensor.config import CouplingConfig, DelayTensorConfig, TensorConfig
    from delay_tensor.reservoir import DelayTensor

    order, outputs = 4, 6
    a_mat, b_mat, c_mat, d_mat = _stable_system(order, outputs, 1)
    model = fit_macromodel(_markov(a_mat, b_mat, c_mat, d_mat, steps=24), order=order)
    path = save_macromodel(model, tmp_path / "model.npz")

    slots = 8
    config = DelayTensorConfig(
        tensor=TensorConfig(wires=order, slots=slots, slot_dt_ns=20.0, loop_period_ns=slots * 20.0),
        coupling=CouplingConfig(topology="extracted", neighbour_strength=0.0, model_path=path),
    )
    reservoir = DelayTensor(config)
    result = reservoir.run(np.ones((2, slots)))

    assert result.traces.shape == (2, order, slots)
    assert np.abs(result.traces).max() > 0.0, "a fitted model must actually respond"
    assert result.health["topology"] == "extracted"
