"""Checks on the time-domain bench -- the ngspice one skips without the bench group.

The synthetic test is the sharp one: feeding a known delay and gain through the
resample-and-FFT path must reproduce that delay and gain exactly, because on
FFT bins a pure delay is a known phase and nothing else is being asked of the
method. It exercises the machinery only -- the factor of two comes from the
Thevenin convention and is checked against a real solver in
:func:`test_transient_fft_matches_the_analytic_s_parameters`.
"""

from __future__ import annotations

import numpy as np
import pytest

from transient import (
    SOURCE_FLOOR,
    LaunchPulse,
    arrival_time,
    band_mask,
    flight_time,
    resample_uniform,
    spice_time,
    transient_transfer,
)


def _gaussian_pulse(n: int, dt: float, centre: float, sigma: float) -> np.ndarray:
    t = np.arange(n) * dt
    return np.exp(-0.5 * ((t - centre) / sigma) ** 2)


def test_spice_time_uses_si_suffixes() -> None:
    assert spice_time(1e-9) == "1n"
    assert spice_time(0.2e-9) == "0.2n"
    assert spice_time(40e-9) == "40n"
    assert spice_time(50e-12) == "50p"
    assert spice_time(0.0) == "0"


def test_pulse_source_names_all_seven_p_spice_arguments() -> None:
    """``PULSE(V1 V2 TD TR TF PW PER)`` -- six arguments is a malformed deck."""
    pulse = LaunchPulse()
    source = pulse.spice_source()
    assert source.startswith("PULSE(") and source.endswith(")")
    assert len(source[len("PULSE(") : -1].split()) == 7, source
    assert source == "PULSE(0 1 1n 0.2n 0.2n 0.5n 40n)"


def test_pulse_fall_defaults_to_rise_and_can_be_set() -> None:
    assert LaunchPulse().fall_time_s == LaunchPulse().rise_s
    assert LaunchPulse(fall_s=0.4e-9).fall_time_s == 0.4e-9
    assert "0.4n" in LaunchPulse(fall_s=0.4e-9).spice_source()


def test_pulse_first_null_comes_from_the_flat_top() -> None:
    assert LaunchPulse(width_s=0.5e-9).first_null_hz() == pytest.approx(2e9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rise_s": 0.0},
        {"width_s": 0.0},
        {"delay_s": -1e-9},
        {"period_s": 1e-9},  # no room for the pulse inside its own period
        {"fall_s": 0.0},
        {"amplitude": float("nan")},
    ],
)
def test_pulse_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LaunchPulse(**kwargs)


def test_resample_uniform_regrids_a_non_uniform_series() -> None:
    times = np.array([0.0, 1e-9, 1.5e-9, 4e-9, 9e-9])
    values = np.array([0.0, 1.0, 1.5, 4.0, 9.0])
    grid, sampled = resample_uniform(times, values, dt=1e-9)

    np.testing.assert_allclose(np.diff(grid), 1e-9)
    # the series is the identity in its own units, so the regrid reproduces it
    np.testing.assert_allclose(sampled, grid * 1e9, atol=1e-9)
    assert grid[0] == times[0]
    assert grid[-1] < times[-1]


def test_resample_uniform_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_uniform(np.array([0.0, 2.0, 1.0]), np.zeros(3), dt=1.0)
    with pytest.raises(ValueError, match="disagree"):
        resample_uniform(np.array([0.0, 1.0]), np.zeros(3), dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        resample_uniform(np.array([0.0, 1.0]), np.zeros(2), dt=0.0)
    with pytest.raises(ValueError, match="at least 2"):
        resample_uniform(np.array([0.0]), np.zeros(1), dt=1.0)


def _live_bins(times: np.ndarray, source: np.ndarray, dt: float) -> np.ndarray:
    """Bins where the pulse carries enough energy for the ratio to mean anything.

    Computed from the same resampled grid ``transient_transfer`` uses, so the
    two agree about where the spectrum has underflowed to zero.
    """
    _, source_u = resample_uniform(times, source, dt=dt)
    spectrum = np.abs(np.fft.rfft(source_u))
    return spectrum > SOURCE_FLOOR * spectrum.max()


def test_transfer_recovers_a_known_delay_and_gain_exactly() -> None:
    """A pure delay is a known phase on FFT bins, so the method is pinned hard."""
    n, dt = 1024, 10e-12
    times = np.arange(n) * dt
    source = _gaussian_pulse(n, dt, centre=2e-9, sigma=0.2e-9)
    gain, delay_samples = 0.5, 37

    response = np.zeros(n)
    response[delay_samples:] = gain * source[: n - delay_samples]

    freq_hz, transfer = transient_transfer(times, source, response, dt=dt)
    expected = 2.0 * gain * np.exp(-2j * np.pi * freq_hz * delay_samples * dt)

    # a Gaussian spectrum underflows to exactly zero at high frequency, where
    # the ratio is 0/0 and blanking is the only honest answer -- so the exact
    # ness claim is made only where the pulse actually has energy
    live = _live_bins(times, source, dt)
    np.testing.assert_allclose(transfer[live], expected[live], atol=1e-10)
    # sigma = 0.2 ns puts the 1e-6 roll-off at 4.18 GHz, which is 43 bins at
    # this record length -- the analytic figure, so the band really is covered
    assert freq_hz[live][-1] > 3.5e9, "the pulse must carry energy into the GHz"
    assert np.all(transfer[~live] == 0.0), "dead bins are blanked, not left as 0/0"


def test_transfer_survives_a_non_uniform_solver_axis() -> None:
    """The whole point of the resample: uneven output must not perturb H."""
    n, dt = 1024, 10e-12
    times = np.arange(n) * dt
    source = _gaussian_pulse(n, dt, centre=2e-9, sigma=0.2e-9)
    gain, delay_samples = 0.5, 37
    response = np.zeros(n)
    response[delay_samples:] = gain * source[: n - delay_samples]

    # pretend ngspice printed every other sample in one stretch only
    keep = np.ones(n, dtype=bool)
    keep[400:700:3] = False

    freq_hz, h_u = transient_transfer(times, source, response, dt=dt)
    _, h_subset = transient_transfer(times[keep], source[keep], response[keep], dt=dt)
    expected = 2.0 * gain * np.exp(-2j * np.pi * freq_hz * delay_samples * dt)

    # the expected phase must be built on the grid actually returned: the
    # resample spans [t0, t_last), so it holds one sample fewer than the input
    live = _live_bins(times, source, dt)
    np.testing.assert_allclose(h_u[live], expected[live], atol=1e-10)
    # dropping samples coarsens the record, but the omitted stretch holds no
    # signal, so interpolating back costs almost nothing
    assert np.abs(h_subset[live] - expected[live]).max() < 0.05


def test_transfer_rejects_a_silent_source() -> None:
    n = 64
    times = np.arange(n) * 1e-11
    with pytest.raises(ValueError, match="identically zero"):
        transient_transfer(times, np.zeros(n), np.ones(n), dt=1e-11)


def test_transfer_blanks_bins_where_the_pulse_has_no_energy() -> None:
    n, dt = 512, 10e-12
    times = np.arange(n) * dt
    source = _gaussian_pulse(n, dt, centre=1e-9, sigma=0.1e-9)
    freq_hz, transfer = transient_transfer(times, source, np.zeros(n), dt=dt)

    # a zero response is a legitimate measurement, not an arithmetic failure
    assert np.all(np.isfinite(transfer))
    assert np.allclose(transfer, 0.0)
    # the grid spans [times[0], times[-1]), never extrapolating past the end
    _, grid = resample_uniform(times, source, dt=dt)
    assert freq_hz.size == grid.size // 2 + 1
    assert freq_hz[0] == 0.0


def test_band_mask_keeps_off_dc_and_stays_ordered() -> None:
    freq_hz = np.array([0.0, 0.1e9, 0.5e9, 1.0e9, 2.0e9])
    mask = band_mask(freq_hz, 0.1e9, 1.0e9)
    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 3])
    assert not mask[0], "DC must stay excluded: the chain matrix is singular there"
    with pytest.raises(ValueError, match="stay off DC"):
        band_mask(freq_hz, 0.0, 1e9)
    with pytest.raises(ValueError, match="exceed"):
        band_mask(freq_hz, 2e9, 1e9)


def test_arrival_time_finds_the_first_crossing() -> None:
    times = np.linspace(0, 10e-9, 1001)
    wave = np.zeros_like(times)
    wave[times >= 4e-9] = np.sin(2 * np.pi * 1e9 * (times[times >= 4e-9] - 4e-9))
    # the 5% threshold is one grid step after the sine leaves zero
    assert arrival_time(times, wave, fraction=0.05) == pytest.approx(4e-9, abs=5e-11)

    with pytest.raises(ValueError, match="fraction"):
        arrival_time(times, wave, fraction=1.5)
    with pytest.raises(ValueError, match="identically zero"):
        arrival_time(times, np.zeros_like(times))


def test_flight_time_is_length_over_the_modal_velocity() -> None:
    assert flight_time(0.08, 3.0) == pytest.approx(0.08 * np.sqrt(3.0) / 299792458.0)
    assert flight_time(4, 1.0) == pytest.approx(4 / 299792458.0)
    with pytest.raises(ValueError):
        flight_time(0.0, 3.0)
    with pytest.raises(ValueError):
        flight_time(1.0, 0.5)


# --------------------------------------------------------------------------
# against a real solver
# --------------------------------------------------------------------------


def test_transient_fft_matches_the_analytic_s_parameters() -> None:
    """ngspice ``.tran``, FFT'd, must land on the analytic S-matrix element.

    Driven port 0. Port order is ``[c0_launch, c1_launch, c0_receive,
    c1_receive]``, so row 2 is through-transmission and row 3 far-end
    crosstalk -- getting that pairing backwards produced an error as large as
    the signal itself before this test existed.
    """
    from rf_netlist import build_deck, node_name
    from topologies import default_topology, scattering

    pytest.importorskip("PySpice.Spice.Parser")
    from spice_runner import run_transient

    topology = default_topology("microstrip_pair")
    sections = topology.sections
    through_node = node_name(0, sections)
    cross_node = node_name(1, sections)

    pulse = LaunchPulse()
    deck = build_deck(topology, driven_port=0, source=pulse.spice_source())
    dt, end_time = 10e-12, 20e-9
    times, values, _ = run_transient(
        deck, ["src", through_node, cross_node], step_time=dt, end_time=end_time
    )

    freq_hz, h_through = transient_transfer(times, values["src"], values[through_node], dt=dt)
    _, h_cross = transient_transfer(times, values["src"], values[cross_node], dt=dt)

    mask = band_mask(freq_hz, 0.1e9, 1.9e9)
    band = freq_hz[mask]
    s_mat = scattering(topology, band, discrete=True)

    through_peak = float(np.abs(s_mat[:, 2, 0]).max())
    cross_peak = float(np.abs(s_mat[:, 3, 0]).max())

    through_error = float(np.abs(h_through[mask] - s_mat[:, 2, 0]).max())
    cross_error = float(np.abs(h_cross[mask] - s_mat[:, 3, 0]).max())
    assert through_error < 0.02 * through_peak, (
        f"through: {through_error:.4e} against peak {through_peak:.4e}"
    )
    assert cross_error < 0.03 * cross_peak, (
        f"crosstalk: {cross_error:.4e} against peak {cross_peak:.4e}"
    )

    # the transient cannot beat the ladder's own discretisation by much, so
    # agreement with the distributed line should be the same order of magnitude
    distributed = scattering(topology, band, discrete=False)
    distributed_gap = float(np.abs(s_mat[:, 2, 0] - distributed[:, 2, 0]).max())
    assert through_error < 5.0 * max(distributed_gap, 1e-12)


def test_measured_flight_time_matches_the_modal_prediction() -> None:
    from rf_netlist import build_deck, node_name
    from topologies import default_topology

    pytest.importorskip("PySpice.Spice.Parser")
    from spice_runner import run_transient

    topology = default_topology("microstrip_pair")
    sections = topology.sections
    receive = node_name(0, sections)

    pulse = LaunchPulse()
    deck = build_deck(topology, driven_port=0, source=pulse.spice_source())
    dt, end_time = 10e-12, 20e-9
    times, values, _ = run_transient(deck, ["src", receive], step_time=dt, end_time=end_time)

    measured = arrival_time(times, np.asarray(values[receive]), fraction=0.05)
    crossing = flight_time(topology.length_m, topology.meta["eps_eff"])
    expected = pulse.delay_s + crossing

    assert measured > pulse.delay_s, "the signal cannot arrive before it is launched"
    assert abs(measured - expected) < 0.25 * crossing, (
        f"arrival {measured:.4e} s vs predicted {expected:.4e} s (flight {crossing:.4e} s)"
    )
