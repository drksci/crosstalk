"""Checks on the scikit-rf bench -- skipped when the bench group is absent.

:func:`test_uncoupled_pair_decouples` is the one that decides whether the modal
construction is right. It reaches the 4-port through a Hadamard transform and
a reordering, which is indirect enough that matching on coupled data would
prove little; in the limit where the two modes are identical the cross terms
*must* vanish, and that is a closed-form answer rather than an opinion.
"""

from __future__ import annotations

import numpy as np
import pytest

from topologies import C_LIGHT, default_topology, microstrip_z0_eps_eff, z_to_s

pytest.importorskip("skrf")

from skrf import Network

from rf_char import (
    characterize_line,
    conductor_alpha_np_per_m,
    coupled_pair_network,
    dielectric_alpha_np_per_m,
    export_touchstone,
    mode_z_matrix,
)

SWEEP = np.array([0.1e9, 0.5e9, 1.0e9, 1.5e9, 2.0e9])
LENGTH_M = 0.02


def _microstrip() -> dict:
    return default_topology("microstrip_pair").meta


def test_mode_z_matrix_matches_the_textbook_lossless_form() -> None:
    """``Zc coth(gl)`` reduces to ``-j Zc cot(theta)`` once loss is switched off."""
    zc, eps, length = 50.0, 3.0, 0.02
    freq_hz = np.array([0.5e9, 1.5e9])
    theta = 2.0 * np.pi * freq_hz / C_LIGHT * np.sqrt(eps) * length

    got = mode_z_matrix(zc, eps, freq_hz, length)
    np.testing.assert_allclose(got[:, 0, 0], -1j * zc / np.tan(theta), rtol=1e-12)
    np.testing.assert_allclose(got[:, 0, 1], -1j * zc / np.sin(theta), rtol=1e-12)
    np.testing.assert_allclose(got[:, 0, 0], got[:, 1, 1])
    np.testing.assert_allclose(got, np.swapaxes(got, -1, -2))


def test_uncoupled_pair_decouples() -> None:
    """With ``z_even == z_odd`` the pair is two independent lines, not a coupled one.

    This is the closed-form check on the Hadamard/reordering construction: the
    off-diagonal 2x2 blocks must come out zero and each diagonal block must
    equal the single line's own S-matrix.
    """
    s_line = z_to_s(mode_z_matrix(50.0, 3.0, SWEEP, LENGTH_M), 50.0)
    network = coupled_pair_network(
        z_even_ohm=50.0,
        eps_even=3.0,
        z_odd_ohm=50.0,
        eps_odd=3.0,
        freq_hz=SWEEP,
        length_m=LENGTH_M,
    )
    s_pair = network.s

    np.testing.assert_allclose(s_pair[:, 0:2, 0:2], s_line, atol=1e-12)
    np.testing.assert_allclose(s_pair[:, 2:4, 2:4], s_line, atol=1e-12)
    assert np.abs(s_pair[:, 0:2, 2:4]).max() < 1e-12, "the lines must not talk to each other"
    assert np.abs(s_pair[:, 2:4, 0:2]).max() < 1e-12


def test_coupled_network_is_reciprocal_and_passive() -> None:
    meta = _microstrip()
    network = coupled_pair_network(
        z_even_ohm=meta["z_even_ohm"],
        eps_even=meta["eps_even"],
        z_odd_ohm=meta["z_odd_ohm"],
        eps_odd=meta["eps_odd"],
        freq_hz=SWEEP,
        length_m=LENGTH_M,
        alpha_even_np_per_m=dielectric_alpha_np_per_m(SWEEP, meta["eps_even"], meta["tan_delta"]),
        alpha_odd_np_per_m=dielectric_alpha_np_per_m(SWEEP, meta["eps_odd"], meta["tan_delta"]),
    )
    s_mat = network.s

    np.testing.assert_allclose(s_mat, np.swapaxes(s_mat, -1, -2), atol=1e-12)
    worst = float(np.linalg.svd(s_mat)[1].max())
    assert worst <= 1.0 + 1e-9, f"a passive line cannot amplify, got {worst}"
    assert worst < 1.0, "dielectric loss must leave the pair strictly passive"


def test_modes_straddle_the_isolated_line() -> None:
    """The modal split opens around the lone line's value as coupling tightens.

    Microstrip is inhomogeneous: the even mode's field sits mostly in the
    substrate while the odd mode's straddles the gap, which is half air. So
    the odd mode ends up with the *lower* effective permittivity and travels
    faster, and an uncoupled line sits between the two. Splitting in log space
    keeps the geometric mean at the isolated ``eps_eff`` while the pair opens
    up, which is the invariant asserted here.
    """
    meta = _microstrip()
    _, lone_eps = microstrip_z0_eps_eff(meta["w_m"], meta["h_m"], meta["eps_r"])

    assert meta["z_even_ohm"] > meta["z_odd_ohm"], "even mode must be the higher impedance"
    assert meta["eps_odd"] < lone_eps < meta["eps_even"]
    np.testing.assert_allclose(np.sqrt(meta["eps_even"] * meta["eps_odd"]), lone_eps, rtol=1e-12)
    assert meta["velocity_odd_m_per_s"] > meta["velocity_even_m_per_s"], (
        "the odd mode, seeing more air, must travel faster"
    )


def test_thickness_lowers_the_impedance() -> None:
    """A finite conductor widens the trace effectively, so Z0 falls."""
    meta = _microstrip()
    common = {
        "w_m": meta["w_m"],
        "h_m": meta["h_m"],
        "freq_hz": SWEEP,
        "length_m": LENGTH_M,
        "eps_r": meta["eps_r"],
        "tan_delta": meta["tan_delta"],
    }
    thin = characterize_line(**common, t_m=None)
    thick = characterize_line(**common, t_m=meta["t_m"])
    assert np.all(thin.z0.real > thick.z0.real), "thickness must lower Z0 at every frequency"


def test_closed_form_agrees_with_scikit_rf_once_thickness_is_excluded() -> None:
    """The quasi-static form has no thickness term, so ``t_m=None`` is the fair test."""
    meta = _microstrip()
    ours_z0, _ = microstrip_z0_eps_eff(meta["w_m"], meta["h_m"], meta["eps_r"])
    thin = characterize_line(
        w_m=meta["w_m"],
        h_m=meta["h_m"],
        freq_hz=SWEEP,
        length_m=LENGTH_M,
        t_m=None,
        eps_r=meta["eps_r"],
        tan_delta=meta["tan_delta"],
    )

    gap = np.abs(thin.z0.real - ours_z0) / ours_z0
    assert gap.max() < 0.025, f"thin-line Z0 differs by up to {gap.max():.4%} from {ours_z0:.4f}"
    assert gap[-1] < 0.005, f"at the top of the band the gap is {gap[-1]:.4%}"


def test_thickness_accounts_for_most_of_the_gap() -> None:
    """Attribution: the part our closed form omits is the part that differs."""
    meta = _microstrip()
    ours_z0, _ = microstrip_z0_eps_eff(meta["w_m"], meta["h_m"], meta["eps_r"])
    common = {
        "w_m": meta["w_m"],
        "h_m": meta["h_m"],
        "freq_hz": SWEEP,
        "length_m": LENGTH_M,
        "eps_r": meta["eps_r"],
        "tan_delta": meta["tan_delta"],
    }
    thin_gap = np.abs(characterize_line(**common, t_m=None).z0.real - ours_z0) / ours_z0
    thick_gap = np.abs(characterize_line(**common, t_m=meta["t_m"]).z0.real - ours_z0) / ours_z0

    assert np.all(thick_gap > thin_gap), (
        f"omitting thickness must be most of the discrepancy, got thin={thin_gap} thick={thick_gap}"
    )


def test_eps_eff_is_dispersion_scikit_rf_brackets_ours() -> None:
    """Our single quasi-static value sits inside scikit-rf's swept band."""
    meta = _microstrip()
    _, ours_eps = microstrip_z0_eps_eff(meta["w_m"], meta["h_m"], meta["eps_r"])
    reference = characterize_line(
        w_m=meta["w_m"],
        h_m=meta["h_m"],
        freq_hz=SWEEP,
        length_m=LENGTH_M,
        t_m=meta["t_m"],
        eps_r=meta["eps_r"],
        tan_delta=meta["tan_delta"],
    )

    assert reference.eps_eff.max() - reference.eps_eff.min() > 0.02, "dispersion is not negligible"
    assert reference.eps_eff.min() <= ours_eps <= reference.eps_eff.max()
    # it falls across this band: the Djordjevic-Svensson broadband dielectric
    # outweighs Kirschning-Jansen geometric dispersion here, which reverses the
    # direction a quasi-static reading would predict
    assert np.all(np.diff(reference.eps_eff) < 0)


def test_attenuation_scales_as_documented() -> None:
    alpha_d = dielectric_alpha_np_per_m(SWEEP, 3.0, 0.02)
    np.testing.assert_allclose(alpha_d, alpha_d[0] * SWEEP / SWEEP[0], rtol=1e-12)
    assert np.all(alpha_d > 0)
    assert np.all(np.diff(dielectric_alpha_np_per_m(SWEEP, 3.0, 0.04)) > 0), (
        "doubling the loss tangent must double the loss"
    )

    alpha_c = conductor_alpha_np_per_m(SWEEP, 1.5e-3, 50.0)
    assert np.all(alpha_c > 0), "a finite conductor must attenuate"
    assert np.all(np.diff(alpha_c) > 0), "skin depth shrinks with frequency"
    # skin resistance grows as sqrt(f), so attenuation does too
    np.testing.assert_allclose(
        alpha_c[1:] / alpha_c[:-1], np.sqrt(SWEEP[1:] / SWEEP[:-1]), rtol=1e-6
    )


def test_characterize_line_renormalizes_to_the_reference() -> None:
    meta = _microstrip()
    reference = characterize_line(
        w_m=meta["w_m"],
        h_m=meta["h_m"],
        freq_hz=SWEEP,
        length_m=LENGTH_M,
        t_m=meta["t_m"],
        eps_r=meta["eps_r"],
        tan_delta=meta["tan_delta"],
        z0_ref=75.0,
    )
    assert reference.z0_ref == 75.0
    np.testing.assert_allclose(np.asarray(reference.network.z0).real, 75.0, rtol=1e-12)


def test_touchstone_round_trips(tmp_path) -> None:
    meta = _microstrip()
    network = coupled_pair_network(
        z_even_ohm=meta["z_even_ohm"],
        eps_even=meta["eps_even"],
        z_odd_ohm=meta["z_odd_ohm"],
        eps_odd=meta["eps_odd"],
        freq_hz=SWEEP,
        length_m=LENGTH_M,
    )
    path = export_touchstone(network, tmp_path / "coupled.s4p")

    back = Network(path)
    assert back.nports == 4
    np.testing.assert_allclose(back.s, network.s, atol=1e-12)
    np.testing.assert_allclose(np.asarray(back.f), SWEEP, rtol=1e-9)


@pytest.mark.parametrize(
    "bad",
    [np.array([1.0e9]), np.zeros((2, 2)), np.array([-1.0e9, 2.0e9]), np.array([2.0e9, 1.0e9])],
)
def test_sweep_validation(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        characterize_line(w_m=1.5e-3, h_m=0.8e-3, freq_hz=bad, length_m=LENGTH_M)
