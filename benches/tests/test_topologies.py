"""Pure-NumPy checks on the line models -- no ngspice, no scikit-rf.

These run in the ordinary uv venv. They pin the two properties the RF bench
rests on and that are easy to break without noticing:

* ``C`` must be a **Maxwell** capacitance matrix -- off-diagonals ``<= 0`` and
  non-negative row sums -- because that is the only shape
  :func:`topologies.spice_shunt_and_bridge` can turn into two-terminal parts.
  A positive off-diagonal wants a negative capacitor, which no deck contains.
* every mode must travel at ``c / sqrt(eps_eff)``, which is what makes the
  line homogeneous rather than an arbitrary ladder.

The first is the regression behind ``mcl``'s worst failure so far: deriving
``C = (eps/c^2) L^-1`` from a *tridiagonal* ``L`` gives ``C`` a checkerboard
sign pattern, so ``C[0,2]`` came out positive and the deck's ``clip`` dropped
it silently -- 0.56 of analytic-versus-ngspice error with nothing in the deck
to point at.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from topologies import (
    C_LIGHT,
    _adjacency,
    _modal_partner,
    coupling_from_gap,
    default_topology,
    microstrip_pair,
    microstrip_z0_eps_eff,
    mode_parameters,
    multiconductor_bus,
    scattering,
    spice_shunt_and_bridge,
)

KINDS = ("mcl", "microstrip_pair", "lc_ladder")


@pytest.mark.parametrize("kind", KINDS)
def test_default_topologies_are_symmetric_and_positive_definite(kind: str) -> None:
    topology = default_topology(kind)
    # atol=0: the defaults would let a 1e-9 asymmetry past a 1e-10 matrix
    for label, matrix in (
        ("R", topology.R),
        ("L", topology.L),
        ("G", topology.G),
        ("C", topology.C),
    ):
        assert np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0), (
            f"{kind} {label} is not symmetric"
        )
    for label, matrix in (("L", topology.L), ("C", topology.C)):
        assert np.all(np.linalg.eigvalsh(matrix) > 0), f"{kind} {label} is not positive definite"


def test_mcl_capacitance_is_tridiagonal_and_maxwell() -> None:
    """Nearest-neighbour coupling lives in ``C``, where a deck can contain it."""
    topology = default_topology("mcl")
    off_diagonal = topology.C - np.diag(np.diag(topology.C))

    assert np.all(off_diagonal <= 0.0), "Maxwell off-diagonals must be <= 0"
    # scale-relative, not np.allclose's absolute default: the entries the old
    # checkerboard bug produced were 2e-11, four orders under that default
    tolerance = 1e-12 * np.abs(topology.C).max()
    assert np.allclose(np.triu(off_diagonal, 2), 0.0, atol=tolerance), "C reached 0 -> 2"
    assert np.allclose(np.tril(off_diagonal, -2), 0.0, atol=tolerance)

    shunt, bridge = spice_shunt_and_bridge(topology)
    assert np.all(shunt >= 0.0) and np.all(bridge >= 0.0)


def test_mcl_modes_are_degenerate() -> None:
    """``L C = (eps_eff/c^2) I`` -- every mode at ``c/sqrt(eps_eff)`` regardless of coupling."""
    topology = default_topology("mcl")
    target = topology.meta["eps_eff"] / C_LIGHT**2
    eigenvalues = np.linalg.eigvals(topology.L @ topology.C).real
    np.testing.assert_allclose(eigenvalues, target, rtol=1e-9)

    modes = mode_parameters(topology, 1e9)
    assert bool(modes["simultaneous"][0]), "L and C must share an eigenbasis"
    np.testing.assert_allclose(
        modes["velocity_m_per_s"], C_LIGHT / np.sqrt(topology.meta["eps_eff"]), rtol=1e-9
    )


def test_mcl_mean_self_inductance_matches_the_configured_value() -> None:
    l_self = 400e-9
    topology = multiconductor_bus(4, l_self=l_self)
    np.testing.assert_allclose(np.mean(np.diag(topology.L)), l_self, rtol=1e-9)


def test_mcl_inductive_tail_decays_geometrically() -> None:
    """Nearest-neighbour dominant: ``d1`` largest, then a geometric fall at ``k_cap``."""
    topology = default_topology("mcl")
    k_cap = topology.meta["k_cap"]
    first = np.abs(np.diag(topology.L, 1))
    second, third = abs(topology.L[0, 2]), abs(topology.L[0, 3])

    assert np.all(first > second), f"d1 must exceed d2, got {first} vs {second}"
    assert second > third > 0.0
    np.testing.assert_allclose(third / second, k_cap, rtol=1e-3)


def test_bus_rejects_k_cap_that_loses_positive_definiteness() -> None:
    with pytest.raises(ValueError, match="k_cap"):
        multiconductor_bus(4, k_cap=0.5)
    with pytest.raises(ValueError, match="k_cap"):
        multiconductor_bus(4, k_cap=-0.1)


def test_pair_formulations_agree_exactly() -> None:
    """For two conductors the two spellings coincide, so ``lc_ladder`` is untouched.

    The inverse of a tridiagonal matrix only acquires its checkerboard signs at
    distance 2 and beyond, so a pair is the one case where building ``L``
    nearest-neighbour and building ``C`` nearest-neighbour give the same line.
    """
    l_self, k_cap, eps_eff = 280e-9, 0.30, 3.0
    topology = multiconductor_bus(2, l_self=l_self, k_cap=k_cap, eps_eff=eps_eff, sections=4)

    np.testing.assert_allclose(topology.L, l_self * (np.eye(2) + k_cap * _adjacency(2)), rtol=1e-12)
    np.testing.assert_allclose(
        topology.C, (eps_eff / C_LIGHT**2) * np.linalg.inv(topology.L), rtol=1e-12
    )


def test_modal_partner_round_trips() -> None:
    """The pairing is symmetric: ``partner(partner(X)) == X`` either way round.

    An eigendecomposition cannot keep structural zeros exact, so the round trip
    lands a few e-26 in ``C``'s distance-2 slots; compare at that scale rather
    than against an absolute zero no decomposition can reproduce.
    """
    eps_eff = [3.0] * 4
    topology = default_topology("mcl")
    tolerance = 1e-12 * np.abs(topology.C).max()

    partner = _modal_partner(topology.C, eps_eff)
    np.testing.assert_allclose(partner, topology.L, rtol=1e-9, atol=tolerance)
    np.testing.assert_allclose(
        _modal_partner(partner, eps_eff), topology.C, rtol=1e-9, atol=tolerance
    )


def test_modal_partner_rejects_indefinite_input() -> None:
    # eigenvalues 3 and -1
    with pytest.raises(ValueError, match="positive definite"):
        _modal_partner(np.array([[1.0, 2.0], [2.0, 1.0]]), [3.0, 3.0])


def test_shunt_and_bridge_refuses_a_positive_off_diagonal() -> None:
    """The silent-clip regression: ``clip`` is how this used to fail with no cause."""
    topology = default_topology("mcl")
    broken = topology.C.copy()
    broken[0, 2] = broken[2, 0] = abs(broken[0, 1])
    with pytest.raises(ValueError, match="positive off-diagonal"):
        spice_shunt_and_bridge(dataclasses.replace(topology, C=broken))


def test_shunt_is_the_row_sum_not_the_diagonal() -> None:
    topology = default_topology("mcl")
    shunt, bridge = spice_shunt_and_bridge(topology)

    np.testing.assert_allclose(shunt, topology.C.sum(axis=1))
    expected_bridge = np.clip(-topology.C, 0.0, None)
    np.fill_diagonal(expected_bridge, 0.0)
    np.testing.assert_allclose(bridge, expected_bridge)
    # rtol only: the absolute default of 1e-8 is four orders above these values
    # and would call the row sum and the diagonal "close" across a 35% gap
    assert not np.allclose(shunt, np.diag(topology.C), rtol=1e-3, atol=0.0), (
        "the diagonal alone would double-count the bridge and inflate the line"
    )


def test_shunt_and_bridge_applies_the_same_split_to_G() -> None:
    """``G = omega tan(delta) C`` is a conductance matrix, so it splits identically."""
    topology = default_topology("microstrip_pair")
    shunt, bridge = spice_shunt_and_bridge(topology)
    g_shunt, g_bridge = spice_shunt_and_bridge(topology, matrix=topology.G, label="G")

    scale = 2.0 * np.pi * topology.loss_freq_hz * topology.meta["tan_delta"]
    np.testing.assert_allclose(g_shunt, scale * shunt, rtol=1e-9)
    np.testing.assert_allclose(g_bridge, scale * bridge, rtol=1e-9)


def test_microstrip_modal_impedances_reproduce_the_gap_coupling() -> None:
    w, gap, h, eps_r = 1.5e-3, 0.4e-3, 0.8e-3, 4.3
    topology = microstrip_pair(w=w, gap=gap, h=h, eps_r=eps_r)
    modes = mode_parameters(topology, 1e9)
    meta = topology.meta

    assert modes["z_mode_ohm"].size == 2, "a pair has two modes"
    np.testing.assert_allclose(
        modes["z_mode_ohm"], [meta["z_even_ohm"], meta["z_odd_ohm"]], rtol=1e-9
    )

    # the stored mode impedances must be the geometry's, not free parameters
    z_even, z_odd = meta["z_even_ohm"], meta["z_odd_ohm"]
    z0, _ = microstrip_z0_eps_eff(w, h, eps_r)
    np.testing.assert_allclose(meta["z0_single_ohm"], z0, rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(z_even * z_odd), z0, rtol=1e-9)

    k = (z_even - z_odd) / (z_even + z_odd)
    np.testing.assert_allclose(k, meta["k_coupling"], rtol=1e-9)
    np.testing.assert_allclose(k, coupling_from_gap(gap, h), rtol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_scattering_is_reciprocal(kind: str) -> None:
    s_mat = scattering(default_topology(kind), np.array([0.2e9, 1.0e9, 2.0e9]))
    np.testing.assert_allclose(s_mat, np.swapaxes(s_mat, -1, -2), atol=1e-12, err_msg=kind)


def test_ladder_converges_to_the_distributed_line_at_second_order() -> None:
    """``O(sections^-2)``: ~4x less error per doubling. That is what makes the
    sectioned deck a discretisation of the line rather than a different model.
    """
    freqs = np.array([0.5e9, 1.0e9])
    errors = []
    for sections in (8, 16, 32, 64):
        topology = microstrip_pair(sections=sections)
        ladder = scattering(topology, freqs, discrete=True)
        distributed = scattering(topology, freqs, discrete=False)
        errors.append(float(np.abs(ladder - distributed).max()))

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(ratio > 3.0 for ratio in ratios), (
        f"expected the ladder error to fall ~4x per doubling, got {ratios} from {errors}"
    )


def test_pair_decouples_as_the_gap_opens() -> None:
    """``k -> 0`` must give two isolated ``Z0`` lines with the modal split closed.

    ``coupling_from_gap`` is ``sech(pi gap / 2h)``, so a wide enough gap drives
    ``k`` below float resolution and ``p = sqrt((1+k)/(1-k))`` rounds to exactly
    1 -- which is what lets the limit be asserted exactly rather than
    approximately.
    """
    pair = microstrip_pair(gap=100.0, sections=4)
    z0, eps_eff = microstrip_z0_eps_eff(pair.meta["w_m"], pair.meta["h_m"], pair.meta["eps_r"])

    np.testing.assert_allclose(pair.meta["z_even_ohm"], z0, rtol=1e-12)
    np.testing.assert_allclose(pair.meta["z_odd_ohm"], z0, rtol=1e-12)
    np.testing.assert_allclose(pair.meta["eps_even"], eps_eff, rtol=1e-12)
    np.testing.assert_allclose(pair.meta["eps_odd"], eps_eff, rtol=1e-12)

    # and the matrices really are two identical, uncoupled lines
    assert abs(pair.L[0, 1]) < 1e-12 * np.abs(pair.L).max()
    assert abs(pair.C[0, 1]) < 1e-12 * np.abs(pair.C).max()


def test_pair_refuses_touching_traces() -> None:
    """``k -> 1`` makes ``Zeven`` infinite; say so instead of dividing by zero."""
    with pytest.raises(ValueError, match="contact"):
        microstrip_pair(gap=0.0, sections=4)
