"""Pure-Python checks on the RF deck generator -- no ngspice required.

``spice_runner`` is the only module that imports the GPL PySpice stack, so
everything here runs in the ordinary uv venv.

Three of these pin bugs that were each invisible until they produced a wrong
number in an unrelated place:

* ``test_series_resistance_and_inductance_are_never_in_parallel`` -- two
  elements written between one node pair are *parallel* in SPICE. The deck
  used to put ``R`` and ``L`` there, so ``0.22`` ohm shunted ``j7.77`` ohm and
  the inductor all but vanished.
* ``test_every_inductor_pair_is_named`` -- ngspice refuses ``run`` unless a
  section's ``K`` set is complete, even for pairs whose coefficient is zero.
* ``test_k_count_survives_a_structurally_zero_mutual`` -- the census used to
  count nonzero ``L`` off-diagonals, which agreed only while ``L`` was dense.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rf_netlist import (
    build_deck,
    deck_census,
    expected_census,
    launch_nodes,
    mid_node,
    node_name,
    port_node,
    receive_nodes,
    s_from_node_voltages,
)
from topologies import default_topology

KINDS = ("mcl", "microstrip_pair", "lc_ladder")


@pytest.mark.parametrize("kind", KINDS)
def test_deck_census_matches_expected(kind: str) -> None:
    topology = default_topology(kind)
    ports = 2 * topology.conductors
    for driven_port in (0, topology.conductors, ports - 1):
        deck = build_deck(topology, driven_port=driven_port)
        assert deck_census(deck) == expected_census(topology), f"{kind} port {driven_port}"


@pytest.mark.parametrize("kind", KINDS)
def test_every_inductor_pair_is_named(kind: str) -> None:
    """ngspice gathers a section's inductors into one system and wants every pair."""
    topology = default_topology(kind)
    n, sections = topology.conductors, topology.sections
    deck = build_deck(topology)
    for s in range(sections):
        for c in range(n):
            for d in range(c + 1, n):
                assert f"K{c}{d}_{s} " in deck, f"{kind}: missing K{c}{d}_{s}"


@pytest.mark.parametrize("kind", KINDS)
def test_series_resistance_and_inductance_are_never_in_parallel(kind: str) -> None:
    """Each ``Rs``/``Ls`` must own a distinct node pair, or SPICE puts them in parallel."""
    deck = build_deck(default_topology(kind))
    pairs: set[frozenset[str]] = set()
    for line in deck.splitlines():
        if line.startswith(("Rs", "Ls")):
            parts = line.split()
            pair = frozenset(parts[1:3])
            assert pair not in pairs, f"{kind}: {parts[0]} reuses the node pair {sorted(pair)}"
            pairs.add(pair)


def test_k_count_survives_a_structurally_zero_mutual() -> None:
    """A zero ``L[c, d]`` still costs a ``K`` element -- ngspice wants the pair named.

    This is the gap the census used to have: counting nonzero off-diagonals
    agreed only while ``L`` was dense, which is exactly the condition the
    nearest-neighbour-in-``C`` flip happened to leave behind.
    """
    topology = default_topology("mcl")
    sparse_l = topology.L.copy()
    sparse_l[0, 2] = sparse_l[2, 0] = 0.0
    sparse = dataclasses.replace(topology, L=sparse_l)

    n, sections = sparse.conductors, sparse.sections
    assert int(np.count_nonzero(np.triu(sparse.L, 1))) == n * (n - 1) // 2 - 1, (
        "the fixture must actually have a zero, or this proves nothing"
    )

    got = deck_census(build_deck(sparse))
    assert got == expected_census(sparse)
    assert got["K"] == n * (n - 1) // 2 * sections


def test_loss_resistors_appear_only_for_lossy_topologies() -> None:
    lossless = default_topology("lc_ladder")  # tan_delta = 0
    census = deck_census(build_deck(lossless))
    assert census["Rg"] == 0 and census["Rgb"] == 0

    lossy = default_topology("microstrip_pair")
    census = deck_census(build_deck(lossy))
    assert census["Rg"] > 0 and census["Rgb"] > 0
    # G = omega tan(delta) C, so every shunt capacitor gets a parallel resistor
    assert census["Rg"] == census["Csh"]
    assert census["Rgb"] == census["Cb"]


def test_shunt_conductance_resistor_is_the_inverse_conductance() -> None:
    topology = default_topology("microstrip_pair")
    dl = topology.section_length_m
    g_shunt = topology.G.sum(axis=1)
    boundary, share = 0, 0.5  # the two ends carry half a pi-cell

    line = next(ln for ln in build_deck(topology).splitlines() if ln.startswith(f"Rg0_{boundary} "))
    np.testing.assert_allclose(float(line.split()[-1]), 1.0 / (g_shunt[0] * dl * share), rtol=1e-12)


def test_port_nodes_are_unique_and_ordered_launch_then_receive() -> None:
    topology = default_topology("mcl")
    n = topology.conductors
    ports = [port_node(topology, p) for p in range(2 * n)]

    assert len(set(ports)) == len(ports)
    assert ports[:n] == launch_nodes(topology)
    assert ports[n:] == receive_nodes(topology)
    with pytest.raises(ValueError, match="port"):
        port_node(topology, 2 * n)


def test_mid_node_is_never_a_port_node() -> None:
    """The private ``R``/``L`` node must not be mistakable for a boundary."""
    topology = default_topology("mcl")
    ports = {port_node(topology, p) for p in range(2 * topology.conductors)}
    assert mid_node(0, 0) not in ports
    assert not mid_node(0, 0).startswith(node_name(0, 0)), "nm must not parse as an n node"


def test_element_names_are_unique() -> None:
    for kind in KINDS:
        deck = build_deck(default_topology(kind))
        names = [ln.split()[0] for ln in deck.splitlines() if ln and ln[0] not in "*. \t"]
        assert len(set(names)) == len(names), f"{kind}: duplicate element name"


def test_s_row_follows_the_matched_and_open_limits() -> None:
    """The two limits the row formula is derived from: ``V_p = Vs/2`` and ``V_p = Vs``."""
    topology = default_topology("microstrip_pair")
    ports = [port_node(topology, p) for p in range(2 * topology.conductors)]

    def row_with(everywhere: float, driven: float) -> np.ndarray:
        voltages = {name: np.array([everywhere + 0.0j]) for name in ports}
        voltages[ports[0]] = np.array([driven + 0.0j])
        matrix = s_from_node_voltages(voltages, topology, driven_port=0)
        assert matrix.shape == (1, len(ports)), "frequency-major: (F, 2N)"
        return matrix[0]

    matched = row_with(0.0, 0.5)
    np.testing.assert_allclose(matched, 0.0, atol=1e-15), "a matched load must give S = 0"

    open_circuit = row_with(0.0, 1.0)
    np.testing.assert_allclose(open_circuit[0], 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(open_circuit[1:], 0.0, atol=1e-15)

    # every port at Vs/2: undriven ports see a full wave, the driven one none
    through = row_with(0.5, 0.5)
    np.testing.assert_allclose(through[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(through[1:], 1.0, rtol=0.0, atol=1e-15)


def test_s_row_rejects_ports_from_different_sweeps() -> None:
    topology = default_topology("microstrip_pair")
    ports = [port_node(topology, p) for p in range(2 * topology.conductors)]
    voltages = {name: np.zeros(3, dtype=complex) for name in ports}
    voltages[ports[1]] = np.zeros(5, dtype=complex)
    with pytest.raises(ValueError, match="frequency sweep"):
        s_from_node_voltages(voltages, topology, driven_port=0)
