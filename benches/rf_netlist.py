"""Write a topology as a sectioned SPICE ladder -- the circuit half of the bench.

``topologies.py`` describes a line with per-unit-length ``R``, ``L``, ``G``,
``C``. ngspice has no element for that, so the line is cut into ``sections``
lumped pi-cells and emitted with basic elements only:

.. code-block:: text

     n0 --[Rs]-- nm --[Ls]-- n1 ...        series R+L per conductor
                |  X  |                    K: mutual inductance
               --- Csh ---                 shunt = row sum of C
     n0 --- Cb --- n1  n1 --- Cb --- n2    bridge = -C off-diagonal

``R`` and ``L`` go through a private ``nm{c}_{s}`` node between them. **Two
elements written between the same node pair are in parallel in SPICE**, and the
parallel combination of ``R*dl`` with ``jw*L*dl`` is dominated by whichever is
smaller -- at 100 MHz on this line ``0.22`` ohm shunts ``j7.77`` ohm, so the
inductor all but disappears and the deck silently models a resistive line.
``topologies.pi_section`` builds ``z_series = (R + jw L) dl``, which is series,
so the deck has to match. The extra node is invisible to :func:`s_from_node_voltages`,
which reads only the port nodes.

The ``X`` is a ``K`` element coupling the series inductors of neighbouring
conductors. SPICE's ``K`` scales two *named* inductors by a coefficient ``m`` and
forms ``M = m sqrt(L1 L2)``, so ``m = L[c, d] / sqrt(L[c, c] L[d, d])``
reproduces the mutual term exactly; naming ``Ls`` rather than the branch as a
whole means ``K`` keeps working across the split. Because ``L`` is positive
definite its normalised form is too, which is the check ngspice applies before
it will run.

The shunt capacitor is the **row sum** of ``C``, not the diagonal -- see
:func:`topologies.spice_shunt_and_bridge` for why those differ and why the
diagonal alone inflates the line's capacitance by the entire mutual term.

Dielectric loss ``G`` is emitted alongside, in the same shunt/bridge split but
as ``Rg``/``Rgb`` resistors rather than capacitors: ``G`` is a conductance
matrix with ``G = omega tan(delta) C``, so ``y_shunt = (G + jw C) dl`` in
:func:`topologies.pi_section` is a capacitor and a resistor in parallel, and
leaving the ``G`` half out of the deck makes every lossy topology disagree with
the analytic chain by about ``tan(delta)``. Topologies with ``tan_delta = 0``
emit no ``Rg``/``Rgb`` at all.

Shunt and bridge capacitors are emitted over *boundaries* (``sections + 1`` of
them), not inside the section loop. A pi-cell splits its shunt in half across
its two ends, so interior boundaries collect a full cell from the sections on
either side while the two ends collect one half; summing over boundaries gives
exactly ``sections`` cells, which is what ``topologies.chain_ladder`` multiplies
out. Emitting them per section instead would double interior nodes and drop the
receive end entirely.

Emitting text is this module's whole job, mirroring ``netlist.py``: it imports
no SPICE binding, so the GPL PySpice stack stays in ``spice_runner.py`` and
never reaches ``delay_tensor``.
"""

from __future__ import annotations

import numpy as np

from topologies import Topology, spice_shunt_and_bridge


def node_name(conductor: int, boundary: int) -> str:
    """SPICE node at ``boundary`` (0 = launch end) on ``conductor``."""
    return f"n{conductor}_{boundary}"


def mid_node(conductor: int, section: int) -> str:
    """Private node splitting ``Rs`` and ``Ls`` inside ``section``.

    Between two boundaries, never on one, so it can never be mistaken for a
    port or collect a shunt capacitor. Named ``nm`` rather than ``n`` so the
    prefix parse in :func:`deck_census` cannot confuse it with ``node_name``.
    """
    return f"nm{conductor}_{section}"


def launch_nodes(topology: Topology) -> list[str]:
    """Nodes at the launch end, in conductor order."""
    return [node_name(c, 0) for c in range(topology.conductors)]


def receive_nodes(topology: Topology) -> list[str]:
    """Nodes at the receive end, in conductor order."""
    return [node_name(c, topology.sections) for c in range(topology.conductors)]


def port_node(topology: Topology, port: int) -> str:
    """SPICE node for ``port`` under the ``[launch..., receive...]`` ordering."""
    ports = 2 * topology.conductors
    if not 0 <= port < ports:
        raise ValueError(f"port must be in [0, {ports}), got {port}")
    return node_name(port % topology.conductors, 0 if port < topology.conductors else topology.sections)


def s_from_node_voltages(
    voltages: dict[str, complex],
    topology: Topology,
    *,
    driven_port: int,
    source_amplitude: complex = 1.0,
) -> np.ndarray:
    """One row of ``S`` from the node voltages of a single-port excitation.

    Every port except the driven one is terminated in ``z0``, so no wave enters
    them from outside and ``b_j = V_j / sqrt(z0)``. At the driven port both
    waves exist, but the source current follows from the Thevenin resistor,
    ``I = (Vs - V_p) / z0``, which collapses the row to

    .. code-block:: text

        S[p, j] = 2 V_j / Vs            j != p   (matched load absorbs)
        S[p, p] = 2 V_p / Vs - 1

    Both limits check out by hand: a matched network gives ``V_p = Vs/2`` and
    therefore ``S[p,p] = 0``, while an open gives ``V_p = Vs`` and ``S[p,p] = 1``.

    ``z0`` cancels out of the ratios -- it sets ``a1`` and ``bj`` identically --
    so it is not a parameter: only the requirement that the source resistance
    and the terminations agree matters, and ``build_deck`` guarantees that.

    Returns ``(F, 2N)``: one row per frequency point, one column per port, all
    from a single sweep. The caller fills ``S[:, :, driven_port]``, i.e. the
    driven port indexes the *column*, because ``S[a, b]`` reads "port ``b`` into
    port ``a``".

    Collecting one column at a time means the full matrix costs ``2N`` runs
    rather than one -- see :func:`build_deck` on why a single run cannot close
    it.
    """
    n = topology.conductors
    ports = 2 * n

    samples = [
        np.asarray(voltages[port_node(topology, port)], dtype=complex).reshape(-1)
        for port in range(ports)
    ]
    count = samples[0].size
    for port, sample in enumerate(samples):
        if sample.size != count:
            raise ValueError(
                f"port {port} has {sample.size} samples, expected {count} -- "
                "every port must come from the same frequency sweep"
            )

    matrix = np.stack(samples, axis=-1) * (2.0 / source_amplitude)
    matrix[:, driven_port] -= 1.0
    return matrix


def build_deck(
    topology: Topology,
    *,
    driven_port: int = 0,
    z0: float = 50.0,
    source: str = "DC 0",
) -> str:
    """Render ``topology`` as a self-contained ngspice deck.

    Ports follow ``topologies``' order -- ``[c0_launch, ..., cN_launch,
    c0_receive, ..., cN_receive]`` -- and exactly one of them is driven; every
    other port is terminated in ``z0``. That is the standard one-port-at-a-time
    S-parameter fixture, so results compare directly against a scikit-rf
    network without de-embedding or renormalisation.

    The driven port may sit at either end. Exciting all ``2N`` ports is what
    reconstructs the *whole* S-matrix: a single launch-side excitation only
    yields one column, and while the ladder is symmetric under reversing the
    conductors, its shunt capacitances differ between outer and inner
    conductors (``[85.5, 39.4, 39.4, 85.5] pF/m`` for the bus), so that symmetry does
    not close the matrix on its own.

    With a matched source the voltage at an undriven receive port is
    ``sqrt(z0)`` times the outgoing wave, so dividing its FFT by the incident
    amplitude gives ``S21`` directly.

    Analysis directives are deliberately absent: ``netlist.py`` follows the
    same convention, and it lets one deck serve an operating point, an ``.ac``
    sweep and a ``.tran`` run without being regenerated.
    """
    n = topology.conductors
    ports = 2 * n
    if not 0 <= driven_port < ports:
        raise ValueError(f"driven_port must be in [0, {ports}), got {driven_port}")
    boundary = 0 if driven_port < n else topology.sections
    driven_conductor = driven_port % n

    shunt, bridge = spice_shunt_and_bridge(topology)
    g_shunt, g_bridge = spice_shunt_and_bridge(topology, matrix=topology.G, label="G")
    dl = topology.section_length_m
    sections = topology.sections

    lines = [
        f"* {topology.name} -- generated by benches/rf_netlist.py",
        f"* {n} conductors x {sections} pi-sections of {dl * 1e3:.4g} mm over "
        f"{topology.length_m * 1e3:.4g} mm",
        f"* loss linearised at {topology.loss_freq_hz / 1e9:.4g} GHz, "
        f"driven port {driven_port} (conductor {driven_conductor} at boundary {boundary})",
        "* reltol: the network is linear, so a tight reltol costs nothing and",
        "* keeps the deck comparable to the analytic chain matrix.",
        ".options reltol=1e-6 abstol=1e-14 vntol=1e-10 chgtol=1e-16",
        "",
        "* --- ports: one Thevenin drive, every other port terminated in z0 ---",
    ]

    for port in range(ports):
        node = node_name(port % n, 0 if port < n else sections)
        if port == driven_port:
            lines.append(f"Vdrv src 0 {source}")
            lines.append(f"Rdrv src {node} {_f(z0)}")
        else:
            lines.append(f"Rt{port} {node} 0 {_f(z0)}")

    lines.append("")
    lines.append(
        "* --- series R+L per section: R and L are split across an nm node, ---"
    )
    lines.append(
        "*     two elements between the same node pair would be in PARALLEL ---"
    )
    for s in range(sections):
        for c in range(n):
            lines.append(
                f"Rs{c}_{s} {node_name(c, s)} {mid_node(c, s)} "
                f"{_f(topology.R[c, c] * dl)}"
            )
            lines.append(
                f"Ls{c}_{s} {mid_node(c, s)} {node_name(c, s + 1)} "
                f"{_f(topology.L[c, c] * dl)}"
            )
        for c in range(n):
            for d in range(c + 1, n):
                product = topology.L[c, c] * topology.L[d, d]
                if product <= 0:
                    raise ValueError(f"{topology.name}: L[{c},{c}] and L[{d},{d}] must be > 0")
                m = topology.L[c, d] / np.sqrt(product)
                if abs(m) > 1e-12:
                    lines.append(f"K{c}{d}_{s} Ls{c}_{s} Ls{d}_{s} {_f(m)}")
                else:
                    # ngspice gathers a section's inductors into one "inductive
                    # system" and refuses to run unless *every* pair is named --
                    # a tridiagonal bus only couples neighbours, so the distant
                    # pairs have to be declared explicitly at zero rather than
                    # left out. The coefficient is the coefficient; m = 0 leaves
                    # the matrix exactly the tridiagonal L we asked for.
                    lines.append(f"K{c}{d}_{s} Ls{c}_{s} Ls{d}_{s} 0")

    lines.append("")
    lines.append("* --- shunt and bridge capacitors: pi-cell split across boundaries ---")
    for b in range(sections + 1):
        share = 0.5 if (b == 0 or b == sections) else 1.0
        for c in range(n):
            if shunt[c] * share > 0:
                lines.append(
                    f"Csh{c}_{b} {node_name(c, b)} 0 {_f(shunt[c] * dl * share)}"
                )
        for c in range(n):
            for d in range(c + 1, n):
                if bridge[c, d] * share > 0:
                    lines.append(
                        f"Cb{c}{d}_{b} {node_name(c, b)} {node_name(d, b)} "
                        f"{_f(bridge[c, d] * dl * share)}"
                    )

    lines.append("")
    lines.append("* --- shunt and bridge conductances: dielectric loss G, same split as C ---")
    for b in range(sections + 1):
        share = 0.5 if (b == 0 or b == sections) else 1.0
        for c in range(n):
            conductance = g_shunt[c] * dl * share
            if conductance > 0:
                lines.append(f"Rg{c}_{b} {node_name(c, b)} 0 {_f(1.0 / conductance)}")
        for c in range(n):
            for d in range(c + 1, n):
                conductance = g_bridge[c, d] * dl * share
                if conductance > 0:
                    lines.append(
                        f"Rgb{c}{d}_{b} {node_name(c, b)} {node_name(d, b)} "
                        f"{_f(1.0 / conductance)}"
                    )

    lines.append("")
    lines.append("* --- receive end is terminated above, with the rest of the ports ---")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def deck_census(deck: str) -> dict[str, int]:
    """Element census parsed from rendered deck text.

    Counting what was actually written -- rather than a parallel arithmetic
    expectation -- means a refactor that changes the deck cannot keep an
    outdated expectation alive, and it keeps ``benches/tests`` runnable in the
    plain uv venv with no ngspice installed.
    """
    prefixes = ("Rs", "Ls", "K", "Csh", "Cb", "Rgb", "Rg", "Rt", "Vdrv", "Rdrv")
    counts = dict.fromkeys(prefixes, 0)
    for line in deck.splitlines():
        if not line or line[0] in "*. \t":
            continue
        name = line.split()[0]
        for prefix in prefixes:
            if _matches(name, prefix):
                counts[prefix] += 1
                break
    return counts


def _matches(name: str, prefix: str) -> bool:
    """Exact-prefix match that stops at the first character after ``prefix``.

    ``Ls3_1`` belongs to ``Ls``; plain ``L`` must not also claim it, ``Rt7``
    belongs to ``Rt`` and not to ``Rtr``-era names, and ``K`` must claim
    ``K01_4`` exactly once.
    """
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix) :]
    if not rest:
        return True
    return rest[0].isdigit() or rest[0] in "_."


def expected_census(topology: Topology, *, driven_port: int = 0) -> dict[str, int]:
    """Element census ``build_deck`` must produce, derived from the topology.

    The port count is ``2N`` and exactly one of them carries the source, so
    there are ``2N - 1`` terminations regardless of which end is driven.
    """
    n, sections = topology.conductors, topology.sections
    # Every c<d pair, not just the ones with m != 0. ngspice gathers a section's
    # inductors into one "inductive system" and refuses to run unless every pair
    # is named, so build_deck emits an explicit zero coefficient for the rest --
    # both branches write a K element. Counting nonzero L off-diagonals instead
    # would agree only while L happens to be dense, which is what made this
    # survive the nearest-neighbour-in-C flip: the old tridiagonal L masked it.
    mutual_pairs = n * (n - 1) // 2
    shunt, bridge = spice_shunt_and_bridge(topology)
    g_shunt, g_bridge = spice_shunt_and_bridge(topology, matrix=topology.G, label="G")
    bridge_nonzero = int(np.count_nonzero(np.triu(bridge, 1)))
    return {
        "Rs": n * sections,
        "Ls": n * sections,
        "K": mutual_pairs * sections,
        "Csh": int(np.count_nonzero(shunt)) * (sections + 1),
        "Cb": bridge_nonzero * (sections + 1),
        "Rg": int(np.count_nonzero(g_shunt > 0)) * (sections + 1),
        "Rgb": int(np.count_nonzero(np.triu(g_bridge, 1) > 0)) * (sections + 1),
        "Rt": 2 * n - 1,
        "Vdrv": 1,
        "Rdrv": 1,
    }


def _f(value: float) -> str:
    """Round-trip a double: ngspice parses with strtod, 17 sig digits is exact."""
    return f"{float(value):.17g}"
