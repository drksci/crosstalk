"""PySpice glue: turn netlist text into solved node voltages.

This is the only module in ``benches/`` that imports PySpice, and PySpice is
GPL-3.0, so nothing under ``packages/`` may ever import *this*. The split is
deliberate: ``netlist.py`` and ``rf_netlist.py`` are pure text generation and
stay redistributable alongside the package, while the GPL stays contained to
the runner.

The bench group carries PySpice and scikit-rf, and it is *not* a default uv
group, so ``uv sync --all-packages`` keeps the test environment and the
delay-tensor wheel both free of GPL code. Run against it::

    uv run --all-packages --group bench ...
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
import warnings

import numpy as np

from netlist import state_node_names

# PySpice keeps a supported-version list that predates ngspice 41 and warns on
# every connect. The handshake works regardless, so drop just that record.
_UNSUPPORTED = "Unsupported Ngspice version"


class _DropVersionBanner(logging.Filter):
    """Drop exactly the ngspice-version record, nothing else."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _UNSUPPORTED not in record.getMessage()


# A logger's own filters run only for records it emits itself; filters on an
# ancestor are never consulted as the record propagates. This record comes from
# NgSpiceShared.getChild(...) under PySpice.Spice.NgSpice.Shared, so the filter
# has to sit on that exact logger to intercept it.
_BANNER_LOGGER = "PySpice.Spice.NgSpice.Shared.NgSpiceShared"


@contextlib.contextmanager
def _quiet_py_spice():
    """Silence the ngspice-version warning without muting real diagnostics.

    Only the filter is needed: a record that fails it is dropped before any
    handler runs, while one that passes still falls through to logging's
    lastResort (stderr), since PySpice installs no handlers of its own. Adding
    a stand-in handler here would instead mark every record handled and
    swallow PySpice's real warnings along with the banner.
    """
    logger = logging.getLogger(_BANNER_LOGGER)
    banner = _DropVersionBanner()
    logger.addFilter(banner)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f".*{_UNSUPPORTED}.*")
            yield
    finally:
        logger.removeFilter(banner)


def solve_operating_point(netlist: str, names: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Solve ``netlist`` at its operating point and read back ``names``.

    Returns the values in the order given, plus a timing breakdown. The deck
    is written to a temp file because PySpice's parser wants a path, not a
    string.
    """
    # Imported lazily so importing this module does not drag in the GPL stack,
    # and so netlist.py's tests never touch it.
    from PySpice.Spice.Parser import SpiceParser

    fd, path = tempfile.mkstemp(suffix=".cir", prefix="delay_tensor_")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(netlist)

        started = time.perf_counter()
        with _quiet_py_spice():
            circuit = SpiceParser(path).build_circuit()
            parsed = time.perf_counter()
            analysis = circuit.simulator(simulator="ngspice-shared").operating_point()
            solved = time.perf_counter()
    finally:
        os.unlink(path)

    values = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        raw = np.asarray(analysis[name]).reshape(-1)[0]
        values[index] = float(np.real(raw))

    return values, {"parse_s": parsed - started, "solve_s": solved - parsed}


def solve_states(
    netlist: str, batch: int, wires: int, slots: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve and reshape straight into ``(batch, wires, slots)``."""
    names = state_node_names(batch, wires, slots)
    values, timings = solve_operating_point(netlist, names)
    return values.reshape(batch, wires, slots), timings


def run_ac(
    netlist: str,
    names: list[str],
    *,
    start_hz: float,
    stop_hz: float,
    points: int,
    variation: str = "lin",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Sweep ``netlist`` and return complex node voltages over frequency.

    ``ngspice``'s ``.ac`` uses ``variation`` (``lin``/``dec``/``oct``) with
    ``points`` steps between the bounds, so the returned frequency axis is
    whatever ngspice actually solved -- read back from the analysis rather than
    reconstructed, because a ``dec`` sweep's last point is not ``stop_hz``.

    Complex node voltages are exactly what the S-parameter extraction needs:
    with a Thevenin source ``a1 = Vs / (2 sqrt(z0))`` and a matched receive
    port giving ``b2 = V2 / sqrt(z0)``, so ``S21 = 2 V2 / Vs`` needs no phase
    reference beyond the node itself.
    """
    from PySpice.Spice.Parser import SpiceParser

    fd, path = tempfile.mkstemp(suffix=".cir", prefix="delay_tensor_")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(netlist)

        started = time.perf_counter()
        with _quiet_py_spice():
            circuit = SpiceParser(path).build_circuit()
            parsed = time.perf_counter()
            simulator = circuit.simulator(simulator="ngspice-shared")
            analysis = simulator.ac(variation, points, start_hz, stop_hz)
            solved = time.perf_counter()
    finally:
        os.unlink(path)

    freqs = np.asarray(analysis.frequency, dtype=np.float64).reshape(-1)
    values = {name: np.asarray(analysis[name], dtype=complex).reshape(-1) for name in names}
    return freqs, values, {"parse_s": parsed - started, "solve_s": solved - parsed}


def run_transient(
    netlist: str,
    names: list[str],
    *,
    step_time: float,
    end_time: float,
    start_time: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run ``.tran`` and return real node voltages against a time axis.

    ``step_time`` is ngspice's *output* interval; internal timesteps are chosen
    by the solver and are typically much finer, so the returned axis is read
    back from ``analysis.time`` rather than assumed to be uniform. The macromodel
    extractor re-samples onto a uniform grid afterwards for exactly that reason.
    """
    from PySpice.Spice.Parser import SpiceParser

    fd, path = tempfile.mkstemp(suffix=".cir", prefix="delay_tensor_")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(netlist)

        started = time.perf_counter()
        with _quiet_py_spice():
            circuit = SpiceParser(path).build_circuit()
            parsed = time.perf_counter()
            simulator = circuit.simulator(simulator="ngspice-shared")
            analysis = simulator.transient(
                step_time=step_time, end_time=end_time, start_time=start_time
            )
            solved = time.perf_counter()
    finally:
        os.unlink(path)

    times = np.asarray(analysis.time, dtype=np.float64).reshape(-1)
    values = {name: np.asarray(analysis[name], dtype=np.float64).reshape(-1) for name in names}
    return times, values, {"parse_s": parsed - started, "solve_s": solved - parsed}
