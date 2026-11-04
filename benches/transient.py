"""Time-domain half of the RF bench: pulse in, waveform out, FFT back to ``S``.

The frequency-domain bench (``.ac``, and :func:`rf_char`'s scikit-rf answer)
already settles what the line does to a sinusoid. This measures the same
thing from a pulse, which is the only way to see the two quantities a sweep
cannot show: how long the line actually takes, and what a real edge looks like
arriving.

Three things have to be right for the comparison to mean anything, and each
cost an iteration to discover:

* **ngspice's ``.tran`` output axis is not uniform.** ``step_time`` is a hint;
  the solver prints where it likes, and the observed ``dt`` ranged over two
  orders of magnitude. Everything here resamples onto a uniform grid first --
  :func:`spice_runner.run_transient`'s docstring says as much.
* **The driven and observed ports must be the right pair.** Port order is
  ``[c0_launch, ..., c0_receive, ...]``, so driving port 0 and reading
  ``c1_receive`` yields far-end crosstalk, not through-transmission, and
  comparing that against ``S[0, 1]`` silently produces an error of order the
  signal itself. Ask for ``row, column`` from the caller rather than guessing.
* **The transfer function needs no window.** The record opens at zero, the
  pulse is finite, and the response has decayed to 1e-4 of its peak by the end
  of the window, so the periodic extension is already seamless. Windowing
  both waves would break ``H = Y/X`` in the first place -- it is only valid
  unwindowed.

``H = 2 * FFT(response) / FFT(source)`` is ``S`` and not ``S/2`` because a
Thevenin source behind ``z0`` fixes ``a1 = Vs / (2 sqrt(z0))`` whatever
reflects back, while a matched receive port gives ``b2 = V2 / sqrt(z0)``.

Pure NumPy: no PySpice import, so it is testable in the plain uv venv. Only
:mod:`spice_runner` touches the GPL stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# below this fraction of the pulse's peak spectrum, the ratio is arithmetic on
# numerical noise rather than a measurement
SOURCE_FLOOR = 1e-6


def spice_time(seconds: float) -> str:
    """Format a duration with an SI suffix ngspice accepts (``1.5n``, ``400p``)."""
    for scale, suffix in ((1e-9, "n"), (1e-12, "p"), (1e-6, "u"), (1e-3, "m")):
        scaled = seconds / scale
        if 0.1 <= abs(scaled) < 1e4:
            return f"{scaled:g}{suffix}"
    return f"{seconds:g}"


@dataclass(frozen=True)
class LaunchPulse:
    """A trapezoidal launch pulse: one spec for both the deck and the analysis.

    ``width_s`` sets where the pulse's own spectrum collapses: a trapezoid's
    transform is ``width * sinc(f * width)`` shaped by the edges, so the first
    null sits at ``1 / width_s``. The transfer function *divides* by that
    spectrum, so accuracy falls away as the band approaches it -- which is
    exactly where the bench's worst bins were.
    """

    amplitude: float = 1.0
    delay_s: float = 1e-9
    rise_s: float = 0.2e-9
    fall_s: float | None = None  # defaults to rise_s in spice_source
    width_s: float = 0.5e-9
    period_s: float = 40e-9

    def __post_init__(self) -> None:
        for field_name in ("amplitude", "delay_s", "rise_s", "width_s", "period_s"):
            value = getattr(self, field_name)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        if self.delay_s < 0 or self.rise_s <= 0 or self.width_s <= 0:
            raise ValueError("delay must be >= 0, and rise and width must be > 0")
        if self.fall_s is not None and self.fall_s <= 0:
            raise ValueError(f"fall must be > 0, got {self.fall_s}")
        if self.period_s <= self.delay_s + self.rise_s + self.width_s:
            raise ValueError(
                f"period {self.period_s} s must leave room for delay+rise+width "
                f"({self.delay_s + self.rise_s + self.width_s} s) or the pulse repeats "
                "inside the record"
            )

    @property
    def fall_time_s(self) -> float:
        """Fall time, symmetric with the rise unless overridden."""
        return self.rise_s if self.fall_s is None else self.fall_s

    def spice_source(self) -> str:
        """``PULSE(V1 V2 TD TR TF PW PER)`` operand for an ngspice source."""
        return (
            f"PULSE(0 {self.amplitude:g} {spice_time(self.delay_s)} "
            f"{spice_time(self.rise_s)} {spice_time(self.fall_time_s)} "
            f"{spice_time(self.width_s)} {spice_time(self.period_s)})"
        )

    def first_null_hz(self) -> float:
        """Frequency of the pulse spectrum's first null, from the flat-top width.

        Band-limit below this. The measured worst case was at 95% of it.
        """
        return 1.0 / self.width_s


def resample_uniform(
    times: np.ndarray, values: np.ndarray, *, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a non-uniform solver output onto a uniform grid of spacing ``dt``.

    Linear interpolation is adequate because the solver's own steps are finer
    than ``dt`` where the waveform is moving and coarser where it is not; the
    resampled grid therefore never steps over structure the solver resolved.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.shape != values.shape:
        raise ValueError(f"times {times.shape} and values {values.shape} disagree")
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"need a 1-D series of at least 2 samples, got {times.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")

    grid = np.arange(times[0], times[-1], dt)
    if grid.size < 4:
        raise ValueError(f"dt={dt} leaves only {grid.size} samples over {times[-1] - times[0]} s")
    return grid, np.interp(grid, times, values)


def transient_transfer(
    times: np.ndarray,
    source_wave: np.ndarray,
    response_wave: np.ndarray,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer function ``H = 2 * FFT(response) / FFT(source)`` on the FFT grid.

    Returns ``(freq_hz, H)``. ``H`` equals ``S`` at the driven/observed port
    pair, with the factor of two from the Thevenin source described in the
    module docstring. Bins where the pulse carries no energy come back as zero
    rather than ``inf`` -- :func:`band_mask` is what keeps a caller out of them.
    """
    grid, source_u = resample_uniform(times, source_wave, dt=dt)
    _, response_u = resample_uniform(times, response_wave, dt=dt)

    freq_hz = np.fft.rfftfreq(grid.size, d=dt)
    source_spectrum = np.fft.rfft(source_u)
    peak = np.abs(source_spectrum).max()
    if peak <= 0:
        raise ValueError("the source wave is identically zero; there is nothing to divide by")

    with np.errstate(divide="ignore", invalid="ignore"):
        transfer = 2.0 * np.fft.rfft(response_u) / source_spectrum
    transfer[np.abs(source_spectrum) < SOURCE_FLOOR * peak] = 0.0
    return freq_hz, np.nan_to_num(transfer, nan=0.0, posinf=0.0, neginf=0.0)


def band_mask(freq_hz: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Boolean mask of the open band ``[low_hz, high_hz]``."""
    if low_hz <= 0:
        raise ValueError(f"low_hz must be > 0 to stay off DC, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz {high_hz} must exceed low_hz {low_hz}")
    return (freq_hz >= low_hz) & (freq_hz <= high_hz)


def arrival_time(times: np.ndarray, wave: np.ndarray, *, fraction: float = 0.05) -> float:
    """First time ``|wave|`` crosses ``fraction`` of its peak: a flight time.

    A threshold rather than the peak because the peak arrives later than the
    front; ``fraction`` is a knob about that, and results should be quoted with
    the fraction that produced them.
    """
    times = np.asarray(times, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    envelope = np.abs(wave)
    peak = envelope.max()
    if peak <= 0:
        raise ValueError("the wave is identically zero; it never crosses anything")
    crossed = np.flatnonzero(envelope >= fraction * peak)
    if crossed.size == 0:
        raise ValueError(f"wave never reaches {fraction:.0%} of its peak")
    return float(times[crossed[0]])


def flight_time(length_m: float, eps_eff: float) -> float:
    """Time to cross ``length_m`` of line with effective permittivity ``eps_eff``."""
    if length_m <= 0:
        raise ValueError(f"length must be > 0, got {length_m}")
    if eps_eff < 1:
        raise ValueError(f"eps_eff must be >= 1, got {eps_eff}")
    return length_m * np.sqrt(eps_eff) / 299792458.0
