"""Fit a compact state-space macromodel to a line's impulse response.

This is the pivot of the whole RF bench. ``topologies.py`` and
``rf_netlist.py`` describe a line and build a deck that simulates it; this
module compresses what comes back out into a discrete system

.. code-block:: text

    x[k+1] = A x[k] + B u[k]
    y[k]   = C x[k] + D u[k]

small enough to run inside the delay-tensor reservoir, so the reservoir's
update stops being an ad-hoc recurrence and becomes the macromodel of a real
line. The fit is the Eigensystem Realization Algorithm (Ho-Kalman in its
deterministic form): it recovers a minimal system exactly from a finite run of
its own impulse response, and truncates gracefully when the data is noisy.

Why this rather than fitting the transfer function, or reading the poles off
``chain_ladder`` directly: the impulse response is the one artefact every
source produces the same way. ngspice's ``.tran``, the analytic chain matrix
and scikit-rf's inverse FFT all yield the same ``h[k]``, so one fit covers all
three and any disagreement between them shows up as a larger residual rather
than as a different model. Fitting poles instead would need each source to be
parsed on its own terms.

Pure NumPy on purpose -- no PySpice, no scikit-rf -- so this runs in the
ordinary uv venv and the macromodel is usable from ``delay_tensor`` without
pulling the GPL stack into the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transient import SOURCE_FLOOR, resample_uniform


@dataclass(frozen=True)
class MacroModel:
    """A discrete state-space realization of a line, fitted from Markov parameters.

    ``A`` is ``(r, r)``, ``B`` is ``(r, m)``, ``C`` is ``(p, r)`` and ``D`` is
    ``(p, m)``, where ``m`` inputs feed ``p`` outputs through ``r`` states.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    singular_values: np.ndarray

    @property
    def order(self) -> int:
        """Number of retained states."""
        return int(self.A.shape[0])

    @property
    def inputs(self) -> int:
        return int(self.B.shape[1])

    @property
    def outputs(self) -> int:
        return int(self.C.shape[0])

    def impulse_response(self, steps: int) -> np.ndarray:
        """Reproduce the Markov parameters, shape ``(steps, p, m)``.

        ``h[0] = D`` and ``h[k] = C A^(k-1) B`` for ``k >= 1``. A line is
        passive, so ``|eig(A)| <= 1`` and the powers decay rather than
        overflowing.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        out = np.zeros((steps, self.outputs, self.inputs))
        out[0] = self.D
        # advance AFTER reading, so h[k] = C A^(k-1) B -- the convention the
        # Hankel factorization in fit_macromodel assumes
        power = np.eye(self.order)
        for k in range(1, steps):
            out[k] = self.C @ power @ self.B
            power = self.A @ power
        return out

    def max_error(self, markov: np.ndarray) -> float:
        """Largest absolute deviation from the Markov parameters it was fitted to."""
        observed = np.asarray(markov, dtype=np.float64)
        return float(np.abs(self.impulse_response(observed.shape[0]) - observed).max())

    def spectral_radius(self) -> float:
        """``max |eig(A)|``. Above 1 the realization is not the passive line it claims."""
        return float(np.abs(np.linalg.eigvals(self.A)).max())


def block_hankel(markov: np.ndarray, blocks: int, *, offset: int = 1) -> np.ndarray:
    """Block Hankel matrix ``H[offset]``, shape ``(p * blocks, m * blocks)``.

    Entry ``(i, j)`` is the Markov parameter at ``offset + i + j``, so both
    diagonals of blocks carry the same parameter -- that repetition is what
    lets the factorization below recover the system rather than merely
    interpolate it.
    """
    steps, outputs, inputs = markov.shape
    if blocks < 1:
        raise ValueError(f"blocks must be >= 1, got {blocks}")
    if steps < offset + 2 * blocks - 1:
        raise ValueError(
            f"{steps} Markov parameters cannot fill a {blocks}x{blocks} Hankel at "
            f"offset {offset}; need at least {offset + 2 * blocks - 1}"
        )

    hankel = np.zeros((outputs * blocks, inputs * blocks))
    for i in range(blocks):
        for j in range(blocks):
            hankel[i * outputs : (i + 1) * outputs, j * inputs : (j + 1) * inputs] = markov[
                offset + i + j
            ]
    return hankel


def fit_macromodel(
    markov: np.ndarray,
    *,
    order: int | None = None,
    tolerance: float = 1e-10,
    blocks: int | None = None,
) -> MacroModel:
    """Fit a realization to ``markov``, shape ``(K + 1, p, m)``.

    ``markov[k]`` is the system's response at lag ``k`` to an impulse, with
    ``markov[0]`` the direct feedthrough. ``h[0]`` enters as ``D`` and the rest
    through a singular-value decomposition of the Hankel matrix built from
    ``h[1]``:

    .. code-block:: text

        H0 = U S V^T          (block Hankel from h[1], h[2], ...)
        H1                     (the same, advanced one lag)
        A = S^-1/2 U^T H1 V S^-1/2
        B = S^1/2 V^T E        E: first m block-columns
        C = F^T U S^1/2        F: first p block-rows

    Truncation is what makes this usable on simulated data. ``order`` sets the
    retained rank directly; otherwise singular values below
    ``tolerance * s[0]`` are dropped, so a noise floor does not become states.
    A realization fitted at rank ``r`` reproduces every lag whose contribution
    lives in those ``r`` directions, which is why the residual, not the rank,
    is the thing to watch.
    """
    observed = np.asarray(markov, dtype=np.float64)
    if observed.ndim != 3:
        raise ValueError(f"markov must be (K+1, p, m), got shape {observed.shape}")
    steps, outputs, inputs = observed.shape
    if steps < 2:
        raise ValueError(f"need at least 2 Markov parameters, got {steps}")

    if blocks is None:
        blocks = max(1, (steps - 1) // 2)

    h0 = block_hankel(observed, blocks, offset=1)
    h1 = block_hankel(observed, blocks, offset=2)

    left, singular, right_t = np.linalg.svd(h0, full_matrices=False)

    if singular[0] <= 0.0:
        # without this the inverse square root below is 1/sqrt(0), and the fit
        # returns all-NaN rather than saying what is wrong
        raise ValueError(
            "the Hankel matrix is identically zero, so there is no system to fit -- "
            "check that markov holds responses rather than zeros"
        )

    if order is None:
        order = int(np.count_nonzero(singular > tolerance * singular[0]))
    ceiling = min(int(np.count_nonzero(singular > 0)), *h0.shape)
    order = max(1, min(order, ceiling))

    left_r = left[:, :order]
    right_r = right_t[:order].T
    inverse_sqrt = 1.0 / np.sqrt(singular[:order])
    root = np.sqrt(singular[:order])

    a_mat = (left_r.T @ h1 @ right_r) * (inverse_sqrt[:, None] * inverse_sqrt[None, :])
    b_mat = root[:, None] * right_r.T[:, :inputs]
    c_mat = left_r[:outputs, :] * root[None, :]

    return MacroModel(
        A=a_mat,
        B=b_mat,
        C=c_mat,
        D=observed[0].copy(),
        singular_values=singular,
    )


def markov_from_transient(
    times: np.ndarray,
    source_wave: np.ndarray,
    response_waves: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Deconvolve the launch pulse out of a transient record into Markov parameters.

    Returns ``h`` of shape ``(steps, outputs, 1)`` -- one input, which is the
    driven port, and one column per observed node -- ready for
    :func:`fit_macromodel`.

    Deconvolving is what makes this usable where :func:`block_hankel` on an
    S-matrix is not: the spectrum's DC bin is ``sum(response) / sum(source)``,
    both finite because every port in the deck is terminated, so there is no
    division by the open-circuit singularity a bare line has at DC. The
    response is taken *as measured* rather than scaled into ``S`` -- the
    factor of two belongs to the Thevenin convention, and ``B`` should carry
    volts out per volt in, not a normalisation.

    ``irfft`` reconstructs the negative frequencies by Hermitian symmetry,
    which is exact for a real impulse response, and the record opens and
    closes at zero so the circular wrap lands where the response has already
    decayed.
    """
    times = np.asarray(times, dtype=np.float64)
    source = np.asarray(source_wave, dtype=np.float64)
    responses = np.asarray(response_waves, dtype=np.float64)
    if responses.ndim == 1:
        responses = responses[:, None]
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"times must be a 1-D series of at least 2 samples, got {times.shape}")
    if source.shape != times.shape:
        raise ValueError(f"source_wave {source.shape} does not match times {times.shape}")
    if responses.shape[0] != times.size:
        raise ValueError(
            f"response_waves {responses.shape} has {responses.shape[0]} samples, "
            f"times has {times.size}"
        )

    grid, source_u = resample_uniform(times, source, dt=dt)
    columns = []
    for index in range(responses.shape[1]):
        _, column = resample_uniform(times, responses[:, index], dt=dt)
        columns.append(column)
    response_u = np.stack(columns, axis=1)

    source_spectrum = np.fft.rfft(source_u)
    peak = np.abs(source_spectrum).max()
    if peak <= 0:
        raise ValueError("the source wave is identically zero; there is nothing to divide by")

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.fft.rfft(response_u, axis=0) / source_spectrum[:, None]
    gain[np.abs(source_spectrum) < SOURCE_FLOOR * peak] = 0.0
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)

    h = np.fft.irfft(gain, n=grid.size, axis=0)
    return h[:, :, None]


def save_macromodel(model: MacroModel, path) -> str:
    """Write a fit as ``.npz`` for ``delay_tensor``'s ``topology: extracted``.

    ``A`` and ``B`` are what the reservoir reads; ``C``, ``D`` and the
    singular values travel along so the file can be re-inspected without
    refitting, and ``order`` records what it was fitted at.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        A=model.A,
        B=model.B,
        C=model.C,
        D=model.D,
        singular_values=model.singular_values,
        order=np.array([model.order], dtype=np.int64),
    )
    return str(destination)
