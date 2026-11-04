"""Minimal sweep helpers that any tier can call.

A sweep is one axis (see contract.SWEEP_AXES), a list of values, and a function
``forward_at(x, value) -> tier output`` plus a ``score(out) -> {accuracy|nmse|mean_kl: float}``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise so that per-example power(x)/power(noise) = snr_db.
    Power is the mean of x^2 over all axes except the batch axis."""
    x = np.asarray(x, dtype=np.float32)
    p = np.mean(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)
    sigma = np.sqrt(p / (10.0 ** (snr_db / 10.0)))
    return x + (sigma * rng.standard_normal(x.shape)).astype(np.float32)


def run_sweep(axis: str, values: list[float], forward_at: Callable, x: np.ndarray,
              score: Callable[[dict], dict[str, float]]) -> dict:
    """Evaluate the tier at each axis value. Returns a contract ``sweeps[]`` entry."""
    rows = [score(forward_at(x, v)) for v in values]
    entry: dict = {"axis": axis, "values": [float(v) for v in values]}
    for key in ("accuracy", "mean_kl", "nmse"):
        if rows and key in rows[0]:
            entry[key] = [float(r[key]) for r in rows]
    return entry


def snr_sweep(forward: Callable, x: np.ndarray, score: Callable, snrs: list[float], seed: int = 0) -> dict:
    """Input-AWGN sweep (axis ``snr_db``). ``forward(x_noisy) -> tier output``.
    Noise is re-seeded per value so results are deterministic."""
    def fwd(xx: np.ndarray, snr: float) -> dict:
        rng = np.random.default_rng(seed + int(round(snr * 10)))
        return forward(add_awgn(xx, snr, rng))
    return run_sweep("snr_db", snrs, fwd, x, score)
