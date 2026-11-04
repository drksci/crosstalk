"""NARMA10 sanity task (deterministic).

y[t+1] = 0.3 y[t] + 0.05 y[t] * sum_{i=0..9} y[t-i] + 1.5 u[t-9] u[t] + 0.1,   u ~ U(0, 0.5)
The target aligned to input u[t] is y[t+1], so it depends only on u[t-9..t] (causal, memory 10).
"""
from __future__ import annotations

import numpy as np

from . import sha256_arrays

SEED = 1234


def narma10_series(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n)
    y = np.zeros(n + 1)
    for t in range(9, n):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    if not np.all(np.isfinite(y)) or np.abs(y).max() > 10:
        raise RuntimeError("NARMA10 diverged; pick another seed")
    return u, y[1:]  # target[t] = y[t+1]


def dataset(n_train: int = 4000, n_test: int = 2000, washout: int = 100, seed: int = SEED) -> dict:
    """Return the dataset dict. x/y have shape [1(batch), 1(channel), T]."""
    u, y = narma10_series(n_train + n_test, seed)
    ds = {
        "name": "narma10", "task": "regression", "washout": washout, "seed": seed,
        "x_train": u[:n_train][None, None], "y_train": y[:n_train][None, None],
        "x_test": u[n_train:][None, None], "y_test": y[n_train:][None, None],
    }
    ds["sha256"] = sha256_arrays({k: ds[k] for k in ("x_train", "y_train", "x_test", "y_test")})
    return ds
