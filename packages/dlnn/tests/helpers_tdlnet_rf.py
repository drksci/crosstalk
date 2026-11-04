"""Test helper for the RF tier: random (untrained) TDL-Net-S, optionally with a 'benign' (less saturated) operating point.

Uses tests/helpers_tdlnet.py (M2a) when present, else builds the same topology itself. The model is NOT trained; it is a
development stand-in, and agreement numbers measured with it are agreement with that model's own V-tier outputs.
"""
from __future__ import annotations

import numpy as np

from dlnn.model.npz import Layer, Model
from dlnn.tiers import virtual

try:
    from helpers_tdlnet import make_random_tdlnet as _make
except Exception:  # pragma: no cover
    _make = None

TOPOLOGY = ((2, 8, 8, 1), (8, 8, 8, 4), (8, 8, 8, 16))


def _fallback(x, y, seed):
    rng = np.random.default_rng(seed)
    layers, h = [], np.asarray(x, np.float32)
    for l, (I, O, K, d) in enumerate(TOPOLOGY):
        W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, I, K))
        ly = Layer(W=W, dilation=d, gain=np.full(O, 1.0, np.float32), bias=np.zeros(O, np.float32), vsat=1.0)
        ly.vsat = float(np.std(virtual.layer_preact(h, ly))) * 1.2 + 1e-3
        layers.append(ly)
        h = virtual.layer_forward(h, ly)
    m = Model(kind="tdlnet", layers=layers, readout_W=np.zeros((8, 8), np.float32), readout_b=np.zeros(8, np.float32),
              readout_from=[2], readout_kind="pooled", input_scale=1.0, input_fullscale=float(np.abs(x).max()))
    return m


def refit_readout(m: Model, x, y, n_out=8) -> Model:
    feats = virtual.features(m, virtual.forward_layers(m, x))
    Y = np.eye(n_out, dtype=np.float32)[y] - 1.0 / n_out
    A = np.concatenate([feats, np.ones((len(feats), 1), np.float32)], axis=1)
    sol = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Y)
    m.readout_W, m.readout_b = sol[:-1].T.astype(np.float32) * 8.0, (sol[-1] * 8.0).astype(np.float32)
    return m


def random_model(ds: dict, seed: int = 0, vsat_scale: float = 1.0, n_calib: int = 2000) -> Model:
    """vsat_scale > 1 moves the operating point toward the linear part of tanh (less chaotic, 'benign')."""
    x, y = ds["x_train"][:n_calib], ds["y_train"][:n_calib]
    m = _make(x, y, seed=seed) if _make else refit_readout(_fallback(x, y, seed), x, y)
    if vsat_scale != 1.0:
        for ly in m.layers:
            ly.vsat = float(ly.vsat) * vsat_scale
        refit_readout(m, x, y)
        m.meta = {**m.meta, "vsat_scale": vsat_scale}
    return m
