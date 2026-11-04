"""Test helper: a RANDOMLY INITIALISED (not trained) TDL-Net-S in the `dlnn-model/1` schema.

Topology (plan section 2): 2->8 K=8 d=1, 8->8 K=8 d=4, 8->8 K=8 d=16, then a pooled float readout
(ridge fitted on the time-means of ALL three layers' outputs, readout_from=[0,1,2], so the logits are
not pure noise; ~45-50 % top-1 on the modrec test set, i.e. far below a trained net).  Ternary taps are random
{-1,0,+1} with P(0)=1/3; vsat per layer is calibrated to the std of the pre-activation on data so tanh is
neither dead nor hard-saturated.  This is a development stand-in for the M1 trained model; it is NOT trained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dlnn.model.npz import Layer, Model
from dlnn.tiers import virtual

TOPOLOGY = ((2, 8, 8, 1), (8, 8, 8, 4), (8, 8, 8, 16))   # (I, O, K, dilation)


def make_random_tdlnet(x_calib: np.ndarray, y_calib: np.ndarray | None = None, seed: int = 0,
                       n_out: int = 8, per_line_gain: bool = False, input_fullscale: float | None = None) -> Model:
    """Random-weight TDL-Net-S calibrated on x_calib [N,2,T] (and ridge readout if y_calib given)."""
    rng = np.random.default_rng(seed)
    # ADC full scale: 99.99th percentile of |x| (clips rare noise peaks; what a trainer would calibrate)
    fs = float(np.percentile(np.abs(x_calib), 99.99)) if input_fullscale is None else input_fullscale
    layers: list[Layer] = []
    pooled: list[np.ndarray] = []
    h = np.asarray(x_calib, np.float32)
    for l, (I, O, K, d) in enumerate(TOPOLOGY):
        W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, I, K), p=[1 / 3, 1 / 3, 1 / 3])
        gain = np.full(O, 1.0 + 0.1 * l, np.float32)
        if per_line_gain:
            gain = (gain * rng.uniform(0.5, 1.5, O)).astype(np.float32)
        bias = rng.standard_normal(O).astype(np.float32)          # in units of vsat (scaled below)
        ly = Layer(W=W, dilation=d, gain=gain, bias=bias, vsat=1.0)
        u = virtual.layer_preact(h, ly)
        ly.vsat = float(np.std(u)) * 1.2 + 1e-3          # calibrate: ~1 sigma of preact maps to ~0.8 of tanh knee
        ly.bias = (ly.bias * ly.vsat).astype(np.float32)  # bias ~ N(0, vsat^2)
        layers.append(ly)
        h = virtual.layer_forward(h, ly)
        pooled.append(h.mean(axis=2))
    feats = np.concatenate(pooled, axis=1)              # pooled features of all layers [N, 24]
    if y_calib is not None:
        Y = np.eye(n_out, dtype=np.float32)[y_calib] - 1.0 / n_out
        A = np.concatenate([feats, np.ones((len(feats), 1), np.float32)], axis=1)
        sol = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ Y)
        Wr, br = sol[:-1].T.astype(np.float32) * 8.0, (sol[-1] * 8.0).astype(np.float32)
    else:
        Wr, br = rng.standard_normal((n_out, feats.shape[1])).astype(np.float32), np.zeros(n_out, np.float32)
    return Model(kind="tdlnet", layers=layers, readout_W=Wr, readout_b=br, readout_from=list(range(len(layers))),
                 readout_kind="pooled", input_scale=1.0, input_fullscale=fs,
                 meta={"note": "RANDOMLY INITIALISED test helper model, not trained", "seed": seed})


def save_random_tdlnet(path: str | Path, dataset: dict, seed: int = 0, n_calib: int = 8000, **kw) -> Path:
    m = make_random_tdlnet(dataset["x_train"][:n_calib], dataset["y_train"][:n_calib], seed=seed, **kw)
    return m.save(path)
