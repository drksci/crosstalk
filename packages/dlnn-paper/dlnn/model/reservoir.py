"""Random ternary tapped-delay-line reservoir + ridge readout (numpy only; M0 walking-skeleton model).

Feed-forward stack of generic ternary TDL layers (random W in {-1,0,+1}, fixed), per-layer vsat
calibrated so that std(u/vsat) ~= `drive` on the training input, random biases; a linear ridge
readout over the concatenated per-step layer outputs is the only trained part. Exports the common
`model.npz` (see dlnn.model.npz).
"""
from __future__ import annotations

import numpy as np

from ..tiers import virtual
from .npz import Layer, Model

DEFAULT_CFG = dict(widths=(1, 48, 48), taps=8, dilations=(1, 2), density=0.6, drive=0.5,
                   input_scale=4.0, bias_range=0.5, seed=7, washout=100)


def random_ternary(shape: tuple, density: float, rng: np.random.Generator) -> np.ndarray:
    nz = rng.random(shape) < density
    return (nz * rng.choice(np.array([-1, 1], dtype=np.int8), shape)).astype(np.int8)


def ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """X [n,F], Y [n,O] -> W [F,O] (X is assumed to carry a bias column)."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def build_layers(x_train: np.ndarray, cfg: dict) -> list[Layer]:
    rng = np.random.default_rng(cfg["seed"])
    w, layers = cfg["widths"], []
    h = (x_train.astype(np.float32) * np.float32(cfg["input_scale"]))
    for l, d in enumerate(cfg["dilations"]):
        W = random_ternary((w[l + 1], w[l], cfg["taps"]), cfg["density"], rng)
        bias = rng.uniform(-cfg["bias_range"], cfg["bias_range"], w[l + 1]).astype(np.float32)
        ly = Layer(W=W, dilation=int(d), gain=np.ones(w[l + 1], np.float32), bias=bias, vsat=1.0)
        u = virtual.layer_preact(h, ly) + bias[None, :, None]
        ly.vsat = float(np.std(u) / cfg["drive"])
        layers.append(ly)
        h = virtual.layer_forward(h, ly)
    return layers


def fit(dataset: dict, cfg: dict | None = None) -> Model:
    """Fit on dataset['x_train'], dataset['y_train'] ([1,1,T] each). Deterministic."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    cfg["widths"] = tuple(cfg["widths"])
    xtr, ytr = dataset["x_train"], dataset["y_train"]
    layers = build_layers(xtr, cfg)
    model = Model(kind="reservoir", layers=layers, readout_W=np.zeros((1, 1), np.float32),
                  readout_b=np.zeros(1, np.float32), readout_from=list(range(len(layers))),
                  readout_kind="per_step", input_scale=float(cfg["input_scale"]), meta={"cfg": _plain(cfg)})
    F = virtual.features(model, virtual.forward_layers(model, xtr))[0].T.astype(np.float64)  # [T,F]
    Y = ytr[0].T.astype(np.float64)                                                           # [T,1]
    wo = cfg["washout"]
    F, Y = F[wo:], Y[wo:]
    Xb = np.hstack([F, np.ones((len(F), 1))])
    ntr = int(0.8 * len(Xb))
    best, best_lam = np.inf, 1e-6
    for lam in 10.0 ** np.arange(-8, 3):          # lambda chosen on a held-out tail of the train set
        W = ridge(Xb[:ntr], Y[:ntr], lam)
        err = np.mean((Xb[ntr:] @ W - Y[ntr:]) ** 2)
        if err < best:
            best, best_lam = err, lam
    W = ridge(Xb, Y, best_lam)
    model.readout_W = W[:-1].T.astype(np.float32)
    model.readout_b = W[-1].astype(np.float32)
    # ADC full-scale for the emulated tier: calibrated on training input (after input_scale)
    model.input_fullscale = float(np.abs(xtr).max() * cfg["input_scale"])
    model.meta.update(ridge_lambda=float(best_lam), heldout_mse=float(best))
    return model


def _plain(cfg: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
