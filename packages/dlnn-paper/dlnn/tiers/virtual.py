"""Tier V (virtual): ideal unit delays, exact ternary sums, exact tanh, float32 numpy.

`layer_forward` / `forward_layers` are generic (any O, I, K, dilation) and are reused by the
reservoir fit and (M1) the TDL-Net. `run()` is the tier entry point of the common contract.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import contract
from ..model.npz import Layer, Model, load_model
from . import common

TOOL = {"name": "numpy-float32", "version": np.__version__}


def layer_preact(x: np.ndarray, ly: Layer) -> np.ndarray:
    """Tapped-delay sum u[b,o,n] = sum_{i,k} W[o,i,k] x[b,i,n-k*d] (no bias). x [B,I,T] -> [B,O,T]."""
    B, I, T = x.shape
    O, _, K = ly.W.shape
    u = np.zeros((B, O, T), dtype=np.float32)
    for k in range(K):
        s = k * ly.dilation
        if s >= T:
            break
        Wk = ly.W[:, :, k].astype(np.float32)
        if not Wk.any():
            continue
        u[:, :, s:] += np.einsum("oi,bit->bot", Wk, x[:, :, :T - s], optimize=True)
    return u


def layer_forward(x: np.ndarray, ly: Layer) -> np.ndarray:
    """x [B,I,T] float32 -> y [B,O,T] = gain * tanh((u + bias) / vsat). Causal (zeros before n=0)."""
    u = layer_preact(x, ly)
    return ly.gain[None, :, None] * np.tanh((u + ly.bias[None, :, None]) / np.float32(ly.vsat))


def forward_layers(model: Model, x: np.ndarray) -> list[np.ndarray]:
    """Return per-layer outputs [B,O_l,T] (float32) for raw input x [B,C,T]."""
    h = (np.asarray(x, dtype=np.float32) * np.float32(model.input_scale))
    outs = []
    for ly in model.layers:
        h = layer_forward(h, ly)
        outs.append(h)
    return outs


def features(model: Model, layer_outs: list[np.ndarray]) -> np.ndarray:
    """Readout features. per_step -> [B,F,T]; pooled -> [B,F]."""
    f = np.concatenate([layer_outs[i] for i in model.readout_from], axis=1)
    return f.mean(axis=2) if model.readout_kind == "pooled" else f


def readout(model: Model, feats: np.ndarray) -> np.ndarray:
    """per_step: [B,F,T] -> [B,n_out,T];  pooled: [B,F] -> [B,n_out]."""
    if model.readout_kind == "pooled":
        return feats @ model.readout_W.T + model.readout_b
    return np.einsum("of,bft->bot", model.readout_W, feats) + model.readout_b[None, :, None]


def forward(model: Model, x: np.ndarray, cfg: dict | None = None) -> dict:
    """Raw tier outputs: {'out': readout output, 'layers': [..]} for input x [B,C,T]."""
    cfg = cfg or {}
    layers = forward_layers(model, x)
    return {"out": readout(model, features(model, layers)), "layers": layers}


def run(model_npz: str | Path, dataset: dict, cfg: dict, outdir: str | Path) -> contract.TierResult:
    t0 = time.time()
    model = load_model(model_npz)
    res = common.evaluate_tier(
        tier="V", status="ok", tool=TOOL, model=model, dataset=dataset, cfg=cfg,
        forward_fn=lambda x, c: forward(model, x, c), ref_fn=None,
        fidelity={"models": ["ideal unit delays", "exact ternary sums", "exact tanh", "float32 arithmetic"],
                  "omits": ["every physical effect: loss, dispersion, noise, quantisation, saturation limits"]},
        sweep_axes=("snr_db",), t0=t0)
    return res
