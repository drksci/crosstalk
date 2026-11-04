"""Export a trained TDLNet to the `dlnn-model/1` container (ternary) or to `dlnn-model-float/1` (float baseline).

Ternary: schema unchanged. Mapping per layer (see docs/model_export.md):
    torch  y = g * tanh(alpha*gamma*u + b)   ->   vsat = 1/(alpha*gamma), bias = b/(alpha*gamma), gain = g (ones without line-gain)
    readout_W = gamma_r * W_t,r (* per-output scale), readout_kind = "pooled", readout_from = [n_layers-1].
Float baseline: a separate schema string `dlnn-model-float/1` (so it can never be mistaken for a ternary model)
with the same keys, but `L{l}_W` float32; `load_float_model` returns a `Model` whose W is float, which
`dlnn.tiers.virtual.forward` evaluates unchanged (it only does W.astype(float32)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .npz import Layer, Model
from .tdlnet_torch import TDLNet

FLOAT_SCHEMA = "dlnn-model-float/1"


@torch.no_grad()
def to_model(net: TDLNet, meta: dict | None = None) -> Model:
    """Ternary net -> Model (int8 taps). Float nets -> Model with float W (not valid for dlnn-model/1 files)."""
    layers = []
    for c in net.convs:
        gain = (c.g.detach().cpu().numpy() if c.g is not None else np.ones(c.c_out)).astype(np.float32)
        if net.float_mode:
            W, vsat, bias = c.W.detach().cpu().numpy().astype(np.float32), 1.0, c.b.detach().cpu().numpy()
        else:
            _, Wt, gamma = c.weights()
            gamma = float(gamma) * float(c.alpha())   # amplifier gain alpha folded in: pre = (alpha*gamma)*u + b
            W = Wt.cpu().numpy().astype(np.int8)
            vsat, bias = 1.0 / gamma, c.b.detach().cpu().numpy() / gamma
        layers.append(Layer(W=W, dilation=c.d, gain=gain, bias=np.asarray(bias, np.float32), vsat=float(vsat)))
    R = net.readout_weights().detach().cpu().numpy().astype(np.float32)
    m = dict(meta or {})
    m["arch"] = net.config()
    return Model(kind="tdlnet", layers=layers, readout_W=R, readout_b=net.rb.detach().cpu().numpy().astype(np.float32),
                 readout_from=[len(layers) - 1], readout_kind="pooled", input_scale=net.input_scale,
                 input_fullscale=net.input_fullscale * net.input_scale, meta=m)


def save_ternary(net: TDLNet, path: str | Path, meta: dict | None = None) -> Model:
    if net.float_mode:
        raise ValueError("float net cannot be exported as dlnn-model/1; use save_float")
    m = to_model(net, meta)
    m.save(path)
    return m


def save_float(net: TDLNet, path: str | Path, meta: dict | None = None) -> Model:
    m = to_model(net, dict(meta or {}, float_baseline=True))
    a = m.to_arrays()  # int8 cast would destroy float taps -> overwrite
    a["schema"] = np.array(FLOAT_SCHEMA)
    for l, ly in enumerate(m.layers):
        a[f"L{l}_W"] = ly.W.astype(np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **a)
    return m


def load_float_model(path: str | Path) -> Model:
    with np.load(path, allow_pickle=False) as z:
        if str(z["schema"]) != FLOAT_SCHEMA:
            raise ValueError(f"not a float baseline file: {z['schema']}")
        layers = [Layer(W=z[f"L{l}_W"].astype(np.float32), dilation=int(z[f"L{l}_dilation"]),
                        gain=z[f"L{l}_gain"], bias=z[f"L{l}_bias"], vsat=float(z[f"L{l}_vsat"]))
                  for l in range(int(z["n_layers"]))]
        fs = float(z["input_fullscale"])
        return Model(kind=str(z["kind"]), layers=layers, readout_W=z["readout_W"], readout_b=z["readout_b"],
                     readout_from=[int(i) for i in z["readout_from"]], readout_kind=str(z["readout_kind"]),
                     input_scale=float(z["input_scale"]), input_fullscale=None if np.isnan(fs) else fs,
                     meta=json.loads(str(z["meta_json"])))
