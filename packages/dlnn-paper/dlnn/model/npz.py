"""Model container + `model.npz` schema "dlnn-model/1" (shared by the reservoir and the M1 TDL-Net).

Layer l (0-based) is stored as keys ``L{l}_W`` (int8 [O,I,K] in {-1,0,1}), ``L{l}_dilation`` (int),
``L{l}_gain`` (float32 [O]), ``L{l}_bias`` (float32 [O]), ``L{l}_vsat`` (float32 scalar):
    u[o,n] = sum_{i,k} W[o,i,k] * x[i, n - k*dilation]     (zeros before n=0)
    y[o,n] = gain[o] * tanh((u[o,n] + bias[o]) / vsat)
Readout: features = concat over layers in ``readout_from`` of (per-step y)  [readout_kind="per_step"]
or of time-mean of y [readout_kind="pooled"];  out = readout_W @ features + readout_b.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "dlnn-model/1"


@dataclass
class Layer:
    W: np.ndarray            # int8 [O, I, K]
    dilation: int
    gain: np.ndarray         # float32 [O]
    bias: np.ndarray         # float32 [O]
    vsat: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.W.shape  # type: ignore[return-value]


@dataclass
class Model:
    kind: str                       # "reservoir" | "tdlnet"
    layers: list[Layer]
    readout_W: np.ndarray           # float32 [n_out, n_feat]
    readout_b: np.ndarray           # float32 [n_out]
    readout_from: list[int]         # layer indices whose outputs form the features
    readout_kind: str = "per_step"  # "per_step" | "pooled"
    input_scale: float = 1.0        # multiplies the raw input (volts-equivalent)
    input_fullscale: float | None = None  # ADC full scale (after input_scale); None -> from data
    meta: dict = field(default_factory=dict)

    def to_arrays(self) -> dict[str, np.ndarray]:
        a: dict[str, np.ndarray] = {
            "schema": np.array(SCHEMA), "kind": np.array(self.kind),
            "n_layers": np.array(len(self.layers)),
            "readout_W": self.readout_W.astype(np.float32), "readout_b": self.readout_b.astype(np.float32),
            "readout_from": np.array(self.readout_from, dtype=np.int64),
            "readout_kind": np.array(self.readout_kind),
            "input_scale": np.array(self.input_scale, dtype=np.float32),
            "input_fullscale": np.array(np.nan if self.input_fullscale is None else self.input_fullscale,
                                        dtype=np.float32),
            "meta_json": np.array(json.dumps(self.meta, sort_keys=True)),
        }
        for l, ly in enumerate(self.layers):
            a[f"L{l}_W"] = ly.W.astype(np.int8)
            a[f"L{l}_dilation"] = np.array(ly.dilation, dtype=np.int64)
            a[f"L{l}_gain"] = np.broadcast_to(ly.gain, (ly.W.shape[0],)).astype(np.float32).copy()
            a[f"L{l}_bias"] = np.broadcast_to(ly.bias, (ly.W.shape[0],)).astype(np.float32).copy()
            a[f"L{l}_vsat"] = np.array(ly.vsat, dtype=np.float32)
        return a

    def sha256(self) -> str:
        """Canonical hash over arrays (independent of zip timestamps)."""
        h = hashlib.sha256()
        for k, v in sorted(self.to_arrays().items()):
            v = np.ascontiguousarray(v)
            h.update(f"{k}|{v.dtype}|{v.shape}|".encode())
            h.update(v.tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.to_arrays())
        return path

    @property
    def n_ternary_weights(self) -> int:
        return int(sum(l.W.size for l in self.layers))


def load_model(path: str | Path) -> Model:
    with np.load(path, allow_pickle=False) as z:
        if str(z["schema"]) != SCHEMA:
            raise ValueError(f"unsupported model schema {z['schema']}")
        layers = [Layer(W=z[f"L{l}_W"].astype(np.int8), dilation=int(z[f"L{l}_dilation"]),
                        gain=z[f"L{l}_gain"].astype(np.float32), bias=z[f"L{l}_bias"].astype(np.float32),
                        vsat=float(z[f"L{l}_vsat"])) for l in range(int(z["n_layers"]))]
        fs = float(z["input_fullscale"])
        return Model(kind=str(z["kind"]), layers=layers, readout_W=z["readout_W"], readout_b=z["readout_b"],
                     readout_from=[int(i) for i in z["readout_from"]], readout_kind=str(z["readout_kind"]),
                     input_scale=float(z["input_scale"]), input_fullscale=None if np.isnan(fs) else fs,
                     meta=json.loads(str(z["meta_json"])))
