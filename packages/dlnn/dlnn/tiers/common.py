"""Shared tier plumbing: metric computation, vs-virtual comparison, TierResult assembly."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .. import contract, sweeps
from ..model.npz import Model

DEFAULT_SNRS = {"regression": [10.0, 20.0, 30.0, 40.0], "classification": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def macro_f1(y: np.ndarray, p: np.ndarray, ncls: int) -> float:
    f = []
    for c in range(ncls):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def mean_kl(p_ref: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    return float(np.mean(np.sum(p_ref * (np.log(p_ref + eps) - np.log(p + eps)), axis=-1)))


def _sl(ds: dict) -> slice:
    return slice(int(ds.get("washout", 0)), None)


def score(out: np.ndarray, ref: np.ndarray, ds: dict) -> dict[str, float]:
    """Sweep score of tier output `out` against dataset targets and virtual reference `ref`."""
    if ds["task"] == "regression":
        yt, yp = ds["y_test"][0, :, _sl(ds)], out[0][:, _sl(ds)]
        return {"nmse": float(np.mean((yp - yt) ** 2) / np.var(yt))}
    return {"accuracy": float(np.mean(out.argmax(1) == ds["y_test"])),
            "mean_kl": mean_kl(softmax(ref), softmax(out))}


def metrics_and_arrays(out: np.ndarray, ref: np.ndarray, ds: dict) -> tuple[dict, dict, int]:
    """Metrics (contract METRIC_IDS), arrays, n_examples for one tier output."""
    if ds["task"] == "regression":
        s = _sl(ds)
        yt, yp, yr = ds["y_test"][0, :, s], out[0][:, s], ref[0][:, s]
        m = {"nmse": float(np.mean((yp - yt) ** 2) / np.var(yt)),
             "nmse_vs_virtual": float(np.mean((yp - yr) ** 2) / np.var(yr))}
        return m, {"y_pred": out[0].astype(np.float32), "y_true": ds["y_test"][0].astype(np.float32)}, int(yt.shape[1])
    y, ncls = ds["y_test"], out.shape[1]
    pred, pref = out.argmax(1), ref.argmax(1)
    m = {"accuracy": float(np.mean(pred == y)), "macro_f1": macro_f1(y, pred, ncls),
         "top1_agreement_vs_virtual": float(np.mean(pred == pref)),
         "mean_kl_virtual_vs_tier": mean_kl(softmax(ref), softmax(out)),
         "logit_nmse_vs_virtual": float(np.sum((out - ref) ** 2) / np.sum(ref ** 2))}
    if "snr_test" in ds:
        m["acc_per_snr"] = {str(int(s)): float(np.mean(pred[ds["snr_test"] == s] == y[ds["snr_test"] == s]))
                            for s in np.unique(ds["snr_test"])}
    return m, {"logits": out.astype(np.float32), "probs": softmax(out).astype(np.float32),
               "pred": pred.astype(np.int64)}, int(len(y))


def evaluate_tier(*, tier: str, status: str, tool: dict, model: Model, dataset: dict, cfg: dict,
                  forward_fn: Callable[[np.ndarray, dict], dict], ref_fn: Callable | None,
                  fidelity: dict, sweep_axes: tuple = ("snr_db",), t0: float | None = None,
                  inputs_from: list[str] | None = None, warnings: list[str] | None = None,
                  extra_sweeps: Callable[[np.ndarray, Callable], list[dict]] | None = None) -> contract.TierResult:
    """Run `forward_fn` on the test set, compare against virtual, run sweeps, build the TierResult.

    forward_fn(x, cfg) -> {'out': readout output, 'layers': [per-layer float outputs]}
    ref_fn(x, cfg) -> same for the virtual tier (None = this IS the virtual tier).
    extra_sweeps(x, score_fn) -> additional contract sweep entries (e.g. adc_bits for E).
    """
    from . import virtual  # lazy: virtual imports this module
    t0 = time.time() if t0 is None else t0
    x = dataset["x_test"]
    res = forward_fn(x, cfg)
    ref = res if ref_fn is None else ref_fn(x, cfg)
    if ref_fn is None and tier != "V":
        ref = virtual.forward(model, x, cfg)
    out, refo = res["out"], ref["out"]
    m, arrays, n = metrics_and_arrays(out, refo, dataset)

    n_tr = int(cfg.get("n_trace", 16))
    for l, h in enumerate(res["layers"]):
        arrays[f"trace_activations.L{l}"] = (h[:n_tr] if dataset["task"] == "classification"
                                             else h[:, :, :500]).astype(np.float32)

    sc = lambda o: score(o["out"], refo, dataset)  # noqa: E731
    snrs = cfg.get("sweep_snrs", DEFAULT_SNRS[dataset["task"]])
    sw: list[dict] = []
    if "snr_db" in sweep_axes:
        sw.append(sweeps.snr_sweep(lambda xn: forward_fn(xn, cfg), x, sc, snrs, seed=int(cfg.get("seed", 0))))
    if extra_sweeps is not None:
        sw += extra_sweeps(x, sc)

    return contract.TierResult(
        tier=tier, status=status, run_id=cfg.get("run_id", "adhoc"), task=dataset["name"], tool=tool,
        model_sha256=model.sha256(), dataset_sha256=dataset["sha256"], n_examples=n, metrics=m,
        sweeps=sw, fidelity=fidelity, inputs_from=inputs_from or [], runtime_s=round(time.time() - t0, 4),
        warnings=warnings or [], arrays=arrays)


def stub_run(*, tier: str, tool_name: str, snr_db: float, loss_db: float, model_npz, dataset: dict,
             cfg: dict, what: str) -> contract.TierResult:
    """Shared body of the C and R STUBS: virtual forward on an input with AWGN and a flat loss.

    This is NOT a circuit or RF simulation; status is always "stub" and fidelity says so.
    """
    from . import virtual
    from ..model.npz import load_model
    t0 = time.time()
    model = load_model(model_npz)
    seed = int(cfg.get("seed", 0)) + (101 if tier == "C" else 202)
    atten = np.float32(10.0 ** (-loss_db / 20.0))

    def fwd(x: np.ndarray, c: dict) -> dict:
        rng = np.random.default_rng(seed)
        xn = sweeps.add_awgn(np.asarray(x, np.float32) * atten, snr_db, rng)
        return virtual.forward(model, xn, c)

    return evaluate_tier(
        tier=tier, status="stub", tool={"name": tool_name, "version": "stub-0"}, model=model,
        dataset=dataset, cfg=cfg, forward_fn=fwd, ref_fn=None,
        fidelity={"models": [f"V forward on input with AWGN ({snr_db:g} dB SNR) and {loss_db:g} dB flat loss"],
                  "omits": [f"STUB / PLACEHOLDER: this is the virtual tier plus noise, NOT {what}",
                            "no transmission-line, coupler, amplifier or parasitic model of any kind"]},
        sweep_axes=("snr_db",), t0=t0,
        inputs_from=[], warnings=[f"{tier} tier is a stub (V + AWGN/loss); not a {what}"])
