"""Tier E (emulated): digital shift-register implementation = numpy fixed-point golden model + Verilog RTL.

Spec (bit-true; `hw/rtl/tdl_layer.v` matches it exactly, see also docs/fixed_point_spec.md):
  input    xq = clip(round(x*input_scale / s0), -L, L) int,  s0 = fullscale/L,  L = 2^(b-1)-1
           (b = ADC bits, default 8 -> L = 127, int8); round = half away from zero
           (np.rint is half-even, so we use floor(v+0.5) for v>=0 and -floor(-v+0.5) for v<0).
  weights  int2 ternary {-1,0,+1} (stored int8)
  acc      int32 = sum_{i,k} W[o,i,k] * xq[i, n-k*d] (zeros before n=0) + bq[o],
           bq = round(bias/s_in) with s_in the real value of one input LSB
  lut idx  z = clip((acc * M + 2^14) >> 15, -128, 127) int8, M = round(2^15 * s_in/vsat * 128/LUT_RANGE)
           (fixed-point multiplier with 15 fractional bits, int64 intermediate; arithmetic shift)
           z indexes a 256-entry LUT: lut[z+128] = round(127*tanh(z*LUT_RANGE/128)) int8  (LUT_RANGE=4)
  out      yq = (lut * R[o] + 2^14) >> 15 with R[o] = round(2^15 * gain[o]/max|gain|) (signed Q.15;
           R = 2^15 is the identity, so equal positive gains cost nothing); real value = yq * s_out,
           s_out = max|gain|/127
  pool     (pooled readout) integer sum over time, divided by T when converting to float
  readout  float32 digital post-processing on dequantised features (as in the plan: ADC + software)

What is RTL and what is numpy (see `run()`): the ADC quantiser, the pooled-sum -> float conversion and the
float readout are numpy.  Layers (delay lines, ternary MAC, scale/clip, LUT, per-line scale, pool sum)
exist as numpy golden model AND as Verilog (`tdl_layer.v`, simulated with iverilog).  The full test set
(and every sweep) runs on the numpy golden model; the RTL is run on a subset and compared bit for bit.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import contract, sweeps
from ..model.npz import Layer, Model, load_model
from . import common, virtual

TOOL = {"name": "numpy-fixedpoint", "version": np.__version__}
LUT_RANGE = 4.0
FRAC = 15
RTL_DIR = Path(__file__).resolve().parents[2] / "hw" / "rtl"
ADC_BITS = (4, 5, 6, 7, 8, 9, 10)
RTL_MAC_BUDGET = 4.5e6   # max (examples * T * O*I*K) MACs per layer sent to the RTL simulator


def rnd(v: np.ndarray) -> np.ndarray:
    """Round half away from zero (matches a simple RTL rounder)."""
    return np.where(v >= 0, np.floor(v + 0.5), -np.floor(-v + 0.5))


def make_lut() -> np.ndarray:
    z = np.arange(-128, 128)
    return rnd(127 * np.tanh(z * LUT_RANGE / 128)).astype(np.int8)


LUT = make_lut()


def quantize_input(x: np.ndarray, fullscale: float, bits: int = 8) -> tuple[np.ndarray, float]:
    """Absmax ADC. Returns (int codes as int32, real value of one LSB)."""
    levels = 2 ** (bits - 1) - 1
    s0 = fullscale / levels
    return np.clip(rnd(x / s0), -levels, levels).astype(np.int32), s0


def layer_params(ly: Layer, s_in: float) -> dict:
    """Integer constants of one layer for input LSB `s_in`: bq [O] int32, M int, R [O] int64 (Q.15), s_out."""
    bq = rnd(ly.bias / s_in).astype(np.int32)
    M = int(rnd(np.float64(2 ** FRAC) * s_in / ly.vsat * 128 / LUT_RANGE))
    gmax = float(np.max(np.abs(ly.gain))) or 1.0
    R = rnd(ly.gain.astype(np.float64) / gmax * 2 ** FRAC).astype(np.int64)
    return {"bq": bq, "M": M, "R": R, "s_out": gmax / 127.0}


def layer_fixed(xq: np.ndarray, s_in: float, ly: Layer) -> tuple[np.ndarray, float]:
    """xq int [B,I,T] with LSB s_in -> (yq int8-range [B,O,T], s_out)."""
    B, I, T = xq.shape
    O, _, K = ly.W.shape
    pr = layer_params(ly, s_in)
    acc = np.zeros((B, O, T), dtype=np.int32)
    for k in range(K):
        s = k * ly.dilation
        if s >= T:
            break
        Wk = ly.W[:, :, k].astype(np.int32)
        if Wk.any():
            acc[:, :, s:] += np.einsum("oi,bit->bot", Wk, xq[:, :, :T - s], optimize=True).astype(np.int32)
    acc += pr["bq"][None, :, None]
    z = np.clip((acc.astype(np.int64) * pr["M"] + (1 << (FRAC - 1))) >> FRAC, -128, 127)
    yq = LUT[z + 128].astype(np.int64)
    yq = ((yq * pr["R"][None, :, None] + (1 << (FRAC - 1))) >> FRAC).astype(np.int32)
    return yq, pr["s_out"]


def adc_codes(model: Model, x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    """ADC codes of the raw input batch x (full scale from the model, else abs-max of this batch)."""
    xs = np.asarray(x, dtype=np.float64) * model.input_scale
    fs = model.input_fullscale if model.input_fullscale is not None else float(np.abs(xs).max())
    return quantize_input(xs, fs, bits)


def forward_int(model: Model, x: np.ndarray, adc_bits: int = 8) -> dict:
    """Integer golden chain: {'xq': ADC codes, 's0', 'yq': [per-layer int arrays], 's': [per-layer LSB]}."""
    xq, s0 = adc_codes(model, x, adc_bits)
    yq, scales, s = [], [], s0
    h = xq
    for ly in model.layers:
        h, s = layer_fixed(h, s, ly)
        yq.append(h)
        scales.append(s)
    return {"xq": xq, "s0": s0, "yq": yq, "s": scales}


def readout_from_int(model: Model, yq: list[np.ndarray], scales: list[float],
                     pools: list[np.ndarray] | None = None) -> np.ndarray:
    """Dequantise + float readout. `pools[l]` (int [B,O]) may replace the time-sum of yq[l]."""
    yq = [np.ascontiguousarray(q) for q in yq]   # same memory layout -> bit-identical float summation order
    outs = [(q * s).astype(np.float32) for q, s in zip(yq, scales)]
    if model.readout_kind == "pooled":
        T = yq[0].shape[2]
        f = np.concatenate([(pools[i] if pools is not None else yq[i].sum(axis=2)) * scales[i] / T
                            for i in model.readout_from], axis=1)
    else:
        f = np.concatenate([outs[i] for i in model.readout_from], axis=1)
    return virtual.readout(model, f.astype(np.float32))


def forward(model: Model, x: np.ndarray, cfg: dict | None = None, adc_bits: int = 8) -> dict:
    """Fixed-point forward pass. Returns float outputs (same keys as virtual.forward) + int traces."""
    r = forward_int(model, x, adc_bits)
    outs = [(q * s).astype(np.float32) for q, s in zip(r["yq"], r["s"])]
    return {"out": readout_from_int(model, r["yq"], r["s"]), "layers": outs}


# --------------------------------------------------------------------------------------------------
# Verilog RTL path (iverilog)
# --------------------------------------------------------------------------------------------------
class RtlError(RuntimeError):
    pass


def rtl_info() -> dict:
    """{'available': bool, 'version': str, 'reason': str}"""
    iv, vv = shutil.which("iverilog"), shutil.which("vvp")
    if not iv or not vv:
        return {"available": False, "version": "", "reason": "iverilog/vvp not found on PATH"}
    try:
        out = subprocess.run([iv, "-V"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Icarus Verilog version (\S+)", out)
        return {"available": True, "version": m.group(1) if m else "unknown", "reason": ""}
    except Exception as e:  # pragma: no cover
        return {"available": False, "version": "", "reason": f"iverilog -V failed: {e}"}


def _hex(v: int, bits: int) -> str:
    return format(int(v) & ((1 << bits) - 1), "0{}x".format((bits + 3) // 4))


def run_layer_rtl(xq: np.ndarray, ly: Layer, params: dict, in_w: int, workdir: Path,
                  tag: str = "L", timeout: float = 900.0) -> tuple[np.ndarray, np.ndarray]:
    """Run `hw/rtl/tdl_layer.v` on integer input codes xq [N,I,T] (|code| < 2^(in_w-1)).

    Returns (y [N,O,T] int, pool [N,O] int).  Raises RtlError on any tool failure."""
    N, I, T = xq.shape
    O, I2, K = ly.W.shape
    if I2 != I:
        raise RtlError(f"input lines {I} != layer inputs {I2}")
    lim = 1 << (in_w - 1)
    if np.abs(xq).max(initial=0) >= lim:
        raise RtlError(f"input code exceeds {in_w}-bit signed range")
    if not (-(1 << 31) <= params["M"] < (1 << 31)):
        raise RtlError("LUT multiplier M does not fit int32")
    wd = Path(workdir) / tag
    wd.mkdir(parents=True, exist_ok=True)
    # --- vectors ---
    xw = I * in_w
    q = xq.transpose(0, 2, 1).reshape(N * T, I)           # [(n,t), i]
    lines = []
    for row in q:
        word = 0
        for i, v in enumerate(row):
            word |= (int(v) & ((1 << in_w) - 1)) << (i * in_w)
        lines.append(_hex(word, xw))
    (wd / "x.hex").write_text("\n".join(lines) + "\n")
    wcode = {0: 0, 1: 1, -1: 3}
    (wd / "w.hex").write_text("\n".join(format(wcode[int(v)], "x") for v in ly.W.reshape(-1)) + "\n")
    (wd / "bq.hex").write_text("\n".join(_hex(v, 32) for v in params["bq"]) + "\n")
    (wd / "r.hex").write_text("\n".join(_hex(v, 32) for v in params["R"]) + "\n")
    (wd / "lut.hex").write_text("\n".join(_hex(v, 8) for v in LUT) + "\n")
    # --- compile + run ---
    sim = wd / "sim.vvp"
    pfx = "-Ptb_tdl_layer."
    cmd = ["iverilog", "-g2012", "-o", str(sim),
           f"{pfx}I={I}", f"{pfx}O={O}", f"{pfx}K={K}", f"{pfx}D={ly.dilation}", f"{pfx}IN_W={in_w}",
           f"{pfx}NEX={N}", f"{pfx}T={T}", str(RTL_DIR / "tdl_layer.v"), str(RTL_DIR / "tb_tdl_layer.v")]
    try:
        c = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if c.returncode != 0:
            raise RtlError(f"iverilog compile failed: {c.stderr.strip()[:500]}")
        run = ["vvp", "-n", str(sim), f"+x={wd/'x.hex'}", f"+w={wd/'w.hex'}", f"+bq={wd/'bq.hex'}",
               f"+r={wd/'r.hex'}", f"+lut={wd/'lut.hex'}", f"+M={params['M']}",
               f"+out={wd/'out.txt'}", f"+pool={wd/'pool.txt'}"]
        r = subprocess.run(run, capture_output=True, text=True, timeout=timeout, cwd=wd)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RtlError(f"{type(e).__name__}: {e}") from e
    if r.returncode != 0 or f"TB_DONE samples={N * T} examples={N}" not in r.stdout:
        raise RtlError(f"vvp failed or incomplete (rc={r.returncode}): {(r.stdout + r.stderr).strip()[-500:]}")
    y = np.loadtxt(wd / "out.txt", dtype=np.int64, ndmin=2).reshape(N, T, O).transpose(0, 2, 1)
    pool = np.loadtxt(wd / "pool.txt", dtype=np.int64, ndmin=2).reshape(N, O)
    return y.astype(np.int32), pool


def verify_chain(model: Model, xq0: np.ndarray, s0: float, workdir: Path, in_w: int = 8) -> dict:
    """Run every layer of `model` in RTL, each fed with the RTL output of the previous layer, and compare
    with the numpy golden chain.  xq0 [N,I,T] ADC codes.  Returns a stats dict (+ 'rtl_yq', 'rtl_pool')."""
    h_gold, s = xq0, s0
    h_rtl = xq0
    layers, ys, pools, scales = [], [], [], []
    for l, ly in enumerate(model.layers):
        pr = layer_params(ly, s)
        g, s_out = layer_fixed(h_gold, s, ly)
        y, pool = run_layer_rtl(h_rtl, ly, pr, in_w if l == 0 else 8, workdir, tag=f"L{l}")
        mism = int(np.sum(y != g))
        pmism = int(np.sum(pool != g.sum(axis=2)))
        layers.append({"layer": l + 1, "shape": list(g.shape), "values_compared": int(g.size),
                       "mismatches": mism, "pool_compared": int(pool.size), "pool_mismatches": pmism})
        ys.append(y); pools.append(pool); scales.append(s_out)
        h_gold, h_rtl, s = g, y, s_out
    return {"layers": layers, "rtl_yq": ys, "rtl_pool": pools, "scales": scales,
            "values_compared": sum(d["values_compared"] + d["pool_compared"] for d in layers),
            "mismatches": sum(d["mismatches"] + d["pool_mismatches"] for d in layers)}


def rtl_crosscheck(model: Model, x: np.ndarray, cfg: dict, adc_bits: int = 8,
                   workdir: str | Path | None = None) -> dict:
    """RTL-vs-golden check on the first `rtl_examples` of x.  ADC quantisation uses the full-set fullscale
    (exactly what the golden full-set run uses).  Never raises: errors are reported in the dict."""
    info = rtl_info()
    res: dict = {"ran": False, "rtl_tool": f"iverilog {info['version']}" if info["available"] else "none",
                 "error": info["reason"]}
    if not info["available"]:
        return res
    try:
        n = int(min(cfg.get("rtl_examples", 64), x.shape[0]))
        full = forward_int(model, x, adc_bits)
        T = x.shape[2]
        macs = max(int(ly.W.size) for ly in model.layers)
        t_use = int(min(T, max(32, RTL_MAC_BUDGET // (n * macs))))
        xq0 = full["xq"][:n, :, :t_use]
        with tempfile.TemporaryDirectory(prefix="dlnn_rtl_") as td:
            wd = Path(workdir) if workdir else Path(td)
            wd.mkdir(parents=True, exist_ok=True)
            st = verify_chain(model, xq0, full["s0"], wd, in_w=adc_bits)
            # end-to-end: float readout driven by RTL (time-sums or per-step codes) vs golden readout
            gold_out = readout_from_int(model, [q[:n, :, :t_use] for q in full["yq"]], full["s"])
            rtl_out = readout_from_int(model, st["rtl_yq"], st["scales"], st["rtl_pool"])
            st["readout_identical"] = bool(np.array_equal(gold_out, rtl_out))
            # ADC-width spot check of layer 1 at every sweep width
            adc = []
            if cfg.get("rtl_adc_check", True):
                n2 = min(16, n)
                for b in ADC_BITS:
                    qf, s0b = adc_codes(model, x, b)      # full-set fullscale, as in the golden run
                    q = qf[:n2, :, :t_use]
                    ly = model.layers[0]
                    pr = layer_params(ly, s0b)
                    g, _ = layer_fixed(q, s0b, ly)
                    y, pool = run_layer_rtl(q, ly, pr, b, wd, tag=f"adc{b}")
                    adc.append({"bits": b, "values_compared": int(g.size + pool.size),
                                "mismatches": int(np.sum(y != g) + np.sum(pool != g.sum(axis=2)))})
            res.update(ran=True, error="", examples=n, T=t_use, T_full=T, adc_bits=adc_bits,
                       layers=st["layers"], values_compared=st["values_compared"],
                       mismatches=st["mismatches"], readout_identical=st["readout_identical"],
                       adc_check=adc,
                       adc_values_compared=sum(a["values_compared"] for a in adc),
                       adc_mismatches=sum(a["mismatches"] for a in adc))
    except Exception as e:  # tool failure -> caller falls back to numpy only
        res.update(ran=False, error=f"{type(e).__name__}: {e}")
    return res


def _fidelity(rtl: dict, n_test: int, n_layers: int) -> tuple[str, dict, list[str]]:
    """(status, fidelity, warnings) from the RTL cross-check result."""
    core = ["numpy fixed-point golden model (bit-true): 8-bit absmax ADC, int2 ternary weights, "
            "int32 accumulate, 256-entry tanh LUT -> int8, Q.15 scale / per-line scale, integer pool sum"]
    base_omits = ["all analog effects (loss, dispersion, noise, coupler errors)",
                  "ADC quantiser and float32 readout/softmax are numpy post-processing, not RTL"]
    if not rtl["ran"]:
        return "fallback", {
            "models": core + [f"test-set metrics and sweeps on all {n_test} test examples from the numpy golden model"],
            "omits": [f"Verilog RTL did NOT run ({rtl['error']}); bit-true model (no RTL), no RTL verification"]
                     + base_omits}, [f"RTL verification unavailable, numpy-only: {rtl['error']}"]
    ok = rtl["mismatches"] == 0 and rtl["adc_mismatches"] == 0 and rtl["readout_identical"]
    per = ", ".join(f"L{d['layer']}: {d['values_compared'] + d['pool_compared']} compared / "
                    f"{d['mismatches'] + d['pool_mismatches']} mismatches" for d in rtl["layers"])
    tlen = f"first {rtl['T']} of {rtl['T_full']} time steps" if rtl["T"] < rtl["T_full"] else f"all {rtl['T']} time steps"
    models = core + [
        f"Verilog RTL hw/rtl/tdl_layer.v ({rtl['rtl_tool']}) run on the first {rtl['examples']} test examples ({tlen}), "
        f"all {n_layers} layers chained (each RTL layer fed by the previous RTL layer's output), vs the numpy golden "
        f"model: {rtl['values_compared']} values (layer outputs + pooled sums) compared, {rtl['mismatches']} mismatches "
        f"[{per}]",
        f"RTL-driven float readout on those {rtl['examples']} examples identical to golden: {rtl['readout_identical']}",
        f"RTL layer 1 also checked at ADC widths {ADC_BITS[0]}-{ADC_BITS[-1]} bits on {min(16, rtl['examples'])} examples: "
        f"{rtl['adc_values_compared']} values compared, {rtl['adc_mismatches']} mismatches",
        f"test-set metrics and all sweeps (snr_db, adc_bits) on all {n_test} test examples come from the numpy golden "
        f"model (verified equal to the RTL on the subset above), not from RTL"]
    omits = [f"RTL not simulated on the full test set (only {rtl['examples']} examples, numpy golden elsewhere)"] + base_omits
    if ok:
        return "ok", {"models": models, "omits": omits}, []
    omits.insert(0, "RTL and numpy golden model DISAGREE: see warnings; RTL is not verified")
    return "fallback", {"models": models, "omits": omits}, [
        f"RTL/numpy mismatch: {rtl['mismatches']} chain values, {rtl['adc_mismatches']} ADC-check values, "
        f"readout identical={rtl['readout_identical']}"]


def run(model_npz: str | Path, dataset: dict, cfg: dict, outdir: str | Path) -> contract.TierResult:
    t0 = time.time()
    model = load_model(model_npz)

    def adc_sweep(x: np.ndarray, sc) -> list[dict]:
        return [sweeps.run_sweep("adc_bits", list(ADC_BITS),
                                 lambda xx, b: forward(model, xx, cfg, adc_bits=int(b)), x, sc)]

    x = dataset["x_test"]
    if cfg.get("rtl", True):
        rtl = rtl_crosscheck(model, x, cfg)
    else:
        rtl = {"ran": False, "error": "RTL disabled by cfg['rtl']=False", "rtl_tool": "none"}
    status, fid, warn = _fidelity(rtl, int(x.shape[0]), len(model.layers))
    res = common.evaluate_tier(
        tier="E", status=status, tool={"name": TOOL["name"], "version": TOOL["version"]}, model=model,
        dataset=dataset, cfg=cfg, forward_fn=lambda xx, c: forward(model, xx, c), ref_fn=None,
        fidelity=fid, sweep_axes=("snr_db",), extra_sweeps=adc_sweep, t0=t0, warnings=warn)
    try:  # sidecar with the raw RTL numbers (not part of the tier contract)
        tdir = Path(outdir) / "tiers"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "E_rtl.json").write_text(json.dumps({k: v for k, v in rtl.items()}, indent=2, default=str))
    except OSError:
        pass
    res.runtime_s = round(time.time() - t0, 4)
    return res
