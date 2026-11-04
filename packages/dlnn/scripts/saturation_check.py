"""Read-only check (M4d): how much of each layer's tanh input is in saturation, per tier / compensation mode.

Uses the tiers' own forward code (dlnn.tiers.circuit: cursor_trim, effective_kernel, _conv_fft; dlnn.tiers.rf: build_channels,
fir_apply, amplifier, stage_sigma) and the fitted parameters of a finished run (default out/final-w32). Nothing is written to out/.
Saturation = |tanh(z)| > 0.95, i.e. |z| > atanh(0.95) = 1.832, with z = (preactivation + bias) / (V_sat * vsat_scale)
(the argument of the amplifier). Fractions are over all (example, output channel, time step) units of the layer.

  .venv/bin/python scripts/saturation_check.py [--run out/final-w32] [--n 400] [--json PATH]
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from dlnn.data import modrec
from dlnn.model.npz import load_model
from dlnn.tiers import circuit, rf, virtual

ZSAT = math.atanh(0.95)


def frac(z):
    return float(np.mean(np.abs(z) > ZSAT))


def c_layers(model, x, line, amp, comp):
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    zs = []
    for ly in model.layers:
        trim = circuit.cursor_trim(ly, line) if comp == "layer" else 1.0
        Wk, _ = circuit.effective_kernel(ly, line, trim, equalise=(comp == "tap"))
        u = circuit._conv_fft(h, Wk) + ly.bias[None, :, None]
        z = u / (ly.vsat * amp.vsat_scale)
        zs.append(z)
        h = ly.gain[None, :, None] * (amp.gain_scale * np.tanh(z) + amp.offset_rel)
    return zs


def r_layers(model, x, rc, chans, p_ref):
    rng = np.random.default_rng(int(rc["rf_seed"]) + 7)
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    zs = []
    for l, (ly, ch) in enumerate(zip(model.layers, chans)):
        u = rf.fir_apply(h, ch)
        if rc["snr_db"] is not None:
            u = u + (rf.stage_sigma(p_ref[l], rc) * rng.standard_normal(u.shape)).astype(np.float32)
        z = (u + ly.bias[None, :, None]) / np.float32(ly.vsat * rc["amp_vsat_scale"])
        zs.append(z)
        h = (ly.gain[None, :, None] * (rc["amp_gain_scale"] * rf.amplifier(z, rc) + rc["amp_offset_rel"])).astype(np.float32)
    return zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/final-w32")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run / "model.npz")
    ds = modrec.dataset()
    x = ds["x_test"][:a.n]
    ex = json.loads((run / "params" / "circuit_extracted.json").read_text())
    fit = ex["line"]["fit"]
    line = circuit.LineModel(fit["R_ohm_per_m"], fit["L_h_per_m"], fit["C_f_per_m"])
    d = ex["diffpair"]
    amp = circuit.AmpModel(d["vsat_scale"], d["gain_scale"], d["offset_rel"])
    out = {"n_examples": int(len(x)), "z_sat_threshold": ZSAT, "run": str(run), "loss_db_per_us_fitted": fit["loss_db_per_us_100mhz"]}
    # V
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    zs = []
    for ly in model.layers:
        u = virtual.layer_preact(h, ly) + ly.bias[None, :, None]
        z = u / np.float32(ly.vsat)
        zs.append(z)
        h = ly.gain[None, :, None] * np.tanh(z)
    out["V"] = [frac(z) for z in zs]
    for comp in ("layer", "none", "tap"):
        out[f"C_{comp}"] = [frac(z) for z in c_layers(model, x, line, amp, comp)]
    ytrue = ds["y_test"][:a.n]
    pv = virtual.forward(model, x)["out"].argmax(1)
    out["acc_V_n"] = float(np.mean(pv == ytrue))
    for comp in ("layer", "none", "tap"):
        o = circuit.calibrated_forward(model, x, line, amp, comp=comp)["out"].argmax(1)
        out[f"C_{comp}_acc"] = float(np.mean(o == ytrue))
        out[f"C_{comp}_agree_V"] = float(np.mean(o == pv))
    out["C_layer_trims"] = [circuit.cursor_trim(ly, line) for ly in model.layers]
    # R, as the tier runs it with --cascade (params/circuit_extracted.json -> loss, amp scales), and standalone defaults
    base = {}
    cfg_c = rf._apply_cascade({"cascade_params": str(run / "params" / "circuit_extracted.json")}, [], [])
    for name, cfg in (("R_cascade_32dBus", cfg_c), ("R_default_5dBus", dict(base)),
                      ("R_cascade_32dBus_tap_equalize", {**cfg_c, "tap_equalize": True})):
        rc = rf.resolve_cfg(cfg)
        chans, _ = rf.build_channels(model, rc)
        p_ref = rf.reference_power(model, ds["x_test"])
        out[name] = [frac(z) for z in r_layers(model, x, rc, chans, p_ref)]
        o = rf.forward_with(model, x, rc, chans, p_ref)["out"].argmax(1)
        out[name + "_acc"], out[name + "_agree_V"] = float(np.mean(o == ytrue)), float(np.mean(o == pv))
        out[name + "_loss_db_per_us"] = rc["loss_db_per_us"]
        out[name + "_comp_gain_db"] = [c.info["required_amp_gain_db"] for c in chans]
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
