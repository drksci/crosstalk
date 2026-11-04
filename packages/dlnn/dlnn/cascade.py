"""Cascade helpers: read `params/circuit_extracted.json` (written by the C tier) and hand its numbers to other tiers.

Schema `dlnn-circuit-extracted/1` is documented in docs/circuit_extraction.md. The numbers come from ngspice
when `source == "ngspice"` and from a closed-form numpy RLC line when `source == "analytic-numpy-rlgc"`
(C tier fallback); consumers must propagate that provenance. All models are GENERIC (no vendor data).

Typical use by the RF tier (`dlnn/tiers/rf.py`, only when `--cascade`):

    ex = cascade.load_extracted(run_dir / "params" / "circuit_extracted.json")
    cfg = cascade.apply_to_rf_cfg(cfg, ex)            # new dict; original untouched
    # cfg["cascade_circuit"]["tap_H"]  -> {"freq_hz", "H" [K,F] complex, "delay_s" [K]} for the d=1 unit cell
    # cascade.line_H(ex, freqs, delay_s) -> H(f) of a tap line of ANY delay (layers 2-3), fitted-RLC extrapolation
    # result.inputs_from = [cascade.EXTRACT_REL_PATH]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "dlnn-circuit-extracted/1"
EXTRACT_REL_PATH = "params/circuit_extracted.json"

_REQUIRED = ("schema", "source", "tool", "symbol_period_s", "line", "taps", "diffpair", "amp")


class ExtractionError(ValueError):
    pass


def load_extracted(path: str | Path) -> dict:
    """Load + validate the extraction JSON. Adds ndarray views: ``freq_hz`` [F], ``H`` complex [K,F],
    ``delay_s`` [K] (nominal tap delays k*d*Ts of the layer-1 unit cell), ``path``."""
    path = Path(path)
    d = json.loads(path.read_text())
    missing = [k for k in _REQUIRED if k not in d]
    if d.get("schema") != SCHEMA or missing:
        raise ExtractionError(f"{path}: not a {SCHEMA} file (schema={d.get('schema')!r}, missing={missing})")
    t = d["taps"]
    H = np.asarray(t["H_re"], np.float64) + 1j * np.asarray(t["H_im"], np.float64)
    if H.ndim != 2 or H.shape[1] != len(t["freq_hz"]) or H.shape[0] != len(t["k"]):
        raise ExtractionError(f"{path}: inconsistent H shape {H.shape}")
    for key in ("R_ohm_per_m", "L_h_per_m", "C_f_per_m"):
        if not np.isfinite(d["line"]["fit"][key]):
            raise ExtractionError(f"{path}: line.fit.{key} not finite")
    d = dict(d)
    d["freq_hz"] = np.asarray(t["freq_hz"], np.float64)
    d["H"] = H
    d["delay_s"] = np.asarray(t["delay_nominal_s"], np.float64)
    d["path"] = str(path)
    return d


def line_H(extracted: dict, freqs, delay_s: float, termination_ohm: float | None = None) -> np.ndarray:
    """H(f) = V_load/V_source of one tap line with nominal delay `delay_s`, from the FITTED R/L/C (any delay).

    The line is cut so that its delay is exactly `delay_s` (length = delay_s * v_p), loaded by `termination_ohm`
    (default: the extraction's 50 ohm). For delay 0 the tap is a direct connection (H = 1)."""
    from .tiers import circuit
    fit = extracted["line"]["fit"]
    vp = fit["vp_m_per_s"]
    return circuit.line_H(np.asarray(freqs, np.float64), delay_s * vp, fit["R_ohm_per_m"], fit["L_h_per_m"],
                          fit["C_f_per_m"], fit.get("G_s_per_m", 0.0),
                          fit.get("termination_ohm", 50.0) if termination_ohm is None else termination_ohm)


def loss_db_per_us(extracted: dict) -> float:
    """Matched-line attenuation at 100 MHz implied by the fitted line (dB per microsecond of delay)."""
    return float(extracted["line"]["fit"]["loss_db_per_us_100mhz"])


def apply_to_rf_cfg(cfg: dict, extracted: dict) -> dict:
    """Return a COPY of the RF-tier cfg with the circuit extraction attached (original `cfg` is not modified).

    Added keys (stable names; see docs/circuit_extraction.md):
      cfg["cascade_circuit"] = {
          "source": "ngspice" | "analytic-numpy-rlgc", "tool": {...}, "path": <json path or None>,
          "loss_db_per_us": float,                         # fitted, at 100 MHz
          "line_rlc": {"R_ohm_per_m","L_h_per_m","C_f_per_m","G_s_per_m","z0_ohm","vp_m_per_s","termination_ohm"},
          "tap_H": {"freq_hz": ndarray[F], "H": complex ndarray[K,F], "delay_s": ndarray[K]},   # SPICE samples
          "amp": {"vsat_scale","gain_scale","offset_rel","vsat_v","vsat_design_v","gain_a_v","r2"},
          "symbol_period_s": float}
      cfg["loss_db_per_us"], cfg["amp_vsat_scale"], cfg["amp_gain_scale"]   # convenience scalars
      cfg["inputs_from"] += ["params/circuit_extracted.json"]               # provenance
    """
    fit, dp, amp = extracted["line"]["fit"], extracted["diffpair"], extracted["amp"]
    out = dict(cfg)
    out["cascade_circuit"] = {
        "source": extracted["source"], "tool": dict(extracted["tool"]), "path": extracted.get("path"),
        "loss_db_per_us": loss_db_per_us(extracted),
        "line_rlc": {k: fit[k] for k in ("R_ohm_per_m", "L_h_per_m", "C_f_per_m", "G_s_per_m", "z0_ohm",
                                          "vp_m_per_s", "termination_ohm")},
        "tap_H": {"freq_hz": extracted["freq_hz"], "H": extracted["H"], "delay_s": extracted["delay_s"]},
        "amp": {"vsat_scale": amp["vsat_scale"], "gain_scale": amp["gain_scale"], "offset_rel": amp["offset_rel"],
                "vsat_v": dp["vsat_v"], "vsat_design_v": dp["vsat_design_v"], "gain_a_v": dp["A_v"],
                "r2": dp["r2"]},
        "symbol_period_s": extracted["symbol_period_s"],
    }
    out["loss_db_per_us"] = loss_db_per_us(extracted)
    out["amp_vsat_scale"] = float(amp["vsat_scale"])
    out["amp_gain_scale"] = float(amp["gain_scale"])
    out["inputs_from"] = list(cfg.get("inputs_from", [])) + [EXTRACT_REL_PATH]
    return out
