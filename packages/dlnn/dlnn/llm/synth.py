"""Synthetic compare.json dicts with injected divergences, for evaluating the commentary LLM (dlnn.llm.eval).

These are SYNTHETIC cards (invented numbers shaped like real modrec results) used only to test whether the
commentator names the right metric and tier. They are never written to out/ and never presented as simulations.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import compare, contract
from . import card as card_mod

SNRS = ["0", "5", "10", "15", "20"]
V_SNR_ACC = {"0": 0.50, "5": 0.70, "10": 0.88, "15": 0.94, "20": 0.96}

_BASE_MODELS = {
    "V": (["ideal unit delays", "exact ternary sums", "exact tanh", "float32 arithmetic"],
          ["every physical effect: loss, dispersion, noise, quantisation, saturation limits"]),
    "E": (["bit-true model (no RTL)", "{adc}-bit absmax ADC", "int2 ternary weights", "int32 accumulate",
           "256-entry tanh LUT -> int8", "Q.15 fixed-point scaling", "layer dilations {dil}"],
          ["Verilog RTL (not written)", "all analog effects"]),
    "C": (["lossy RLGC coax lines at {loss} dB/us", "ideal tap VCVS", "BJT diff-pair tanh, {bias}", "RC integrator"],
          ["layout parasitics", "full network end-to-end SPICE (unit cell only)"]),
    "R": (["scikit-rf coax lines at {rloss} dB/us", "couplers with isolation {iso} dB", "Wilkinson combiner",
           "amplifier Rapp AM/AM", "AWGN per stage", "layer dilations {rdil}"],
          ["transistor-level behaviour", "EM / layout effects"]),
}


@dataclass
class Scenario:
    sid: str
    title: str
    tier: str | None                       # injected tier (None = control)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)   # overrides for the injected tier: acc_delta, f1_delta, top1, kl, lnmse
    snr_weights: list[float] | None = None
    statuses: dict[str, str] = field(default_factory=dict)    # tier -> stub/fallback
    primary_hint: str = ""


SCENARIOS: list[Scenario] = [
    Scenario("adc6", "6-bit ADC in the emulated tier", "E", {"adc": 6},
             {"acc": -0.046, "f1": -0.049, "top1": 0.905, "kl": 0.162, "lnmse": 0.086}, [0.4, 0.6, 1.0, 1.4, 1.6],
             primary_hint="top1_agreement_vs_virtual"),
    Scenario("loss30", "30 dB/us line loss in the circuit tier", "C", {"loss": 30},
             {"acc": -0.121, "f1": -0.13, "top1": 0.781, "kl": 0.448, "lnmse": 0.352}, [0.5, 0.8, 1.0, 1.2, 1.2],
             primary_hint="mean_kl_virtual_vs_tier"),
    Scenario("iso15", "15 dB coupler isolation in the RF tier (circuit tier is a fallback)", "R", {"iso": 15},
             {"acc": -0.068, "f1": -0.071, "top1": 0.862, "kl": 0.281, "lnmse": 0.19}, [0.7, 1.0, 1.1, 1.0, 1.0],
             {"C": "fallback"}, primary_hint="top1_agreement_vs_virtual"),
    Scenario("vsat", "mis-biased diff pair (V_sat shift) in the circuit tier", "C", {"bias": "bias offset +60 mV, V_sat shifted 1.4x"},
             {"acc": -0.093, "f1": -0.097, "top1": 0.822, "kl": 0.331, "lnmse": 0.503}, [0.2, 0.6, 1.0, 1.5, 1.8],
             primary_hint="logit_nmse_vs_virtual"),
    Scenario("dil", "wrong dilation (layer 3 tap spacing 8 instead of 16) in the RF tier", "R", {"rdil": "1,4,8 (V uses 1,4,16)"},
             {"acc": -0.512, "f1": -0.547, "top1": 0.308, "kl": 2.11, "lnmse": 1.41}, [0.9, 1.0, 1.0, 1.0, 1.0],
             primary_hint="mean_kl_virtual_vs_tier"),
    Scenario("ctrl1", "control: no divergence", None, {}, {}),
    Scenario("ctrl2", "control: no divergence, circuit and RF tiers are stubs", None, {}, {},
             statuses={"C": "stub", "R": "stub"}),
    Scenario("ctrl3", "control: no divergence, different seed", None, {}, {}),
]

_CLEAN = {"adc": 8, "dil": "1,4,16", "loss": 2, "bias": "bias offset 0 mV, V_sat nominal", "rloss": 2, "iso": 30, "rdil": "1,4,16"}


def _sweeps(tier: str, status: str) -> list[dict[str, Any]]:
    sw: list[dict[str, Any]] = []
    snr = [0, 5, 10, 15, 20]
    base = [V_SNR_ACC[str(s)] for s in snr]
    sw.append({"axis": "snr_db", "values": snr, "accuracy": base})
    v = np.array(base)
    if tier == "E":
        bits = [4, 5, 6, 7, 8, 9, 10]
        pen = [0.30, 0.15, 0.047, 0.015, 0.004, 0.001, 0.0]
        sw.append({"axis": "adc_bits", "values": bits, "accuracy": [round(0.808 - p, 3) for p in pen]})
    if tier in ("C", "R"):
        loss = [0, 5, 10, 20, 30, 40]
        pen = [0.0, 0.005, 0.02, 0.06, 0.12, 0.22]
        sw.append({"axis": "loss_db_per_us", "values": loss, "accuracy": [round(0.808 - p, 3) for p in pen]})
    if tier == "R":
        iso = [10, 15, 20, 25, 30, 40]
        pen = [0.15, 0.07, 0.04, 0.01, 0.003, 0.0]
        sw.append({"axis": "isolation_db", "values": iso, "accuracy": [round(0.808 - p, 3) for p in pen]})
        tap = [0.0, 0.25, 0.5, 1.0, 2.0]
        pen = [0.0, 0.002, 0.01, 0.03, 0.08]
        sw.append({"axis": "tap_amp_err_db", "values": tap, "accuracy": [round(0.808 - p, 3) for p in pen]})
    return sw


def make_compare(sc: Scenario, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(1000 + seed)
    params = dict(_CLEAN)
    params.update(sc.params)
    params["rloss"] = params.get("rloss", 2)
    acc_v = float(np.mean([V_SNR_ACC[s] for s in SNRS]))
    tiers = ["V", "E", "C", "R"]
    info: dict[str, Any] = {}
    mm: dict[str, dict[str, Any]] = {m: {} for m in
                                      ("accuracy", "macro_f1", "acc_per_snr", "top1_agreement_vs_virtual",
                                       "mean_kl_virtual_vs_tier", "logit_nmse_vs_virtual")}
    for t in tiers:
        models, omits = _BASE_MODELS[t]
        models = [m.format(**params) for m in models]
        status = sc.statuses.get(t, "ok")
        if status == "stub":
            models = ["V forward on input with AWGN (35 dB SNR) and 0.3 dB flat loss"]
            omits = [f"STUB / PLACEHOLDER: this is the virtual tier plus noise, NOT the real {t} simulation"]
        elif status == "fallback":
            omits = omits + ["FALLBACK: closed-form analytic line model replaced the real SPICE run"]
        warns = [f"{t} tier is a {status}; not the real tool"] if status != "ok" else []
        tool = {"V": "numpy-float32", "E": "numpy-fixedpoint", "C": "ngspice" if status == "ok" else "analytic-rlgc",
                "R": "scikit-rf"}[t]
        if status == "stub":
            tool = "stub(V+AWGN)"
        info[t] = {"status": status, "tool": {"name": tool, "version": "x"},
                   "fidelity": {"models": models, "omits": omits}, "runtime_s": 1.0, "warnings": warns,
                   "inputs_from": [], "n_examples": 4000}
        inj = (t == sc.tier)
        if t == "V":
            acc, f1, top1, kl, ln = acc_v, acc_v - 0.006, 1.0, 0.0, 0.0
            w = [1.0] * 5
            d = 0.0
        elif inj:
            m = sc.metrics
            acc, f1 = acc_v + m["acc"], acc_v - 0.006 + m["f1"]
            top1, kl, ln = m["top1"], m["kl"], m["lnmse"]
            w = sc.snr_weights or [1.0] * 5
            d = m["acc"]
        else:
            d = float(rng.normal(0, 0.003))
            acc, f1 = acc_v + d, acc_v - 0.006 + d
            top1 = float(np.clip(0.992 + rng.normal(0, 0.003), 0.98, 1.0))
            kl = float(abs(rng.normal(0.004, 0.0015)))
            ln = float(abs(rng.normal(0.008, 0.003)))
            w = [1.0] * 5
        mm["accuracy"][t] = round(acc, 4)
        mm["macro_f1"][t] = round(f1, 4)
        mm["top1_agreement_vs_virtual"][t] = round(top1, 4)
        mm["mean_kl_virtual_vs_tier"][t] = round(kl, 4)
        mm["logit_nmse_vs_virtual"][t] = round(ln, 4)
        mm["acc_per_snr"][t] = {s: round(V_SNR_ACC[s] + d * wi * (1.0 if t != "V" else 0), 3) for s, wi in zip(SNRS, w)}
    dv: dict[str, dict[str, Any]] = {}
    for mid, row in mm.items():
        dv[mid] = {}
        for t, v in row.items():
            if t == "V":
                continue
            dv[mid][t] = ({k: v[k] - row["V"][k] for k in v} if isinstance(v, dict) else v - row["V"])
    sweeps: dict[str, dict[str, Any]] = {}
    knees: list[dict[str, Any]] = []
    for t in tiers:
        for sw in _sweeps(t, info[t]["status"]) if t != "V" else _sweeps("V", "ok")[:1]:
            sweeps.setdefault(sw["axis"], {})[t] = {k: v for k, v in sw.items() if k != "axis"}
            knees.append({"axis": sw["axis"], "tier": t, "metric": "accuracy",
                          "knee_value": compare._knee(sw["axis"], sw["values"], sw["accuracy"], "accuracy"),
                          "rule": compare.KNEE_RULES["accuracy"][2]})
    cmp = {"schema_version": contract.COMPARE_SCHEMA_VERSION, "run_id": f"synthetic-{sc.sid}", "task": "modrec",
           "model_sha256": "0" * 64, "dataset_sha256": "1" * 64, "tiers": tiers, "tier_info": info,
           "metric_matrix": mm, "deltas_vs_virtual": dv, "sweeps": sweeps, "knees": knees}
    return contract.validate_compare(contract._jsonable(cmp))


def expected(sc: Scenario, cmp: dict[str, Any]) -> dict[str, Any]:
    """Ground truth for scoring: the injected tier and the metrics the card flags as material for that tier."""
    if sc.tier is None:
        return {"tier": None, "metrics": [], "primary": None}
    dvs = [r for r in card_mod.divergences(cmp) if r["tier"] == sc.tier]
    return {"tier": sc.tier, "metrics": [r["metric_id"] for r in dvs],
            "primary": dvs[0]["metric_id"] if dvs else None}


def all_cards() -> list[tuple[Scenario, dict[str, Any]]]:
    out = []
    for i, sc in enumerate(SCENARIOS):
        out.append((sc, make_compare(sc, seed=i)))
    return out


def example_regression() -> dict[str, Any]:
    """Tiny invented regression compare used ONLY as the in-context example of the prompt (different task and
    different injected defects than every eval card)."""
    def info(status, tool, models, omits, warns=()):
        return {"status": status, "tool": {"name": tool, "version": "x"}, "fidelity": {"models": models, "omits": omits},
                "runtime_s": 0.5, "warnings": list(warns), "inputs_from": [], "n_examples": 1900}
    ti = {"V": info("ok", "numpy-float32", ["ideal unit delays", "exact tanh"], ["every physical effect"]),
          "E": info("ok", "numpy-fixedpoint", ["bit-true model (no RTL)", "5-bit absmax ADC", "int2 ternary weights"],
                    ["all analog effects"]),
          "C": info("fallback", "analytic-rlgc", ["closed-form lossy line model at 2 dB/us"],
                    ["FALLBACK: analytic line model replaced the real SPICE run", "layout parasitics"],
                    ["C tier is a fallback; not the real tool"]),
          "R": info("ok", "scikit-rf", ["coax lines at 2 dB/us", "couplers with isolation 12 dB", "Wilkinson combiner"],
                    ["EM / layout effects"])}
    mm = {"nmse": {"V": 0.152, "E": 0.231, "C": 0.154, "R": 0.203},
          "nmse_vs_virtual": {"V": 0.0, "E": 0.118, "C": 0.004, "R": 0.071}}
    dv = {m: {t: v - row["V"] for t, v in row.items() if t != "V"} for m, row in mm.items()}
    sweeps = {"adc_bits": {"E": {"values": [4, 5, 6, 7, 8], "nmse": [0.40, 0.231, 0.17, 0.155, 0.152]}},
              "isolation_db": {"R": {"values": [10, 20, 30], "nmse": [0.26, 0.17, 0.152]}},
              "snr_db": {t: {"values": [10, 20, 30], "nmse": [0.5, 0.2, 0.152]} for t in ("V", "E", "C", "R")}}
    knees = [{"axis": "adc_bits", "tier": "E", "metric": "nmse", "knee_value": 6.0, "rule": compare.KNEE_RULES["nmse"][2]},
             {"axis": "isolation_db", "tier": "R", "metric": "nmse", "knee_value": 20.0, "rule": compare.KNEE_RULES["nmse"][2]}]
    knees += [{"axis": "snr_db", "tier": t, "metric": "nmse", "knee_value": 20.0, "rule": compare.KNEE_RULES["nmse"][2]}
              for t in ("V", "E", "C", "R")]
    cmp = {"schema_version": contract.COMPARE_SCHEMA_VERSION, "run_id": "example-regression", "task": "narma10",
           "model_sha256": "0" * 64, "dataset_sha256": "1" * 64, "tiers": ["V", "E", "C", "R"], "tier_info": ti,
           "metric_matrix": mm, "deltas_vs_virtual": dv, "sweeps": sweeps, "knees": knees}
    return contract.validate_compare(contract._jsonable(cmp))


EXAMPLE_ANSWER = {
    "summary": "E and R diverge from V (nmse_vs_virtual=0.118 and 0.071); C stays close to V but is a fallback, so that closeness says nothing about circuit physics.",
    "findings": [
        {"metric_id": "nmse_vs_virtual", "tiers": ["E"],
         "observation": "E has nmse_vs_virtual=0.118 and nmse=0.231 (+0.079 versus V).",
         "likely_cause": "E's own models line lists a 5-bit absmax ADC, so quantisation is the candidate cause; its omits line says all analog effects are not modelled by E."},
        {"metric_id": "nmse_vs_virtual", "tiers": ["R"],
         "observation": "R has nmse_vs_virtual=0.071 and nmse=0.203 (+0.051 versus V).",
         "likely_cause": "R's own models line lists couplers with isolation 12 dB, so coupler leakage is the candidate cause; EM and layout effects are not modelled by R."}],
    "caveats": ["C is a FALLBACK (closed-form line model), not the real SPICE simulation; its agreement with V is not evidence about circuit behaviour."],
    "questions": ["Does E recover to the V level with more ADC bits, and does R with better isolation (see the sweeps)?"],
}
