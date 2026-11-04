#!/usr/bin/env python
"""Regenerate docs/results.md (and the headline table block of docs/README.md) from finished run directories.

    .venv/bin/python scripts/make_results_md.py                       # out/final-w32 + out/final-w8 -> docs/results.md
    .venv/bin/python scripts/make_results_md.py --w32 out/modrec-w32-full --w8 out/modrec-w8-full --llm out/modrec-w32-qwen-pg

Every number in the output is read from files (compare.json, tiers/*.json, params/*.json, models/train_report*.json,
models/llm/eval_report.json, out/<run>/r_ideal30.txt and docs/saturation_check_*.json when present). Only the prose between the tables is static, and it
contains no result numbers (the attribution paragraph of section 2 and the LLM paragraphs are generated from the files). If an input file is missing the corresponding section says so instead of inventing anything.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["V", "E", "C", "R"]
AXES = [("snr_db", "SNR sweep"), ("adc_bits", "ADC bits"), ("loss_db_per_us", "line loss (dB per microsecond)"),
        ("isolation_db", "coupler isolation (dB)"), ("tap_amp_err_db", "tap amplitude error sigma (dB)")]
AXIS_NOTE = {
    "snr_db": "V, E, C: AWGN added to the input on top of the dataset's own noise. R: per-stage amplifier SNR (a different quantity).",
    "adc_bits": "E: input ADC width of the fixed-point model. R: readout ADC width.",
    "loss_db_per_us": "C and R: line loss at 100 MHz; each tier's own per-layer gain compensation is re-applied at every point (C: default mode `layer`; the `none` and `tap` modes are in section 2).",
    "isolation_db": "R only.", "tap_amp_err_db": "R only."}


def jl(p: Path):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if x != 0 and abs(x) < 1e-3:
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_info(d: Path):
    cmp = jl(d / "compare.json")
    if cmp is None:
        return None
    return {"dir": d, "cmp": cmp, "tiers": {t: jl(d / "tiers" / f"{t}.json") for t in TIERS}, "rtl": jl(d / "tiers" / "E_rtl.json"),
            "ext": jl(d / "params" / "circuit_extracted.json"), "diag": jl(d / "params" / "circuit_diagnostics.json"),
            "com": jl(d / "llm" / "commentary.json"), "card": (d / "llm" / "card.txt").read_text() if (d / "llm" / "card.txt").exists() else None,
            "ideal30": parse_ideal30(d / "r_ideal30.txt")}


def parse_ideal30(p: Path):
    if not p.exists():
        return None
    res = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^(\S+)\s+(\{.*\})\s*$", line)
        if m:
            try:
                res[m.group(1)] = json.loads(m.group(2))
            except json.JSONDecodeError:
                pass
    return res or None


# ---------------------------------------------------------------- tables
def headline_table(r):
    c = r["cmp"]
    mm = c["metric_matrix"]
    rows = []
    for t in TIERS:
        ti = c["tier_info"][t]
        tool = f"{ti['tool']['name']} {ti['tool']['version']}"
        if t == "C" and ti["tool"]["name"] == "ngspice":      # the 4000-example numbers are NOT computed by ngspice
            tool = f"numpy behavioural model calibrated on {tool} (ngspice itself ran only on a 16-example layer-1 cell)"
        elif t == "E":
            tool += " (golden model; Icarus RTL cross-check on a subset)"
        rows.append([f"**{t}**", ti["status"], tool, ti["n_examples"], f(mm["accuracy"][t], 4),
                     f(mm["macro_f1"][t], 4), f(mm["top1_agreement_vs_virtual"][t], 4), f(mm["mean_kl_virtual_vs_tier"][t], 4),
                     f(mm["logit_nmse_vs_virtual"][t], 4), f"{ti['runtime_s']:.0f} s"])
    ref = c.get("reference")
    if ref:
        rows.append(["*float32 reference (not a tier)*", "reference", ref["tool"], ref["n_examples"], f(ref["accuracy"]), f(ref["macro_f1"]),
                     f(ref["top1_agreement_with_ternary_V"]) + " (vs ternary V)", "", "", ""])
    return table(["tier", "status", "tool", "n", "accuracy", "macro-F1", "top-1 agreement vs V", "mean KL(V||tier) nats", "logit NMSE vs V", "runtime"], rows)


def snr_table(r):
    c = r["cmp"]
    snrs = sorted(c["metric_matrix"]["acc_per_snr"]["V"], key=float)
    rows = [[t] + [f(c["metric_matrix"]["acc_per_snr"][t][s], 3) for s in snrs] for t in TIERS]
    ref = c.get("reference")
    if ref:
        rows.append(["float32 reference"] + [f(ref["acc_per_snr"][s], 3) for s in snrs])
    return table(["tier \\ SNR (dB)"] + snrs, rows)


def short_headline(r):
    c = r["cmp"]
    mm = c["metric_matrix"]
    ref = c.get("reference") or {}
    rows = []
    for t in TIERS:
        rows.append([t, c["tier_info"][t]["status"], f(mm["accuracy"][t], 3), f(mm["top1_agreement_vs_virtual"][t], 3)])
    rows.append(["float32 reference", "reference", f(ref.get("accuracy"), 3), ""])
    return table(["tier", "status", "accuracy (4000 test examples)", "top-1 agreement vs V"], rows)


def sweep_tables(r):
    c = r["cmp"]
    out = []
    for ax, title in AXES:
        sw = c["sweeps"].get(ax)
        if not sw:
            continue
        vals = None
        for t in TIERS:
            if t in sw:
                vals = sw[t]["values"]
                break
        # tiers can have different value grids on the same axis (e.g. snr_db); one table per distinct grid
        grids = {}
        for t in TIERS:
            if t in sw:
                grids.setdefault(tuple(sw[t]["values"]), []).append(t)
        out.append(f"#### {title}\n\n{AXIS_NOTE[ax]}\n")
        for grid, ts in grids.items():
            rows = []
            for t in ts:
                rows.append([f"{t} accuracy"] + [f(a, 3) for a in sw[t]["accuracy"]])
            for t in ts:
                rows.append([f"{t} mean KL"] + [f(a, 3) for a in sw[t]["mean_kl"]])
            out.append(table([ax] + [f(v, 3).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in grid], rows) + "\n")
    return "\n".join(out)


def knee_table(r):
    rows = []
    for k in r["cmp"]["knees"]:
        rows.append([k["axis"], k["tier"], k["metric"], "none within the swept range" if k["knee_value"] is None else f(k["knee_value"], 3).rstrip("0").rstrip(".")])
    return table(["axis", "tier", "metric", "knee (first value >= 0.05 below the best end)"], rows)


def timing_table(r):
    tm = r["cmp"].get("timing")
    if not tm:
        return "(no timing block in this run)"
    rows = [[k, f"{v:.1f}"] for k, v in tm["stages_s"].items()]
    rows.append(["**total (compare.json timing.total_s)**", f"{tm['total_s']:.1f}"])
    return table(["stage", "seconds"], rows)


def loss_section(r):
    c = r["cmp"]
    sw = c["sweeps"].get("loss_db_per_us", {})
    ext = r["ext"] or {}
    diag = r["diag"]
    out = []
    vals = sw.get("C", {}).get("values")
    if vals:
        rows = []
        rows.append(["C, mode `layer` (tier default: one ideal gain trim per layer)"] + [f(a, 3) for a in sw["C"]["accuracy"]])
        if diag:
            for mode, lab in (("none", "C, mode `none` (no compensation)"), ("tap", "C, mode `tap` (ideal per-tap repeater)")):
                m = diag["modes"].get(mode)
                if m:
                    rows.append([lab] + [f(a, 3) for a in m["loss_sweep"]["accuracy"]])
        if "R" in sw:
            rows.append(["R (one gain per layer, long taps droop)"] + [f(a, 3) for a in sw["R"]["accuracy"]])
        out.append(table(["accuracy at loss (dB/us) ->"] + [str(int(v)) for v in vals], rows))
    trims = ext.get("loss_comp_gain_trim_per_layer")
    gains = None
    for w in (r["tiers"]["R"] or {}).get("warnings", []):
        m = re.search(r"Required amplifier gain per layer \(dB\): (.*)$", w)
        if m:
            gains = m.group(1)
    extra = []
    if trims:
        extra.append("C's per-layer gain trim at the fitted loss (what the single trim per layer has to supply): " + ", ".join(f(t, 2) for t in trims) + " (linear).")
    if gains:
        extra.append("R's required amplifier gain per layer at the run's operating point (dB): " + gains + ".")
    if "loss_db_per_us" in ext.get("line", {}).get("fit", {}) or ext:
        fit = ext.get("line", {}).get("fit", {})
        if fit.get("loss_db_per_us_100mhz") is not None:
            extra.append(f"Fitted line loss that `--cascade` hands to R: {fit['loss_db_per_us_100mhz']:.2f} dB/us at 100 MHz.")
    return "\n".join(out) + "\n\n" + "\n\n".join(extra) if out else "(no loss sweep in this run)"


def attribution_section(r):
    """Section 2 numbers: how much of C's gap to V is the compensation mode and how much is line loss (generated from files)."""
    c, diag = r["cmp"], r["diag"]
    mm = c["metric_matrix"]["accuracy"]
    if not diag or "C" not in mm:
        return ""
    v, cl = mm["V"], mm["C"]
    none = diag["modes"].get("none", {}).get("metrics_at_fitted_loss", {}).get("accuracy")
    tap = diag["modes"].get("tap", {}).get("metrics_at_fitted_loss", {}).get("accuracy")
    if none is None:
        return ""
    gap, gap_none = 100 * (v - cl), 100 * (v - none)
    out = [f"**How much of C's gap is line loss?** At the cascade loss C scores {f(cl, 3)} with its default `{diag.get('tier_default', 'layer')}` mode, {f(none, 3)} with `none` (no compensation at all)"
           + (f" and {f(tap, 3)} with `tap` (ideal per-tap repeaters)" if tap is not None else "")
           + f", against V's {f(v, 3)}. Of C's {gap:.1f}-point gap to V, {gap - gap_none:.1f} points belong to the choice of compensation mode and {gap_none:.1f} points to the line loss itself (mode `none`)."
           + (f" Ideal per-tap repeaters change C's accuracy by {100 * (tap - none):+.1f} points relative to doing nothing." if tap is not None else "")
           + " Both statements belong with the headline: the cascade operating point is not buildable without repeaters, **and** the default compensation mode over-drives layer 3."]
    sat = jl(ROOT / "docs" / "saturation_check_w32_n400.json")
    sat4 = jl(ROOT / "docs" / "saturation_check_w32_n4000.json")
    if sat and Path(sat.get("run", "")).name == r["dir"].name:
        def row(lab, key, accn=None):
            vals = sat.get(key)
            if vals is None:
                return None
            l3 = f(vals[2], 3) + (f" ({f(sat4[key][2], 3)})" if sat4 and key in sat4 else "")
            return [lab] + [f(x, 3) for x in vals[:2]] + [l3, f(sat.get(accn), 3) if accn else ""]
        rows = [row("V (ideal)", "V", "acc_V_n"), row("C, mode `layer` (tier default)", "C_layer", "C_layer_acc"), row("C, mode `none`", "C_none", "C_none_acc"),
                row("C, mode `tap` (ideal per-tap repeaters)", "C_tap", "C_tap_acc"), row("R, `--cascade` (tier default gain rule)", "R_cascade_32dBus", "R_cascade_32dBus_acc"),
                row("R, `--cascade` with `tap_equalize`", "R_cascade_32dBus_tap_equalize", "R_cascade_32dBus_tap_equalize_acc"),
                row("R, standalone defaults (5 dB/us)", "R_default_5dBus", "R_default_5dBus_acc")]
        trims = sat.get("C_layer_trims")
        out.append(f"**Why: measured saturation of the tanh inputs** (`scripts/saturation_check.py`, the tiers' own forward code, read-only, first {sat['n_examples']} test examples; "
                   f"saturated = |tanh| > 0.95; layer 3 on all 4000 examples in brackets; results in `docs/saturation_check_w32_n400.json`):\n\n"
                   + table(["model", "layer 1", "layer 2", "layer 3", f"accuracy ({sat['n_examples']})"], [x for x in rows if x]))
        out.append("In layer 3 the tap main cursors run from 1 (no line) to about -35 dB, so the single per-layer trim"
                   + (f" (here {f(trims[2], 2)} in layer 3, {f(trims[0], 2)} and {f(trims[1], 2)} in layers 1 and 2)" if trims else "")
                   + " also multiplies the almost unattenuated short taps and pushes layer 3 into saturation, well above V's own level; `tap` restores V's level. "
                   "R's per-layer gain rule shows the same layer-3 saturation at the cascade loss, and R's own per-tap equaliser (optimistic: no repeater noise) raises its accuracy there; "
                   "that R's rule is a modelling artefact was not shown and is not claimed. Details: `docs/circuit_extraction.md`, \"Where C's gap comes from\".")
    return "\n\n".join(out)


def llm_ablation_sentence():
    rep = jl(ROOT / "models" / "llm" / "eval_report.json")
    if not rep:
        return ""
    sm = {(x["backend"], x["rank_section"]): x for x in rep["summaries"]}
    w, wo = sm.get(("qwen", True)), sm.get(("qwen", False))
    if not (w and wo):
        return ""
    return (f"With the list, Qwen attributes {w['injected_correct']}/{w['injected_total']} injected cards (loose rule; {w['injected_strict']}/{w['injected_total']} under the strict most-severe-metric rule) and gets "
            f"{w['controls_correct']}/{w['controls_total']} controls right. **Without** the list it attributes **{wo['injected_correct']}/{wo['injected_total']} injected cards** "
            f"and gets {wo['controls_correct']}/{wo['controls_total']} controls right; the {wo['attribution_correct']}/{wo['attribution_total']} in the table's first column is a control with no divergence, not a found divergent tier. "
            f"The other controls are false alarms (the summary claims a divergence where there is none). It restates what the card says and does not discover which tier diverges. "
            f"BitNet's {sm[('bitnet', True)]['attribution_correct']}/{sm[('bitnet', True)]['attribution_total']} is likewise the {sm[('bitnet', True)]['controls_correct']} controls only, reached by writing almost nothing, with {sm[('bitnet', True)]['injected_correct']}/{sm[('bitnet', True)]['injected_total']} injected cards attributed "
            f"(output degenerate here, `docs/llm_eval.md`).")


def commentary_verdict(r):
    com = r["com"]
    if not com:
        return ""
    bits = [f"`numbers_verified = {com.get('numbers_verified')}`", f"`ids_verified = {com.get('ids_verified')}`"]
    miss = [u for u in com.get("unverified", []) if u.get("kind") == "missing_tier"]
    s = "**Verdict of the guard on this commentary:** " + ", ".join(bits) + "."
    if com.get("ids_verified") is False:
        s += " Numbers pass, but the identifier check fails"
        if miss:
            s += ": the divergent tier(s) " + ", ".join(f"`{u['token']}`" for u in miss) + " have no finding (they appear only in the summary and in a caveat)"
        s += ". \"All numbers verified\" therefore does not mean the commentary is right."
    n = audit(r)
    if n:
        s += f" The independent audit below found {len(n)} further problem(s) that the guard does not catch."
    return s


def class_table(rep_w8, rep_w32):
    if not (rep_w8 and rep_w32):
        return "(training reports missing)"
    classes = list(rep_w32["ternary"]["acc_per_class"])
    head = ["class", "w8 float", "w8 ternary", "w32 float", "w32 ternary"]
    rows = []
    for k in classes:
        rows.append([k, f(rep_w8["float"]["acc_per_class"][k], 3), f(rep_w8["ternary"]["acc_per_class"][k], 3),
                     f(rep_w32["float"]["acc_per_class"][k], 3), f(rep_w32["ternary"]["acc_per_class"][k], 3)])
    rows.append(["**mixed SNR**", f(rep_w8["float"]["acc"], 4), f(rep_w8["ternary"]["acc"], 4), f(rep_w32["float"]["acc"], 4), f(rep_w32["ternary"]["acc"], 4)])
    rows.append(["**SNR >= 10 dB**", f(rep_w8["float"]["acc_ge10dB"], 4), f(rep_w8["ternary"]["acc_ge10dB"], 4), f(rep_w32["float"]["acc_ge10dB"], 4),
                 f(rep_w32["ternary"]["acc_ge10dB"], 4)])
    return table(head, rows)


def w8_vs_w32_table(r8, r32):
    rows = []
    for t in TIERS:
        rows.append([t, f(r32["cmp"]["metric_matrix"]["accuracy"][t], 3), f(r8["cmp"]["metric_matrix"]["accuracy"][t], 3),
                     f(r32["cmp"]["metric_matrix"]["top1_agreement_vs_virtual"][t], 3), f(r8["cmp"]["metric_matrix"]["top1_agreement_vs_virtual"][t], 3),
                     f(r32["cmp"]["metric_matrix"]["mean_kl_virtual_vs_tier"][t], 3), f(r8["cmp"]["metric_matrix"]["mean_kl_virtual_vs_tier"][t], 3)])
    rf32, rf8 = r32["cmp"].get("reference", {}), r8["cmp"].get("reference", {})
    rows.append(["float32 reference", f(rf32.get("accuracy"), 3), f(rf8.get("accuracy"), 3), "", "", "", ""])
    return table(["tier", "accuracy w32", "accuracy w8", "agreement w32", "agreement w8", "mean KL w32", "mean KL w8"], rows)


def rtl_section(r):
    rtl, e = r["rtl"], r["tiers"]["E"]
    if not rtl:
        return "(no E_rtl.json in this run)"
    rows = [[f"layer {l['layer']}", str(l["shape"]), l["values_compared"], l["mismatches"], l["pool_compared"], l["pool_mismatches"]] for l in rtl["layers"]]
    rows.append(["**all layers**", "", rtl["values_compared"], rtl["mismatches"], "", ""])
    s = table(["", "(examples, lines, steps)", "layer outputs compared", "mismatches", "pooled sums compared", "pooled mismatches"], rows)
    s += (f"\n\nRTL tool: {rtl.get('rtl_tool')}; {rtl['examples']} test examples, first {rtl['T']} of {rtl['T_full']} time steps per example "
          f"(causal, so the prefix is exact), ADC {rtl['adc_bits']} bits; tier status `{e['status']}`.")
    if rtl["T"] < rtl["T_full"]:
        K, d3 = 8, 16                               # TDL-Net topology: K = 8 taps, dilations 1, 4, 16
        dead = [k for k in range(K) if k * d3 >= rtl["T"]]
        s += (f"\n\nLimit of a prefix check: with only {rtl['T']} of {rtl['T_full']} steps, the layer-3 taps at lags {', '.join(str(k * d3) for k in dead)} (k = {dead[0]}..{dead[-1]}) only ever see the zero history, so "
              f"{len(dead)} of the {K} layer-3 tap positions are never driven with data in this run, and the pooled sums are {rtl['T']}-step prefix pools, not the classification pools. "
              "The prefix argument (causality) is exact for the steps compared; the long-tap logic is exercised by the width-8 full-length run below.")
    for line in e["fidelity"]["models"]:
        if "ADC widths" in line:
            s += f"\n\nAs recorded in the tier JSON: \"{line}\""
    return s


def c_validation(r):
    ext = r["ext"]
    if not ext:
        return "(no circuit extraction in this run)"
    u = ext.get("unit_cell_tran") or {}
    d = ext.get("diffpair", {})
    rows = [["ngspice version", ext.get("ngspice_version")],
            ["extraction source", ext.get("source")],
            ["diff-pair `.dc` tanh fit R^2", f(d.get("r2"), 8)],
            ["diff-pair fitted V_sat (mV)", f(1e3 * d.get("vsat_v", float("nan")), 2)],
            ["fitted line loss at 100 MHz (dB/us)", f(ext["line"]["fit"].get("loss_db_per_us_100mhz"), 3)],
            ["unit-cell `.tran`: examples", u.get("n_examples")],
            ["unit-cell layer-1 NMSE vs V (variance-normalised, with gain trim)", f(u.get("nmse"), 4)],
            ["same, energy-normalised", f(u.get("nmse_energy_normalised"), 4)],
            ["same cell without gain trim", f((u.get("no_trim") or {}).get("nmse"), 4)],
            ["calibrated behavioural layer 1 vs the SPICE traces (NMSE)", f(u.get("nmse_calibrated_model_vs_spice"), 8)],
            ["RC pool (tau = N*Ts) vs the uniform mean the model was trained with (NMSE)", f(u.get("pool_rc_vs_uniform_mean_nmse"), 4)],
            ["SPICE wall time of the 16-example unit cell (s)", f(u.get("spice_wall_s"), 1)]]
    return table(["quantity", "value"], rows)


def ideal30_section(r):
    d = r["ideal30"]
    if not d:
        return "`scripts/r_ideal30.py` output not found for this run (expected `r_ideal30.txt` next to `compare.json`)."
    rows = []
    for name, v in d.items():
        s30 = v.get("stage_snr_30.0", {})
        rows.append([Path(name).name, f(v.get("stage_snr_None", {}).get("agree_mean"), 4), f(v.get("stage_snr_40.0", {}).get("agree_mean"), 4),
                     f"**{f(s30.get('agree_mean'), 4)}** (" + " / ".join(f(a, 4) for a in s30.get("agree_per_seed", [])) + ")", f(v.get("V_input_awgn30_vs_Vclean"), 4)])
    return table(["model", "R ideal, noiseless", "R ideal, 40 dB per stage", "R ideal, 30 dB per stage (3 seeds)", "V(30 dB input AWGN) vs V(clean)"], rows)


def llm_eval_section():
    rep = jl(ROOT / "models" / "llm" / "eval_report.json")
    if not rep:
        return "(models/llm/eval_report.json missing)"
    rows = []
    for s in rep["summaries"]:
        rows.append([f"{s['backend']} ({s['source']})", "with" if s["rank_section"] else "without",
                     f"{s['attribution_correct']}/{s['attribution_total']}", f"{s['injected_correct']}/{s['injected_total']} / {s['injected_strict']}/{s['injected_total']}",
                     f"{s['controls_correct']}/{s['controls_total']}", s["stub_flagged"], s["numbers_total"], s["numbers_unverified_raw"],
                     s["numbers_unverified_final_flagged"], s["numbers_exposed_as_verified"], f"{s['id_failures_raw']} / {s['id_failures_final']}",
                     s["decode_tok_per_s_mean"]])
    return (f"Generated {rep['generated']}.\n\n" + table(
        ["backend", "divergence list on card", "attribution (8 cards)", "injected 5 (loose / strict)", "controls", "stub flagged", "numbers checked",
         "unverified raw", "unverified after retry (flagged)", "exposed as verified", "id/term failures raw / final", "decode tok/s"], rows))


def commentary_section(r, label):
    com, card = r["com"], r["card"]
    if not com:
        return f"(no `llm/commentary.json` in {label})"
    out = []
    llm = com.get("llm") or {}
    out.append(f"Source: `{com['source']}`, backend `{com['backend']}`" + (f", model {llm.get('model')} (GGUF sha256 `{llm.get('model_sha256')}`), temperature {llm.get('temperature')}, "
               f"seed {llm.get('seed')}, {llm.get('decoding')}" if llm else "") + f". `numbers_verified = {com.get('numbers_verified')}`"
               + (f", `ids_verified = {com.get('ids_verified')}`" if "ids_verified" in com else "") + ".")
    if card:
        out.append(f"Card the model was given: `llm/card.txt`, {len(card)} characters, {len(card.splitlines())} lines.")
    v = com.get("verification")
    if v:
        out.append(f"Guard: {v['n_numbers']} numbers checked; unverified in the first answer {v['n_unverified_raw']}, after the retry {v['n_unverified_final']} "
                   f"(retried: {v['retried']}); id/consistency failures raw {v.get('n_id_failures_raw')}, final {v.get('n_id_failures_final')}.")
    perf = llm.get("perf")
    if perf:
        out.append(f"Generation: {perf['completion_tokens']} tokens, {perf['total_s']} s (prefill {perf['prefill_s']} s, {perf['decode_tok_per_s']} tok/s, {perf['n_threads']} threads)"
                   + (f"; retry: {llm['perf_retry']['completion_tokens']} tokens, {llm['perf_retry']['total_s']} s" if llm.get("perf_retry") else "") + ".")
    q = []
    q.append(f"> **Summary** ({'verified' if com.get('summary_verified', True) else 'UNVERIFIED'}): {com['summary']}")
    for i, fd in enumerate(com["findings"]):
        q.append(f"> **Finding {i + 1}** `{fd['metric_id']}` on {', '.join(fd['tiers'])} ({'verified' if fd.get('verified', True) else 'UNVERIFIED'}): {fd['observation']}  \n"
                 f"> *Likely cause (as written by the model):* {fd['likely_cause']}")
    cv = com.get("caveats_verified", [])
    for i, cav in enumerate(com["caveats"]):
        ok = cv[i] if i < len(cv) else True
        q.append(f"> **Caveat {i + 1}** ({'verified' if ok else 'UNVERIFIED'}): {cav}")
    qv = com.get("questions_verified", [])
    for i, qq in enumerate(com["questions"]):
        ok = qv[i] if i < len(qv) else True
        q.append(f"> **Question {i + 1}** ({'verified' if ok else 'UNVERIFIED'}): {qq}")
    out.append("\n>\n".join(q))
    if com.get("unverified"):
        out.append("Items the guard flagged (`unverified[]` in the JSON):\n\n" + table(["where", "kind", "token", "sentence"],
                   [[u["where"], u["kind"], f"`{u['token']}`", u["sentence"]] for u in com["unverified"]]))
    else:
        out.append("The guard flagged nothing in the numbers.")
    return "\n\n".join(out)


def audit(r):
    """Independent consistency check of the commentary text against compare.json (does not alter the commentary)."""
    com, c = r["com"], r["cmp"]
    if not com:
        return ""
    txt = " ".join([com["summary"]] + [fd["observation"] + " " + fd["likely_cause"] for fd in com["findings"]] + com["caveats"] + com["questions"])
    notes = []
    for t in TIERS:
        st = c["tier_info"][t]["status"]
        if st == "ok" and re.search(rf"\b{t}\b[^.]*\b(fallback|stub|placeholder)\b", txt, re.I):
            notes.append(f"the text calls tier {t} a fallback/stub/placeholder, but its status in compare.json is `{st}`")
    for t in TIERS:
        if c["tier_info"][t]["status"] == "ok" and re.search(rf"\b{t}\b[^.]*not the real (tool|simulation)", txt, re.I):
            notes.append(f"a caveat says tier {t} is \"not the real tool/simulation\", but its status in compare.json is `ok` (the real tool ran; see its `fidelity` block for what was calibrated)")
    for i, fd in enumerate(com["findings"]):
        for m_ in re.finditer(r"(\w+=-?[\d.]+) \([^)]*\) and \1 ", fd["observation"] + " "):
            notes.append(f"finding {i + 1} states `{m_.group(1)}` twice in one sentence")
            break
    adc = c["tier_info"]["C"]["fidelity"]["models"]
    m = re.findall(r"(\d+)-bit", txt)
    cardbits = set(re.findall(r"(\d+)[- ]bit", " ".join(" ".join(c["tier_info"][t]["fidelity"]["models"]) for t in TIERS)))
    for b in m:
        if b not in cardbits:
            notes.append(f"the text mentions a {b}-bit quantity; no tier's `fidelity.models` mentions {b}-bit")
    return notes


def commentary_audit_section(r):
    n = audit(r)
    if not r["com"]:
        return ""
    if not n:
        return "Independent audit against `compare.json` (status words and bit widths only; cannot judge causal claims): nothing found."
    return "Independent audit against `compare.json` (status words and bit widths only):\n\n" + "\n".join(f"- {x}" for x in n)


def fig(rel, alt):
    return f"![{alt}]({rel})"


# ---------------------------------------------------------------- document
def build(r32, r8, rllm, a):
    c = r32["cmp"]
    mm = c["metric_matrix"]
    ref = c.get("reference", {})
    rep32 = jl(ROOT / "models" / "train_report_w32.json")
    rep8 = jl(ROOT / "models" / "train_report.json")
    L = []
    A = L.append
    A("# Results\n")
    A("<!-- GENERATED by scripts/make_results_md.py from compare.json / tier JSONs / build artefacts. Do not edit by hand; rerun the script. -->\n")
    A("Simulation only, on synthetic data. No hardware was measured. All numbers below are read from files by `scripts/make_results_md.py`; "
      "the paragraphs between tables contain no result numbers of their own. Read `docs/honesty.md` before quoting anything.\n")
    A(f"* Headline run: `{r32['dir'].relative_to(ROOT) if r32['dir'].is_relative_to(ROOT) else r32['dir']}` (run id `{c['run_id']}`), model sha256 `{c['model_sha256']}`, dataset sha256 `{c['dataset_sha256']}`.")
    if r8:
        A(f"* Secondary run: `{r8['dir'].name}` (run id `{r8['cmp']['run_id']}`), model sha256 `{r8['cmp']['model_sha256']}`.")
    rm = c.get("run_meta", {})
    A(f"* Cascade flag: {rm.get('cascade')}; C/R subset: {rm.get('cr_subset') or 'none (all examples)'}; test examples per tier: "
      + ", ".join(f"{t}={c['tier_info'][t]['n_examples']}" for t in TIERS) + ".\n")

    A("## 1. Headline: width-32 ternary TDL-Net, all four tiers\n")
    A(headline_table(r32) + "\n")
    A("Status and tool are copied from the tier JSONs. **C** = ngspice 42 on a layer-1 unit cell plus a circuit-calibrated behavioural model for the "
      "4000-example numbers and for layers 2-3 (see section 6); **E** = numpy bit-true fixed point, RTL cross-checked on a subset (section 7). "
      "KL is KL(softmax(V) || softmax(tier)) in nats, per example, averaged.\n")
    A("Accuracy per SNR:\n")
    A(snr_table(r32) + "\n")
    A(fig("figures/fig4_outputs_light.png", "Figure 4: class probabilities of four examples per tier"))
    A("\n*Figure 4. Class probabilities of four test examples for each tier (titles carry the true label; KL to V printed per tier).*\n")
    A(fig("figures/fig6_confusion_light.png", "Figure 6: V confusion matrix and tier minus V"))
    A("\n*Figure 6. V's confusion matrix and (tier minus V) for E, C, R.*\n")
    A(fig("figures/fig7_agreement_light.png", "Figure 7: tier-by-tier top-1 agreement"))
    A("\n*Figure 7. Top-1 agreement, tier by tier.*\n")

    A("## 2. Why C and R sit well below V: the cascade operating point\n")
    ext = r32["ext"] or {}
    fl, nl = ext.get("line", {}).get("fit", {}).get("loss_db_per_us_100mhz"), ext.get("line", {}).get("loss_db_per_us_nominal_100mhz")
    fe = ext.get("line", {}).get("fit", {}).get("fit_max_abs_err")
    casc_note = (f" The fitted loss ({fl:.6f} dB/us) recovers the value that was typed into the netlist ({nl:.1f} dB/us; fit residual max |dH| = {fe:.1e}), because the LTRA line and the closed-form fit are the same RLC line; "
                 "so the cascade demonstrates plumbing and provenance, not independent information. The SPICE `tap_H` samples are not used by R."
                 if fl is not None and nl is not None and fe is not None else "")
    A("With `--cascade`, C is run first and the loss it fitted to its SPICE `.ac` data (the RG-58 figure that was put into the netlist) is handed to R." + casc_note + " "
      "Both tiers therefore run at that loss with **no repeaters**, which the plan itself calls not buildable. R's own default (5 dB/us) is not the headline when `--cascade` is on. "
      "Both tiers carry a `loss_db_per_us` sweep, so the same tiers at lower loss are in the data:\n")
    A(loss_section(r32) + "\n")
    A("Two things follow, and they must be kept apart. (1) At 0 dB/us C and R return to (or near) V's accuracy, so the tiers are not broken, and the 32 dB/us, no-repeater operating point is "
      "one the plan itself calls not buildable. (2) **Most of C's gap is not line loss but a modelling choice**: C's default compensation (`layer`: one ideal gain trim per layer) "
      "scores far lower at the cascade loss than leaving the loss uncompensated (`none`) or using ideal per-tap repeaters (`tap`). `params/circuit_diagnostics.json` holds those two alternative modes; they come from the calibrated "
      "behavioural model, not from SPICE, and the C tier's headline was left at the default `layer` mode (the published numbers are as measured). Whether `layer` or `none` is the more faithful "
      "model of a real build is not decided by these runs.\n")
    A(attribution_section(r32) + "\n")
    A(fig("figures/fig5_sweeps_light.png", "Figure 5: accuracy and mean KL versus the five sweep axes"))
    A("\n*Figure 5. Accuracy and mean KL versus SNR, ADC bits, line loss, isolation and tap amplitude error. R's SNR axis is per-stage.*\n")
    A(fig("figures/fig3_channel_light.png", "Figure 3: tap-path responses, |H(f)| and combiner waveforms"))
    A("\n*Figure 3. Tap-path impulse responses, |H(f)| (C from ngspice `.ac`, R from scikit-rf) and folded combiner waveforms at layer 1 (not a true eye: the sums are multi-level; the R panel shows no eye). Different pulse shapes are used by the two tiers (stated on the figure).*\n")
    A(fig("figures/fig2_traces_light.png", "Figure 2: one example through the layers"))
    A("\n*Figure 2. One example's line voltage after each layer, V/E/C/R overlaid with residual vs V; hollow circles are ngspice samples (layer 1 only).*\n")

    A("## 3. Sweeps (all from the run's `compare.json`)\n")
    A(sweep_tables(r32))
    A("Knees (rule: first value, walking from the best end of the axis, where accuracy is at least 0.05 below its best-end value):\n")
    A(knee_table(r32) + "\n")

    A("## 4. Float versus ternary, and width 8 versus width 32\n")
    A("The float32 reference has the same topology with unconstrained float taps. It is a reference (`is_tier: false`), not a fifth tier: it cannot run through E, C or R.\n")
    A(class_table(rep8, rep32) + "\n")
    A("Source: `models/train_report.json` (width 8) and `models/train_report_w32.json` (width 32), test set of 4000. "
      "The width-8 ternary net, which is the topology the plan specified, **misses the plan's 85 % target at SNR >= 10 dB and collapses the 8PSK class**; "
      "the width-32 ternary net recovers it only partly (per-class row above) and also stays below its float counterpart. QPSK gets worse at width 32 (ternary recall below the width-8 ternary net's), "
      "and the QPSK / 8PSK pair is the main residual error of the headline model (see V's confusion matrix, figure 6). Both models are reported; neither is hidden.\n")
    if r8:
        A("Same two models through the four tiers (`--cascade`, 4000 examples each, accuracy / top-1 agreement vs V / mean KL):\n")
        A(w8_vs_w32_table(r8, r32) + "\n")
        A("Width-8 run in full:\n")
        A(headline_table(r8) + "\n")
        A(snr_table(r8) + "\n")
        A("Width-8 figures (light):\n")
        A(fig("figures/w8/fig6_confusion_light.png", "Figure 6 (width 8): confusion matrices") + "\n")
        A(fig("figures/w8/fig5_sweeps_light.png", "Figure 5 (width 8): sweeps") + "\n")
        A("Width-8 knees:\n")
        A(knee_table(r8) + "\n")

    A("## 5. Timing\n")
    A(timing_table(r32) + "\n")
    if r8:
        A("Width-8 run:\n")
        A(timing_table(r8) + "\n")
    A(f"Load average at the start/end of the headline run (from `run_meta`): {rm.get('load_avg_start')} / {rm.get('load_avg_end')}. "
      "The box has 4 shared cores; see `docs/reproducibility.md`.\n")

    A("## 6. What the circuit tier did and did not compute\n")
    A(c_validation(r32) + "\n")
    A("What this validates: that the closed-form behavioural line + tanh fit reproduce the SPICE layer-1 traces. What it does not: layers 2-3 use the same line "
      "model extrapolated to delays of up to 1.12 microseconds, **never simulated in SPICE**. ngspice's LTRA line has constant R and G = 0, so low-frequency "
      "loss is overstated and the tap delay carries an excess group-delay error (`taps.delay_error_s` in `params/circuit_extracted.json`).\n")

    A("## 7. RTL versus the numpy golden model (tier E)\n")
    A(rtl_section(r32) + "\n")
    if r8:
        A("Width-8 run:\n")
        A(rtl_section(r8) + "\n")
    A("The RTL is run on this subset only (iverilog throughput); the 4000-example numbers and the ADC/SNR sweeps come from the numpy golden model that the RTL was shown equal to.\n")

    A("## 8. The ideal-channel agreement criterion (R at 30 dB)\n")
    A(ideal30_section(r32) + "\n")
    A("This is R with `preset=ideal` (0 dB/us, perfect couplers, no mismatch, NF 0, ideal ADC) and **per-stage** SNR; the plan's criterion text does not say "
      "whether 30 dB is per stage or at the input, see `docs/honesty.md`. It is not R's headline operating point.\n")

    A("## 9. LLM commentary\n")
    A("### 9.1 Evaluation on synthetic cards (`python -m dlnn.llm.eval`, from `models/llm/eval_report.json`)\n")
    A(llm_eval_section() + "\n")
    A("The 8/8 row is the card as designed, which contains a Python-computed list of material divergences. " + llm_ablation_sentence() + "\n")
    A("### 9.2 The commentary of the final run, shown as produced\n")
    if rllm:
        A(commentary_section(rllm, "the LLM run") + "\n")
        A(commentary_verdict(rllm) + "\n")
        A(commentary_audit_section(rllm) + "\n")
        A(fig("figures/fig8_llm_light.png", "Figure 8: LLM commentary panel") + "\n")
        A("*Figure 8. The commentary panel: unverified findings are dimmed, the header shows the number and identifier verification state, and the footer lists the independent audit.*\n")
        A("The commentary file is not edited after generation. Whatever the guard or the audit flags above is the honest result of this run; "
          "it is not a statement about the simulation, and the commentary must not be read as analysis (`docs/llm_commentary.md`).\n")
    else:
        A("(no commentary available)\n")

    A("## 10. Figures 1 and the remaining ones\n")
    A(fig("figures/fig1_network_light.png", "Figure 1: the network drawn from the real weights"))
    A("\n*Figure 1. The network from the real ternary weights of output unit 0: tap polarity markers, tap length in metres (RG-58, 10 ns symbols), loss of the longest tap at the cascade loss.*\n")
    A("Dark variants of every figure are in `docs/figures/*_dark.png`; SVG versions sit next to the PNGs. The files in `docs/figures/` have opaque backgrounds (white for light, the dark surface colour for dark) so that they read on any page.\n")
    return "\n".join(L)


def update_readme_block(r32, path: Path):
    if not path.exists():
        return
    txt = path.read_text()
    b, e = "<!-- headline:begin -->", "<!-- headline:end -->"
    if b not in txt:
        return
    c = r32["cmp"]
    block = (f"{b}\n_Generated by `scripts/make_results_md.py` from `out/{c['run_id']}/compare.json` (width-32 ternary model, all four tiers on all "
             f"{c['tier_info']['V']['n_examples']} test examples, `--cascade`)._\n\n{short_headline(r32)}\n{e}")
    txt = re.sub(re.escape(b) + r".*?" + re.escape(e), lambda m: block, txt, flags=re.S)
    path.write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w32", default="out/final-w32")
    ap.add_argument("--w8", default="out/final-w8")
    ap.add_argument("--llm", default=None, help="run whose llm/commentary.json is shown (default: the --w32 run)")
    ap.add_argument("--out", default="docs/results.md")
    a = ap.parse_args()
    r32 = run_info(ROOT / a.w32)
    if r32 is None:
        raise SystemExit(f"missing {a.w32}/compare.json")
    r8 = run_info(ROOT / a.w8)
    rllm = run_info(ROOT / a.llm) if a.llm else r32
    (ROOT / a.out).write_text(build(r32, r8, rllm, a))
    update_readme_block(r32, ROOT / "docs" / "README.md")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
