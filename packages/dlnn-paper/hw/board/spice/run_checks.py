#!/usr/bin/env python3
"""M1a ngspice acceptance checks for TILE-8 (hand-written netlists; M3 will generate them from KiCad).

Run:  .venv/bin/python hw/board/spice/run_checks.py            (about 1-2 minutes, 1 core)
Writes hw/board/spice/results.json and the decks/data under hw/board/spice/_work/.

Checks
  A  full 56-section line, .ac : tap gains, loss vs tap 0, group delay, tap spacing, ripple; cross-checked against the
     closed-form ABCD recursion in derive_params.ladder_transfer (max deviation reported).
  B  full line, .tran : step overshoot ("ringing") at tap 7 for the datasheet-min-Q inductor, an ASSUMED typical Q, and the
     lossless bound; 1 us pulse (one PAM symbol) response; 20 random-tolerance trials.
  C  crossbar "0" isolation: DNP footprint modelled as C_dnp, summer = TI TLV9062 macro (PSpice model downloaded from ti.com
     at run time, NOT committed).
  D  summer (TLV9062 macro) with 8 populated 10 kohm inputs: gain, bandwidth, 1 us settling.
  E  diff pair as tanh: DC sweep, fit R^2, generic NPN fitted to BC847BS datasheet TABLE points (curves not digitised).
  F  slice: summer -> diff pair -> diff amp -> track/hold, symbol-rate transient vs the ideal tanh.

All models are lumped / generic.  The inductor is DATASHEET-DERIVED from JLCPCB listing values (L, DCR, SRF, Q_min),
the BJT is a GENERIC fitted model, the op-amp is TI's own PSpice macro (TLV9062; TLV9064 is the same die per TI e2e forum).
Nothing here is measured hardware.
"""
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import linelib as ll                                  # noqa: E402
import derive_params as dp                            # noqa: E402

D = dp.DESIGN
IM = dp.inductor_model(D)
MODELS = HERE / "_work" / "models"
TLV_URL = "https://www.ti.com/lit/zip/SBOMAG1E"       # TI 'TLV9062 PSpice model' (Rev E on the product page)
TLV_LIB = MODELS / "tlv9062.lib"
RESULTS = {"note": "ngspice 42 results; lumped, generic/datasheet-derived models; not measured hardware"}


def ensure_tlv():
    if TLV_LIB.exists():
        return True
    MODELS.mkdir(parents=True, exist_ok=True)
    z = MODELS / "SBOMAG1E.zip"
    r = subprocess.run(["curl", "-sS", "-m", "90", "-L", "-o", str(z), TLV_URL], capture_output=True, text=True)
    try:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.lower().endswith("tlv9062.lib"):
                    TLV_LIB.write_bytes(zf.read(n))
        return TLV_LIB.exists()
    except Exception as e:                            # noqa: BLE001
        print("TLV9062 model download failed:", e, r.stderr)
        return False


# ------------------------------------------------------------------------------------------------------ A: AC
def line_deck_ac(L, Rp_target_Q, prefilter=True, dcr=None, Lvals=None, Cvals=None):
    kw = dict(L=D["L_h"], C=D["C_f"], n=D["n_sections"], tap_every=D["tap_every"], dcr=D["ind_dcr_ohm"] if dcr is None else dcr,
              srf=D["ind_srf_hz"], Q=Rp_target_Q, f_q=D["ind_Q_fq_hz"], rs=D["Rs_ohm"], rl=D["Rl_ohm"], prefilter=prefilter,
              rc_r=D["input_rc_R_ohm"], f_rc=1.0 / (2 * math.pi * D["input_rc_R_ohm"] * D["input_rc_C_f"]), c_tap=D["c_tap_load_f"], cend=D["C_end_f"],
              Lvals=Lvals, Cvals=Cvals)
    return kw


def check_A():
    out = {}
    f_ana = None
    for tag, q in (("Qmin", D["ind_Q_min"]), ("Qtyp_assumed", D["ind_Q_assumed_typ"])):
        kw = line_deck_ac(None, q)
        body, taps = ll.line_netlist(**kw)
        vec = " ".join(f"v({t})" for t in taps)
        deck = f"""A line ac {tag}
{body}
.control
set filetype=ascii
ac dec 200 1k 3meg
wrdata ac_{tag}.txt {vec}
.endc
.end
"""
        o, _ = ll.run_ngspice(deck, [f"ac_{tag}.txt"], tag=f"A_{tag}")
        d = o[f"ac_{tag}.txt"]
        f = d[:, 0]
        H = np.array([d[:, 3 * i + 1] + 1j * d[:, 3 * i + 2] for i in range(len(taps))])
        # group delay from a separate LINEAR-frequency sweep (uniform grid, smooth derivative)
        deck_l = deck.replace("ac dec 200 1k 3meg", "ac lin 1500 1k 301k").replace(f"ac_{tag}.txt", f"aclin_{tag}.txt")
        o2, _ = ll.run_ngspice(deck_l, [f"aclin_{tag}.txt"], tag=f"A_{tag}_lin")
        dl = o2[f"aclin_{tag}.txt"]
        fl = dl[:, 0]
        Hl = np.array([dl[:, 3 * i + 1] + 1j * dl[:, 3 * i + 2] for i in range(len(taps))])
        gdl = -np.gradient(np.unwrap(np.angle(Hl), axis=1), 2 * np.pi * fl, axis=1)
        gd, fgd = gdl, fl
        # analytic cross-check at the same frequencies
        rp = IM["Rp_ohm_Qmin"] if tag == "Qmin" else IM["Rp_ohm_Qtyp_assumed"]
        # analytic includes the prefilter as a buffered RC in front: multiply both by same factor => compare ratios H/H0 and absolute with RC
        Hana = dp.ladder_transfer(f, D, rp)
        frc = 1.0 / (2 * math.pi * D["input_rc_R_ohm"] * D["input_rc_C_f"])
        Hana = Hana * (1.0 / (1 + 1j * f / frc))
        dev_db = np.max(np.abs(20 * np.log10(np.abs(H) / np.abs(Hana))))
        dev_ph = np.max(np.abs(np.angle(H / Hana)))
        fsel = lambda x: int(np.argmin(abs(f - x)))                       # noqa: E731
        loss = {f"{int(x)}": [float(20 * np.log10(abs(H[k, fsel(x)] / H[0, fsel(x)]))) for k in range(len(taps))] for x in (1e3, 125e3, 250e3, 500e3, 1e6)}
        fselg = lambda x: int(np.argmin(abs(fgd - x)))                    # noqa: E731
        gdd = {f"{int(x)}": [float(gd[k, fselg(x)]) for k in range(len(taps))] for x in (50e3, 125e3, 250e3)}
        spacing = {x: [gdd[x][k + 1] - gdd[x][k] for k in range(len(taps) - 1)] for x in gdd}
        sel = fgd <= 250e3
        ripple = float(100 * (gd[-1][sel].max() - gd[-1][sel].min()) / gd[-1][sel].mean())
        # per-tap ripple of the spacing (tap k+1 minus tap k) over 20..250 kHz: peak-to-peak / mean
        selr = (fgd >= 20e3) & (fgd <= 250e3)
        spc = np.diff(gd, axis=0)[:, selr]
        spacing_ripple_pct = float(100 * (spc.max(axis=1) - spc.min(axis=1)).max() / spc.mean())
        # spacing deviation across taps at 125 kHz (excluding first interval which contains the source node)
        sp = np.array(spacing["125000"])
        out[tag] = {
            "tap_loss_db_rel_tap0": loss, "tap7_loss_db_at_250kHz": loss["250000"][-1], "tap7_loss_db_at_125kHz": loss["125000"][-1],
            "tap7_loss_db_at_500kHz": loss["500000"][-1], "tap7_loss_db_at_1MHz": loss["1000000"][-1],
            "group_delay_s": gdd, "tap_spacing_s": spacing, "spacing_mean_s": float(sp.mean()),
            "spacing_min_max_s": [float(sp.min()), float(sp.max())], "spacing_spread_pct": float(100 * (sp.max() - sp.min()) / sp.mean()),
            "tap7_group_delay_ripple_pct_0_250kHz": ripple, "worst_spacing_ripple_pct_20_250kHz": spacing_ripple_pct,
            "tap7_abs_gain_db_dc": float(20 * np.log10(abs(H[-1, 0]))),
            "f_3dB_tap7_hz": float(f[np.argmax(np.abs(H[-1]) < np.abs(H[-1, 0]) * 10 ** (-3 / 20))]) if (np.abs(H[-1]) < np.abs(H[-1, 0]) * 10 ** (-3 / 20)).any() else None,
            "analytic_vs_spice_max_dev_db": float(dev_db), "analytic_vs_spice_max_dev_rad": float(dev_ph),
        }
    return out


# ------------------------------------------------------------------------------------------------------ B: TRAN
def tran_run(pulse, kw, tstop=45e-6, tag="tr"):
    body, taps = ll.line_netlist(src=pulse, **kw)
    vec = " ".join(f"v({t})" for t in taps)
    deck = f"""B line tran
{body}
.control
set filetype=ascii
tran 5n {tstop} 0 5n
wrdata {tag}.txt {vec}
.endc
.end
"""
    o, _ = ll.run_ngspice(deck, [f"{tag}.txt"], tag=f"B_{tag}")
    d = o[f"{tag}.txt"]
    t = d[:, 0]
    return t, np.array([d[:, 2 * i + 1] for i in range(len(taps))])


def step_metrics(t, v, t0=1e-6):
    init = v[t < t0 * 0.9].mean()
    fin = v[(t > t[-1] - 4e-6)].mean()
    h = fin - init
    pk = v[t > t0].max()
    over = 100 * (pk - fin) / h
    # 50 % crossing time
    thr = init + 0.5 * h
    i50 = np.argmax((v > thr) & (t > t0))
    t50 = np.interp(thr, [v[i50 - 1], v[i50]], [t[i50 - 1], t[i50]]) - t0
    i90 = np.argmax((v > init + 0.9 * h) & (t > t0))
    t90 = t[i90]
    late = (t > t90 + 3e-6) & (t < t[-1] - 4e-6)
    ring_late = 100 * np.max(np.abs(v[late] - fin)) / h if late.any() else 0.0
    # undershoot after the step (below initial) = ringing before arrival is not expected
    return {"overshoot_pct": float(over), "t50_s": float(t50), "ring_after_t90_plus_3us_pct": float(ring_late), "final": float(fin), "init": float(init)}


def check_B(n_mc=20, seed=1):
    out = {}
    frc = 1.0 / (2 * math.pi * D["input_rc_R_ohm"] * D["input_rc_C_f"])
    base = line_deck_ac(None, D["ind_Q_min"])
    variants = {
        "Qmin": base,
        "Qtyp_assumed": line_deck_ac(None, D["ind_Q_assumed_typ"]),
        "lossless_bound": {**line_deck_ac(None, 1e5, dcr=0.0)},
    }
    step = "PULSE(0 1 1u 50n 50n 80u 200u)"          # 50 ns edge, as in the plan's ringing criterion
    for tag, kw in variants.items():
        t, V = tran_run(step, kw, tag=f"step_{tag}")
        m = [step_metrics(t, V[k]) for k in range(V.shape[0])]
        t50 = [x["t50_s"] for x in m]
        out[tag] = {"tap7": m[-1], "tap0": m[0], "t50_per_tap_s": t50,
                    "t50_spacing_s": [t50[k + 1] - t50[k] for k in range(len(t50) - 1)]}
    # no input filter (the plan's 'ringing 20 %' situation) and the plan's 1.5 MHz RC, for the design-change table
    for tag, frc_x in (("no_prefilter_Qtyp", None), ("rc_1p5MHz_Qtyp", 1.5e6), ("rc_1p0MHz_Qtyp", 1.0e6), ("rc_0p5MHz_Qtyp", 0.482e6), ("rc_1p5MHz_lossless", 1.5e6), ("rc_0p5MHz_lossless", 0.482e6)):
        kw = dict(line_deck_ac(None, D["ind_Q_assumed_typ"] if "lossless" not in tag else 1e5, dcr=None if "lossless" not in tag else 0.0))
        if frc_x is None:
            kw["prefilter"] = False
        else:
            kw["rc_r"] = 1000.0
            kw["f_rc"] = frc_x
        t, V = tran_run(step, kw, tag=f"var_{tag}")
        out.setdefault("design_variants_tap7_overshoot_pct", {})[tag] = step_metrics(t, V[-1])["overshoot_pct"]
    # one PAM symbol (1/fs_cal), tap 7, Qmin: ISI at the neighbouring symbol instants
    tsym = 1.0 / 1.0468e6
    pulse = f"PULSE(0 1 5u 20n 20n {tsym} 200u)"
    t, V = tran_run(pulse, base, tag="sym")
    v = V[-1]
    pk_i = np.argmax(v)
    tpk = t[pk_i]
    samp = [float(np.interp(tpk + j * tsym, t, v)) for j in range(-3, 6)]
    out["symbol_pulse_tap7_Qmin"] = {"peak": float(v.max()), "samples_at_peak_plus_j_Ts_j_minus3_to_5": samp,
                                     "isi_rel_to_peak_pct": [100 * x / v.max() for x in samp]}
    # Monte-Carlo ringing (tolerance) at Qtyp
    rng = np.random.default_rng(seed)
    ov = []
    for i in range(n_mc):
        Lv = D["L_h"] * (1 + rng.uniform(-D["tol_L"], D["tol_L"], D["n_sections"]))
        Cv = D["C_f"] * (1 + rng.uniform(-D["tol_C"], D["tol_C"], D["n_sections"] + 1))
        Cv[0] = D["C_end_f"] * Cv[0] / D["C_f"]; Cv[-1] = D["C_end_f"] * Cv[-1] / D["C_f"]
        kw = line_deck_ac(None, D["ind_Q_assumed_typ"], Lvals=Lv, Cvals=Cv)
        t, V = tran_run(step, kw, tag="mc")
        ov.append(step_metrics(t, V[-1])["overshoot_pct"])
    out["mc_tolerance_overshoot_pct_Qtyp"] = {"n": n_mc, "mean": float(np.mean(ov)), "max": float(np.max(ov)), "min": float(np.min(ov))}
    return out


def check_B2_mc_delay(n=2000, seed=2):
    """Analytic (validated against SPICE in A) Monte Carlo of tap-7 / adjacent-tap group delay at 125 kHz."""
    rng = np.random.default_rng(seed)
    f = np.array([100e3, 125e3, 150e3])
    rp = IM["Rp_ohm_Qmin"]
    d7, d1 = [], []
    for _ in range(n):
        Lv = D["L_h"] * (1 + rng.uniform(-D["tol_L"], D["tol_L"], D["n_sections"]))
        Cv = D["C_f"] * (1 + rng.uniform(-D["tol_C"], D["tol_C"], D["n_sections"] + 1))
        Cv[0] = D["C_end_f"] * Cv[0] / D["C_f"]; Cv[-1] = D["C_end_f"] * Cv[-1] / D["C_f"]
        H = dp.ladder_transfer(f, D, rp, L=Lv, C=Cv)
        g = np.array([dp.group_delay(f, H[k])[1] for k in range(H.shape[0])])
        d7.append(g[-1]); d1.append(g[-1] - g[-2])
    Hn = dp.ladder_transfer(f, D, rp)
    gn = np.array([dp.group_delay(f, Hn[k])[1] for k in range(Hn.shape[0])])
    # end-capacitor finding: full C vs C/2 at both ends (analytic ladder, validated against SPICE in check A)
    ff = np.linspace(20e3, 250e3, 600)
    endcap = {}
    for tag, cend in (("full_C_120pF", D["C_f"]), ("half_C_60pF", D["C_end_f"])):
        Cv = np.full(D["n_sections"] + 1, D["C_f"]); Cv[0] = Cv[-1] = cend
        H = dp.ladder_transfer(ff, D, rp, C=Cv)
        g = np.array([dp.group_delay(ff, H[k]) for k in range(H.shape[0])])
        sp = np.diff(g, axis=0)[1:]               # exclude the source-side interval
        endcap[tag] = {"spacing_ripple_p2p_pct_of_mean_20_250kHz": float(100 * np.max(np.ptp(sp, axis=1)) / sp.mean()),
                       "spacing_mean_ns": float(sp.mean() * 1e9)}
    return {"end_capacitor_comparison": endcap, "n": n, "tap7_delay_nominal_s": float(gn[-1]), "tap7_delay_sigma_ns": float(np.std(d7) * 1e9),
            "tap7_delay_sigma_fraction_of_spacing": float(np.std(d7) / np.mean(np.diff(gn))),
            "last_spacing_sigma_ns": float(np.std(d1) * 1e9), "tap7_delay_p2p_ns": float((np.max(d7) - np.min(d7)) * 1e9)}


# ------------------------------------------------------------------------------------------------------ C/D: crossbar + summer
def summer_deck(kind, rf, rin=None, n_pop=8, cf=None, cdnp=None, extra_cap=10e-12):
    rin = rin or D["R_in_ohm"]
    s = [f".include {TLV_LIB}", "Vcc vcc 0 5", "Vref ref 0 2.5", "Vsig sig 0 DC 2.5 AC 1"]
    if kind == "pop":
        s += [f"Rin0 sig sn {rin}"]
        for i in range(1, n_pop):
            s += [f"Rinx{i} ref sn {rin}"]               # other populated inputs sit at the reference (AC zero)
    elif kind == "dnp":
        s += [f"Cdnp sig sn {cdnp}"]
        for i in range(1, n_pop + 1):
            s += [f"Rinx{i} ref sn {rin}"]
    s += [f"Csn sn 0 {extra_cap}", f"Rf out sn {rf}"]
    if cf:
        s += [f"Cf out sn {cf}"]
    s += ["X1 ref sn vcc 0 out TLV9062", "Cl out 0 10p"]
    return "\n".join(s)


def check_CD():
    if not ensure_tlv():
        return {"status": "tlv9062 model unavailable"}, {"status": "tlv9062 model unavailable"}
    out_c, out_d = {}, {}
    rf = 866.0
    res = {}
    for tag, kw in (("pop", dict(kind="pop")), ("dnp", dict(kind="dnp", cdnp=D["c_dnp_pad_f"]))):
        body = summer_deck(rf=rf, **kw)
        deck = f"""C/D summer {tag}
{body}
.control
set filetype=ascii
ac dec 100 1k 10meg
wrdata cd_{tag}.txt v(out)
.endc
.end
"""
        o, _ = ll.run_ngspice(deck, [f"cd_{tag}.txt"], psa=True, tag=f"CD_{tag}")
        d = o[f"cd_{tag}.txt"]
        res[tag] = (d[:, 0], d[:, 1] + 1j * d[:, 2])
    f, Hp = res["pop"]
    _, Hd = res["dnp"]
    iso = {}
    for x in (125e3, 250e3, 500e3, 1e6, 2e6):
        i = int(np.argmin(abs(f - x)))
        iso[f"{int(x)}"] = float(20 * np.log10(abs(Hd[i]) / abs(Hp[i])))
    out_c = {"isolation_db_vs_frequency_TLV9062_macro": iso, "c_dnp_pF": D["c_dnp_pad_f"] * 1e12, "worst_in_band_0_to_500kHz_db": iso["500000"],
             "criterion_db": -45.0, "pass": bool(iso["500000"] <= -45.0 and iso["1000000"] <= -45.0),
             "analytic_db_at_1MHz": dp.crossbar_isolation_db(D, 1e6),
             "max_c_dnp_pF_for_minus45dB_at_1MHz": 10 ** (-45 / 20) / (2 * math.pi * 1e6 * D["R_in_ohm"]) * 1e12,
             "populated_gain_expected": -rf / D["R_in_ohm"], "populated_gain_spice_dc": float(Hp[0].real)}
    # D: summer bandwidth and settling with Cf
    ds = {}
    for cf in (None, 10e-12, 15e-12, 22e-12):
        body = summer_deck(kind="pop", rf=rf, cf=cf, n_pop=8)
        deck = f"""D summer cf={cf}
{body}
.control
set filetype=ascii
ac dec 100 1k 30meg
wrdata d_ac.txt v(out)
alter Vsig PULSE(2.5 3.0 1u 20n 20n 3u 10u)
tran 2n 6u
wrdata d_tr.txt v(out)
.endc
.end
"""
        # the 'alter' on a source with AC spec: use separate decks for clarity
        deck_ac = deck.split("alter")[0] + ".endc\n.end\n"
        o, _ = ll.run_ngspice(deck_ac, ["d_ac.txt"], psa=True, tag="D_ac")
        d = o["d_ac.txt"]
        f = d[:, 0]; H = d[:, 1] + 1j * d[:, 2]
        g0 = abs(H[0])
        f3 = float(f[np.argmax(abs(H) < g0 / math.sqrt(2))]) if (abs(H) < g0 / math.sqrt(2)).any() else None
        pk = float(20 * np.log10(abs(H).max() / g0))
        # transient
        body_t = body.replace("Vsig sig 0 DC 2.5 AC 1", "Vsig sig 0 PULSE(2.5 3.0 1u 20n 20n 3u 10u)")
        deck_t = f"""D summer tran
{body_t}
.control
set filetype=ascii
tran 2n 6u
wrdata d_tr.txt v(out)
.endc
.end
"""
        o, _ = ll.run_ngspice(deck_t, ["d_tr.txt"], psa=True, tag="D_tr")
        dt = o["d_tr.txt"]
        t, v = dt[:, 0], dt[:, 1]
        v0 = v[t < 0.9e-6].mean()
        vf = v[(t > 3.5e-6) & (t < 4e-6)].mean()
        step = vf - v0
        i = np.argmax((np.abs(v - vf) < 0.001 * abs(step)) & (t > 1e-6) & (np.arange(len(t)) > 0))
        # settle time: last time outside 0.1 % band after the step
        outside = np.where((np.abs(v - vf) > 0.001 * abs(step)) & (t > 1.0e-6) & (t < 4e-6))[0]
        ts = float(t[outside[-1]] - 1.0e-6) if outside.size else 0.0
        ds[str(cf)] = {"dc_gain": float(H[0].real), "f3dB_hz": f3, "peaking_db": pk, "step_gain": float(step / 0.5),
                       "settle_0p1pct_s": ts, "overshoot_pct": float(100 * max(0.0, (vf - v.min()) if step < 0 else (v.max() - vf)) / abs(step))}
    out_d = {"Rf_ohm": rf, "n_populated_inputs": 8, "C_summing_node_pF": 10.0, "by_Cf": ds}
    return out_c, out_d


# ------------------------------------------------------------------------------------------------------ E: diff pair
def npn_model_line(name="qbc847", vbe_on=D["vbe_on_v"], ic_ref=2e-3, vt=0.025852, bf=D["beta_bjt"], vaf=100.0, rb=10.0, tf=0.6e-9,
                   cjc=4.8e-12, cje=13e-12, dis=1.0):
    """GENERIC NPN with IS fitted so that VBE(IC=2 mA) = 0.665 V (BC847BS datasheet VBE(on) typ), BF = 300 (hFE 200-450 @ 2 mA, geometric mean),
    CJC from CCBO = 2 pF @ VCB=10 V (C(0) = 2 pF * (1+10/0.75)^0.33 ~ 4.8 pF), CJE from CEBO = 11 pF @ VEB=0.5 V (C(0)=11*(1+0.5/0.75)^0.33~13 pF),
    TF = 0.6 ns (fT ~ 250 MHz listed for BCM847BS; BC847BS datasheet only gives fT >= 100 MHz).
    NOT from the datasheet (assumed): VAF = 100 V, RB = 10 ohm, MJC/MJE = 0.33, VJ = 0.75."""
    ic_vbe = ic_ref / (bf / (bf + 1))
    is_ = ic_vbe / math.exp(vbe_on / vt) / dis
    return f".model {name} npn (IS={is_:.4e} BF={bf} VAF={vaf} RB={rb} TF={tf} CJE={cje} CJC={cjc} MJE=0.33 MJC=0.33 VJE=0.75 VJC=0.75)"


def diffpair_deck(re=None, tail="resistor", dvbe=0.0, temp=27, vdmax=0.4, step=0.002):
    """tail: 'resistor' (R_E to ground), 'ideal' (ideal current sink) or 'bjt' (BC847B current source, diode-connected Q4 + R_ref mirror)."""
    isq = float(npn_model_line().split("IS=")[1].split()[0])
    q2 = isq * math.exp(dvbe / 0.025852)
    mdl = npn_model_line()
    tp = dp.diffpair_params(D)
    if tail == "ideal":
        tl = f"Iee tail 0 DC {D['I_EE_a']}"
    elif tail == "resistor":
        tl = f"Re tail 0 {tp['R_E_E96_ohm']}"
    else:
        tl = (f"Q3 tail b3 0 qbc847\nQ4 b3 b3 0 qbc847\nRref vcc b3 {tp['tail_bjt']['Rref_E96_ohm']}")
    return f"""E diffpair
{mdl}
.model qbc847b npn (IS={q2:.4e} BF={D['beta_bjt']} VAF=100 RB=10 TF=0.6n CJE=13p CJC=4.8p MJE=0.33 MJC=0.33 VJE=0.75 VJC=0.75)
.temp {temp}
Vcc vcc 0 5
Vcm cm 0 2.5
Vd vd 0 0
Ebp bp cm vd 0 0.5
Ebn bn cm vd 0 -0.5
Q1 c1 bp tail qbc847
Q2 c2 bn tail qbc847b
{tl}
Rl1 vcc c1 {D['R_L_ohm']}
Rl2 vcc c2 {D['R_L_ohm']}
Eo out 0 c2 c1 1
.control
set filetype=ascii
dc Vd -{vdmax} {vdmax} {step}
wrdata dp_dc.txt v(out) i(Vcc)
.endc
.end
"""


def fit_tanh(vd, y):
    from scipy.optimize import curve_fit
    def f(v, A, vs, off, shift):
        return A * np.tanh((v - shift) / vs) + off
    p, _ = curve_fit(f, vd, y, p0=[1.0, 0.05, 0.0, 0.0])
    res = y - f(vd, *p)
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    return {"A_v": float(p[0]), "vsat_mV": float(abs(p[1]) * 1e3), "offset_mV": float(p[2] * 1e3), "input_shift_mV": float(p[3] * 1e3),
            "R2": float(r2), "rms_resid_mV": float(np.sqrt(np.mean(res ** 2)) * 1e3), "max_resid_mV": float(np.max(np.abs(res)) * 1e3)}


def check_E():
    out = {"model": "GENERIC NPN fitted to BC847BS datasheet table points (VBE(on) 665 mV @2 mA, hFE ~300, CCBO 2 pF, CEBO 11 pF); curves not digitised; VAF/RB assumed",
           "model_line": npn_model_line()}
    cases = {"bjt_tail_27C": dict(tail="bjt"), "bjt_tail_0C": dict(tail="bjt", temp=0), "bjt_tail_50C": dict(tail="bjt", temp=50),
             "bjt_tail_dVBE_2mV": dict(tail="bjt", dvbe=2e-3), "ideal_tail_27C": dict(tail="ideal"),
             "resistor_tail_27C": dict(tail="resistor"), "resistor_tail_dVBE_2mV": dict(tail="resistor", dvbe=2e-3)}
    for tag, kw in cases.items():
        o, _ = ll.run_ngspice(diffpair_deck(**kw), ["dp_dc.txt"], tag=f"E_{tag}")
        d = o["dp_dc.txt"]
        vd = d[:, 0]; y = d[:, 1]
        r = fit_tanh(vd, y)
        # reference: ideal tanh with fitted nothing: alpha*I_EE*R_L*tanh(v/(2VT_T))
        T = kw.get("temp", 27) + 273.15
        vt = dp.K_B * T / dp.Q_E
        ideal = D["beta_bjt"] / (D["beta_bjt"] + 1) * D["I_EE_a"] * D["R_L_ohm"] * np.tanh(vd / (2 * vt))
        r["rms_vs_ideal_tanh_mV"] = float(np.sqrt(np.mean((y - ideal) ** 2)) * 1e3)
        r["ideal_vsat_mV"] = float(2 * vt * 1e3)
        r["R2_vs_ideal_tanh"] = float(1 - np.sum((y - ideal) ** 2) / np.sum((y - y.mean()) ** 2))
        out[tag] = r
    # transient: step of 100 mV differential, 10-90 % time of the output
    tdeck = diffpair_deck(tail="bjt").replace("Vd vd 0 0", "Vd vd 0 PULSE(-0.1 0.1 100n 10n 10n 500n 2u)").split(".control")[0] + """.control
set filetype=ascii
tran 0.5n 700n
wrdata dp_tr.txt v(out)
.endc
.end
"""
    o, _ = ll.run_ngspice(tdeck, ["dp_tr.txt"], tag="E_tr")
    d = o["dp_tr.txt"]; t, v = d[:, 0], d[:, 1]
    lo, hi = v[t < 90e-9].mean(), v[(t > 500e-9) & (t < 590e-9)].mean()
    t10 = t[np.argmax((v > lo + 0.1 * (hi - lo)) & (t > 100e-9))]
    t90 = t[np.argmax((v > lo + 0.9 * (hi - lo)) & (t > 100e-9))]
    out["transient_10_90_ns"] = float((t90 - t10) * 1e9)
    return out


# ------------------------------------------------------------------------------------------------------ F: slice
def check_F(cfd=8.2e-12, rfa=None):
    if not ensure_tlv():
        return {"status": "tlv9062 model unavailable"}
    """summer P/N (TLV macro) -> diff pair (generic NPN) -> diff amp (TLV macro, gain 0.5) -> 4066 track/hold (Ron 100) -> hold buffer (TLV macro).
    Inputs: 6 populated paths (+1,+1,+1,-1,-1,+1... weights) driven by 1 us PAM symbols (ideal tap buffers).  Expected y = tanh(u/vsat)."""
    model = dict(np.load(dp.REPO / "models" / "tdlnet_modrec_ternary.npz", allow_pickle=True))
    lp = dp.layer_params(D, model)[1]                         # layer 2 values: R_f = 866 ohm, diff-amp R_f from the derivation
    rfs = lp["Rf_E96_ohm"]
    rfa = rfa or lp["diffamp_Rf_E96_ohm"][0]
    vt = dp.thermal_voltage(D["temp_c"])
    tsym = 1.0 / 1.0e6
    rng = np.random.default_rng(3)
    weights = [1, 1, 1, -1, -1, 1]
    nsym = 24
    sym = rng.uniform(-1, 1, (len(weights), nsym))                # model-unit activations
    A = D["A_act_v"]
    lines = [f".include {TLV_LIB}", npn_model_line(),
             "Vcc vcc 0 5", "Vref ref 0 2.5"]
    for i, (w, s) in enumerate(zip(weights, sym)):
        pts = []
        for k, a in enumerate(s):
            tk = k * tsym
            if k == 0:
                pts += [f"0 {2.5 + A * a:.6g}"]
            pts += [f"{tk + 20e-9:.9g} {2.5 + A * a:.6g}", f"{tk + tsym:.9g} {2.5 + A * a:.6g}"]
        # PWL with 20 ns edges at each symbol boundary
        pwl = " ".join(pts)
        lines.append(f"Vin{i} in{i} 0 PWL({pwl})")
        node = "snp" if w > 0 else "snn"
        # convention in this deck: W=+1 -> node snn (inverting output rises less...)  see board_design.md; here we just make y = +sum(w a)
        node = "snn" if w > 0 else "snp"
        lines.append(f"Rin{i} in{i} {node} {D['R_in_ohm']}")
    lines += [f"Rfp outp snp {rfs}", f"Rfn outn snn {rfs}", f"Cfp outp snp 15p", f"Cfn outn snn 15p",
              "Xsp ref snp vcc 0 outp TLV9062", "Xsn ref snn vcc 0 outn TLV9062",
              "Csp snp 0 10p", "Csn snn 0 10p",
              # diff pair: bases = outp (non-inverting side), outn; with W=+1 on snn the outn falls, outp rises? snn node out = 2.5 - Rf*I -> y positive => outp rises relative to outn
              "Q1 c1 outn tail qbc847", "Q2 c2 outp tail qbc847", "Q3 tail b3 0 qbc847", "Q4 b3 b3 0 qbc847", f"Rref vcc b3 {dp.diffpair_params(D)['tail_bjt']['Rref_E96_ohm']}",
              f"Rl1 vcc c1 {D['R_L_ohm']}", f"Rl2 vcc c2 {D['R_L_ohm']}",
              # difference amp: single TLV, gain 0.5 about 2.5 V : out = 2.5 + 0.5*(c2 - c1)... c2-c1 = Rl*(Ic1-Ic2)
              "Rda1 c1 inpd 20k", f"Rda2 inpd ref {rfa}", "Rda3 c2 invd 20k", f"Rda4 outd invd {rfa}", f"Cfd outd invd {cfd}",
              "Xda inpd invd vcc 0 outd TLV9062",
              "* hold: 4066 switch + 620 pF + buffer",
              "Vclk clk 0 PULSE(0 5 0 5n 5n 450n 1u)",
              f".model SWM VSWITCH(Ron={dp.DESIGN['ron_4066_ohm']} Roff=1e9 Von=3 Voff=2)",
              "S1 outd hold clk 0 SWM", f"Ch hold 0 {dp.sample_hold_params(D)['C_hold_f']}", "Xhb hold outh vcc 0 outh TLV9062", "Clb outh 0 10p"]
    deck = "F slice\n" + "\n".join(lines) + f"""
.control
set filetype=ascii
tran 2n {nsym*tsym}
wrdata f_tr.txt v(outh) v(outd) v(outp) v(outn) v(c1) v(c2) v(snp)
.endc
.end
"""
    o, _ = ll.run_ngspice(deck, ["f_tr.txt"], psa=True, tag="F")
    d = o["f_tr.txt"]
    t = d[:, 0]; outh = d[:, 1]
    # the diff-pair analytic reference with the REALISED R_f and fitted vsat: u_phys = (Rf/Rin) * A * sum(w a);  y = A*tanh(u_phys/(2VT*...))
    u_phys = (rfs / D["R_in_ohm"]) * A * (np.array(weights)[:, None] * sym).sum(axis=0)
    alpha = D["beta_bjt"] / (D["beta_bjt"] + 1)
    expected = 2.5 + 0.5 * alpha * D["I_EE_a"] * D["R_L_ohm"] * np.tanh(u_phys / (2 * vt))
    # sample the hold output at the end of each hold window (0.95 us after the start of the symbol), one symbol later (hold valid next symbol)
    got = np.array([np.interp((k + 1) * tsym - 30e-9 + 1e-6 * 0, t, outh) for k in range(nsym)])
    # S/H samples the diff-amp output at the end of track window (450 ns) of symbol k and holds through symbol k; read at 900 ns
    got = np.array([np.interp(k * tsym + 900e-9, t, outh) for k in range(nsym)])
    skip = 2
    err = got[skip:] - expected[skip:]
    # best-fit gain/offset between got and expected
    pfit = np.polyfit(expected[skip:], got[skip:], 1)
    resid = got[skip:] - np.polyval(pfit, expected[skip:])
    return {"n_symbols": nsym, "weights": weights, "note": "ideal tap sources; TLV9062 macros; generic NPN; 4066 as VSWITCH Ron=118 (TI HC4066 85C max); layer-2 R_f and diff-amp R_f from derive_params", "Rf_ohm": rfs, "diffamp_Rf_ohm": rfa,
            "rms_error_vs_ideal_tanh_mV": float(np.sqrt(np.mean(err ** 2)) * 1e3), "max_error_vs_ideal_tanh_mV": float(np.max(np.abs(err)) * 1e3),
            "full_scale_mV": 500.0, "rms_error_pct_full_scale": float(100 * np.sqrt(np.mean(err ** 2)) / 0.5),
            "fit_gain": float(pfit[0]), "fit_offset_mV": float(pfit[1] * 1e3), "rms_resid_after_gain_offset_mV": float(np.sqrt(np.mean(resid ** 2)) * 1e3),
            "output_range_v": [float(got.min()), float(got.max())]}


def main():
    import time
    t0 = time.time()
    R = RESULTS
    R["A_ac"] = check_A();                    print("A done", round(time.time() - t0))
    R["B_tran"] = check_B();                  print("B done", round(time.time() - t0))
    R["B2_mc_delay"] = check_B2_mc_delay();   print("B2 done", round(time.time() - t0))
    R["C_crossbar_isolation"], R["D_summer"] = check_CD(); print("CD done", round(time.time() - t0))
    R["E_diffpair"] = check_E();              print("E done", round(time.time() - t0))
    R["F_slice"] = check_F();                 print("F done", round(time.time() - t0))
    # acceptance table
    ovs = R["B_tran"]["Qmin"]["tap7"]["overshoot_pct"], R["B_tran"]["Qtyp_assumed"]["tap7"]["overshoot_pct"], R["B_tran"]["lossless_bound"]["tap7"]["overshoot_pct"]
    loss_min = R["A_ac"]["Qmin"]["tap7_loss_db_at_250kHz"]
    loss_typ = R["A_ac"]["Qtyp_assumed"]["tap7_loss_db_at_250kHz"]
    iso = R["C_crossbar_isolation"].get("worst_in_band_0_to_500kHz_db") if isinstance(R["C_crossbar_isolation"], dict) else None
    R["acceptance"] = {
        "ringing_tap7_overshoot_pct": {"Qmin": ovs[0], "Qtyp_assumed": ovs[1], "lossless_bound": ovs[2], "criterion": "<= 10", "pass": bool(max(ovs) <= 10.0)},
        "loss_tap7_vs_tap0_db_at_250kHz": {"Qmin": -loss_min, "Qtyp_assumed": -loss_typ, "criterion": "<= 3", "pass": bool(-loss_min <= 3.0 and -loss_typ <= 3.0)},
        "crossbar_zero_isolation_db_at_500kHz": {"value": iso, "criterion": "<= -45", "pass": bool(iso is not None and iso <= -45)},
        "diffpair_tanh_R2": R["E_diffpair"]["bjt_tail_27C"]["R2"],
        "definitions": {"ringing": "step (50 ns edge) at the line input incl. the input RC; overshoot = (peak - final)/(final - initial) at tap 7",
                        "loss": "|H(tap7)/H(tap0)| in dB at 250 kHz (2x the 125 kBd symbol rate); tap 0 = same node with no line; DCR+Q loss only",
                        "isolation": "summer output with the footprint DNP (pad C) relative to the populated-footprint path, at 500 kHz (Nyquist of 1 MS/s) and 1 MHz"},
    }
    R["runtime_s"] = time.time() - t0
    with open(HERE / "results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(R["acceptance"], indent=1))


if __name__ == "__main__":
    main()
