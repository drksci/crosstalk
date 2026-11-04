#!/usr/bin/env python3
"""derive_params.py : TILE-8 hardware parameters, derived (not typed) -> hw/board/params.json

Usage:  python hw/board/derive_params.py [--model models/tdlnet_modrec_ternary.npz] [--out hw/board/params.json]

Every number in params.json is computed here from (a) DESIGN INPUTS below (each tagged with its source: a vendor
datasheet / JLCPCB listing, a plan choice, or an explicit ASSUMPTION), (b) the model npz (V_sat, gain, bias, weights per
layer) and (c) closed-form / numpy formulas whose sources are cited in each function's docstring.  Time-domain
acceptance numbers (ringing, isolation, diff-pair fit) come from ngspice: hw/board/spice/run_checks.py writes
hw/board/spice/results.json and this script merges it in if present (else acceptance = "not run").

All of this is a LUMPED-ELEMENT design at ~1 MHz.  It is not RF, nothing here is measured, and every vendor-part
number is a datasheet/listing value, not a measurement.  Costs live in cost_model.py (estimate, needs a live quote).

Units are SI (ohm, farad, henry, second, hertz, volt, ampere) unless a key name says otherwise.  Functions assert
unit-consistency where it is cheap to do so (e.g. Z0 = sqrt(L/C) must reproduce 1/(wC) * sqrt(...)).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

K_B = 1.380649e-23        # J/K   (SI 2019 exact)
Q_E = 1.602176634e-19     # C     (SI 2019 exact)
EPS0 = 8.8541878128e-12   # F/m
C0 = 299792458.0

# --------------------------------------------------------------------------------------------------------------
# DESIGN INPUTS.  source tags:  [plan]  docs/phase2_plan.md section 1.1     [JLC]  JLCPCB part page listing
#   [TI]  TI TLV9061/2/4 datasheet (SBOS...; sections "Electrical Characteristics")   [DIO] Diodes Inc. BC847BS (DS30222 Rev 16) / BCM847BS (DS41216 Rev 2)
#   [ASSUME] a design assumption that is NOT from a datasheet and must be checked/measured     [CHOICE] design decision made in M1a
# --------------------------------------------------------------------------------------------------------------
DESIGN = {
    "L_h": 120e-6,                     # [plan] ladder inductance
    "C_f": 120e-12,                    # [plan] C0G 0603 +-5 %
    "C_end_f": 60e-12,                 # [CHOICE] half-value shunt C at both ends of the ladder (2 x 30 pF C0G in parallel; 62 pF not found on JLC): a pi-ladder terminates in C/2,
                                       #   a full C at the ends is a 60 pF shunt discontinuity -> +-40 % tap-spacing zigzag (M1a finding)
    "n_sections": 56,                  # [plan]
    "tap_every": 8,                    # [plan] taps at sections 0,8,...,56
    "n_taps": 8,                       # [plan] K
    "n_lines": 8,                      # [plan]
    "n_outputs": 8,                    # [plan] width 8
    "Rs_ohm": 1000.0,                  # [plan] source termination = Z0
    "Rl_ohm": 1000.0,                  # [plan] end termination = Z0
    "input_rc_R_ohm": 1000.0,          # [CHOICE] changed from the plan's 1.5 MHz; see board_design.md section 5
    "input_rc_C_f": 330e-12,           # [CHOICE] E24, C0G -> 482 kHz
    "tap_buffer_gain": 1.0,
    "c_tap_load_f": 6e-12,             # [TI] Cic 4 pF + [ASSUME] 2 pF pad/trace per tap buffer input
    "tol_L": 0.05,                     # [JLC] FHW1812IF121JST +-5 %
    "tol_C": 0.05,                     # [plan] C0G +-5 %
    "ind_dcr_ohm": 3.3,                # [JLC] C394134 listing (treated as max)
    "ind_srf_hz": 3.3e6,               # [JLC] C394134 listing
    "ind_Q_min": 10.0,                 # [JLC] C394134 listing "10@796kHz" (Q minimum)
    "ind_Q_fq_hz": 796e3,              # [JLC]
    "ind_Q_assumed_typ": 30.0,         # [ASSUME] typical = 3x min for ferrite-core wirewound chips; NOT a datasheet value
    "R_in_ohm": 10e3,                  # [plan] crossbar resistor, 0402 +-1 %
    "A_act_v": 0.5,                    # [plan] layer-to-layer PAM amplitude: 2.5 V +- 0.5 V per +-1 model unit
    "A_in0_v_per_unit": 0.3,           # [CHOICE] layer-1 input: 4.5 model-unit full scale -> +-1.35 V
    "V_cm_v": 2.5,                     # [plan] mid-rail reference
    "Vdd_v": 5.0,                      # [plan]
    "temp_c": 27.0,                    # [ASSUME] nominal; the ngspice default
    "I_EE_a": 1.0e-3,                  # [plan]
    "R_L_ohm": 1000.0,                 # [plan]
    "beta_bjt": 300.0,                 # [DIO] hFE 200..450 @2 mA -> geometric mean
    "vbe_on_v": 0.665,                 # [DIO] BC847BS VBE(on) typ @ VCE=5 V, IC=2 mA
    "vbe_match_v": 2e-3,               # [DIO] BCM847BS |VBE1-VBE2| <= 2 mV @ 2 mA
    "dac_bits": 12,                    # [JLC] MCP4728
    "dac_vref_v": 5.0,                 # [CHOICE] MCP4728 with VDD as reference (gain x1)
    "ron_4066_ohm": 118.0,             # [TI] CD74HC4066 datasheet: R_on(VCC=4.5 V, Vin=VCC..GND) typ 35 / max 95 ohm @25 C, 118 ohm @85 C.  The 74HCT4066 (Nexperia
                                       #   datasheet blocked) is ASSUMED equal; JLC lists 54 ohm for it
    "t_track_s": 450e-9,               # [CHOICE] S/H track window inside 1 symbol
    "opamp_gbw_hz": 10e6,              # [TI] TLV906x
    "opamp_iq_a": 538e-6,              # [TI] per amplifier, VS=5.5 V
    "opamp_en_1khz": 16e-9,            # [TI] V/sqrt(Hz) @1 kHz
    "opamp_en_10khz": 10e-9,           # [TI]
    "opamp_cic_f": 4e-12,              # [TI]
    "opamp_cid_f": 2e-12,              # [TI]
    "c_dnp_pad_f": 0.05e-12,           # [ASSUME] design value for an unpopulated 0402 footprint's pad-to-pad C incl. stray;
                                       #   the coplanar-strip estimate below gives ~0.02 pF (see pad_gap_capacitance)
    "pad_w_m": 0.5e-3, "pad_gap_m": 0.5e-3, "pad_len_m": 0.5e-3, "copper_t_m": 35e-6,   # [ASSUME] IPC-7351 0402 nominal land pattern
    "er_fr4": 4.4,                     # [ASSUME] JLC04161H-7628 prepreg/core Er ~4.4 (JLC stackup page quotes 4.4 at 1 MHz-ish)
    "n_symbol_sps": 8,                 # [plan] 8 samples/symbol
    "fs_nominal_hz": 1.0e6,            # [plan]
    "pitch_m": 3.0e-3,                 # [plan]
    "supply_rail_headroom_v": 0.2,     # [TI] RRIO: output within ~15-20 mV of rail at 10 kohm; 0.2 V keeps margin for 1.2k loads
}

# E-series
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E96 = [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58,
       1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55,
       2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
       4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
       6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76]


def nearest_series(x, series):
    """Nearest standard value (log distance) from an E-series mantissa list."""
    if x <= 0:
        raise ValueError("value must be positive")
    dec = math.floor(math.log10(x))
    best = min(((abs(math.log(x / (m * 10 ** d))), m * 10 ** d) for d in (dec - 1, dec, dec + 1) for m in series))
    return best[1]


def thermal_voltage(temp_c):
    """V_T = k T / q.  [Sze, Physics of Semiconductor Devices; SI-2019 constants]  25.85 mV at 27 C."""
    return K_B * (temp_c + 273.15) / Q_E


# ------------------------------------------------------------------------------------------------ delay line
def line_basic(d):
    """Constant-k LC ladder (low-pass): Z0 = sqrt(L/C), per-section delay tau0 = sqrt(L C), cut-off f_c = 1/(pi sqrt(LC)).
    Source: Pozar, Microwave Engineering 4e, section 2.? 'Periodic structures / artificial transmission lines'
    (Z0, tau0 = sqrt(LC), w_c = 2/sqrt(LC)); Brillouin, Wave Propagation in Periodic Structures (1953).
    Sanity assertion: Z0 * w_c * C = 2 (units check)."""
    L, C = d["L_h"], d["C_f"]
    z0 = math.sqrt(L / C)
    tau0 = math.sqrt(L * C)
    wc = 2.0 / tau0
    assert abs(z0 * wc * C - 2.0) < 1e-9
    n, te = d["n_sections"], d["tap_every"]
    return {
        "Z0_ohm": z0, "tau_section_s": tau0, "f_cutoff_ideal_hz": wc / (2 * math.pi),
        "tap_spacing_ideal_s": te * tau0, "total_delay_ideal_s": n * tau0,
        "n_taps": n // te + 1, "Rs_matches_Z0": abs(d["Rs_ohm"] - z0) / z0 < 0.01 and abs(d["Rl_ohm"] - z0) / z0 < 0.01,
    }


def inductor_model(d):
    """Datasheet-derived inductor model.  C_p from the self-resonance: f_srf = 1/(2 pi sqrt(L Cp)).
    R_p (parallel, frequency independent) such that Q(f_q) = datasheet Q: 1/Q = DCR/(w L) + w L / R_p.
    Source: standard parallel-RLC inductor equivalent circuit (Coilcraft Doc 'Modeling Inductors'; TDK app notes)."""
    L = d["L_h"]
    cp = 1.0 / ((2 * math.pi * d["ind_srf_hz"]) ** 2 * L)
    out = {"Cp_f": cp}
    for tag, q in (("Qmin", d["ind_Q_min"]), ("Qtyp_assumed", d["ind_Q_assumed_typ"])):
        w = 2 * math.pi * d["ind_Q_fq_hz"]
        x = 1.0 / q - d["ind_dcr_ohm"] / (w * L)
        out["Rp_ohm_" + tag] = w * L / x if x > 0 else float("inf")
    return out


def section_impedance(f, d, Rp):
    """Series-arm impedance of one inductor: (DCR + jwL) || Rp || 1/(jwCp)."""
    w = 2 * np.pi * np.asarray(f, float)
    L = d["L_h"]
    cp = 1.0 / ((2 * math.pi * d["ind_srf_hz"]) ** 2 * L)
    zl = d["ind_dcr_ohm"] + 1j * w * L
    y = 1.0 / zl + 1.0 / Rp + 1j * w * cp
    return 1.0 / y


def ladder_transfer(f, d, Rp, c_tap=None, L=None, C=None):
    """Exact node-voltage recursion of the whole ladder (same circuit as the ngspice deck):
    Vsrc -> Rs -> node0 [C]-Zs-node1 [C]- ... node N [C || Rl].  Returns H[k, f] = V(node 8k)/V_buf for k=0..K-1... K taps (n/tap_every+1 nodes).
    Optional per-section L, C arrays (Monte Carlo).  c_tap added at tap nodes."""
    f = np.atleast_1d(np.asarray(f, float))
    w = 2 * np.pi * f
    n, te = d["n_sections"], d["tap_every"]
    ct = d["c_tap_load_f"] if c_tap is None else c_tap
    Lv = np.full(n, d["L_h"]) if L is None else np.asarray(L, float)
    if C is None:
        Cv = np.full(n + 1, d["C_f"])
        Cv[0] = Cv[n] = d["C_end_f"]
    else:
        Cv = np.asarray(C, float)
    cp = 1.0 / ((2 * math.pi * d["ind_srf_hz"]) ** 2 * d["L_h"])
    # backward recursion
    Y = 1j * w * Cv[n] + 1.0 / d["Rl_ohm"] + (1j * w * ct if n % te == 0 else 0)
    ratios = [None] * (n + 1)
    nodeY = [None] * (n + 1)
    nodeY[n] = Y
    for m in range(n, 0, -1):
        zs = 1.0 / (1.0 / (d["ind_dcr_ohm"] + 1j * w * Lv[m - 1]) + 1.0 / Rp + 1j * w * cp)
        zn = 1.0 / nodeY[m]
        ratios[m] = zn / (zs + zn)                       # V_m / V_{m-1}
        Ym1 = 1j * w * Cv[m - 1] + 1.0 / (zs + zn) + (1j * w * ct if (m - 1) % te == 0 else 0)
        nodeY[m - 1] = Ym1
    z0n = 1.0 / nodeY[0]
    v0 = z0n / (d["Rs_ohm"] + z0n)                       # V_0 / V_buf
    v = [v0]
    for m in range(1, n + 1):
        v.append(v[-1] * ratios[m])
    return np.array([v[k * te] for k in range(n // te + 1)])


def group_delay(f, H):
    """tau_g = -d(phi)/d(w), phi unwrapped.  Standard definition (Oppenheim & Schafer)."""
    w = 2 * np.pi * np.asarray(f)
    ph = np.unwrap(np.angle(H))
    return -np.gradient(ph, w)


def line_analysis(d):
    b = line_basic(d)
    im = inductor_model(d)
    f = np.linspace(1e3, 1.0e6, 2000)
    out = {}
    fc_cp = b["f_cutoff_ideal_hz"] / math.sqrt(1 + 4 * im["Cp_f"] / d["C_f"])
    out["f_cutoff_with_Cp_hz"] = fc_cp                       # from cos(theta) = -1 with series arm jwL/(1-w^2 L Cp)
    tau0 = b["tau_section_s"]
    te = d["tap_every"]
    fr = np.array([50e3, 125e3, 250e3, 500e3, 1e6])
    # ideal constant-k group delay of one tap spacing:  tau_g = te * tau0 / sqrt(1-(f/fc)^2)
    out["group_delay_ideal_tap_spacing_s"] = {f"{int(x)}": float(te * tau0 / math.sqrt(1 - (x / b["f_cutoff_ideal_hz"]) ** 2)) for x in fr}
    out["group_delay_increase_at_1MHz_pct_ideal"] = float(100 * (1 / math.sqrt(1 - (1e6 / b["f_cutoff_ideal_hz"]) ** 2) - 1))
    # exact ladder with parasitics, lossless Rp=inf variant and Q-min variant
    res = {}
    for tag, rp in (("Qmin", im["Rp_ohm_Qmin"]), ("Qtyp_assumed", im["Rp_ohm_Qtyp_assumed"])):
        H = ladder_transfer(f, d, rp)
        res[tag] = {"H": H}
    H = res["Qmin"]["H"]
    gd = np.array([group_delay(f, H[k]) for k in range(H.shape[0])])
    sel = f <= 250e3
    taps_gd = {f"{int(x)}": [float(gd[k][np.argmin(abs(f - x))]) for k in range(H.shape[0])] for x in (50e3, 125e3, 250e3)}
    out["tap_group_delay_s_by_freq_exact_Qmin"] = taps_gd
    spacing = {x: [float(taps_gd[x][k + 1] - taps_gd[x][k]) for k in range(len(taps_gd[x]) - 1)] for x in taps_gd}
    out["tap_spacing_s_by_freq_exact_Qmin"] = spacing
    # spacing nominal for the clock = mean spacing at 125 kHz
    out["tap_spacing_nominal_s"] = float(np.mean(spacing["125000"][1:]))   # exclude tap0->tap1 (contains the source-side RC node)
    out["fs_calibrated_hz"] = 1.0 / out["tap_spacing_nominal_s"]
    out["group_delay_ripple_tap7_0_250kHz_pct_exact_Qmin"] = float(100 * (gd[-1][sel].max() - gd[-1][sel].min()) / gd[-1][sel].mean())
    # loss of every tap relative to tap 0 (same node = no line), exact analytic, at reference frequencies
    loss = {}
    for tag in res:
        Hh = res[tag]["H"]
        loss[tag] = {f"{int(x)}": [float(20 * np.log10(abs(Hh[k][np.argmin(abs(f - x))] / Hh[0][np.argmin(abs(f - x))]))) for k in range(Hh.shape[0])]
                     for x in (1e3, 125e3, 250e3, 500e3)}
    out["tap_loss_db_rel_tap0_exact"] = loss
    out["tap7_loss_db_at_250kHz_Qmin"] = loss["Qmin"]["250000"][-1]
    out["tap7_loss_db_at_250kHz_Qtyp_assumed"] = loss["Qtyp_assumed"]["250000"][-1]
    # DC gain of every tap (absolute, V_tap / V_buf):  resistive divider with the cumulated DCR
    dcr = d["ind_dcr_ohm"]
    n = d["n_sections"]
    dc = []
    for k in range(n // te + 1):
        rem = (n - k * te) * dcr + d["Rl_ohm"]
        dc.append(rem / (d["Rs_ohm"] + k * te * dcr + rem))
    out["tap_dc_gain"] = [float(x) for x in dc]
    out["tap_dc_gain_ideal_lossless"] = 0.5
    out["dc_series_resistance_total_ohm"] = n * dcr
    out["dc_loss_tap7_vs_ideal_db"] = float(20 * math.log10(dc[-1] / 0.5))
    # analytic low-loss attenuation estimate per section: alpha = R_eq / (2 Z0), R_eq(f) = DCR + (w L)^2 / Rp
    def alpha_np_per_section(fx, rp):
        w = 2 * math.pi * fx
        return (dcr + (w * d["L_h"]) ** 2 / rp) / (2 * b["Z0_ohm"])
    out["analytic_low_loss_tap7_db_at_250kHz_Qmin"] = float(8.6859 * n * alpha_np_per_section(250e3, im["Rp_ohm_Qmin"]))
    out["reflection_coeff_at_term_pct"] = {f"{int(x)}": float(100 * abs((b["Z0_ohm"] / math.sqrt(max(1e-9, 1 - (x / b["f_cutoff_ideal_hz"]) ** 2)) - b["Z0_ohm"]) /
                                                         (b["Z0_ohm"] / math.sqrt(1 - (x / b["f_cutoff_ideal_hz"]) ** 2) + b["Z0_ohm"]))) for x in (125e3, 250e3, 500e3, 1e6)}
    # tap-buffer loading
    c_tot_line = (d["n_sections"] + 1) * d["C_f"]
    n_tap_nodes = d["n_sections"] // te + 1
    out["tap_buffer_loading"] = {
        "c_per_tap_f": d["c_tap_load_f"],
        "added_C_fraction": n_tap_nodes * d["c_tap_load_f"] / c_tot_line,
        "delay_increase_pct": float(100 * (math.sqrt(1 + n_tap_nodes * d["c_tap_load_f"] / c_tot_line) - 1)),   # tau ~ sqrt(L C)
        "buffer_input_resistance_ohm": ">1e12 (TLV9064 Ib 0.5 pA)",
        "source": "[TI] TLV906x Cic; delay ~ sqrt(L*C)",
    }
    # dispersion from Cp (low-frequency effective inductance L/(1-(f/f_srf)^2))
    out["Leff_increase_pct_at_250kHz"] = float(100 * (1 / (1 - (250e3 / d["ind_srf_hz"]) ** 2) - 1))
    # tolerance: systematic and random delay (tau_total = N sqrt(L C); each section +-tol uniform)
    tl, tc = d["tol_L"], d["tol_C"]
    sig_L, sig_C = tl / math.sqrt(3), tc / math.sqrt(3)           # uniform +-tol -> sigma = tol/sqrt(3)
    sec_sig = 0.5 * math.sqrt(sig_L ** 2 + sig_C ** 2)
    out["delay_tolerance"] = {
        "systematic_worst_case_pct": float(100 * (math.sqrt((1 + tl) * (1 + tc)) - 1)),
        "random_sigma_per_section_pct": float(100 * sec_sig),
        "random_sigma_tap7_ns": float(sec_sig * tau0 * math.sqrt(n) * 1e9),
        "random_sigma_tap7_fraction_of_Ts": float(sec_sig * tau0 * math.sqrt(n) / out["tap_spacing_nominal_s"]),
        "random_sigma_between_adjacent_taps_ns": float(sec_sig * tau0 * math.sqrt(te) * 1e9),
        "source": "tau = N sqrt(LC); uniform +-tol -> sigma=tol/sqrt(3); first-order error propagation (independent sections)",
    }
    out["input_rc"] = {"R_ohm": d["input_rc_R_ohm"], "C_f": d["input_rc_C_f"],
                       "f3db_hz": 1.0 / (2 * math.pi * d["input_rc_R_ohm"] * d["input_rc_C_f"])}
    return b, im, out


# ------------------------------------------------------------------------------------------------ crossbar etc.
def pad_gap_capacitance(d):
    """Pad-to-pad capacitance of an UNPOPULATED 0402 footprint, modelled as two coplanar strips (CPS) of width w
    separated by gap s on FR4, evaluated for a length l (the pad width).  CPS: k = s/(s+2w),
    C' = eps0 * eps_eff * K(k')/K(k), eps_eff ~ (1+er)/2 (substrate/air half-space).
    Source: Simons, Coplanar Waveguide Circuits, Components and Systems (2001), ch.2 (CPS formulas); Gupta et al.,
    Microstrip Lines and Slotlines.  Thickness correction ignored (t/s = 0.07).  This is an ESTIMATE; real boards
    add via/trace stray, so the design value c_dnp_pad_f (0.05 pF) is ~2-3x larger."""
    from scipy.special import ellipk
    w, s, l = d["pad_w_m"], d["pad_gap_m"], d["pad_len_m"]
    k = s / (s + 2 * w)
    kp = math.sqrt(1 - k * k)
    eps_eff = (1 + d["er_fr4"]) / 2
    cprime = EPS0 * eps_eff * ellipk(kp ** 2) / ellipk(k ** 2)
    return cprime * l


def crossbar_isolation_db(d, f):
    """'0' weight (footprint DNP) leakage relative to a populated footprint, summing node at virtual ground:
    I_leak / I_pop = (j w C_dnp) / (1/R_in) -> |.| = 2 pi f C_dnp R_in.   Returned in dB (negative = isolated)."""
    return 20 * math.log10(2 * math.pi * f * d["c_dnp_pad_f"] * d["R_in_ohm"])


def layer_params(d, model):
    """Per-layer analog scaling derived from the model npz.
    Physical chain for output o of layer l (inputs are ZOH PAM about V_cm, a_phys = A_in * a_model):
        diff summer:   v_d = R_f * sum_i W_i a_phys_i / R_in  +  bias term          (W in {-1,0,+1})
        diff pair:     V_out_diff = alpha * I_EE * R_L * tanh(v_d / (2 V_T))
        model:         y = gain_o * tanh(u / vsat),   u = sum W a_model + bias_o
    Matching tanh arguments:  v_d / (2 V_T) = u / vsat  ->   R_f = R_in * 2 V_T / (A_in * vsat).
    Output scaling: y_phys = A_next * gain_o * tanh(.) -> diff-amp gain G_d,o = A_next * gain_o / (alpha I_EE R_L)
    with alpha = beta/(beta+1).   Bias: DAC channel -> R_b -> summing node: (V_dac - V_cm)/R_b = A_in*bias_o/R_in
    -> R_b = R_in * V_range / (A_in * |bias|_max * margin)."""
    vt = thermal_voltage(d["temp_c"])
    alpha = d["beta_bjt"] / (d["beta_bjt"] + 1.0)
    n_layers = int(model["n_layers"])
    layers = []
    for l in range(n_layers):
        W = model[f"L{l}_W"].astype(int)           # [O, I, K]
        vsat = float(model[f"L{l}_vsat"])
        gain = model[f"L{l}_gain"].astype(float)
        bias = model[f"L{l}_bias"].astype(float)
        a_in = d["A_in0_v_per_unit"] if l == 0 else d["A_act_v"]
        a_next = d["A_act_v"]
        amp_in_units = float(model["input_fullscale"]) if l == 0 else 1.0
        rf_exact = d["R_in_ohm"] * 2 * vt / (a_in * vsat)
        rf = nearest_series(rf_exact, E96)
        # effective vsat with the realised R_f
        vsat_eff = d["R_in_ohm"] * 2 * vt / (a_in * rf)
        n_pos = (W == 1).sum(axis=(1, 2))
        n_neg = (W == -1).sum(axis=(1, 2))
        n_nz = n_pos + n_neg
        # worst-case single summer output excursion (all inputs at full scale, same polarity) for the busiest node
        node_i_max = a_in * amp_in_units * np.maximum(n_pos, n_neg) / d["R_in_ohm"]
        v_node_max = rf * node_i_max
        headroom = d["Vdd_v"] / 2 - d["supply_rail_headroom_v"]
        # tap-buffer loading: number of populated resistors on each (line, tap) = sum over outputs of |W|
        loads = (W != 0).sum(axis=0)        # [I, K]
        vd_total_worst = rf * a_in * amp_in_units * n_nz / d["R_in_ohm"]
        bias_max = float(np.max(np.abs(bias)))
        v_range = d["dac_vref_v"] / 2.0
        rb_exact = d["R_in_ohm"] * v_range / (a_in * max(bias_max, 1e-9) * 1.25)
        rb = nearest_series(rb_exact, E96)
        lsb_v = d["dac_vref_v"] / 2 ** d["dac_bits"]
        bias_lsb_units = lsb_v * d["R_in_ohm"] / (rb * a_in)
        ra = 20e3
        # diff-amp input network loads each collector with (Ra + Rfa) ~ 30 kohm: R_L_eff = R_L || (Ra + g Ra)  (one fixed-point iteration)
        gd0 = a_next * gain / (alpha * d["I_EE_a"] * d["R_L_ohm"])
        rl_eff = d["R_L_ohm"] * (ra * (1 + gd0)) / (d["R_L_ohm"] + ra * (1 + gd0))
        gd_exact = a_next * gain / (alpha * d["I_EE_a"] * rl_eff)
        rfa = np.array([nearest_series(g * ra, E96) for g in gd_exact])
        layers.append({
            "layer": l + 1, "n_in_lines": int(W.shape[1]), "n_taps": int(W.shape[2]), "n_out": int(W.shape[0]),
            "vsat_model_units": vsat, "gain_model": gain.tolist(), "bias_model_units": bias.tolist(),
            "A_in_v_per_unit": a_in, "input_amplitude_units": amp_in_units,
            "Rf_exact_ohm": rf_exact, "Rf_E96_ohm": rf, "vsat_realised_over_design": vsat_eff / vsat,
            "nonzero_weights": int(n_nz.sum()), "of_total": int(W.size), "nonzero_per_output": n_nz.tolist(),
            "plus_per_output": n_pos.tolist(), "minus_per_output": n_neg.tolist(),
            "populated_0402": int(n_nz.sum()), "dnp_0402_footprints": int(2 * W.size - n_nz.sum()),
            "worst_node_swing_v_per_side": v_node_max.tolist(), "node_swing_limit_v": headroom,
            "worst_case_node_clips": bool((v_node_max > headroom).any()),
            "worst_case_vd_diff_v_max": float(vd_total_worst.max()),
            "max_load_resistors_per_tap_buffer": int(loads.max()), "mean_load_resistors_per_tap_buffer": float(loads.mean()),
            "tap_buffer_peak_current_a_worst": float(a_in * amp_in_units * loads.max() / d["R_in_ohm"]),
            "bias_max_model_units": bias_max, "Rb_E96_ohm": rb, "bias_lsb_model_units": bias_lsb_units,
            "diffamp_gain_exact": gd_exact.tolist(), "diffamp_Rin_ohm": ra, "diffamp_Rf_E96_ohm": rfa.tolist(),
            "diffamp_gain_realised": (rfa / ra).tolist(), "diffamp_Cf_f": 8.2e-12, "diffamp_Cf_note": "SPICE-tuned (TLV9062 macro slice, spice/results.json F_slice): 0 pF rings, 22 pF too slow",
            "summer_Cf_f": 15e-12, "summer_Cf_note": "SPICE (D_summer): no peaking, f3dB ~7 MHz with 10 pF summing-node C",
            "summer_vsat_phys_mv": 2 * vt * 1e3,
        })
    return layers


def tail_bjt(d):
    """BJT current-mirror tail (replaces the plan's resistor tail; see board_design.md).  Q4 (diode-connected, shared by all 8 pairs of a tile)
    sets V_BE; R_ref from +5 V gives I_ref = (Vdd - V_BE)/R_ref, each pair's tail transistor Q3 (same type, emitter to GND) copies it
    (error ~ 2/beta + Early term).  Because V_BE cancels, I_EE has ~0.04 %/K tempco (supply-limited), unlike a V_b3/R_E tail (0.4 %/K,
    +-8 % amplitude over 0..50 C in ngspice) or the plan's resistor tail (I_EE rises ~10 % when the larger base swings +0.2 V: A = 1.057, R^2 0.99966).
    Source: Gray & Meyer 5e, ch.4 (current mirror); V_BE(I) = VBE(2 mA) - V_T ln(2 mA / I)."""
    vt = thermal_voltage(d["temp_c"])
    vbe = d["vbe_on_v"] - vt * math.log(2e-3 / d["I_EE_a"])
    rref = (d["Vdd_v"] - vbe) / d["I_EE_a"]
    rref_e = nearest_series(rref, E96)
    return {"VBE_ref_v": vbe, "Rref_exact_ohm": rref, "Rref_E96_ohm": rref_e, "Iref_realised_a": (d["Vdd_v"] - vbe) / rref_e,
            "mirror_beta_error_pct": 100 * 2 / d["beta_bjt"], "I_EE_tempco_pct_per_K_est": 0.04,
            "I_EE_supply_sensitivity_pct_per_pct": 1.0}


def diffpair_params(d):
    """BJT emitter-coupled pair as tanh.  Ic1 - Ic2 = alpha I_EE tanh(v_d / (2 V_T)) (Gray & Meyer, Analysis and Design
    of Analog ICs 5e, eq. 3.?/'differential pair large-signal'), V_T = kT/q.  Tail: resistor R_E = (V_cm - V_BE)/I_EE
    with V_BE = VBE(on)(2 mA) - V_T ln(2 mA / (I_EE/2))   (I_C per device = I_EE/2).
    Temperature: the tanh width 2 V_T scales as T (+0.33 %/K): the realised V_sat of every layer drifts together."""
    vt = thermal_voltage(d["temp_c"])
    alpha = d["beta_bjt"] / (d["beta_bjt"] + 1.0)
    vbe = d["vbe_on_v"] - vt * math.log(2e-3 / (d["I_EE_a"] / 2))
    re = (d["V_cm_v"] - vbe) / d["I_EE_a"]
    p = {
        "V_T_mV": vt * 1e3, "vsat_phys_mV": 2 * vt * 1e3, "alpha": alpha,
        "out_swing_diff_v": alpha * d["I_EE_a"] * d["R_L_ohm"],
        "V_BE_at_Ic_V": vbe, "R_E_exact_ohm": re, "R_E_E96_ohm": nearest_series(re, E96),
        "I_EE_realised_a": (d["V_cm_v"] - vbe) / nearest_series(re, E96),
        "vsat_temp_coeff_pct_per_K": 100 / (d["temp_c"] + 273.15),
        "offset_from_VBE_mismatch_mV_max": d["vbe_match_v"] * 1e3,
        "offset_in_vsat_units": d["vbe_match_v"] / (2 * vt),
        "collector_dc_v": d["Vdd_v"] - 0.5 * d["I_EE_a"] * d["R_L_ohm"],
        "p_diffpair_mw": d["Vdd_v"] * d["I_EE_a"] * 1e3,
        "tail_bjt": tail_bjt(d),
        "tail_cm_sensitivity_note": "resistor tail: emitter node moves ~(v_d/2) on large input -> I_EE varies a few % (ngspice DC sweep in spice/results.json quantifies it)",
    }
    return p


def sample_hold_params(d):
    """Track-and-hold: 74HC/HCT4066 R_on + C_hold.  Acquisition error after t_track = exp(-t/(R_on C_h)) (single pole);
    choose C_h so error <= 0.15 % of full step at the worst-case R_on.  kT/C noise = sqrt(k T / C_h) (Johns & Martin, Analog IC Design, sampling noise).
    Hold droop = I_b * T_hold / C_h with I_b = 0.5 pA [TI].  Charge injection Q_inj/C_h with Q_inj ASSUMED 2 pC (not in datasheet reading)."""
    ron = d["ron_4066_ohm"]
    t = d["t_track_s"]
    c_max = t / (math.log(1 / 1.5e-3) * ron)        # error <= 0.15 % at the 85 C worst-case R_on
    # largest E24 value (in pF decades) not exceeding c_max
    E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]   # E12: 620 pF C0G was not found on JLC, 560 pF was
    cands = [m * 10 ** dd for dd in (0, 1, 2, 3, 4) for m in E12 if m * 10 ** dd * 1e-12 <= c_max]
    ch = max(cands) * 1e-12
    tau = ron * ch
    kt_c = math.sqrt(K_B * (d["temp_c"] + 273.15) / ch)
    return {
        "C_hold_max_for_0p15pct_f": c_max, "C_hold_f": ch, "C_hold_value": "C0G 0603",
        "tau_acq_s": tau, "acq_error_pct": 100 * math.exp(-t / tau), "t_track_s": t,
        "t_hold_s": 1.0 / d["fs_nominal_hz"] - t,
        "kTC_noise_uV_rms": kt_c * 1e6,
        "droop_uV_per_symbol": 0.5e-12 * (1.0 / d["fs_nominal_hz"]) / ch * 1e6,
        "charge_injection_mV_assumed_2pC": 2e-12 / ch * 1e3,
        "R_on_ohm_85C_max_TI_HC4066": ron,
    }


def summer_noise(d, layers):
    """Output-referred voltage noise of one summing node, white part only (single-pole closed loop):
    e_out^2 = [ e_n^2 (1 + R_f/R_par)^2 + 4kT R_f + 4kT R_f^2/R_par ... ] * ENBW,  R_par = R_in / n_pop.
    Uses e_n = 16 nV/rtHz (1 kHz, conservative; 10 nV at 10 kHz) [TI].  Bandwidth = the 482 kHz input RC -> ENBW=pi/2 f3.
    Result compared against V_sat(phys) = 2 V_T = 52 mV."""
    out = []
    f3 = 1.0 / (2 * math.pi * d["input_rc_R_ohm"] * d["input_rc_C_f"])
    enbw = math.pi / 2 * f3
    for lp in layers:
        npop = max(1, int(round(np.mean(np.maximum(lp["plus_per_output"], lp["minus_per_output"])))))
        rpar = d["R_in_ohm"] / npop
        rf = lp["Rf_E96_ohm"]
        en = d["opamp_en_1khz"]
        gn = 1 + rf / rpar
        e2 = (en * gn) ** 2 + 4 * K_B * (d["temp_c"] + 273.15) * rf * (1 + rf / rpar)
        out.append({"layer": lp["layer"], "n_pop_per_node_mean": npop, "noise_gain": gn, "e_out_nV_rtHz": math.sqrt(e2) * 1e9,
                    "v_noise_uV_rms_in_ENBW": math.sqrt(e2 * enbw) * 1e6,
                    "ratio_to_vsat": math.sqrt(e2 * enbw) / (2 * thermal_voltage(d["temp_c"]))})
    return out


def power_budget(d, layers):
    """Per-tile analog power (5 V).  Op-amp count per FULL tile: 64 tap buffers + 16 summers (P,N per output) + 8 diff-amps
    + 8 hold buffers + 8 line-input buffers + 2 (Vref buffer, spare) = 106.  I_Q 538 uA each [TI].  Diff pairs: 8 x I_EE.  Crossbar: worst-case sum of
    peak tap-buffer currents is overly pessimistic; we use a duty factor 0.4 (ASSUME: rms/peak of a PAM signal with ~uniform amplitudes
    and 50 % of populated paths), DAC 2 x MCP4728 at 800 uA typ (datasheet)."""
    n_amp = 64 + 16 + 8 + 8 + 8 + 2
    i_amp = n_amp * d["opamp_iq_a"]
    i_dp = d["n_outputs"] * d["I_EE_a"]
    cb = 0.0
    for lp in layers:
        cb = max(cb, lp["populated_0402"] * lp["A_in_v_per_unit"] * lp["input_amplitude_units"] / d["R_in_ohm"] * 0.4)
    i_dac = 2 * 800e-6                       # [MCP4728 datasheet DS22187E] I_DD typ 800 uA each @ VDD=5.5 V (max 1.4 mA)
    per_tile = i_amp + i_dp + cb + i_dac
    # layer-1 tile: 2 lines populated -> 16 tap buffers
    n_amp1 = 16 + 16 + 8 + 8 + 2 + 2
    per_tile1 = n_amp1 * d["opamp_iq_a"] + i_dp + (layers[0]["populated_0402"] * layers[0]["A_in_v_per_unit"] * layers[0]["input_amplitude_units"] / d["R_in_ohm"] * 0.4) + i_dac
    tiles = [per_tile1] + [per_tile] * (len(layers) - 1)
    carrier = 0.03
    tot = sum(tiles) + carrier
    return {"n_opamps_full_tile": n_amp, "i_opamps_a": i_amp, "i_diffpairs_a": i_dp, "i_crossbar_a_est": cb, "i_dac_a_typ": i_dac,
            "i_tile_full_a": per_tile, "i_tile_layer1_a": per_tile1, "i_total_system_a": tot,
            "p_tile_full_w": per_tile * d["Vdd_v"], "p_total_system_w": tot * d["Vdd_v"],
            "ldo_dissipation_w_at_6V_in": tot * (6.0 - d["Vdd_v"]),
            "note": "estimate: op-amp I_Q from datasheet; crossbar duty and DAC current are assumptions"}


def signal_to_noise(d, layers):
    """Quantisation / kTC / noise vs full scale.  S/H kT/C vs A_act, DAC LSB.  Dynamic range in dB = 20 log10(A_act / noise)."""
    sh = sample_hold_params(d)
    return {"kTC_dynamic_range_db": 20 * math.log10(d["A_act_v"] / (sh["kTC_noise_uV_rms"] * 1e-6)),
            "input_dac_lsb_mv": 3300 / 4096.0, "input_dac_lsb_in_model_units": (3.3 / 4096) / d["A_in0_v_per_unit"]}


def connector_pinout():
    """26-pin 2x13 0.1\" ribbon: inter-tile signal bus.  8 signals each with an adjacent GND (rows alternate), power, I2C,
    S/H clock, sync.  The plan's 'partial-sum pairs' (16 pins) CANNOT share this 2x13: moved to a separate 2x8 header
    J_PS (fitted only for W>8 stitching, DNP in the width-8 demonstrator)."""
    pins = {}
    for i in range(8):
        pins[2 * i + 1] = f"SIG{i}"       # odd pins: signal
        pins[2 * i + 2] = "GND"           # even pins: ground next to each signal
    pins.update({17: "+5VA", 18: "GND", 19: "+5VA", 20: "GND", 21: "+3V3", 22: "SDA", 23: "SCL", 24: "SH_CLK", 25: "SYNC", 26: "GND"})
    assert len(pins) == 26 and sorted(pins) == list(range(1, 27))
    jps = {}
    for i in range(8):
        jps[2 * i + 1] = f"PSP{i}"
        jps[2 * i + 2] = f"PSN{i}"
    return {"J_SIG_2x13": {str(k): v for k, v in sorted(pins.items())}, "J_PS_2x8_stitching_only": {str(k): v for k, v in sorted(jps.items())},
            "tile_id": "3 solder jumpers on tile (not on the ribbon); I2C bus muxed per tile on the carrier (TCA9548A-class)",
            "monitor": "line-0 tap-7 test point + SMA-less 1x2 header (MON, GND) on each tile",
            "ribbon_max_length_m": 0.10}


def mutual_inductance_estimate(d):
    """Order-of-magnitude coupling coefficient k between two neighbouring unshielded wirewound chips (ASSUMPTION-heavy).
    Model: each chip is an open-circuit magnetic rod (ferrite drum) of body volume V; to first order its field is a point dipole.
    For two identical dipoles side by side with PARALLEL axes the mutual inductance is M = mu0 m1 m2 / (4 pi p^3 I1 I2) and the
    self-inductance of an open rod is L ~ mu0 m^2/(I^2 V_eff) -> k = M/L ~ V_eff / (4 pi p^3).  We take V_eff = 0.1 * body volume
    (ASSUME: effective magnetic volume of a demagnetisation-limited rod is ~10 % of its body) and a factor 2 for the
    dipole-field orientation (Bz at transverse position vs on-axis): k_par = V_eff / (2 pi p^3).
    Axes ORTHOGONAL (alternating orientation) -> first-order coupling of ideal dipoles is zero; real parts retain ~10 % of it (ASSUME).
    Not a field solve; measure the real parts before trusting (see risks)."""
    v_body = 4.5e-3 * 3.2e-3 * 3.2e-3
    out = {}
    for pitch in (3e-3, 5e-3, 8e-3, 12e-3):
        kk = min(1.0, 0.1 * v_body / (2 * math.pi * pitch ** 3))
        out[f"{pitch*1e3:.0f}mm"] = {"k_parallel_axes": kk, "k_orthogonal_axes_assumed": 0.1 * kk}
    return {"v_body_mm3": v_body * 1e9, "per_pitch": out,
            "formula": "k_par = 0.1 V_body / (2 pi p^3); k_orth = 0.1 k_par (assumed)"}


def forward_xtalk_db(d, k):
    """Forward (co-directional) crosstalk of two identical coupled LC lines with inductive coupling coefficient k and no
    capacitive coupling:  |V_f / V_in| ~ (k / 2) * (w * tau_total) at low frequency  [Cohn / Matthaei-Young-Jones; coupled-line forward
    coupling = (Lm/L - Cm/C)/2 * w * T_d].  Evaluated at 250 kHz, tau_total = N tau0."""
    b = line_basic(d)
    w = 2 * math.pi * 250e3
    td = b["total_delay_ideal_s"]
    x = 0.5 * k * w * td
    return 20 * math.log10(max(x, 1e-12))


def build(model_path):
    d = dict(DESIGN)
    model = dict(np.load(model_path, allow_pickle=True))
    b, im, la = line_analysis(d)
    layers = layer_params(d, model)
    dp = diffpair_params(d)
    sh = sample_hold_params(d)
    xt = mutual_inductance_estimate(d)
    xt["forward_xtalk_db_at_250kHz_parallel_by_pitch"] = {k: forward_xtalk_db(d, v["k_parallel_axes"]) for k, v in xt["per_pitch"].items()}
    xt["forward_xtalk_db_at_250kHz_orthogonal_by_pitch"] = {k: forward_xtalk_db(d, v["k_orthogonal_axes_assumed"]) for k, v in xt["per_pitch"].items()}
    cdnp_est = pad_gap_capacitance(d)
    iso = {
        "c_dnp_design_pf": d["c_dnp_pad_f"] * 1e12,
        "c_dnp_cps_estimate_pf": cdnp_est * 1e12,
        "isolation_db_at": {f"{int(f)}": crossbar_isolation_db(d, f) for f in (125e3, 250e3, 500e3, 1e6)},
        "isolation_db_at_1MHz_cps_estimate": 20 * math.log10(2 * math.pi * 1e6 * cdnp_est * d["R_in_ohm"]),
        "criterion_db": -45.0,
        "formula": "20 log10(2 pi f C_dnp R_in)",
    }
    P = {
        "schema": "dlnn-board-params/1",
        "note": "ESTIMATE / DESIGN values derived by hw/board/derive_params.py; designed, not built, nothing measured",
        "model_path": str(Path(model_path).as_posix()),
        "model_meta": json.loads(str(model.get("meta_json", "{}"))) if "meta_json" in model else {},
        "design_inputs": d,
        "line": {**b, **{k: v for k, v in la.items()}},
        "inductor_model": im,
        "crossbar": {"R_in_ohm": d["R_in_ohm"], "A_act_v": d["A_act_v"], "A_in0_v_per_unit": d["A_in0_v_per_unit"],
                     "footprints_per_tile": 2 * d["n_lines"] * d["n_taps"] * d["n_outputs"], "isolation": iso,
                     "polarity": "W=+1: populate the '+' footprint (-> summing node SN_B, whose inverting output drives the diff pair's base B_N... see board_design.md 4.3); W=-1: '-' footprint; W=0: both DNP"},
        "layers": layers,
        "summer_noise": summer_noise(d, layers),
        "diffpair": dp,
        "sample_hold": sh,
        "snr": signal_to_noise(d, layers),
        "power": power_budget(d, layers),
        "connector": connector_pinout(),
        "crosstalk": xt,
        "tile_counts": {"demonstrator_tiles": len(layers), "inductors_per_full_tile": d["n_lines"] * d["n_sections"],
                        "inductors_layer1_tile": 2 * d["n_sections"], "caps_per_full_tile": d["n_lines"] * (d["n_sections"] + 1)},
    }
    res = HERE / "spice" / "results.json"
    P["acceptance"] = json.loads(res.read_text()) if res.exists() else {"status": "not run: execute hw/board/spice/run_checks.py"}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models" / "tdlnet_modrec_ternary.npz"))
    ap.add_argument("--out", default=str(HERE / "params.json"))
    a = ap.parse_args()
    P = build(a.model)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        return o
    with open(a.out, "w") as f:
        json.dump(clean(P), f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
