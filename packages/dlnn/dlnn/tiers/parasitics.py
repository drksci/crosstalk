"""Tier P (parasitics and tolerance): lumped-element model of the TILE-8 board at about 1 MBd.

NOT RF: at ~1 MHz the board is a lumped-element circuit (wavelength >> board). scikit-rf is used only as an ABCD/
two-port cascade engine for the LC ladders. Designed, not built, nothing measured; see docs/parasitics_model.md.

Chain of one layer (each layer = one tile):
  x[n] (ZOH PAM, clock T)  -> input RC (1 kohm + 330 pF) -> buffer (GBW) -> [ribbon, NEXT]
     -> Rs 1k -> 56-section LC ladder (L+DCR, Cp, Rp, C; mutual inductance to neighbour lines) -> tap k (8 taps) -> tap buffer
     -> crossbar (10 kohm +-1 %, DNP pad leakage) -> inverting summer (R_f error, GBW pole, noise)
     -> BJT diff pair (tanh, V_BE mismatch, temperature) -> diff amp (gain error) -> track-and-hold (acquisition error,
        charge-injection pedestal, phase error) -> next tile.
Everything linear is a frequency response H(f) on a fine grid (40 ns), evaluated at the S/H instants t0 + m*T
(arbitrary fractional positions, direct inverse DFT) giving a symbol-rate MIMO FIR; the FIR is applied with the existing
FFT-convolution engine of tier R (`rf.fir_apply`). The nonlinearity is exactly the model's tanh.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import contract
from ..model.npz import load_model
from . import common, virtual
from .rf import fir_apply

try:  # scikit-rf = ABCD two-port cascade engine
    import skrf
    HAVE_SKRF, SKRF_VERSION = True, skrf.__version__
except Exception:  # pragma: no cover
    skrf, HAVE_SKRF, SKRF_VERSION = None, False, "n/a"

REPO = Path(__file__).resolve().parents[2]
PARAMS_JSON = REPO / "hw" / "board" / "params.json"

# ------------------------------------------------------------------------------------------------ parameters
# Sources: [P] hw/board/params.json (datasheet-derived / design), [A] assumed here (documented in docs/parasitics_model.md)
DEFAULTS = dict(
    # ladder / inductor / capacitor
    L_h=120e-6, C_f=120e-12, C_end_f=60e-12, n_sections=56, tap_every=8, dcr_ohm=3.3, srf_hz=3.3e6,   # [P]
    Q_fq_hz=796e3, Rs_ohm=1000.0, Rl_ohm=1000.0, c_tap_f=6e-12,                                        # [P]
    q_inductor=10.0,        # datasheet minimum (Q >= 10 @ 796 kHz); 30 = assumed typical (corner)       [P]/[A]
    l_tol_pct=5.0, c_tol_pct=5.0,   # half-width of the uniform tolerance band used when a trial is drawn [P] (FHW1812 +-5 %, C0G +-5 %)
    lot_fraction=0.5,       # share of a tile's L/C tolerance budget that is common to the whole tile (same reel) [A]
    # input stage / buffers
    edge_ns=None,           # 10-90 % edge of the input RC; None -> 2.2 * R*C of params (726 ns = 482 kHz)       [P]
    opamp_gbw_hz=10e6,      # TLV906x                                                                       [P]
    gain_trim_db="auto",    # "auto" = M1a normalisation (mean on-lag tap gain of tile 1 line 0, averaged over the Q = 10 and Q = 30 corners, = 1;
                            # the level the board-aware models were trained on, models/board_aware/line_fir_m1a.py); a number = fixed dB. Level correction between the T/H output (A_in*x) and the summer. derive_params.py sizes R_f as if the tap
                            # amplitude were A_in*x, but the matched ladder (Rs = Rl = 1 k) halves it (tap DC gain 0.46-0.54, in params.json).
                            # 6.02 dB = the intended operating point (R_f and R_b doubled / a x2 driver); 0 = the board AS DESIGNED (finding F1)
    # crossbar
    r_tol_pct=1.0, R_in_ohm=10e3, c_dnp_f=50e-15, c_pad_tol_pct=30.0,                                    # [P]/[A]
    # summer / diff pair / diff amp / T-H
    gain_tol_pct=1.0,       # R_f and diff-amp ratio resistor tolerance                                      [P]
    vbe_mismatch_mv=2.0,    # BCM847BS: |dV_BE| <= 2 mV, NOT trimmed away (conservative)                    [P]
    temperature_c=27.0, acq_error=0.0011, pedestal_mv=3.57, th_phase_err_ns=6.0,                          # [P]/[A]
    # clock
    clock_err_pct=0.5,      # residual error of the MCU symbol-period calibration (half-width)               [A]
    # crosstalk
    crosstalk_k=None,       # nearest-neighbour coupling coefficient; None -> from pitch/layout              [P]
    crosstalk_pitch_mm=3.0, crosstalk_layout="alternating",   # "alternating" (orthogonal neighbours) | "parallel"
    k_orth_over_par=0.1,    # [A] board_design.md 'k_orth = 0.1 k_par'
    v_body_mm3=46.08,       # 1812 inductor body                                                             [P]
    # ribbon / connector
    ribbon_c_pf_per_m=30.0, ribbon_len_m=0.1, ribbon_cm_pf_per_m=5.0, ribbon_rout_ohm=50.0,               # [P]/[A]
    # noise / readout
    stage_snr_db=None,      # None -> physical stage noise from params (summer_noise.ratio_to_vsat)         [P]
    adc_bits=12, dac_bits=12,
    cal_method="centroid",  # symbol-period calibration: "centroid" (tap-7 minus tap-0 pulse-response centroid, as M1a line_fir_m1a.py) | "group_delay" (at ~122 kHz)
    # grid
    dt_ns=40.0, nfft=1024, lag_lo=-2, lag_hi=13,
    apply_tol=False,        # draw tolerances (MC). False = nominal parts
    seed=0,
)
IS_TOL = ("l_tol_pct", "c_tol_pct", "r_tol_pct", "gain_tol_pct", "clock_err_pct", "vbe_mismatch_mv", "th_phase_err_ns",
          "c_pad_tol_pct")


def load_board_params(path: str | Path | None = None) -> dict:
    p = Path(path) if path else PARAMS_JSON
    if p.exists():
        return json.loads(p.read_text())
    return {}


def resolve(cfg: dict | None, bp: dict | None = None) -> dict:
    """cfg (user) over DEFAULTS, with board parameters from params.json where present."""
    cfg = dict(cfg or {})
    bp = load_board_params(cfg.get("board_params")) if bp is None else bp
    d = dict(DEFAULTS)
    di = bp.get("design_inputs", {})
    mapping = {"L_h": "L_h", "C_f": "C_f", "C_end_f": "C_end_f", "n_sections": "n_sections", "tap_every": "tap_every",
               "dcr_ohm": "ind_dcr_ohm", "srf_hz": "ind_srf_hz", "Q_fq_hz": "ind_Q_fq_hz", "Rs_ohm": "Rs_ohm",
               "Rl_ohm": "Rl_ohm", "c_tap_f": "c_tap_load_f", "opamp_gbw_hz": "opamp_gbw_hz", "R_in_ohm": "R_in_ohm",
               "c_dnp_f": "c_dnp_pad_f", "temperature_c": "temp_c", "v_body_mm3": None}
    for k, pk in mapping.items():
        if pk and pk in di:
            d[k] = float(di[pk]) if k not in ("n_sections", "tap_every") else int(di[pk])
    d["rc_R"] = float(di.get("input_rc_R_ohm", 1000.0))
    d["rc_C"] = float(di.get("input_rc_C_f", 330e-12))
    d["_have_params"] = bool(bp)
    sh = bp.get("sample_hold", {})
    if "acq_error_pct" in sh:
        d["acq_error"] = float(sh["acq_error_pct"]) / 100.0
    if "charge_injection_mV_assumed_2pC" in sh:
        d["pedestal_mv"] = float(sh["charge_injection_mV_assumed_2pC"])
    d["vT2_mv"] = 2 * 25.8649   # 2 V_T at 27 C
    if "diffpair" in bp:
        d["vT2_mv"] = float(bp["diffpair"].get("vsat_phys_mV", d["vT2_mv"]))
    cb = bp.get("crosstalk", {}).get("v_body_mm3")
    if cb:
        d["v_body_mm3"] = float(cb)
    d["layers_bp"] = bp.get("layers", [])
    d["summer_noise_bp"] = bp.get("summer_noise", [])
    d["input_dac_lsb_units"] = float(bp.get("snr", {}).get("input_dac_lsb_in_model_units", 0.002685546875))
    d.update({k: v for k, v in cfg.items() if k in DEFAULTS or k in ("rc_R", "rc_C")})
    if d["edge_ns"] is None:
        d["edge_ns"] = 2.2 * d["rc_R"] * d["rc_C"] * 1e9
    d["rc_tau_s"] = d["edge_ns"] * 1e-9 / 2.2
    d["T_nom_s"] = 8.0 * float(np.sqrt(d["L_h"] * d["C_f"]))
    d["cp_f"] = 1.0 / ((2 * np.pi * d["srf_hz"]) ** 2 * d["L_h"])
    return d


def inductor_rp(L: float, dcr: float, Q: float, f_q: float) -> float:
    """Parallel R_p reproducing the datasheet Q at f_q given the series DCR (same definition as hw/board/spice/linelib.py)."""
    w = 2 * np.pi * f_q
    x = 1.0 / Q - dcr / (w * L)
    return float(w * L / x) if x > 0 else 1e12


def k_coupling(pc: dict) -> np.ndarray:
    """Coupling-coefficient matrix [8,8] between the inductors of the 8 lines of one tile (section j to section j).

    Dipole estimate of two 1812 inductor bodies side by side, parallel axes (board_design.md / params.json crosstalk):
        k_par(p) = 0.1 * V_body / (2 pi p^3)        (V_body = 46.08 mm^3: 0.027 at 3 mm, -17 dB forward crosstalk @250 kHz)
    Orthogonal axes: k_orth = k_orth_over_par * k_par (assumed 0.1).  'alternating' layout: neighbours (distance 1) are
    orthogonal, distance-2 lines are parallel again at 2p.  'parallel': all lines parallel.  If `crosstalk_k` is given it is
    the nearest-neighbour coefficient and the other entries keep the dipole 1/d^3 and orientation scaling relative to it.
    """
    n, p = 8, float(pc["crosstalk_pitch_mm"])
    kpar = lambda pm: 0.1 * pc["v_body_mm3"] / (2 * np.pi * pm ** 3)  # noqa: E731
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dd = abs(i - j)
            par = (pc["crosstalk_layout"] == "parallel") or (dd % 2 == 0)
            K[i, j] = kpar(p * dd) * (1.0 if par else pc["k_orth_over_par"])
    if pc.get("crosstalk_k") is not None:
        k1 = K[0, 1]
        K = K * (float(pc["crosstalk_k"]) / k1) if k1 > 0 else K
    return K


# ------------------------------------------------------------------------------------------------ ladder solvers
def ladder_taps_numpy(f: np.ndarray, Lv: np.ndarray, Cv: np.ndarray, pc: dict, Lmut: np.ndarray | None = None) -> np.ndarray:
    """Tap transfer V_tap/V_driver for `Lines` independent ladders. Lv [Lines,N] (H, inductor value used in jwL), Cv [Lines,N+1]
    (node shunt C, node 0..N). Backward ABCD-equivalent recursion (node voltage + branch current), vectorised over lines and f.
    Returns [Lines, K, F] with K taps at nodes 0, te, 2te, ..."""
    N, te = int(pc["n_sections"]), int(pc["tap_every"])
    w = 2 * np.pi * np.asarray(f, float)
    jw = 1j * w
    rp0 = inductor_rp(pc["L_h"], pc["dcr_ohm"], pc["q_inductor"], pc["Q_fq_hz"])
    Lines = Lv.shape[0]
    F = len(w)
    V = np.ones((Lines, F), complex)
    I = np.zeros((Lines, F), complex)
    taps = {}
    with np.errstate(all="ignore"):
        for j in range(N, 0, -1):
            Ysh = jw[None, :] * (Cv[:, j, None] + (pc["c_tap_f"] if j % te == 0 else 0.0))
            if j == N:
                Ysh = Ysh + 1.0 / pc["Rl_ohm"]
            if j % te == 0:
                taps[j] = V.copy()
            J = I + Ysh * V
            Yser = 1.0 / (pc["dcr_ohm"] + jw[None, :] * Lv[:, j - 1, None]) + jw[None, :] * pc["cp_f"] \
                + (Lref_ratio(Lv[:, j - 1, None], pc["L_h"]) / rp0)
            V = V + J / Yser
            I = J
        Ysh0 = jw[None, :] * (Cv[:, 0, None] + pc["c_tap_f"])
        taps[0] = V.copy()
        J0 = I + Ysh0 * V
        Vb = V + pc["Rs_ohm"] * J0
        out = np.stack([taps[t * te] / Vb for t in range(N // te + 1)], axis=1)
    return out


def Lref_ratio(Lj, Lref):
    """Rp scales with L (Rp ~ Q w L), so 1/Rp ~ (Lref/L)/Rp0; returned as the multiplier of 1/rp0."""
    return Lref / Lj if np.ndim(Lj) else Lref / Lj


def ladder_taps_skrf(f: np.ndarray, pc: dict, Lval: float | None = None, Cval: float | None = None) -> np.ndarray:
    """Tap transfers of ONE ladder by a scikit-rf ABCD cascade (`**`). Prefix = Rs, C0, (series, shunt)*j ; suffix = the rest
    to the open end (Rl is part of the last shunt branch).  V_tap/V_driver = 1/(A_p + B_p * Y_rest).  Returns [K, F]."""
    N, te = int(pc["n_sections"]), int(pc["tap_every"])
    f = np.asarray(f, float)
    w = 2 * np.pi * f
    jw = 1j * w
    L = pc["L_h"] if Lval is None else Lval
    C = pc["C_f"] if Cval is None else Cval
    rp0 = inductor_rp(pc["L_h"], pc["dcr_ohm"], pc["q_inductor"], pc["Q_fq_hz"])
    freq = skrf.Frequency.from_f(f, unit="hz")
    z0 = 1000.0

    def net(a: np.ndarray):
        n = skrf.Network(frequency=freq, s=np.zeros((len(f), 2, 2), complex), z0=np.full((len(f), 2), z0))
        n.a = a
        return n

    def series(Z):
        a = np.zeros((len(f), 2, 2), complex); a[:, 0, 0] = 1; a[:, 1, 1] = 1; a[:, 0, 1] = Z
        return net(a)

    def shunt(Y):
        a = np.zeros((len(f), 2, 2), complex); a[:, 0, 0] = 1; a[:, 1, 1] = 1; a[:, 1, 0] = Y
        return net(a)

    ser = series(1.0 / (1.0 / (pc["dcr_ohm"] + jw * L + 0j) + jw * pc["cp_f"] + 1.0 / rp0))
    def sh(j):
        Y = jw * ((pc["C_end_f"] if j in (0, N) else C) + (pc["c_tap_f"] if j % te == 0 else 0.0))
        if j == N:
            Y = Y + 1.0 / pc["Rl_ohm"]
        return shunt(Y)
    sections = {j: ser ** sh(j) for j in range(1, N + 1)}
    pre = {0: series(np.full(len(f), pc["Rs_ohm"], complex)) ** sh(0)}
    for j in range(1, N + 1):
        pre[j] = pre[j - 1] ** sections[j]
    suf = {N: None}
    cur = None
    for j in range(N - 1, -1, -1):
        cur = sections[j + 1] if cur is None else sections[j + 1] ** cur
        suf[j] = cur
    out = []
    with np.errstate(all="ignore"):
        for t in range(N // te + 1):
            j = t * te
            P = pre[j].a
            if suf[j] is None:
                Yr = np.zeros(len(f), complex)
            else:
                S = suf[j].a
                Yr = S[:, 1, 0] / S[:, 0, 0]
            out.append(1.0 / (P[:, 0, 0] + P[:, 0, 1] * Yr))
    return np.array(out)


# ------------------------------------------------------------------------------------------------ random draws
@dataclass
class Units:
    """Unit draws u ~ U(-1, 1) for one board; tolerances scale them (common random numbers across sweeps)."""
    uL_lot: np.ndarray; uL: list; uC_lot: np.ndarray; uC: list; uRc: list; uCrc: list
    uRin: list; uPad: list; uRf: list; uDA: list; uVbe: list; uClk: float; uTh: np.ndarray; noise_seed: int


def draw_units(model, seed: int) -> Units:
    rng = np.random.default_rng(seed)
    U = lambda *s: rng.uniform(-1, 1, size=s)  # noqa: E731
    lays = model.layers
    nl = len(lays)
    I = [ly.W.shape[1] for ly in lays]
    O = [ly.W.shape[0] for ly in lays]
    K = [ly.W.shape[2] for ly in lays]
    return Units(uL_lot=U(nl), uL=[U(I[l], 56) for l in range(nl)], uC_lot=U(nl), uC=[U(I[l], 57) for l in range(nl)],
                 uRc=[U(I[l]) for l in range(nl)], uCrc=[U(I[l]) for l in range(nl)],
                 uRin=[U(O[l], I[l], K[l], 2) for l in range(nl)], uPad=[U(O[l], I[l], K[l], 2) for l in range(nl)],
                 uRf=[U(O[l]) for l in range(nl)], uDA=[U(O[l]) for l in range(nl)], uVbe=[U(O[l]) for l in range(nl)],
                 uClk=float(U(1)[0]), uTh=U(nl), noise_seed=int(rng.integers(1 << 31)))


def zero_units(model) -> Units:
    u = draw_units(model, 0)
    z = lambda a: np.zeros_like(a)  # noqa: E731
    return Units(z(u.uL_lot), [z(a) for a in u.uL], z(u.uC_lot), [z(a) for a in u.uC], [z(a) for a in u.uRc],
                 [z(a) for a in u.uCrc], [z(a) for a in u.uRin], [z(a) for a in u.uPad], [z(a) for a in u.uRf],
                 [z(a) for a in u.uDA], [z(a) for a in u.uVbe], 0.0, z(u.uTh), 12345)


# ------------------------------------------------------------------------------------------------ plant (linear part)
@dataclass
class Plant:
    h: list            # per layer float32 [O, I, M]
    mlo: int
    t0: list
    T: float
    info: dict


class Context:
    """Caches the frequency grid and nominal (coupled) ladders for one resolved parameter set."""

    def __init__(self, pc: dict, use_skrf: bool | None = None):
        self.pc = pc
        self.dt = pc["dt_ns"] * 1e-9
        self.N = int(pc["nfft"])
        self.f = np.fft.rfftfreq(self.N, self.dt)
        self.df = 1.0 / (self.N * self.dt)
        self.use_skrf = HAVE_SKRF if use_skrf is None else use_skrf
        self.engine = "skrf-abcd" if self.use_skrf else "numpy-abcd"
        N = pc["n_sections"]
        self._Lnom = np.full((1, N), pc["L_h"])
        Cn = np.full((1, N + 1), pc["C_f"]); Cn[:, 0] = Cn[:, -1] = pc["C_end_f"]
        self._Cnom = Cn
        self._Hnom = None
        self._Tcoup = None

    def H_nominal(self) -> np.ndarray:           # [K, F] uncoupled nominal ladder
        if self._Hnom is None:
            if self.use_skrf:
                self._Hnom = ladder_taps_skrf(self.f, self.pc)
            else:
                self._Hnom = ladder_taps_numpy(self.f, self._Lnom, self._Cnom, self.pc)[0]
        return self._Hnom

    def T_coupled(self) -> np.ndarray | None:    # [8,8,K,F]: line i tap k response to line j drive, nominal parts, modal solution
        K = k_coupling(self.pc)
        if not np.any(K):
            return None
        if self._Tcoup is None:
            lam, U = np.linalg.eigh(K)
            pcs = self.pc
            Lm = np.array([pcs["L_h"] * (1.0 + l) for l in lam])[:, None] * np.ones((1, int(pcs["n_sections"])))
            Cm = np.repeat(self._Cnom, len(lam), axis=0)
            Hm = np.stack([ladder_taps_skrf(self.f, pcs, Lval=Lm[m, 0]) for m in range(len(lam))]) if self.use_skrf \
                else ladder_taps_numpy(self.f, Lm, Cm, pcs)       # [8modes, K, F]
            self._Tcoup = np.einsum("im,mkf,jm->ijkf", U, Hm, U)
        return self._Tcoup


def _tol(pc: dict, key: str, scale_ok: bool) -> float:
    return float(pc[key]) / 100.0 if scale_ok else 0.0


def build_plant(model, ctx: Context, pc: dict, units: Units, cal_dataset_n: int = 0) -> Plant:
    """Linear symbol-rate MIMO FIR kernels for every layer."""
    if any(ly.dilation != 1 for ly in model.layers):
        raise ValueError("tier P models the TILE-8 board: fixed tap spacing, dilation must be 1 in every layer")
    f, df, dt, N = ctx.f, ctx.df, ctx.dt, ctx.N
    w = 2 * np.pi * f
    jw = 1j * w
    tol = bool(pc["apply_tol"])
    tl, tc = _tol(pc, "l_tol_pct", tol), _tol(pc, "c_tol_pct", tol)
    a = float(pc["lot_fraction"])
    Hnom = ctx.H_nominal()
    Tcoup = ctx.T_coupled()
    lags = np.arange(int(pc["lag_lo"]), int(pc["lag_hi"]) + 1)
    M = len(lags)
    nl = len(model.layers)
    wk = np.full(len(f), 2.0); wk[0] = 1.0; wk[-1] = 1.0
    # all toleranced lines of all layers in one solver call
    sizes = [ly.W.shape[1] for ly in model.layers]
    Nsec = int(pc["n_sections"])
    Lall, Call = [], []
    for l in range(nl):
        Lall.append(pc["L_h"] * (1 + tl * (a * units.uL_lot[l] + (1 - a) * units.uL[l])))
        Cl = pc["C_f"] * (1 + tc * (a * units.uC_lot[l] + (1 - a) * units.uC[l]))
        Cl[:, 0] = pc["C_end_f"] * (1 + tc * (a * units.uC_lot[l] + (1 - a) * units.uC[l][:, 0]))
        Cl[:, -1] = pc["C_end_f"] * (1 + tc * (a * units.uC_lot[l] + (1 - a) * units.uC[l][:, -1]))
        Call.append(Cl)
    if tol and (tl > 0 or tc > 0):
        Htol_all = ladder_taps_numpy(f, np.concatenate(Lall), np.concatenate(Call), pc)    # [sumI, K, F]
    else:
        Htol_all = None
    K = Nsec // int(pc["tap_every"]) + 1
    gbw = pc["opamp_gbw_hz"]
    Hbuf = 1.0 / (1 + jw / (2 * np.pi * gbw))                       # unity-gain op-amp pole (tap buffer and driver)
    Hrib = 1.0 / (1 + jw * pc["ribbon_rout_ohm"] * pc["ribbon_c_pf_per_m"] * 1e-12 * pc["ribbon_len_m"])
    Cm = pc["ribbon_cm_pf_per_m"] * 1e-12 * pc["ribbon_len_m"]
    Xt = jw * pc["ribbon_rout_ohm"] * Cm / (1 + jw * pc["ribbon_rout_ohm"] * (Cm + pc["ribbon_c_pf_per_m"] * 1e-12 * pc["ribbon_len_m"]))
    Tclk = pc["T_nom_s"]
    plants, t0s = [], []
    start = 0
    info = {"layers": []}
    T_cal = None
    for l, ly in enumerate(model.layers):
        O, I, Kt = ly.W.shape
        # --- tap transfers of the I driven lines [I,I,K,F] (diag toleranced line + coupling correction)
        if Htol_all is not None:
            Hl = Htol_all[start:start + I]
        else:
            Hl = np.repeat(Hnom[None], I, axis=0)
        start += I
        T = np.zeros((I, I, K, len(f)), complex)
        for i in range(I):
            T[i, i] = Hl[i]
        if Tcoup is not None:
            D = Tcoup[:I, :I].copy()
            for i in range(I):
                D[i, i] = D[i, i] - Hnom
            T = T + D
        T = T * Hbuf[None, None, None, :]                           # tap buffer pole
        # --- input stage: RC (per line, tolerance on R and C0G), driver buffer, ribbon + NEXT (tiles 2..)
        rc_tau = pc["rc_tau_s"] * (1 + _tol(pc, "r_tol_pct", tol) * units.uRc[l][:, None]) * \
            (1 + _tol(pc, "c_tol_pct", tol) * units.uCrc[l][:, None])
        Min = np.zeros((I, I, len(f)), complex)
        for i in range(I):
            Min[i, i] = (1.0 / (1 + jw * rc_tau[i, 0])) * Hbuf * 10.0 ** (float(pc["gain_trim_db"]) / 20.0)
        if l > 0:
            for i in range(I):
                Min[i, i] *= Hrib
                for j in (i - 1, i + 1):
                    if 0 <= j < I:
                        Min[i, j] += Xt                           # NEXT from the adjacent signal wire (captured at wire i's input)
        R = np.einsum("imkf,mjf->ijkf", T, Min)                     # driven line j -> tap k of line i
        # --- calibration (global clock from tile 1 line 0, S/H phase per tile from that tile's line 0)
        if T_cal is None:
            g = R[0, 0]
            if pc["cal_method"] == "group_delay":                   # old definition: group delay at ~122 kHz (centre-of-band)
                ph = np.angle(g[:, 6] * np.conj(g[:, 4]))           # (bins 4,6 ~ 98 / 146 kHz)
                tau = -ph / (2 * np.pi * (f[6] - f[4]))
                T_cal = float((tau[-1] - tau[0]) / (K - 1))
            else:                                                   # "centroid" = definition of models/board_aware/line_fir_m1a.py:
                Tg = pc["T_nom_s"]                                  # T = (centroid(tap 7) - centroid(tap 0))/7 of the pulse response, t < 30 us
                Pg = Tg * np.sinc(f * Tg) * np.exp(-1j * np.pi * f * Tg)
                hh = np.fft.irfft(g * Pg[None, :], n=N, axis=1)
                tt = np.arange(N) * dt
                sel = tt < 30e-6
                cen = (tt[sel] * hh[:, sel]).sum(1) / hh[:, sel].sum(1)
                T_cal = float((cen[-1] - cen[0]) / (K - 1))
            Tclk = T_cal * (1 + (float(pc["clock_err_pct"]) / 100.0 * units.uClk if tol else 0.0))
            info["T_cal_us"] = T_cal * 1e6
            info["T_clk_us"] = Tclk * 1e6
        P = Tclk * np.sinc(f * Tclk) * np.exp(-1j * np.pi * f * Tclk)
        def E_of(t0):
            return (wk * df)[None, :] * np.exp(2j * np.pi * f[None, :] * (t0 + lags[:, None] * Tclk))
        # S/H phase: maximise the on-lag fraction of tap responses of line 0 (the monitor line)
        best, bt0 = -1.0, 0.0
        g0 = R[0, 0] * P[None, :]
        for t0 in np.linspace(0.2 * Tclk, 2.0 * Tclk, 181):
            tt = t0 + np.arange(-3, K + 6)[:, None] * Tclk
            r = np.real(np.exp(2j * np.pi * f[None, None, :] * tt[None]) * (wk * df)[None, None, :] * g0[:, None, :]).sum(-1)  # [K, lagrange]
            on = np.array([r[k, k + 3] for k in range(K)])
            score = float(np.mean(on / (np.abs(r).sum(1) + 1e-30)))
            if score > best:
                best, bt0 = score, t0
        t0 = bt0 + (float(pc["th_phase_err_ns"]) * 1e-9 * units.uTh[l] if tol else 0.0)
        t0s.append(t0)
        E = E_of(t0)
        # --- crossbar coefficients A [O,I,K] (1/R) and B [O,I,K] (pad capacitance * R_in)
        W = ly.W.astype(int)
        eR = _tol(pc, "r_tol_pct", tol) * units.uRin[l]
        cp = pc["c_dnp_f"] * (1 + _tol(pc, "c_pad_tol_pct", tol) * units.uPad[l])
        A = np.zeros(W.shape); B = np.zeros(W.shape)
        for s, sign in ((0, +1), (1, -1)):
            pop = (W == sign)
            A += np.where(pop, sign / (1 + eR[..., s]), 0.0)
            B += np.where(~pop, sign * cp[..., s] * pc["R_in_ohm"], 0.0)
        # --- summer pole
        nbp = pc["summer_noise_bp"]
        ng = float(nbp[l]["noise_gain"]) if l < len(nbp) else 2.0
        Hsum = 1.0 / (1 + jw / (2 * np.pi * gbw / ng))
        Rs_ = np.einsum("oik,ijkf->ojf", A, R) + jw[None, None, :] * np.einsum("oik,ijkf->ojf", B, R)
        Rs_ = Rs_ * (P * Hsum)[None, None, :]
        h = np.real(np.einsum("mf,ojf->ojm", E, Rs_)).astype(np.float32)
        plants.append(h)
        # diagnostics: energy beyond the lag window, DC tap gain
        info["layers"].append({"on_lag_gain": [float(np.real(np.sum(g0[k] * np.exp(2j * np.pi * f * (bt0 + (k + 0) * Tclk)) * (wk * df)))) for k in range(K)],
                               "dc_tap_gain": [float(np.abs(R[0, 0, k, 0])) for k in range(K)],
                               "on_lag_score": best, "t0_us": t0 * 1e6})
    return Plant(h=plants, mlo=int(lags[0]), t0=t0s, T=Tclk, info=info)


class _Ch:
    def __init__(self, h, mlo): self.h, self.mlo = h, mlo


# ------------------------------------------------------------------------------------------------ forward
def ref_z_power(model, x, n=256) -> list:
    h = np.asarray(x[:n], np.float32) * np.float32(model.input_scale)
    out = []
    for ly in model.layers:
        z = (virtual.layer_preact(h, ly) + ly.bias[None, :, None]) / np.float32(ly.vsat)
        out.append(float(np.mean(z ** 2)))
        h = virtual.layer_forward(h, ly)
    return out


def forward_plant(model, plant: Plant, pc: dict, units: Units, x: np.ndarray, p_ref: list) -> dict:
    tol = bool(pc["apply_tol"])
    lays = []
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    lsb = pc["input_dac_lsb_units"] if pc["dac_bits"] else 0.0
    if lsb:
        h = np.round(h / lsb).astype(np.float32) * np.float32(lsb)
    rng = np.random.default_rng(units.noise_seed)
    Tfac = (273.15 + 27.0) / (273.15 + float(pc["temperature_c"]))         # 2 V_T ~ absolute temperature
    for l, ly in enumerate(model.layers):
        bp = pc["layers_bp"]
        r_vs = float(bp[l]["vsat_realised_over_design"]) if l < len(bp) else 1.0
        dar = (float(bp[l]["diffamp_gain_realised"][0]) / float(bp[l]["diffamp_gain_exact"][0])) if l < len(bp) else 1.0
        S = fir_apply(h, _Ch(plant.h[l], plant.mlo))                       # sum over taps/lines, model units
        O = S.shape[1]
        b = ly.bias.astype(np.float32)
        if lsb and l < len(bp) and "bias_lsb_model_units" in bp[l]:
            lb = float(bp[l]["bias_lsb_model_units"]); b = np.round(b / lb).astype(np.float32) * np.float32(lb)
        gt = _tol(pc, "gain_tol_pct", tol)
        eRf = 1.0 + gt * units.uRf[l]
        z = (S + b[None, :, None]) * (eRf[None, :, None] * Tfac / (ly.vsat * r_vs))
        z = z + (pc["vbe_mismatch_mv"] / pc["vT2_mv"]) * units.uVbe[l][None, :, None] * (1.0 if True else 0.0) \
            if pc["vbe_mismatch_mv"] and tol else z
        # stage noise (in z units)
        if pc["stage_snr_db"] is not None:
            sig = np.sqrt(p_ref[l] * 10.0 ** (-float(pc["stage_snr_db"]) / 10.0))
        else:
            nb = pc["summer_noise_bp"]
            sig = float(nb[l]["ratio_to_vsat"]) if l < len(nb) else 5e-4
        z = z + np.float32(sig) * rng.standard_normal(z.shape).astype(np.float32)
        y = np.tanh(z)
        y = y * (1.0 + 2 * gt * units.uDA[l])[None, :, None].astype(np.float32) * np.float32(dar) if tol else y * np.float32(dar)
        y = ly.gain[None, :, None] * y
        # track-and-hold: incomplete acquisition (memory of the previous held value) + charge-injection pedestal
        e = np.float32(pc["acq_error"])
        yh = y * (1 - e)
        yh[:, :, 1:] += e * y[:, :, :-1]
        yh = yh + np.float32(pc["pedestal_mv"] * 1e-3 / 0.5)
        lays.append(yh.astype(np.float32))
        h = yh.astype(np.float32)
    f = np.concatenate([lays[i] for i in model.readout_from], axis=1)
    if pc["adc_bits"]:
        lsb_a = 3.3 / 2 ** int(pc["adc_bits"]) / 0.5
        f = (np.clip(np.round((f * 0.5 + 2.5) / (3.3 / 2 ** int(pc["adc_bits"]))), 0, 2 ** int(pc["adc_bits"]) - 1)
             * (3.3 / 2 ** int(pc["adc_bits"])) - 2.5) / 0.5
        f = f.astype(np.float32)
    if model.readout_kind == "pooled":
        f = f.mean(axis=2)
    return {"out": virtual.readout(model, f), "layers": lays}


class Simulator:
    """Holds model, resolved cfg, caches of Context and calibration. `evaluate(over, units)` -> forward dict."""

    def __init__(self, model, x_cal: np.ndarray, cfg: dict | None = None, use_skrf: bool | None = None):
        self.model = model
        self.cfg = dict(cfg or {})
        self.pc0 = resolve(self.cfg)
        self.ctxs: dict = {}
        self.use_skrf = use_skrf
        self.p_ref = ref_z_power(model, x_cal)
        self._plants: dict = {}

    def pc(self, over: dict | None = None) -> dict:
        pc = resolve({**self.cfg, **(over or {})}, None) if over else self.pc0
        if isinstance(pc["gain_trim_db"], str):
            pc = {**pc, "gain_trim_db": self.auto_trim_db(pc)}
        return pc

    def auto_trim_db(self, pc: dict) -> float:
        """Trim (dB) that makes the mean on-lag tap gain of tile 1 line 0 equal 1 averaged over the Q = 10 / 30 corners (M1a convention).
        Computed once per simulator on nominal parts and the base (non-swept) line, then held fixed (R_f is a resistor: sweeping Q changes the loss, not the trim)."""
        if getattr(self, "_auto_trim", None) is None:
            g = []
            for q in (10.0, 30.0):
                c = {**pc, "gain_trim_db": 0.0, "q_inductor": q, "apply_tol": False}
                pl = build_plant(self.model, Context(c, self.use_skrf), c, zero_units(self.model))
                g.append(float(np.mean(pl.info["layers"][0]["on_lag_gain"])))
            self._auto_trim = float(-20 * np.log10(np.mean(g)))
        return self._auto_trim

    def ctx(self, pc: dict) -> Context:
        key = tuple(pc[k] for k in ("q_inductor", "crosstalk_k", "crosstalk_pitch_mm", "crosstalk_layout", "L_h", "C_f",
                                    "dcr_ohm", "srf_hz", "n_sections", "c_tap_f", "dt_ns", "nfft"))
        if key not in self.ctxs:
            if len(self.ctxs) > 6:
                self.ctxs.pop(next(iter(self.ctxs)))
            nom = all(pc[k] == 0 for k in ())  # engine choice: skrf for nominal-grid builds
            self.ctxs[key] = Context(pc, self.use_skrf)
        return self.ctxs[key]

    def plant(self, pc: dict, units: Units, tag=None) -> Plant:
        keys = ("q_inductor", "crosstalk_k", "crosstalk_pitch_mm", "crosstalk_layout", "edge_ns", "clock_err_pct", "l_tol_pct",
                "c_tol_pct", "r_tol_pct", "c_pad_tol_pct", "th_phase_err_ns", "apply_tol", "gain_trim_db", "opamp_gbw_hz",
                "lot_fraction", "ribbon_len_m", "cal_method")
        key = (tag,) + tuple(pc[k] for k in keys)
        if tag is not None and key in self._plants:
            return self._plants[key]
        pl = build_plant(self.model, self.ctx(pc), pc, units)
        if tag is not None:
            if len(self._plants) > 40:
                self._plants.pop(next(iter(self._plants)))
            self._plants[key] = pl
        return pl

    def forward(self, x, over: dict | None = None, units: Units | None = None, tag=None) -> dict:
        pc = self.pc(over)
        units = units if units is not None else zero_units(self.model)
        pl = self.plant(pc, units, tag)
        return forward_plant(self.model, pl, pc, units, x, self.p_ref)


# ------------------------------------------------------------------------------------------------ Monte Carlo
def stratified(ds: dict, n: int) -> np.ndarray:
    from ..pipeline import stratified_subset
    return stratified_subset(ds["y_test"], ds["snr_test"], n)


def _stats(acc: np.ndarray, thr: float) -> dict:
    return {"n_trials": int(len(acc)), "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "accuracy_p5": float(np.percentile(acc, 5)), "accuracy_min": float(acc.min()), "accuracy_max": float(acc.max()),
            "yield_threshold": float(thr), "yield": float(np.mean(acc >= thr))}


def monte_carlo(sim: Simulator, x: np.ndarray, y: np.ndarray, v_pred: np.ndarray, n_trials: int, seed: int = 1000,
                over: dict | None = None, yield_drop: float = 0.05) -> dict:
    over = {**(over or {}), "apply_tol": True}
    acc, agr = np.zeros(n_trials), np.zeros(n_trials)
    v_acc = float(np.mean(v_pred == y))
    t0 = time.time()
    for t in range(n_trials):
        u = draw_units(sim.model, seed + t)
        o = sim.forward(x, over, u)["out"]
        p = o.argmax(1)
        acc[t], agr[t] = np.mean(p == y), np.mean(p == v_pred)
    st = _stats(acc, v_acc - yield_drop)
    st.update({"v_accuracy_same_subset": v_acc, "agreement_mean": float(agr.mean()), "runtime_s": time.time() - t0,
               "n_examples": int(len(y)), "seed0": seed})
    return {"stats": st, "accuracy": acc, "agreement": agr}


# ------------------------------------------------------------------------------------------------ sweeps
SWEEP_PLAN = {   # axis: (values, cfg key, boards to average (random axes), notes)
    "l_tol_pct": ([0, 2.5, 5, 7.5, 10, 15], "l_tol_pct", True),
    "c_tol_pct": ([0, 2.5, 5, 7.5, 10, 15], "c_tol_pct", True),
    "clock_err_pct": ([0, 0.5, 1, 2, 4, 8], "clock_err_pct", True),
    "stage_snr_db": ([60, 50, 40, 30, 20, 10], "stage_snr_db", False),
    "edge_ns": ([100, 250, 400, 726, 1000, 1500], "edge_ns", False),
    "crosstalk_k": ([0, 0.0027, 0.0068, 0.0136, 0.0272, 0.05], "crosstalk_k", False),
    "vbe_mismatch_mv": ([0, 1, 2, 4, 8, 16], "vbe_mismatch_mv", True),
    "q_inductor": ([10, 15, 20, 30, 60, 100], "q_inductor", False),
    "temperature_c": ([-20, 0, 27, 50, 70, 85], "temperature_c", False),
    "crosstalk_pitch_mm": ([3, 4, 5, 8, 12], "crosstalk_pitch_mm", False),
    "gain_trim_db": ([0, 2, 4, 6.0206, 8, 10], "gain_trim_db", False),
}
SWEEP_LAYOUT = {"crosstalk_pitch_mm": "parallel"}   # parallel-axes worst case; the alternating layout is stored as sweep.*_alt arrays


def run_sweeps(sim: Simulator, x, y, v_out, boards: int = 4, only=None, extra: dict | None = None) -> tuple[list, dict]:
    v_prob = common.softmax(v_out)
    vpred = v_out.argmax(1)
    entries, arrays = [], {}

    def point(over, rand):
        accs, kls, ags = [], [], []
        for b in range(boards if rand else 1):
            u = draw_units(sim.model, 5000 + b) if rand else zero_units(sim.model)
            o = sim.forward(x, {**over, "apply_tol": bool(rand)}, u)["out"]
            accs.append(np.mean(o.argmax(1) == y)); ags.append(np.mean(o.argmax(1) == vpred))
            kls.append(common.mean_kl(v_prob, common.softmax(o)))
        return float(np.mean(accs)), float(np.mean(kls)), float(np.mean(ags))

    for axis, (vals, key, rand) in SWEEP_PLAN.items():
        if only and axis not in only:
            continue
        base = dict(extra or {})
        if axis in SWEEP_LAYOUT:
            base["crosstalk_layout"] = SWEEP_LAYOUT[axis]
        rows = [point({**base, key: v}, rand) for v in vals]
        entries.append({"axis": axis, "values": [float(v) for v in vals], "accuracy": [r[0] for r in rows],
                        "mean_kl": [r[1] for r in rows]})
        arrays[f"sweep.{axis}.values"] = np.array(vals, float)
        arrays[f"sweep.{axis}.top1_agreement_vs_virtual"] = np.array([r[2] for r in rows])
        if axis == "crosstalk_pitch_mm":
            rows2 = [point({"crosstalk_pitch_mm": v, "crosstalk_layout": "alternating"}, False) for v in vals]
            arrays["sweep.crosstalk_pitch_mm_alt.values"] = np.array(vals, float)
            arrays["sweep.crosstalk_pitch_mm_alt.accuracy"] = np.array([r[0] for r in rows2])
            arrays["sweep.crosstalk_pitch_mm_alt.top1_agreement_vs_virtual"] = np.array([r[2] for r in rows2])
    return entries, arrays


# ------------------------------------------------------------------------------------------------ grid
GRID_AXES = [
    ("l_tol_pct", [0, 2.5, 5, 7.5, 10], "% (half-width, uniform)"),
    ("clock_err_pct", [0, 0.5, 1, 2, 4], "% (half-width)"),
    ("stage_snr_db", [60, 50, 40, 30, 20], "dB"),
    ("crosstalk_k", [0, 0.0027, 0.0068, 0.0136, 0.0272], "coupling coefficient (nearest neighbour)"),
    ("vbe_mismatch_mv", [0, 1, 2, 4, 8], "mV (half-width)"),
]


def run_grid(sim: Simulator, x, y, v_out, board_seed: int = 777, axes=None, progress=None) -> dict:
    """accuracy / agreement-vs-V on the full Cartesian grid of `axes` (C order, last axis fastest).
    One fixed 'golden' board (unit draws fixed, tolerances scale them), so the surface is smooth in each tolerance."""
    axes = axes or GRID_AXES
    names = [a[0] for a in axes]
    vals = [a[1] for a in axes]
    shape = [len(v) for v in vals]
    acc = np.zeros(shape); agr = np.zeros(shape)
    vpred = v_out.argmax(1)
    u = draw_units(sim.model, board_seed)
    import itertools
    idx_axes = [names.index(n) for n in names]
    # order loops so the expensive plant keys (l_tol, clock, crosstalk) are outermost; snr, vbe inner
    plant_ax = [i for i, n in enumerate(names) if n in ("l_tol_pct", "clock_err_pct", "crosstalk_k", "c_tol_pct", "edge_ns", "q_inductor")]
    inner_ax = [i for i in range(len(names)) if i not in plant_ax]
    t0 = time.time()
    for po in itertools.product(*[range(shape[i]) for i in plant_ax]):
        over = {names[i]: vals[i][j] for i, j in zip(plant_ax, po)}
        over["apply_tol"] = True
        pc = sim.pc(over)
        tag = ("grid",) + tuple(po)
        pl = sim.plant(pc, u, tag)
        for io in itertools.product(*[range(shape[i]) for i in inner_ax]):
            o2 = {**over, **{names[i]: vals[i][j] for i, j in zip(inner_ax, io)}}
            pc2 = sim.pc(o2)
            out = forward_plant(sim.model, pl, pc2, u, x, sim.p_ref)["out"]
            pred = out.argmax(1)
            full = [0] * len(names)
            for i, j in zip(plant_ax, po): full[i] = j
            for i, j in zip(inner_ax, io): full[i] = j
            acc[tuple(full)] = np.mean(pred == y); agr[tuple(full)] = np.mean(pred == vpred)
        sim._plants.clear()
        if progress:
            progress(po, time.time() - t0)
    return {"axes": [{"name": n, "values": [float(v) for v in vs], "unit": un} for (n, _, un), vs in zip(axes, vals)],
            "accuracy": acc, "agreement": agr, "runtime_s": time.time() - t0}


def export_grid(path: str | Path, grids: dict, meta: dict) -> dict:
    """Write docs/artifacts/data/sweeps_grid.json (schema dlnn-sweeps-grid/1) with run id and sha256 of the payload."""
    payload = {"schema": "dlnn-sweeps-grid/1", "run_id": meta["run_id"], "tier": "P",
               "note": "Simulation only (lumped-element parasitics + tolerance model, hw/board/params.json). Designed, not built, nothing measured. "
                       "One fixed 'golden' board: unit tolerance draws are fixed and scaled by the axis values; stage_snr_db overrides the physical stage noise.",
               "meta": meta, "models": {}}
    for name, g in grids.items():
        payload["models"][name] = {
            "axes": g["axes"], "order": "C (last axis fastest)", "shape": [len(a["values"]) for a in g["axes"]],
            "n_points": int(g["accuracy"].size), "accuracy": np.round(g["accuracy"], 5).ravel().tolist(),
            "agreement_vs_virtual": np.round(g["agreement"], 5).ravel().tolist(), "v_accuracy_same_subset": g.get("v_accuracy"),
            "runtime_s": round(g["runtime_s"], 1)}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    payload["sha256_of"] = "canonical JSON (sort_keys, compact separators) of this object without the sha256 and sha256_of keys"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=None, separators=(",", ":")))
    return payload


# ------------------------------------------------------------------------------------------------ tier entry point
def run(model_npz: str | Path, dataset: dict, cfg: dict, outdir: str | Path) -> contract.TierResult:
    t0 = time.time()
    model = load_model(model_npz)
    cfg = dict(cfg or {})
    x, y = dataset["x_test"], dataset["y_test"]
    sim = Simulator(model, x, cfg)
    pc = sim.pc()
    res = sim.forward(x)                                     # nominal parts, full test set
    v = virtual.forward(model, x, cfg)
    ref = v["out"]
    status = "ok" if HAVE_SKRF else "fallback"
    diag = {}
    for tag, over in (("as_designed_gain_trim_0dB", {"gain_trim_db": 0.0}), ("trim_6dB_level_fix", {"gain_trim_db": 6.0206}), ("Q30_assumed_typical", {"q_inductor": 30.0}),
                      ("Q10_datasheet_min", {"q_inductor": 10.0})):
        o = sim.forward(x, over)["out"]
        diag[tag] = (float(np.mean(o.argmax(1) == y)), float(np.mean(o.argmax(1) == ref.argmax(1))))
    tool = {"name": "scikit-rf" if HAVE_SKRF else "numpy-abcd-fallback", "version": SKRF_VERSION if HAVE_SKRF else np.__version__}

    n_sub = int(cfg.get("p_subset", 1000))
    idx = stratified(dataset, n_sub) if n_sub < len(y) else np.arange(len(y))
    xs, ys = x[idx], y[idx]
    vs = ref[idx]
    sw_entries, sw_arrays = [], {}
    if not cfg.get("skip_sweeps"):
        sw_entries, sw_arrays = run_sweeps(sim, xs, ys, vs, boards=int(cfg.get("p_sweep_boards", 4)), only=cfg.get("p_sweep_axes"))
    arrays_extra = dict(sw_arrays)
    mc_n = int(cfg.get("mc_trials", 1000))
    mc_info = {}
    warns = []
    if mc_n > 0:
        n_mc = int(cfg.get("mc_subset", 400))
        imc = stratified(dataset, n_mc)
        mc = monte_carlo(sim, x[imc], y[imc], ref[imc].argmax(1), mc_n, seed=int(cfg.get("mc_seed", 1000)),
                         yield_drop=float(cfg.get("yield_drop", 0.05)))
        mc_info = mc["stats"]
        arrays_extra["mc.accuracy"] = mc["accuracy"].astype(np.float32)
        arrays_extra["mc.agreement_vs_virtual"] = mc["agreement"].astype(np.float32)
        arrays_extra["mc.subset_indices"] = imc.astype(np.int64)
        Path(outdir, "tiers").mkdir(parents=True, exist_ok=True)
        Path(outdir, "tiers", "P_montecarlo.json").write_text(json.dumps(mc_info, indent=2))
        warns.append(f"Monte Carlo: {mc_n} boards on a fixed stratified subset of {len(imc)} test examples: accuracy "
                     f"{mc_info['accuracy_mean']:.4f} +- {mc_info['accuracy_std']:.4f} (p5 {mc_info['accuracy_p5']:.4f}), yield "
                     f"{mc_info['yield']:.3f} at >= V-5pt ({mc_info['yield_threshold']:.4f} on that subset), {mc_info['runtime_s']:.0f} s")
    models = [f"{tool['name']} {tool['version']}: ABCD cascade of {pc['n_sections']} constant-k LC sections per line "
              f"(L {pc['L_h']*1e6:g} uH + DCR {pc['dcr_ohm']:g} ohm, Cp {pc['cp_f']*1e12:.1f} pF from SRF {pc['srf_hz']/1e6:g} MHz, Rp from Q={pc['q_inductor']:g}; C {pc['C_f']*1e12:g} pF), "
              f"Rs/Rl {pc['Rs_ohm']:g} ohm, input RC {1/(2*np.pi*pc['rc_tau_s'])/1e3:.0f} kHz (edge {pc['edge_ns']:.0f} ns), tap buffers GBW {pc['opamp_gbw_hz']/1e6:g} MHz",
              "ladder responses -> impulse responses at the S/H instants by direct inverse DFT; FIR engine of tier R (rf.fir_apply)",
              f"crossbar 10 kohm +-{pc['r_tol_pct']:g} %, DNP-pad leakage, summer pole and R_f error, tanh with V_BE mismatch {pc['vbe_mismatch_mv']:g} mV and temperature, diff-amp gain error, track-and-hold acquisition error and phase error",
              f"inter-line mutual inductance k (dipole estimate, {pc['crosstalk_layout']} layout, {pc['crosstalk_pitch_mm']:g} mm pitch), ribbon 30 pF/m + NEXT",
              f"clock calibrated on tile-1 line 0, residual error {pc['clock_err_pct']:g} %; stage noise from params.json summer_noise; 12-bit DAC/ADC",
              "nominal metrics: nominal parts (no tolerance draw) on the FULL test set of "
              f"{len(y)}; sweeps on a fixed stratified subset of {len(idx)} examples (tolerance axes: mean of {cfg.get('p_sweep_boards', 4)} boards); "
              + (f"Monte Carlo on {mc_info['n_examples']} examples" if mc_info else "Monte Carlo not run")]
    omits = ["LUMPED-ELEMENT circuit model at ~1 MBd: NOT RF (wavelength >> board); no EM/layout/field solving, no hardware measurement",
             "inductor Q is only known as a datasheet minimum (>= 10 @ 796 kHz); Q = 30 is an assumed typical; Rp frequency independent; no inductor tempco or saturation",
             "mutual inductance is a dipole estimate between equal sections of neighbour lines (no capacitive/ground-return coupling); tolerance enters first order on top of the coupled nominal ladder",
             "op-amps are single-pole (GBW 10 MHz) with no slew limit, offset or output impedance beyond the ribbon driver (50 ohm assumed); diff pair is ideal tanh + V_BE offset (no beta, Early, emitter-tail variation)",
             "no clock jitter (static phase/period error only), no supply noise, no ground bounce, no temperature dependence except V_T",
             "tolerances are uniform within +-tol (a bounded distribution), independent per part with a shared tile-level component"]
    chk = common.metrics_and_arrays(res["out"], ref, dataset)
    m, arrays, n = chk
    for l, hh in enumerate(res["layers"]):
        arrays[f"trace_activations.L{l}"] = hh[:int(cfg.get("n_trace", 16))].astype(np.float32)
    arrays.update(arrays_extra)
    for k, (a_, g_) in diag.items():
        arrays[f"diag.{k}.accuracy"] = np.array(a_)
        arrays[f"diag.{k}.top1_agreement_vs_virtual"] = np.array(g_)
    warns.append("Finding F1 (gain trim): the board AS DESIGNED (R_f sized for tap amplitude A_in*x, but the matched ladder halves it) gives "
                 f"accuracy {diag['as_designed_gain_trim_0dB'][0]:.4f} vs V {float(np.mean(ref.argmax(1) == y)):.4f}; the headline numbers use the "
                 f"M1a normalisation (mean on-lag tap gain = 1, auto trim {pc['gain_trim_db']:.2f} dB; the fixed 6.02 dB level fix gives {diag['trim_6dB_level_fix'][0]:.4f}). Q corners (trimmed): Q=10 {diag['Q10_datasheet_min'][0]:.4f}, "
                 f"Q=30 {diag['Q30_assumed_typical'][0]:.4f}.")
    warns.append("Lumped-element parasitics/tolerance model, not RF; simulation only, nothing measured. "
                 f"Calibrated symbol period {sim.plant(pc, zero_units(model), 'info').info['T_clk_us']:.3f} us; "
                 f"DC tap gain tap0..7 (tile 1): " + ", ".join(f"{g:.3f}" for g in sim.plant(pc, zero_units(model), 'info').info['layers'][0]['dc_tap_gain']))
    return contract.TierResult(
        tier="P", status=status, run_id=cfg.get("run_id", "adhoc"), task=dataset["name"], tool=tool, model_sha256=model.sha256(),
        dataset_sha256=dataset["sha256"], n_examples=n, metrics=m, sweeps=sw_entries, fidelity={"models": models, "omits": omits},
        inputs_from=["hw/board/params.json"] if pc["_have_params"] else [], runtime_s=round(time.time() - t0, 3), warnings=warns, arrays=arrays)


# ------------------------------------------------------------------------------------------------ notebook API
# Stateless entry points: every function takes parameters (+ optional outdir) and returns plain dicts / arrays.
def _model(m):
    return load_model(m) if isinstance(m, (str, Path)) else m


def _write(outdir, name, obj):
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        Path(outdir, name).write_text(json.dumps(obj, indent=1))


def build_line(params: dict | None = None, corner: str = "Q10", M: int = 12) -> dict:
    """Tap FIRs of the M1a line (tile 1, line 0: input RC, buffers, ladder, ZOH pulse) at symbol rate: {'fir': [8, M], 'T_us', 't0_us'}.
    corner 'Q10' | 'Q30' sets q_inductor; params override DEFAULTS/params.json. Same clock/phase rules as build_plant (centroid clock)."""
    pc = resolve({**(params or {}), "q_inductor": {"Q10": 10.0, "Q30": 30.0}.get(corner, (params or {}).get("q_inductor", 10.0))})
    ctx = Context(pc)
    f, df, dt, N = ctx.f, ctx.df, ctx.dt, ctx.N
    jw = 2j * np.pi * f
    Hb = 1.0 / (1 + jw / (2 * np.pi * pc["opamp_gbw_hz"]))
    g = ctx.H_nominal() * ((1.0 / (1 + jw * pc["rc_tau_s"])) * Hb * Hb)[None, :]
    Tg = pc["T_nom_s"]
    Pg = Tg * np.sinc(f * Tg) * np.exp(-1j * np.pi * f * Tg)
    hh = np.fft.irfft(g * Pg[None, :], n=N, axis=1)
    tt = np.arange(N) * dt
    sel = tt < 30e-6
    cen = (tt[sel] * hh[:, sel]).sum(1) / hh[:, sel].sum(1)
    K = g.shape[0]
    T = float((cen[-1] - cen[0]) / (K - 1))
    Pf = T * np.sinc(f * T) * np.exp(-1j * np.pi * f * T)
    wk = np.full(len(f), 2.0); wk[0] = 1.0; wk[-1] = 1.0
    G = g * Pf[None, :]

    def fir(t0):
        E = (wk * df)[None, :] * np.exp(2j * np.pi * f[None, :] * (t0 + np.arange(M)[:, None] * T))
        return np.real(G @ E.T)
    best, bt = -1.0, 0.0
    for t0 in np.linspace(0.2 * T, 2.0 * T, 181):
        F = fir(t0)
        sc = np.mean([abs(F[k, k]) / (np.abs(F[k]).sum() + 1e-30) for k in range(K)])
        if sc > best:
            best, bt = sc, t0
    return {"fir": fir(bt), "T_us": T * 1e6, "t0_us": bt * 1e6, "corner": corner}


def run_nominal(model, dataset: dict, cfg: dict | None = None, outdir=None, n: int | None = None) -> dict:
    """Nominal-parts P vs V (and V-ideal) at Q = 10 and Q = 30 (+ board-as-designed and 6 dB trim diagnostics)."""
    m = _model(model)
    x, y = dataset["x_test"][:n], dataset["y_test"][:n]
    ref = virtual.forward(m, x)["out"]
    sim = Simulator(m, x, cfg)
    out = _nominal_chunk(sim, x, y, ref)
    _write(outdir, "nominal.json", out)
    return out


def mc_study(model, dataset: dict, n_trials: int = 50, subset: int = 200, seed: int = 1000, cfg: dict | None = None, outdir=None,
             yield_drop: float = 0.05) -> dict:
    """Monte Carlo over all tolerances on a fixed stratified subset: stats dict + per-trial arrays."""
    m = _model(model)
    x, y = dataset["x_test"], dataset["y_test"]
    ref = virtual.forward(m, x)["out"]
    idx = stratified(dataset, subset) if subset < len(y) else np.arange(len(y))
    sim = Simulator(m, x, cfg)
    r = monte_carlo(sim, x[idx], y[idx], ref[idx].argmax(1), n_trials, seed=seed, yield_drop=yield_drop)
    out = {"stats": r["stats"], "accuracy": r["accuracy"].tolist(), "agreement": r["agreement"].tolist()}
    _write(outdir, "mc.json", out)
    return out


def sweep(model, dataset: dict, axis: str, values=None, cfg: dict | None = None, subset: int = 500, boards: int = 2, outdir=None) -> dict:
    """One sweep axis (any key of SWEEP_PLAN): {'axis', 'values', 'accuracy', 'mean_kl', 'agreement'}; `values` overrides the default points."""
    m = _model(model)
    x, y = dataset["x_test"], dataset["y_test"]
    ref = virtual.forward(m, x)["out"]
    idx = stratified(dataset, subset) if subset < len(y) else np.arange(len(y))
    sim = Simulator(m, x, cfg)
    plan = dict(SWEEP_PLAN)
    if values is not None:
        plan = {axis: (list(values), plan[axis][1], plan[axis][2])}
    old = dict(SWEEP_PLAN)
    SWEEP_PLAN.clear(); SWEEP_PLAN.update({axis: plan[axis]})
    try:
        entries, arrays = run_sweeps(sim, x[idx], y[idx], ref[idx], boards=boards, only=[axis])
    finally:
        SWEEP_PLAN.clear(); SWEEP_PLAN.update(old)
    e = entries[0]
    out = {**e, "agreement": arrays[f"sweep.{axis}.top1_agreement_vs_virtual"].tolist()}
    _write(outdir, f"sweep_{axis}.json", out)
    return out


def make_grid(model, dataset: dict, axes=None, subset: int = 200, cfg: dict | None = None, outdir=None, run_id: str = "adhoc", name: str = "model") -> dict:
    """Full Cartesian grid over `axes` ([(name, values, unit), ...], default GRID_AXES) -> dict in the sweeps_grid.json schema
    (also written to outdir/sweeps_grid.json when outdir is given)."""
    m = _model(model)
    x, y = dataset["x_test"], dataset["y_test"]
    ref = virtual.forward(m, x)["out"]
    idx = stratified(dataset, subset) if subset < len(y) else np.arange(len(y))
    sim = Simulator(m, x, cfg)
    g = run_grid(sim, x[idx], y[idx], ref[idx], axes=axes)
    g["v_accuracy"] = float(np.mean(ref[idx].argmax(1) == y[idx]))
    meta = {"run_id": run_id, "n_examples_per_point": int(len(idx)), "dataset_sha256": dataset["sha256"]}
    path = Path(outdir) / "sweeps_grid.json" if outdir else Path(os.devnull)
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
    return export_grid(path, {name: g}, meta)


def export_web(outdir, model, dataset: dict, n_trials: int = 50, mc_subset: int = 200, grid_subset: int = 200, grid_points: int = 3,
               cfg: dict | None = None, run_id: str = "adhoc") -> dict:
    """Run a small end-to-end P study and dump what the article uses into outdir: p_web.json (nominal, MC stats, sweeps, line FIRs)
    and sweeps_grid.json (grid_points^5 grid). Defaults finish in about 2 minutes."""
    out = Path(outdir)
    res = {"run_id": run_id, "nominal": run_nominal(model, dataset, cfg, out),
           "monte_carlo": mc_study(model, dataset, n_trials, mc_subset, cfg=cfg, outdir=out)["stats"],
           "line_fir": {c: build_line(cfg, c)["fir"].round(5).tolist() for c in ("Q10", "Q30")}, "sweeps": {}}
    for ax in ("stage_snr_db", "clock_err_pct", "l_tol_pct", "crosstalk_k"):
        res["sweeps"][ax] = sweep(model, dataset, ax, cfg=cfg, subset=mc_subset, boards=2, outdir=out)
    axes = [(n, [v[int(round(i))] for i in np.linspace(0, len(v) - 1, grid_points)], u) for n, v, u in GRID_AXES]
    g = make_grid(model, dataset, axes, grid_subset, cfg, out, run_id)
    res["grid"] = {"file": "sweeps_grid.json", "sha256": g["sha256"]}
    _write(out, "p_web.json", res)
    return res


# ------------------------------------------------------------------------------------------------ CLI (checkpointed, resumable)
TOL_KEYS = ("l_tol_pct", "c_tol_pct", "r_tol_pct", "gain_tol_pct", "clock_err_pct", "vbe_mismatch_mv", "th_phase_err_ns", "c_pad_tol_pct")
CROSSTALK_CASES = [("parallel", 3.0), ("alternating", 3.0), ("parallel", 5.0), ("alternating", 5.0)]


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _save_chunk(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)                                    # atomic: a killed job never leaves a half-written chunk


def _chunk(path: Path, fn, log):
    """Run fn() and store its dict unless the chunk file already exists (resume)."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            path.unlink()
    t0 = time.time()
    obj = fn()
    obj["chunk_runtime_s"] = round(time.time() - t0, 2)
    _save_chunk(path, obj)
    log(f"done {path.parent.name}/{path.name} in {obj['chunk_runtime_s']} s")
    return obj


def _nominal_chunk(sim, x, y, ref):
    out = {}
    vp, vprob = ref.argmax(1), common.softmax(ref)
    for tag, over in (("Q10", {"q_inductor": 10.0}), ("Q30", {"q_inductor": 30.0}),
                      ("Q10_as_designed_trim0dB", {"q_inductor": 10.0, "gain_trim_db": 0.0}),
                      ("Q10_ideal_wire_no_xtalk", {"q_inductor": 10.0, "crosstalk_k": 0.0})):
        o = sim.forward(x, over)["out"]
        out[tag] = {"accuracy": float(np.mean(o.argmax(1) == y)), "agreement_vs_v": float(np.mean(o.argmax(1) == vp)),
                    "mean_kl_vs_v": float(common.mean_kl(vprob, common.softmax(o)))}
    out["v_accuracy"] = float(np.mean(vp == y))
    out["n_examples"] = int(len(y))
    return out


M1A_FIR_JSON = REPO / "models" / "board_aware" / "line_fir_m1a.json"


def v_on_m1a_fir(model_npz: str | Path, ds: dict) -> dict:
    """Reference V (float forward of the trained ternary net, torch) with the M1a ngspice line FIR per tap (Q = 10 / Q = 30 corners,
    models/board_aware/line_fir_m1a.json) instead of the ideal unit delay. The line is ideally buffered there (no tolerance, no
    crossbar/summer/diff-pair/T-H parasitics), so it is the V-with-real-line reference to compare tier P against."""
    import torch
    from ..model import train
    from ..model.tdlnet_torch import TDLNet
    mp = Path(model_npz)
    npz = np.load(mp, allow_pickle=False)
    cfg = json.loads(str(npz["meta_json"]))["arch"]
    sd = torch.load(mp.with_suffix(".pt"))
    F = json.loads(M1A_FIR_JSON.read_text())["corners"]
    out = {}
    for tag, fir in (("ideal", None), ("Q10", F["Q10"]["fir"]), ("Q30", F["Q30"]["fir"])):
        c = dict(cfg); c["line_fir"] = fir
        net = TDLNet.from_config(c); net.load_state_dict(sd); net.eval()
        lg = train.predict(net, ds["x_test"])
        out[tag] = {"accuracy": float(np.mean(lg.argmax(1) == ds["y_test"])), "pred": lg.argmax(1).tolist()}
    return out


def _vfir_chunk(mp, ds, sim_pred):
    r = v_on_m1a_fir(mp, ds)
    res = {}
    for tag in ("Q10", "Q30"):
        pv = np.array(r[tag]["pred"])
        res[tag] = {"v_m1a_fir_accuracy": r[tag]["accuracy"], "p_accuracy": sim_pred[tag]["accuracy"],
                    "p_minus_v_m1a_fir": sim_pred[tag]["accuracy"] - r[tag]["accuracy"], "agreement_p_vs_v_m1a_fir": sim_pred[tag]["agr"](pv)}
    res["v_ideal_accuracy"] = r["ideal"]["accuracy"]
    return res


def _sweep_chunk(sim, xs, ys, vs, axis, boards):
    entries, arrays = run_sweeps(sim, xs, ys, vs, boards=boards, only=[axis])
    return {"entry": entries[0], "arrays": {k: v.tolist() for k, v in arrays.items()}}


def _xtalk_cases_chunk(sim, xs, ys, vs):
    vp, vprob = vs.argmax(1), common.softmax(vs)
    rows = []
    for lay, pit in CROSSTALK_CASES:
        o = sim.forward(xs, {"crosstalk_layout": lay, "crosstalk_pitch_mm": pit})["out"]
        k01 = float(k_coupling({**sim.pc({"crosstalk_layout": lay, "crosstalk_pitch_mm": pit})})[0, 1])
        rows.append({"layout": lay, "pitch_mm": pit, "k_nearest": k01, "accuracy": float(np.mean(o.argmax(1) == ys)),
                     "agreement_vs_v": float(np.mean(o.argmax(1) == vp)), "mean_kl_vs_v": float(common.mean_kl(vprob, common.softmax(o)))})
    return {"cases": rows}


def _mc_chunk(sim, xm, ym, vm, seed0, n):
    r = monte_carlo(sim, xm, ym, vm, n, seed=seed0)
    return {"seed0": seed0, "accuracy": r["accuracy"].tolist(), "agreement": r["agreement"].tolist()}


def _attrib_chunk(sim, xm, ym, vm, n):
    """One tolerance at a time at its nominal width (all other tolerances zero): which one hurts most."""
    out = {}
    base = sim.forward(xm, {"apply_tol": False})["out"].argmax(1)
    out["nominal_accuracy"] = float(np.mean(base == ym))
    for k in TOL_KEYS:
        over = {kk: 0.0 for kk in TOL_KEYS}
        over[k] = DEFAULTS[k]
        r = monte_carlo(sim, xm, ym, vm, n, seed=9000, over=over)
        out[k] = {"width": DEFAULTS[k], "accuracy_mean": r["stats"]["accuracy_mean"], "accuracy_std": r["stats"]["accuracy_std"],
                  "accuracy_p5": r["stats"]["accuracy_p5"]}
    return out


def _grid_chunk(sim, xg, yg, vg, li, board_seed=777):
    axes = [(n, ([v[li]] if n == GRID_AXES[0][0] else v), u) for n, v, u in GRID_AXES]
    g = run_grid(sim, xg, yg, vg, board_seed=board_seed, axes=axes)
    return {"l_tol_index": li, "accuracy": g["accuracy"].tolist(), "agreement": g["agreement"].tolist()}


def run_model(mp: Path, ds: dict, out_root: Path, a, log) -> dict:
    name = Path(mp).stem.replace("_ternary", "")
    sha = _sha_file(mp)
    od = out_root / name
    mfile = od / "meta.json"
    if mfile.exists() and json.loads(mfile.read_text()).get("model_file_sha256") != sha:   # model file replaced: old chunks are stale
        old = json.loads(mfile.read_text())["model_file_sha256"][:8]
        od.rename(out_root / f"{name}.stale-{old}")
        log(f"{name}: model file changed (was {old}), moved old chunks aside")
    od.mkdir(parents=True, exist_ok=True)
    _save_chunk(mfile, {"model": str(mp), "model_file_sha256": sha})
    model = load_model(mp)
    x, y = ds["x_test"], ds["y_test"]
    ref = virtual.forward(model, x)["out"]
    sim = Simulator(model, x, {"run_id": a.run_id})
    log(f"{name} sha256 {sha[:16]} V acc {np.mean(ref.argmax(1) == y):.4f}")
    nom = _chunk(od / "nominal.json", lambda: _nominal_chunk(sim, x, y, ref), log)

    def _vf():
        sp = {}
        for tag, q in (("Q10", 10.0), ("Q30", 30.0)):
            o = sim.forward(x, {"q_inductor": q})["out"].argmax(1)
            sp[tag] = {"accuracy": float(np.mean(o == y)), "agr": (lambda pv, o=o: float(np.mean(o == pv)))}
        return _vfir_chunk(mp, ds, sp)
    _chunk(od / "v_m1a_fir.json", _vf, log)
    idx = stratified(ds, a.sweep_subset)
    xs, ys, vs = x[idx], y[idx], ref[idx]
    for axis in SWEEP_PLAN:
        _chunk(od / f"sweep_{axis}.json", lambda ax=axis: _sweep_chunk(sim, xs, ys, vs, ax, a.boards), log)
    _chunk(od / "sweep_crosstalk_cases.json", lambda: _xtalk_cases_chunk(sim, xs, ys, vs), log)
    imc = stratified(ds, a.mc_subset)
    xm, ym, vm = x[imc], y[imc], ref[imc].argmax(1)
    for b in range(0, a.mc, 100):
        n = min(100, a.mc - b)
        _chunk(od / f"mc_{b:04d}.json", lambda b=b, n=n: _mc_chunk(sim, xm, ym, vm, 1000 + b, n), log)
    _chunk(od / "attrib.json", lambda: _attrib_chunk(sim, xm, ym, vm, a.attrib_trials), log)
    gi = stratified(ds, a.grid_subset)
    vg = ref[gi]
    for li in range(len(GRID_AXES[0][1])):
        _chunk(od / f"grid_l{li}.json", lambda li=li: _grid_chunk(sim, x[gi], y[gi], vg, li), log)
    return {"name": name, "sha": sha, "model": model, "ref": ref, "nominal": nom, "mc_idx": imc, "grid_idx": gi, "dir": od}


def assemble(info: dict, ds: dict, mp: Path, a, log) -> tuple[dict, np.ndarray]:
    """Collect the chunks of one model into the contract tier JSON (+ P_montecarlo.json, summary.json) and the grid array."""
    od: Path = info["dir"]
    y = ds["y_test"]
    r = run(mp, ds, {"run_id": a.run_id, "mc_trials": 0, "skip_sweeps": True}, od)
    entries, arrays = [], {}
    for axis in SWEEP_PLAN:
        c = json.loads((od / f"sweep_{axis}.json").read_text())
        entries.append(c["entry"])
        arrays.update({k: np.array(v) for k, v in c["arrays"].items()})
    accs, agrs = [], []
    for b in range(0, a.mc, 100):
        c = json.loads((od / f"mc_{b:04d}.json").read_text())
        accs += c["accuracy"]; agrs += c["agreement"]
    accs, agrs = np.array(accs), np.array(agrs)
    imc = info["mc_idx"]
    v_acc = float(np.mean(info["ref"][imc].argmax(1) == y[imc]))
    st = _stats(accs, v_acc - 0.05)
    st.update({"v_accuracy_same_subset": v_acc, "agreement_mean": float(agrs.mean()), "n_examples": int(len(imc)), "seed0": 1000,
               "yield_rule": "accuracy >= V accuracy on the same subset - 5 points",
               "runtime_s": float(sum(json.loads((od / f"mc_{b:04d}.json").read_text())["chunk_runtime_s"] for b in range(0, a.mc, 100)))})
    (od / "tiers").mkdir(parents=True, exist_ok=True)
    (od / "tiers" / "P_montecarlo.json").write_text(json.dumps(st, indent=2))
    r.sweeps = entries
    r.arrays.update(arrays)
    r.arrays["mc.accuracy"] = accs.astype(np.float32)
    r.arrays["mc.agreement_vs_virtual"] = agrs.astype(np.float32)
    r.arrays["mc.subset_indices"] = imc.astype(np.int64)
    r.warnings.append(f"Monte Carlo: {a.mc} boards on a fixed stratified subset of {len(imc)} test examples: accuracy {st['accuracy_mean']:.4f} "
                      f"+- {st['accuracy_std']:.4f} (p5 {st['accuracy_p5']:.4f}), yield {st['yield']:.3f} at >= V-5pt ({st['yield_threshold']:.4f} on that subset), "
                      f"{st['runtime_s']:.0f} s of compute")
    r.fidelity["models"][-1] = (f"nominal metrics: nominal parts on the FULL test set of {len(y)}; sweeps on a fixed stratified subset of "
                                f"{a.sweep_subset} examples (tolerance axes: mean of {a.boards} boards); Monte Carlo {a.mc} trials on {len(imc)} examples")
    contract.validate(r)
    contract.save(r, od)
    nominal = info["nominal"]
    summary = {"model": info["name"], "model_file_sha256": info["sha"], "nominal": nominal, "monte_carlo": st,
               "attribution_one_at_a_time": json.loads((od / "attrib.json").read_text()),
               "v_on_m1a_fir": json.loads((od / "v_m1a_fir.json").read_text()),
               "crosstalk_cases": json.loads((od / "sweep_crosstalk_cases.json").read_text())["cases"]}
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    g = None
    parts = [json.loads((od / f"grid_l{li}.json").read_text()) for li in range(len(GRID_AXES[0][1]))]
    acc = np.concatenate([np.array(p["accuracy"]) for p in parts], 0)
    agr = np.concatenate([np.array(p["agreement"]) for p in parts], 0)
    rt = sum(p["chunk_runtime_s"] for p in parts)
    gi = info["grid_idx"]
    g = {"axes": [{"name": n, "values": [float(v) for v in vs], "unit": un} for n, vs, un in GRID_AXES], "accuracy": acc, "agreement": agr,
         "runtime_s": rt, "v_accuracy": float(np.mean(info["ref"][gi].argmax(1) == y[gi])), "model_file_sha256": info["sha"]}
    return summary, g


def main(argv=None):  # pragma: no cover
    import argparse
    from ..data import modrec
    ap = argparse.ArgumentParser(description="Tier P: checkpointed + resumable nominal run, sweeps, Monte Carlo, attribution, grid")
    ap.add_argument("--models", nargs="+", default=["models/tdlnet_b_w8_ternary.npz", "models/tdlnet_b_w8_aware_ternary.npz",
                                                      "models/tdlnet_b_w8_aware2_ternary.npz"])
    ap.add_argument("--out", default="out/P2-M2b")
    ap.add_argument("--run-id", default="P2-M2b")
    ap.add_argument("--mc", type=int, default=1000)
    ap.add_argument("--mc-subset", type=int, default=800)
    ap.add_argument("--sweep-subset", type=int, default=1000)
    ap.add_argument("--boards", type=int, default=4)
    ap.add_argument("--attrib-trials", type=int, default=100)
    ap.add_argument("--grid-subset", type=int, default=800)
    ap.add_argument("--grid-out", default="docs/artifacts/data/sweeps_grid.json")
    ap.add_argument("--assemble", action="store_true", help="after the chunks exist (run without it first), build tier JSONs + grid file")
    a = ap.parse_args(argv)
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    ds = modrec.dataset()
    infos = {}
    for mp in a.models:
        infos[mp] = run_model(Path(mp), ds, out_root, a, log)
    if a.assemble:
        grids, sums = {}, {}
        t0 = time.time()
        for mp, info in infos.items():
            sums[info["name"]], grids[info["name"]] = assemble(info, ds, Path(mp), a, log)
            log(f"assembled {info['name']}")
        meta = {"run_id": a.run_id, "n_examples_per_point": int(a.grid_subset),
                "subset": "stratified over (class, SNR) cells, first n/cells of each, test-set order (dlnn.pipeline.stratified_subset)",
                "dataset_sha256": ds["sha256"], "q_inductor": 10.0, "gain_trim_db": DEFAULTS["gain_trim_db"],
                "model_file_sha256": {k: v["model_file_sha256"] for k, v in grids.items()},
                "base_cfg": "Q = 10; nominal parts for all non-swept tolerance axes except one fixed 'golden' board (unit draws fixed, scaled by the axes); board resolved from hw/board/params.json"}
        res = export_grid(a.grid_out, grids, meta)
        (out_root / "summary_all.json").write_text(json.dumps({"models": sums, "grid_sha256": res["sha256"]}, indent=2))
        log(f"grid sha256 {res['sha256']}")


if __name__ == "__main__":  # pragma: no cover
    main()
