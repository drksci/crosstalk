"""Tier R (RF): channel-level S-parameter simulation of the tapped-delay-line network.

This is a CHANNEL-LEVEL simulation (scikit-rf two-port cascades + numpy convolution). There is no EM/layout
modelling and no hardware measurement. See docs/rf_model.md for the equations, the parameter table and what is
not modelled.

Signal flow of one layer (everything linear is a frequency response H(f); the amplifier is memoryless):

  x_i[n] (pulses, end-to-end raised cosine, beta=1, 10 ns symbol)
     -> for every tap path (o,i,k):  coupler (coupling, amp/phase error, finite isolation)
                                     -> mismatch Gamma -> lossy dispersive coax of length k*d*T_s*v_p
                                     -> mismatch Gamma -> 180-degree hybrid (sign) -> Wilkinson branch (1/sqrt(N))
     -> sum of all paths (+ crosstalk between neighbouring input lines)
     -> amplifier: gain compensates the mean tap loss, + AWGN (stage SNR, NF), Rapp (or tanh) AM/AM
     -> ideal re-timing sample&hold at the symbol instants + pulse regeneration (the next layer's input)
  -> RC pool (time mean) -> ADC (adc_bits) -> software readout.

`H_{o,i,k}(f)` is built with `skrf` (`skrf.media.Coaxial` + flat two-port networks, cascaded with `**`), turned
into impulse responses with an IFFT on a fine grid (default 8 points per symbol) and convolved with the whole
test set with numpy FFT convolution. Because the re-timing samples the combiner output at the symbol instants,
the pulse-shape * channel convolution sampled at symbol spacing is an exact symbol-rate MIMO FIR; the
fine-grid (waveform) path is implemented separately (`fine_waveform`) and checked against it in the tests.

If scikit-rf cannot be imported the same channel is built from closed-form RLGC/skin-effect equations in numpy and
the result is labelled status="fallback".
"""
from __future__ import annotations

import json
import time
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import contract
from ..model.npz import Layer, Model, load_model
from . import common, virtual

try:  # scikit-rf is the real tool; without it we fall back to a closed-form numpy RLGC line
    import skrf as _rf
    from skrf.media import Coaxial as _Coaxial
    HAVE_SKRF = True
    SKRF_VERSION = str(_rf.__version__)
except Exception:  # pragma: no cover - exercised only when scikit-rf is missing
    _rf = None
    _Coaxial = None
    HAVE_SKRF = False
    SKRF_VERSION = "n/a"

C0 = 299792458.0
ETA0 = 376.730313

DEFAULTS: dict = dict(
    symbol_ns=10.0,          # T_s
    vp_cm_per_ns=19.8,       # RG-58 phase velocity
    oversample=8,            # fine-grid points per symbol (waveforms / IFFT grid)
    z0=50.0,                 # system impedance, ohm
    f_ref_mhz=100.0,         # frequency at which loss_db_per_us is specified (skin effect ~ sqrt(f) below/above)
    trim_mhz=50.0,           # line delay is trimmed to nominal at this frequency
    loss_db_per_us=5.0,      # coax loss at f_ref, dB per microsecond of delay (RG-58 ~32, LMR-400 ~14, 0 = lossless)
    coupling_db=10.0,        # coupler through->tap coupling (sets required amplifier gain only)
    isolation_db=30.0,       # coupler isolation: a "0" tap leaks 10^(-iso/20) of a unit tap (inf = no leak)
    tap_amp_err_db=0.5,      # sigma of per-tap amplitude error, dB
    tap_phase_err_deg=5.0,   # sigma of per-tap phase error at f_ref, degrees (realised as a small delay skew)
    gamma_mismatch=0.05,     # reflection coefficient at both ends of every tap line (ripple/echo)
    crosstalk_db=-35.0,      # coupling between neighbouring input lines (None = off)
    snr_db=30.0,             # per-stage SNR at the amplifier input (None = noiseless)
    nf_db=3.0,               # amplifier noise figure (degrades the stage SNR)
    adc_bits=10,             # readout ADC (None = ideal)
    amp_model="tanh",        # "tanh" (BJT diff pair = the model activation) or "rapp" (Rapp AM/AM, smoothness fitted to tanh)
    rapp_p="fit",            # Rapp smoothness; "fit" = least-squares fit of the unit Rapp curve to tanh
    tap_equalize=False,      # True: per-tap gain trim restores every tap's main-lobe gain (no extra noise modelled)
    rf_seed=None,            # seed of the static tap errors/noise (None -> cfg["seed"] + 303)
    sweep_baseline="nominal",  # sweeps vary one parameter around the resolved cfg ("ideal" = around IDEAL)
    snr_sweep_mode="stage",  # "stage": per-stage SNR axis; "input": AWGN on the input (as V/E do)
    amp_vsat_scale=1.0,      # multiplies every layer's V_sat (set by the cascade hook from the diff-pair fit)
    amp_gain_scale=1.0,      # multiplies every layer's amplifier gain (cascade hook)
    amp_offset_rel=0.0,      # output offset of the amplifier relative to its gain (cascade hook)
    cascade_params=None,     # path of params/circuit_extracted.json (cascade hook)
    n_waveforms=2,           # examples whose fine-grid layer-1 waveforms are stored
)
IDEAL: dict = dict(loss_db_per_us=0.0, isolation_db=float("inf"), tap_amp_err_db=0.0, tap_phase_err_deg=0.0,
                   gamma_mismatch=0.0, crosstalk_db=None, nf_db=0.0, adc_bits=None, amp_model="tanh",
                   tap_equalize=False)

_RAPP_P_FIT: float | None = None
TOL = 1e-4  # impulse-response samples below this (relative to a unit tap) are dropped


def rapp_p_fit() -> float:
    """Smoothness p of the unit Rapp curve z/(1+|z|^2p)^(1/2p) that best matches tanh(z) (Gaussian-weighted LSQ)."""
    global _RAPP_P_FIT
    if _RAPP_P_FIT is None:
        z = np.linspace(-4, 4, 2001)
        w = np.exp(-z ** 2 / 2)
        ps = np.arange(0.8, 3.0, 0.01)
        err = [np.sum(w * (z / (1 + np.abs(z) ** (2 * p)) ** (1 / (2 * p)) - np.tanh(z)) ** 2) for p in ps]
        _RAPP_P_FIT = float(ps[int(np.argmin(err))])
    return _RAPP_P_FIT


def resolve_cfg(cfg: dict | None) -> dict:
    """Defaults < (preset=="ideal" overrides) < user keys. Unknown keys are ignored."""
    cfg = dict(cfg or {})
    rc = dict(DEFAULTS)
    if cfg.get("preset") == "ideal":
        rc.update(IDEAL)
    rc.update({k: cfg[k] for k in DEFAULTS if k in cfg})
    if rc["rf_seed"] is None:
        rc["rf_seed"] = int(cfg.get("seed", 0)) + 303
    return rc


# ------------------------------------------------------------------------------------------------ geometry / grid
@dataclass
class Grid:
    U: int
    nsym: int            # window length in symbols (power of two)
    T: float             # symbol period, s
    vp: float            # m/s
    N: int = 0
    fs: float = 0.0
    f: np.ndarray = field(default_factory=lambda: np.zeros(0))
    P: np.ndarray = field(default_factory=lambda: np.zeros(0))   # raised-cosine (beta=1) spectrum, p(0)=1

    def __post_init__(self):
        self.N = self.U * self.nsym
        self.fs = self.U / self.T
        self.f = np.arange(self.N // 2 + 1) * (self.fs / self.N)
        x = np.clip(self.f * self.T, 0, 1)
        self.P = np.where(self.f * self.T <= 1.0, 0.5 * self.T * (1 + np.cos(np.pi * x)), 0.0)


def make_grid(rc: dict, max_lag_symbols: int) -> Grid:
    nsym = 256
    while nsym < 4 * max_lag_symbols + 128:
        nsym *= 2
    return Grid(U=int(rc["oversample"]), nsym=nsym, T=rc["symbol_ns"] * 1e-9, vp=rc["vp_cm_per_ns"] * 1e7)


# ------------------------------------------------------------------------------------------------ line physics
_LINE_CACHE: dict = {}


def _coax_geometry(rc: dict) -> tuple[float, float, float]:
    eps = (C0 / (rc["vp_cm_per_ns"] * 1e7)) ** 2
    dint = 0.9e-3
    dout = dint * np.exp(rc["z0"] * 2 * np.pi * np.sqrt(eps) / ETA0)
    return eps, dint, dout


def _alpha_db_per_us(gamma_real: float, vp: float) -> float:
    return float(gamma_real * 8.685889638 * vp * 1e-6)


_SIGMA_CACHE: dict = {}


def coax_parameters(rc: dict) -> tuple[float, float]:
    """(sigma, tan_delta) such that the Coaxial model has loss_db_per_us at f_ref (conductor ~ sqrt f, dielectric ~ f)."""
    L = float(rc["loss_db_per_us"])
    key = (L, rc["z0"], rc["vp_cm_per_ns"], rc["f_ref_mhz"])
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    if L <= 1e-9:
        out = (1e30, 0.0)
    else:
        eps, dint, dout = _coax_geometry(rc)
        vp = rc["vp_cm_per_ns"] * 1e7
        fr = np.array([rc["f_ref_mhz"] * 1e6])
        tand = 2e-4 * L / 32.0          # dielectric share ~2 % of the RG-58 loss, scaled with the target
        if HAVE_SKRF:
            fq = _rf.Frequency.from_f(fr, unit="hz")
            ad = _alpha_db_per_us(_Coaxial(fq, Dint=dint, Dout=dout, epsilon_r=eps, tan_delta=tand,
                                           sigma=1e30).gamma.real[0], vp)
            ac0 = _alpha_db_per_us(_Coaxial(fq, Dint=dint, Dout=dout, epsilon_r=eps, tan_delta=0.0,
                                            sigma=5.8e7).gamma.real[0], vp)
        else:
            ad = _numpy_alpha_diel(rc, tand)
            ac0 = _numpy_alpha_cond(rc, 5.8e7)
        ac = max(L - ad, 0.05 * L)
        out = (5.8e7 * (ac0 / ac) ** 2, tand)
    _SIGMA_CACHE[key] = out
    return out


def _numpy_alpha_cond(rc: dict, sigma: float) -> float:
    eps, dint, dout = _coax_geometry(rc)
    f = rc["f_ref_mhz"] * 1e6
    rs = np.sqrt(np.pi * f * 4e-7 * np.pi / sigma)
    r = rs / (2 * np.pi) * (1 / (dint / 2) + 1 / (dout / 2))
    a = r / (2 * rc["z0"])
    return _alpha_db_per_us(a, rc["vp_cm_per_ns"] * 1e7)


def _numpy_alpha_diel(rc: dict, tand: float) -> float:
    eps, _, _ = _coax_geometry(rc)
    f = rc["f_ref_mhz"] * 1e6
    return _alpha_db_per_us(np.pi * f * np.sqrt(eps) * tand / C0, rc["vp_cm_per_ns"] * 1e7)


def _line_gamma_numpy(f: np.ndarray, rc: dict) -> np.ndarray:
    """Closed-form propagation constant (fallback): skin effect alpha_s*sqrt(f/f_ref)*(1+j) + dielectric + lossless beta."""
    sigma, tand = coax_parameters(rc)
    vp = rc["vp_cm_per_ns"] * 1e7
    fr = rc["f_ref_mhz"] * 1e6
    a_c = _numpy_alpha_cond(rc, sigma) / (8.685889638 * vp * 1e-6) if sigma < 1e29 else 0.0
    a_d = _numpy_alpha_diel(rc, tand) / (8.685889638 * vp * 1e-6) if tand > 0 else 0.0
    ff = np.maximum(f, 1.0)
    return (a_c * np.sqrt(ff / fr) * (1 + 1j) + a_d * ff / fr + 1j * 2 * np.pi * ff / vp)


def _flat2(freq, s11, s21, s22, z0):
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 0, 0], s[:, 1, 1], s[:, 0, 1], s[:, 1, 0] = s11, s22, s21, s21
    return _rf.Network(frequency=freq, s=s, z0=z0)


def _junction(freq, gam, z0):
    """Lossless mismatch two-port (unitary): S11=Gamma, S22=-Gamma, S21=S12=sqrt(1-Gamma^2)."""
    return _flat2(freq, gam, np.sqrt(1 - gam ** 2), -gam, z0)


def line_response(rc: dict, g: Grid, n_samples: int, n_combiner: int) -> tuple[np.ndarray, float]:
    """Complex frequency response (on the rfft grid g.f) of ONE tap path with line length n_samples*T_s*v_p,
    delay-trimmed to nominal. Returns (H_eff[f], static_gain) where static_gain = coupler * Wilkinson branch amplitude.

    skrf cascade: coupler(c) ** junction(Gamma) ** Coaxial.line ** junction(Gamma) ** Wilkinson branch(1/sqrt(N)).
    H_eff = S21 * exp(+j*2*pi*f*tau_g(f_trim)): the designer trims the physical length so the group delay at f_trim
    equals the nominal k*d*T_s; what remains (dispersion, droop, echoes) is the channel degradation.
    """
    hkey = ("H", rc["loss_db_per_us"], rc["z0"], rc["vp_cm_per_ns"], rc["f_ref_mhz"], rc["trim_mhz"], rc["coupling_db"],
            float(rc["gamma_mismatch"]), g.U, g.nsym, rc["symbol_ns"], n_samples, n_combiner, HAVE_SKRF)
    if hkey in _LINE_CACHE:
        return _LINE_CACHE[hkey]
    c = 10 ** (-rc["coupling_db"] / 20)
    w = 1.0 / np.sqrt(n_combiner)
    gam = float(rc["gamma_mismatch"])
    length = n_samples * g.T * g.vp
    f = g.f[1:]
    if HAVE_SKRF:
        key = (rc["loss_db_per_us"], rc["z0"], rc["vp_cm_per_ns"], rc["f_ref_mhz"], g.U, g.nsym, rc["symbol_ns"], n_samples)
        if key not in _LINE_CACHE:
            sigma, tand = coax_parameters(rc)
            eps, dint, dout = _coax_geometry(rc)
            freq = _rf.Frequency.from_f(f, unit="hz")
            med = _Coaxial(freq, z0_port=rc["z0"], Dint=dint, Dout=dout, epsilon_r=eps, tan_delta=tand, sigma=sigma)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                ln = med.line(length, unit="m")
                beta = np.asarray(med.gamma.imag, dtype=float)
            _LINE_CACHE[key] = (freq, ln, beta)
            if len(_LINE_CACHE) > 400:
                _LINE_CACHE.pop(next(iter(_LINE_CACHE)))
        freq, ln, beta = _LINE_CACHE[key]
        z0 = rc["z0"]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            net = (_flat2(freq, 0, c, 0, z0) ** _junction(freq, gam, z0) ** ln ** _junction(freq, gam, z0)
                   ** _flat2(freq, 0, w, 0, z0))
        s21 = np.asarray(net.s[:, 1, 0], dtype=complex)
    else:
        gm = _line_gamma_numpy(f, rc)
        beta = gm.imag
        e1 = np.exp(-gm * length)
        s21 = c * w * (1 - gam ** 2) * e1 / (1 + gam ** 2 * e1 ** 2)
    k_trim = int(np.argmin(np.abs(f - rc["trim_mhz"] * 1e6)))
    tau_g = float(np.gradient(beta * length, 2 * np.pi * f)[k_trim]) if n_samples > 0 else 0.0
    H = np.empty(len(g.f), dtype=complex)
    H[1:] = s21 * np.exp(1j * 2 * np.pi * f * tau_g)
    H[0] = H[1].real                      # f=0 point is outside the skin-effect model: take the lowest bin
    H[-1] = H[-1].real
    _LINE_CACHE[hkey] = (H, c * w)
    return H, c * w


# ------------------------------------------------------------------------------------------------ channel build
@dataclass
class LayerChan:
    h: np.ndarray                 # float32 [O, I, M]   symbol-rate MIMO FIR incl. gain compensation
    mlo: int                      # lag of h[..., 0] (<= 0)
    comp: np.ndarray              # [K] amplitude compensation per tap index (scalar broadcast unless tap_equalize)
    a: np.ndarray                 # [O, I, K] tap amplitudes (sign, error, leak), without comp
    delta: np.ndarray             # [O, I, K] delay skew, s
    xt: float
    Hk: np.ndarray                # [K, F] complex H_eff * static (uncompensated)
    lags: np.ndarray              # [K] nominal lag k*d in symbols
    info: dict = field(default_factory=dict)


def _fir_from_H(H: np.ndarray, g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Sample (p*h)(jT) and its derivative at symbol lags j in [-nsym/2, nsym/2). Returns (g_sym, dg_sym)."""
    G = g.P * H
    gt = g.fs * np.fft.irfft(G, n=g.N)
    w = 2j * np.pi * g.f
    dgt = g.fs * np.fft.irfft(w * G, n=g.N)
    j = np.arange(-g.nsym // 2, g.nsym // 2)
    idx = (j * g.U) % g.N
    return gt[idx], dgt[idx]


def build_layer(ly: Layer, rc: dict, g: Grid, layer_index: int) -> LayerChan:
    O, I, K = ly.W.shape
    d = int(ly.dilation)
    N = I * K                                   # Wilkinson combiner ports for one output
    lens = [k * d for k in range(K)]
    jaxis = np.arange(-g.nsym // 2, g.nsym // 2)
    Hk, gs, dgs, stat = [], [], [], 0.0
    for L in lens:
        H, stat = line_response(rc, g, L, N)
        Hk.append(H)
        gk, dgk = _fir_from_H(H, g)
        gs.append(gk)
        dgs.append(dgk)
    gs, dgs, Hk = np.array(gs), np.array(dgs), np.array(Hk)

    # --- amplifier gain compensation (design-time, from the nominal ideal-coupler main-lobe gains)
    main = gs[:, g.nsym // 2]                       # j = 0
    n_act = (ly.W != 0).sum(axis=(0, 1)).astype(float)     # active taps per k
    if rc["tap_equalize"]:
        comp = 1.0 / np.maximum(np.abs(main), 1e-12)
    else:
        mean_main = float(np.sum(n_act * np.abs(main)) / max(n_act.sum(), 1.0))
        comp = np.full(K, 1.0 / max(mean_main, 1e-12))

    # --- static random tap errors (common random numbers across sweep points)
    rng = np.random.default_rng([int(rc["rf_seed"]), layer_index, O, I, K])
    z_amp = rng.standard_normal((O, I, K))
    z_ph = rng.standard_normal((O, I, K))
    z_sgn = rng.choice([-1.0, 1.0], size=(O, I, K))
    eps_amp = 10 ** (rc["tap_amp_err_db"] * z_amp / 20)
    leak = 0.0 if not np.isfinite(rc["isolation_db"]) else 10 ** (-rc["isolation_db"] / 20)
    a = np.where(ly.W != 0, ly.W.astype(float), z_sgn * leak) * eps_amp
    delta = (rc["tap_phase_err_deg"] / 360.0) / (rc["f_ref_mhz"] * 1e6) * z_ph

    # --- extent of the impulse responses -> dense lag axis
    gn = gs * comp[:, None]
    keep = np.abs(gn) > TOL
    jlo = np.array([jaxis[k][0] if (k := np.where(keep[i])[0]).size else 0 for i, _ in enumerate(lens)])
    jhi = np.array([jaxis[k][-1] if (k := np.where(keep[i])[0]).size else 0 for i, _ in enumerate(lens)])
    jlo = np.minimum(jlo, -1) - 1
    jhi = np.maximum(jhi, 1) + 1
    lag = np.array(lens)
    mlo = int(min(0, np.min(lag + jlo)))
    mhi = int(np.max(lag + jhi))
    M = mhi - mlo + 1
    Gb = np.zeros((K, M))
    Gd = np.zeros((K, M))
    for ki in range(K):
        sl = slice(lag[ki] + jlo[ki] - mlo, lag[ki] + jhi[ki] - mlo + 1)
        src = slice(jlo[ki] + g.nsym // 2, jhi[ki] + g.nsym // 2 + 1)
        Gb[ki, sl] = gs[ki, src] * comp[ki]
        Gd[ki, sl] = dgs[ki, src] * comp[ki]
    h = np.einsum("oik,km->oim", a, Gb) - np.einsum("oik,oik,km->oim", a, delta, Gd)

    xt = 0.0
    if rc["crosstalk_db"] is not None:
        xt = 10 ** (-abs(float(rc["crosstalk_db"])) / 20)
        hx = h.copy()
        hx[:, 1:, :] += xt * h[:, :-1, :]
        hx[:, :-1, :] += xt * h[:, 1:, :]
        h = hx
    info = dict(required_amp_gain_db=float(-20 * np.log10(np.mean(np.abs(main)) + 1e-30)), M=M, mlo=mlo,
                n_paths=int(O * I * K), n_combiner=N,
                tap_main_gain_db=[float(20 * np.log10(abs(cc * m) + 1e-30)) for cc, m in zip(comp, main)])
    return LayerChan(h=h.astype(np.float32), mlo=mlo, comp=comp, a=a, delta=delta, xt=xt, Hk=Hk, lags=lag, info=info)


def build_channels(model: Model, rc: dict) -> tuple[list[LayerChan], Grid]:
    max_lag = max(int(ly.dilation) * (ly.W.shape[2] - 1) for ly in model.layers)
    g = make_grid(rc, 3 * max_lag + 16)
    return [build_layer(ly, rc, g, l) for l, ly in enumerate(model.layers)], g


# ------------------------------------------------------------------------------------------------ convolution
def _next_fast(n: int) -> int:
    from scipy.fft import next_fast_len
    return int(next_fast_len(n, real=True))


def fir_apply(x: np.ndarray, ch: LayerChan) -> np.ndarray:
    """y[b,o,n] = sum_{i,m} h[o,i,m] x[b,i,n-m]  (m from mlo; zeros outside the record). x [B,I,T] -> [B,O,T]."""
    B, I, T = x.shape
    O, _, M = ch.h.shape
    nfft = _next_fast(T + M - 1)
    Hf = np.fft.rfft(ch.h, n=nfft, axis=-1).astype(np.complex64)   # [O,I,F]
    out = np.empty((B, O, T), dtype=np.float32)
    F = nfft // 2 + 1
    step = max(1, int(3.0e7 // (max(I, O) * F)))
    for s in range(0, B, step):
        X = np.fft.rfft(x[s:s + step].astype(np.float32), n=nfft, axis=-1)          # [b,I,F]
        Y = np.zeros((X.shape[0], O, F), dtype=np.complex64)
        for i in range(I):                                                           # sum over input lines
            Y += X[:, i, None, :] * Hf[None, :, i, :]
        y = np.fft.irfft(Y, n=nfft, axis=-1)                                         # [b,O,nfft]
        out[s:s + step] = y[:, :, -ch.mlo:-ch.mlo + T]
    return out


def amplifier(z: np.ndarray, rc: dict) -> np.ndarray:
    """Unit AM/AM curve of the saturating amplifier, z = (u + bias)/vsat. tanh = exact BJT diff pair; Rapp = generic."""
    if rc["amp_model"] == "tanh":
        return np.tanh(z)
    p = rapp_p_fit() if rc["rapp_p"] == "fit" else float(rc["rapp_p"])
    return z / (1.0 + np.abs(z) ** (2 * p)) ** (1.0 / (2 * p))


def quantize_adc(f: np.ndarray, bits: int | None) -> np.ndarray:
    """Symmetric mid-tread ADC, full scale = max |f| (calibrated on the data, like the E tier)."""
    if bits is None:
        return f
    levels = 2 ** (int(bits) - 1) - 1
    fs = float(np.abs(f).max()) or 1.0
    s = fs / levels
    return (np.clip(np.where(f >= 0, np.floor(f / s + 0.5), -np.floor(-f / s + 0.5)), -levels, levels) * s).astype(np.float32)


def reference_power(model: Model, x: np.ndarray, n: int = 256) -> list[float]:
    """Mean-square amplifier-input level of each stage from the ideal (V) forward on the first n examples."""
    h = np.asarray(x[:n], np.float32) * np.float32(model.input_scale)
    p = []
    for ly in model.layers:
        u = virtual.layer_preact(h, ly) + ly.bias[None, :, None]
        p.append(float(np.mean(u ** 2)))
        h = ly.gain[None, :, None] * np.tanh(u / np.float32(ly.vsat))
    return p


def stage_sigma(p_ref: float, rc: dict) -> float:
    """RMS of the AWGN added at the amplifier input: P_ref / SNR, degraded by the noise figure."""
    return float(np.sqrt(p_ref * 10 ** ((float(rc["nf_db"]) - float(rc["snr_db"])) / 10)))


def forward_with(model: Model, x: np.ndarray, rc: dict, chans: list[LayerChan], p_ref: list[float]) -> dict:
    rng = np.random.default_rng(int(rc["rf_seed"]) + 7)
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    outs = []
    for l, (ly, ch) in enumerate(zip(model.layers, chans)):
        u = fir_apply(h, ch)
        if rc["snr_db"] is not None:
            sigma = stage_sigma(p_ref[l], rc)
            u = u + (sigma * rng.standard_normal(u.shape)).astype(np.float32)
        z = (u + ly.bias[None, :, None]) / np.float32(ly.vsat * rc["amp_vsat_scale"])
        h = (ly.gain[None, :, None] * (rc["amp_gain_scale"] * amplifier(z, rc) + rc["amp_offset_rel"])
             ).astype(np.float32)                                              # re-timing S/H: sampled values
        outs.append(h)
    feats = quantize_adc(virtual.features(model, outs), rc["adc_bits"])
    return {"out": virtual.readout(model, feats.astype(np.float32)), "layers": outs}


def forward(model: Model, x: np.ndarray, cfg: dict | None = None) -> dict:
    """Standalone RF forward pass (builds the channel each call)."""
    rc = resolve_cfg(cfg)
    chans, _ = build_channels(model, rc)
    return forward_with(model, x, rc, chans, reference_power(model, x))


# ------------------------------------------------------------------------------------------------ fine-grid waveforms
def fine_waveform(ly: Layer, ch: LayerChan, g: Grid, x: np.ndarray) -> np.ndarray:
    """Layer combiner output on the fine time grid (U points/symbol) for ONE example x [I,T] (before the amplifier,
    gain-compensated, no noise). y(t) = sum_{i,k} a e^{-j w skew} H_k(f) e^{-j w k d T_s} [X_i + xt X_neighbours]."""
    I, T = x.shape
    O, _, K = ly.W.shape
    u = np.zeros((I, g.N))
    u[:, :T * g.U:g.U] = x
    X = np.fft.rfft(u, axis=-1) * (g.fs * g.P)[None, :]                    # pulse-shaped line inputs
    Xl = X.copy()
    if ch.xt:
        Xl[1:] += ch.xt * X[:-1]
        Xl[:-1] += ch.xt * X[1:]
    w = 2 * np.pi * g.f
    Y = np.zeros((O, len(g.f)), dtype=complex)
    for k in range(K):
        dly = np.exp(-1j * w * ch.lags[k] * g.T)
        Hk = ch.Hk[k] * ch.comp[k] * dly
        # sum over i with per-path skew phase: [O,I,F]
        ph = np.exp(-1j * w[None, None, :] * ch.delta[:, :, k][:, :, None])
        Y += np.einsum("oi,oif,if->of", ch.a[:, :, k], ph, Xl) * Hk[None, :]
    return np.fft.irfft(Y, n=g.N, axis=-1)


# ------------------------------------------------------------------------------------------------ cascade hook
_CASCADE_PY = Path(__file__).resolve().parents[1] / "cascade.py"


def _apply_cascade(cfg: dict, warnings: list[str], inputs_from: list[str]) -> dict:
    p = cfg.get("cascade_params")
    if not p:
        return cfg
    casc = _CASCADE_PY
    pp = Path(p)
    if not casc.exists():
        warnings.append(f"cascade_params={p} given but dlnn/cascade.py does not exist yet; ignored")
        return cfg
    if not pp.exists():
        warnings.append(f"cascade_params file {p} not found; ignored")
        return cfg
    try:
        from .. import cascade
        extracted = cascade.load_extracted(pp) if hasattr(cascade, "load_extracted") else json.loads(pp.read_text())
        new = cascade.apply_to_rf_cfg(dict(cfg), extracted)
        new = dict(cfg if new is None else new)
        cc = new.get("cascade_circuit") or {}
        if "amp_offset_rel" not in cfg and cc.get("amp"):
            new["amp_offset_rel"] = float(cc["amp"].get("offset_rel", 0.0))
        added = [q for q in new.get("inputs_from", []) if q not in cfg.get("inputs_from", [])]
        inputs_from.extend(added or [str(p)])
        src = cc.get("source") or extracted.get("source", "?")
        warnings.append(f"cascade: loss_db_per_us={new.get('loss_db_per_us')} and amp scales taken from {p} "
                        f"(circuit extraction source={src}); the SPICE tap_H samples are NOT used directly "
                        "(layers 2-3 need delays beyond the layer-1 unit cell)")
        return new
    except Exception as e:  # keep the tier alive; say so
        warnings.append(f"cascade hook failed ({type(e).__name__}: {e}); cascade ignored")
        return cfg


# ------------------------------------------------------------------------------------------------ tier entry
def _sweep_plan(task: str, rc: dict, cfg: dict) -> list[tuple[str, list, str]]:
    snrs = [float(s) for s in cfg.get("sweep_snrs", common.DEFAULT_SNRS[task])]
    plan = [("snr_db", snrs, "snr_db"),
            ("loss_db_per_us", [0.0, 5.0, 14.0, 32.0], "loss_db_per_us"),
            ("isolation_db", [15.0, 20.0, 25.0, 30.0, 35.0, 40.0], "isolation_db"),
            ("tap_amp_err_db", [0.0, 0.25, 0.5, 1.0, 2.0], "tap_amp_err_db"),
            ("adc_bits", [4, 5, 6, 7, 8, 9, 10], "adc_bits")]
    return plan


def run(model_npz: str | Path, dataset: dict, cfg: dict, outdir: str | Path) -> contract.TierResult:
    t0 = time.time()
    model = load_model(model_npz)
    warns: list[str] = []
    inputs_from: list[str] = []
    cfg = _apply_cascade(dict(cfg), warns, inputs_from)
    rc = resolve_cfg(cfg)
    x = dataset["x_test"]
    clf = dataset["task"] == "classification"
    p_ref = reference_power(model, x)
    cache: dict = {}

    def chans_for(r: dict):
        key = tuple(r[k] for k in ("loss_db_per_us", "isolation_db", "tap_amp_err_db", "tap_phase_err_deg",
                                   "gamma_mismatch", "crosstalk_db", "tap_equalize", "coupling_db", "rf_seed",
                                   "z0", "oversample", "vp_cm_per_ns", "symbol_ns", "f_ref_mhz", "trim_mhz"))
        if key not in cache:
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            cache[key] = build_channels(model, r)
        return cache[key]

    def fwd(xx: np.ndarray, c: dict) -> dict:
        r = rc if c is cfg else resolve_cfg({**cfg, **c})
        return forward_with(model, xx, r, chans_for(r)[0], p_ref)

    ref_out = virtual.forward(model, x, cfg)["out"]
    extra_arrays: dict = {}

    def extra_sweeps(xx: np.ndarray, sc) -> list[dict]:
        entries = []
        base = dict(IDEAL) if rc_base_ideal else {}
        for axis, values, key in _sweep_plan(dataset["task"], rc, cfg):
            if axis == "snr_db" and cfg.get("snr_sweep_mode", rc["snr_sweep_mode"]) == "input":
                from .. import sweeps as _sw
                entry = _sw.snr_sweep(lambda xn: fwd(xn, cfg), xx, sc, values, seed=int(cfg.get("seed", 0)))
                entries.append(entry)
                continue
            rows, agree = [], []
            for v in values:
                o = fwd(xx, {**base, key: v})
                rows.append(sc(o))
                if clf:
                    agree.append(float(np.mean(o["out"].argmax(1) == ref_out.argmax(1))))
                else:
                    yr, yo = ref_out[0], o["out"][0]
                    sl = slice(int(dataset.get("washout", 0)), None)
                    agree.append(float(np.mean((yo[:, sl] - yr[:, sl]) ** 2) / np.var(yr[:, sl])))
            e = {"axis": axis, "values": [float(v) for v in values]}
            for m in ("accuracy", "mean_kl", "nmse"):
                if m in rows[0]:
                    e[m] = [float(r[m]) for r in rows]
            entries.append(e)
            name = "top1_agreement_vs_virtual" if clf else "nmse_vs_virtual"
            extra_arrays[f"sweep.{axis}.values"] = np.array(values, dtype=np.float64)
            extra_arrays[f"sweep.{axis}.{name}"] = np.array(agree, dtype=np.float64)
        return entries

    rc_base_ideal = str(rc["sweep_baseline"]) == "ideal"
    chans, grid = chans_for(rc)
    status = "ok" if HAVE_SKRF else "fallback"
    tool = {"name": "scikit-rf" if HAVE_SKRF else "numpy-rlgc-fallback",
            "version": SKRF_VERSION if HAVE_SKRF else np.__version__}
    omits = ["no EM / layout / connector effects: channel-level two-port cascades only",
             "no hardware measurement of any kind; coax is a generic RG-58-like model (not vendor data)",
             "amplifier is memoryless (no bandwidth limit, no AM/PM, no bias drift); noise is white and added "
             "once per stage at the amplifier input",
             "ideal re-timing (sample at the symbol instants, no jitter or clock skew) and ideal pulse regeneration "
             "between layers; input DAC/pulse generator is ideal",
             "crosstalk is flat and only between neighbouring input lines (no NEXT/FEXT frequency shape); "
             "hybrid isolation is not modelled separately from the coupler isolation_db",
             "the snr_db sweep axis of tier R is the per-stage amplifier SNR, not input AWGN as in V/E"]
    if not HAVE_SKRF:
        omits.insert(0, "FALLBACK: scikit-rf is not installed; closed-form numpy RLGC/skin-effect line used instead")
        warns.append("scikit-rf unavailable: closed-form numpy RLGC fallback used (status=fallback)")
    models = [f"{tool['name']} {tool['version']}: skrf.media.Coaxial lossy dispersive line (skin effect ~sqrt(f), "
              f"{rc['loss_db_per_us']:g} dB/us at {rc['f_ref_mhz']:g} MHz, v_p={rc['vp_cm_per_ns']:g} cm/ns) per tap path"
              if HAVE_SKRF else "closed-form skin-effect RLGC coax per tap path",
              f"coupler tap ({rc['coupling_db']:g} dB, isolation {rc['isolation_db']:g} dB, amp err sigma "
              f"{rc['tap_amp_err_db']:g} dB, phase err sigma {rc['tap_phase_err_deg']:g} deg as delay skew)",
              f"mismatch Gamma={rc['gamma_mismatch']:g} at both line ends (multi-reflection ripple), 180-degree "
              "hybrid sign, Wilkinson N-way combiner (-10log10 N dB)",
              f"amplifier gain compensating mean tap loss, {rc['amp_model']} AM/AM, stage SNR {rc['snr_db']} dB, "
              f"NF {rc['nf_db']:g} dB",
              f"neighbour crosstalk {rc['crosstalk_db']} dB, readout ADC {rc['adc_bits']} bits",
              f"IFFT impulse responses on a fine grid ({rc['oversample']} pts/symbol), raised-cosine beta=1 pulses, "
              "ideal symbol-rate re-timing between layers"]
    res = common.evaluate_tier(
        tier="R", status=status, tool=tool, model=model, dataset=dataset, cfg=cfg,
        forward_fn=lambda xx, c: fwd(xx, cfg), ref_fn=None,
        fidelity={"models": models, "omits": omits}, sweep_axes=(), extra_sweeps=extra_sweeps, t0=t0,
        inputs_from=inputs_from, warnings=warns)
    res.arrays.update(extra_arrays)
    # fine-grid layer-1 combiner waveforms for the first examples (contract: waveforms.<example_id>)
    for e in range(min(int(rc["n_waveforms"]), len(x))):
        nw = max(8, min(x.shape[-1], grid.nsym - chans[0].info["M"] - 16))      # keep the record inside the IFFT window
        xe = np.asarray(x[e][:, :nw], np.float32) * np.float32(model.input_scale)
        res.arrays[f"waveforms.{e}"] = fine_waveform(model.layers[0], chans[0], grid, xe)[
            :, :(xe.shape[1] + chans[0].info["M"] + 16) * grid.U].astype(np.float32)
    res.warnings.append(
        "Channel-level simulation (scikit-rf two-ports + numpy convolution): no EM/layout effects, no hardware "
        "measurement. Required amplifier gain per layer (dB): "
        + ", ".join(f"L{l}={c.info['required_amp_gain_db']:.1f}" for l, c in enumerate(chans)))
    res.runtime_s = round(time.time() - t0, 4)
    return res
