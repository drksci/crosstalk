"""Tier C (circuit): ngspice unit cell + parameter extraction + circuit-calibrated behavioural model.

What SPICE (ngspice) computes here, and what it does not
--------------------------------------------------------
SPICE computes, for ONE layer-1 unit cell only (a K-tap bank of lossy RLC coax lines, ideal +-1/0 VCVS
taps into a summing node, a generic BJT differential pair, an RC integrator):
  (a) `.ac`  : the complex transfer function H_k(f) of every tap line (loss, delay error, dispersion);
      `.dc`  : the diff-pair transfer curve, fitted with A*tanh(v/V_sat) + offset (R^2, V_sat, gain);
  (b) `.tran`: the unit cell driven end-to-end by PAM pulse trains for `c_n_examples` (16) test examples;
      the sampled layer-1 activations are compared with the virtual tier (NMSE).
SPICE does NOT compute the full test set, layers 2-3, or the readout.  The full test set is run by the
CIRCUIT-CALIBRATED BEHAVIOURAL MODEL: an LTI line model (fitted to the SPICE H_k(f), evaluated on a 1 ns
grid for each tap delay k*d*T_s, sampled with the same symbol timing as the .tran stimulus) + tanh with the
fitted (V_sat, gain, offset).  Its numbers are never "SPICE results".

If ngspice is missing or fails, the tier falls back to a numpy closed-form RLGC line + ideal tanh and is
labelled status="fallback" / tool "numpy-analytic-rlgc".

Symbol timing used by stimulus AND behavioural model: symbol n occupies [n*Ts, (n+1)*Ts]; its leading edge
is a linear ramp of `T_RISE` starting at n*Ts; the output is sampled at n*Ts + SAMPLE_PHASE*Ts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import contract, sweeps
from ..model.npz import Layer, Model, load_model
from . import common, virtual

SPICE_DIR = Path(__file__).resolve().parents[2] / "hw" / "spice"
EXTRACT_SCHEMA = "dlnn-circuit-extracted/1"
EXTRACT_REL_PATH = "params/circuit_extracted.json"

TS = 10e-9                 # symbol period (s)
T_RISE = 1e-9              # PAM edge (linear ramp)
SAMPLE_PHASE = 0.8         # sampling instant inside the symbol
DT = 1e-9                  # grid of the behavioural line model
NFFT = 4096                # 4.096 us window -> 244 kHz bins up to 500 MHz
V_T_DESIGN = 0.0259        # V (plan)
Z0 = 50.0
L_PER_M = 250e-9
C_PER_M = 100e-12
VP = 1.0 / math.sqrt(L_PER_M * C_PER_M)          # 2.0e8 m/s
RG58_DB_PER_US = 32.0      # at 100 MHz (plan)
RL_OHM = 1000.0            # diff-pair load resistors
BF_NOM = 100.0
ALPHA_NOM = BF_NOM / (BF_NOM + 1.0)
VCM = 2.0                  # input common-mode DC (V)
VCC = 5.0
POOL_R = 1000.0
DB_PER_NP = 20.0 / math.log(10.0)
AC_NPTS = 1024
AC_FMAX = 500e6

GENERIC_NOTICE = ("GENERIC textbook models only: RLC coax (L=250 nH/m, C=100 pF/m, constant R) and a generic NPN "
                  "diff pair; not vendor models, no measured hardware.")


class SpiceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------- ngspice plumbing
def ngspice_path() -> str | None:
    return shutil.which("ngspice")


def ngspice_version() -> str:
    p = ngspice_path()
    if p is None:
        return "unavailable"
    try:
        out = subprocess.run([p, "-v"], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout
    except Exception:  # noqa: BLE001
        return "unknown"
    m = re.search(r"ngspice-(\d+(?:\.\d+)*)", out)
    return m.group(1) if m else "unknown"


def read_raw(path: str | Path) -> list[dict]:
    """Minimal binary ngspice raw-file reader. Returns [{plot, names(lower), data[npts, nvars]}] (complex for .ac)."""
    b = Path(path).read_bytes()
    plots, pos = [], 0
    while pos < len(b):
        hdr: dict[str, str] = {}
        names: list[str] = []
        while True:
            e = b.index(b"\n", pos)
            line = b[pos:e].decode("latin1")
            pos = e + 1
            if line.startswith("Variables:"):
                nv = int(hdr["No. Variables"])
                rest = line[len("Variables:"):].split()
                for j in range(nv):
                    if j == 0 and rest:
                        t = rest
                    else:
                        e = b.index(b"\n", pos)
                        t = b[pos:e].decode("latin1").split()
                        pos = e + 1
                    names.append(t[1].lower())
                continue
            if line.startswith("Binary:"):
                break
            if ":" in line:
                k, v = line.split(":", 1)
                hdr[k.strip()] = v.strip()
        nv, npts = int(hdr["No. Variables"]), int(hdr["No. Points"])
        cplx = "complex" in hdr.get("Flags", "")
        per = nv * (2 if cplx else 1)
        a = np.frombuffer(b, dtype="<f8", count=per * npts, offset=pos).reshape(npts, per)
        pos += 8 * per * npts
        if cplx:
            a = a[:, 0::2] + 1j * a[:, 1::2]
        plots.append({"plot": hdr["Plotname"], "names": names, "data": a})
    return plots


def run_ngspice(netlist: Path, timeout: float = 600.0) -> list[dict]:
    """Run `ngspice -b` on a netlist, return parsed raw plots. Raises SpiceError on any failure."""
    exe = ngspice_path()
    if exe is None:
        raise SpiceError("ngspice not found on PATH")
    raw, log = netlist.with_suffix(".raw"), netlist.with_suffix(".log")
    try:
        p = subprocess.run([exe, "-b", "-r", str(raw), "-o", str(log), str(netlist)], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise SpiceError(f"ngspice timeout on {netlist.name}") from e
    txt = log.read_text(errors="replace") if log.exists() else p.stdout
    bad = [l for l in txt.splitlines() if re.search(r"fatal|singular|aborted|no such|not supported|unknown", l, re.I)]
    if p.returncode != 0 or bad or not raw.exists():
        raise SpiceError(f"ngspice failed on {netlist.name}: {'; '.join(bad[:3]) or p.stderr[-200:]}")
    plots = read_raw(raw)
    if not plots or plots[0]["data"].shape[0] == 0:
        raise SpiceError(f"ngspice produced no data for {netlist.name}")
    return plots


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


def lib_hashes() -> dict[str, str]:
    return {n: hashlib.sha256((SPICE_DIR / n).read_bytes()).hexdigest() for n in ("coax_rg58.lib", "diffpair.lib")}


# ----------------------------------------------------------------------------------------- analytic line
def r_from_loss(loss_db_per_us: float) -> float:
    """Series R (ohm/m) giving matched-line attenuation `loss_db_per_us` for Z0=50 ohm, v_p=2e8 m/s."""
    alpha_np_per_m = loss_db_per_us / DB_PER_NP / (VP * 1e-6)
    return 2.0 * Z0 * alpha_np_per_m


def line_H(f: np.ndarray, length_m: float, R: float, L: float = L_PER_M, C: float = C_PER_M,
           G: float = 0.0, zl: float = Z0) -> np.ndarray:
    """Closed-form H = V_load/V_source of an RLGC line fed by an ideal source, loaded by `zl` (ohm)."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 1.0)      # 1 Hz stands in for DC (as in the .ac sweep)
    if length_m <= 0:
        return np.ones(f.shape, dtype=complex)
    w = 2 * np.pi * f
    zs, yp = R + 1j * w * L, G + 1j * w * C
    gam, z0c = np.sqrt(zs * yp), np.sqrt(zs / yp)
    gl = gam * length_m
    return zl / (zl * np.cosh(gl) + z0c * np.sinh(gl))


def tap_lengths(K: int, d: int) -> np.ndarray:
    """Coax length (m) of tap k: chosen so that delay = k*d*Ts exactly (v_p = 1/sqrt(LC))."""
    return np.arange(K) * d * TS * VP


def fit_line(freq: np.ndarray, H: np.ndarray, lengths: np.ndarray, nominal_r: float) -> dict:
    """Least-squares fit of (R, L, C) of the closed-form terminated line to the extracted H_k(f), k>=1."""
    from scipy.optimize import least_squares
    ks = [k for k in range(len(lengths)) if lengths[k] > 0]
    sel = freq <= AC_FMAX

    def resid(p):
        R, L, C = p[0] * max(nominal_r, 1e-3), p[1] * L_PER_M, p[2] * C_PER_M
        r = [(line_H(freq[sel], lengths[k], R, L, C) - H[k, sel]) for k in ks]
        r = np.concatenate(r)
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, x0=[1.0, 1.0, 1.0], x_scale=[0.1, 0.01, 0.01], bounds=([0.0, 0.5, 0.5], [10, 2, 2]))
    R, L, C = sol.x[0] * max(nominal_r, 1e-3), sol.x[1] * L_PER_M, sol.x[2] * C_PER_M
    err = np.abs(np.stack([line_H(freq[sel], lengths[k], R, L, C) - H[k, sel] for k in ks]))
    vp = 1 / math.sqrt(L * C)
    z0 = math.sqrt(L / C)
    # matched-line loss at 100 MHz from the fitted R (alpha = Re gamma)
    w = 2 * np.pi * 100e6
    gam = np.sqrt((R + 1j * w * L) * (1j * w * C))
    return {"R_ohm_per_m": float(R), "L_h_per_m": float(L), "C_f_per_m": float(C), "G_s_per_m": 0.0,
            "z0_ohm": z0, "vp_m_per_s": vp, "termination_ohm": Z0,
            "loss_db_per_us_100mhz": float(gam.real * DB_PER_NP * vp * 1e-6),
            "fit_rms_abs_err": float(np.sqrt(np.mean(err ** 2))), "fit_max_abs_err": float(err.max())}


# ----------------------------------------------------------------------------------------- behavioural model
def _pulse() -> np.ndarray:
    """Unit symbol (trapezoid, leading ramp T_RISE, trailing ramp T_RISE after Ts) on the DT grid, length NFFT."""
    t = np.arange(NFFT) * DT
    p = np.clip(t / T_RISE, 0, 1) - np.clip((t - TS) / T_RISE, 0, 1)
    return p


def tap_response(Hfun, max_lag: int) -> np.ndarray:
    """Sampled response s[m], m=0..max_lag, of a line with transfer Hfun(f) to one unit symbol."""
    f = np.fft.rfftfreq(NFFT, DT)
    Hf = Hfun(f)
    Hf[-1] = Hf[-1].real
    y = np.fft.irfft(np.fft.rfft(_pulse()) * Hf, n=NFFT)
    idx = np.round((np.arange(max_lag + 1) + SAMPLE_PHASE) * TS / DT).astype(int)
    idx = idx[idx < NFFT]
    s = np.zeros(max_lag + 1)
    s[:len(idx)] = y[idx]
    return s


class LineModel:
    """Closed-form (fitted) RLC line used by the calibrated behavioural model. R is scalable (loss sweeps)."""

    def __init__(self, R: float, L: float = L_PER_M, C: float = C_PER_M):
        self.R, self.L, self.C = float(R), float(L), float(C)

    @property
    def vp(self) -> float:
        return 1 / math.sqrt(self.L * self.C)

    def with_loss(self, loss_db_per_us: float, ref_loss_db_per_us: float) -> "LineModel":
        return LineModel(self.R * (loss_db_per_us / ref_loss_db_per_us) if ref_loss_db_per_us > 0 else 0.0,
                         self.L, self.C)

    def tap_fir(self, k: int, d: int, tail: int = 80) -> np.ndarray:
        """Sampled response of the tap with delay k*d*Ts (length of line cut to give exactly that delay)."""
        length = k * d * TS * self.vp
        max_lag = min(k * d + tail, int(NFFT * DT / TS) - 1)
        return tap_response(lambda f: line_H(f, length, self.R, self.L, self.C), max_lag)


def effective_kernel(ly: Layer, line: LineModel, trim: float = 1.0, equalise: bool = False) -> tuple:
    """(W_eff[o,i,lag], S[k,lag]) with W_eff = trim * sum_k W[o,i,k] * s_k[lag]  (sampled impulse response of the
    tap network). equalise=True divides every tap by its own main-cursor gain (ideal per-tap repeater/equaliser)."""
    O, I, K = ly.W.shape
    firs = [line.tap_fir(k, ly.dilation) for k in range(K)]
    nlag = max(len(s) for s in firs)
    S = np.zeros((K, nlag))
    for k, s in enumerate(firs):
        S[k, :len(s)] = s / (s[k * ly.dilation] if equalise and abs(s[k * ly.dilation]) > 1e-12 else 1.0)
    return trim * np.einsum("oik,kl->oil", ly.W.astype(np.float64), S), S


def cursor_trim(ly: Layer, line: LineModel) -> float:
    """1 / (|W|-weighted mean main-cursor gain of the active taps): the ideal gain trim that undoes the mean tap loss."""
    O, I, K = ly.W.shape
    cur = np.array([line.tap_fir(k, ly.dilation)[k * ly.dilation] for k in range(K)])
    wk = np.abs(ly.W.astype(np.float64)).sum(axis=(0, 1))
    return float(wk.sum() / max((wk * cur).sum(), 1e-12))


def _conv_fft(x: np.ndarray, Wk: np.ndarray) -> np.ndarray:
    """Causal conv: x [B,I,T], Wk [O,I,Lag] -> [B,O,T]."""
    B, I, T = x.shape
    n = 1 << int(np.ceil(np.log2(T + Wk.shape[2] - 1)))
    X = np.fft.rfft(x.astype(np.float32), n=n, axis=2)
    Wf = np.fft.rfft(Wk.astype(np.float32), n=n, axis=2)
    U = np.einsum("oif,bif->bof", Wf, X, optimize=True)
    return np.fft.irfft(U, n=n, axis=2)[:, :, :T]


class AmpModel:
    """tanh(v/vsat_scale) with fitted gain/offset (relative to the design diff pair)."""

    def __init__(self, vsat_scale: float = 1.0, gain_scale: float = 1.0, offset_rel: float = 0.0):
        self.vsat_scale, self.gain_scale, self.offset_rel = float(vsat_scale), float(gain_scale), float(offset_rel)


COMP_MODES = ("none", "layer", "tap")


def calibrated_forward(model: Model, x: np.ndarray, line: LineModel, amp: AmpModel, comp: str = "layer",
                       trims: list[float] | None = None) -> dict:
    """Circuit-calibrated behavioural forward pass (NOT SPICE). x [B,C,T] -> {'out','layers'}.

    comp = loss compensation: "none" (raw line loss), "layer" (one ideal gain trim per layer = what a single
    amplifier stage can do; default), "tap" (ideal per-tap repeater/equaliser: every tap's main cursor -> 1)."""
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    outs = []
    for l, ly in enumerate(model.layers):
        trim = (trims[l] if trims is not None else cursor_trim(ly, line)) if comp == "layer" else 1.0
        Wk, _ = effective_kernel(ly, line, trim, equalise=(comp == "tap"))
        u = _conv_fft(h, Wk) + ly.bias[None, :, None]
        y = ly.gain[None, :, None] * (amp.gain_scale * np.tanh(u / (ly.vsat * amp.vsat_scale)) + amp.offset_rel)
        h = y
        outs.append(y.astype(np.float32))
    h32 = [o for o in outs]
    return {"out": virtual.readout(model, virtual.features(model, h32)), "layers": h32}


# ----------------------------------------------------------------------------------------- netlist generation
def _f(v: float) -> str:
    return f"{v:.9g}"


def _pwl(samples: np.ndarray, volts_per_unit: float) -> str:
    """PWL text for symbols n=0.. (value samples[n]*volts_per_unit); leading linear ramp T_RISE at n*Ts."""
    pts, prev = [(0.0, 0.0)], 0.0
    for n, v in enumerate(samples * volts_per_unit):
        pts += [(n * TS + T_RISE, float(v))]
        if n + 1 < len(samples):
            pts += [((n + 1) * TS, float(v))]
        prev = v
    pts += [(len(samples) * TS, float(prev))]
    chunks = [" ".join(f"{_f(t)} {_f(v)}" for t, v in pts[i:i + 6]) for i in range(0, len(pts), 6)]
    return "PWL(" + "\n+ ".join(chunks) + ")"


_REL_INCLUDES = False   # True only while regenerating the checked-in hw/spice/unit_cell.cir


def _header(title: str, r_ohm_per_m: float) -> list[str]:
    inc = (lambda n: n) if _REL_INCLUDES else (lambda n: (SPICE_DIR / n).as_posix())
    return [f"* {title}", f"* {GENERIC_NOTICE}", "* GENERATED by dlnn.tiers.circuit - do not hand-edit",
            f".param coax_r={_f(r_ohm_per_m)}", f".include {inc('coax_rg58.lib')}", f".include {inc('diffpair.lib')}"]


def netlist_ac(K: int, d: int, r_ohm_per_m: float) -> str:
    """K tap lines (k=1..K-1; tap 0 is the source node itself), each loaded by 50 ohm; .ac for H_k(f)."""
    L = _header(f"DLNN tap-line AC extraction, K={K}, d={d}", r_ohm_per_m)
    L.append("V1 in 0 DC 0 AC 1")
    for k, ln in enumerate(tap_lengths(K, d)):
        if k == 0:
            continue
        L += [f"Xline{k} in tap{k} coax_rg58 len={_f(ln)}", f"Rterm{k} tap{k} 0 {_f(Z0)}"]
    L += [f".ac lin {AC_NPTS + 1} 1 {_f(AC_FMAX)}",
          ".save " + " ".join(f"v(tap{k})" for k in range(1, K)),
          ".end"]
    return "\n".join(L) + "\n"


def netlist_dc(vmax: float = 0.3, step: float = 0.002, iee: float = 1e-3) -> str:
    L = _header("DLNN diff-pair DC transfer", r_from_loss(RG58_DB_PER_US))
    L += [f"Vcc vcc 0 DC {_f(VCC)}", f"Vcm cm 0 DC {_f(VCM)}", "Vd d 0 DC 0",
          "Ep inp cm d 0 0.5", "En inn cm d 0 -0.5",
          f"Xdp inp inn outp outn vcc diffpair rl={_f(RL_OHM)} iee={_f(iee)}",
          "Eo vo 0 outp outn 1", f".dc Vd {_f(-vmax)} {_f(vmax)} {_f(step)}", ".save v(vo) v(d)", ".end"]
    return "\n".join(L) + "\n"


def netlist_unit_cell(ly: Layer, outputs: list[int], x_units: np.ndarray, volts_per_unit: float,
                      r_ohm_per_m: float, trim: float = 1.0, with_pool: bool = True,
                      tap_gain: np.ndarray | None = None) -> str:
    """Layer-1 unit cell(s): shared input lines/taps, one summing node + diff pair + RC pool per output.

    x_units [I,N]: the model-unit input samples (already multiplied by input_scale). Volts = unit*volts_per_unit.
    u_V = volts_per_unit * u_units.  Pre-amp: v_dp = 2 V_T * (u_V/(volts_per_unit*trim^-1) + bias)/vsat  (see code).
    Output volts == model units (1 V per unit): Iee_o = gain_o / (Rl * alpha_nom).
    """
    O, I, K = ly.W.shape
    N = x_units.shape[1]
    L = _header(f"DLNN layer-1 unit cell: {I} input line(s) x {K} taps (d={ly.dilation}), outputs {outputs}",
                r_ohm_per_m)
    L += [f"Vcc vcc 0 DC {_f(VCC)}", f"Vcm cm 0 DC {_f(VCM)}"]
    lens = tap_lengths(K, ly.dilation)
    used = {(i, k) for o in outputs for i in range(I) for k in range(K) if ly.W[o, i, k] != 0}
    for i in range(I):
        L.append(f"Vin{i} in{i} 0 {_pwl(x_units[i], volts_per_unit)}")
    for (i, k) in sorted(used):
        if k == 0:
            L.append(f"* tap {i},0: direct connection (zero delay)")
            continue
        L += [f"Xline_{i}_{k} in{i} tap_{i}_{k} coax_rg58 len={_f(lens[k])}", f"Rterm_{i}_{k} tap_{i}_{k} 0 {_f(Z0)}"]
    saves = []
    for o in outputs:
        nodes = [(i, k, int(ly.W[o, i, k])) for i in range(I) for k in range(K) if ly.W[o, i, k] != 0]
        prev = "0"
        for j, (i, k, w) in enumerate(nodes):
            tap = f"in{i}" if k == 0 else f"tap_{i}_{k}"
            cur = f"sum{o}" if j == len(nodes) - 1 else f"s{o}_{j}"
            L.append(f"Etap_{o}_{j} {cur} {prev} {tap} 0 {_f(w * (1.0 if tap_gain is None else tap_gain[k]))}")
            prev = cur
        sumnode = f"sum{o}" if nodes else "0"
        apre = 2 * V_T_DESIGN * trim / (ly.vsat * volts_per_unit)
        bias_dp = 2 * V_T_DESIGN * float(ly.bias[o]) / ly.vsat
        iee = float(ly.gain[o]) / (RL_OHM * ALPHA_NOM)
        L += [f"Epre{o} vdp{o} bn{o} {sumnode} 0 {_f(apre)}", f"Vb{o} bn{o} 0 DC {_f(bias_dp)}",
              f"Ep{o} inp{o} cm vdp{o} 0 0.5", f"En{o} inn{o} cm vdp{o} 0 -0.5",
              f"Xdp{o} inp{o} inn{o} outp{o} outn{o} vcc diffpair rl={_f(RL_OHM)} iee={_f(iee)}",
              f"Eout{o} y{o} 0 outp{o} outn{o} 1"]
        saves.append(f"v(y{o})")
        if with_pool:
            L += [f"Rpool{o} y{o} p{o} {_f(POOL_R)}", f"Cpool{o} p{o} 0 {_f(N * TS / POOL_R)}"]
            saves.append(f"v(p{o})")
    saves += [f"v(in{i})" for i in range(I)]
    L += [f".tran 0.5n {_f(N * TS)} 0 0.5n", ".save " + " ".join(saves), ".end"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------- extraction (a)
def _group_delay_error(freq: np.ndarray, h: np.ndarray, tau: float) -> float:
    """Delay error (s): -d(phase)/d(omega) of H*exp(+j w tau) over 5..60 MHz (excess group delay vs nominal)."""
    m = (freq >= 5e6) & (freq <= 60e6)
    ph = np.unwrap(np.angle(h[m] * np.exp(1j * 2 * np.pi * freq[m] * tau)))
    return float(-np.polyfit(2 * np.pi * freq[m], ph, 1)[0])


def extract_ac(workdir: Path, K: int, d: int, r_nom: float, use_spice: bool) -> dict:
    lens = tap_lengths(K, d)
    nl = netlist_ac(K, d, r_nom)
    p = workdir / "tap_lines_ac.cir"
    if use_spice:
        p.write_text(nl)
        plots = run_ngspice(p)
        pl = next(q for q in plots if q["plot"].lower().startswith("ac"))
        names = pl["names"]
        freq = pl["data"][:, names.index("frequency")].real
        H = np.ones((K, len(freq)), dtype=complex)
        for k in range(1, K):
            H[k] = pl["data"][:, names.index(f"v(tap{k})")]
    else:
        freq = np.linspace(1, AC_FMAX, AC_NPTS + 1)
        H = np.stack([line_H(freq, lens[k], r_nom) for k in range(K)])
    fit = fit_line(freq, H, lens, r_nom)
    taps = []
    i100 = int(np.argmin(np.abs(freq - 100e6)))
    for k in range(K):
        tau = k * d * TS
        taps.append({"k": k, "delay_nominal_s": tau, "length_m": float(lens[k]),
                     "dc_gain_db": float(20 * np.log10(abs(H[k, 0]))),
                     "gain_100mhz_db": float(20 * np.log10(abs(H[k, i100]))),
                     "delay_error_s": 0.0 if k == 0 else _group_delay_error(freq, H[k], tau)})
    return {"freq_hz": freq, "H": H, "fit": fit, "taps": taps, "netlist": nl}


def tanh_fit(v: np.ndarray, y: np.ndarray) -> dict:
    from scipy.optimize import curve_fit

    def fn(x, a, vs, off):
        return a * np.tanh(x / vs) + off

    p0 = [float(np.max(np.abs(y))), 2 * V_T_DESIGN, 0.0]
    popt, _ = curve_fit(fn, v, y, p0=p0)
    res = y - fn(v, *popt)
    r2 = 1.0 - float(np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    return {"A_v": float(popt[0]), "vsat_v": float(abs(popt[1])), "offset_v": float(popt[2]), "r2": r2,
            "rms_resid_v": float(np.sqrt(np.mean(res ** 2)))}


def extract_dc(workdir: Path, use_spice: bool) -> dict:
    iee = 1e-3
    nl = netlist_dc(iee=iee)
    if use_spice:
        p = workdir / "diffpair_dc.cir"
        p.write_text(nl)
        pl = run_ngspice(p)[0]
        v, y = pl["data"][:, pl["names"].index("v(d)")].real, pl["data"][:, pl["names"].index("v(vo)")].real
    else:
        v = np.arange(-0.3, 0.3 + 1e-9, 0.002)
        y = RL_OHM * ALPHA_NOM * iee * np.tanh(v / (2 * V_T_DESIGN))
    ft = tanh_fit(v, y)
    ft.update({"iee_a": iee, "rl_ohm": RL_OHM, "vsat_design_v": 2 * V_T_DESIGN,
               "vsat_scale": ft["vsat_v"] / (2 * V_T_DESIGN),
               "gain_scale": ft["A_v"] / (RL_OHM * ALPHA_NOM * iee),
               "offset_rel": ft["offset_v"] / (RL_OHM * ALPHA_NOM * iee),
               "v_t_fitted_v": ft["vsat_v"] / 2, "v_t_ngspice_27c_v": 8.617333262e-5 * 300.15,
               "sweep_v": [float(v[0]), float(v[-1])], "n_points": int(len(v))})
    return {"fit": ft, "netlist": nl, "v": v, "y": y}


# ----------------------------------------------------------------------------------------- unit cell (b)
def pick_examples(dataset: dict, n: int, T: int = 128) -> np.ndarray:
    """[n, C, Tw] test examples (classification: first n; regression: n windows of the test series)."""
    x = np.asarray(dataset["x_test"], np.float32)
    if dataset["task"] == "classification":
        return x[:n]
    Tx = x.shape[2]
    T = min(T, Tx)
    starts = np.linspace(0, Tx - T, n).astype(int)
    return np.stack([x[0, :, s:s + T] for s in starts])


def _run_cell(args) -> dict:
    workdir, name, nl, outputs, N = args
    p = workdir / f"{name}.cir"
    p.write_text(nl)
    pl = run_ngspice(p)[0]
    nm, d = pl["names"], pl["data"]
    t = d[:, nm.index("time")].real
    ts = (np.arange(N) + SAMPLE_PHASE) * TS
    res = {"y": np.stack([np.interp(ts, t, d[:, nm.index(f"v(y{o})")].real) for o in outputs]),
           "pool_end": np.array([np.interp(N * TS, t, d[:, nm.index(f"v(p{o})")].real) for o in outputs]),
           "n_points": int(len(t))}
    tg = np.arange(0, N * TS, DT)
    res["wave"] = np.stack([np.interp(tg, t, d[:, nm.index(f"v(in{i})")].real) for i in range(0, 2)
                            if f"v(in{i})" in nm] + [np.interp(tg, t, d[:, nm.index(f"v(y{o})")].real)
                                                     for o in outputs]).astype(np.float32)
    p.with_suffix(".raw").unlink(missing_ok=True)
    return res


def run_unit_cell(workdir: Path, model: Model, dataset: dict, cfg: dict, r_nom: float, trim: float,
                  line: "LineModel | None" = None, amp: "AmpModel | None" = None, tag: str = "") -> tuple:
    """ngspice .tran of the layer-1 unit cell(s) on n test examples; compare with the virtual tier.

    Returns (summary dict, y_spice [n,n_out,N], y_virtual, examples). If `line`/`amp` are given the summary also
    holds the NMSE of the calibrated behavioural model's layer 1 against the SPICE traces (calibration quality)."""
    ly = model.layers[0]
    n_ex = int(cfg.get("c_n_examples", 16))
    ex = pick_examples(dataset, n_ex)
    xs = ex * np.float32(model.input_scale)                       # [n, I, N] model units
    N = xs.shape[2]
    outputs = list(range(min(ly.W.shape[0], int(cfg.get("c_outputs", 8)))))
    vpu = 0.5 / float(np.abs(xs).max())                           # peak input = +-0.5 V
    jobs = [(workdir, f"unit_cell{tag}_ex{e:02d}", netlist_unit_cell(ly, outputs, xs[e], vpu, r_nom, trim),
             outputs, N) for e in range(n_ex)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(cfg.get("c_workers", 2))) as pool:
        res = list(pool.map(_run_cell, jobs))
    y_c = np.stack([r["y"] for r in res])                          # [n, n_out, N]
    y_v = virtual.layer_forward(xs, ly)[:, outputs, :]
    err = y_c - y_v
    var = y_v.var(axis=(0, 2))                                    # per channel (over examples x time)
    nmse_ch = (err ** 2).mean(axis=(0, 2)) / var
    nmse = float((err ** 2).mean(axis=(0, 2)).sum() / var.sum())
    nmse_energy = float((err ** 2).sum() / (y_v ** 2).sum())
    nmse_cal = None
    if line is not None and amp is not None:
        Wk, _ = effective_kernel(ly, line, trim)
        u = _conv_fft(xs, Wk)[:, outputs] + ly.bias[None, outputs, None]
        y_cal = ly.gain[None, outputs, None] * (amp.gain_scale * np.tanh(u / (ly.vsat * amp.vsat_scale)) + amp.offset_rel)
        nmse_cal = float(((y_cal - y_c) ** 2).sum() / ((y_c - y_c.mean(axis=(0, 2), keepdims=True)) ** 2).sum())
    # RC pool: SPICE end value vs (i) the RC filter applied to the sampled SPICE y, (ii) the uniform time mean
    tau, a = N * TS, math.exp(-TS / (N * TS))
    pool_c = np.stack([r["pool_end"] for r in res])               # [n, n_out]
    w_rc = (1 - a) * a ** np.arange(N - 1, -1, -1)
    y0 = (ly.gain[outputs] * np.tanh(ly.bias[outputs] / ly.vsat))[None, :]       # DC level before the first symbol
    pool_rc = np.einsum("beN,N->be", y_c, w_rc) + y0 * a ** N
    pool_uni = y_v.mean(axis=2)
    out = {"n_examples": n_ex, "outputs": outputs, "volts_per_unit": vpu, "gain_trim": trim,
           "nmse": nmse, "nmse_energy_normalised": nmse_energy, "nmse_per_channel": nmse_ch.tolist(),
           "max_abs_err_units": float(np.abs(err).max()),
           "nmse_calibrated_model_vs_spice": nmse_cal,
           "pool_rc_tau_s": tau,
           "pool_spice_vs_rc_of_sampled_y_nmse": float(((pool_c - pool_rc) ** 2).sum() / (pool_rc ** 2).sum()),
           "pool_rc_vs_uniform_mean_nmse": float(((pool_c - pool_uni) ** 2).sum() / (pool_uni ** 2).sum()),
           "spice_wall_s": round(time.time() - t0, 2), "tran_timepoints_per_run": res[0]["n_points"],
           "tran_options": ".tran 0.5n N*Ts 0 0.5n (tmax=0.5 ns), ngspice defaults otherwise",
           "wave": np.stack([r["wave"] for r in res[:4]])}
    return out, y_c, y_v, ex


# ----------------------------------------------------------------------------------------- reference netlist
def write_reference_netlist(path: str | Path | None = None) -> Path:
    """Regenerate hw/spice/unit_cell.cir: output 0 of a small fixed demo layer driven by a short PAM burst."""
    rng = np.random.default_rng(3)
    W = rng.choice(np.array([-1, 0, 1], np.int8), size=(1, 2, 8), p=[0.3, 0.4, 0.3])
    ly = Layer(W=W, dilation=1, gain=np.array([1.0], np.float32), bias=np.array([0.0], np.float32), vsat=1.0)
    x = rng.choice(np.array([-1.0, 0.0, 1.0]), size=(2, 24))
    global _REL_INCLUDES
    path = Path(path) if path else SPICE_DIR / "unit_cell.cir"
    _REL_INCLUDES = path.resolve().parent == SPICE_DIR.resolve()
    try:
        nl = netlist_unit_cell(ly, [0], x, 0.5, r_from_loss(RG58_DB_PER_US))
    finally:
        _REL_INCLUDES = False
    head = ("* hw/spice/unit_cell.cir : REFERENCE layer-1 unit cell (2 input lines x 8 taps, d=1, ONE output), "
            "regenerated by\n* dlnn.tiers.circuit.write_reference_netlist(); the tier generates the real netlists "
            "from model.npz.\n* Run from this directory:  cd hw/spice && ngspice -b unit_cell.cir\n")
    path.write_text(head + nl)
    return path


# ----------------------------------------------------------------------------------------- tier entry point
def _write_extracted(outdir: Path, doc: dict) -> Path:
    p = Path(outdir) / EXTRACT_REL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return p


def _r6(a: np.ndarray) -> list:
    return np.round(np.asarray(a, np.float64), 7).tolist()


def _write_diagnostics(outdir: Path, model: Model, dataset: dict, line: LineModel, amp: AmpModel, fit: dict,
                       comp: str, cfg: dict) -> Path:
    """params/circuit_diagnostics.json: full-set results of the OTHER loss-compensation modes (behavioural model,
    not SPICE) so the effect of the single-trim assumption is visible. Not part of the tier contract."""
    x = dataset["x_test"]
    ref = virtual.forward(model, x)["out"]
    vals = [float(v) for v in cfg.get("loss_sweep", (0.0, 5.0, 14.0, 32.0))]
    diag: dict = {"note": "circuit-calibrated BEHAVIOURAL model (not SPICE); loss compensation modes at the fitted "
                          "line loss and over the loss sweep; 'tier_default' marks the one used in tiers/C.json",
                  "tier_default": comp, "modes": {}}
    for mode in COMP_MODES:
        if mode == comp:
            continue
        fwd = lambda xx, v: calibrated_forward(model, xx, line.with_loss(v, fit["loss_db_per_us_100mhz"]), amp, comp=mode)  # noqa: E731
        sc = lambda o: common.score(o["out"], ref, dataset)  # noqa: E731
        sw = sweeps.run_sweep("loss_db_per_us", vals, fwd, x, sc)
        m, _, _ = common.metrics_and_arrays(calibrated_forward(model, x, line, amp, comp=mode)["out"], ref, dataset)
        diag["modes"][mode] = {"metrics_at_fitted_loss": {k: v for k, v in m.items() if k != "acc_per_snr"}, "loss_sweep": sw}
    p = Path(outdir) / "params" / "circuit_diagnostics.json"
    p.write_text(json.dumps(contract._jsonable(diag), indent=1))
    return p


def run(model_npz: str | Path, dataset: dict, cfg: dict, outdir: str | Path) -> contract.TierResult:
    t0 = time.time()
    outdir = Path(outdir)
    model = load_model(model_npz)
    ly0 = model.layers[0]
    O, I, K = ly0.W.shape
    d0 = ly0.dilation
    work = outdir / "spice"
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    loss_nom = float(cfg.get("c_loss_db_per_us", RG58_DB_PER_US))
    r_nom = r_from_loss(loss_nom)
    use_spice = bool(cfg.get("c_use_spice", True)) and ngspice_path() is not None
    version = ngspice_version() if use_spice else "n/a"
    spice_ok = use_spice
    fallback_reason = None if use_spice else "ngspice not found on PATH (or c_use_spice=False)"

    # (a) extraction ------------------------------------------------------------------------------------------
    try:
        ac = extract_ac(work, K, d0, r_nom, spice_ok)
        dc = extract_dc(work, spice_ok)
    except Exception as e:  # noqa: BLE001  ngspice failed: use the closed-form numpy line, labelled as such
        if not use_spice:
            raise
        spice_ok, fallback_reason = False, f"{type(e).__name__}: {e}"
        warnings.append(f"ngspice extraction failed ({fallback_reason}); using analytic numpy RLGC line + ideal tanh")
        ac, dc = extract_ac(work, K, d0, r_nom, False), extract_dc(work, False)
    fit, dfit = ac["fit"], dc["fit"]
    if fit["fit_max_abs_err"] > 0.02:
        warnings.append(f"closed-form line fit to H_k(f) is poor (max |dH| = {fit['fit_max_abs_err']:.3g})")
    if dfit["r2"] < 0.99:
        warnings.append(f"diff-pair tanh fit R^2 = {dfit['r2']:.4f} < 0.99")
    line = LineModel(fit["R_ohm_per_m"], fit["L_h_per_m"], fit["C_f_per_m"])
    amp = AmpModel(dfit["vsat_scale"], dfit["gain_scale"], dfit["offset_rel"])
    comp = str(cfg.get("c_loss_comp", "layer"))
    if comp not in COMP_MODES:
        raise ValueError(f"c_loss_comp must be one of {COMP_MODES}")
    trims = [cursor_trim(ly, line) if comp == "layer" else 1.0 for ly in model.layers]

    # (b) unit-cell .tran ------------------------------------------------------------------------------------
    tran, waves = None, {}
    if spice_ok:
        try:
            tran, _, _, _ = run_unit_cell(work, model, dataset, cfg, r_nom, trims[0], line, amp)
            if comp == "layer" and bool(cfg.get("c_tran_untrimmed", True)):   # diagnostic: same cell, no gain trim
                tran_nt = run_unit_cell(work, model, dataset, cfg, r_nom, 1.0, line, amp, tag="_notrim")[0]
                tran["no_trim"] = {k: v for k, v in tran_nt.items() if k != "wave"}
            waves = {f"waveforms.ex{e}": tran["wave"][e] for e in range(tran["wave"].shape[0])}
            if tran["nmse"] > 0.05:
                warnings.append(f"unit-cell layer-1 NMSE vs V = {tran['nmse']:.3f} exceeds the 0.05 target")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"ngspice .tran of the unit cell failed ({type(e).__name__}: {e})")
            fallback_reason = f"unit-cell .tran failed: {e}"
            tran = None
    spice_full = spice_ok and tran is not None

    # extraction document -----------------------------------------------------------------------------------
    nl_ex0 = netlist_unit_cell(ly0, list(range(min(O, int(cfg.get("c_outputs", 8))))),
                               pick_examples(dataset, 1)[0] * np.float32(model.input_scale), 0.05, r_nom, trims[0])
    doc = {
        "schema": EXTRACT_SCHEMA,
        "source": "ngspice" if spice_ok else "analytic-numpy-rlgc",
        "generic_model_notice": GENERIC_NOTICE,
        "tool": {"name": "ngspice" if spice_ok else "numpy", "version": version if spice_ok else np.__version__},
        "ngspice_version": version, "netlist_sha256": {
            **lib_hashes(), "tap_lines_ac": sha256_text(ac["netlist"]), "diffpair_dc": sha256_text(dc["netlist"]),
            "unit_cell_example0": sha256_text(nl_ex0)},
        "units": {"freq_hz": "Hz", "H": "dimensionless V_tap/V_source, 50 ohm load", "delay": "s", "length": "m",
                  "vsat_v": "V at diff-pair input", "A_v": "V differential output", "loss": "dB/us"},
        "symbol_period_s": TS, "dilation": d0, "n_taps": K,
        "timing": {"t_rise_s": T_RISE, "sample_phase": SAMPLE_PHASE},
        "line": {"model": "ltra-rlc-constant-R" if spice_ok else "analytic-rlc-constant-R",
                 "R_ohm_per_m_nominal": r_nom, "L_h_per_m_nominal": L_PER_M, "C_f_per_m_nominal": C_PER_M,
                 "G_s_per_m_nominal": 0.0, "loss_db_per_us_nominal_100mhz": loss_nom, "fit": fit,
                 "omits": ["skin-effect sqrt(f) loss/dispersion", "dielectric loss (G=0)", "connectors", "crosstalk"]},
        "taps": {"k": list(range(K)), "delay_nominal_s": [k * d0 * TS for k in range(K)],
                 "length_m": [t["length_m"] for t in ac["taps"]],
                 "dc_gain_db": [t["dc_gain_db"] for t in ac["taps"]],
                 "gain_100mhz_db": [t["gain_100mhz_db"] for t in ac["taps"]],
                 "delay_error_s": [t["delay_error_s"] for t in ac["taps"]],
                 "freq_hz": _r6(ac["freq_hz"]),
                 "H_re": [_r6(ac["H"][k].real) for k in range(K)], "H_im": [_r6(ac["H"][k].imag) for k in range(K)]},
        "diffpair": dfit,
        "amp": {"vsat_scale": amp.vsat_scale, "gain_scale": amp.gain_scale, "offset_rel": amp.offset_rel},
        "loss_comp": comp, "loss_comp_gain_trim_per_layer": trims,
        "unit_cell_tran": None if tran is None else {k: v for k, v in tran.items() if k != "wave"},
    }
    ex_path = _write_extracted(outdir, doc)

    # (c) circuit-calibrated behavioural model on the FULL test set --------------------------------------
    cal = lambda x, c: calibrated_forward(model, x, line, amp, comp=comp, trims=trims)  # noqa: E731

    def loss_sweep(x, score):
        vals = [float(v) for v in cfg.get("loss_sweep", (0.0, 5.0, 14.0, 32.0))]
        return [sweeps.run_sweep(
            "loss_db_per_us", vals,
            lambda xx, v: calibrated_forward(model, xx, line.with_loss(v, fit["loss_db_per_us_100mhz"]), amp, comp=comp),
            x, score)]

    n_cell = tran["n_examples"] if tran else 0
    if spice_full:
        status = "ok"
        tool = {"name": "ngspice", "version": version}
        tran_txt = (f"ngspice .tran of the layer-1 unit cell ({n_cell} test examples, {len(tran['outputs'])} output lines): "
                    f"layer-1 activations vs V NMSE = {tran['nmse']:.4f} (variance-normalised; "
                    f"{tran['nmse_energy_normalised']:.4f} energy-normalised), noise off"
                    + (f"; same cell without the gain trim: NMSE = {tran['no_trim']['nmse']:.4f}" if "no_trim" in tran else "")
                    + f"; calibrated behavioural layer 1 vs these SPICE traces: NMSE = {tran['nmse_calibrated_model_vs_spice']:.4f}")
        models = [
            "SPICE (ngspice " + version + ") computed: (a) .ac per-tap complex H_k(f) of the lossy RLC coax lines "
            f"(RG-58-like generic model, {loss_nom:g} dB/us at 100 MHz, constant R) and .dc diff-pair transfer "
            f"(tanh fit R^2 = {dfit['r2']:.5f}, V_sat = {dfit['vsat_v']*1e3:.2f} mV, gain A = {dfit['A_v']:.4f} V); "
            "(b) " + tran_txt,
            "The FULL test-set numbers (metrics, sweeps, traces) come from the CIRCUIT-CALIBRATED BEHAVIOURAL MODEL "
            "(closed-form RLC line fitted to the SPICE H_k(f), evaluated per tap delay k*d*T_s, sampled at "
            f"{SAMPLE_PHASE:g}*T_s; tanh with the SPICE-fitted V_sat/gain/offset; loss compensation "
            f"'{comp}' = " + {"none": "none (raw line loss)", "layer": "one ideal gain trim per layer (no per-tap repeaters)",
                              "tap": "ideal per-tap repeater/equaliser"}[comp] + "). They are NOT SPICE results.",
            "Layer-1 SPICE unit cell(s) only; layers 2-3 (d=4, 16) use the same fitted line model, extrapolated "
            "to longer delays and NOT simulated in SPICE.",
        ]
        omits = [
            "full network in SPICE (only layer 1, " + str(n_cell) + " examples, was run end-to-end in ngspice)",
            "layers 2-3 line responses are extrapolated from the layer-1 fit; no SPICE verification of long lines",
            "skin-effect sqrt(f) loss/dispersion (constant-R LTRA line; DC loss of long lines is overstated)",
            "dielectric loss, connectors, layout parasitics, crosstalk, coupler imperfections (ideal VCVS taps)",
            "noise (all sources off), temperature/bias drift; BJT is a GENERIC model, not a vendor part",
            "RC pool is only checked inside the unit cell; the tier's readout pooling is the exact digital mean",
            "readout and softmax are digital (not circuit)",
        ]
    else:
        status = "fallback"
        tool = ({"name": "ngspice-partial", "version": version} if spice_ok
                else {"name": "numpy-analytic-rlgc", "version": np.__version__})
        models = [f"analytic: numpy closed-form RLC line (R, L, C as in hw/spice/coax_rg58.lib, {loss_nom:g} dB/us) "
                  "and ideal tanh with V_sat = 2*V_T; ngspice results NOT available",
                  "full test set run with the circuit-calibrated behavioural model built from the ANALYTIC extraction"]
        omits = [f"FALLBACK: ngspice did not run ({fallback_reason}); every number is analytic, none is a SPICE result",
                 "no unit-cell transient comparison", "all items listed for the SPICE run are also omitted"]
        warnings.append("C tier is the analytic fallback (no ngspice results): " + str(fallback_reason))

    res = common.evaluate_tier(
        tier="C", status=status, tool=tool, model=model, dataset=dataset, cfg=cfg, forward_fn=cal, ref_fn=None,
        fidelity={"models": models, "omits": omits}, sweep_axes=("snr_db",), t0=t0,
        inputs_from=[EXTRACT_REL_PATH], warnings=warnings, extra_sweeps=loss_sweep)
    res.arrays.update(waves)
    if bool(cfg.get("c_diagnostics", True)):
        _write_diagnostics(outdir, model, dataset, line, amp, fit, comp, cfg)
    res.warnings.append(f"extraction parameters written to {EXTRACT_REL_PATH} "
                        f"(source: {doc['source']}); schema {EXTRACT_SCHEMA}")
    res.runtime_s = round(time.time() - t0, 3)
    return res
