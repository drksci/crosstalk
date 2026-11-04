"""Tier C, BOARD mode: ngspice on the KiCad-exported TILE-8 netlist (M2a).

The phase-1 coax unit-cell path in `circuit.py` is untouched and is the "coax study".  This module re-points the C tier at the
real board: the per-layer tile schematic (`hw/board/gen_tile.py --layer L --model M`, KiCad ERC clean) is exported with
`kicad-cli sch export netlist --format spice`, the stand-in parts library of the export is replaced by `hw/spice/parts/*.lib`
(datasheet-derived inductor, fitted single-pole TLV906x, generic BC847 NPN, 4066 = R_on + C_off) and ngspice runs
  * `.ac`  per tap of the whole tile (taps, input RC, tap buffers),
  * `.dc`  of the whole bias -> summer -> BJT diff pair -> difference amp -> S/H chain (all 8 channels at once),
  * `.tran` of the WHOLE tile (8 lines x 56 LC sections, 64 tap buffers, 1024 crossbar footprints, 8 summers/pairs/diff amps/S/H)
    on test examples, layer 1 -> S/H waveform -> layer 2 -> S/H waveform -> layer 3 (no extrapolated layers).
A per-layer circuit-calibrated BEHAVIOURAL model is then built from the SPICE results and runs the 4000-example test set.
Its numbers are never SPICE results.  Everything is simulation of a design that is not built; nothing is measured.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from ..model.npz import Model, load_model
from . import circuit, virtual

REPO = Path(__file__).resolve().parents[2]
PARTS = REPO / "hw" / "spice" / "parts"
TLV_LIB = REPO / "hw" / "board" / "spice" / "_work" / "models" / "tlv9062.lib"
EDA_PY = REPO / "hw" / ".venv-eda" / "bin" / "python"
GEN_TILE = REPO / "hw" / "board" / "gen_tile.py"
PARAMS_JSON = REPO / "hw" / "board" / "params.json"

VCM = 2.5
A_ACT = 0.5            # volts per model unit between tiles (derive_params A_act_v)
A_IN0 = 0.3            # volts per model unit at the layer-1 inputs (derive_params A_in0_v_per_unit)
R_IN = 10e3            # crossbar resistor
Q_FQ = 796e3
DCR0, DCR_TC = 3.3, 0.00393        # DCR at 25 C and copper tempco (1/K)
RON25, RON_TC = 95.0, 0.3833       # 74HC(T)4066 R_on at 25 C (max) and ohm/K (95 -> 118 ohm for 25 -> 85 C)
DAC_VREF = 2.048                   # MCP4728 internal reference (M3b: DAC on 3.3 V, range 0..2.048 V)
T_TRACK = 450e-9                   # S/H track window (design: board_design.md)
EDGE = 20e-9                       # DAC / test-source edge
NOTICE = ("SIMULATION of a design that is not built; nothing measured. Inductor = datasheet-derived lumped model (L, DCR, C_p from SRF, "
          "R_p from Q_min 10 / assumed Q_typ 30); op-amp = single-pole fit to TLV9062 (TI macro used only in the slice cross-check); "
          "NPN = generic BC847-fitted; 4066 = R_on + C_off; no layout parasitics, no crosstalk, no noise.")


def sha_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------------------------- inductor model numbers
def inductor_rp(q: float, dcr: float = DCR0, L: float = 120e-6, fq: float = Q_FQ) -> float:
    """R_p (parallel, constant) reproducing datasheet Q at f_q given the series DCR (same as hw/board/spice/linelib.py)."""
    w = 2 * math.pi * fq
    x = 1.0 / q - dcr / (w * L)
    return 1e12 if x <= 0 else w * L / x


# --------------------------------------------------------------------------------------------- export from KiCad
def export_tiles(model_npz: str | Path, outroot: str | Path, layers=(1, 2, 3)) -> dict:
    """gen_tile.py --model per layer into outroot/layerN, then kicad-cli spice netlist.  Returns paths + shas (re-exported every call)."""
    model_npz, outroot = Path(model_npz).resolve(), Path(outroot)
    env = dict(os.environ, KICAD9_FOOTPRINT_DIR=os.environ.get("KICAD9_FOOTPRINT_DIR", "/usr/share/kicad/footprints"))
    info = {"model_file": str(model_npz.relative_to(REPO)) if REPO in model_npz.parents else str(model_npz),
            "model_file_sha256": sha_file(model_npz), "gen_tile_sha256": sha_file(GEN_TILE),
            "kicad_cli": subprocess.run(["kicad-cli", "--version"], capture_output=True, text=True).stdout.strip(), "layers": {}}
    for L in layers:
        d = outroot / f"layer{L}"
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(EDA_PY), str(GEN_TILE), "--layer", str(L), "--model", str(model_npz), "--out", str(d)],
                           capture_output=True, text=True, cwd=REPO, env=env)
        if r.returncode:
            raise RuntimeError(f"gen_tile layer {L} failed: {r.stderr[-800:]}")
        cir = d / "tile.cir"
        r = subprocess.run(["kicad-cli", "sch", "export", "netlist", "--format", "spice", "-o", str(cir), str(d / "tile.kicad_sch")],
                           capture_output=True, text=True, env=env)
        if r.returncode or not cir.exists():
            raise RuntimeError(f"kicad-cli netlist layer {L} failed: {r.stderr[-800:]}")
        sch = hashlib.sha256()
        for p in sorted(d.glob("*.kicad_sch")):
            sch.update(p.read_bytes())
        info["layers"][str(L)] = {"cir": str(cir), "cir_sha256": sha_file(cir), "schematic_sha256": sch.hexdigest(),
                                  "gen_tile_stdout": r.stdout[-200:] if False else ""}
    return info


# --------------------------------------------------------------------------------------------- deck builder
_RE_IND = re.compile(r"^(XL\d+_\d+)\s+(\S+)\s+(\S+)\s+IND_FH1812\s*$")
_RE_LC = re.compile(r"^(C\d+_\d+(?:_[12])?)\s+(\S+)\s+(\S+)\s+([0-9.]+[pnu]?)\s*$")
_RE_RF = re.compile(r"^(RF[PN]\d+|RB\d+|RO\d+)\s+(\S+)\s+(\S+)\s+([0-9.]+k?)\s*$")


E96 = [1.0, 1.02, 1.05, 1.07, 1.1, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.3, 1.33, 1.37, 1.4, 1.43, 1.47, 1.5, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.0, 2.05, 2.1, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.8, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32, 3.4, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.9, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.5, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76]


def _e96_nearest(v: float) -> float:
    d = math.floor(math.log10(v))
    cands = [m * 10 ** (d + k) for k in (-1, 0, 1) for m in E96]
    return min(cands, key=lambda c: abs(math.log(c / v)))


def _val(s: str) -> float:
    m = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "k": 1e3}
    return float(s[:-1]) * m[s[-1]] if s[-1] in m else float(s)


class Tile:
    """A KiCad-exported tile netlist plus the run configuration.  `deck(...)` returns a complete ngspice input."""

    def __init__(self, cir: str | Path, layer: int, model: Model, opamp: str = "fit", q: float = 10.0, temp_c: float = 25.0,
                 rf_scale: float = 2.0, mc: dict | None = None, rt_to_vref: bool = True):
        self.cir, self.layer, self.model = Path(cir).resolve(), layer, model
        self.opamp, self.q, self.temp_c, self.rf_scale, self.mc = opamp, q, temp_c, rf_scale, mc
        self.rt_to_vref = rt_to_vref
        self.raw = self.cir.read_text().splitlines()
        self.lines_idx = sorted({int(m.group(1)) for l in self.raw for m in [re.match(r"^RP(\d+)\s", l)] if m})
        self.n_lines = len(self.lines_idx)
        self.rf = {}
        for l in self.raw:
            m = _RE_RF.match(l)
            if m:
                self.rf[m.group(1)] = _val(m.group(4)) * (rf_scale if m.group(1)[:2] in ("RF", "RB", "RO") else 1.0)
        self.O = model.layers[layer - 1].W.shape[0]
        ly = model.layers[layer - 1]
        self.R_f = np.array([self.rf[f"RFN{o}"] for o in range(self.O)])
        self.R_b = np.array([self.rf[f"RB{o}"] for o in range(self.O)])
        self.R_b_design = self.R_b / rf_scale            # the bias DAC is sized with the as-designed R_b
        self.a_in = A_IN0 if layer == 1 else A_ACT
        self.ly = ly

    # ---------------------------------------------------------------- netlist rewriting
    def body(self) -> str:
        t = self.temp_c
        dcr = DCR0 * (1 + DCR_TC * (t - 25.0))
        rp = inductor_rp(self.q, dcr)
        mc = self.mc
        out = []
        for l in self.raw:
            if l.startswith((".include", ".end", ".title")) or not l.strip():
                continue
            m = re.match(r"^(RT\d+)\s+(\S+)\s+GND\s+(\S+)\s*$", l)
            if m and self.rt_to_vref:           # ECO-1: line termination to mid-rail instead of ground (see docs/circuit_extraction.md)
                out.append(f"{m.group(1)} {m.group(2)} VREF25 {m.group(3)}")
                continue
            m = _RE_IND.match(l)
            if m:
                pars = f"R={dcr:.5g} Rp={rp:.6g}"
                if mc:
                    pars += f" L={{agauss(120u,{mc['sigma_L'] * 120e-6:.6g},1)}}"
                out.append(f"{m.group(1)} {m.group(2)} {m.group(3)} IND_FH1812 {pars}")
                continue
            m = _RE_LC.match(l)
            if m and mc:
                v = _val(m.group(4))
                out.append(f"{m.group(1)} {m.group(2)} {m.group(3)} {{agauss({v:.6g},{mc['sigma_C'] * v:.6g},1)}}")
                continue
            m = _RE_RF.match(l)
            if m and self.rf_scale != 1.0 and m.group(1)[:2] in ("RF", "RB", "RO"):
                out.append(f"{m.group(1)} {m.group(2)} {m.group(3)} {self.rf[m.group(1)]:.6g}")
                continue
            out.append(l)
        return "\n".join(out)

    def lib_lines(self) -> str:
        ron = RON25 + RON_TC * (self.temp_c - 25.0)
        lib = (PARTS / "dlnn_board_parts.lib").read_text().replace("Ron=95 Roff", f"Ron={ron:.5g} Roff")
        wd = self.cir.parent / "_lib"
        wd.mkdir(exist_ok=True)
        p = wd / f"parts_T{self.temp_c:g}.lib"
        p.write_text(lib)
        op = PARTS / ("opa_quad_ti.lib" if self.opamp == "ti" else "opa_quad_fit.lib")
        s = f".include {p}\n.include {op}\n"
        if self.opamp == "ti":
            s += f".include {TLV_LIB}\n"
        return s

    def dac_volts(self) -> np.ndarray:
        """Bias DAC level per output exactly as gen_tile.py (M3b version) computes it: MCP4728 on 3.3 V with its 2.048 V reference,
        R_o (+5VA) centring the range, design R_b; 12-bit quantised."""
        rb = self.R_b_design
        ro = np.array([_e96_nearest(r * 2.5 / (2.5 - 1.024)) for r in rb])
        v = VCM + rb * self.a_in * self.ly.bias.astype(float) / R_IN - rb * 2.5 / ro
        v = np.clip(v, 0.0, 2.047)
        return np.round(v / DAC_VREF * 4096) / 4096 * DAC_VREF

    def deck(self, analysis: str, sources: str, save: list[str], bias_v: np.ndarray | None = None, dc_sweep_bias: bool = False,
             extra_options: str = "") -> str:
        bv = self.dac_volts() if bias_v is None else bias_v
        head = [f"* {self.cir.name} layer {self.layer} opamp={self.opamp} Q={self.q} T={self.temp_c} rf_scale={self.rf_scale}",
                self.lib_lines(), f".temp {self.temp_c:g}", ".options rshunt=1e9",
                "Vcc +5VA 0 5", "V33 +3V3 0 3.3", extra_options]
        if self.mc:
            head.append(f".option seed={int(self.mc['seed'])}")
        body = self.body()
        # DAC outputs: per-run levels (or the common DC-sweep source driving all eight BIAS nets)
        outl = []
        for l in body.splitlines():
            m = re.match(r"^(XUDAC(\d))\s+(.*)\s+DAC4_IDEAL\s*$", l)
            if m:
                d = int(m.group(2))
                if dc_sweep_bias:
                    for c in range(4):
                        outl.append(f"Bb{d * 4 + c} BIAS{d * 4 + c} 0 V=v(vsw)")
                    # keep the supply/ground loads so the nets exist
                    outl.append(f"Rdacd{d} +5VA 0 1e9")
                    continue
                vs = " ".join(f"V{'ABCD'[c]}={bv[d * 4 + c]:.6g}" for c in range(4))
                outl.append(f"{m.group(1)} {m.group(3)} DAC4_IDEAL {vs}")
            else:
                outl.append(l)
        sv = " ".join(save)
        return "\n".join(head) + "\n" + "\n".join(outl) + "\n" + sources + "\n" + (f".save {sv}\n" if save else "") + analysis + "\n.end\n"


def run_ngspice(deck: str, workdir: Path, tag: str, psa: bool = False, timeout: float = 3600.0) -> list[dict]:
    """ngspice -b on a deck; binary raw file parsed with circuit.read_raw.  psa: PSpice dialect for the TI macro."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cir = workdir / f"{tag}.cir"
    cir.write_text(deck)
    if psa:
        (workdir / ".spiceinit").write_text("set ngbehavior=psa\n")
    elif (workdir / ".spiceinit").exists():
        (workdir / ".spiceinit").unlink()
    raw, log = workdir / f"{tag}.raw", workdir / f"{tag}.log"
    p = subprocess.run(["ngspice", "-b", "-r", str(raw), "-o", str(log), str(cir)], capture_output=True, text=True,
                       timeout=timeout, cwd=workdir, stdin=subprocess.DEVNULL)
    txt = log.read_text(errors="replace") if log.exists() else p.stdout
    if not raw.exists() or "doAnalyses" in txt and "aborted" in txt:
        raise circuit.SpiceError(f"ngspice failed on {tag}: {txt[-600:]}")
    plots = circuit.read_raw(raw)
    raw.unlink(missing_ok=True)
    return plots


def vec(plot: dict, name: str) -> np.ndarray:
    return plot["data"][:, plot["names"].index(name.lower())]


# --------------------------------------------------------------------------------------------- stimuli
def pwl_symbols(vals: np.ndarray, t_start: float, T: float, edge: float = EDGE, lead: float = 0.0, name: str = "V") -> str:
    """PWL of a ZOH symbol train: symbol n starts at t_start + n*T (linear edge of `edge`), value vals[n] volts."""
    pts = [(0.0, VCM)]
    prev = VCM
    for n, v in enumerate(vals):
        t = t_start + n * T
        pts += [(t, prev), (t + edge, float(v))]
        prev = float(v)
    pts.append((t_start + len(vals) * T + 2 * T, prev))
    return " ".join(f"{a:.9g} {b:.6g}" for a, b in pts)


def clock_pulse(t_start: float, T: float, t0: float, n_sym: int, pre: int = 2) -> str:
    """SH_CLK: high for T_TRACK, FALLING edge (sample) at t_start + n*T + t0 for n = -pre..n_sym-1 (period T)."""
    first_fall = t_start - pre * T + t0
    rise = first_fall - T_TRACK
    return f"PULSE(0 5 {rise:.9g} 5n 5n {T_TRACK - 5e-9:.9g} {T:.9g})"


# --------------------------------------------------------------------------------------------- analyses
def sig_dc_sources(tile: Tile, ac_line: int | None = None, level: float = VCM) -> str:
    s = []
    for i in tile.lines_idx:
        s.append(f"Vs{i} SIG{i} 0 DC {level:g}" + (" AC 1" if ac_line == i else ""))
    s.append("Vclk SH_CLK 0 DC 5")      # switch closed: the hold node follows the diff amp at the operating point
    return "\n".join(s)


def ac_taps(tile: Tile, workdir: Path, fmax: float = 20e6, npts: int = 1000, line: int = 0) -> dict:
    """`.ac` of the whole tile with line `line` driven.  H[k, f] = V(T{line}_{k}) / V(SIG{line}), k = 0..7 (tap buffers' outputs),
    plus the node after the input RC / buffer.  Uniform grid df = fmax/npts so that the symbol-rate FIR can be built by IFFT."""
    nm = [f"v(t{line}_{k})" for k in range(8)] + [f"v(/line{line}/ib)"]
    deck = tile.deck(f".ac lin {npts} {fmax / npts:g} {fmax:g}", sig_dc_sources(tile, line), nm)
    p = run_ngspice(deck, workdir, f"ac_L{tile.layer}")[-1]
    f = vec(p, "frequency").real
    H = np.array([vec(p, n) for n in nm])
    return {"freq_hz": f, "H": H[:8], "H_ib": H[8]}


def fir_from_ac(f: np.ndarray, H: np.ndarray, T: float, M: int = 12, edge: float = EDGE, oversample: int = 8, tau_lp: float = 0.0):
    """Dense impulse-response-of-a-ZOH-symbol h_k(t) from the .ac transfer functions.  Returns (t, h[K, len(t)]) with t spacing 1/(2 fmax)."""
    df = f[1] - f[0]
    n = len(f)
    Hf = np.concatenate([H[:, :1], H], axis=1)             # DC point = first point (flat extrapolation)
    ff = np.concatenate([[0.0], f])
    P = T * np.sinc(ff * T) * np.sinc(ff * edge) * np.exp(-1j * np.pi * ff * (T + edge))
    if tau_lp:
        P = P / (1 + 2j * np.pi * ff * tau_lp)          # first-order low-pass of the post-chain (summer, pair, diff amp, S/H) after the taps
    spec = Hf * P[None, :]
    nfft = 2 * n * oversample
    h = np.fft.irfft(spec, n=nfft, axis=1) * (df * nfft)    # x(t) = sum_f X(f) e^{j2 pi f t} df
    dt = 1.0 / (nfft * df)
    return np.arange(nfft) * dt, h


def calibrate_clock(t: np.ndarray, h: np.ndarray, mask_after: float = 12e-6) -> tuple[float, np.ndarray]:
    """T = (centroid(tap7) - centroid(tap0)) / 7 (the MCU-calibrated symbol clock, as in models/board_aware/line_fir_m1a.py)."""
    sel = t < mask_after
    cen = np.array([(t[sel] * h[k, sel]).sum() / h[k, sel].sum() for k in range(h.shape[0])])
    return float((cen[-1] - cen[0]) / (h.shape[0] - 1)), cen


def sample_fir(t: np.ndarray, h: np.ndarray, t0: float, T: float, M: int = 12) -> np.ndarray:
    pts = t0 + np.arange(M) * T
    return np.array([np.interp(pts, t, h[k]) for k in range(h.shape[0])])         # [K, M]


def best_phase(t, h, T, M=12, grid=121):
    """S/H phase t0 in [0, 2T) maximising the mean on-lag fraction fir_k[k] / sum_m |fir_k[m]| (same criterion as tier P)."""
    best, bt = -1.0, 0.0
    for t0 in np.linspace(0.0, 2 * T, grid):
        F = sample_fir(t, h, t0, T, M)
        sc = np.mean([abs(F[k, k]) / (np.abs(F[k]).sum() + 1e-30) for k in range(F.shape[0])])
        if sc > best:
            best, bt = sc, float(t0)
    return bt, best


def dc_sweep(tile: Tile, workdir: Path, vlo: float = -1.0, vhi: float = 3.0, step: float = 0.01, in_level: np.ndarray | None = None) -> dict:
    """`.dc` of the whole bias -> summers -> BJT pair (mirror tail) -> difference amp -> S/H -> buffer chain, all 8 channels at once.
    vsw = V_bias (absolute DAC voltage, 0..2.048 V on the board) of every channel; inputs held at 2.5 V (zero).  Returns OUT_o(vsw) and the pair input v_d(vsw)."""
    nm = [f"v(out{o})" for o in range(tile.O)] + [f"v(/ch{o}/op)" for o in range(tile.O)] + [f"v(/ch{o}/on)" for o in range(tile.O)]
    src = sig_dc_sources(tile) + "\nVsw vsw 0 0"
    deck = tile.deck(f".dc Vsw {vlo:g} {vhi:g} {step:g}", src, nm, dc_sweep_bias=True)
    p = run_ngspice(deck, workdir, f"dc_L{tile.layer}")[-1]
    v = vec(p, "v-sweep").real if "v-sweep" in p["names"] else p["data"][:, 0].real
    O = tile.O
    out = np.array([vec(p, nm[o]).real for o in range(O)])
    op = np.array([vec(p, nm[O + o]).real for o in range(O)])
    on = np.array([vec(p, nm[2 * O + o]).real for o in range(O)])
    return {"vsw": v, "out": out, "vd": op - on}


def tanh_fit(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.optimize import curve_fit
    f = lambda v, A, vs, s, c: A * np.tanh((v - s) / vs) + c        # noqa: E731
    best = None
    for sign in (1.0, -1.0):
        try:
            p, _ = curve_fit(f, x, y, p0=[sign * 0.5 * (y.max() - y.min()) / 1.0, (x.max() - x.min()) / 6, x[np.argmin(np.abs(y - y.mean()))], y.mean()],
                             maxfev=20000)
        except Exception:                                    # noqa: BLE001
            continue
        r = y - f(x, *p)
        r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if best is None or r2 > best["r2"]:
            best = {"A": float(p[0]), "vs": float(abs(p[1])), "shift": float(p[2]), "offset": float(p[3]), "r2": float(r2),
                    "rms_resid": float(np.sqrt((r ** 2).mean()))}
    return best


def calibrate_bias(tile: Tile, dc: dict, y_target: np.ndarray) -> dict:
    """DC trim of the bias DAC (what the MCU would do over I2C): choose V_dac,o so OUT_o = 2.5 + A_ACT*y_target_o at zero input,
    by interpolating the `.dc` sweep; limited to the DAC range [0.01, 2.04] V (0.5 mV LSB, 12 bit); `clipped` flags channels that cannot reach it."""
    O = tile.O
    vsw_cal, clipped = np.zeros(O), np.zeros(O, bool)
    ref = tile.dac_volts()
    for o in range(O):
        y = dc["out"][o]
        tgt = VCM + A_ACT * y_target[o]
        idx = np.where((y[:-1] - tgt) * (y[1:] - tgt) <= 0)[0]
        if len(idx) == 0:
            vsw_cal[o], clipped[o] = dc["vsw"][np.argmin(np.abs(y - tgt))], True
        else:
            i = idx[np.argmin(np.abs(dc["vsw"][idx] - ref[o]))]
            vsw_cal[o] = np.interp(tgt, [y[i], y[i + 1]], [dc["vsw"][i], dc["vsw"][i + 1]]) if y[i + 1] != y[i] else dc["vsw"][i]
        if not 0.01 <= vsw_cal[o] <= 2.04:
            vsw_cal[o], clipped[o] = float(np.clip(vsw_cal[o], 0.01, 2.04)), True
    vb = np.round(vsw_cal / DAC_VREF * 4096) / 4096 * DAC_VREF
    return {"vb": vb, "vsw": vb, "clipped": clipped, "design_vb": ref}


def y_zero_input(ly) -> np.ndarray:
    return ly.gain.astype(float) * np.tanh(ly.bias.astype(float) / float(ly.vsat))


# --------------------------------------------------------------------------------------------- transient of one tile
def tran_tile(tile: Tile, x_volts: np.ndarray, T: float, t0: float, vb: np.ndarray, workdir: Path, tag: str,
              t_start: float | None = None, sample_dt: float = 250e-9) -> dict:
    """Whole-tile `.tran`.  x_volts [I, N] = input voltage of every populated line for symbols 0..N-1 (ZOH, 20 ns edges, symbol n launched
    at t_start + n T; lines idle at 2.5 V before).  SH_CLK falls (= S/H samples) at t_start + n T + t0.  Returns the held outputs
    OUT_o read `sample_dt` after each falling edge (this is the 'S/H sample') and the node-voltage extrema (rail check)."""
    I, N = x_volts.shape
    pre = 3
    t_start = (pre + 1) * T if t_start is None else t_start
    src = []
    for i in tile.lines_idx:
        src.append(f"Vs{i} SIG{i} 0 PWL({pwl_symbols(x_volts[i], t_start, T)})")
    src.append(f"Vclk SH_CLK 0 {clock_pulse(t_start, T, t0, N, pre)}")
    tstop = t_start + (N + 1) * T + t0 + sample_dt + 50e-9
    save = [f"v(out{o})" for o in range(tile.O)]
    deck = tile.deck(f".tran 10n {tstop:.9g} 0 10n", "\n".join(src), save, bias_v=vb)
    p = run_ngspice(deck, workdir, tag)[-1]
    t = vec(p, "time").real
    ts = t_start + np.arange(N) * T + t0 + sample_dt
    out = np.array([np.interp(ts, t, vec(p, f"v(out{o})").real) for o in range(tile.O)])       # [O, N]
    allv = np.array([vec(p, f"v(out{o})").real for o in range(tile.O)])
    return {"out_v": out, "t_start": t_start, "n_points": int(len(t)), "out_min": float(allv.min()), "out_max": float(allv.max())}


# --------------------------------------------------------------------------------------------- calibrated behavioural layer
class LayerCal:
    """Per-layer circuit-calibrated behavioural model built from SPICE results (NOT SPICE).
    Linear part: symbol-rate FIR of the tap voltages (from `.ac`, ZOH launch); crossbar + summer: vsw_eq = (R_b/R_in) sum W tap
    (the summer/pair see signals exactly like the bias current); amplifier: out = A tanh((vsw_cal + vsw_eq - s)/vs) + c per channel
    (fitted to the `.dc` sweep); S/H: first-order memory e of the previous held value (fitted to calibration transients)."""

    def __init__(self, layer: int, ly, R_b, fir, fits, vsw_cal, a_in, e_sh=0.0, tau_post=0.0, pad_steady: bool = False):
        self.layer, self.ly, self.R_b, self.fir, self.fits, self.vsw_cal = layer, ly, np.asarray(R_b, float), fir, fits, np.asarray(vsw_cal, float)
        self.a_in, self.e_sh = a_in, e_sh
        self.Wl = np.einsum("oik,km->oim", ly.W.astype(float), fir)         # [O, I, M] volts per volt

    def set_fir(self, fir):
        self.fir = fir
        self.Wl = np.einsum("oik,km->oim", self.ly.W.astype(float), fir)

    def forward_volts(self, d: np.ndarray) -> np.ndarray:
        """d [B, I, N] input deviation from 2.5 V (volts) -> held output deviation [B, O, N] (volts)."""
        B, I, N = d.shape
        M = self.Wl.shape[2]
        acc = np.zeros((B, self.Wl.shape[0], N))
        for m in range(M):
            if m >= N:
                break
            acc[:, :, m:] += np.einsum("oi,bin->bon", self.Wl[:, :, m], d[:, :, :N - m], optimize=True)
        veq = acc * (self.R_b[None, :, None] / R_IN)
        A = np.array([f["A"] for f in self.fits])[None, :, None]
        vs = np.array([f["vs"] for f in self.fits])[None, :, None]
        s = np.array([f["shift"] for f in self.fits])[None, :, None]
        c = np.array([f["offset"] for f in self.fits])[None, :, None]
        y = A * np.tanh((self.vsw_cal[None, :, None] + veq - s) / vs) + c
        if self.e_sh:
            yh = y.copy()
            yh[:, :, 1:] = (1 - self.e_sh) * y[:, :, 1:] + self.e_sh * y[:, :, :-1]
            y = yh
        return y

    def forward_units(self, h_units: np.ndarray) -> np.ndarray:
        """model-unit activations in -> model-unit activations out (adds the 2.5 V / A_act scaling)."""
        return (self.forward_volts(h_units * self.a_in) / A_ACT).astype(np.float32)


def v_forward_fir(model: Model, x: np.ndarray, fir_by_layer=None, fir=None) -> dict:
    """Tier V with the M1a line FIR: tap k sees x through h_k[m] (the exact rule of TDLNet.forward with line_fir): V on the board line,
    ideal buffers/summer/tanh.  fir [K, M] shared by the d = 1 layers."""
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    outs = []
    for ly in model.layers:
        Wl = np.einsum("oik,km->oim", ly.W.astype(np.float32), np.asarray(fir, np.float32))
        B, I, N = h.shape
        u = np.zeros((B, Wl.shape[0], N), np.float32)
        for m in range(Wl.shape[2]):
            if m >= N:
                break
            u[:, :, m:] += np.einsum("oi,bin->bon", Wl[:, :, m], h[:, :, :N - m], optimize=True)
        h = ly.gain[None, :, None] * np.tanh((u + ly.bias[None, :, None]) / np.float32(ly.vsat))
        outs.append(h.astype(np.float32))
    return {"out": virtual.readout(model, virtual.features(model, outs)), "layers": outs}


# --------------------------------------------------------------------------------------------- per-layer setup + chain
class LayerSetup:
    """Everything SPICE gives for one tile at one corner: AC FIR, clock/phase, DC fits, DAC trim."""

    def __init__(self, tile: Tile, workdir: Path, T: float | None = None, y_target=None, vb_trim=None, tau_post: float = 0.0):
        self.tile = tile
        self.workdir = Path(workdir)
        self.ac = ac_taps(tile, self.workdir)
        self.t_ir, self.h_ir = fir_from_ac(self.ac["freq_hz"], self.ac["H"], T or 0.95e-6)
        self.T_cal, self.centroids = calibrate_clock(self.t_ir, self.h_ir)
        self.T = T or self.T_cal
        if T:   # re-evaluate the pulse response at the given (global) clock
            self.t_ir, self.h_ir = fir_from_ac(self.ac["freq_hz"], self.ac["H"], self.T)
        self.t0_opt, self.phase_score = best_phase(self.t_ir, self.h_ir, self.T)
        self.t0 = self.t0_opt + 200e-9           # clock (S/H falling edge) phase after launch; the post-chain lag is added, refitted by fit_post
        self.dc = dc_sweep(tile, self.workdir)
        self.y_target = y_zero_input(tile.ly) if y_target is None else y_target
        self.cal = calibrate_bias(tile, self.dc, self.y_target) if vb_trim is None else vb_trim
        self.fits = [tanh_fit(self.dc["vsw"], self.dc["out"][o] - VCM) for o in range(tile.O)]
        self.tau_post = tau_post
        self.tau_lp = 0.0
        self.fir = sample_fir(self.t_ir, self.h_ir, self.t0 - tau_post, self.T)

    e_fit = 0.0

    def calibrated(self, e_sh: float | None = None) -> LayerCal:
        t = self.tile
        return LayerCal(t.layer, t.ly, t.R_b, self.fir, self.fits, self.cal["vsw"], t.a_in, self.e_fit if e_sh is None else e_sh)

    def set_tau(self, tau_post: float, tau_lp: float | None = None):
        """tau_post = pure delay of the post-chain, tau_lp = its first-order low-pass time constant (both fitted to SPICE transients)."""
        if tau_lp is not None and tau_lp != self.tau_lp:
            self.tau_lp = tau_lp
            self.t_ir, self.h_ir = fir_from_ac(self.ac["freq_hz"], self.ac["H"], self.T, tau_lp=tau_lp)
        self.tau_post = tau_post
        self.fir = sample_fir(self.t_ir, self.h_ir, self.t0 - tau_post, self.T)

    def set_clock(self, T: float, t0_eff: float):
        """Operate the tile at symbol period T with the TAP-sampling phase t0_eff; the S/H clock phase is t0_eff + tau_post."""
        self.T = T
        self.t_ir, self.h_ir = fir_from_ac(self.ac["freq_hz"], self.ac["H"], T, tau_lp=self.tau_lp)
        self.t0 = t0_eff + self.tau_post
        self.fir = sample_fir(self.t_ir, self.h_ir, t0_eff, T)


def spice_chain(setups: list[LayerSetup], x: np.ndarray, input_scale: float, workdir: Path, tag: str, n_layers: int = 3) -> dict:
    """One example [I0, N] through the three tiles.  Tile l+1 is driven by ZOH symbols = the S/H samples of tile l (launched at tile l's
    sampling instant).  Returns per-layer held samples in MODEL UNITS ((OUT - 2.5)/0.5) and the voltages."""
    T = setups[0].T
    h_in = np.asarray(x, np.float64) * input_scale
    t_start = 4 * T
    ys, volts, lo, hi = [], [], 9.0, -9.0
    cur = VCM + A_IN0 * h_in                                   # volts [I, N]
    for l, su in enumerate(setups[:n_layers]):
        r = tran_tile(su.tile, cur, T, su.t0, su.cal["vb"], workdir, f"{tag}_L{l + 1}", t_start=t_start)
        volts.append(r["out_v"])
        ys.append((r["out_v"] - VCM) / A_ACT)
        lo, hi = min(lo, r["out_min"]), max(hi, r["out_max"])
        t_start = t_start + su.t0                                # next tile's symbols launch at this tile's sampling instant
        cur = r["out_v"]
    return {"y": ys, "volts": volts, "out_min": lo, "out_max": hi}


# --------------------------------------------------------------------------------------------- metrics, subsets, calibration of the post chain
def nmse(a: np.ndarray, ref: np.ndarray) -> float:
    """sum over channels of mean-square error / sum over channels of the variance of the reference (same definition as circuit.py)."""
    return float(((a - ref) ** 2).mean(axis=(0, 2)).sum() / ref.var(axis=(0, 2)).sum())


def stratified_idx(ds: dict, n: int, offset: int = 0) -> np.ndarray:
    """Deterministic class x SNR balanced subset of the test set (round robin over the (class, SNR) cells)."""
    y, s = np.asarray(ds["y_test"]), np.asarray(ds["snr_test"])
    cells = [np.where((y == c) & (s == r))[0] for c in np.unique(y) for r in np.unique(s)]
    out, k = [], offset
    while len(out) < n:
        for c in cells:
            if k < len(c) and len(out) < n:
                out.append(int(c[k]))
        k += 1
    return np.array(out[:n])


ADC_BITS, ADC_VREF = 12, 3.3


def adc_units(y_units: np.ndarray) -> np.ndarray:
    v = VCM + A_ACT * y_units
    lsb = ADC_VREF / 2 ** ADC_BITS
    q = np.clip(np.round(v / lsb), 0, 2 ** ADC_BITS - 1) * lsb
    return ((q - VCM) / A_ACT).astype(np.float32)


def cal_forward(model: Model, cals: list[LayerCal], x: np.ndarray, adc: bool = True) -> dict:
    """Circuit-calibrated behavioural forward (NOT SPICE) on x [B, 2, N]."""
    h = np.asarray(x, np.float32) * np.float32(model.input_scale)
    outs = []
    for c in cals:
        h = c.forward_units(h)
        outs.append(h)
    feats = [adc_units(o) if adc else o for o in outs]
    return {"out": virtual.readout(model, virtual.features(model, feats)), "layers": outs}


def readout_from_samples(model: Model, ys: list[np.ndarray], adc: bool = True) -> np.ndarray:
    feats = [adc_units(o) if adc else o for o in ys]
    return virtual.readout(model, virtual.features(model, feats))


def fit_post(setups: list[LayerSetup], inputs: list[np.ndarray], outputs: list[np.ndarray], taus=None, lps=None, es=None) -> dict:
    """Fit, per layer, the post-chain parameters of the behavioural model against SPICE transients of CALIBRATION examples (train set):
    pure delay tau_post, first-order low-pass tau_lp and S/H memory e.  Minimises NMSE(behavioural layer driven by the SPICE input, SPICE output)."""
    taus = np.arange(-100e-9, 1001e-9, 50e-9) if taus is None else taus
    lps = (0.0, 50e-9, 100e-9, 150e-9, 200e-9, 300e-9, 450e-9) if lps is None else lps
    es = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3) if es is None else es
    res = []
    for l, su in enumerate(setups):
        best = (9e9, 0.0, 0.0, 0.0)
        for lp in lps:
            su.set_tau(0.0, lp)
            for tau in taus:
                su.set_tau(float(tau))
                for e in es:
                    sc = nmse(su.calibrated(e).forward_units(inputs[l]), outputs[l])
                    if sc < best[0]:
                        best = (sc, float(tau), float(e), float(lp))
        su.set_tau(best[1], best[3])
        su.e_fit = best[2]
        res.append({"layer": l + 1, "nmse": best[0], "tau_post_s": best[1], "e_sh": best[2], "tau_lp_s": best[3]})
    return {"layers": res}


# --------------------------------------------------------------------------------------------- orchestration
def build_setups(info: dict, model: Model, workdir: Path, q: float = 10.0, temp_c: float = 25.0, mc: dict | None = None, T: float | None = None,
                 vb_trims: list | None = None, rf_scale: float = 2.0, rt_to_vref: bool = True, opamp: str = "fit") -> list[LayerSetup]:
    out = []
    for L in (1, 2, 3):
        mc_l = None if mc is None else dict(mc, seed=int(mc["seed"]) * 10 + L)
        tl = Tile(info["layers"][str(L)]["cir"], L, model, opamp=opamp, q=q, temp_c=temp_c, rf_scale=rf_scale, mc=mc_l, rt_to_vref=rt_to_vref)
        out.append(LayerSetup(tl, Path(workdir) / f"L{L}", T=(T if T else (out[0].T if out else None)),
                              vb_trim=None if vb_trims is None else vb_trims[L - 1]))
    return out


def setup_summary(su: LayerSetup) -> dict:
    t = su.tile
    return {"layer": t.layer, "T_s": su.T, "T_cal_from_centroids_s": su.T_cal, "t0_s": su.t0, "phase_score": su.phase_score, "tau_post_s": su.tau_post,
            "tau_lp_s": su.tau_lp, "e_sh": su.e_fit, "vb": su.cal["vb"].tolist(), "vb_design": su.cal["design_vb"].tolist(),
            "clipped_channels": int(su.cal["clipped"].sum()), "amp_fit": su.fits, "fir": su.fir.tolist(),
            "dc_tap_gain": [float(abs(su.ac["H"][k, 0])) for k in range(8)],
            "tap_gain_at_125kHz": [float(abs(np.interp(125e3, su.ac["freq_hz"], np.abs(su.ac["H"][k])))) for k in range(8)],
            "f3db_tap7_hz": float(su.ac["freq_hz"][np.argmax(np.abs(su.ac["H"][7]) < np.abs(su.ac["H"][7, 0]) * 0.7071)]),
            "R_f": t.R_f.tolist(), "R_b": t.R_b.tolist()}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(path)


def run_chain_batches(setups, model, X, indices, outdir: Path, tag: str, log, batch: int = 8) -> list[np.ndarray]:
    """SPICE chain over examples X[indices]; checkpointed per batch of `batch` examples.  Returns y per layer [n, 8, N]."""
    ys = [[] for _ in range(3)]
    for b0 in range(0, len(indices), batch):
        f = outdir / f"{tag}_{b0:03d}.npz"
        if f.exists():
            z = np.load(f)
            bl = [z[f"y{l}"] for l in range(3)]
        else:
            t = time.time()
            res = [spice_chain(setups, X[i], model.input_scale, outdir / "_w", f"{tag}_{i}") for i in indices[b0:b0 + batch]]
            bl = [np.stack([r["y"][l] for r in res]) for l in range(3)]
            np.savez(f, **{f"y{l}": bl[l] for l in range(3)}, idx=indices[b0:b0 + batch], rail=[min(r["out_min"] for r in res), max(r["out_max"] for r in res)])
            log(f"{tag} batch {b0}-{b0 + len(res)} done in {time.time() - t:.0f} s")
        for l in range(3):
            ys[l].append(bl[l])
    return [np.concatenate(y) for y in ys]


def acc(out: np.ndarray, y: np.ndarray) -> float:
    return float((out.argmax(1) == y).mean())


def layer_metrics(model, setups, x, y_spice, fir_m1a) -> dict:
    """Per-layer NMSE: SPICE vs V (ideal), SPICE vs V-on-M1a-FIR, calibrated(layer driven by the SPICE input) vs SPICE."""
    vi = virtual.forward_layers(model, x)
    vf = v_forward_fir(model, x, fir=fir_m1a)["layers"]
    out = []
    for l in range(3):
        inp = y_spice[l - 1] if l else x * np.float32(model.input_scale)
        cal = setups[l].calibrated().forward_units(inp)
        out.append({"layer": l + 1, "nmse_spice_vs_V": nmse(y_spice[l], vi[l]), "nmse_spice_vs_V_on_M1a_FIR": nmse(y_spice[l], vf[l]),
                    "nmse_calibrated_vs_spice": nmse(cal, y_spice[l]), "n_examples": int(len(x))})
    return {"layers": out}


def choose_operating_point(model, setups, x, y, Ts=None, phases=None, log=None) -> dict:
    """MCU-style bring-up: pick the symbol clock T and the tap-sampling phase t0_eff that maximise the TRAIN-subset accuracy of the
    behavioural model (validated against SPICE).  One common (T, t0_eff) for the three tiles; each tile's S/H clock phase = t0_eff + its tau_post."""
    Ts = np.array([0.86, 0.88, 0.89, 0.90, 0.92, 0.946, 0.97]) * 1e-6 if Ts is None else Ts
    phases = np.linspace(0.70e-6, 1.50e-6, 17) if phases is None else phases
    best = (-1.0, None, None)
    grid = []
    for T in Ts:
        for ph in phases:
            for su in setups:
                su.set_clock(float(T), float(ph))
            a = acc(cal_forward(model, cal_models(setups), x)["out"], y)
            grid.append((float(T), float(ph), a))
            if a > best[0]:
                best = (a, float(T), float(ph))
    for su in setups:
        su.set_clock(best[1], best[2])
    if log:
        log(f"operating point: T={best[1] * 1e6:.3f} us t0_eff={best[2] * 1e6:.3f} us train-subset acc={best[0]:.4f}")
    return {"train_subset_acc": best[0], "T_us": best[1] * 1e6, "t0_eff_us": best[2] * 1e6, "grid_best5": sorted(grid, key=lambda g: -g[2])[:5]}


def cal_models(setups) -> list[LayerCal]:
    return [su.calibrated() for su in setups]


def run_model(model_path: str | Path, outroot: str | Path, n_test: int = 64, n_calib: int = 4, corners=(10.0, 30.0), n_mc: int = 20,
              temps=(0.0, 25.0, 50.0), n_temp_ex: int = 3, n_mc_ex: int = 2, full: bool = True, rf_scale: float = 2.0, n_test_q30: int = 16) -> dict:
    """Everything for one model.  Resumable: every SPICE batch is a checkpoint file under outroot."""
    from ..data import modrec
    model_path, outroot = Path(model_path), Path(outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    logf = outroot / "progress.log"

    def log(msg):
        with open(logf, "a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")

    t_all = time.time()
    prev_all = json.loads((outroot / "result.json").read_text()).get("corners", {}) if (outroot / "result.json").exists() else {}
    model = load_model(model_path)
    ds = modrec.dataset()
    name = model_path.stem.replace("_ternary", "")
    fir_m1a = json.loads((REPO / "models" / "board_aware" / "line_fir_m1a.json").read_text())
    info = export_tiles(model_path, outroot / "tile")
    info["model_sha256_canonical"] = model.sha256()
    save_json(outroot / "export.json", info)
    log(f"{name}: exported tiles, cir sha {[v['cir_sha256'][:8] for v in info['layers'].values()]}, model file sha {info['model_file_sha256'][:12]}")
    idx = stratified_idx(ds, n_test)
    xt = ds["x_test"]
    Xs = xt[idx]
    # calibration examples: TRAIN set, never test
    ytr, str_ = np.asarray(ds["y_train"]), np.asarray(ds["snr_train"])
    cidx = np.array([int(np.where((ytr == c) & (str_ == 10))[0][0]) for c in (0, 2, 3, 4)][:n_calib])
    Xc = ds["x_train"][cidx]
    result = {"model": name, "model_file": info["model_file"], "model_file_sha256": info["model_file_sha256"], "tile_cir_sha256": {k: v["cir_sha256"] for k, v in info["layers"].items()},
              "gen_tile_sha256": info["gen_tile_sha256"], "kicad_cli": info["kicad_cli"], "notice": NOTICE, "eco": {"rt_to_vref25": True, "rf_rb_ro_scale": rf_scale}, "test_subset_idx": idx.tolist(),
              "corners": {}}
    v_full = virtual.forward(model, xt)["out"]
    ytest = ds["y_test"]
    result["V_ideal_accuracy_4000"] = acc(v_full, ytest)
    result["V_on_M1a_FIR_accuracy_4000"] = {c: acc(v_forward_fir(model, xt, fir=fir_m1a["corners"][c]["fir"])["out"], ytest) for c in ("Q10", "Q30")}
    nominal_trims = None
    for q in corners:
        qn = f"Q{int(q)}"
        wd = outroot / qn
        t_q = time.time()
        if q != corners[0]:
            idx = stratified_idx(ds, n_test_q30)
            Xs = xt[idx]
        su = build_setups(info, model, wd, q=q, rf_scale=rf_scale)
        log(f"{qn}: setups done T={su[0].T * 1e6:.4f} us t0={[round(s.t0 * 1e9) for s in su]} ns clipped={[int(s.cal['clipped'].sum()) for s in su]}")
        # calibration transients -> post-chain parameters; clock/phase chosen once (stored in result.json and REUSED on resume so that the
        # SPICE checkpoints stay valid)
        prev = prev_all.get(qn)
        if prev and "operating_point" in prev:
            opp = prev["operating_point"]
            for s_, st in zip(su, prev["setups"]):
                s_.set_clock(st["T_s"], st["t0_s"])
                s_.t0 = st["t0_s"]
            log(f"{qn}: reusing stored operating point {opp['T_us']:.3f} us / {opp['t0_eff_us']:.3f} us")
        else:
            ycal = run_chain_batches(su, model, Xc, np.arange(len(Xc)), wd, "calib0", log, batch=len(Xc))
            post = fit_post(su, [Xc * np.float32(model.input_scale)] + ycal[:2], ycal)
            log(f"{qn}: post-chain fit (initial clock) {post}")
            rs = np.random.default_rng(5).permutation(len(ds["x_train"]))[:1500]
            opp = choose_operating_point(model, su, ds["x_train"][rs], np.asarray(ds["y_train"])[rs], log=log)
        ycal = run_chain_batches(su, model, Xc, np.arange(len(Xc)), wd, "calib", log, batch=len(Xc))
        inputs = [Xc * np.float32(model.input_scale)] + ycal[:2]
        post = fit_post(su, inputs, ycal)
        log(f"{qn}: post-chain fit (final clock) {post}")
        ysp = run_chain_batches(su, model, Xs, np.arange(len(Xs)), wd, "test", log)
        lm = layer_metrics(model, su, Xs, ysp, fir_m1a["corners"][qn]["fir"])
        # accuracy on the subset
        o_sp = readout_from_samples(model, ysp)
        o_v = virtual.forward(model, Xs)["out"]
        o_vf = v_forward_fir(model, Xs, fir=fir_m1a["corners"][qn]["fir"])["out"]
        cal_sub = cal_forward(model, cal_models(su), Xs)
        # calibrated chain vs SPICE chain (whole network from the input)
        chain_nmse = [nmse(cal_sub["layers"][l], ysp[l]) for l in range(3)]
        ysub = ds["y_test"][idx]
        # full 4000 with the calibrated model
        full_out = cal_forward(model, cal_models(su), xt)
        cacc = acc(full_out["out"], ytest)
        pred_c, pred_v = full_out["out"].argmax(1), v_full.argmax(1)
        zero_pad = None
        result["corners"][qn] = {
            "Q": q, "T_us": su[0].T * 1e6, "operating_point": opp, "setups": [setup_summary(s) for s in su], "post_fit": post, "layer_metrics": lm["layers"],
            "calibrated_chain_vs_spice_chain_nmse": chain_nmse,
            "subset": {"n": int(len(idx)), "acc_spice": acc(o_sp, ysub), "acc_V": acc(o_v, ysub), "acc_V_on_FIR": acc(o_vf, ysub), "acc_calibrated": acc(cal_sub["out"], ysub),
                       "agree_spice_vs_V": float((o_sp.argmax(1) == o_v.argmax(1)).mean()), "agree_spice_vs_calibrated": float((o_sp.argmax(1) == cal_sub["out"].argmax(1)).mean())},
            "full4000": {"acc_C_calibrated": cacc, "acc_V_ideal": result["V_ideal_accuracy_4000"],
                         "acc_V_on_M1a_FIR": result["V_on_M1a_FIR_accuracy_4000"][qn], "C_minus_V_points": 100 * (cacc - result["V_ideal_accuracy_4000"]),
                         "C_minus_V_on_FIR_points": 100 * (cacc - result["V_on_M1a_FIR_accuracy_4000"][qn]),
                         "agreement_with_V": float((pred_c == pred_v).mean()),
                         "acc_per_snr": {str(int(s)): float((pred_c[np.asarray(ds['snr_test']) == s] == ytest[np.asarray(ds['snr_test']) == s]).mean()) for s in np.unique(ds['snr_test'])}},
            "rail_check_v": [float(np.load(f)["rail"][0]) for f in sorted(wd.glob("test_*.npz"))][:1] + [float(np.load(f)["rail"][1]) for f in sorted(wd.glob("test_*.npz"))][:1],
            "wall_s": time.time() - t_q}
        save_json(outroot / "result.json", result)
        log(f"{qn}: C acc(4000)={cacc:.4f} V={result['V_ideal_accuracy_4000']:.4f} Vfir={result['V_on_M1a_FIR_accuracy_4000'][qn]:.4f}; nmse cal-vs-spice {[round(x['nmse_calibrated_vs_spice'], 4) for x in lm['layers']]}")
        if q == corners[0]:
            nominal_trims = [s.cal for s in su]
            nominal_su = su
    # ---- temperature (Q10 nominal board, DAC trim and clock frozen at 25 C)
    if temps:
        tres = {}
        T0 = nominal_su[0].T
        for tc in temps:
            f = outroot / f"temp_{tc:g}.json"
            if f.exists():
                tres[str(tc)] = json.loads(f.read_text())
                continue
            sut = build_setups(info, model, outroot / "temp" / f"T{tc:g}", q=corners[0], temp_c=tc, T=T0, vb_trims=nominal_trims, rf_scale=rf_scale)
            for a, b in zip(sut, nominal_su):
                a.set_tau(b.tau_post, b.tau_lp); a.e_fit = b.e_fit
            ytt = run_chain_batches(sut, model, Xs, np.arange(n_temp_ex), outroot / "temp" / f"T{tc:g}", "t", log, batch=n_temp_ex)
            vi = virtual.forward_layers(model, Xs[:n_temp_ex])
            fo = cal_forward(model, cal_models(sut), xt)
            tres[str(tc)] = {"temp_c": tc, "nmse_spice_vs_V": [nmse(ytt[l], vi[l]) for l in range(3)],
                             "nmse_calibrated_vs_spice": [nmse(sut[l].calibrated().forward_units(ytt[l - 1] if l else Xs[:n_temp_ex] * np.float32(model.input_scale)), ytt[l]) for l in range(3)],
                             "acc_C_calibrated_4000": acc(fo["out"], ytest), "ron_ohm": RON25 + RON_TC * (tc - 25), "dcr_ohm": DCR0 * (1 + DCR_TC * (tc - 25)),
                             "T_cal_us": sut[0].T_cal * 1e6, "dac_trim_frozen_at_25C": True}
            save_json(f, tres[str(tc)])
            log(f"temp {tc}: {tres[str(tc)]}")
        result["temperature"] = tres
    # ---- Monte Carlo on L and C (agauss in the netlist), board re-trimmed (DAC) and clock re-calibrated per draw
    if n_mc:
        mcres = []
        for k in range(n_mc):
            f = outroot / "mc" / f"mc_{k:02d}.json"
            if f.exists():
                mcres.append(json.loads(f.read_text()))
                continue
            sig = {"sigma_L": 0.05 / math.sqrt(3), "sigma_C": 0.05 / math.sqrt(3)}
            sm = build_setups(info, model, outroot / "mc" / f"w{k}", q=corners[0], mc=dict(seed=1000 + k, **sig), rf_scale=rf_scale)
            for a, b in zip(sm, nominal_su):
                a.set_tau(b.tau_post, b.tau_lp); a.e_fit = b.e_fit
            ymc = run_chain_batches(sm, model, Xs, np.arange(n_mc_ex), outroot / "mc" / f"w{k}", "m", log, batch=n_mc_ex)
            vi = virtual.forward_layers(model, Xs[:n_mc_ex])
            fo = cal_forward(model, cal_models(sm), xt)
            rec = {"run": k, "seed": 1000 + k, "T_us": sm[0].T * 1e6, "sigma_L_rel": sig["sigma_L"], "sigma_C_rel": sig["sigma_C"],
                   "nmse_spice_vs_V": [nmse(ymc[l], vi[l]) for l in range(3)],
                   "nmse_calibrated_vs_spice": [nmse(sm[l].calibrated().forward_units(ymc[l - 1] if l else Xs[:n_mc_ex] * np.float32(model.input_scale)), ymc[l]) for l in range(3)],
                   "acc_C_calibrated_4000": acc(fo["out"], ytest), "clipped": [int(s.cal["clipped"].sum()) for s in sm]}
            save_json(f, rec)
            mcres.append(rec)
            log(f"mc {k}: acc {rec['acc_C_calibrated_4000']:.4f} T {rec['T_us']:.4f}")
        a = np.array([r["acc_C_calibrated_4000"] for r in mcres])
        result["monte_carlo"] = {"n": len(mcres), "acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "acc_min": float(a.min()), "acc_max": float(a.max()),
                                 "T_us_mean": float(np.mean([r["T_us"] for r in mcres])), "T_us_std": float(np.std([r["T_us"] for r in mcres], ddof=1)),
                                 "runs": mcres, "model_of_variation": "L, C of every ladder element ~ agauss(nominal, nominal*0.05/sqrt(3)) (sigma matches a uniform +-5 % band); DAC re-trimmed and clock re-calibrated per draw; SPICE chain on 2 examples per draw, accuracy from the re-extracted calibrated model on 4000"}
    result["wall_s"] = time.time() - t_all
    save_json(outroot / "result.json", result)
    log(f"{name}: done in {result['wall_s'] / 60:.1f} min")
    return result


# --------------------------------------------------------------------------------------------- notebook API (no hidden state)
def export_netlist(model_npz: str | Path, layer: int, outdir: str | Path) -> Path:
    """gen_tile.py --model + kicad-cli spice export for ONE layer; returns the .cir path (a few seconds)."""
    info = export_tiles(model_npz, Path(outdir), layers=(layer,))
    return Path(info["layers"][str(layer)]["cir"])


def calibrate(model_npz: str | Path, outdir: str | Path, q: float = 10.0, temp_c: float = 25.0, T: float | None = None, rf_scale: float = 2.0) -> dict:
    """`.ac` + `.dc` of the three tiles (about 15-30 s): FIR, amplifier fits, DAC trim, clock/phase.  Returns {'setups': [LayerSetup x3],
    'summary': [dict x3], 'info': export info}; writes outdir/calibration_Q<q>.json."""
    outdir = Path(outdir)
    model = load_model(model_npz)
    info = export_tiles(model_npz, outdir / "tile")
    su = build_setups(info, model, outdir / f"spice_Q{int(q)}", q=q, temp_c=temp_c, T=T, rf_scale=rf_scale)
    summ = [setup_summary(s) for s in su]
    save_json(outdir / f"calibration_Q{int(q)}.json", {"model_file_sha256": info["model_file_sha256"], "tile_cir_sha256": {k: v["cir_sha256"] for k, v in info["layers"].items()},
                                                      "layers": summ})
    return {"setups": su, "summary": summ, "info": info}


def run_spice(setups: list[LayerSetup], model: Model, examples: np.ndarray, outdir: str | Path, tag: str = "ex") -> dict:
    """Whole-tile `.tran` of the 3-layer chain for `examples` [n, 2, N] (about 20 s per example).  Returns y per layer [n, 8, N] in model
    units (S/H samples), plus the rail check.  No checkpoints; use run_model for resumable batch runs."""
    outdir = Path(outdir)
    res = [spice_chain(setups, x, model.input_scale, outdir / "_w", f"{tag}{i}") for i, x in enumerate(np.asarray(examples))]
    return {"y": [np.stack([r["y"][l] for r in res]) for l in range(3)], "out_min_v": min(r["out_min"] for r in res), "out_max_v": max(r["out_max"] for r in res)}


def export_web(outdir: str | Path, model_npz: str | Path, run_id: str = "adhoc", rf_scale: float = 2.0) -> dict:
    """Browser exports (unit_cell.template.cir, tap_response.json, diffpair_dc.json, manifest.json with sha256) into `outdir`; about 1 minute."""
    outdir = Path(outdir)
    work = outdir / "_work"
    a, b = calibrate(model_npz, work, 10.0, rf_scale=rf_scale), calibrate(model_npz, work, 30.0, rf_scale=rf_scale)
    return export_browser(outdir, a["setups"], b["setups"], Path(model_npz).stem, run_id,
                          {"tile_cir_sha256": {k: v["cir_sha256"] for k, v in a["info"]["layers"].items()}, "model_file_sha256": a["info"]["model_file_sha256"], "eco": {"rt_to_vref25": True, "rf_rb_ro_scale": rf_scale}})


def run(model_npz, dataset: dict, cfg: dict, outdir):
    """Tier entry point of the common contract (board mode).  Re-exports the KiCad netlists for this model, re-runs `.ac`/`.dc` on every tile,
    applies the operating point and S/H parameters stored by `run_model` (out/P2-M2a/<model>/result.json, key cfg['c_board_result']) or the
    defaults, and evaluates the test set with the CIRCUIT-CALIBRATED BEHAVIOURAL model (not SPICE).  Use `python -m dlnn.tiers.circuit_board`
    for the SPICE .tran runs."""
    from . import common
    t0 = time.time()
    outdir = Path(outdir)
    model_npz = Path(model_npz)
    model = load_model(model_npz)
    q = float(cfg.get("c_board_q", 10.0))
    info = export_tiles(model_npz, outdir / "tile")
    su = build_setups(info, model, outdir / "spice", q=q)
    res_path = cfg.get("c_board_result")
    stored = None
    if res_path and Path(res_path).exists():
        stored = json.loads(Path(res_path).read_text())["corners"].get(f"Q{int(q)}")
    for l, s_ in enumerate(su):
        if stored:
            pf = stored["post_fit"]["layers"][l]
            s_.tau_post, s_.e_fit = pf["tau_post_s"], pf["e_sh"]
            op = stored["operating_point"]
            s_.set_clock(op["T_us"] * 1e-6, op["t0_eff_us"] * 1e-6)
        else:
            s_.tau_post, s_.e_fit = 0.22e-6, 0.01
            s_.set_clock(s_.T, s_.t0_opt)
    cals = cal_models(su)
    ver = circuit.ngspice_version()
    fid = {"models": [f"ngspice {ver} on the KiCad-exported TILE-8 netlist (hw/board/gen_tile.py --model, kicad-cli spice export, sha256 "
                      f"{', '.join(v['cir_sha256'][:12] for v in info['layers'].values())}): .ac per tap and .dc of the bias->pair->S/H chain per tile; "
                      "the SPICE .tran of the whole chain (64 test examples) is run by `python -m dlnn.tiers.circuit_board` and stored in out/P2-M2a/",
                      "the numbers of THIS result (full test set, sweeps) come from the CIRCUIT-CALIBRATED BEHAVIOURAL model built from those SPICE results "
                      "(tap FIR from .ac, per-channel tanh from .dc, S/H memory and alignment fitted to calibration transients); NOT SPICE results",
                      NOTICE],
           "omits": ["layout parasitics, crosstalk, mutual inductance, noise (this is the P tier)", "TI TLV9062 macro in the full tile (OP/transient too slow; single-pole fit used)",
                     "ECO-1/ECO-2 netlist edits (line termination to VREF25, R_f/R_b/R_o x2) are applied: the exported schematic as committed does not work",
                     "ribbon between tiles (S/H samples are handed over as ideal ZOH symbols)", "ADC 12 bit is modelled, DAC bias trim is ideal-calibrated"]}
    return common.evaluate_tier(tier="C", status="ok", tool={"name": "ngspice", "version": ver}, model=model, dataset=dataset, cfg=cfg,
                                forward_fn=lambda x, c: cal_forward(model, cals, x), ref_fn=None, fidelity=fid, sweep_axes=("snr_db",), t0=t0,
                                inputs_from=[], warnings=[f"C board mode, Q = {q:g}: behavioural model calibrated to SPICE"])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="C tier, board mode (ngspice on the KiCad tile netlist). Resumable.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--corners", default="10,30")
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--no-temp", action="store_true")
    ap.add_argument("--rf-scale", type=float, default=2.0, help="ECO-2: R_f, R_b, R_o multiplier (2.0 = tap-gain restoration for an ideal-line net; ~2.9 for the M1a-FIR board nets)")
    ap.add_argument("--n-test-q30", type=int, default=16)
    a = ap.parse_args(argv)
    mp = Path(a.model)
    out = Path(a.out) if a.out else REPO / "out" / "P2-M2a" / mp.stem.replace("_ternary", "")
    r = run_model(mp, out, n_test=a.n_test, corners=tuple(float(c) for c in a.corners.split(",")), n_mc=a.mc, temps=() if a.no_temp else (0.0, 25.0, 50.0), rf_scale=a.rf_scale, n_test_q30=a.n_test_q30)
    print(json.dumps({k: v for k, v in r.items() if k in ("model", "V_ideal_accuracy_4000", "V_on_M1a_FIR_accuracy_4000", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------------------- browser export
UNIT_CELL_TEMPLATE = """* TILE-8 unit cell for the browser (ngspice-wasm): ONE delay line (56 LC sections) + 8 taps + one inverting summer + BJT diff pair.
* SIMULATION of a design that is not built.  Reduced model, no proprietary TI macro: the op-amps are SINGLE-POLE op-amps
* (A0 = 1e5, GBW = 10 MHz, labelled fit, not the TLV9062 macro); inductor = L + DCR + Cp(SRF) + Rp(Q) (datasheet-derived lumped);
* NPN = generic BC847-fitted.  Placeholders {{...}} are filled by the widget: {{Q_RP}} Rp for the chosen Q (Q=10: 6350.9, Q=30: 21561.8),
* {{RXP0}}..{{RXP7}} / {{RXN0}}..{{RXN7}} = 10k if the tap weight is +1 / -1 on that side, else 1G (footprint not populated), {{VIN_PWL}} PWL of the input voltage about 2.5 V.
.param RP={{Q_RP}} DCR=3.3 CP=19.383p
.model QNPN npn (IS=1.3520e-14 BF=300 VAF=100 RB=10 TF=0.6n CJE=13p CJC=4.8p MJE=0.33 MJC=0.33 VJE=0.75 VJC=0.75)
.subckt OPA inp inn out
E1 x 0 inp inn 1e5
R1 x y 1k
C1 y 0 1.5915e-06
Eo z 0 y 0 1
Ro z out 100
.ends OPA
Vdd vdd 0 5
Vref ref 0 2.5
Vin vin 0 PWL({{VIN_PWL}})
Rp0 vin rc 1k
Cp0 rc 0 330p
Xib rc ib ib OPA
Rs ib n0 1k
C00 n0 0 60p
%(ladder)s
Rt n56 ref 1k
%(taps)s
* crossbar: W=+1 populates Rxp (to sn), W=-1 populates Rxn (to sp), as on the board
%(xbar)s
Xs1 ref sn on OPA
Rfn sn on 1058
Cfn sn on 15p
Xs2 ref sp op OPA
Rfp sp op 1058
Cfp sp op 15p
* generic BJT pair, mirror-less 1 mA ideal tail (the board uses a BJT mirror; see tap/dc exports for the full tile)
Q1 c1 on tail QNPN
Q2 c2 op tail QNPN
Itail tail 0 1m
Rl1 vdd c1 1k
Rl2 vdd c2 1k
Eout out 0 c2 c1 1
.end
"""


def export_browser(outdir: Path, su10: list[LayerSetup], su30: list[LayerSetup], model_name: str, run_id: str, extra: dict) -> dict:
    """docs/artifacts/data/spice/: unit_cell.template.cir, tap_response.json, diffpair_dc.json (schema + sha)."""
    outdir.mkdir(parents=True, exist_ok=True)
    ladder = []
    for k in range(1, 57):
        ladder.append(f"Rd{k} n{k - 1} m{k} {{DCR}}\nL{k} m{k} n{k} 120u\nCpp{k} n{k - 1} n{k} {{CP}}\nRpp{k} n{k - 1} n{k} {{RP}}\nC{k} n{k} 0 {'60p' if k == 56 else '120p'}")
    taps = "\n".join(f"Xt{k} n{8 * k} t{k} t{k} OPA" for k in range(8))
    xbar = []
    for k in range(8):
        xbar.append(f"Rxp{k} t{k} sn {{{{RXP{k}}}}}\nRxn{k} t{k} sp {{{{RXN{k}}}}}")
    text = UNIT_CELL_TEMPLATE % {"ladder": "\n".join(ladder), "taps": taps, "xbar": "\n".join(xbar)}
    (outdir / "unit_cell.template.cir").write_text(text)
    # tap response (SPICE .ac of the whole layer-2 tile, line 0): magnitude (dB) and phase (deg), log-thinned
    f = su10[1].ac["freq_hz"]
    sel = np.unique(np.round(np.geomspace(1, len(f), 160)).astype(int) - 1)
    def block(su):
        H = su.ac["H"]
        return {"mag_db": [[round(float(20 * np.log10(abs(H[k, i]))), 3) for i in sel] for k in range(8)],
                "phase_deg": [[round(float(np.degrees(np.angle(H[k, i]))), 2) for i in sel] for k in range(8)],
                "T_us": round(su.T * 1e6, 4), "fir_symbol_rate": np.round(su.fir, 4).tolist()}
    tap = {"schema": "dlnn-spice-tap-response/1", "run_id": run_id, "model": model_name, "source": "ngspice 42 .ac of the KiCad-exported TILE-8 layer-2 tile, line 0, TLV906x single-pole fit",
           "notice": NOTICE, "freq_hz": [round(float(f[i]), 1) for i in sel], "Q10": block(su10[1]), "Q30": block(su30[1]),
           "taps": "T0_0..T0_7 = outputs of the tap buffers at sections 0,8,..,56; H = V(tap)/V(SIG0)"}
    (outdir / "tap_response.json").write_text(json.dumps(tap))
    dc = su10[1].dc
    ch = 0
    dpj = {"schema": "dlnn-spice-diffpair-dc/1", "run_id": run_id, "model": model_name,
           "source": "ngspice 42 .dc of the whole bias -> summer -> BJT diff pair (mirror tail) -> difference amp -> S/H -> buffer chain, layer-2 tile, channel 0, inputs at 2.5 V",
           "notice": NOTICE, "x": "V_dac (V) at the bias DAC output (0..2.048 V on the board)", "y": "tile output OUT0 (V)",
           "v_dac": np.round(dc["vsw"][::4], 4).tolist(), "out_v": np.round(dc["out"][ch][::4], 5).tolist(),
           "fit": {"form": "out = 2.5 + A tanh((v_dac - shift)/vs) + offset", "A": su10[1].fits[ch]["A"], "vs": su10[1].fits[ch]["vs"], "shift": su10[1].fits[ch]["shift"],
                   "offset": su10[1].fits[ch]["offset"], "r2": su10[1].fits[ch]["r2"]}}
    (outdir / "diffpair_dc.json").write_text(json.dumps(dpj))
    meta = {"schema": "dlnn-spice-export/1", "run_id": run_id, "model": model_name, "files": {}, **extra}
    for fn in ("unit_cell.template.cir", "tap_response.json", "diffpair_dc.json"):
        meta["files"][fn] = {"sha256": sha_file(outdir / fn), "bytes": (outdir / fn).stat().st_size}
    (outdir / "manifest.json").write_text(json.dumps(meta, indent=1))
    return meta
