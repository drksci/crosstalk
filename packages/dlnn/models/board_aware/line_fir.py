"""LC delay-line tap FIRs from ngspice (board-aware training input).

Netlist: DAC step -> Rs 1k (source termination) with C_in 106 pF (=> 1.5 MHz RC), 56 constant-k sections
(series L 120 uH with DCR 3 ohm, shunt C 120 pF), end termination 1k.  Taps at nodes 0,8,..,56 (K = 8).
A single-sample (one T_s wide) pulse is applied; each tap's response is sampled on the symbol clock.
Output: line_fir.json {"fir": [K][M], ...}, line.cir.   Run: python models/board_aware/line_fir.py
"""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
L, DCR, C, Z0, SECT, K, STEP = 120e-6, 3.0, 120e-12, 1000.0, 56, 8, 8
CIN = 1 / (2 * np.pi * Z0 * 1.5e6)           # 1.5 MHz input RC with the 1k source resistance
TS_NOM = 8 * np.sqrt(L * C)                  # 0.96 us
TAPS = list(range(0, SECT + 1, STEP))
M = 12                                       # FIR length (lags) stored


def netlist(tpulse=TS_NOM, tstop=24e-6, tr=10e-9):
    ln = [f"* DLNN LC delay line, 56 sections, 1k terminated, 1.5 MHz input RC",
          f"Vin src 0 PULSE(0 1 1n {tr:.3e} {tr:.3e} {tpulse - tr:.6e} 1)",
          f"Rs src n0 {Z0:g}", f"Cin n0 0 {CIN:.4e}", f"Cs0 n0 0 {C / 2:.3e}"]
    for i in range(1, SECT + 1):
        ln += [f"L{i} n{i-1} m{i} {L:.3e}", f"RL{i} m{i} n{i} {DCR:g}", f"C{i} n{i} 0 {C if i < SECT else C / 2:.3e}"]
    ln += [f"Rt n{SECT} 0 {Z0:g}", f".tran 2n {tstop:.3e} 0 2n",
           ".control", "run", "set wr_vecnames", "option numdgt=7",
           "wrdata OUT.txt " + " ".join(f"v(n{t})" for t in TAPS), ".endc", ".end"]
    return "\n".join(ln) + "\n"


def run():
    (HERE / "line.cir").write_text(netlist())
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "l.cir").write_text(netlist())
        subprocess.run(["ngspice", "-b", "l.cir"], cwd=d, check=True, capture_output=True)
        raw = np.loadtxt(Path(d) / "OUT.txt", skiprows=1)
    t = raw[:, 0]
    v = np.stack([raw[:, 1 + 2 * i] for i in range(len(TAPS))])        # [K, T] pulse responses
    return t, v


def main():
    t, v = run()
    # peak time of each tap's single-sample response -> calibrated clock (MCU-calibrated to tap 7)
    tpk = np.array([t[np.argmax(v[k])] for k in range(K)])
    # centroid-based delay (robust to ringing)
    cen = np.array([(t * v[k]).sum() / v[k].sum() for k in range(K)])
    ts_eff = (cen[K - 1] - cen[0]) / (K - 1)
    # sample phase c (common S/H instant): scan one clock period and keep the phase that puts the most response
    # energy on the nominal lags (tap k <-> lag k); lag m is c + m*ts_eff after the pulse launch (lags m >= 0 only).
    def build(c):
        f = np.zeros((K, M))
        for k in range(K):
            for m in range(M):
                f[k, m] = np.interp(c + m * ts_eff, t, v[k])
        return f
    best = None
    for c in np.linspace(0.05e-6, 1.6e-6, 160):
        f = build(c)
        score = sum(f[k, k] ** 2 / (f[k] ** 2).sum() for k in range(K))
        if best is None or score > best[0]:
            best = (score, c, f)
    _, c0, fir = best
    fir = fir / np.mean([fir[k, k] for k in range(K)])      # mean on-lag gain = 1 (summer gain is learnt per layer)
    ideal = np.zeros_like(fir)
    for k in range(K):
        ideal[k, k] = 1.0
    main_lag = fir.argmax(1)
    # fraction of the response energy NOT at the nominal lag k (ringing / skew)
    off = [float(1 - fir[k, k] ** 2 / (fir[k] ** 2).sum()) for k in range(K)]
    out = dict(fir=fir.round(6).tolist(), ts_eff_us=float(ts_eff * 1e6), ts_nom_us=TS_NOM * 1e6,
               centroid_us=(cen * 1e6).round(4).tolist(), peak_us=(tpk * 1e6).round(4).tolist(),
               peak_gain_vs_tap0_db=(20 * np.log10(v.max(1) / v[0].max())).round(2).tolist(),
               main_lag=main_lag.tolist(), sample_phase_us=float(c0 * 1e6), off_nominal_energy_frac=[round(x, 4) for x in off],
               note="ngspice transient of the 56-section ladder; fir[k][m] = tap-k response to a one-sample pulse, sampled m clock "
                    "periods after the tap-0 centroid, clock = (centroid7-centroid0)/7; normalised by tap-0 DC sum.",
               params=dict(L=L, DCR=DCR, C=C, Z0=Z0, sections=SECT, taps=TAPS, cin=CIN))
    (HERE / "line_fir.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("ts_eff_us", "centroid_us", "peak_gain_vs_tap0_db", "main_lag", "off_nominal_energy_frac")}, indent=1))
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    print(fir)


if __name__ == "__main__":
    main()
