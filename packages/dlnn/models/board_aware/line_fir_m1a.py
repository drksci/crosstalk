"""Per-tap symbol-rate responses of the REAL M1a line (hw/board/spice/linelib.py, hw/board/params.json) at Q = 10 and Q = 30.

Netlist = linelib.line_netlist (56 sections, L 120 uH + DCR 3.3, C 120 pF, half-value end caps, 6 pF tap loads, 1k/1k terminations,
input RC 1 kohm + 330 pF = 482 kHz behind an IDEAL unity buffer: linelib's line deck does not instantiate the TLV9062 macro, so
buffers are not in the FIR).  Input = one zero-order-hold symbol (pulse of width T_s).  Taps = nodes 0, 8, ..., 56.
Clock T_s = mean over both corners of (centroid tap7 - centroid tap0)/7 (MCU-calibrated, common to both corners); one common S/H phase
(maximises on-lag energy at Q = 10).  Common gain normalisation = mean on-lag gain of the two corners (so Q = 10 vs Q = 30 loss is kept).
Stores fir[corner][K][M] at symbol rate AND dense[corner][K][(M+1)*P+1] at T_s/P (P = 8) for the +-14 % delay-jitter augmentation.
Run: .venv/bin/python models/board_aware/line_fir_m1a.py
"""
import hashlib, json, math, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hw/board/spice"))
import linelib as ll                                   # noqa: E402

K, M, P = 8, 12, 8
D = json.loads((ROOT / "hw/board/params.json").read_text())["design_inputs"]


def corner_kw(Q):
    return dict(L=D["L_h"], C=D["C_f"], n=D["n_sections"], tap_every=D["tap_every"], dcr=D["ind_dcr_ohm"], srf=D["ind_srf_hz"],
                Q=Q, f_q=D["ind_Q_fq_hz"], rs=D["Rs_ohm"], rl=D["Rl_ohm"], prefilter=True, rc_r=D["input_rc_R_ohm"],
                f_rc=1.0 / (2 * math.pi * D["input_rc_R_ohm"] * D["input_rc_C_f"]), c_tap=D["c_tap_load_f"], cend=D["C_end_f"])


def pulse_response(Q, tpulse, tstop=30e-6, dt=2e-9):
    src = f"PULSE(0 1 1u 10n 10n {tpulse - 10e-9:.6e} 1)"
    body, taps = ll.line_netlist(src=src, **corner_kw(Q))
    vec = " ".join(f"v({t})" for t in taps)
    deck = f"M1a FIR pulse\n{body}\n.control\nset filetype=ascii\ntran {dt} {tstop} 0 {dt}\nwrdata fir_{int(Q)}.txt {vec}\n.endc\n.end\n"
    o, _ = ll.run_ngspice(deck, [f"fir_{int(Q)}.txt"], tag=f"fir_{int(Q)}")
    d = o[f"fir_{int(Q)}.txt"]
    return d[:, 0], np.array([d[:, 2 * i + 1] for i in range(len(taps))]), deck


def main():
    ver = subprocess.run(["ngspice", "--version"], capture_output=True, text=True).stdout
    ngv = re.search(r"ngspice-\d+", ver).group(0)
    corners = {"Q10": 10.0, "Q30": 30.0}
    # 1) calibrate the clock from a first pass with a nominal pulse width
    ts = 0.96e-6
    for _ in range(2):
        cens = []
        for Q in corners.values():
            t, v, _ = pulse_response(Q, ts)
            cens.append([(t * v[k]).sum() / v[k].sum() for k in range(K)])
        ts = float(np.mean([(c[K - 1] - c[0]) / (K - 1) for c in cens]))
    resp, decks = {}, {}
    for name, Q in corners.items():
        resp[name] = pulse_response(Q, ts)[:2]
        decks[name] = pulse_response(Q, ts)[2]
    # 2) S/H phase from Q10 (centroid of tap 0 response is the reference launch time)
    t10, v10 = resp["Q10"]

    def sample(t, v, c, offs):
        return np.array([[np.interp(c + o * ts, t, v[k]) for o in offs] for k in range(K)])
    c_best, s_best = None, -1
    for c in np.linspace(1e-6, 1e-6 + 1.2 * ts, 241):          # S/H instant measured from the pulse start (1 us in the deck)
        f = sample(t10, v10, c, range(M))
        s = sum(f[k, k] ** 2 / (f[k] ** 2).sum() for k in range(K))
        if s > s_best:
            c_best, s_best = c, s
    c0 = c_best
    firs, dense, stats = {}, {}, {}
    for name in corners:
        t, v = resp[name]
        firs[name] = sample(t, v, c0, range(M))
        dense[name] = sample(t, v, c0 - ts, np.arange((M + 1) * P + 1) / P)      # dense[k, j] = h_k(c0 + (j/P - 1) T_s)
    gain = np.mean([firs[n][k, k] for n in corners for k in range(K)])
    for name in corners:
        f = firs[name] / gain
        firs[name], dense[name] = f, dense[name] / gain
        on = np.array([f[k, k] for k in range(K)])
        energy_off = np.array([1 - f[k, k] ** 2 / (f[k] ** 2).sum() for k in range(K)])
        stats[name] = dict(on_lag_gain=on.round(4).tolist(), off_nominal_energy_frac=energy_off.round(4).tolist(),
                           max_neighbour_leak=float(max(abs(f[k, k + s]) / abs(f[k, k]) for k in range(K) for s in (-1, 1) if 0 <= k + s < M)),
                           main_lag=f.argmax(1).tolist())
    sha = hashlib.sha256("\n".join(decks[n] for n in corners).encode()).hexdigest()
    out = dict(schema="dlnn-line-fir-m1a/1", K=K, M=M, P=P, ts_us=ts * 1e6, sample_phase_after_pulse_start_us=(c_best - 1e-6) * 1e6,
               corners={n: dict(Q=corners[n], Q_at_hz=D["ind_Q_fq_hz"], fir=firs[n].round(6).tolist(),
                                dense=dense[n].round(6).tolist(), **stats[n]) for n in corners},
               dense_grid="dense[k][j] = h_k(c0 + (j/P - 1) T_s), j = 0..(M+1)P; fir[k][m] = dense[k][P + m P]",
               provenance=dict(netlist_sha256=sha, ngspice=ngv, linelib_sha256=hashlib.sha256((ROOT / "hw/board/spice/linelib.py").read_bytes()).hexdigest(),
                               params_json="hw/board/params.json", generator="models/board_aware/line_fir_m1a.py",
                               contents="line + input RC only (ideal unity buffer after the RC); no TLV9062 macro, no noise, no crosstalk, no mismatch",
                               design=dict(sections=D["n_sections"], L_h=D["L_h"], C_f=D["C_f"], C_end_f=D["C_end_f"], dcr=D["ind_dcr_ohm"],
                                           srf_hz=D["ind_srf_hz"], rc_ohm=D["input_rc_R_ohm"], rc_c_f=D["input_rc_C_f"], c_tap_f=D["c_tap_load_f"],
                                           Rs=D["Rs_ohm"], Rl=D["Rl_ohm"]),
                               normalisation="common: mean on-lag gain over both corners and all taps = 1"))
    (ROOT / "models/board_aware/line_fir_m1a.json").write_text(json.dumps(out))
    print("ts_us", ts * 1e6, "phase_us", (c_best - 1e-6) * 1e6, "sha", sha[:12], ngv)
    for n in corners:
        print(n, json.dumps(stats[n]))
        np.set_printoptions(precision=2, suppress=True, linewidth=170)
        print(np.array(out["corners"][n]["fir"]))


if __name__ == "__main__":
    main()
