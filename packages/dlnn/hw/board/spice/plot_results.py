#!/usr/bin/env python3
"""Figures for docs/board_design.md (re-simulates the cheap decks; run after run_checks.py).  Output: hw/board/spice/figures/*.png"""
import sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import linelib as ll, derive_params as dp, run_checks as rc
D = dp.DESIGN
out = HERE / "figures"; out.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# 1: tap magnitude + group-delay spacing
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for tag, q, ls in (("Q = 10 (datasheet min)", D["ind_Q_min"], "-"), ("Q = 30 (assumed typ)", D["ind_Q_assumed_typ"], "--")):
    kw = rc.line_deck_ac(None, q); body, taps = ll.line_netlist(**kw)
    vec = " ".join(f"v({t})" for t in taps)
    deck = f"x\n{body}\n.control\nset filetype=ascii\nac lin 1500 1k 1501k\nwrdata a.txt {vec}\n.endc\n.end\n"
    o, _ = ll.run_ngspice(deck, ["a.txt"], tag="plot_a"); d = o["a.txt"]; f = d[:, 0]
    H = np.array([d[:, 3 * i + 1] + 1j * d[:, 3 * i + 2] for i in range(len(taps))])
    for k in (0, 3, 7):
        ax[0].plot(f / 1e3, 20 * np.log10(abs(H[k] / H[0])), ls, label=f"tap {k}, {tag}" if k == 7 else None, color=f"C{[0,3,7].index(k)}")
    gd = -np.gradient(np.unwrap(np.angle(H), axis=1), 2 * np.pi * f, axis=1)
    m = (f > 20e3) & (f < 400e3)
    for k in range(1, 8):
        if ls == "-":
            ax[1].plot(f[m] / 1e3, (gd[k] - gd[k - 1])[m] * 1e6, lw=0.8)
ax[0].set_xlim(0, 1000); ax[0].set_ylim(-30, 1); ax[0].axhline(-3, color="k", lw=0.5); ax[0].axvline(250, color="k", lw=0.5, ls=":")
ax[0].set_xlabel("kHz"); ax[0].set_ylabel("dB rel. tap 0"); ax[0].set_title("tap gain vs frequency (taps 0, 3, 7)"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("kHz"); ax[1].set_ylabel("tap-to-tap delay (us)"); ax[1].set_title("tap spacing vs frequency (Q min)"); ax[1].set_ylim(0.8, 1.2)
# 2: steps
step = "PULSE(0 1 1u 50n 50n 80u 200u)"
for tag, kw, c in (("Q min", rc.line_deck_ac(None, D["ind_Q_min"]), "C0"), ("Q 30 (assumed)", rc.line_deck_ac(None, D["ind_Q_assumed_typ"]), "C1"),
                   ("lossless bound", rc.line_deck_ac(None, 1e5, dcr=0.0), "C2")):
    t, V = rc.tran_run(step, kw, tag="plot_s")
    ax[2].plot((t - 1e-6) * 1e6, V[-1] / V[-1][-200:].mean(), color=c, label=tag)
kw = rc.line_deck_ac(None, 1e5, dcr=0.0); kw["f_rc"] = 1.5e6; kw["rc_r"] = 1000.0
t, V = rc.tran_run(step, kw, tag="plot_s"); ax[2].plot((t - 1e-6) * 1e6, V[-1] / V[-1][-200:].mean(), "k:", lw=0.8, label="plan: 1.5 MHz RC, lossless")
ax[2].set_xlim(4, 14); ax[2].set_xlabel("us after the edge"); ax[2].set_title("tap-7 step response, 482 kHz input RC"); ax[2].legend(fontsize=7); ax[2].axhline(1.1, color="r", lw=0.5)
fig.tight_layout(); fig.savefig(out / "line_response.png"); plt.close(fig)

# 3: diff pair
fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
for tag, kw in (("current-mirror tail", dict(tail="bjt")), ("resistor tail (plan)", dict(tail="resistor"))):
    o, _ = ll.run_ngspice(rc.diffpair_deck(**kw), ["dp_dc.txt"], tag="plot_e"); d = o["dp_dc.txt"]
    r = rc.fit_tanh(d[:, 0], d[:, 1])
    ax[0].plot(d[:, 0] * 1e3, d[:, 1], label=f"{tag}: R2={r['R2']:.5f}, vsat={r['vsat_mV']:.1f} mV, A={r['A_v']:.3f} V")
    ax[1].plot(d[:, 0] * 1e3, (d[:, 1] - (r['A_v'] * np.tanh((d[:, 0] - r['input_shift_mV'] * 1e-3) / (r['vsat_mV'] * 1e-3)) + r['offset_mV'] * 1e-3)) * 1e3)
ax[0].set_xlabel("v_d (mV)"); ax[0].set_ylabel("V_out diff (V)"); ax[0].legend(fontsize=6); ax[0].set_title("diff pair (generic NPN fitted to BC847BS table points)")
ax[1].set_xlabel("v_d (mV)"); ax[1].set_ylabel("fit residual (mV)"); ax[1].set_title("residual of A tanh(v/vsat)+off")
fig.tight_layout(); fig.savefig(out / "diffpair_tanh.png"); plt.close(fig)
print("figures written to", out)
