"""Figures 1-8 of plan section 5 (matplotlib; PNG + SVG; light and dark; transparent backgrounds).

    fig1_network   tapped-delay-line network from the model's real weights (polarity markers, lengths in metres)
    fig2_traces    one test example's line voltages after each layer: V/E/C/R overlaid + residual vs V
    fig3_channel   tap-path responses and |H(f)| from C (ngspice .ac) vs R (scikit-rf); folded combiner waveforms at layer 1 (not a true eye)
    fig4_outputs   per-tier class probabilities for 4 examples, KL annotated, TRUE label from the dataset
    fig5_sweeps    accuracy and mean KL vs the sweep axes, tiers as series, one block per axis
    fig6_confusion confusion-matrix difference of each tier vs V
    fig7_agreement top-1 agreement heatmap tier x tier
    fig8_llm       LLM commentary panel with citation chips; unverified dimmed; TEMPLATE labelled as such

Entry points: ``render(name, run_dir, theme, ...)`` builds + saves one figure, ``make_all(run_dir)`` all of them for both
themes (never raises: a failing figure is skipped and reported in ``make_all.errors``), ``build(name, run, theme)``
returns the matplotlib Figure (used by the notebooks). Stub/fallback tiers are labelled as such.
Colours: tiers use the validated dataviz categorical slots 1-4; honesty-relevant context is annotated directly.
"""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .common import (AXIS_LABEL, INK, LINESTYLES, LOSS_RG58, MARKERS, SEQ, T_S_NS, TIER_COL, TIER_LONG,  # noqa: E402
                     TIER_SHORT, TIERS, V_P_CM_NS, Run, conf_matrix, ctx, end_labels, lum, save_fig, tier_label, wrap)

M_PER_TAP_UNIT = T_S_NS * V_P_CM_NS / 100.0    # metres of RG-58 per symbol of delay (1.98 m)
AXIS_ORDER = ["snr_db", "adc_bits", "loss_db_per_us", "isolation_db", "tap_amp_err_db"]


def comp_modes(run: Run) -> dict | None:
    """C accuracy under the three loss-compensation modes at the run's fitted loss ({'layer','none','tap'}), from
    params/circuit_diagnostics.json and compare.json; None when unavailable (regression task, no diagnostics, C not the default 'layer')."""
    try:
        d = json.loads((run.dir / "params" / "circuit_diagnostics.json").read_text())
        if d.get("tier_default") != "layer":
            return None
        out = {"layer": float(run.cmp["metric_matrix"]["accuracy"]["C"])}
        for m in ("none", "tap"):
            out[m] = float(d["modes"][m]["metrics_at_fitted_loss"]["accuracy"])
        return out
    except Exception:  # noqa: BLE001
        return None


def trim_matters(run: Run) -> bool:
    """True when C's default per-layer gain trim scores clearly below the other modes in this run (>= 2 accuracy points)."""
    m = comp_modes(run)
    return bool(m) and max(m["none"], m["tap"]) - m["layer"] >= 0.02


def _foot(fig, text: str, theme: str, width: int = 118, y: float = 0.0, size: float = 7.2):
    lines = []
    for para in text.split("\n"):
        lines += wrap(para, width)
    fig.text(0.01, y, "\n".join(lines), ha="left", va="top", fontsize=size, color=INK[theme]["sec"], linespacing=1.35)


def _mono_ok(run: Run) -> bool:
    return run.task == "modrec"


# ======================================================================================================== fig 1
def build_fig1(run: Run, theme: str):
    """Network / tap diagram from the model in the run: delay lines as bars, tap polarity markers (the actual weights of
    output unit 0), tap positions in metres of RG-58 (v_p = 19.8 cm/ns, T_s = 10 ns), loss of each tap at 32 dB/us."""
    ink = INK[theme]
    model = run.model
    layers = model.layers
    n = len(layers)
    hdr_in, ruler_in = 0.86, 0.12
    data_in = [0.5 + 0.074 * l.W.shape[1] for l in layers]
    rows = [0.55]
    for dd in data_in:
        rows += [hdr_in, ruler_in + dd]
    rows += [0.7]
    fig = plt.figure(figsize=(7.0, sum(rows) + 1.0))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows, hspace=0.0, left=0.02, right=0.98, top=0.99, bottom=0.10)

    # --- header: input + legend of markers
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.text(0.0, 0.78, "Input: 2 baseband lines (I, Q), 128 samples, one sample per T_s = 10 ns (PAM train, +/-0.5 V)",
            fontsize=8, color=ink["text"], fontweight="bold", transform=ax.transAxes, va="center")
    x0 = 0.0
    for lab, mk, fill in (("+1 : coupler straight to combiner", "^", True), ("-1 : through a 180 deg hybrid", "v", False),
                          ("0 : no tap", ".", None)):
        if fill is None:
            ax.plot([x0 + 0.01], [0.28], ".", color=ink["muted"], ms=5, transform=ax.transAxes)
        else:
            ax.plot([x0 + 0.01], [0.28], mk, mfc=ink["text"] if fill else "none", mec=ink["text"], ms=5.5, mew=1.1,
                    transform=ax.transAxes)
        ax.text(x0 + 0.03, 0.28, lab, fontsize=7.2, color=ink["sec"], va="center", transform=ax.transAxes)
        x0 += 0.36

    total_loss = []
    for l, ly in enumerate(layers):
        O, I, K = ly.W.shape
        d = int(ly.dilation)
        Wl = ly.W
        o_show = 0
        nz = (Wl != 0).mean()
        pos, neg = (Wl > 0).mean(), (Wl < 0).mean()
        tap_m = np.arange(K) * d * M_PER_TAP_UNIT
        tap_ns = np.arange(K) * d * T_S_NS
        loss_db = tap_ns * 1e-3 * LOSS_RG58
        total_loss.append(loss_db[-1])
        N = I * K
        h = fig.add_subplot(gs[1 + 2 * l]); h.axis("off")
        h.text(0.0, 0.80, f"Layer {l + 1}: {I} -> {O} lines, K = {K} taps, dilation d = {d}  (tap spacing {d * T_S_NS:.0f} ns = "
               f"{d * M_PER_TAP_UNIT:.2f} m)", fontsize=8.4, fontweight="bold", color=ink["text"], va="center", transform=h.transAxes)
        h.text(0.0, 0.50, f"nonzero taps {100 * nz:.0f} % (+1: {100 * pos:.0f} %, -1: {100 * neg:.0f} %) over all {O * I * K} (o,i,k) "
               f"entries; markers below = output unit {o_show}", fontsize=7, color=ink["sec"], va="center", transform=h.transAxes)
        ax = fig.add_subplot(gs[2 + 2 * l])
        ax.set_xlim(-1.35, K + 2.55)
        ax.set_ylim(I - 0.4, -0.6)
        ax.axis("off")
        xa = ax.get_xaxis_transform()
        for k in range(K):
            ax.annotate(f"{tap_m[k]:.1f} m", (k, 1.0), xycoords=xa, xytext=(0, 17), textcoords="offset points", ha="center",
                        va="center", fontsize=6.8, color=ink["text"])
            ax.annotate(f"{tap_ns[k]:.0f} ns", (k, 1.0), xycoords=xa, xytext=(0, 8), textcoords="offset points", ha="center",
                        va="center", fontsize=6.2, color=ink["sec"])
            ax.plot([k, k], [-0.35, I - 0.65], color=ink["grid"], lw=0.6, zorder=0)
        for txt, off in (("length", 17), ("delay", 8)):
            ax.annotate(txt, (-0.45, 1.0), xycoords=xa, xytext=(0, off), textcoords="offset points", ha="right", va="center",
                        fontsize=6.4, color=ink["sec"])
        lw_bar = 3.0 if I <= 4 else 1.4
        ms = 5.4 if I <= 4 else 3.0
        for i in range(I):
            ax.plot([-0.15, K - 0.85], [i, i], color=ink["axis"], lw=lw_bar, solid_capstyle="round", zorder=1)
            ax.text(-0.55, i, ("I", "Q")[i] if I == 2 else (f"{i}" if i % 8 == 0 or i == I - 1 else ""), ha="right",
                    va="center", fontsize=6.8, color=ink["sec"])
            for k in range(K):
                w = int(Wl[o_show, i, k])
                if w == 1:
                    ax.plot(k, i, "^", mfc=ink["text"], mec=ink["text"], ms=ms, zorder=3)
                elif w == -1:
                    ax.plot(k, i, "v", mfc="none", mec=ink["text"], ms=ms, mew=1.0, zorder=3)
                else:
                    ax.plot(k, i, ".", color=ink["muted"], ms=2.4 if I > 4 else 4, zorder=2)
        yc = (I - 1) / 2
        ax.annotate("", xy=(K + 0.2, yc), xytext=(K - 0.75, yc), arrowprops=dict(arrowstyle="-|>", color=ink["sec"], lw=1.0))
        ax.text(K + 1.45, yc, f"N = {N} port\ncombiner (-{10 * np.log10(N):.0f} dB)\n+ amp gain g, tanh", ha="center", va="center",
                fontsize=6.8, color=ink["text"], linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.5", fc="none", ec=ink["axis"], lw=0.9))
        ax.annotate(f"longest tap: {loss_db[-1]:.1f} dB\nat {LOSS_RG58:.0f} dB/us", (K + 1.45, yc), xytext=(0, -32),
                    textcoords="offset points", ha="center", va="top", fontsize=6.6, color=ink["warn"], linespacing=1.25)
    # --- footer
    ax = fig.add_subplot(gs[-1]); ax.axis("off")
    feat = model.readout_W.shape[1]
    ax.text(0.0, 0.5, "RC average pool\ntau = T * T_s (passive)", fontsize=7.3, color=ink["text"], va="center", ha="left",
            linespacing=1.3, bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.5", fc="none", ec=ink["axis"], lw=0.9),
            transform=ax.transAxes)
    ax.annotate("", xy=(0.31, 0.5), xytext=(0.22, 0.5), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=ink["sec"], lw=1.0))
    ax.text(0.33, 0.5, f"ADC 8 bit -> linear {feat} -> {model.readout_W.shape[0]} -> softmax\n(the only digital stage)", fontsize=7.3,
            color=ink["text"], va="center", ha="left", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.5", fc="none", ec=ink["axis"], lw=0.9), transform=ax.transAxes)
    longest = max(total_loss)
    _foot(fig, f"Geometry: RG-58, v_p = {V_P_CM_NS} cm/ns, T_s = {T_S_NS:.0f} ns, so one symbol of delay = {M_PER_TAP_UNIT:.2f} m. "
               f"At {LOSS_RG58:.0f} dB/us the layer-3 dilation-16 taps reach {longest:.0f} dB of loss: this is why the plan says they "
               "need re-amplification. C and R run at that loss without repeaters in the later figures and fall behind V"
               + ("; for C most of that gap comes from its default per-layer gain-trim model, not from the loss alone (Fig 5, docs/circuit_extraction.md). "
                  if trim_matters(run) else ". ")
               + "Markers are the trained ternary weights of output unit 0. Simulation only, no hardware.",
          theme, width=112, y=0.085)
    return fig


# ======================================================================================================== fig 2
def build_fig2(run: Run, theme: str, example: int = 0):
    """Line voltages after each layer for one test example: V/E/C/R overlaid + residual vs V (activations are the
    arrays stored in each tier's npz; the C layer-1 SPICE samples come from the stored unit-cell waveforms)."""
    ink, col = INK[theme], TIER_COL[theme]
    tiers = run.tiers
    n_layers = len(run.model.layers) if run.model is not None else 3
    fig = plt.figure(figsize=(7.0, 1.05 + 2.55 * n_layers + (0.9 if _mono_ok(run) else 0)))
    nrow = 2 * n_layers + (1 if _mono_ok(run) else 0)
    hr = ([0.9] if _mono_ok(run) else []) + [2.1, 1.0] * n_layers
    gs = fig.add_gridspec(nrow, 1, height_ratios=hr, hspace=0.42, left=0.09, right=0.84, top=0.93, bottom=0.04)
    r0 = 0
    snr_txt = ""
    if _mono_ok(run):
        x = run.x_test[example]
        ax = fig.add_subplot(gs[0])
        ax.plot(x[0], color=ink["sec"], lw=1.0, label="I")
        ax.plot(x[1], color=ink["muted"], lw=1.0, ls="--", label="Q")
        ax.set_ylabel("input")
        ax.set_xticklabels([])
        cls = run.classes[int(run.labels[example])]
        snr_txt = f"{cls}, {run.snr[example]:.0f} dB SNR"
        r0 = 1
        ax.set_title(f"Test example {example}: {snr_txt} (true class); V predicts {run.classes[int(run.res['V'].arrays['pred'][example])]}.  Input: solid I, dashed Q",
                     fontsize=8.5)
    nm_rows = []
    cal_spice = None
    wv = run.res["C"].arrays.get(f"waveforms.ex{example}") if "C" in run.res else None
    for l in range(n_layers):
        tv = run.trace("V", l)
        if tv is None:
            continue
        ch = run.pick_channel(l, example, limit=8 if l == 0 else None)
        a = fig.add_subplot(gs[r0 + 2 * l])
        b = fig.add_subplot(gs[r0 + 2 * l + 1], sharex=a)
        v = tv[example, ch]
        nr = {}
        end_items = []
        for t in tiers:
            tr = run.trace(t, l)
            if tr is None:
                continue
            y = tr[example, ch]
            a.plot(y, color=col[t], ls=LINESTYLES[t], lw=2.6 if t == "V" else 1.5, marker=MARKERS[t] if t != "V" else None,
                   ms=3.0, markevery=6 if t != "V" else None, zorder=3 if t == "V" else 4, label=t)
            end_items.append((len(y) - 1, float(y[-1]), t))
            if t != "V":
                r = y - v
                b.plot(r, color=col[t], ls=LINESTYLES[t], lw=1.3)
                nr[t] = float(np.mean(r ** 2) / max(np.var(v), 1e-12))
        if l == 0 and wv is not None and "C" in tiers and ch < wv.shape[0] - 2:
            idx = np.arange(128) * 10 + 8
            ys = wv[2 + ch, idx]
            a.plot(np.arange(128), ys, "o", mfc="none", mec=ink["text"], ms=3.2, mew=0.7, zorder=6, label="SPICE")
            cal_spice = float(np.mean((ys - run.trace("C", 0)[example, ch]) ** 2) / np.var(ys))
            end_items.append((127, float(ys[-1]) - 0.0, ""))
        a.set_ylabel(f"layer {l + 1}\nline {ch}")
        a.set_xticklabels([])
        b.set_ylabel("tier - V")
        txt = "   ".join(f"NMSE {t} {nr[t]:.1e}" if nr[t] < 0.01 else f"NMSE {t} {nr[t]:.2f}" for t in nr)
        b.text(1.0, 1.02, txt, transform=b.transAxes, ha="right", va="bottom", fontsize=6.6, color=ink["sec"])
        nm_rows.append(nr)
        if l < n_layers - 1:
            b.set_xticklabels([])
    b.set_xlabel("sample n (one per T_s = 10 ns; 128 samples = 1.28 us)")
    h, lab = [], []
    for t in tiers:
        h.append(Line2D([], [], color=col[t], ls=LINESTYLES[t], lw=2.4 if t == "V" else 1.5, marker=MARKERS[t] if t != "V" else None, ms=3.5))
        lab.append(tier_label(t, run.status[t]))
    if wv is not None and "C" in tiers:
        h.append(Line2D([], [], ls="", marker="o", mfc="none", mec=ink["text"], ms=4))
        lab.append("ngspice samples, layer-1 unit cell")
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.035))
    lossnote = ""
    if run.cascade and "C" in run.sweeps_by_axis("loss_db_per_us"):
        sw = run.sweeps_by_axis("loss_db_per_us")["C"]
        lossnote = (f" With --cascade, C and R run at the extracted {LOSS_RG58:.0f} dB/us (equal to the value put into the netlist) with no repeaters, so their layer-2/3 "
                    f"lines shrink and smear (residuals grow with depth)." + (" C's default per-layer gain trim additionally over-drives layer 3 (docs/circuit_extraction.md)." if trim_matters(run) else "")
                    + f" At 0 dB/us C's accuracy is {sw['accuracy'][0]:.2f} (Fig 5).")
    spice = (f" The hollow circles are ngspice samples of the layer-1 unit cell (NMSE of the calibrated model vs SPICE here: "
             f"{cal_spice:.1e}). " if cal_spice is not None else " ")
    _foot(fig, "V = ideal math; E = bit-true fixed point; C layers 1-3 on the full test set = circuit-calibrated behavioural model "
               "(layers 2-3 extrapolated, NOT SPICE)." + spice + "R = scikit-rf channel model + noise." + lossnote, theme, width=116, y=-0.005)
    return fig


# ======================================================================================================== fig 3
def _spice_combiner_wave(run: Run, example: int, ch: int):
    """ngspice .tran of the layer-1 unit cell for ONE output, saving the summing node. Cached in figures/.eye_cache.
    Returns (t_s, u_units) on a 1 ns grid, gain-trim applied; None if ngspice is unavailable."""
    cache = run.dir / "figures" / f".eye_cache_ex{example}_o{ch}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["u"]
    from ..tiers import circuit
    if circuit.ngspice_path() is None or run.params is None or not run.params.get("unit_cell_tran") \
            or run.params.get("source") != "ngspice":
        return None
    p = run.params
    model = run.model
    ly = model.layers[0]
    ex = run.x_test[example:example + 1].astype(np.float32) * np.float32(model.input_scale)
    vpu = float(p["unit_cell_tran"]["volts_per_unit"])
    r_nom = float(p["line"]["R_ohm_per_m_nominal"])
    trim = float(p["loss_comp_gain_trim_per_layer"][0])
    N = ex.shape[2]
    nl = circuit.netlist_unit_cell(ly, [ch], ex[0], vpu, r_nom, trim, with_pool=False)
    nl = nl.replace(".save ", f".save v(sum{ch}) ", 1)
    with tempfile.TemporaryDirectory(prefix="dlnn_eye_") as td:
        f = Path(td) / "eye.cir"
        f.write_text(nl)
        pl = circuit.run_ngspice(f)[0]
    nm, d = pl["names"], pl["data"]
    t = d[:, nm.index("time")].real
    tg = np.arange(0, N * circuit.TS, 1e-9)
    u = np.interp(tg, t, d[:, nm.index(f"v(sum{ch})")].real) / vpu * trim
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, t=tg, u=u)
    except OSError:
        pass
    return tg, u


def _fold_eye(ax, t_ns, u, t_sample_ns, color, n_skip=12, n_end=8, Ts=T_S_NS, lw=0.5, alpha=0.5):
    """Fold u(t) onto a 2*Ts window centred on the sampling instants."""
    n_sym = int((t_ns[-1] - t_sample_ns[0]) / Ts)
    for n in range(n_skip, n_sym - n_end):
        c = t_sample_ns[0] + n * Ts
        m = (t_ns >= c - Ts) & (t_ns <= c + Ts)
        ax.plot(t_ns[m] - c, u[m], color=color, lw=lw, alpha=alpha)


def build_fig3(run: Run, theme: str, example: int = 0, allow_spice: bool = True):
    """Tap-path responses and |H(f)| of C (ngspice .ac of the layer-1 line bank) vs R (scikit-rf cascade), and the eye
    at the layer-1 combiner (C: ngspice summing node, one output, 1 example; R: stored scikit-rf fine-grid waveform)."""
    from ..tiers import circuit, rf
    ink, col = INK[theme], TIER_COL[theme]
    p = run.params
    if p is None:
        raise FileNotFoundError("params/circuit_extracted.json missing: figure 3 needs the C-tier extraction")
    model = run.model
    ly = model.layers[0]
    O, I, K = ly.W.shape
    d = int(ly.dilation)
    # ---------------- C: SPICE H_k(f)
    ft = np.asarray(p["taps"]["freq_hz"])
    Hc = np.asarray(p["taps"]["H_re"]) + 1j * np.asarray(p["taps"]["H_im"])
    # ---------------- R: scikit-rf at the run's operating point
    cfg = {}
    if "R" in run.res and run.res["R"].inputs_from:
        from .. import cascade
        cfg = cascade.apply_to_rf_cfg({}, cascade.load_extracted(run.dir / "params" / "circuit_extracted.json"))
    rc = rf.resolve_cfg(cfg)
    g = rf.make_grid(rc, 3 * d * (K - 1) + 16)
    Hr, stat = {}, None
    ks = [k for k in (1, 3, 7) if k < K]
    for k in range(1, K):
        H, stat = rf.line_response(rc, g, k * d, I * K)
        Hr[k] = H / stat
    c_src = p.get("source") == "ngspice"
    c_lab = "C: ngspice H_k(f)" if c_src else "C: analytic RLC (FALLBACK, not SPICE)"
    # response of each tap path to one raised-cosine (beta=1, 10 ns) symbol = impulse response seen at the signal bandwidth
    Ts = T_S_NS * 1e-9

    def rc_spec(f):
        x = np.clip(f * Ts, 0, 1)
        return np.where(f * Ts <= 1.0, 0.5 * Ts * (1 + np.cos(np.pi * x)), 0.0)

    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.5, 1.7, 1.7, 0.05], hspace=0.7, wspace=0.3, left=0.085, right=0.97,
                          top=0.93, bottom=0.02)
    # ---- (a) responses
    f_c = np.fft.rfftfreq(4096, 1e-9)
    Hci = np.stack([np.interp(f_c, ft, Hc[k].real) + 1j * np.interp(f_c, ft, Hc[k].imag) for k in range(K)])
    dt_r = T_S_NS / rc["oversample"]
    for j, k in enumerate(ks):
        a = fig.add_subplot(gs[0, j])
        yc = np.fft.irfft(Hci[k] * rc_spec(f_c), n=4096) * 1e9
        tc = np.arange(4096) * 1.0
        yr = np.fft.irfft(Hr[k] * rc_spec(g.f), n=g.N) * g.fs
        tr = np.arange(g.N) * dt_r
        tau = k * d * T_S_NS
        for sym in range(-1, 4):
            a.axvline(sym * T_S_NS, color=ink["grid"], lw=0.7, zorder=0)
        # C: t = 0 is the nominal tap delay (its line is cut to exactly that delay); R is already delay-trimmed
        a.plot(tc - tau, yc, color=col["C"], ls=LINESTYLES["C"], lw=1.7)
        a.plot(np.where(tr > g.N * dt_r / 2, tr - g.N * dt_r, tr), yr, color=col["R"], ls=LINESTYLES["R"], lw=1.4)
        a.set_xlim(-12, 38)
        a.set_ylim(-0.2, 1.08)
        a.set_title(f"tap k={k} ({k * d * M_PER_TAP_UNIT:.1f} m)", fontsize=8)
        a.set_xlabel("t - k*d*T_s (ns)")
        if j == 0:
            a.set_ylabel("response to one\n10 ns RC symbol")
        else:
            a.set_yticklabels([])
        a.axhline(0, color=ink["axis"], lw=0.6)
        if j == len(ks) - 1:
            a.plot([], [], color=col["C"], ls=LINESTYLES["C"], label=c_lab)
            a.plot([], [], color=col["R"], ls=LINESTYLES["R"], label="R: scikit-rf")
            a.legend(loc="upper right", frameon=False, fontsize=6.8)
    fig.text(0.085, 0.955, "(a) Tap-path impulse response (as seen by a 10 ns raised-cosine symbol; grid = symbol instants)",
             fontsize=8.5, fontweight="bold", color=ink["text"], va="bottom")
    # ---- (b) |H(f)|
    b = fig.add_subplot(gs[1, :])
    fmax = 400e6
    items = []
    for k in ks:
        m = ft <= fmax
        b.plot(ft[m] / 1e6, 20 * np.log10(np.abs(Hc[k, m])), color=col["C"], ls=LINESTYLES["C"], lw=1.7)
        mr = g.f <= fmax
        hr = np.abs(Hr[k][mr])
        b.plot(g.f[mr] / 1e6, 20 * np.log10(np.maximum(hr, 1e-6)), color=col["R"], ls=LINESTYLES["R"], lw=1.3)
        items.append((fmax / 1e6, float(20 * np.log10(abs(Hc[k, np.argmin(abs(ft - fmax))]))), f"k={k}"))
    b.axvline(100, color=ink["muted"], lw=0.8, ls=":")
    b.text(101, b.get_ylim()[0] if False else 0.5, "", fontsize=6)
    b.set_xlim(0, 400)
    b.set_xlabel("frequency (MHz)")
    b.set_ylabel("|H| of the tap path (dB)")
    b.set_title(("(b) |H(f)|: ngspice constant-R line (C)  vs  scikit-rf skin-effect + mismatch ripple (R)" if c_src else
                 "(b) |H(f)|: analytic constant-R line (C, FALLBACK: no ngspice)  vs  scikit-rf (R)"), fontsize=8.5)
    end_labels(b, items, dx=4)
    yl = b.get_ylim()
    b.text(102, yl[0] + 0.04 * (yl[1] - yl[0]), "1/T_s = 100 MHz", fontsize=6.6, color=ink["sec"])
    # ---- (c) eyes
    ch = run.pick_channel(0, example, limit=8)
    wr = run.res["R"].arrays.get(f"waveforms.{example}") if "R" in run.res else None
    wave_c = _spice_combiner_wave(run, example, ch) if allow_spice else (
        (lambda cp: (np.load(cp)["t"], np.load(cp)["u"]) if cp.exists() else None)(run.dir / "figures" / f".eye_cache_ex{example}_o{ch}.npz"))
    ca = fig.add_subplot(gs[2, :2])
    cb = fig.add_subplot(gs[2, 2], sharey=ca)
    lim = 0.0
    if wave_c is not None:
        tg, u = wave_c
        _fold_eye(ca, tg * 1e9, u, np.array([0.8 * T_S_NS]), col["C"])
        lim = max(lim, float(np.abs(u).max()))
        ca.set_title(f"C: ngspice summing node, output {ch}, 1 example", fontsize=8)
        ca.set_xlim(-T_S_NS, T_S_NS)
        ca.axvline(0, color=ink["muted"], lw=0.8, ls=":")
        ca.set_xlabel("time relative to the sampling instant (ns)")
        ca.set_ylabel("combiner voltage\n(model units, gain-trimmed)")
    else:
        ca.text(0.5, 0.5, "no ngspice summing-node data for C in this run: eye skipped", transform=ca.transAxes, ha="center", color=ink["sec"])
        ca.set_title("C: unavailable", fontsize=8)
    if wr is not None:
        U = rc["oversample"]
        wv = wr[ch]
        t_ns = np.arange(len(wv)) * dt_r
        _fold_eye(cb, t_ns, wv, np.array([0.0]), col["R"], alpha=0.35)
        lim = max(lim, float(np.abs(wv[: 128 * U]).max()))
        cb.set_xlim(-T_S_NS, T_S_NS)
        cb.axvline(0, color=ink["muted"], lw=0.8, ls=":")
        cb.set_title("R: scikit-rf combiner", fontsize=8)
        cb.set_xlabel("time rel. to sampling instant (ns)")
        plt.setp(cb.get_yticklabels(), visible=False)
    fig.text(0.085, ca.get_position().y1 + 0.03, "(c) Folded combiner waveforms at layer 1 (not a true eye: multi-level sums; gain-trimmed, model units; same example and output line)",
             fontsize=8.5, fontweight="bold", color=ink["text"], va="bottom")
    ca.set_ylim(-lim * 1.05, lim * 1.05)
    ca.axhline(0, color=ink["axis"], lw=0.6)
    cb.axhline(0, color=ink["axis"], lw=0.6)
    _foot(fig, "Both channels have the same nominal geometry (RG-58, v_p = 19.8 cm/ns for R). C's SPICE line is a constant-R LTRA line, so its DC loss is "
               "overstated (about -3.6 dB vs -2.2 dB at 100 MHz for the longest layer-1 tap); R is skin-effect (~sqrt f) loss with 5 % end mismatch (ripple) "
               "and a 10 dB coupler / N-way Wilkinson (their fixed gain is divided out here). The eyes differ in more than the line: C drives trapezoid "
               "pulses (1 ns edges) through ideal VCVS taps, R uses raised-cosine pulses and includes tap amplitude/phase errors, leakage and crosstalk.", theme, width=116, y=0.0)
    return fig


# ======================================================================================================== fig 4
def pick_examples(run: Run, n: int = 4, pool: int = 16) -> list[int]:
    """Deterministic, informative examples among the first `pool`: (all tiers agree & right), (V right, C or R wrong),
    (V wrong), then fill with the first unused. Falls back to 0..n-1 when the labels are not available."""
    try:
        y = run.labels
    except OSError:
        return list(range(n))
    pr = run.preds()
    pool = min(pool, len(y))
    right = {t: pr[t][:pool] == y[:pool] for t in pr}
    all_right = [i for i in range(pool) if all(right[t][i] for t in right)]
    cr_wrong = [i for i in range(pool) if right["V"][i] and any(not right[t][i] for t in ("C", "R") if t in right)]
    v_wrong = [i for i in range(pool) if not right["V"][i]]
    out: list[int] = []
    for lst in (all_right, cr_wrong, cr_wrong[1:], v_wrong, all_right[1:], list(range(pool))):
        for i in lst:
            if i not in out:
                out.append(i)
                break
        if len(out) >= n:
            break
    for i in range(pool):
        if len(out) >= n:
            break
        if i not in out:
            out.append(i)
    return sorted(out[:n])


def _kl(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def build_fig4(run: Run, theme: str, examples: list[int] | None = None, n_show: int = 4, window: int = 200):
    """Per-tier outputs side by side. modrec: class probabilities for 4 examples (title = TRUE label from the dataset,
    KL(V||tier) annotated). narma10: prediction vs target + residual vs V."""
    ink, col = INK[theme], TIER_COL[theme]
    tiers = run.tiers
    if run.task == "narma10":
        fig, axs = plt.subplots(2, len(tiers), figsize=(1.9 * len(tiers) + 0.6, 4.4), sharex=True, squeeze=False,
                                gridspec_kw={"height_ratios": [3, 1.6]})
        sl = slice(100, 100 + window)
        yv = run.res["V"].arrays["y_pred"][0, sl]
        for j, t in enumerate(tiers):
            a, r = axs[0, j], run.res[t]
            a.plot(r.arrays["y_true"][0, sl], color=ink["muted"], lw=1.1, label="target")
            a.plot(r.arrays["y_pred"][0, sl], color=col[t], ls=LINESTYLES[t], lw=1.2, label="prediction")
            a.set_title(tier_label(t, r.status).replace(" (", "\n("), fontsize=7.5)
            axs[1, j].plot(r.arrays["y_pred"][0, sl] - yv, color=col[t], lw=1.0)
            axs[1, j].set_xlabel("test step")
            if j == 0:
                a.set_ylabel("output"); a.legend(loc="upper right", fontsize=6.5)
                axs[1, j].set_ylabel("tier - V")
        fig.suptitle(f"Per-tier outputs: narma10 (run {run.cmp['run_id']})", fontsize=9, x=0.01, ha="left")
        return fig
    ex = examples or pick_examples(run, n_show)
    y, snr, cls = run.labels, run.snr, run.classes
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.0, 1.55, 1.0], wspace=0.05, hspace=0.32, left=0.1, right=0.99,
                          top=0.9, bottom=0.2)
    ncls = len(cls)
    w = 0.8 / len(tiers)
    for pi, e in enumerate(ex[:4]):
        r_, c_ = divmod(pi, 2)
        a = fig.add_subplot(gs[r_, 2 * c_])
        t_ax = fig.add_subplot(gs[r_, 2 * c_ + 1]); t_ax.axis("off")
        for j, t in enumerate(tiers):
            a.barh(np.arange(ncls) + (j - (len(tiers) - 1) / 2) * w, run.res[t].arrays["probs"][e], height=w * 0.86,
                   color=col[t], zorder=3)
        a.set_yticks(np.arange(ncls)); a.set_yticklabels(cls, fontsize=7.2)
        a.set_xlim(0, 1.0)
        a.grid(axis="y", visible=False)
        a.set_ylim(ncls - 0.45, -0.55)
        lab = int(y[e])
        a.get_yticklabels()[lab].set_fontweight("bold")
        a.axhspan(lab - 0.5, lab + 0.5, color=ink["grid"], alpha=0.8, zorder=0, lw=0)
        if c_ == 1:
            a.set_yticklabels([])
        pv = int(run.res["V"].arrays["pred"][e])
        a.set_title(f"example {e}: true {cls[lab]} @ {snr[e]:.0f} dB", fontsize=8, loc="left")
        if r_ == 1:
            a.set_xlabel("probability")
        kl = {t: _kl(run.res["V"].arrays["probs"][e], run.res[t].arrays["probs"][e]) for t in tiers if t != "V"}
        ok = {t: int(run.res[t].arrays["pred"][e]) == lab for t in tiers}
        t_ax.text(0.08, 0.97, f"V predicts {cls[pv]}", fontsize=6.9, color=ink["sec"], va="top", transform=t_ax.transAxes)
        for i, t in enumerate(tiers):
            yy = 0.84 - i * 0.145
            t_ax.add_patch(Rectangle((0.06, yy - 0.03), 0.07, 0.07, transform=t_ax.transAxes, color=col[t], clip_on=False))
            s1 = "V: reference" if t == "V" else f"KL(V||{t}) = {kl[t]:.3g}"
            t_ax.text(0.17, yy + 0.04, s1, transform=t_ax.transAxes, fontsize=6.9, color=ink["text"], va="center")
            t_ax.text(0.17, yy - 0.035, f"top-1: {cls[int(run.res[t].arrays['pred'][e])]} ({'right' if ok[t] else 'wrong'})",
                      transform=t_ax.transAxes, fontsize=6.3, color=ink["sec"], va="center")
    h = [Line2D([], [], color=col[t], lw=6) for t in tiers]
    fig.legend(h, [tier_label(t, run.status[t]) for t in tiers], loc="upper center", ncol=2, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, 1.0))
    _foot(fig, "Bold class name + shaded row = the TRUE label from the dataset (not V's prediction; V is itself wrong on some examples). "
               "KL is in nats, KL(softmax_V || softmax_tier) on that example. "
               + ("C and R run at 32 dB/us (cascade), so their probabilities are flatter or shifted to neighbouring classes." if run.cascade else ""),
          theme, width=118, y=0.075)
    return fig


# ======================================================================================================== fig 5
def build_fig5(run: Run, theme: str):
    """Accuracy (or NMSE) and mean KL vs each sweep axis, tiers as series, one block (2 stacked panels) per axis."""
    ink, col = INK[theme], TIER_COL[theme]
    cls_task = run.task == "modrec"
    axes_ids = [a for a in AXIS_ORDER if a in run.cmp["sweeps"]]
    metrics = ["accuracy", "mean_kl"] if cls_task else ["nmse"]
    nb = len(axes_ids)
    ncol = 2
    nrow = int(np.ceil((nb + 1) / ncol))
    fig = plt.figure(figsize=(7.0, 3.3 * nrow + 0.5))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.42, wspace=0.34, left=0.11, right=0.93, top=0.955, bottom=0.04)
    status = run.status
    vclean = run.cmp["metric_matrix"].get("accuracy", {}).get("V")
    legend_ax = None
    for bi, ax_id in enumerate(axes_ids + ["__legend__"]):
        r, c = divmod(bi, ncol)
        if ax_id == "__legend__":
            legend_ax = fig.add_subplot(outer[r, c]); legend_ax.axis("off")
            continue
        inner = outer[r, c].subgridspec(len(metrics), 1, height_ratios=[1.35, 1.0][:len(metrics)], hspace=0.12)
        axs = [fig.add_subplot(inner[i]) for i in range(len(metrics))]
        for m, a in zip(metrics, axs):
            items = []
            for t, sw in run.cmp["sweeps"][ax_id].items():
                if m not in sw:
                    continue
                a.plot(sw["values"], sw[m], color=col[t], ls=LINESTYLES[t], marker=MARKERS[t], ms=4.2, lw=1.5,
                       zorder=3, mec=INK[theme]["surface"] if False else col[t])
                items.append((sw["values"][-1], sw[m][-1], t + ("*" if status[t] != "ok" else "")))
            if m == "accuracy" and vclean is not None:
                a.axhline(vclean, color=ink["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
                a.text(0.02, vclean, "V clean", transform=a.get_yaxis_transform(), fontsize=6.2, color=ink["sec"], va="bottom")
            if m == "mean_kl":
                a.set_yscale("symlog", linthresh=1e-3)
                a.set_ylim(bottom=0)
            if m == "nmse":
                a.set_yscale("log")
            a.margins(x=0.1)
            a.set_ylabel({"accuracy": "accuracy", "mean_kl": "mean KL (nats)", "nmse": "NMSE"}[m], fontsize=7.6)
            if m != metrics[-1]:
                a.set_xticklabels([])
            end_labels(a, items, dx=4, fontsize=7)
            if ax_id == "loss_db_per_us":
                op = 32.0 if run.cascade else 5.0
                a.axvline(op, color=ink["warn"], lw=1.0, ls=":", zorder=1)
                if m == metrics[0]:
                    a.text(op - 0.8, 0.97, f"headline run:\n{op:.0f} dB/us" + ("\n(cascade, no repeaters)" if run.cascade else ""),
                           transform=a.get_xaxis_transform(), fontsize=6.4, color=ink["warn"], ha="right", va="top")
        axs[0].set_title(ax_id, fontsize=8.5)
        axs[-1].set_xlabel(AXIS_LABEL[ax_id], fontsize=7.8)
        if ax_id == "loss_db_per_us":
            axs[-1].set_xticks([0, 5, 14, 32])
        if ax_id == "adc_bits":
            axs[-1].set_xticks([4, 6, 8, 10])
        if ax_id == "snr_db" and "R" in run.cmp["sweeps"][ax_id]:
            axs[0].text(0.98, 0.04, "R: per-stage SNR\nV, E, C: input AWGN", transform=axs[0].transAxes, fontsize=6.2,
                        color=ink["sec"], ha="right", va="bottom")
    # legend block
    h = [Line2D([], [], color=col[t], ls=LINESTYLES[t], marker=MARKERS[t], ms=5, lw=1.6) for t in run.tiers]
    legend_ax.legend(h, [tier_label(t, status[t]) for t in run.tiers], loc="upper left", frameon=False, fontsize=7.4,
                     title="tier", title_fontsize=7.8, bbox_to_anchor=(0.0, 1.0))
    note = ["Dotted grey = V accuracy on the clean test set.",
            "Mean KL is KL(V on clean input || tier at the swept value), symlog axis (V and E are ~0).",
            "Each curve is drawn only for tiers that implement that axis."]
    if run.cascade:
        cm = comp_modes(run)
        extra = ""
        if cm and trim_matters(run):
            extra = (f" Line loss is not the whole gap to V: C's default per-layer gain trim over-drives layer 3 (C scores {cm['layer']:.2f} with it, "
                     f"{cm['none']:.2f} with no compensation, {cm['tap']:.2f} with ideal per-tap repeaters at this loss; docs/circuit_extraction.md).")
        note.insert(0, "Amber dotted line on the loss axis = the headline run: with --cascade, C and R run at 32 dB/us (RG-58, NO repeaters; "
                       "the value equals the one put into the netlist), an operating point that is not buildable." + extra
                       + " At 0 dB/us both tiers are within ~0.2 points of V.")
    ytxt = 0.52
    for s in note:
        ls = wrap(s, 46)
        legend_ax.text(0.0, ytxt, "\n".join(ls), transform=legend_ax.transAxes, fontsize=6.8, color=ink["sec"], va="top", linespacing=1.3)
        ytxt -= 0.062 * len(ls) + 0.03
    fig.suptitle(f"Metric vs sweep axis (run {run.cmp['run_id']})", x=0.01, y=0.998, ha="left", fontsize=9.5, fontweight="bold")
    return fig


# ======================================================================================================== fig 6/7
def _div_cmap(theme: str):
    k = INK[theme]
    return LinearSegmentedColormap.from_list("div", [k["neg"], k["mid"], k["pos"]])


def build_fig6(run: Run, theme: str):
    """Row-normalised confusion matrix of V, and the difference (tier - V) for E, C and R. Shared diverging scale."""
    ink = INK[theme]
    y, cls = run.labels, run.classes
    n = len(cls)
    others = [t for t in run.tiers if t != "V"]
    pv_all = run.res["V"].arrays["pred"]
    cmV = conf_matrix(y, pv_all, n)
    rnV = cmV / np.maximum(cmV.sum(1, keepdims=True), 1)
    diffs, cm, nex = {}, {"V": cmV}, {}
    for t in others:
        pt, pv, idx = run.paired_preds(t, "V")          # tier and V on the SAME examples (C and R may run on a subset)
        cm[t] = conf_matrix(y[idx], pt, n)
        cmv = conf_matrix(y[idx], pv, n)
        diffs[t] = cm[t] / np.maximum(cm[t].sum(1, keepdims=True), 1) - cmv / np.maximum(cmv.sum(1, keepdims=True), 1)
        nex[t] = len(idx)
    rn = {"V": rnV}
    lim = max(0.05, max(np.abs(d).max() for d in diffs.values()))
    seq = LinearSegmentedColormap.from_list("seq", SEQ[theme])
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.6), squeeze=False)
    axl = axs.ravel()
    short = [c.replace("QAM", "QAM") for c in cls]
    for i, a in enumerate(axl):
        if i == 0:
            m, cmap, vmin, vmax, title = rn["V"], seq, 0, 1, f"V (reference): recall per true class, accuracy {np.trace(cm['V']) / cm['V'].sum():.3f}"
        else:
            if i - 1 >= len(others):
                a.axis("off"); continue
            t = others[i - 1]
            m, cmap, vmin, vmax = diffs[t], _div_cmap(theme), -lim, lim
            pt_, pv_, idx_ = run.paired_preds(t, "V")
            title = (f"{t} - V: accuracy {np.mean(pt_ == y[idx_]):.3f} (V {np.mean(pv_ == y[idx_]):.3f})"
                     + (f", {len(idx_)} paired ex." if len(idx_) != len(y) else ""))
        im = a.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        if i > 0 and np.abs(m).max() < 0.03:
            a.text(0.5, 0.5, f"no visible difference\nmax |change| = {np.abs(m).max():.3f}", transform=a.transAxes,
                   ha="center", va="center", fontsize=7.6, color=ink["sec"])
        a.set_xticks(range(n)); a.set_xticklabels(short, rotation=60, ha="right", fontsize=6.8)
        a.set_yticks(range(n)); a.set_yticklabels(short, fontsize=6.8)
        a.grid(False)
        a.set_title(title, fontsize=7.8)
        a.set_xlabel("predicted", fontsize=7.4)
        if i % 2 == 0:
            a.set_ylabel("true", fontsize=7.4)
        thr = 0.02 if i == 0 else 0.04
        for r_ in range(n):
            for c_ in range(n):
                v = m[r_, c_]
                if abs(v) >= thr or (i == 0 and r_ == c_):
                    rgba = cmap((v - vmin) / (vmax - vmin))
                    hexc = "#%02x%02x%02x" % tuple(int(255 * q) for q in rgba[:3])
                    a.text(c_, r_, f"{v:+.2f}" if i else f"{v:.2f}", ha="center", va="center", fontsize=5.6 if i else 6.0,
                           color="#0b0b0b" if lum(hexc) > 0.45 else "#ffffff")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    parts = []
    for t in others:
        d = diffs[t]
        if np.abs(d).max() < 0.03:
            parts.append(f"{t}: no cell differs from V by more than {np.abs(d).max():.3f}.")
            continue
        off = [(d[r_, c_], r_, c_) for r_ in range(n) for c_ in range(n) if r_ != c_]
        top = sorted(off, reverse=True)[:2]
        dg = sorted((d[k, k], k) for k in range(n))[:2]
        parts.append(f"{t}: largest gains off the diagonal " + ", ".join(f"{cls[r_]}->{cls[c_]} ({v:+.2f})" for v, r_, c_ in top if v > 0.04)
                     + "; largest recall losses " + ", ".join(f"{cls[k]} ({v:+.2f})" for v, k in dg if v < -0.04) + ".")
    _foot(fig, "Cells are fractions of the true class (rows sum to 1). Diverging panels: blue = the tier puts MORE of that class there than V, red = LESS "
               "(grey = no change); on the diagonal blue is better, off the diagonal red is better. One shared scale "
               f"(+/-{lim:.2f}) so E ~ V shows up as blank.\n" + "  ".join(parts), theme, width=118, y=0.065)
    return fig


def build_fig7(run: Run, theme: str):
    """Top-1 agreement (fraction of test examples with the same argmax) for every pair of tiers."""
    ink = INK[theme]
    ts = [t for t in run.tiers if "pred" in run.res[t].arrays]
    M = np.array([[np.mean(np.equal(*run.paired_preds(a, b)[:2])) for b in ts] for a in ts])
    nn = {t: len(run.index_of(t)) for t in ts}
    subset = len(set(nn.values())) > 1
    seq = LinearSegmentedColormap.from_list("seq", SEQ[theme])
    fig, ax = plt.subplots(figsize=(4.9, 4.7))
    lo = min(0.5, float(M.min()) - 0.02)
    im = ax.imshow(M, cmap=seq, vmin=lo, vmax=1.0)
    ax.set_xticks(range(len(ts))); ax.set_yticks(range(len(ts)))
    lab = [f"{t}\n{TIER_SHORT[t].split(' ', 1)[1]}" for t in ts]
    ax.set_xticklabels(lab, fontsize=7.4); ax.set_yticklabels(lab, fontsize=7.4)
    ax.grid(False)
    ax.xaxis.tick_top()
    for i in range(len(ts)):
        for j in range(len(ts)):
            rgba = seq((M[i, j] - lo) / (1 - lo))
            hexc = "#%02x%02x%02x" % tuple(int(255 * q) for q in rgba[:3])
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold" if i == j else "normal",
                    color="#0b0b0b" if lum(hexc) > 0.45 else "#ffffff")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    cb.set_label("top-1 agreement", fontsize=7.4)
    n = max(nn.values())
    fig.suptitle(f"Prediction agreement between tiers ({n} test examples" + (", pairs on their common examples)" if subset else ")"), x=0.02, ha="left", fontsize=9.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.97))
    if run.cascade and set("VECR") <= set(ts):
        i = {t: k for k, t in enumerate(ts)}
        txt = (f"E reproduces V almost exactly ({M[i['V'], i['E']]:.3f}): 8-bit fixed point is nearly free. C and R agree with V on only "
               f"{M[i['V'], i['C']]:.2f} and {M[i['V'], i['R']]:.2f} of the examples because, in this run, both sit at 32 dB/us without repeaters (not buildable)" + (" and C's default per-layer gain trim over-drives layer 3" if trim_matters(run) else "") + "; "
               f"they agree with each other {M[i['C'], i['R']]:.2f}" + (", more than with V, since the same operating point hurts both." if M[i['C'], i['R']] > max(M[i['V'], i['C']], M[i['V'], i['R']]) else "."))
    else:
        txt = "Agreement of argmax predictions, all pairs."
    if subset:
        txt += " Tiers " + ", ".join(f"{t} ({nn[t]})" for t in ts) + " ran on different numbers of examples; each cell uses the examples both ran on."
    _foot(fig, txt, theme, width=96, y=0.07)
    return fig


# ======================================================================================================== fig 8
def audit_commentary(com: dict, status: dict[str, str], cmp: dict | None = None) -> list[str]:
    """Independent checks of commentary statements against compare.json (not part of the verifier; same rules as the audit in
    scripts/make_results_md.py): (1) a tier with status ok called a fallback/stub/placeholder, (2) an ok tier called "not the real
    tool/simulation", (3) the same metric=value stated twice in one sentence, (4) an N-bit quantity that no tier's fidelity text mentions
    (only when ``cmp`` = the run's compare.json dict is given). It cannot judge causal claims."""
    import re
    out = []
    texts = [("summary", com.get("summary", ""))] + [(f"caveats[{i}]", c) for i, c in enumerate(com.get("caveats", []))] + \
            [(f"findings[{i}]", f"{f.get('observation', '')} {f.get('likely_cause', '')}") for i, f in enumerate(com.get("findings", []))] + \
            [(f"questions[{i}]", q) for i, q in enumerate(com.get("questions", []))]
    for where, t in texts:
        for tier, st in status.items():
            if st == "ok" and re.search(rf"\b{tier}\b[^.]{{0,40}}\b(fallback|stub|placeholder)\b", t, re.I):
                out.append(f"{where}: says tier {tier} is a fallback/stub, but compare.json has {tier} status=ok")
            if st == "ok" and re.search(rf"\b{tier}\b[^.]*not the real (tool|simulation)", t, re.I):
                out.append(f"{where}: calls tier {tier} \"not the real tool\", but compare.json has {tier} status=ok (the real tool ran)")
    for i, f in enumerate(com.get("findings", [])):
        m = re.search(r"(\w+=-?[\d.]+) \([^)]*\) and \1 ", f.get("observation", "") + " ")
        if m:
            out.append(f"findings[{i}]: states {m.group(1)} twice in one sentence")
    if cmp:
        cardbits = set(re.findall(r"(\d+)[- ]bit", " ".join(" ".join(v.get("fidelity", {}).get("models", [])) for v in cmp.get("tier_info", {}).values())))
        alltxt = " ".join(t for _, t in texts)
        for b in sorted(set(re.findall(r"(\d+)-bit", alltxt))):
            if b not in cardbits:
                out.append(f"text mentions a {b}-bit quantity; no tier's fidelity text mentions {b}-bit")
    return out


def build_fig8(run: Run, theme: str, commentary: dict | str | Path | None = None):
    """LLM commentary panel: summary, findings with citation chips (metric_id, tier chips in the tier colour, verification
    chip), unverified findings dimmed and labelled; template commentary is labelled TEMPLATE, not model output."""
    ink, col = INK[theme], TIER_COL[theme]
    if commentary is None:
        commentary = run.dir / "llm" / "commentary.json"
    com = commentary if isinstance(commentary, dict) else json.loads(Path(commentary).read_text())
    is_tmpl = com.get("source") == "template"
    verified_all = com.get("numbers_verified")
    unver = com.get("unverified", [])
    W, FS, LH = 7.0, 7.6, 0.158           # figure width, text size, line height (in)
    wrap_w = 100

    blocks = []   # (kind, payload)
    blocks.append(("head", None))
    blocks.append(("h2", "Summary"))
    blocks.append(("text", com.get("summary", ""), com.get("summary_verified", True) is not False))
    blocks.append(("h2", f"Findings ({len(com.get('findings', []))})"))
    for f in com.get("findings", []):
        blocks.append(("finding", f))
    blocks.append(("h2", "Caveats"))
    cv = com.get("caveats_verified", [True] * len(com.get("caveats", [])))
    for c, ok in zip(com.get("caveats", []), cv):
        blocks.append(("bullet", c, ok is not False))
    if com.get("questions"):
        blocks.append(("h2", "Questions"))
        qv = com.get("questions_verified", [True] * len(com["questions"]))
        for q, ok in zip(com["questions"], qv):
            blocks.append(("bullet", q, ok is not False))
    aud = audit_commentary(com, run.status, run.cmp) if not is_tmpl else []
    if not is_tmpl:
        blocks.append(("h2", "Independent audit against compare.json (not part of the verifier)"))
        if aud:
            for a_ in aud:
                blocks.append(("bullet", a_, True, "warn"))
        else:
            blocks.append(("bullet", "nothing found by the status-word, duplicate-value and bit-width checks (causal claims are not checked)", True))
    blocks.append(("foot", None))

    def lines(txt, w=wrap_w):
        return wrap(txt, w)

    # height pass
    H = 0.1
    for b in blocks:
        k = b[0]
        if k == "head": H += 0.72 if is_tmpl else 1.0
        elif k == "h2": H += 0.30
        elif k == "text": H += LH * len(lines(b[1])) + 0.1
        elif k == "bullet": H += LH * len(lines(b[1], wrap_w - 4)) + 0.07
        elif k == "finding":
            f = b[1]
            H += 0.36 + LH * (len(lines("Observation: " + f["observation"], wrap_w - 4)) + len(lines("Likely cause: " + f["likely_cause"], wrap_w - 4))) + 0.18
        elif k == "foot": H += 0.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    r = fig.canvas.get_renderer()
    y = 0.1

    def chip(x, yc, text, fc, tc, ec=None, fs=6.8, alpha=1.0, bold=False):
        t = ax.text(x + 0.07, yc, text, fontsize=fs, color=tc, va="center", ha="left", alpha=alpha,
                    fontweight="bold" if bold else "normal", zorder=6)
        bb = t.get_window_extent(r)
        wid = bb.width / fig.dpi + 0.14
        ax.add_patch(FancyBboxPatch((x, yc - 0.1), wid, 0.2, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec or "none",
                                    lw=0.8, alpha=alpha, zorder=5))
        return x + wid + 0.07

    for b in blocks:
        k = b[0]
        if k == "head":
            src = "TEMPLATE (fixed text, not model output)" if is_tmpl else f"MODEL OUTPUT: {com.get('backend', '?')}"
            ax.text(0.1, y + 0.13, "LLM commentary on the comparison card", fontsize=10, fontweight="bold", color=ink["text"], va="center")
            x = chip(0.1, y + 0.47, src, ink["warn"] if is_tmpl else ink["text"], ink["surface"] if not is_tmpl else "#ffffff", bold=True)
            if not is_tmpl:
                llm = com.get("llm", {})
                x = chip(x, y + 0.47, llm.get("model", ""), "none", ink["sec"], ec=ink["axis"])
                x = chip(x, y + 0.47, "grammar-constrained JSON", "none", ink["sec"], ec=ink["axis"])
                ok = verified_all
                x = chip(0.1, y + 0.76, "numbers_verified: true" if ok else "numbers_verified: FALSE (1+ number UNVERIFIED)", "none", ink["text"] if ok else ink["warn"],
                         ec=ink["axis"] if ok else ink["warn"], bold=not ok)
                if "ids_verified" in com:
                    idok = com.get("ids_verified") is not False
                    x = chip(x, y + 0.76, "ids_verified: true" if idok else "ids_verified: FALSE (a tier has no finding)", "none",
                             ink["text"] if idok else ink["warn"], ec=ink["axis"] if idok else ink["warn"], bold=not idok)
            y += 0.72 if is_tmpl else 1.0
        elif k == "h2":
            ax.text(0.1, y + 0.17, b[1].upper(), fontsize=7.2, fontweight="bold", color=ink["sec"], va="center")
            ax.plot([0.1, W - 0.1], [y + 0.28, y + 0.28], color=ink["grid"], lw=0.8)
            y += 0.30
        elif k == "text":
            ls = lines(b[1]); alpha = 1.0 if b[2] else 0.45
            ax.text(0.14, y + 0.03, "\n".join(ls), fontsize=FS, color=ink["text"], va="top", linespacing=1.32, alpha=alpha)
            y += LH * len(ls) + 0.1
        elif k == "bullet":
            ls = lines(b[1], wrap_w - 4); alpha = 1.0 if b[2] else 0.45
            bcol = ink["warn"] if len(b) > 3 and b[3] == "warn" else ink["text"]
            ax.text(0.14, y + 0.03, "-", fontsize=FS, color=ink["sec"], va="top", alpha=alpha)
            ax.text(0.3, y + 0.03, "\n".join(ls), fontsize=FS, color=bcol, va="top", linespacing=1.32, alpha=alpha)
            y += LH * len(ls) + 0.07
        elif k == "finding":
            f = b[1]
            ver = f.get("verified", True) is not False
            alpha = 1.0 if ver else 0.5
            ob = lines("Observation: " + f["observation"], wrap_w - 4)
            lc = lines("Likely cause: " + f["likely_cause"], wrap_w - 4)
            hh = 0.36 + LH * (len(ob) + len(lc)) + 0.1
            ax.add_patch(FancyBboxPatch((0.1, y), W - 0.2, hh, boxstyle="round,pad=0,rounding_size=0.08", fc="none",
                                        ec=ink["axis"] if ver else ink["warn"], lw=0.9, ls="-" if ver else (0, (3, 2)), alpha=alpha))
            x = chip(0.2, y + 0.19, f["metric_id"], ink["grid"], ink["text"], alpha=alpha)
            for t in f["tiers"]:
                tc = col.get(t, ink["muted"])
                x = chip(x, y + 0.19, f"tier {t}", tc, "#0b0b0b" if lum(tc) > 0.5 and theme == "light" or t == "R" else "#ffffff", alpha=alpha, bold=True)
            if not ver:
                x = chip(x + 0.05, y + 0.19, "UNVERIFIED: a number is not on the card", "none", ink["warn"], ec=ink["warn"], bold=True)
            ax.text(0.24, y + 0.33, "\n".join(ob), fontsize=FS, color=ink["text"], va="top", linespacing=1.32, alpha=alpha)
            ax.text(0.24, y + 0.33 + LH * len(ob) + 0.03, "\n".join(lc), fontsize=FS, color=ink["text"], va="top", linespacing=1.32, alpha=alpha)
            y += hh + 0.08
        elif k == "foot":
            msg = ("The text above is a FIXED TEMPLATE: no language model produced it." if is_tmpl else
                   "A language model commented on a <=1.5k-token text card; it ran no simulation. The verifier checks that numbers and metric/tier ids "
                   "match the card; it does NOT check whether the explanation is right. Dimmed = flagged unverified.")
            if unver and not is_tmpl:
                msg += " Guard flags: " + "; ".join((f"tier {u['token']} has no finding" if u.get("kind") == "missing_tier" else f"{u['where']}: '{u['token']}'") for u in unver) + "."
            ax.text(0.1, y + 0.08, "\n".join(wrap(msg, 118)), fontsize=6.8, color=ink["sec"], va="top", linespacing=1.35)
    return fig


# ======================================================================================================== registry
BUILDERS = {"fig1_network": build_fig1, "fig2_traces": build_fig2, "fig3_channel": build_fig3, "fig4_outputs": build_fig4,
            "fig5_sweeps": build_fig5, "fig6_confusion": build_fig6, "fig7_agreement": build_fig7, "fig8_llm": build_fig8}
APPLIES = {"fig1_network": ("modrec", "narma10"), "fig2_traces": ("modrec",), "fig3_channel": ("modrec",),
           "fig4_outputs": ("modrec", "narma10"), "fig5_sweeps": ("modrec", "narma10"), "fig6_confusion": ("modrec",),
           "fig7_agreement": ("modrec",), "fig8_llm": ("modrec", "narma10")}


def build(name: str, run: Run | str | Path, theme: str = "light", **kw):
    """Return the matplotlib Figure (not saved, not closed)."""
    run = run if isinstance(run, Run) else Run(run)
    with ctx(theme):
        return BUILDERS[name](run, theme, **kw)


def render(name: str, run_dir: str | Path, theme: str = "light", outdir: str | Path | None = None, opaque: bool = False, **kw) -> list[Path]:
    """Build and save `<outdir or run/figures>/<name>_<theme>.{png,svg}`; returns the paths. ``opaque=True`` draws the page colour
    (white for light, the dark surface for dark) into the files instead of leaving them transparent (used for the docs export)."""
    run = run_dir if isinstance(run_dir, Run) else Run(run_dir)
    with ctx(theme):
        fig = BUILDERS[name](run, theme, **kw)
        return save_fig(fig, Path(outdir) if outdir else run.dir / "figures", name, theme, opaque=opaque)


def make_all(run_dir: str | Path, themes=("light", "dark"), outdir: str | Path | None = None,
             names: list[str] | None = None, opaque: bool = False) -> list[Path]:
    """Render every applicable figure in both themes. Never raises: failures are collected in ``make_all.errors``."""
    make_all.errors = {}
    out: list[Path] = []
    try:
        run = Run(run_dir)
    except Exception as e:  # noqa: BLE001
        make_all.errors["run"] = f"{type(e).__name__}: {e}"
        return out
    for name in names or list(BUILDERS):
        if run.task not in APPLIES[name]:
            continue
        for theme in themes:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out += render(name, run, theme, outdir=outdir, opaque=opaque)
            except Exception as e:  # noqa: BLE001  a missing optional input must not lose the other figures
                make_all.errors[f"{name}/{theme}"] = f"{type(e).__name__}: {e}"
                plt.close("all")
    return out


make_all.errors = {}  # type: ignore[attr-defined]

# backwards-compatible names used by earlier code / notebooks
def fig4_outputs(run_dir, theme="light", **kw):
    return render("fig4_outputs", run_dir, theme, **kw)


def fig5_sweeps(run_dir, theme="light", **kw):
    return render("fig5_sweeps", run_dir, theme, **kw)
