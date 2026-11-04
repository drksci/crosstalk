"""Regenerates the notebook sources (unexecuted). Run:  .venv/bin/python notebooks/_build.py
Then execute (saves real outputs):
    for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=dlnn --ExecutePreprocessor.timeout=900 $n; done
Heavy results are read from an existing run (env DLNN_RUN, default out/modrec-w32-full); the 10-minute pipeline is NOT rerun.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "dlnn", "display_name": "dlnn (.venv)", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c.strip("\n")) if k == "md" else nbf.v4.new_code_cell(c.strip("\n")) for k, c in cells]
    nbf.write(n, HERE / f"{name}.ipynb")


SETUP = '''import json, os, time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from dlnn.viz import nb, figures
from dlnn.viz.common import TIER_COL, MARKERS, LINESTYLES, TIER_SHORT, INK, conf_matrix

nb.setup()
run = nb.load_run()          # env DLNN_RUN, default out/modrec-w32-full (read-only; the pipeline is NOT rerun here)
cmp = run.cmp
print(f"run {run.dir.name} | model {os.path.basename(cmp['run_meta']['model_file'])} | cascade={run.cascade} | "
      f"tiers {cmp['tiers']} | n_test {cmp['tier_info']['V']['n_examples']}")'''

# ==================================================================================================== 00
nb([
("md", r"""
# 00 Overview: a delay-line neural network, simulated at four fidelities

**One trained network, four levels of physical realism.** The network is a *ternary tapped-delay-line convolution net* (TDL-Net): every layer is a bank of coax delay lines with taps, a passive combiner, and a saturating amplifier. The weights are only ever **-1, 0, +1**, so a weight is literally *a tap polarity*. The task is 8-class modulation recognition (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK) on synthetic I/Q at 0-20 dB SNR.

For layer $l$ with input lines $i$, output lines $o$, $K$ taps, dilation $d_l$ and symbol period $T_s=10\,\mathrm{ns}$:

$$u_o[n]=\sum_{i}\sum_{k=0}^{K-1} W_{o,i,k}\;x_i[n-k\,d_l],\qquad W_{o,i,k}\in\{-1,0,+1\}$$

$$y_o[n]=g_l\,\tanh\!\Big(\frac{u_o[n]+b_o}{V_{sat}}\Big),\qquad \text{feature}_o=\frac1T\sum_n y_o[n]\ \ (\text{RC average pool}),\qquad \text{logits}=W_r\,\text{feature}+b_r$$

| math | physical element | note |
|---|---|---|
| delay $k\,d_l\,T_s$ | coax of length $k\,d_l\,T_s\,v_p$ (RG-58: $v_p$ = 19.8 cm/ns, so **1.98 m per 10 ns**) | dilation = longer line |
| $W=+1$ / $0$ / $-1$ | coupler tap into the combiner / no tap / tap through a 180 deg hybrid | BitNet-b1.58 ternary maps 1:1 onto tap polarity |
| $\sum_{i,k}$ | N-way Wilkinson combiner | passive, $-10\log_{10}N$ dB |
| $\tanh$ | BJT differential pair, $I=I_{EE}\tanh(v/2V_T)$ | natively tanh |
| $g_l,\ V_{sat},\ b_o$ | amplifier gain, bias, DC offset | compensates loss, adds noise |
| global average pool | RC integrator, $\tau\approx N\,T_s$ | passive |
| readout + softmax | ADC + software | the only honestly digital part |

## The four tiers

| tier | tool | what it computes | what it does **not** |
|---|---|---|---|
| **V** virtual | numpy float32 | ideal unit delays, exact ternary sums, exact tanh (the math; the reference for every delta) | every physical effect |
| **E** emulated | numpy bit-true fixed point + Icarus Verilog RTL | 8-bit ADC, int2 weights, int32 accumulate, 256-entry tanh LUT, integer pool | analog effects; RTL is run on a subset only |
| **C** circuit | ngspice 42 | SPICE on a layer-1 unit cell: per-tap `.ac`, diff-pair `.dc`, `.tran`; full-set numbers from the **circuit-calibrated behavioural model** | layers 2-3 and the 4000-example set are *not* SPICE |
| **R** RF | scikit-rf 2.2 | channel-level lossy dispersive line, coupler, hybrid, Wilkinson, mismatch, crosstalk, noise, amplifier | EM, layout, transistor level |

The float32 baseline (same topology, unconstrained taps) is a **reference, not a fifth tier**.
"""),
("code", SETUP),
("md", "## Headline results (read live from the run)"),
("code", '''nb.md(f"**Headline run `{run.dir.name}`** (task `{cmp['task']}`, width-32 ternary TDL-Net, all four tiers on all {cmp['tier_info']['V']['n_examples']} test examples)")
nb.results_table(cmp)
rt = cmp.get("run_meta", {})
print("pipeline stage timing (s):", {k: round(v) for k, v in cmp.get("timing", {}).get("stages_s", {}).items()})'''),
("code", '''# the last pipeline run (out/LATEST), which is not necessarily the modrec headline run
latest_name = (nb.ROOT / "out" / "LATEST").read_text().strip() if (nb.ROOT / "out" / "LATEST").exists() else None
print("out/LATEST ->", latest_name)
if latest_name and (nb.ROOT / "out" / latest_name / "compare.json").exists():
    lc = json.loads((nb.ROOT / "out" / latest_name / "compare.json").read_text())
    if lc["run_id"] != cmp["run_id"]:
        nb.md(f"**`out/LATEST` = `{latest_name}`** (task `{lc['task']}`). It is a different run from the headline above:")
        if lc["task"] == "narma10":
            mm = lc["metric_matrix"]
            rows = ["| tier | status | NMSE | NMSE vs V |", "|---|---|---|---|"]
            for t in lc["tiers"]:
                rows.append(f"| {t} | {lc['tier_info'][t]['status']} | {nb.fmt(mm['nmse'].get(t))} | {nb.fmt(mm['nmse_vs_virtual'].get(t))} |")
            nb.md("\\n".join(rows))
        else:
            nb.results_table(lc)'''),
("code", '''# the plan-spec width-8 net is a documented SECONDARY model: it misses its accuracy target (drops 8PSK entirely)
rows = ["| model | ternary weights | V acc | E acc | C acc | R acc | float32 ref | C/R op. point |", "|---|---|---|---|---|---|---|---|"]
for name in ("modrec-w32-full", "modrec-w8-full"):
    p = nb.ROOT / "out" / name / "compare.json"
    if not p.exists():
        continue
    c = json.loads(p.read_text()); a = c["metric_matrix"]["accuracy"]
    nw = "16,896 conv + 256 readout (width 32, headline)" if "w32" in name else "1,152 conv + 64 readout = 1,216, plan spec (secondary)"
    rows.append(f"| {name} | {nw} | " + " | ".join(nb.fmt(a.get(t), 3) for t in "VECR") +
                f" | {nb.fmt(c['reference']['accuracy'], 3)} | 32 dB/us, no repeaters (C: default gain trim matters, see below) |")
nb.md("\\n".join(rows))
nb.md("*Width 8 reaches only 0.735 mixed-SNR (0.763 at >= 10 dB) and never predicts 8PSK; width 32 is the headline model. Both are reported; neither is hidden (notebook 01).*")'''),
("md", "## The network, drawn from the trained weights (figure 1)\nEach horizontal bar is one delay line; the markers are the actual ternary weights of one output unit; lengths are metres of RG-58."),
("code", "nb.show('fig1_network', run)"),
("md", "## What the tiers agree on (figure 7)"),
("code", "nb.show('fig7_agreement', run)"),
("code", '''nb.md(nb.honesty_box(run))'''),
("md", """
## Notebook map

| notebook | what it does | live compute |
|---|---|---|
| 01 task and model | the 8 signal classes, training report, width 8 vs 32, ternary weights | seconds |
| 02 virtual | the ideal forward pass step by step | seconds |
| 03 emulated | fixed-point spec, ADC sweep, RTL vs numpy bit-exact check | ~1 min (iverilog) |
| 04 circuit | the SPICE netlists, live ngspice `.dc` / `.ac` / `.tran`, calibrated vs SPICE | ~1 min (ngspice) |
| 05 rf | scikit-rf line networks, impulse responses, live parameter sweeps | ~2 min |
| 06 compare and LLM | the side-by-side matrix, figures 4-8, the card and the verified commentary | seconds |

Everything heavy (the 10-minute pipeline) is read from `out/<run>/`; change the run with `DLNN_RUN=modrec-w8-full`.
"""),
], "00_overview")

# ==================================================================================================== 01
nb([
("md", r"""
# 01 Task and model

**Task.** 8-class modulation recognition on synthetic complex baseband I/Q: 128 samples at 8 samples/symbol, random carrier phase, carrier-frequency offset $\pm1\,\%$ of $f_s$, random timing offset, AWGN at SNR $\in\{0,5,10,15,20\}$ dB. 16 000 train / 4 000 test examples, generated in-repo and deterministic (sha256 below).

**Model.** `TDL-Net`: 2 lines (I, Q) $\to$ L1 ($K=8$, $d=1$) $\to$ L2 ($d=4$) $\to$ L3 ($d=16$) $\to$ RC average pool $\to$ ternary linear $\to$ softmax. Receptive field $1+7\,(1+4+16)=148\ge128$. Trained offline with ternary quantisation-aware training (BitNet b1.58 rule $W_t=\mathrm{clip}(\mathrm{round}(W/\gamma),-1,1)$, $\gamma=\mathrm{mean}|W|$, straight-through estimator).
"""),
("code", SETUP + '''
from dlnn.data import modrec
ds = modrec.dataset()
print("dataset sha256 regenerated:", ds["sha256"][:16], "| recorded in the run:", cmp["dataset_sha256"][:16], "| identical:", ds["sha256"] == cmp["dataset_sha256"])
print("train", ds["x_train"].shape, "test", ds["x_test"].shape, "classes", modrec.CLASSES)'''),
("md", "## The eight signal classes (20 dB SNR examples)"),
("code", '''g = modrec.generate(80, snrs=(20,))            # 10 per class, all at 20 dB
x, y = g["x"], g["y"]
cls = list(modrec.CLASSES)
col = TIER_COL["light"]["V"]
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey=True)
for c, ax in enumerate(axs.ravel()):
    e = x[y == c][0]
    ax.plot(e[0, :96], color=col, lw=1.2, label="I"); ax.plot(e[1, :96], color=INK["light"]["muted"], lw=1.0, ls="--", label="Q")
    ax.set_title(cls[c], fontsize=9)
axs[0, 0].legend(fontsize=7); fig.supxlabel("sample n (8 per symbol)"); fig.supylabel("amplitude")
fig.suptitle("Waveforms: first 96 samples of one example per class", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); nb.show_fig(fig)'''),
("code", '''fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
for c, ax in enumerate(axs.ravel()):
    e = x[y == c][:6]
    ax.plot(e[:, 0, ::2].ravel(), e[:, 1, ::2].ravel(), ".", ms=2.0, color=col, alpha=0.6)
    ax.set_title(cls[c], fontsize=9); ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
fig.suptitle("Constellations (every 2nd sample = 4 per symbol, transitions included; 6 examples per class): carrier phase and CFO rotate and smear them", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
fig.tight_layout(); nb.show_fig(fig)'''),
("code", '''fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey=True)
f = np.fft.fftshift(np.fft.fftfreq(128))
for c, ax in enumerate(axs.ravel()):
    z = x[y == c][:, 0] + 1j * x[y == c][:, 1]
    P = np.mean(np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(128), axis=1), axes=1)) ** 2, axis=0)
    ax.plot(f, 10 * np.log10(P / P.max()), color=col, lw=1.2); ax.set_title(cls[c], fontsize=9); ax.set_ylim(-45, 3)
fig.supxlabel("normalised frequency (cycles/sample)"); fig.supylabel("power (dB rel. peak)")
fig.suptitle("Average spectrum per class (10 examples at 20 dB)", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); nb.show_fig(fig)'''),
("md", "## Training report: float32 vs ternary, and width 8 (plan spec) vs width 32 (headline)"),
("code", '''rep = {w: json.load(open(nb.ROOT / "models" / f)) for w, f in ((8, "train_report.json"), (32, "train_report_w32.json"))}
rows = ["| width | variant | weights | accuracy (mixed SNR) | accuracy >= 10 dB | train time (s) | note |", "|---|---|---|---|---|---|---|"]
for w in (8, 32):
    r = rep[w]
    rows.append(f"| {w} | float32 | unconstrained | {r['float']['acc']:.4f} | {r['float']['acc_ge10dB']:.4f} | {r['float']['seconds']:.0f} | reference only |")
    note = "plan spec; **misses the 85 % >= 10 dB target, 8PSK = 0 %**" if w == 8 else "**headline model of the demo**"
    rows.append(f"| {w} | ternary | {r['ternary']['n_ternary_weights']} taps | {r['ternary']['acc']:.4f} | {r['ternary']['acc_ge10dB']:.4f} | {r['ternary']['seconds']:.0f} | {note} |")
nb.md("\\n".join(rows))
r = rep[32]["recipe"]
print("recipe (both widths):", {k: r[k] for k in ("optimizer", "lr", "schedule", "batch", "epochs", "iq_rotation_aug", "restarts", "soft_frac")})
print("Not the plan recipe (lr 3e-3, 20 epochs underfits badly: float 62 %, ternary 45 %); see docs/build-log-M1.md for everything that was tried.")'''),
("code", '''# per-SNR and per-class accuracy of the four trained models (test set, 4000 examples)
SER = [("w8 float", rep[8]["float"], "#e87ba4", "//"), ("w8 ternary", rep[8]["ternary"], "#e87ba4", None),
       ("w32 float", rep[32]["float"], "#6250d6", "//"), ("w32 ternary", rep[32]["ternary"], "#6250d6", None)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.7]})
ax = axs[0]
for name, r_, c_, h in SER:
    s = sorted(r_["acc_per_snr"], key=int)
    ax.plot([int(k) for k in s], [r_["acc_per_snr"][k] for k in s], marker="o" if h is None else "s", color=c_, ls="-" if h is None else "--", label=name)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.set_title("accuracy vs SNR")
ax = axs[1]; w = 0.2
for j, (name, r_, c_, h) in enumerate(SER):
    ax.bar(np.arange(8) + (j - 1.5) * w, [r_["acc_per_class"][c] for c in cls], w * 0.92, color=c_, hatch=h, edgecolor=c_, alpha=1.0 if h is None else 0.4)
ax.set_xticks(range(8)); ax.set_xticklabels(cls, rotation=35, ha="right", fontsize=7.5); ax.set_ylabel("recall"); ax.set_title("recall per class")
ax.annotate("w8 ternary: 8PSK = 0", (3 - 0.5 * w, 0.02), xytext=(3.4, 0.35), fontsize=7.5, arrowprops=dict(arrowstyle="->", color=INK["light"]["sec"]))
fig.tight_layout(); nb.show_fig(fig)
nb.md("*Solid = ternary, hatched/light = float32; magenta = width 8, violet = width 32 (these are model variants, not the tier colours of the later figures). Width 32 recovers 8PSK to 0.63 recall and 16QAM to 0.84; QPSK stays the hard class (0.47).*")'''),
("code", '''# where the 8PSK collapse goes: confusion matrices of the two ternary models from the stored runs (V tier, test set)
def vconf(name):
    d = nb.ROOT / "out" / name
    if not (d / "tiers" / "V.npz").exists():
        return None
    return conf_matrix(np.load(d / "labels.npy"), np.load(d / "tiers" / "V.npz")["pred"], 8)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, name, ttl in zip(axs, ("modrec-w8-full", "modrec-w32-full"), ("width 8 ternary (V)", "width 32 ternary (V)")):
    cm = vconf(name)
    if cm is None:
        ax.axis("off"); ax.text(0.5, 0.5, f"{name} not present", ha="center"); continue
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1); ax.grid(False); ax.set_title(ttl, fontsize=9)
    ax.set_xticks(range(8)); ax.set_xticklabels(cls, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(8)); ax.set_yticklabels(cls, fontsize=7)
    for i in range(8):
        for j in range(8):
            if rn[i, j] >= 0.1:
                ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if rn[i, j] > 0.55 else "black")
fig.supxlabel("predicted", fontsize=8); fig.supylabel("true", fontsize=8); fig.tight_layout(); nb.show_fig(fig)'''),
("md", "## Weights: ternary tap polarity vs the float baseline"),
("code", '''from dlnn.model.export import load_float_model
from dlnn.model.npz import load_model
mt, mf = load_model(nb.ROOT / "models" / "tdlnet_modrec_w32_ternary.npz"), load_float_model(nb.ROOT / "models" / "tdlnet_modrec_w32_float.npz")
m8 = load_model(nb.ROOT / "models" / "tdlnet_modrec_ternary.npz")
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex="row")
for l in range(3):
    W = mt.layers[l].W
    cnt = [(W == -1).mean(), (W == 0).mean(), (W == 1).mean()]
    axs[0, l].bar(["-1", "0", "+1"], cnt, color=["#898781", "#c3c2b7", "#898781"], width=0.6)
    for xx, v in zip(range(3), cnt):
        axs[0, l].text(xx, v + 0.01, f"{100 * v:.0f} %", ha="center", fontsize=7.5)
    axs[0, l].set_ylim(0, 0.75); axs[0, l].set_title(f"L{l + 1} ternary taps (d={mt.layers[l].dilation})", fontsize=8.5)
    axs[1, l].hist(mf.layers[l].W.ravel(), bins=60, color="#6250d6", alpha=0.8)
    axs[1, l].set_title(f"L{l + 1} float baseline taps", fontsize=8.5); axs[1, l].set_xlabel("weight value")
axs[0, 0].set_ylabel("fraction"); axs[1, 0].set_ylabel("count")
fig.tight_layout(); nb.show_fig(fig)
print("width 32: ", mt.n_ternary_weights, "ternary conv taps; nonzero fractions", [round(float((l.W != 0).mean()), 2) for l in mt.layers])
print("width 8 : ", m8.n_ternary_weights, "ternary conv taps; nonzero fractions", [round(float((l.W != 0).mean()), 2) for l in m8.layers])
print("per-layer V_sat (amplifier saturation):", [round(l.vsat, 2) for l in mt.layers], "| per-layer gain g (all ones: no per-line scale):", [float(l.gain.mean()) for l in mt.layers])'''),
("md", """
**Reading this.** A ternary tap has three states; about 37-45 % of them are *no tap at all* (width 32) (a free sparsity), the rest split evenly between the two polarities. The float baseline is the same topology with free real taps: it is better (0.919 vs 0.863 mixed-SNR) and is shown only as the reference the ternary constraint is judged against.

Honest limits: the width-8 net of the plan (1 216 taps) is a capacity-limited model on this task; width 32 (16 896 taps) is what the four-tier comparison uses.
"""),
], "01_task_and_model")

# ==================================================================================================== 02
nb([
("md", r"""
# 02 Virtual tier (V): the ideal forward pass, step by step

Tier V is the math with nothing physical in it: unit delays, exact ternary sums, exact $\tanh$, float32. It reproduces the torch evaluation of the trained model to $2\times10^{-6}$ in the logits (M1 acceptance), and it is the **reference** for every "agreement", "KL" and "delta vs V" number in this project.

**Layer** (causal, zeros before $n=0$):

$$u_o[n]=\sum_{i,k}W_{o,i,k}\,x_i[n-k\,d],\qquad y_o[n]=g_o\tanh\!\Big(\frac{u_o[n]+b_o}{V_{sat}}\Big)$$

**Readout:** $f_o=\frac1T\sum_n y_o^{(3)}[n]$ (the RC pool), $z=W_r f+b_r$, $p=\mathrm{softmax}(z)$.
"""),
("code", SETUP + '''
from dlnn.tiers import virtual
from dlnn.tiers.common import softmax
model = run.model
EX = 6      # test example to walk through
y_true = int(run.labels[EX]); snr = float(run.snr[EX]); cls = run.classes
x = run.x_test[EX:EX + 1]
print(f"example {EX}: true class {cls[y_true]}, SNR {snr:.0f} dB, x shape {x.shape}")'''),
("md", "### Step 1. The input lines"),
("code", '''fig, ax = plt.subplots(figsize=(7.2, 2.2))
ax.plot(x[0, 0], color=TIER_COL["light"]["V"], label="I"); ax.plot(x[0, 1], color=INK["light"]["muted"], ls="--", label="Q")
ax.set_xlabel("sample n (T_s = 10 ns each)"); ax.set_ylabel("input"); ax.legend(ncol=2, fontsize=7); ax.set_title(f"example {EX}: {cls[y_true]} @ {snr:.0f} dB")
fig.tight_layout(); nb.show_fig(fig)'''),
("md", """
### Step 2. One output unit of layer 1, tap by tap
Each non-zero tap contributes $\\pm$ the input line delayed by $k\\,d$ samples (a coax of length $k\\,d\\,T_s\\,v_p$); the combiner adds them to give $u$."""),
("code", '''ly = model.layers[0]
o = int(np.argmax((ly.W != 0).sum(axis=(1, 2))))             # the unit with the most taps, for a busy picture
xin = x[0] * model.input_scale
taps = [(i, k, int(ly.W[o, i, k])) for i in range(2) for k in range(ly.W.shape[2]) if ly.W[o, i, k] != 0]
T = xin.shape[1]; u = np.zeros(T)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
for j, (i, k, w) in enumerate(taps):
    sh = np.zeros(T); sh[k * ly.dilation:] = xin[i, :T - k * ly.dilation]
    u += w * sh
    axs[0].plot(w * sh + 3.2 * j, lw=0.9, color=INK["light"]["sec"])
    axs[0].text(T + 1, 3.2 * j, f"{'+' if w > 0 else '-'}{'IQ'[i]}[n-{k * ly.dilation}]  ({k * ly.dilation * 1.98:.1f} m)", fontsize=6.5, va="center", clip_on=False)
axs[0].set_yticks([]); axs[0].set_xlim(0, T - 1); axs[0].set_title(f"layer 1, output unit {o}: {len(taps)} non-zero taps, each = signed delayed copy of an input line", fontsize=9)
ref = virtual.layer_preact(x * model.input_scale, ly)[0, o]
axs[1].plot(u, color=TIER_COL["light"]["V"], lw=2.2, label="sum of taps (this cell)"); axs[1].plot(ref, "k:", lw=1.2, label="virtual.layer_preact")
axs[1].set_xlabel("sample n"); axs[1].set_ylabel("u[n]"); axs[1].legend(fontsize=7)
fig.tight_layout(); nb.show_fig(fig)
print("max |difference| vs the library implementation:", float(np.abs(u - ref).max()))'''),
("md", "### Step 3. The saturating amplifier (diff pair): $y=g\\tanh((u+b)/V_{sat})$"),
("code", '''b, vs, g = float(ly.bias[o]), float(ly.vsat), float(ly.gain[o])
yo = g * np.tanh((u + b) / vs)
zz = np.linspace(-4 * vs, 4 * vs, 300)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
axs[0].plot(zz, g * np.tanh(zz / vs), color=INK["light"]["sec"]); axs[0].plot(u + b, yo, ".", ms=3, color=TIER_COL["light"]["V"], alpha=0.7)
axs[0].set_xlabel("u + b  (model units)"); axs[0].set_ylabel("y"); axs[0].set_title(f"V_sat = {vs:.2f}, g = {g:.2f}, b = {b:+.2f}", fontsize=9)
axs[1].plot(yo, color=TIER_COL["light"]["V"]); axs[1].set_xlabel("sample n"); axs[1].set_title("layer-1 output of this unit", fontsize=9)
fig.tight_layout(); nb.show_fig(fig)
layers = virtual.forward_layers(model, x)
print("matches virtual.layer_forward:", float(np.abs(layers[0][0, o] - yo).max()) < 1e-5)'''),
("md", "### Step 4. All three layers (dilations 1, 4, 16) and the pool"),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for l, ax in enumerate(axs):
    im = ax.imshow(layers[l][0], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"); ax.grid(False)
    ax.set_title(f"layer {l + 1} (d={model.layers[l].dilation})", fontsize=9); ax.set_xlabel("sample n")
axs[0].set_ylabel("output line"); fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02).set_label("y")
nb.show_fig(fig)
feat = layers[-1][0].mean(axis=1)                     # RC average pool
logits = model.readout_W @ feat + model.readout_b
p = softmax(logits[None])[0]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].bar(range(len(feat)), feat, color=TIER_COL["light"]["V"]); axs[0].set_xlabel("pooled feature index (layer-3 line)"); axs[0].set_ylabel("mean of y")
axs[1].barh(range(8), p, color=TIER_COL["light"]["V"]); axs[1].set_yticks(range(8)); axs[1].set_yticklabels(cls, fontsize=7.5); axs[1].invert_yaxis()
axs[1].get_yticklabels()[y_true].set_fontweight("bold"); axs[1].set_xlabel("softmax probability"); axs[1].grid(axis="y", visible=False)
fig.tight_layout(); nb.show_fig(fig)
stored = run.res["V"].arrays["probs"][EX]
print("pred:", cls[int(p.argmax())], "| true:", cls[y_true], "| max |p - stored V tier probs|:", float(np.abs(p - stored).max()))'''),
("md", "### Step 5. Cost in hardware terms, and the receptive field"),
("code", '''rows = ["| layer | lines in -> out | K | d | longest tap | length (RG-58) | non-zero taps | ternary adds / sample |", "|---|---|---|---|---|---|---|---|"]
for l, ly_ in enumerate(model.layers):
    O, I, K = ly_.W.shape; nz = int((ly_.W != 0).sum()); d = ly_.dilation
    rows.append(f"| {l + 1} | {I} -> {O} | {K} | {d} | {(K - 1) * d * 10} ns | {(K - 1) * d * 1.98:.1f} m | {nz} of {ly_.W.size} | {nz} |")
nb.md("\\n".join(rows))
print("receptive field:", 1 + sum((l.W.shape[2] - 1) * l.dilation for l in model.layers), "samples (>= 128 input samples)")
print("no multiplications in the taps: every weight is -1, 0 or +1, so a layer is additions and subtractions only.")'''),
("md", "### Step 6. V on many examples (live) against the stored full-test-set result"),
("code", '''N = nb.n_subset(500)
t0 = time.time()
out = virtual.forward(model, run.x_test[:N])["out"]
pred = out.argmax(1)
acc = float((pred == run.labels[:N]).mean())
print(f"live V on the first {N} test examples: accuracy {acc:.4f} ({time.time() - t0:.1f} s)")
print(f"stored V tier, same {N} examples: {float((run.res['V'].arrays['pred'][:N] == run.labels[:N]).mean()):.4f} | whole test set (4000): {cmp['metric_matrix']['accuracy']['V']:.4f}")
print("identical predictions:", bool((pred == run.res['V'].arrays['pred'][:N]).all()))
sw = cmp["sweeps"]["snr_db"]["V"]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.plot(sw["values"], sw["accuracy"], marker="o", color=TIER_COL["light"]["V"], label="V, extra input AWGN (stored, 4000 ex.)")
snrs = run.snr; ax.plot(sorted(set(snrs)), [float((pred[snrs[:N] == s] == run.labels[:N][snrs[:N] == s]).mean()) for s in sorted(set(snrs))], "s--", color=INK["light"]["sec"], label=f"V per dataset SNR (live, {N} ex.)")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); fig.tight_layout(); nb.show_fig(fig)'''),
], "02_virtual")

# ==================================================================================================== 03
nb([
("md", r"""
# 03 Emulated tier (E): fixed-point arithmetic and the RTL cross-check

Tier E replaces float math by what a digital implementation would do. The spec (full text: `docs/fixed_point_spec.md`, single source: the docstring of `dlnn/tiers/emulated.py`):

| step | formula | width |
|---|---|---|
| ADC | $x_q=\mathrm{clip}(\mathrm{rnd}(x/s_0),-L,L)$, $L=2^{b-1}-1$, $s_0=\text{fullscale}/L$ | $b$ bit signed (8 default) |
| ternary MAC | $\mathrm{acc}=b_q[o]+\sum_{i,k}W_{o,i,k}\,x_q[i,n-kd]$ | int32 |
| scale | $z=\mathrm{clip}((\mathrm{acc}\cdot M+2^{14})\gg15,-128,127)$ | int64 intermediate |
| tanh LUT | $v=\mathrm{LUT}[z+128]$, $\mathrm{LUT}[j]=\mathrm{rnd}(127\tanh((j-128)\cdot4/128))$ | int8 |
| per-line gain | $y=(v\,R[o]+2^{14})\gg15$ (Q.15) | int8 |
| pool | $P[o]=\sum_n y[o,n]$ | int32 |

**What is verified how.** The numpy "golden model" runs all 4000 test examples and all sweeps. The Icarus Verilog RTL (`hw/rtl/tdl_layer.v`) is compared **bit for bit** with the golden model on a subset (the first 64 examples, all 3 layers chained, plus layer 1 at ADC widths 4-10): a tier E with any mismatch would be `fallback`, not `ok`. The ADC quantiser, the pooled-sum to float conversion and the float readout are numpy post-processing in both paths.
"""),
("code", SETUP + '''
from dlnn.tiers import emulated, virtual
from dlnn.tiers.common import softmax, mean_kl
model = run.model
x_all, y_all = run.x_test, run.labels
cls = run.classes'''),
("md", "### The LUT and the ADC"),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
j = np.arange(256)
axs[0].plot((j - 128) * 4 / 128, emulated.LUT / 127, color=TIER_COL["light"]["E"], drawstyle="steps-mid", label="256-entry LUT")
axs[0].plot((j - 128) * 4 / 128, np.tanh((j - 128) * 4 / 128), color=INK["light"]["muted"], lw=0.8, label="tanh")
axs[0].set_title("activation LUT (int8)", fontsize=9); axs[0].set_xlabel("pre-activation"); axs[0].legend(fontsize=6.5)
xs = (x_all[0] * model.input_scale)[0]
for bits, ls in ((4, ":"), (6, "--"), (8, "-")):
    q, s0 = emulated.quantize_input(xs[None, None], model.input_fullscale, bits)
    axs[1].plot(q[0, 0] * s0, ls=ls, lw=1.2 if bits < 8 else 1.6, label=f"{bits} bit")
axs[1].plot(xs, color=INK["light"]["text"], lw=0.5, label="input"); axs[1].set_title("ADC reconstruction (one line)", fontsize=9); axs[1].set_xlabel("sample n"); axs[1].set_xlim(0, 60); axs[1].legend(fontsize=6.5)
err = {b: (emulated.quantize_input(xs, model.input_fullscale, b)[0] * emulated.quantize_input(xs, model.input_fullscale, b)[1] - xs) for b in range(4, 11)}
axs[2].semilogy(list(err), [float(np.sqrt(np.mean(e ** 2))) for e in err.values()], "s-", color=TIER_COL["light"]["E"]); axs[2].set_title("ADC RMS error", fontsize=9); axs[2].set_xlabel("bits")
fig.tight_layout(); nb.show_fig(fig)'''),
("md", "### Layer 1 in integers, one output unit, one example\nThe same quantities that exist as signals in the RTL."),
("code", '''EX, o = 6, 3
xq, s0 = emulated.adc_codes(model, x_all[EX:EX + 1], 8)
ly = model.layers[0]
pr = emulated.layer_params(ly, s0)
acc = np.zeros(xq.shape[2], dtype=np.int64)
for k in range(ly.W.shape[2]):
    s = k * ly.dilation
    acc[s:] += np.einsum("i,it->t", ly.W[o, :, k].astype(np.int64), xq[0, :, :xq.shape[2] - s])
acc += pr["bq"][o]
z = np.clip((acc * pr["M"] + (1 << 14)) >> 15, -128, 127)
yq = (emulated.LUT[z + 128].astype(np.int64) * pr["R"][o] + (1 << 14)) >> 15
y_gold, s_out = emulated.layer_fixed(xq, s0, ly)
y_float = virtual.layer_forward(x_all[EX:EX + 1] * model.input_scale, ly)[0, o]
n0 = 40
rows = ["| n | x_q (I) | x_q (Q) | acc (int32) | z | y_q (int8) | y_q*s_out | y float (V) |", "|---|---|---|---|---|---|---|---|"]
for n in range(n0, n0 + 8):
    rows.append(f"| {n} | {xq[0, 0, n]} | {xq[0, 1, n]} | {acc[n]} | {z[n]} | {yq[n]} | {yq[n] * s_out:+.4f} | {y_float[n]:+.4f} |")
nb.md("\\n".join(rows))
print("this cell == emulated.layer_fixed (bit-exact):", bool((yq == y_gold[0, o]).all()), "| max |dequantised - float|:", float(np.abs(yq * s_out - y_float).max()))
print(f"ADC step s0 = {s0:.4f}, LUT range +-{emulated.LUT_RANGE}, M = {pr['M']}, R[o] = {pr['R'][o]} (Q.15), s_out = {s_out:.5f}")'''),
("md", "### ADC-bits sweep, live on a subset (compare with the stored full-set sweep)"),
("code", '''N = nb.n_subset(500)
xs_, ys_ = x_all[:N], y_all[:N]
ref = virtual.forward(model, xs_)["out"]
p_ref = softmax(ref)
rows = []
t0 = time.time()
for bits in range(4, 11):
    o_ = emulated.forward(model, xs_, adc_bits=bits)["out"]
    p_ = softmax(o_)
    rows.append((bits, float((o_.argmax(1) == ys_).mean()), float((o_.argmax(1) == ref.argmax(1)).mean()), mean_kl(p_ref, p_)))
print(f"7 ADC widths on {N} examples: {time.time() - t0:.1f} s")
sw = cmp["sweeps"]["adc_bits"]["E"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
bits_ = [r[0] for r in rows]
axs[0].plot(sw["values"], sw["accuracy"], "s:", color=TIER_COL["light"]["E"], label="stored, 4000 ex.")
axs[0].plot(bits_, [r[1] for r in rows], "o", mfc="none", mec=INK["light"]["text"], label=f"live, {N} ex.")
axs[0].axhline(cmp["metric_matrix"]["accuracy"]["V"], color=INK["light"]["muted"], ls="--", lw=0.8); axs[0].text(4, cmp["metric_matrix"]["accuracy"]["V"] + 0.002, "V", fontsize=7)
axs[0].set_xlabel("ADC bits"); axs[0].set_ylabel("accuracy"); axs[0].legend(fontsize=7)
axs[1].semilogy(sw["values"], np.maximum(sw["mean_kl"], 1e-7), "s:", color=TIER_COL["light"]["E"], label="stored, 4000 ex.")
axs[1].semilogy(bits_, [max(r[3], 1e-7) for r in rows], "o", mfc="none", mec=INK["light"]["text"], label="live")
axs[1].set_xlabel("ADC bits"); axs[1].set_ylabel("mean KL(V || E)  (nats)"); axs[1].legend(fontsize=7)
fig.tight_layout(); nb.show_fig(fig)
nb.md("| ADC bits | accuracy (live) | top-1 agreement with V | mean KL(V\\\\|\\\\|E) |\\n|---|---|---|---|\\n" + "\\n".join(f"| {b} | {a:.4f} | {g:.4f} | {k:.2e} |" for b, a, g, k in rows))'''),
("md", "### RTL vs numpy: bit-exact check (live, small count)\nIf `iverilog` is missing this cell reports it and continues (the tier would then be labelled `fallback: bit-true model (no RTL)`)."),
("code", '''info = emulated.rtl_info()
print("iverilog:", info)
if info["available"]:
    t0 = time.time()
    res = emulated.rtl_crosscheck(model, x_all[:8], {"rtl_examples": 8, "rtl_adc_check": False})
    print(f"live RTL run: {res['examples']} examples x first {res['T']} of {res['T_full']} time steps, tool {res['rtl_tool']}, {time.time() - t0:.0f} s")
    rows = ["| layer | values compared | mismatches | pooled sums compared | pooled mismatches |", "|---|---|---|---|---|"]
    for d_ in res["layers"]:
        rows.append(f"| {d_['layer']} | {d_['values_compared']} | {d_['mismatches']} | {d_['pool_compared']} | {d_['pool_mismatches']} |")
    nb.md("\\n".join(rows))
    print(f"TOTAL: {res['values_compared']} values compared, {res['mismatches']} mismatches; RTL-driven readout identical to golden: {res['readout_identical']}")
else:
    print("iverilog not available: skipping the live RTL run (see the stored result below).")
stored = json.loads((run.dir / "tiers" / "E_rtl.json").read_text()) if (run.dir / "tiers" / "E_rtl.json").exists() else None
if stored and stored.get("ran"):
    print(f"\\nSTORED (the pipeline run): {stored['examples']} examples, {stored['values_compared']} values, {stored['mismatches']} mismatches; "
          f"ADC widths 4-10 on layer 1: {stored['adc_values_compared']} values, {stored['adc_mismatches']} mismatches; readout identical: {stored['readout_identical']}")'''),
("md", "### Mismatch accounting: how E differs from V on the whole test set"),
("code", '''pv, pe = run.res["V"].arrays["pred"], run.res["E"].arrays["pred"]
dis = np.where(pv != pe)[0]
print(f"E vs V: {len(dis)} of {len(pv)} examples change the top-1 class (agreement {1 - len(dis) / len(pv):.5f}); accuracy V {float((pv == y_all).mean()):.5f} vs E {float((pe == y_all).mean()):.5f}")
Pv = run.res["V"].arrays["probs"]
margin = np.sort(Pv, axis=1)
gap = margin[:, -1] - margin[:, -2]
print("V top-1 minus top-2 probability on the disagreeing examples:", np.round(gap[dis], 2).tolist())
print("median over all examples:", round(float(np.median(gap)), 2), "-> the flips happen on near-ties, as expected from 8-bit rounding")
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.hist(gap, bins=40, color=INK["light"]["muted"], label="all examples"); ax.hist(gap[dis], bins=40, color=TIER_COL["light"]["E"], label="E differs from V")
ax.set_yscale("log"); ax.set_xlabel("V margin (top-1 minus top-2 probability)"); ax.legend(fontsize=7); fig.tight_layout(); nb.show_fig(fig)
rows = ["| layer | NMSE of E layer output vs V (first 16 examples) |", "|---|---|"]
for l in range(3):
    a, b = run.trace("V", l), run.trace("E", l)
    rows.append(f"| {l + 1} | {np.mean((a - b) ** 2) / np.var(a):.2e} |")
nb.md("\\n".join(rows))'''),
("md", """
**Takeaways.** 8-bit fixed point is nearly free (top-1 agreement with V 0.9972, accuracy unchanged); the accuracy only starts to drop below 5 ADC bits. The RTL equals the numpy golden model bit for bit on the checked subset. **Not covered by E:** every analog effect (loss, dispersion, noise, coupler errors), the ADC quantiser itself in hardware, and the float readout; the RTL has not been run on the full 4000 examples.
"""),
], "03_emulated")

# ==================================================================================================== 04
nb([
("md", r"""
# 04 Circuit tier (C): ngspice on a unit cell, and what is *not* SPICE

**What SPICE computes here (ngspice 42):**
1. `.ac` per-tap complex transfer functions $H_k(f)$ of lossy RLC coax lines (generic RG-58-like: $L$=250 nH/m, $C$=100 pF/m, constant series $R$ giving 32 dB/us at 100 MHz) into 50 ohm;
2. `.dc` transfer of a generic BJT differential pair, fitted with $A\tanh(v/V_{sat})+\mathrm{offset}$;
3. `.tran` of the **layer-1 unit cell** (taps as ideal VCVS into a summing node, diff pair, RC pool) on 16 test examples, compared with V.

**What is *not* SPICE:** the full 4000-example numbers, and layers 2-3 (dilations 4 and 16, much longer lines), come from the **circuit-calibrated behavioural model**: a closed-form RLC line fitted to the SPICE $H_k(f)$ (max $|\Delta H|$ about $10^{-13}$) and extrapolated to the longer taps, plus the fitted tanh. Nothing about layers 2-3 is verified in SPICE.
"""),
("code", SETUP + '''
import tempfile
from pathlib import Path
from dlnn.tiers import circuit, virtual
model = run.model
P = run.params
ly0 = model.layers[0]
r_nom = circuit.r_from_loss(32.0)
print("ngspice:", circuit.ngspice_path(), "| version", circuit.ngspice_version())
print("extraction source recorded in the run:", P["source"], "| tool", P["tool"])'''),
("md", "## The netlists"),
("code", '''def head(txt, n=22):
    ls = txt.splitlines(); print("\\n".join(ls[:n]) + (f"\\n... ({len(ls) - n} more lines)" if len(ls) > n else ""))
print("=== hw/spice/coax_rg58.lib (generic lossy line, LTRA) ==="); head((nb.ROOT / "hw/spice/coax_rg58.lib").read_text(), 24)
print("\\n=== diff-pair .dc netlist (generated by circuit.netlist_dc) ==="); head(circuit.netlist_dc(), 20)'''),
("code", '''print("=== tap-line .ac netlist (K=8 taps, d=1) ==="); head(circuit.netlist_ac(8, 1, r_nom), 24)
cell = (run.dir / "spice" / "unit_cell_ex00.cir")
print("=== layer-1 unit-cell .tran netlist, example 0 (from the run). Shown: header, supplies, 2 of the tap lines, ALL elements of output 0; the PWL input data and outputs 1-7 are elided ===")
import re
txt = cell.read_text().splitlines()
pick = [l for l in txt if l.startswith(("*", ".")) or re.match(r"^(Vcc|Vcm|Xline_0_[12] |Rterm_0_[12] |Etap_0_|Epre0 |Vb0 |Ep0 |En0 |Xdp0 |Eout0 |Rpool0 |Cpool0 )", l)]
print(f"[{len(txt)} lines in the file]")
print("\\n".join(l[:150] for l in pick))
print("Vin0 in0 0 PWL(0 0 ... 128 symbols, 1 ns edges ...)   <- input line I as a PWL source (elided)")'''),
("md", "## Live `.dc`: the diff pair is a tanh"),
("code", '''t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    dc = circuit.extract_dc(Path(td), True)
v, yv, ft = dc["v"], dc["y"], dc["fit"]
print(f"ngspice .dc: {time.time() - t0:.1f} s | fit A = {ft['A_v']:.4f} V, V_sat = {1e3 * ft['vsat_v']:.2f} mV (2*V_T design 51.8 mV), R^2 = {ft['r2']:.8f}, rms residual {1e3 * ft['rms_resid_v']:.3f} mV")
print(f"fitted V_T = {1e3 * ft['v_t_fitted_v']:.2f} mV vs ngspice thermal voltage at 27 C = {1e3 * ft['v_t_ngspice_27c_v']:.2f} mV")
print(f"stored extraction (run): V_sat {1e3 * P['diffpair']['vsat_v']:.2f} mV, R^2 {P['diffpair']['r2']:.8f}  -> same")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
axs[0].plot(1e3 * v, yv, color=TIER_COL["light"]["C"], lw=2.2, label="ngspice .dc")
axs[0].plot(1e3 * v, ft["A_v"] * np.tanh(v / ft["vsat_v"]) + ft["offset_v"], "k:", lw=1.2, label="A tanh(v/V_sat) fit")
axs[0].set_xlabel("differential input (mV)"); axs[0].set_ylabel("differential output (V)"); axs[0].legend(fontsize=7)
axs[1].plot(1e3 * v, 1e3 * (yv - (ft["A_v"] * np.tanh(v / ft["vsat_v"]) + ft["offset_v"])), color=INK["light"]["sec"])
axs[1].set_xlabel("input (mV)"); axs[1].set_ylabel("fit residual (mV)")
fig.tight_layout(); nb.show_fig(fig)'''),
("md", "## Live `.ac`: per-tap transfer functions $H_k(f)$ and the closed-form fit"),
("code", '''t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    ac = circuit.extract_ac(Path(td), 8, 1, r_nom, True)
f, H, fit = ac["freq_hz"], ac["H"], ac["fit"]
print(f"ngspice .ac: {time.time() - t0:.1f} s, {len(f)} points up to {f[-1] / 1e6:.0f} MHz")
print(f"closed-form RLC fit: R = {fit['R_ohm_per_m']:.4f} ohm/m, L = {1e9 * fit['L_h_per_m']:.1f} nH/m, C = {1e12 * fit['C_f_per_m']:.1f} pF/m -> loss {fit['loss_db_per_us_100mhz']:.2f} dB/us @100 MHz, max |dH| = {fit['fit_max_abs_err']:.1e}")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for k in range(1, 8):
    axs[0].plot(f / 1e6, 20 * np.log10(np.abs(H[k])), lw=1.3, color=TIER_COL["light"]["C"], alpha=0.35 + 0.09 * k)
    fitH = circuit.line_H(f, circuit.tap_lengths(8, 1)[k], fit["R_ohm_per_m"], fit["L_h_per_m"], fit["C_f_per_m"])
    axs[0].plot(f[::40] / 1e6, 20 * np.log10(np.abs(fitH[::40])), "k.", ms=2.5)
axs[0].text(300, 20 * np.log10(abs(H[7, np.argmin(abs(f - 300e6))])) - 0.6, "k=7", fontsize=7); axs[0].text(300, 20 * np.log10(abs(H[1, np.argmin(abs(f - 300e6))])) + 0.2, "k=1", fontsize=7)
axs[0].set_xlim(0, 400); axs[0].set_xlabel("frequency (MHz)"); axs[0].set_ylabel("|H_k| (dB)"); axs[0].set_title("SPICE (lines) and closed-form fit (dots)", fontsize=9)
dly = np.array([t["delay_error_s"] for t in ac["taps"]]) * 1e12
axs[1].bar(range(8), dly, color=TIER_COL["light"]["C"]); axs[1].set_xlabel("tap k"); axs[1].set_ylabel("group-delay error (ps)"); axs[1].set_title("excess delay vs k*T_s", fontsize=9)
fig.tight_layout(); nb.show_fig(fig)
rows = ["| k | length (m) | DC gain (dB) | gain @100 MHz (dB) | delay error (ps) |", "|---|---|---|---|---|"]
for t in ac["taps"]:
    rows.append(f"| {t['k']} | {t['length_m']:.2f} | {t['dc_gain_db']:.2f} | {t['gain_100mhz_db']:.2f} | {1e12 * t['delay_error_s']:.0f} |")
nb.md("\\n".join(rows))
nb.md("*The LTRA line has constant series R and no shunt G, so the low-frequency loss is overstated (DC about -3.6 dB vs -2.2 dB at 100 MHz for the longest layer-1 tap): a known limit of this SPICE line model, stated in the tier's `fidelity.omits`.*")'''),
("md", "## Live `.tran`: the layer-1 unit cell on a few examples, SPICE vs V vs calibrated model"),
("code", '''from dlnn.model.npz import Layer
ds_min = {"x_test": run.x_test, "task": "classification"}
fitp = P["line"]["fit"]; ampp = P["amp"]
line = circuit.LineModel(fitp["R_ohm_per_m"], fitp["L_h_per_m"], fitp["C_f_per_m"])
amp = circuit.AmpModel(ampp["vsat_scale"], ampp["gain_scale"], ampp["offset_rel"])
trim = float(P["loss_comp_gain_trim_per_layer"][0])
t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    tran, y_c, y_v, ex = circuit.run_unit_cell(Path(td), model, ds_min, {"c_n_examples": 3, "c_outputs": 8, "c_workers": 2}, r_nom, trim, line, amp)
print(f"3 examples x 8 output lines in ngspice: {time.time() - t0:.0f} s (tran: {tran['tran_timepoints_per_run']} time points each)")
print(f"layer-1 NMSE SPICE vs V (variance-normalised): {tran['nmse']:.4f} | calibrated model vs SPICE: {tran['nmse_calibrated_model_vs_spice']:.2e}")
cal = circuit.calibrated_forward(model, ex, line, amp, comp="layer", trims=[trim, 1, 1])["layers"][0][:, :8]
o = 4
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
e = 0
axs[0].plot(y_v[e, o], color=TIER_COL["light"]["V"], lw=2.4, label="V (ideal)")
axs[0].plot(cal[e, o], color=TIER_COL["light"]["C"], ls="--", lw=1.4, label="C calibrated model")
axs[0].plot(y_c[e, o], "o", mfc="none", mec=INK["light"]["text"], ms=3.2, label="ngspice .tran samples")
axs[0].set_ylabel(f"layer-1 output {o}"); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
axs[1].plot(y_c[e, o] - y_v[e, o], color=INK["light"]["text"], lw=1.0, label="SPICE - V"); axs[1].plot(cal[e, o] - y_c[e, o], color=TIER_COL["light"]["C"], lw=1.0, ls="--", label="calibrated - SPICE")
axs[1].set_xlabel("sample n"); axs[1].set_ylabel("residual"); axs[1].legend(fontsize=7)
fig.tight_layout(); nb.show_fig(fig)
u16 = P["unit_cell_tran"]
print(f"\\nStored (run, 16 examples x 8 lines): NMSE vs V = {u16['nmse']:.4f} (no gain trim: {u16['no_trim']['nmse']:.4f}); calibrated vs SPICE = {u16['nmse_calibrated_model_vs_spice']:.1e}; SPICE wall {u16['spice_wall_s']:.0f} s")'''),
("md", """
### What is SPICE and what is calibrated (for this run)

| quantity | where it comes from |
|---|---|
| per-tap $H_k(f)$, $k\\le7$, $d=1$ | **ngspice `.ac`** |
| diff-pair tanh ($V_{sat}$, gain, offset, $R^2$) | **ngspice `.dc`** + fit |
| layer-1 activations on 16 examples | **ngspice `.tran`** (unit cell, 8 outputs) |
| layer-1 activations, the other 3984 examples | calibrated behavioural model (reproduces SPICE to NMSE $\\approx 10^{-7}$ on the 16) |
| layers 2-3 (all examples) | calibrated model, **line fit extrapolated to 4x and 16x longer taps. Not SPICE.** |
| accuracy / agreement / KL / sweeps of tier C | calibrated model |
| RC pool | SPICE uses a leaky average ($\\tau=NT_s$); the model trains with a uniform mean (NMSE 0.22 between them): not matched |
"""),
("md", "## The loss sweep, and the choice of loss compensation (full test set, calibrated model, not SPICE)"),
("code", '''diag = json.loads((run.dir / "params" / "circuit_diagnostics.json").read_text())
swC = cmp["sweeps"]["loss_db_per_us"]["C"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(swC["values"], swC["accuracy"], "^--", color=TIER_COL["light"]["C"], label=f"C default: one gain trim per layer")
sty = {"none": ("v", ":"), "tap": ("d", "-.")}
for mode, v in diag["modes"].items():
    s = v["loss_sweep"]; ax.plot(s["values"], s["accuracy"], marker=sty[mode][0], ls=sty[mode][1], color=INK["light"]["sec"], label={"none": "no compensation", "tap": "ideal per-tap repeater / equaliser"}[mode])
ax.axhline(cmp["metric_matrix"]["accuracy"]["V"], color=INK["light"]["muted"], ls="--", lw=0.8); ax.text(0.5, cmp["metric_matrix"]["accuracy"]["V"] + 0.003, "V", fontsize=7.5)
ax.axvline(32, color=INK["light"]["warn"], ls=":"); ax.text(31, 0.64, "32 dB/us\\nheadline", fontsize=7, color=INK["light"]["warn"], ha="right")
ax.set_xlabel("line loss at 100 MHz (dB / us)"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); nb.show_fig(fig)
rows = ["| compensation (at 32 dB/us) | accuracy | top-1 agreement with V | mean KL |", "|---|---|---|---|",
        f"| one ideal gain trim per layer (tier default) | {cmp['metric_matrix']['accuracy']['C']:.3f} | {cmp['metric_matrix']['top1_agreement_vs_virtual']['C']:.3f} | {cmp['metric_matrix']['mean_kl_virtual_vs_tier']['C']:.3f} |"]
for mode, v in diag["modes"].items():
    m = v["metrics_at_fitted_loss"]; rows.append(f"| {mode} | {m['accuracy']:.3f} | {m['top1_agreement_vs_virtual']:.3f} | {m['mean_kl_virtual_vs_tier']:.3f} |")
nb.md("\\n".join(rows))
nb.md("**Reading the gap.** At 0 dB/us the calibrated model reproduces V (0.863). The C headline of 0.663 is the **32 dB/us operating point with the tier's default loss compensation** (a single trim per layer). The diagnostics in `params/circuit_diagnostics.json` show the compensation choice matters as well: with no compensation or an ideal per-tap equaliser the same loss gives about 0.80. Most of C's 20-point gap to V is therefore the default per-layer gain trim (0.663 vs 0.804 with none), not line loss (about 6 points); in layer 3 the trim of 2.85 puts 72.8 % of the activations into tanh saturation against 49.9 % in V (`docs/circuit_extraction.md`, measured with `scripts/saturation_check.py`). Per-tap repeaters are what the plan says the dilation-16 layer needs, and the line loss alone makes the operating point unbuildable without them; this simulation does not include them in the headline.")'''),
("md", "## The same example through all tiers (figure 2) and the C vs R channels (figure 3)"),
("code", "nb.show('fig2_traces', run)"),
("code", "nb.show('fig3_channel', run)"),
], "04_circuit")

# ==================================================================================================== 05
nb([
("md", r"""
# 05 RF tier (R): scikit-rf channel model

Each tap path is a scikit-rf two-port cascade
$$\text{coupler}\ \star\ \text{mismatch junction}(\Gamma)\ \star\ \mathrm{Coaxial.line}(k\,d\,T_s\,v_p)\ \star\ \text{mismatch junction}(\Gamma)\ \star\ \text{Wilkinson branch}(1/\sqrt N)$$
and its $S_{21}(f)$ becomes a frequency response $H_{o,i,k}(f)$ (delay-trimmed to the nominal $kdT_s$ at 50 MHz). IFFT gives impulse responses, a symbol-rate FIR per layer is applied with FFT convolution to all examples, then the amplifier (tanh), stage noise and a 10-bit ADC. *Channel level only: no EM, layout or connector effects; the amplifier is memoryless.*

**Per-stage SNR convention (important).** R's `snr_db` is the SNR at **each amplifier input**: with reference power $P_{ref,l}$ of the ideal pre-activation of layer $l$ and noise figure NF,
$$\sigma_l^2=P_{ref,l}\;10^{(\mathrm{NF}-\mathrm{SNR})/10},$$
three independent noise sources (one per layer). V, E and C instead add AWGN at the **input** only. So the `snr_db` axes of R and V/E are not the same quantity; the demonstration below makes the difference explicit.
"""),
("code", SETUP + '''
import skrf
from dlnn.tiers import rf, virtual
from dlnn import cascade, sweeps
from dlnn.tiers.common import softmax, mean_kl
model = run.model
x_all, y_all = run.x_test, run.labels
N = nb.n_subset(500)
xs, ys = x_all[:N], y_all[:N]
# the run's own operating point for R: defaults, plus the cascade hook (SPICE line loss + diff-pair scales) if the run used --cascade
cfg_run = {}
if run.cascade:
    ex_ = cascade.load_extracted(run.dir / "params" / "circuit_extracted.json")
    cfg_run = cascade.apply_to_rf_cfg({}, ex_); cfg_run["amp_offset_rel"] = float(ex_["amp"]["offset_rel"])
rc = rf.resolve_cfg(cfg_run)
print("scikit-rf", skrf.__version__, "| R operating point of this run: loss", round(rc["loss_db_per_us"], 2), "dB/us, isolation", rc["isolation_db"], "dB, tap err", rc["tap_amp_err_db"], "dB /", rc["tap_phase_err_deg"], "deg, Gamma", rc["gamma_mismatch"],
      "crosstalk", rc["crosstalk_db"], "dB, stage SNR", rc["snr_db"], "dB, NF", rc["nf_db"], "dB, ADC", rc["adc_bits"], "bit")
print("subset for live computations: first", N, "test examples (not class/SNR balanced); the stored sweeps below use all 4000")'''),
("md", "## A scikit-rf line network, built by hand"),
("code", '''import skrf as rfk
from skrf.media import Coaxial
eps, dint, dout = rf._coax_geometry(rc)
sigma, tand = rf.coax_parameters(rc)
freq = rfk.Frequency(1, 400, 400, "MHz")
med = Coaxial(freq, z0_port=50, Dint=dint, Dout=dout, epsilon_r=eps, tan_delta=tand, sigma=sigma)
print(f"Coaxial medium: Dint {1e3 * dint:.2f} mm, Dout {1e3 * dout:.2f} mm (sets Z0 = 50 ohm), eps_r {eps:.3f} (v_p = {rc['vp_cm_per_ns']} cm/ns), sigma {sigma:.3g} S/m, tan_delta {tand:.2g}")
print("(sigma and tan_delta were solved so that the matched-line loss at 100 MHz equals the requested dB/us)")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for L, c in zip((1.98, 7.9, 13.9), ("#86b6ef", "#3987e5", "#184f95")):
    ln = med.line(L, unit="m")
    s21 = ln.s[:, 1, 0]
    axs[0].plot(freq.f / 1e6, 20 * np.log10(np.abs(s21)), color=c, label=f"{L} m"); axs[1].plot(freq.f / 1e6, np.unwrap(np.angle(s21)) / (2 * np.pi), color=c)
axs[0].set_xlabel("MHz"); axs[0].set_ylabel("|S21| (dB)"); axs[0].legend(fontsize=7, title=f"line length @ {rc['loss_db_per_us']:.0f} dB/us", title_fontsize=7)
axs[1].set_xlabel("MHz"); axs[1].set_ylabel("phase (cycles)")
fig.suptitle("skrf.media.Coaxial.line: skin-effect loss ~ sqrt(f) and dispersion", x=0.01, ha="left", fontsize=9.5, fontweight="bold"); fig.tight_layout(); nb.show_fig(fig)
print("skrf network object of one line:", ln)'''),
("md", "## One complete tap path (coupler, mismatch, line, mismatch, Wilkinson) and its dependence on loss"),
("code", '''g = rf.make_grid(rc, 3 * 16 * 7 + 16)
Ts = 10e-9
def rc_spec(f):
    x_ = np.clip(f * Ts, 0, 1); return np.where(f * Ts <= 1.0, 0.5 * Ts * (1 + np.cos(np.pi * x_)), 0.0)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.8))
cols = {0.0: "#cde2fb", 5.0: "#86b6ef", 14.0: "#3987e5", 32.0: "#184f95"}
for (kk, dd), axr in zip(((7, 1), (7, 16)), axs):
    for loss, c in cols.items():
        rc_ = dict(rc, loss_db_per_us=loss)
        H, stat = rf.line_response(rc_, g, kk * dd, 16 if dd == 1 else 256)
        Hn = H / stat
        axr[0].plot(g.f[1:] / 1e6, 20 * np.log10(np.abs(Hn[1:]) + 1e-9), color=c, label=f"{loss:g} dB/us")
        yy = np.fft.irfft(Hn * rc_spec(g.f), n=g.N) * g.fs
        tt = np.arange(g.N) * Ts / g.U; tt = np.where(tt > tt[-1] / 2, tt - tt[-1] - tt[1], tt)
        o_ = np.argsort(tt); axr[1].plot(tt[o_] * 1e9, yy[o_], color=c)
    axr[0].set_xlim(0, 200); axr[0].set_ylim(-45 if dd == 16 else -14, 3); axr[0].set_ylabel(f"|H| tap k={kk}, d={dd}\\n({kk * dd * 1.98:.1f} m) (dB)"); axr[0].set_xlabel("MHz")
    axr[1].set_xlim(-15, 60); axr[1].set_ylabel("response to 10 ns RC symbol"); axr[1].set_xlabel("t - nominal delay (ns)")
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); nb.show_fig(fig)
nb.md("*(top: the longest layer-1 tap, 13.9 m; bottom: the longest layer-3 tap, 222 m = 1.12 us. Static coupler / combiner gain divided out.)* At 32 dB/us the layer-3 tap loses about **36 dB** and is visibly smeared; without per-tap repeaters this operating point is not buildable. It is not the whole reason C and R fall behind V in the full-set tables: for C the default per-layer gain trim accounts for most of the gap (notebook 04, loss-compensation section).")'''),
("md", "## Live parameter sweeps on the subset (compared with the stored 4000-example sweeps)"),
("code", '''t0 = time.time()
pref = rf.reference_power(model, xs)
Vclean = virtual.forward(model, xs)["out"]; pV = softmax(Vclean)
def score(out):
    return float((out.argmax(1) == ys).mean()), float((out.argmax(1) == Vclean.argmax(1)).mean()), mean_kl(pV, softmax(out))
plan = {"snr_db": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0], "loss_db_per_us": [0.0, 5.0, 14.0, 32.0], "isolation_db": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        "tap_amp_err_db": [0.0, 0.25, 0.5, 1.0, 2.0], "adc_bits": [4, 5, 6, 7, 8, 9, 10]}
chans, _ = rf.build_channels(model, rc)
live = {}
for axis, vals in plan.items():
    res = []
    for v in vals:
        rcv = dict(rc, **{axis: v})
        ch = chans if axis in ("snr_db", "adc_bits") else rf.build_channels(model, rcv)[0]
        res.append(score(rf.forward_with(model, xs, rcv, ch, pref)["out"]))
    live[axis] = np.array(res)
    print(f"{axis:15s} done ({time.time() - t0:.0f} s)")
print(f"total live sweep time on {N} examples: {time.time() - t0:.0f} s")'''),
("code", '''fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
for ax, (axis, vals) in zip(axs.ravel(), plan.items()):
    st = cmp["sweeps"][axis]["R"]
    ax.plot(st["values"], st["accuracy"], color=TIER_COL["light"]["R"], ls=LINESTYLES["R"], marker="D", ms=4, label="stored (4000 ex.)")
    ax.plot(vals, live[axis][:, 0], "o", mfc="none", mec=INK["light"]["text"], ms=5, label=f"live ({N} ex.)")
    ax.set_xlabel(axis); ax.set_ylabel("accuracy")
    if axis == "loss_db_per_us":
        ax.axvline(rc["loss_db_per_us"], color=INK["light"]["warn"], ls=":"); ax.legend(fontsize=6.5)
axs[1, 2].axis("off")
axs[1, 2].text(0, 0.95, "R sweep axes\\n\\n* snr_db: PER-STAGE SNR\\n* loss: dB/us at 100 MHz\\n* isolation: coupler\\n* tap_amp_err: sigma, dB\\n* adc_bits\\n\\nvertical dotted line = this\\nrun's operating point", va="top", fontsize=7.5, color=INK["light"]["sec"])
fig.tight_layout(); nb.show_fig(fig)
rows = ["| axis | value | live accuracy | live agreement with V | live mean KL | stored accuracy (4000) |", "|---|---|---|---|---|---|"]
for axis, vals in plan.items():
    for v, r_, sa in zip(vals, live[axis], cmp["sweeps"][axis]["R"]["accuracy"]):
        rows.append(f"| {axis} | {v:g} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]:.3f} | {sa:.3f} |")
nb.md("\\n".join(rows))
nb.md("*The live subset is the first 500 test examples, so its numbers differ from the stored ones by sampling noise (a few points), not by method. The loss axis is the dominant one; isolation and tap errors are second-order; ADC bits are flat (R's 10-bit ADC is not the bottleneck).*")'''),
("md", "## Per-stage SNR vs input AWGN: why the two `snr_db` axes differ"),
("code", '''rng = np.random.default_rng(0)
ideal = dict(rf.resolve_cfg({"preset": "ideal"}))
rows = ["| condition | accuracy | top-1 agreement with clean V |", "|---|---|---|"]
for snr_db in (30.0, 20.0, 10.0):
    xin = sweeps.add_awgn(xs, snr_db, rng)
    out = virtual.forward(model, xin)["out"]; a_, g_, _ = score(out)
    rows.append(f"| V, AWGN {snr_db:.0f} dB at the INPUT | {a_:.3f} | {g_:.3f} |")
chI, _ = rf.build_channels(model, ideal)
for snr_db in (30.0, 20.0, 10.0):
    out = rf.forward_with(model, xs, dict(ideal, snr_db=snr_db), chI, pref)["out"]; a_, g_, _ = score(out)
    rows.append(f"| R ideal channel (0 dB/us, no errors), {snr_db:.0f} dB per STAGE (x3 stages) | {a_:.3f} | {g_:.3f} |")
nb.md("\\n".join(rows))
nb.md("With an ideal channel, R at 30 dB per stage agrees with V on ~98 % of examples (`docs/tiers.md`, 4000 examples, 3 seeds: 0.986). Three noise sources accumulate through the layers: at 10 dB per stage R falls to about 0.75 accuracy while V with 10 dB of input AWGN keeps about 0.85 (the dataset already contains 0-20 dB noise). That is why the R curve in the SNR panel lies below V's and why the two `snr_db` axes must not be read as the same quantity.")'''),
("md", "## Figures: C vs R channels (figure 3) and the sweeps (figure 5)"),
("code", "nb.show('fig3_channel', run)"),
("code", "nb.show('fig5_sweeps', run)"),
], "05_rf")

# ==================================================================================================== 06
nb([
("md", r"""
# 06 Compare the tiers, then read the LLM commentary

Everything here is read from the finished run (`compare.json`, tier arrays, `llm/commentary.json`); nothing is recomputed. The small LLM (Qwen2.5-1.5B-Instruct Q4_K_M via llama.cpp, grammar-constrained JSON) is **only a commentator**: it reads a $\le$1.5k-token text card rendered from `compare.json` and writes findings. It runs no simulation and is not the model under study (that is the ternary delay-line net).
"""),
("code", SETUP),
("md", "## Side-by-side matrix"),
("code", '''nb.results_table(cmp)
rows = ["| tier | " + " | ".join(f"{s} dB" for s in ("0", "5", "10", "15", "20")) + " |", "|---|---|---|---|---|---|"]
for t in cmp["tiers"]:
    a = cmp["metric_matrix"]["acc_per_snr"][t]
    rows.append(f"| {t} | " + " | ".join(f"{a[s]:.3f}" for s in ("0", "5", "10", "15", "20")) + " |")
nb.md("**Accuracy per SNR**\\n\\n" + "\\n".join(rows))
nb.md(nb.honesty_box(run))
for t in cmp["tiers"]:
    i = cmp["tier_info"][t]
    print(f"{t}: {i['status']} | omits: " + " | ".join(i["fidelity"]["omits"]))
    for w in i["warnings"][:2]:
        print("    warning:", w[:200])'''),
("md", "## Figure 4: the same four examples through every tier (true label from the dataset)"),
("code", "nb.show('fig4_outputs', run)"),
("md", "## Figure 5: sweeps"),
("code", "nb.show('fig5_sweeps', run)"),
("md", """
**What the sweeps say.** E is flat down to 5 ADC bits. C and R are not bugs: at the headline operating point (`--cascade`: the fitted 32 dB/us line loss, equal to the netlist input, no repeaters; not buildable) they are 16-20 points below V, and for C most of that is its default per-layer gain trim rather than line loss (0.663 vs 0.804 with no compensation); at 0 dB/us they match V (0.863 / 0.861). V's own accuracy falls only below 10 dB input SNR. R's per-stage SNR axis is a different quantity from the input-AWGN axis of V/E/C (notebook 05).
"""),
("md", "## Figures 6 and 7: where the differences are"),
("code", "nb.show('fig6_confusion', run)"),
("code", "nb.show('fig7_agreement', run)"),
("md", "## The card the commentator reads"),
("code", '''from dlnn.llm import card
cp = run.dir / "llm" / "card.txt"
ctext = cp.read_text() if cp.exists() else card.render_card(cmp)
print(f"card: {len(ctext)} chars, ~{card.approx_tokens(ctext)} tokens by card.approx_tokens (conservative 2.4 chars/token; budget in dlnn/llm/card.py: MAX_TOKENS={card.MAX_TOKENS}, MAX_CHARS={card.MAX_CHARS}); "
      f"{'within' if len(ctext) <= card.MAX_CHARS else 'OVER'} the character budget\\n")
print(ctext)'''),
("md", "## The commentary, with verification status (figure 8)"),
("code", '''lrun = nb.llm_run_dir()
com = json.loads((lrun / "llm" / "commentary.json").read_text())
print(f"commentary read from {lrun.name}: source = {com['source']}, backend = {com['backend']}, numbers_verified = {com.get('numbers_verified')}, ids_verified = {com.get('ids_verified')}")
if lrun != run.dir:
    print(f"(the headline run {run.dir.name} carries a template commentary; {lrun.name} is a run of the same model with real Qwen output)")
nb.show('fig8_llm', run, commentary=com)
ver = com.get("verification")
if ver:
    print("verifier:", {k: ver[k] for k in ("n_numbers", "retried", "n_unverified_raw", "n_unverified_final")})
for u in com.get("unverified", []):
    print("UNVERIFIED:", u["where"], "token", repr(u["token"]), "|", u["sentence"])
aud = figures.audit_commentary(com, run.status, run.cmp)
print("independent audit against compare.json statuses:", aud if aud else "no contradictions found")'''),
("code", '''# the template commentary (labelled as such), for contrast
tc = run.dir / "llm" / "commentary.json"
t = json.loads(tc.read_text())
if t["source"] == "template":
    print("headline run commentary is a TEMPLATE (backend 'null'), not model output:")
    nb.show('fig8_llm', run, commentary=t)'''),
("code", '''# optional: re-run the LLM now (needs the GGUF; off by default because it takes about a minute and the box is shared)
if os.environ.get("RUN_LLM") == "1" and nb.llm_available():
    from dlnn.llm import backend
    t0 = time.time()
    fresh = backend.get_backend("qwen").comment(cmp, ctext)
    print(f"live Qwen run took {time.time() - t0:.0f} s; source = {fresh['source']}, numbers_verified = {fresh.get('numbers_verified')}")
    nb.show('fig8_llm', run, commentary=fresh)
else:
    print("RUN_LLM != 1 (or the GGUF is missing): showing the saved commentary only. Set RUN_LLM=1 to run Qwen live.")'''),
("md", """
**How to read the commentary.** The verifier checks every number and every `metric_id` / tier id against the card, retries once, and dims what is still wrong. It does **not** check whether an *explanation* is true: in the saved Qwen output the finding "C models a 6-bit ADC" is flagged (6 is not on the card), and a caveat calls R a "FALLBACK" although `compare.json` has `R: status ok` (the independent audit above catches that one). The saved commentary also has `ids_verified: false` (tier R has no finding), which the figure now shows next to `numbers_verified: true`. The cause of the C and R gap in this run (32 dB/us line loss with no repeaters, plus C's default per-layer gain trim) is **not** mentioned by the model; the card does not state the operating point. Treat the commentary as a summary of the card, never as analysis.
"""),
], "06_compare_and_llm")

print("wrote", sorted(p.name for p in HERE.glob("0*.ipynb")))
