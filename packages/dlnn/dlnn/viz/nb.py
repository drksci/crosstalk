"""Helpers shared by the notebooks (kept out of the notebooks so each one stays readable).

Environment variables
    DLNN_RUN       run directory (name under out/ or a path); default ``out/modrec-w32-full``
    DLNN_LLM_RUN   run directory whose llm/commentary.json is shown; default: the newest run under out/ whose
                   commentary is real model output for the same model, else DLNN_RUN
    DLNN_NB_N      subset size for the live sweeps (default 500)
    RUN_LLM        "1" re-runs the LLM commentary (only if the GGUF file is present); default: read the saved one
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import figures
from .common import INK, TIER_COL, TIER_SHORT, Run, ctx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "modrec-w32-full"


def run_dir(env: str = "DLNN_RUN", default: str = DEFAULT_RUN) -> Path:
    v = os.environ.get(env, default)
    p = Path(v)
    if not p.is_absolute():
        p = ROOT / "out" / v if not (ROOT / v).exists() else ROOT / v
    return p


def load_run(env: str = "DLNN_RUN", default: str = DEFAULT_RUN) -> Run:
    p = run_dir(env, default)
    if not (p / "compare.json").exists():
        raise FileNotFoundError(f"{p}: no compare.json. Set DLNN_RUN to a finished run under out/ "
                                f"(e.g. produced by `python run.py --task modrec --tiers V,E,C,R --cascade --llm none`).")
    return Run(p)


def llm_run_dir() -> Path:
    if os.environ.get("DLNN_LLM_RUN"):
        return run_dir("DLNN_LLM_RUN")
    main = run_dir()
    mm = json.loads((main / "compare.json").read_text()).get("model_sha256")
    best = None
    for d in sorted((ROOT / "out").glob("*")):
        cj, cm = d / "compare.json", d / "llm" / "commentary.json"
        if not (cj.exists() and cm.exists()):
            continue
        try:
            if json.loads(cj.read_text()).get("model_sha256") == mm and json.loads(cm.read_text()).get("source") == "model":
                best = d
        except Exception:  # noqa: BLE001
            continue
    return best or main


def n_subset(default: int = 500) -> int:
    return int(os.environ.get("DLNN_NB_N", default))


def setup() -> None:
    """Light, opaque notebook style (the exported PNG/SVG files are transparent; notebook inline figures are not)."""
    k = INK["light"]
    plt.rcParams.update({"figure.facecolor": k["surface"], "axes.facecolor": k["surface"], "savefig.facecolor": k["surface"],
                         "figure.dpi": 110, "savefig.dpi": 140, "text.color": k["text"], "axes.labelcolor": k["text"],
                         "axes.edgecolor": k["axis"], "xtick.color": k["sec"], "ytick.color": k["sec"], "grid.color": k["grid"],
                         "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "legend.frameon": False, "figure.figsize": (7.0, 3.4)})


def png(fig, theme: str = "light", dpi: int = 130):
    """Return an IPython Image of a matplotlib figure on an opaque surface; closes the figure."""
    from IPython.display import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=INK[theme]["surface"], transparent=False, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return Image(data=buf.getvalue())


def show(name: str, run: Run | str | Path, theme: str = "light", **kw):
    """Build one of the exported figures (same code as the PNG/SVG files) and display it inline on an opaque surface."""
    from IPython.display import display
    run = run if isinstance(run, Run) else Run(run)
    with ctx(theme):
        fig = figures.BUILDERS[name](run, theme, **kw)
        display(png(fig, theme))


def show_fig(fig, theme: str = "light"):
    from IPython.display import display
    display(png(fig, theme))


def md(text: str):
    from IPython.display import Markdown, display
    display(Markdown(text))


def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float) and v != 0 and abs(v) < 10 ** (-nd + 1):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def results_table(cmp: dict, metrics=("accuracy", "macro_f1", "top1_agreement_vs_virtual", "mean_kl_virtual_vs_tier",
                                      "logit_nmse_vs_virtual")):
    """Markdown table: tiers x metrics + status/tool/runtime."""
    mm = cmp["metric_matrix"]
    head = "| tier | status | tool | n | " + " | ".join(metrics) + " | runtime (s) |"
    rows = [head, "|" + "---|" * (len(metrics) + 5)]
    for t in cmp["tiers"]:
        i = cmp["tier_info"][t]
        vals = [fmt(mm.get(m, {}).get(t)) for m in metrics]
        rows.append(f"| **{t}** {TIER_SHORT[t].split(' ', 1)[1]} | {i['status']} | {i['tool']['name']} {i['tool']['version']} | "
                    f"{i['n_examples']} | " + " | ".join(vals) + f" | {i['runtime_s']:.0f} |")
    ref = cmp.get("reference")
    if ref:
        rows.append(f"| *float32 reference (not a tier)* | n/a | numpy float32 | {ref['n_examples']} | {fmt(ref['accuracy'])} | "
                    f"{fmt(ref['macro_f1'])} | n/a | n/a | n/a | n/a |")
    md("\n".join(rows))


def honesty_box(run: Run | None = None) -> str:
    base = """> **What is and is not simulated (read this before the numbers).**
>
> * **Simulation only, synthetic data. No hardware was built or measured.** Coax and transistor models are generic textbook ones (RG-58-like RLC line, generic NPN), not vendor data.
> * **V** = ideal numpy math (the reference). **E** = numpy bit-true fixed point on all 4000 test examples; the Icarus Verilog RTL is bit-compared with it on a subset (first 64 examples, 0 mismatches), the rest is numpy.
> * **C** = ngspice on a layer-1 *unit cell* (per-tap `.ac`, diff-pair `.dc`, `.tran` on 16 examples). **The 4000-example numbers and layers 2-3 come from the circuit-calibrated behavioural model, not from SPICE.**
> * **R** = scikit-rf *channel-level* model (two-port cascades, IFFT, numpy convolution). Not EM, not layout.
> * The float32 baseline is a *reference*, not a tier."""
    if run is not None and run.cascade:
        base += """
> * **With `--cascade` the R tier uses the line loss that C fitted (32 dB/us, equal to the value put into the netlist) with no repeaters, and C runs at the same loss** (default: one ideal gain trim per layer). That is a far harsher operating point than R's standalone default (5 dB/us), and it is not buildable without repeaters. C/R accuracy of about 0.66/0.70 vs V 0.863 is not a bug: their loss sweeps recover ~0.86 at 0 dB/us. But for C most of the gap is the default per-layer gain trim over-driving layer 3, not line loss (C: 0.663 default, 0.804 with no compensation, 0.805 with ideal per-tap repeaters at 32 dB/us; `docs/circuit_extraction.md`)."""
    return base


def llm_available() -> bool:
    return (ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf").exists()
