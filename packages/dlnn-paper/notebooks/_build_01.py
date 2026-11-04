from _common import *

NB = "01_small_nets_and_arena"
cells = [
("md", r"""
# 01 Small nets and the arena

[![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/01_small_nets_and_arena.ipynb)

**What this notebook does** (simulation only; the board is designed, not built, nothing is measured):

1. **Trains the small DLNN** (TDL-Net-B, width 8, d = 1, ternary taps) and the **arena baselines** (MLP-S, MLP-L, 1-D CNN, delay reservoir, cumulant features) with *lite* settings (minutes, not the full recipe), and compares them with the shipped models.
2. **Exports the arena data** (`arena_models.json`, `examples_arena.json`, `examples_gen_stats.json`, `energy_refs.json`, plus the small phase-1 net's `model_*.json` / `examples_*.json`) from the *shipped* models into `docs/artifacts/data/`, the exact files the article's widgets read.
3. **Checks the JavaScript forward passes** the widgets run (`docs/artifacts/assets/js/*.js`) against the Python reference, headless in `node` (skipped if `node` is absent).

The shipped models were trained with the full recipe (60 epochs, 3 restarts); the lite models here use fewer epochs and one seed, so their accuracy is expected to be a few points lower. They are exported to `docs/artifacts/data/lite/` (a set in which the retrained models replace their shipped copies; the feature MLPs and the board-aware DLNN-B stay shipped) and never replace the article's files.
"""),
("code", COLAB),
("code", SETUP % "01 Small nets and the arena"),
("md", "## 1. The task and the data\n\nEight synthetic modulation classes (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK), 128 complex samples, SNR 0 to 20 dB. The generator is deterministic; the dataset hash is checked against the one the shipped models were trained on."),
("code", '''from dlnn.data import modrec
from dlnn.model.npz import load_model
from dlnn.tiers import virtual
ds = modrec.dataset()
ref_report = paper.read_json("models/arena/arena_report.json")
print("classes", ds["classes"])
print("train", ds["x_train"].shape, "test", ds["x_test"].shape)
print("dataset sha256", ds["sha256"][:16], "| arena report", ref_report["dataset_sha256"][:16])
assert ds["sha256"] == ref_report["dataset_sha256"], "dataset differs from the one the shipped models were trained on"

def acc_V(model_path):
    """Accuracy of a dlnn-model/1 file on the 4000-example test set with the numpy ideal-math tier (V)."""
    m = load_model(model_path)
    return float((virtual.forward(m, ds["x_test"])["out"].argmax(1) == ds["y_test"]).mean())'''),
("md", "## 2. Train the small DLNN, lite\n\nThe shipped recipe (`models/tdlnet_b_w8_report.json`): Adam, lr 0.03, cosine schedule, batch 256, **60 epochs, 3 restarts**, ternarisation ramped in over the first half. Lite: **20 epochs, 1 restart**. Two variants: a ternary readout (the all-analogue net) and a float digital readout (what the Nucleo would run, `DLNN-F`)."),
("code", '''import contextlib, io
from dlnn.model import train
LITE = WORK / "lite" / "models"; LITE.mkdir(parents=True, exist_ok=True)
# start from a copy of the shipped models; whatever is retrained below overwrites its copy (the others, e.g. the feature MLPs and the
# board-aware DLNN-B, which needs the M1a line-FIR search, stay shipped and are listed as such)
import shutil
for f in [*Path("models").glob("tdlnet_b_w8*_ternary.npz"), *Path("models").glob("*.json")]:
    shutil.copyfile(f, LITE / f.name)
shutil.copytree("models/arena", LITE / "arena", dirs_exist_ok=True)
EPOCHS = int(os.environ.get("DLNN_LITE_EPOCHS", 20))
timing = {}
for name, extra in (("tdlnet_b_w8", []), ("tdlnet_b_w8_fro", ["--float-readout"])):
    t = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train.main(["--name", name, "--width", "8", "--depth", "3", "--dilations", "1,1,1", "--epochs", str(EPOCHS),
                    "--restarts", "1", "--ternary-only", "--out", str(LITE)] + extra)
    timing[name] = time.time() - t
    print(buf.getvalue().strip().splitlines()[-2])
    print(f"  {name}: {timing[name]:.0f} s")'''),
("code", '''rows = []
shipped = {"tdlnet_b_w8": paper.read_json("models/tdlnet_b_w8_report.json")["ternary"]["acc"],
           "tdlnet_b_w8_fro": ref_report["models"]["tdlnet_b_w8_fro"]["acc"]}
lite_acc = {}
for name in shipped:
    # shipped file re-evaluated here must reproduce its report exactly; the lite file only has to be in the right region
    s = acc_V(f"models/{name}_ternary.npz")
    l = acc_V(LITE / f"{name}_ternary.npz")
    lite_acc[name] = l
    paper.check(rows, f"{name}: shipped model, V accuracy vs report", s, shipped[name], 0.0005)
    paper.check(rows, f"{name}: lite ({EPOCHS} epochs, 1 seed) vs shipped", l, shipped[name], 0.10, note="lite recipe: expected a few points below")
    assert l > 0.55, "lite training did not learn"
df = paper.show_rows(rows)'''),
("md", "## 3. Arena baselines, lite\n\nSame data, same 4000 test examples. Lite settings: 8 epochs for the torch models, the reservoir and cumulant baselines unchanged (they are cheap). The shipped MLP and CNN numbers in the article come from larger searches (`models/arena/arena_report.json`)."),
("code", '''from dlnn.model import arena
ARENA = LITE / "arena"; ARENA.mkdir(exist_ok=True)
t = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    arena.main(["--only", "mlp_s,mlp_l,cnn,reservoir,cumulant,tdlnet_b_w8,tdlnet_b_w8_fro", "--out", str(ARENA),
                "--dlnn", str(LITE / "tdlnet_b_w8_ternary.npz"), "--epochs", "8"])
timing["arena"] = time.time() - t
print("\\n".join(l for l in buf.getvalue().splitlines() if l[:12].strip() in ("cumulant","mlp_s","mlp_l","cnn","reservoir","tdlnet_b_w8","tdlnet_b_w8_fro")))
print(f"arena lite: {timing['arena']:.0f} s")
lite_rep = paper.read_json(ARENA / "arena_report.json")'''),
("code", '''rows = []
for k in ("tdlnet_b_w8", "tdlnet_b_w8_fro", "mlp_s", "mlp_l", "cnn", "reservoir", "cumulant"):
    a_l, a_s = lite_rep["models"][k]["acc"], ref_report["models"][k]["acc"]
    ok = paper.check(rows, f"{k}: lite vs shipped accuracy", a_l, a_s, 0.15, note=f"ops adds lite {lite_rep['models'][k]['ops']['adds']} / shipped {ref_report['models'][k]['ops']['adds']}")
paper.show_rows(rows);
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
ks = ["tdlnet_b_w8", "tdlnet_b_w8_fro", "mlp_s", "mlp_l", "cnn", "reservoir", "cumulant"]
x = np.arange(len(ks))
ax.bar(x - 0.2, [ref_report["models"][k]["acc"] for k in ks], 0.4, label="shipped (full recipe)")
ax.bar(x + 0.2, [lite_rep["models"][k]["acc"] for k in ks], 0.4, label="lite (this notebook)")
ax.set_xticks(x, ks, rotation=30, ha="right", fontsize=8); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8); fig.tight_layout()'''),
("md", "## 4. Export the data the article reads\n\nFrom the **shipped** models (not the lite ones) into `docs/artifacts/data/`. The export is deterministic, so the files are compared byte for byte with the reference snapshot in `expected/data/`. Lite exports go to `docs/artifacts/data/lite/`."),
("code", '''from dlnn.viz import export_web
t = time.time()
p1 = export_web.export_modrec("models/tdlnet_modrec_ternary.npz", DATA, n=40)         # the phase-1 width-8 net used by the intro widget
p2 = export_web.export_arena("models", DATA, n=40)                                      # arena_models, examples_arena, examples_gen_stats, energy_refs
print("export", f"{time.time() - t:.1f} s")
rows = []
for k, p in {**p1, **p2}.items():
    ref = EXPECTED / "data" / Path(p).name
    same = ref.exists() and paper.sha256_file(p) == paper.sha256_file(ref)
    rows.append({"file": f"docs/artifacts/data/{Path(p).name}", "bytes": Path(p).stat().st_size, "sha256_16": paper.sha256_file(p)[:16],
                 "identical to reference snapshot": same})
import pandas as pd; print(pd.DataFrame(rows).to_string(index=False))'''),
("code", '''t = time.time()
LITE_OUT = DATA / "lite"
export_web.export_arena(LITE, LITE_OUT, n=40) if (LITE / "arena" / "arena_report.json").exists() else None
print("lite export", f"{time.time() - t:.1f} s", sorted(p.name for p in LITE_OUT.glob("*.json")))'''),
("md", "## 5. JavaScript forward pass vs Python\n\nThe widgets run `docs/artifacts/assets/js/tdlnet.js` and `arena.js` in the browser. Here the same code runs headless in `node` on the exported JSON and is compared with the numpy reference logits (tolerance 1e-5, top-1 must agree on every example). The JS signal generator is also checked statistically against the Python generator (`examples_gen_stats.json`)."),
("code", '''import subprocess, shutil
NODE = shutil.which("node")
js = {}
if NODE is None:
    print("node not found: JS check SKIPPED (install nodejs to run it)")
else:
    print(paper.tool_version("node"))
    r = subprocess.run([NODE, "tests/js/check_tdlnet.mjs", str(p1["model"]), str(p1["examples"])], capture_output=True, text=True, timeout=300)
    js["tdlnet"] = json.loads(r.stdout.strip().splitlines()[-1]); print("tdlnet.js  ", js["tdlnet"])
    for tag, d in (("arena", DATA), ("arena_lite", LITE_OUT)):
        if not (Path(d) / "arena_models.json").exists():
            continue
        r = subprocess.run([NODE, "tests/js/check_arena.mjs", str(d)], capture_output=True, text=True, timeout=600)
        try:
            js[tag] = json.loads(r.stdout)
        except Exception:
            js[tag] = {"ok": False, "stdout": r.stdout[-500:], "stderr": r.stderr[-500:]}
        print(f"arena.js ({tag}): ok={js[tag].get('ok')}", {k: round(v["max_abs_err"], 9) for k, v in js[tag].get("models", {}).items()})
    assert js["tdlnet"]["max_abs_err"] <= 1e-5 and js["tdlnet"]["top1_agree"] == js["tdlnet"]["n"]
    assert js["arena"]["ok"] and all(m["max_abs_err"] <= 1e-5 for m in js["arena"]["models"].values())
    print("JS == Python within 1e-5 on every example")'''),
("code", '''paper.write_json(DATA / "train_lite.json", {
    "schema": "dlnn-train-lite/1", "note": "lite settings (%d epochs, 1 seed); shipped models use 60 epochs, 3 restarts. Simulation only." % EPOCHS,
    "epochs": EPOCHS, "seconds": {k: round(v, 1) for k, v in timing.items()},
    "acc_lite": {k: v["acc"] for k, v in lite_rep["models"].items()}, "acc_shipped": {k: ref_report["models"][k]["acc"] for k in lite_rep["models"]},
    "js_check": {k: {"max_abs_err": v.get("max_abs_err"), "ok": v.get("ok")} for k, v in js.items()}, "node": ENV["node"]}, indent=1)
''' + END % NB),
]
build(f"{NB}.ipynb", NB, cells)
