from _common import *

NB = "00_verify"
cells = [
("md", r"""
# 00 Verify: PASS / FAIL for every claim in the article

[![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/<owner>/dlnn-paper/blob/main/notebooks/00_verify.ipynb)

Every number the article shows in an evidence chip is a **claim** in `docs/artifacts/data/claims.json` (value, tolerance, fidelity label, source). This notebook checks each one two ways:

1. **re-read**: the value is read again from its source today (a JSON key path, a regex in a document, or a `derive` formula over named source values) and compared with the claim within its tolerance;
2. **recompute** (where the repo can do it in minutes without training anything): the number is *computed again from the shipped models and the regenerated test set*, not just read. Tier V and E are re-run on all 4000 test examples; the Verilog RTL is re-run through `iverilog` on a few examples; the arena models are re-evaluated with the independent numpy forward passes; the P-tier nominal accuracies, the cost model and the hardware parameters (`derive_params.py`) are regenerated. For the slow tiers C and R (ngspice unit cells, scikit-rf channels) the full-test-set accuracy is computed from the stored per-example predictions of the full run, and a **fresh run on a stratified subset** must reproduce those predictions.

**Fast path**: no torch training, a few minutes. Set `DLNN_VERIFY_CR_N=0` to skip the fresh C/R subset runs.

A PASS means *the number in the article is what this repo's code and data give today, within the stated tolerance*. It does not mean the physics is right: everything is simulation of a design that is not built; nothing is measured. Claims labelled `designed` or `RTL` say so in the table.
"""),
("code", COLAB),
("code", SETUP % "00 Verify"),
("code", '''import math, re, tempfile, subprocess, shutil
import pandas as pd
from dlnn.data import modrec
from dlnn.model.npz import load_model
from dlnn.tiers import virtual, emulated
from dlnn import pipeline

CL = paper.read_json(DATA / "claims.json")
claims = CL["claims"]
EVID = DATA / "evidence"
print(f"{len(claims)} claims; groups:", pd.Series([c.get("group") for c in claims]).value_counts().to_dict())

def src_path(f):
    """Source files named in claims.json are paths in the development repo; out/* run files ship as evidence."""
    p = ROOT / f
    return p if p.exists() else EVID / f

def dig(d, path):
    for p in path.split("."):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d

def read_source(src):
    p = src_path(src["file"])
    if p.suffix == ".json":
        v = dig(json.loads(p.read_text()), src["key"]) if src.get("key") else p.read_text()
    else:
        v = p.read_text()
    if src.get("regex"):
        m = re.search(src["regex"], str(v), re.S)
        if not m:
            raise ValueError(f"regex not found in {src['file']}")
        v = m.group(1).replace(",", "")
    return float(v)

def source_value(c):
    s = c["source"]
    if "derive" in s:
        env = {n: read_source(spec) for n, spec in s["derive"]["inputs"].items()}
        env.update({"pi": math.pi, "math": math})
        return float(eval(s["derive"]["expr"], {"__builtins__": {}}, env))
    return read_source(s)

def within(a, b, tol):
    return abs(a - b) <= max(tol, 1e-9 * max(1.0, abs(b)))

R = {}                     # claim id -> dict(row)
for c in claims:
    row = {"id": c["id"], "group": c.get("group", ""), "fidelity": c.get("fidelity", ""), "claim": c["value"], "tol": c.get("tolerance", 0.0),
           "reread": None, "reread_ok": None, "recomputed": None, "recompute_ok": None, "how": "", "note": ""}
    try:
        row["reread"] = source_value(c)
        row["reread_ok"] = within(row["reread"], c["value"], row["tol"])
    except Exception as e:                                   # noqa: BLE001
        row["note"] = f"cannot read source: {type(e).__name__}: {e}"[:140]
        row["reread_ok"] = False
    row["how"] = "formula over source values" if "derive" in c["source"] else "re-read"
    R[c["id"]] = row
print("re-read:", sum(bool(r["reread_ok"]) for r in R.values()), "ok of", len(R))

EXTRA = []                 # checks that are not claims themselves; each may name the claims it supports
def recompute(cid, value, how, tol=None):
    if cid not in R:
        return
    r = R[cid]
    r["recomputed"], r["how"] = float(value), how
    r["recompute_ok"] = within(float(value), r["claim"], r["tol"] if tol is None else tol)
def extra(name, got, want, tol, affects=(), note=""):
    ok = within(got, want, tol)
    EXTRA.append({"check": name, "got": got, "reference": want, "tol": tol, "result": "PASS" if ok else "FAIL", "note": note})
    for cid in affects:
        if cid in R and not ok:
            R[cid]["note"] = (R[cid]["note"] + " | " + name)[:160]; R[cid]["recompute_ok"] = False
    return ok
STAGE_S = {}
def stage(name):
    def deco(fn):
        t = time.time()
        try:
            fn()
        except Exception as e:                               # noqa: BLE001  a failing stage must not hide the others
            import traceback; traceback.print_exc()
            EXTRA.append({"check": f"stage {name} raised", "got": None, "reference": None, "tol": None, "result": "FAIL", "note": f"{type(e).__name__}: {e}"[:140]})
        STAGE_S[name] = time.time() - t
        print(f"[stage {name}] {STAGE_S[name]:.1f} s")
        return fn
    return deco
ds = modrec.dataset()
y, x, snr = ds["y_test"], ds["x_test"], ds["snr_test"]
acc = lambda out, yy=None: float(np.mean(np.asarray(out).argmax(1) == (y if yy is None else yy)))'''),
("md", "## Phase 1: tiers V and E, the float reference, the RTL\n\nFresh numpy runs on all 4000 test examples for the width-32 and width-8 ternary models (the coax study). `E` is the bit-true fixed-point golden model; its Verilog RTL is run through `iverilog` on a few examples (the stored run compared 202 752 values on 64 examples) and must show **0 mismatches**."),
("code", '''from dlnn.model import export
RUNS = {"w32": ("models/tdlnet_modrec_w32_ternary.npz", "models/tdlnet_modrec_w32_float.npz"), "w8": ("models/tdlnet_modrec_ternary.npz", "models/tdlnet_modrec_float.npz")}
PHASE1 = {}
@stage("phase1 V/E/float/RTL")
def _():
    for run, (mp, fp) in RUNS.items():
        m = load_model(mp)
        V = virtual.forward(m, x)["out"]; E = emulated.forward(m, x)["out"]
        PHASE1[run] = {"V": V, "E": E}
        recompute(f"p1.{run}.acc.V", acc(V), "tier V (numpy), 4000 ex")
        recompute(f"p1.{run}.acc.E", acc(E), "tier E (fixed-point golden), 4000 ex")
        recompute(f"p1.{run}.agree.E", float(np.mean(E.argmax(1) == V.argmax(1))), "E vs V top-1, 4000 ex")
        recompute(f"p1.{run}.nmse.E", float(np.sum((E - V) ** 2) / np.sum(V ** 2)), "logit NMSE E vs V, 4000 ex")
        recompute(f"p1.{run}.acc.float", acc(virtual.forward(export.load_float_model(fp), x)["out"]), "float32 reference model, 4000 ex")
        info = emulated.rtl_info()
        if info["available"]:
            r = emulated.rtl_crosscheck(m, x, {"rtl_examples": 4, "rtl_adc_check": False})
            if r.get("ran"):
                recompute(f"p1.{run}.rtl.mismatches", r["mismatches"], f"iverilog on {r['examples']} examples x {r['T']} steps, {r['values_compared']} values compared")
            else:
                print(run, "RTL did not run:", r.get("error"))
        else:
            print("iverilog not available: RTL claims are re-read only")
        # C and R: accuracy from the stored per-example predictions of the full run, checked against the regenerated labels
        pz = np.load(EVID / "out" / f"final-{run}" / "tier_preds.npz")
        for t in "CR":
            recompute(f"p1.{run}.acc.{t}", float(np.mean(pz[f"{t}_pred"] == y)), f"stored per-example predictions of tier {t} vs regenerated labels")
        extra(f"{run}: stored V predictions == fresh V", float(np.mean(pz["V_pred"] == V.argmax(1))), 1.0, 1e-9, note="stored full run vs fresh numpy V")
        extra(f"{run}: stored E predictions == fresh E", float(np.mean(pz["E_pred"] == E.argmax(1))), 1.0, 1e-9, note="stored full run vs fresh E golden")'''),
("md", "## Phase 1: tiers C and R, fresh subset run\n\nC (ngspice unit cells + calibrated behavioural model, 32 dB/us cascade) and R (scikit-rf channel) are slow on 4000 examples (about 5 minutes each on a fast box), so a stratified subset is re-run here and its predictions compared with the stored full-run predictions at the same indices. C is deterministic (expect 100 % agreement); R draws static tap errors and noise, so its agreement is high but not exact."),
("code", '''from dlnn.tiers import circuit, rf
N_CR = int(os.environ.get("DLNN_VERIFY_CR_N", 200))
@stage("phase1 C/R subset")
def _():
    if N_CR <= 0 or not shutil.which("ngspice"):
        print("C/R fresh subset skipped (DLNN_VERIFY_CR_N=0 or no ngspice): claims for C and R are checked from stored predictions only")
        return
    idx = pipeline.stratified_subset(y, snr, N_CR)
    sub = pipeline.subset_dataset(ds, idx)
    for run, (mp, _fp) in RUNS.items():
        pz = np.load(EVID / "out" / f"final-{run}" / "tier_preds.npz")
        with tempfile.TemporaryDirectory(prefix="dlnn_verify_") as td:
            t = time.time()
            rc = circuit.run(mp, sub, {"run_id": "verify", "seed": 0, "c_n_examples": 2, "c_tran_untrimmed": False, "c_diagnostics": False}, td)
            pc = np.asarray(rc.arrays["logits"]).argmax(1)
            tc = time.time() - t
            t = time.time()
            rr = rf.run(mp, sub, {"run_id": "verify", "seed": 0, "cascade_params": f"{td}/params/circuit_extracted.json"}, td)
            pr = np.asarray(rr.arrays["logits"]).argmax(1)
            tr = time.time() - t
        print(f"{run}: C ({rc.status}, {tc:.0f} s) and R ({rr.status}, {tr:.0f} s) on {N_CR} examples")
        extra(f"{run}: C fresh subset agrees with stored full-run predictions", float(np.mean(pc == pz["C_pred"][idx])), 1.0, 0.01, [f"p1.{run}.acc.C"], f"{N_CR} stratified examples, ngspice {ENV['ngspice']}")
        extra(f"{run}: R fresh subset agrees with stored full-run predictions", float(np.mean(pr == pz["R_pred"][idx])), 1.0, 0.10, [f"p1.{run}.acc.R"], f"{N_CR} stratified examples; R draws random channel noise")'''),
("md", "## Arena: the small DLNN and its competitors\n\nAccuracy at all SNRs and at SNR >= 10 dB from the independent numpy forward passes of `dlnn.viz.export_web.reference_logits` (the same ones the JS forward passes are checked against) and `arena.predict_mlp_npz` for the feature MLPs; parameters and operation counts for the DLNN from `arena.dlnn_ops`."),
("code", '''from dlnn.viz import export_web
from dlnn.model import arena
ARENA_MAP = {"dlnn": ("tdlnet_b_w8", "models/tdlnet_b_w8_ternary.npz"), "dlnn_fro": ("tdlnet_b_w8_fro", "models/tdlnet_b_w8_fro_ternary.npz")}
for k in ("mlp_s", "mlp_l", "cnn", "reservoir", "cumulant"):
    ARENA_MAP[k] = (k, f"models/arena/{k}.npz")
@stage("arena")
def _():
    rep = paper.read_json("models/arena/arena_report.json")
    hi = snr >= 10
    for cid_k, (rk, npz) in ARENA_MAP.items():
        out = export_web.reference_logits(cid_k, npz, x)
        recompute(f"arena.{cid_k}.acc", acc(out), f"independent numpy forward, {npz}")
        recompute(f"arena.{cid_k}.ge10", float(np.mean(out.argmax(1)[hi] == y[hi])), "same, SNR >= 10 dB")
        if cid_k == "dlnn":
            recompute("arena.dlnn.8psk", float(np.mean(out.argmax(1)[y == 3] == 3)), "same, class 8PSK")
    for k in ("mlp_feat_s", "mlp_feat_l", "mlp_pow_s", "mlp_pow_l"):
        if Path(f"models/arena/{k}.npz").exists():
            pred, fset = arena.predict_mlp_npz(f"models/arena/{k}.npz")
            recompute(f"arena.{k}.acc", acc(pred(x)), f"numpy forward of models/arena/{k}.npz (feature set {fset})")
    m = load_model("models/tdlnet_b_w8_ternary.npz")
    o, p = arena.dlnn_ops(m), arena.dlnn_params(m)
    recompute("arena.dlnn.ops", o["adds"], "arena.dlnn_ops(model)"); recompute("arena.dlnn.adds", o["adds"], "arena.dlnn_ops(model)")
    recompute("arena.dlnn.mults", o["mults"], "arena.dlnn_ops(model)"); recompute("arena.dlnn.nonlin", o["nonlin"], "arena.dlnn_ops(model)")
    recompute("arena.dlnn.params", p["params"], "arena.dlnn_params(model)")
    mf = load_model("models/tdlnet_b_w8_fro_ternary.npz"); recompute("arena.dlnn_fro.ops", arena.dlnn_ops(mf)["adds"], "arena.dlnn_ops(model)")
    for k in ("mlp_s", "mlp_l"):               # plain MLP on flattened I/Q: params = weights + biases, mults = weights, adds = params
        z = np.load(f"models/arena/{k}.npz", allow_pickle=False)
        names = [str(n) for n in z["layer_names"]]
        W = [z[n.replace(".", "_")] for n in names if n.endswith("weight")]; B = [z[n.replace(".", "_")] for n in names if n.endswith("bias")]
        pa, mu = sum(w.size for w in W) + sum(b.size for b in B), sum(w.size for w in W)
        recompute(f"arena.{k}.params", pa, "count of the npz arrays"); recompute(f"arena.{k}.mults", mu, "count of the npz weights")
        recompute(f"arena.{k}.adds", pa, "mults + biases = params (one add per weight and bias)"); recompute(f"arena.{k}.ops", pa, "same")
    recompute("arena.n_test", len(y), "regenerated dataset"); recompute("arena.n_train", len(ds["y_train"]), "regenerated dataset")'''),
("md", "## Board-family models, the M1a line and the P tier\n\nThe board-aware models are evaluated with tier V (ideal), and with V on the **ngspice-extracted M1a line FIR** (`models/board_aware/line_fir_m1a.json`) at the Q = 10 and Q = 30 inductor corners. The P-tier nominal accuracies are recomputed with `parasitics.run_nominal` on all 4000 examples."),
("code", '''from dlnn.tiers import circuit_board as cb, parasitics as P
CLASSES = list(ds["classes"])
@stage("board-family models")
def _():
    fir = paper.read_json("models/board_aware/line_fir_m1a.json")["corners"]
    ev = lambda mp: load_model(mp)
    # TDL-Net-B (d = 1) reports
    for cid, mp, key in (("d1.w8.acc", "models/tdlnet_b_w8_ternary.npz", "acc"), ("d1.w8.ge10", "models/tdlnet_b_w8_ternary.npz", "ge10"),
                         ("d1.w16.acc", "models/tdlnet_b_w16_ternary.npz", "acc"), ("d1.v32.acc", "models/v32_modrec_ternary.npz", "acc"),
                         ("d1.v32.float", "models/v32_modrec_float.npz", "acc")):
        m = export.load_float_model(mp) if "float" in mp else ev(mp)
        o = virtual.forward(m, x)["out"]
        recompute(cid, acc(o) if key == "acc" else float(np.mean(o.argmax(1)[snr >= 10] == y[snr >= 10])), f"tier V on {mp}")
    recompute("d1.w8.nternary", ev("models/tdlnet_b_w8_ternary.npz").n_ternary_weights, "model.n_ternary_weights")
    mdl = {"fro": ev("models/tdlnet_b_w8_aware2_fro_ternary.npz"), "plain": ev("models/tdlnet_b_w8_ternary.npz"),
           "tern": ev("models/tdlnet_b_w8_aware2_ternary.npz"), "collapsed": ev("models/tdlnet_b_w8_aware2_collapsed_ternary.npz")}
    def on_fir(m, c): return cb.v_forward_fir(m, x, fir=fir[c]["fir"])["out"]
    for tag, m in mdl.items():
        o10 = on_fir(m, "Q10")
        recompute(f"aware.{tag}.q10", acc(o10), f"tier V on the ngspice M1a line FIR (Q10), {tag} model")
        if tag == "fro":
            recompute("aware.fro.ideal", acc(virtual.forward(m, x)["out"]), "tier V, ideal line")
            recompute("aware.fro.q30", acc(on_fir(m, "Q30")), "tier V on the M1a line FIR (Q30)")
            recompute("aware.fro.ge10", float(np.mean(o10.argmax(1)[snr >= 10] == y[snr >= 10])), "same, SNR >= 10 dB")
        if tag == "tern":
            recompute("aware.tern.ideal", acc(virtual.forward(m, x)["out"]), "tier V, ideal line")
        for cn, cid_tag in (("QPSK", "qpsk"), ("8PSK", "8psk")):
            k = CLASSES.index(cn)
            recompute(f"aware.{tag}.{cid_tag}", float(np.mean(o10.argmax(1)[y == k] == k)), f"per-class accuracy, {cn}, V on Q10 FIR")
    # P tier nominal
    nom = P.run_nominal("models/tdlnet_b_w8_aware2_fro_ternary.npz", ds, None, None)
    recompute("p.nom.q10", nom["Q10"]["accuracy"], "parasitics.run_nominal, 4000 ex"); recompute("p.nom.q30", nom["Q30"]["accuracy"], "parasitics.run_nominal, 4000 ex")
    recompute("p.nom.untrimmed", nom["Q10_as_designed_trim0dB"]["accuracy"], "parasitics.run_nominal, board as designed (0 dB trim)")
    recompute("p.v.m1a", acc(on_fir(mdl["fro"], "Q10")), "tier V on the M1a line FIR (Q10)")
    recompute("p.p_minus_v", nom["Q10"]["accuracy"] - acc(on_fir(mdl["fro"], "Q10")), "P nominal minus V on M1a FIR (Q10)")'''),
("md", "## Hardware parameters and cost model, regenerated\n\n`hw/board/derive_params.py` recomputes every number of `hw/board/params.json` from the design inputs (the time-domain acceptance numbers are merged from the committed ngspice results `hw/board/spice/results.json`); `hw/board/cost_model.py` recomputes the BOM, PCB, assembly and scaling table from the stored JLC price tiers (offline). The claims that point into these files are compared with the regenerated values."),
("code", '''@stage("params + cost regeneration")
def _():
    WK = WORK / "verify"; WK.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "hw/board/derive_params.py", "--model", "models/tdlnet_modrec_ternary.npz", "--out", str(WK / "params.json")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-300:]
    r = subprocess.run([sys.executable, "hw/board/cost_model.py", "--out", str(WK / "cost.json"), "--parts", str(WK / "parts.csv")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-300:]
    newp, newc = paper.read_json(WK / "params.json"), paper.read_json(WK / "cost.json")
    n = 0
    for c in claims:
        s = c["source"]
        if s.get("file") in ("hw/board/params.json", "hw/board/cost.json") and s.get("key") and not s.get("regex"):
            try:
                v = float(dig(newp if s["file"].endswith("params.json") else newc, s["key"]))
            except (KeyError, IndexError, TypeError):
                continue
            recompute(c["id"], v, "regenerated by " + ("derive_params.py" if s["file"].endswith("params.json") else "cost_model.py")); n += 1
    print(n, "claims recomputed from regenerated params.json / cost.json")'''),
("md", "## Integrity: shipped models and netlists against `expected/hashes.json`"),
("code", '''H = paper.read_json(EXPECTED / "hashes.json")
bad = [k for k, v in H["models"].items() if not Path(k).exists() or paper.sha256_file(k) != v["sha256"]]
bad += [k for k, v in H["netlists"].items() if not Path(k).exists() or paper.sha256_file(k) != v]
EXTRA.append({"check": f"sha256 of {len(H['models'])} model files and {len(H['netlists'])} netlist/parameter files match expected/hashes.json", "got": len(bad), "reference": 0, "tol": 0,
              "result": "PASS" if not bad else "FAIL", "note": ", ".join(bad)[:120]})
print("hash mismatches:", bad or "none")'''),
("md", "## Result"),
("code", '''rows = []
for cid, r in R.items():
    oks = [v for v in (r["reread_ok"], r["recompute_ok"]) if v is not None]
    r["result"] = "PASS" if all(oks) and oks else "FAIL"
    r["level"] = "recomputed" if r["recompute_ok"] is not None else ("formula" if r["how"].startswith("formula") else "re-read")
    rows.append(r)
tab = pd.DataFrame(rows)
summ = tab.groupby(["level", "result"]).size().unstack(fill_value=0)
print(summ.to_string()); print()
n_pass, n_fail = int((tab.result == "PASS").sum()), int((tab.result == "FAIL").sum())
print(f"CLAIMS: {n_pass} PASS, {n_fail} FAIL of {len(tab)}   ({int((tab.level == 'recomputed').sum())} recomputed, {int((tab.level == 'formula').sum())} formula, {int((tab.level == 're-read').sum())} re-read only)")
ex = pd.DataFrame(EXTRA)
if len(ex):
    print(f"\\nEXTRA CHECKS: {int((ex.result == 'PASS').sum())} PASS, {int((ex.result == 'FAIL').sum())} FAIL")
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(ex.to_string(index=False))'''),
("code", '''show = tab[["id", "group", "fidelity", "claim", "tol", "reread", "recomputed", "level", "result", "how", "note"]].copy()
with pd.option_context("display.max_rows", 400, "display.max_colwidth", 60, "display.width", 250, "display.float_format", "{:.6g}".format):
    print(show[show.result == "FAIL"].to_string(index=False) if n_fail else "no FAIL rows")
    print()
    print(show.drop(columns=["how", "note"]).to_string(index=False))'''),
("code", '''# pipeline reproducibility: the data files the notebooks (re)write vs the reference snapshot in expected/data
repro = []
for name in ("arena_models.json", "examples_arena.json", "examples_gen_stats.json", "energy_refs.json", "model_tdlnet_modrec_ternary.json",
             "examples_tdlnet_modrec_ternary.json", "scale_story.json", "board_cost.json", "spice/tap_response.json", "spice/diffpair_dc.json"):
    a, b = DATA / name, EXPECTED / "data" / name
    if a.exists() and b.exists():
        repro.append({"file": name, "identical to reference snapshot": paper.read_json(a) == paper.read_json(b)})
print(pd.DataFrame(repro).to_string(index=False))
tm = DATA / "runtimes.json"
rt = paper.read_json(tm) if tm.exists() else {}
paper.write_json(DATA / "verify_results.json", {
    "schema": "dlnn-verify-results/1", "note": "Output of notebooks/00_verify.ipynb. PASS = value in claims.json equals what the repo's code and data give today within the claim tolerance. Simulation of a design that is not built; nothing measured.",
    "env": ENV, "n_pass": n_pass, "n_fail": n_fail, "levels": tab.level.value_counts().to_dict(), "stage_seconds": {k: round(v, 1) for k, v in STAGE_S.items()},
    "claims": {r["id"]: {k: r[k] for k in ("result", "level", "claim", "tol", "reread", "recomputed", "how", "note")} for r in rows},
    "extra_checks": EXTRA, "notebook_cell": "00_verify.ipynb#Result"}, indent=1)
paper.record_runtime("00_verify", time.time() - T0, {"n_pass": n_pass, "n_fail": n_fail})
print(f"{time.time() - T0:.0f} s wall")
if os.environ.get("DLNN_VERIFY_STRICT") == "1":
    assert n_fail == 0 and (len(ex) == 0 or (ex.result == "PASS").all())'''),
]
build(f"{NB}.ipynb", NB, cells)
