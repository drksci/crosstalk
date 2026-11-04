"""Build docs/artifacts/data/claims.json (schema "dlnn-claims/1") from the stored results.

  python docs/artifacts/article/build_claims.py            (run from the repo root)

Every claim = id, text, value (read from the source file at build time), tolerance (abs), fidelity label, the source
file + key path, and slots for the verify-notebook cell tag and the pytest id (filled by M6). The page's evidence chips
(assets/js/chips.js) resolve `data-claim="<id>"` against this file.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/artifacts/data/claims.json"
RUNS = {"w32": ("out/final-w32/compare.json", "width-32 ternary model (models/tdlnet_modrec_w32_ternary.npz), coax study"),
        "w8": ("out/final-w8/compare.json", "width-8 ternary model (models/tdlnet_modrec_ternary.npz), coax study")}
TIER_NAME = {"V": "ideal math (tier V)", "E": "bit-true fixed point (tier E)", "C": "SPICE-calibrated circuit model (tier C)",
             "R": "RF / signal-integrity channel model (tier R)"}
FIDELITY = {"V": "simulated", "E": "RTL", "C": "simulated", "R": "simulated"}


def dig(d, path):
    for p in path.split("."):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]


claims = []


def add(cid, text, value, unit, tol, fidelity, src_file, key, group, **extra):
    claims.append({"id": cid, "text": text, "value": value, "unit": unit, "tolerance": tol, "fidelity": fidelity,
                   "group": group, "source": {"file": src_file, "key": key, "file_sha256_16": sha(src_file)},
                   "notebook_cell": None, "pytest": None, "status": "pending-verify", **extra})


for run, (f, desc) in RUNS.items():
    d = json.loads((ROOT / f).read_text())
    for t in "VECR":
        key = f"metric_matrix.accuracy.{t}"
        add(f"p1.{run}.acc.{t}", f"Mixed-SNR accuracy of the {desc} on the synthetic 8-class set, {TIER_NAME[t]}.",
            round(dig(d, key), 5), "fraction", 0.0005, FIDELITY[t], f, key, "phase1-headline", tier=t, run=run)
    key = "metric_matrix.top1_agreement_vs_virtual.E"
    add(f"p1.{run}.agree.E", f"Top-1 agreement of the fixed-point tier with the ideal tier ({desc}).",
        round(dig(d, key), 5), "fraction", 0.0005, "RTL", f, key, "phase1-headline", tier="E", run=run)
    key = "metric_matrix.logit_nmse_vs_virtual.E"
    add(f"p1.{run}.nmse.E", f"Logit NMSE of the fixed-point tier against the ideal tier ({desc}).",
        dig(d, key), "ratio", 1e-5, "RTL", f, key, "phase1-headline", tier="E", run=run)
    txt = dig(d, "tier_info.E.fidelity.models.1")
    m = re.search(r"(\d+) values .*?compared, (\d+) mismatches", txt)
    add(f"p1.{run}.rtl.mismatches", f"Verilog RTL vs numpy golden model, {m.group(1)} chained layer values compared ({desc}).",
        int(m.group(2)), "count", 0, "RTL", f, "tier_info.E.fidelity.models.1", "phase1-headline", tier="E", run=run,
        compared=int(m.group(1)))
    key = "reference.accuracy"
    add(f"p1.{run}.acc.float", f"Float32 reference with the same topology and unconstrained taps ({desc}); a reference, not a tier.",
        round(dig(d, key), 5), "fraction", 0.0005, "simulated", f, key, "phase1-headline", run=run)

d = json.loads((ROOT / "models/arena/arena_report.json").read_text())
names = {"tdlnet_b_w8": "DLNN (TDL-Net-B w8, d=1)", "tdlnet_b_w8_fro": "DLNN-F (float readout)", "mlp_s": "MLP-S", "mlp_l": "MLP-L", "cnn": "1-D CNN", "reservoir": "delay reservoir", "cumulant": "cumulant baseline"}
for k, n in names.items():
    mid = {"tdlnet_b_w8": "dlnn", "tdlnet_b_w8_fro": "dlnn_fro"}.get(k, k)
    add(f"arena.{mid}.acc", f"Test accuracy of the {n} on the 4000-example synthetic test set (mixed SNR).", round(d["models"][k]["acc"], 5),
        "fraction", 0.0005, "simulated", "models/arena/arena_report.json", f"models.{k}.acc", "arena", model=mid)
    add(f"arena.{mid}.ops", f"Operations per classification of one 128-sample window ({n}): adds.", d["models"][k]["ops"]["adds"], "ops", 0,
        "simulated", "models/arena/arena_report.json", f"models.{k}.ops.adds", "arena", model=mid)

doc = {"schema": "dlnn-claims/1",
       "note": "Headline claims for the article's evidence chips. Values are read from the source files by build_claims.py. "
               "Nothing here is measured. 'fidelity' is one of simulated | RTL | designed | measured. "
               "notebook_cell / pytest are filled when the verify notebook (M6) exists; until then status is pending-verify.",
       "fidelity_labels": ["simulated", "RTL", "designed", "measured"],
       "schema_fields": {"id": "stable claim id shown in the chip", "text": "what is claimed", "value": "number", "unit": "string",
                         "tolerance": "absolute, for the verify notebook", "fidelity": "label", "source": "{file, key (dot path), file_sha256_16}",
                         "notebook_cell": "tag in 00_verify.ipynb or null", "pytest": "test id or null", "status": "pending-verify | PASS | FAIL"},
       "claims": claims}
OUT.write_text(json.dumps(doc, indent=1))
print(f"{len(claims)} claims -> {OUT}")
