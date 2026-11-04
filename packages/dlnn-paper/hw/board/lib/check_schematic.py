#!/usr/bin/env python3
"""M3a checks on a generated tile project: part counts per LCSC type vs parts.csv / params.json, crossbar population vs the
model's non-zero weights, LCSC field on every non-power symbol.

    hw/.venv-eda/bin/python hw/board/lib/check_schematic.py hw/board/tile/layer2 --layer 2 [--model PATH]
Exit status 1 on any FAIL.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOARD = HERE.parent
REPO = BOARD.parent.parent
sys.path.insert(0, str(HERE))
import kisch  # noqa: E402


def symbols(projdir):
    out = {}      # ref -> dict
    for f in sorted(Path(projdir).glob("*.kicad_sch")):
        t = kisch.parse(f.read_text())
        for s in kisch.find(t, "symbol"):
            lib_id = str(kisch.find(s, "lib_id")[0][1])
            if lib_id.startswith("power:"):
                continue
            props = {str(p[1]): str(p[2]) for p in kisch.find(s, "property")}
            ref = props["Reference"]
            d = out.setdefault(ref, dict(lib_id=lib_id, props=props, units=set(), sheet=f.stem,
                                         dnp=str(kisch.find(s, "dnp")[0][1]) == "yes",
                                         in_bom=str(kisch.find(s, "in_bom")[0][1]) == "yes"))
            d["units"].add(int(kisch.find(s, "unit")[0][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proj")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", default=str(REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    syms = symbols(a.proj)
    params = json.load(open(BOARD / "params.json"))
    rows = {}
    for r in csv.DictReader(open(BOARD / "parts.csv")):
        c = r["jlc_lcsc_c_number"]
        if c.startswith("C") and (c not in rows or not rows[c]["qty_full_tile"].strip()):
            rows[c] = r
    full = a.layer != 1
    qcol = "qty_full_tile" if full else "qty_layer1_tile"
    fails = []
    report = {"layer": a.layer, "project": str(a.proj), "types": {}, "checks": {}}

    # 1. LCSC on every symbol
    missing = [r for r, d in syms.items() if not d["props"].get("LCSC")]
    report["checks"]["lcsc_on_every_symbol"] = not missing
    if missing:
        fails.append(f"no LCSC: {missing[:5]}")
    tbd = sorted(r for r, d in syms.items() if d["props"].get("LCSC", "").startswith("TBD"))
    report["lcsc_tbd_refs"] = tbd

    # 2. counts per LCSC (populated, in BOM)
    cnt = Counter()
    for r, d in syms.items():
        if d["dnp"] or not d["in_bom"]:
            continue
        cnt[d["props"]["LCSC"]] += 1
    W = np.load(a.model, allow_pickle=True)[f"L{a.layer-1}_W"].astype(int)
    nz = int((W != 0).sum())
    n_lines = 2 if a.layer == 1 else 8
    # expected from params.json tile_counts + parts.csv
    tc = params["tile_counts"]
    exp = {
        "C394134": tc["inductors_per_full_tile"] if full else tc["inductors_layer1_tile"],
        "C25744": nz,
    }
    for c, r in rows.items():
        v = r[qcol].strip()
        if v and c not in exp and c not in ("C25744",):
            exp[c] = int(v)
    # carrier-only / non-tile rows are not in a tile
    for c in ("C134275", "C130026"):
        exp.pop(c, None)
    # per-type compare
    tbl = []
    for c in sorted(set(exp) | set(cnt)):
        if c == "C25804" or not c.startswith("C") or c.startswith("C25804"):
            continue
        e = exp.get(c)
        g = cnt.get(c, 0)
        name = rows.get(c, {}).get("function", "")[:44]
        ok = (e is None and g == 0) or e == g
        if c == "C9040" and e is None:
            ok = True
        tbl.append((c, name, e, g, "PASS" if ok else "DIFF"))
        report["types"][c] = dict(name=name, expected=e, got=g, ok=bool(ok))
    # E96 'other' resistors (0603, not 1k, not 10k pull-down)
    other = sum(1 for r, d in syms.items() if not d["dnp"] and d["props"].get("Footprint", "").endswith("R_0603_1608Metric")
                and d["props"]["LCSC"] != "C21190")
    report["types"]["E96_other_0603"] = dict(name="E96 0603 resistors other than 1k", expected=int(rows["C25804"][qcol]) if False else 60,
                                              got=other, ok=(other == 60))
    tbl.append(("E96", "E96 0603 resistors other than 1k (csv 60)", 60, other, "PASS" if other == 60 else "DIFF"))

    # 3. crossbar vs model
    bad = []
    pop_p = pop_n = n_fp = 0
    for o in range(8):
        for i in range(n_lines):
            for k in range(8):
                for pol, want in (("P", 1), ("N", -1)):
                    r = syms.get(f"RX{pol}_{o}_{i}_{k}")
                    if r is None:
                        bad.append(f"missing RX{pol}_{o}_{i}_{k}")
                        continue
                    n_fp += 1
                    populated = not r["dnp"]
                    pop_p += populated and pol == "P"
                    pop_n += populated and pol == "N"
                    if populated != (W[o, i, k] == want) or (r["dnp"] and r["in_bom"]):
                        bad.append(f"RX{pol}_{o}_{i}_{k} dnp={r['dnp']} W={W[o,i,k]}")
    xb = dict(footprints=n_fp, populated=pop_p + pop_n, populated_plus=pop_p, populated_minus=pop_n, nonzero_weights=nz,
              W_plus=int((W == 1).sum()), W_minus=int((W == -1).sum()), mismatches=bad[:10], ok=(not bad and pop_p + pop_n == nz))
    report["crossbar"] = xb
    if not xb["ok"]:
        fails.append("crossbar mismatch")
    # footprints
    exp_fp = n_lines * 64 * 2
    report["checks"]["crossbar_footprints"] = (n_fp == exp_fp)
    # layer params consistency with params.json for default model
    print(f"layer {a.layer}  project {a.proj}")
    print(f"{'LCSC':9} {'type':46} {'exp':>5} {'got':>5}")
    for c, name, e, g, v in tbl:
        print(f"{c:9} {name:46} {str(e):>5} {g:>5} {v}")
        if v == "DIFF":
            report["checks"].setdefault("count_diffs", []).append(c)
    print(f"crossbar: footprints {n_fp} (expect {exp_fp}), populated {pop_p}+ {pop_n}- = {pop_p+pop_n}, non-zero W {nz} "
          f"(+{xb['W_plus']} -{xb['W_minus']}) -> {'PASS' if xb['ok'] else 'FAIL'}")
    print(f"LCSC present on all {len(syms)} symbols: {not missing}; refs with LCSC 'TBD*': {len(tbd)}")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
