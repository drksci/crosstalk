#!/usr/bin/env python3
"""Cross-check the article's evidence chips against docs/artifacts/data/claims.json and the source files.

  python scripts/check_article_numbers.py            (run from anywhere; exit 1 on any failure)

For every <span class="ev" data-claim="ID">text</span> in docs/artifacts/article/content/sections.json:
  1. ID must exist in claims.json;
  2. the number in `text` (times the claim's optional display_scale, abs() if display_abs) must equal the claim value to
     within max(tolerance*scale, half a unit of the last displayed digit);
  3. the claim value must still equal what the source file says today (JSON key path, optional regex group 1, or a
     `derive` expression over named source inputs), within the claim tolerance.
Also reports claims that are not used by any chip (info only).
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "docs/artifacts/data/claims.json"
SECTIONS = ROOT / "docs/artifacts/article/content/sections.json"
CHIP = re.compile(r'<span class="ev" data-claim="([^"]+)">(.*?)</span>', re.S)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def dig(d, path):
    for p in path.split("."):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d


def read_source(src):
    """Value a source spec {file, key?, regex?} gives today."""
    p = ROOT / src["file"]
    if p.suffix == ".json":
        v = dig(json.loads(p.read_text()), src["key"]) if src.get("key") else p.read_text()
    else:
        v = p.read_text()
    if src.get("regex"):
        m = re.search(src["regex"], str(v), re.S)
        if not m:
            raise ValueError(f"regex not found in {src['file']}: {src['regex']}")
        v = m.group(1).replace(",", "")
    return float(v) if not isinstance(v, bool) else float(v)


def claim_value_now(c):
    s = c["source"]
    if "derive" in s:
        env = {n: read_source(spec) for n, spec in s["derive"]["inputs"].items()}
        env.update({"pi": math.pi, "math": math})
        return float(eval(s["derive"]["expr"], {"__builtins__": {}}, env))
    return read_source(s)


def main():
    claims = {c["id"]: c for c in json.loads(CLAIMS.read_text())["claims"]}
    sections = json.loads(SECTIONS.read_text())
    fails, used, n_chips = [], set(), 0
    for sid, html in sections.items():
        for cid, text in CHIP.findall(html):
            n_chips += 1
            used.add(cid)
            where = f"[{sid}] {cid} '{text}'"
            c = claims.get(cid)
            if c is None:
                fails.append(f"{where}: claim id missing from claims.json")
                continue
            m = NUM.search(re.sub(r"<[^>]+>", "", text))
            if not m:
                fails.append(f"{where}: no number in chip text")
                continue
            raw = m.group(0).replace(",", "")
            shown = float(raw)
            dec = len(raw.split(".")[1]) if "." in raw else 0
            scale = c.get("display_scale", 1)
            val = c["value"] * scale
            if c.get("display_abs"):
                val, shown = abs(val), abs(shown)
            allow = max(c.get("tolerance", 0) * abs(scale), 0.5 * 10 ** (-dec)) + 1e-9
            if abs(shown - val) > allow:
                fails.append(f"{where}: shows {shown} but claim value*scale = {val:.6g} (allowed +-{allow:.3g})")
    for cid in sorted(used & set(claims)):
        c = claims[cid]
        try:
            now = claim_value_now(c)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{cid}: cannot read source: {e}")
            continue
        if abs(now - c["value"]) > max(c.get("tolerance", 0), 1e-9 * max(1, abs(now))):
            fails.append(f"{cid}: claims.json value {c['value']} != source value {now} (tol {c.get('tolerance', 0)})")
    unused = sorted(set(claims) - used)
    print(f"chips: {n_chips}, distinct claim ids used: {len(used)}, claims in file: {len(claims)}, unused claims: {len(unused)}")
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
