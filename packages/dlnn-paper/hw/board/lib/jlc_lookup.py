#!/usr/bin/env python3
"""JLCPCB parts-library lookup (M3b): keyword -> C-number, library type, description, from the public JLC component search the
jlcpcb.com parts page uses (POST selectSmtComponentList).  Results are cached in lcsc_cache.json (checked in: it is the evidence).

    python3 jlc_lookup.py KEYWORD [KEYWORD ...]        print the first matches
    python3 jlc_lookup.py --mpn 0603WAF9310T5E ...     exact-MPN match only; cache it
A C-number goes into the design only if the MPN (or description) of a returned row matches the requested part exactly.
"""
import json
import subprocess
import sys
from pathlib import Path

CACHE = Path(__file__).with_name("lcsc_cache.json")
URL = "https://jlcpcb.com/api/overseas-pcb-order/v1/shoppingCart/smtGood/selectSmtComponentList"


def query(keyword, n=8):
    body = json.dumps({"currentPage": 1, "pageSize": n, "keyword": keyword, "firstSortName": "", "secondSortName": "",
                       "searchSource": "search"})
    r = subprocess.run(["curl", "-sS", "-m", "40", "-X", "POST", URL, "-H", "Content-Type: application/json", "-d", body],
                       capture_output=True, text=True)
    d = json.loads(r.stdout)
    out = []
    for c in ((d.get("data") or {}).get("componentPageInfo") or {}).get("list") or []:
        out.append(dict(lcsc=c.get("componentCode") or c["urlSuffix"].split("/")[-1], mpn=c.get("componentModelEn"),
                        brand=c.get("componentBrandEn"), pkg=c.get("componentSpecificationEn"),
                        lib="Basic" if c.get("componentLibraryType") == "base" else "Extended",
                        desc=c.get("describe") or c.get("erpComponentName"), stock=c.get("stockCount"),
                        price1=(c.get("componentPrices") or [{}])[0].get("productPrice")))
    return out


def load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def lookup_mpn(mpn, refresh=False):
    cache = load()
    if mpn in cache and not refresh:
        return cache[mpn]
    hit = None
    for row in query(mpn):
        if (row["mpn"] or "").strip().upper() == mpn.strip().upper():
            hit = row
            break
    cache[mpn] = hit
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return hit


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--mpn":
        for m in a[1:]:
            print(m, lookup_mpn(m))
    else:
        for k in a:
            print("==", k)
            for r in query(k):
                print("  ", r)
