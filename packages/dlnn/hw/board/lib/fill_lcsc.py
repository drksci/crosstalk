#!/usr/bin/env python3
"""Resolve every 'TBD' LCSC field of generated projects through the JLC parts search (jlc_lookup.py) into lcsc_cache.json.
usage: fill_lcsc.py PROJDIR [PROJDIR ...]   (then regenerate: dlnn_parts.res_e96 reads the cache)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_schematic as cs  # noqa: E402
import jlc_lookup as J  # noqa: E402
mp = set()
for d in sys.argv[1:]:
    for r, v in cs.symbols(d).items():
        if v["props"].get("LCSC", "").startswith("TBD"):
            mp.add(v["props"]["MPN"])
for m in sorted(mp):
    r = J.lookup_mpn(m, refresh=False)
    print(m, (r["lcsc"], r["lib"], r["stock"]) if r else "NOT FOUND")
